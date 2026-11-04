//! Box-bounded constrained minimisation: an augmented Lagrangian with a bound-projected Newton
//! inner solve.
//!
//! A problem is handed over as decision variables with box bounds and a callable returning the
//! objective, the equalities `h(x) = 0` and the inequalities `g(x) >= 0` at a point, each as a
//! [`Dual`] carrying its exact gradient. Nothing here knows what the variables mean; a projection
//! that does implements [`Constrained`] and keeps its own reading of the answer.
//!
//! # Why not a plain penalty
//!
//! A quadratic penalty reaches the constrained optimum only as its coefficient goes to infinity, and
//! long before that the inner problem is too ill-conditioned to solve. The augmented Lagrangian adds
//! a multiplier estimate that absorbs the constraint's own gradient, so the penalty stays moderate
//! and the residual falls with the multiplier rather than with the conditioning. It also reaches an
//! optimum that sits ON an inequality: a method that only approached feasibility from inside would
//! never find it, and one that only approached from outside would report an infeasible point.
//!
//! Moved from transformation's `hymeko_optimize::backend` (2026-11-01). The Newton systems use
//! [`solve_in_place`], this crate's one small dense solve, instead of the private elimination they
//! had there; answers agree to the solver tolerance, not bit for bit, because the singular-pivot
//! threshold and the back-substitution order differ.

use crate::dual::Dual;
use crate::linalg::dense::solve_in_place;

/// One point of a constrained problem: the objective, `h(x) = 0` and `g(x) >= 0`, each with its
/// gradient with respect to the decision variables.
#[derive(Debug, Clone)]
pub struct Point {
    pub objective: Dual,
    /// `h(x)`, one per equality.
    pub equalities: Vec<Dual>,
    /// `g(x) >= 0`, one per inequality.
    pub inequalities: Vec<Dual>,
}

impl Point {
    /// How far outside the feasible set the point is: the largest equality residual or
    /// inequality shortfall.
    pub fn violation(&self) -> f64 {
        let equalities = self
            .equalities
            .iter()
            .map(|h| h.value.abs())
            .fold(0.0, f64::max);
        let inequalities = self
            .inequalities
            .iter()
            .map(|g| (-g.value).max(0.0))
            .fold(0.0, f64::max);
        equalities.max(inequalities)
    }
}

/// A box-bounded constrained problem with exact first derivatives.
pub trait Constrained {
    /// Why a point could not be evaluated.
    type Error;

    /// The number of decision variables.
    fn dimension(&self) -> usize;

    /// `(lower, upper)` for each decision variable, in the problem's own order.
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// The objective and the constraints at `x`.
    fn point(&self, x: &[f64]) -> Result<Point, Self::Error>;
}

/// Why a solve returned no answer.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError<E> {
    /// The problem could not be evaluated at a point the search asked about.
    Evaluation(E),
    /// The search ran out of iterations with the constraints or stationarity unmet.
    NotConverged {
        iterations: usize,
        violation: f64,
        stationarity: f64,
    },
}

/// A converged search: the point, and the inner iterations it took.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    pub x: Vec<f64>,
    pub iterations: usize,
}

/// Augmented Lagrangian with a bound-projected Newton inner solve.
#[derive(Debug, Clone)]
pub struct AugmentedLagrangian {
    /// Constraint violation and stationarity both below this counts as converged.
    pub tolerance: f64,
    pub outer_iterations: usize,
    pub inner_iterations: usize,
    pub penalty: f64,
    pub penalty_max: f64,
}

impl Default for AugmentedLagrangian {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            outer_iterations: 200,
            inner_iterations: 200,
            penalty: 10.0,
            penalty_max: 1e10,
        }
    }
}

impl AugmentedLagrangian {
    /// Search from `start`, clamped into the bounds.
    ///
    /// # Errors
    /// [`SolveError::Evaluation`] when the problem rejects a point the search needs, and
    /// [`SolveError::NotConverged`] when the iteration budget runs out first.
    pub fn solve<P: Constrained>(
        &self,
        problem: &P,
        start: &[f64],
    ) -> Result<Solution, SolveError<P::Error>> {
        let n = problem.dimension();
        let bounds = problem.bounds();
        let mut x: Vec<f64> = start
            .iter()
            .enumerate()
            .map(|(k, v)| v.clamp(bounds[k].0, bounds[k].1))
            .collect();

        let probe = problem.point(&x).map_err(SolveError::Evaluation)?;
        let mut state = Augmented {
            problem,
            lambda: vec![0.0; probe.equalities.len()],
            mu: vec![0.0; probe.inequalities.len()],
            rho: self.penalty,
        };

        let mut previous = f64::INFINITY;
        let mut iterations = 0usize;
        let mut converged = false;
        let mut stationarity = f64::INFINITY;

        for _ in 0..self.outer_iterations {
            let inner = self
                .minimise(&state, &mut x, &bounds, n)
                .map_err(SolveError::Evaluation)?;
            iterations += inner;

            let (_, grad, point) = state.at(&x).map_err(SolveError::Evaluation)?;
            stationarity = free_norm(&x, &grad, &bounds);
            let violation = point.violation();

            if violation <= self.tolerance && stationarity <= self.tolerance {
                converged = true;
                break;
            }

            for (i, h) in point.equalities.iter().enumerate() {
                state.lambda[i] -= state.rho * h.value;
            }
            for (j, g) in point.inequalities.iter().enumerate() {
                state.mu[j] = (state.mu[j] - state.rho * g.value).max(0.0);
            }
            // Only tighten when the last round did not buy enough feasibility; raising the penalty
            // on a round that IS converging just spends conditioning for nothing.
            if violation > 0.25 * previous {
                state.rho = (state.rho * 10.0).min(self.penalty_max);
            }
            previous = violation;
        }

        if !converged {
            let point = problem.point(&x).map_err(SolveError::Evaluation)?;
            return Err(SolveError::NotConverged {
                iterations,
                violation: point.violation(),
                stationarity,
            });
        }
        Ok(Solution { x, iterations })
    }

    /// Minimise the augmented Lagrangian at fixed multipliers, inside the bounds.
    fn minimise<P: Constrained>(
        &self,
        state: &Augmented<'_, P>,
        x: &mut Vec<f64>,
        bounds: &[(f64, f64)],
        n: usize,
    ) -> Result<usize, P::Error> {
        let mut damping = 1e-8;
        for iteration in 0..self.inner_iterations {
            let (value, grad, _) = state.at(x)?;
            let free: Vec<usize> = (0..n)
                .filter(|&k| is_free(x[k], grad[k], bounds[k]))
                .collect();
            if free.is_empty() || free_norm(x, &grad, bounds) <= self.tolerance * 1e-2 {
                return Ok(iteration);
            }

            // Newton on the free block, with the Hessian from central differences of the EXACT
            // gradient. The gradient is exact to rounding, so a differenced Hessian is accurate to
            // about `sqrt(eps)` relatively — plenty to steer a step whose acceptance is judged on
            // the exact value and whose stopping test is the exact gradient.
            let hessian = state.hessian(x, &free, bounds)?;
            let mut direction = vec![0.0; n];
            let mut solved = false;
            for _ in 0..12 {
                let mut a = hessian.clone();
                for i in 0..free.len() {
                    a[i * free.len() + i] += damping;
                }
                let mut step: Vec<f64> = free.iter().map(|&k| -grad[k]).collect();
                if solve_in_place(&mut a, &mut step, free.len()).is_some() {
                    let slope: f64 = free.iter().zip(&step).map(|(&k, s)| grad[k] * s).sum();
                    if slope < 0.0 && step.iter().all(|s| s.is_finite()) {
                        for (i, &k) in free.iter().enumerate() {
                            direction[k] = step[i];
                        }
                        solved = true;
                        break;
                    }
                }
                damping *= 10.0;
            }
            if !solved {
                // Steepest descent is always available and always a descent direction.
                for &k in &free {
                    direction[k] = -grad[k];
                }
            }

            // Armijo backtracking, projecting onto the bounds at every trial so a step that would
            // leave the box is shortened rather than rejected.
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..60 {
                let trial: Vec<f64> = (0..n)
                    .map(|k| (x[k] + step * direction[k]).clamp(bounds[k].0, bounds[k].1))
                    .collect();
                let moved: f64 = trial
                    .iter()
                    .zip(x.iter())
                    .map(|(t, o)| (t - o).abs())
                    .fold(0.0, f64::max);
                if moved == 0.0 {
                    break;
                }
                if let Ok((candidate, _, _)) = state.at(&trial) {
                    let predicted: f64 = (0..n).map(|k| grad[k] * (trial[k] - x[k])).sum();
                    if candidate.is_finite() && candidate <= value + 1e-4 * predicted {
                        *x = trial;
                        accepted = true;
                        break;
                    }
                }
                step *= 0.5;
            }
            if !accepted {
                return Ok(iteration);
            }
            damping = (damping * 0.1).max(1e-12);
        }
        Ok(self.inner_iterations)
    }
}

/// The augmented Lagrangian at a set of multipliers.
struct Augmented<'a, P> {
    problem: &'a P,
    /// One per equality; free in sign.
    lambda: Vec<f64>,
    /// One per inequality; non-negative.
    mu: Vec<f64>,
    rho: f64,
}

impl<P: Constrained> Augmented<'_, P> {
    /// `(L, dL, the underlying point)`.
    fn at(&self, x: &[f64]) -> Result<(f64, Vec<f64>, Point), P::Error> {
        let point = self.problem.point(x)?;
        let n = x.len();
        let mut value = point.objective.value;
        let mut grad = point.objective.grad.clone();

        for (i, h) in point.equalities.iter().enumerate() {
            value += -self.lambda[i] * h.value + 0.5 * self.rho * h.value * h.value;
            let coefficient = self.rho * h.value - self.lambda[i];
            for k in 0..n {
                grad[k] += coefficient * h.grad[k];
            }
        }
        // `(1/2rho)(max(0, mu - rho g)^2 - mu^2)`: zero and flat where the constraint is slack with
        // a settled multiplier, quadratic where it is violated, and once differentiable at the join —
        // which is what lets a Newton step approach an active constraint instead of chattering on it.
        for (j, g) in point.inequalities.iter().enumerate() {
            let active = (self.mu[j] - self.rho * g.value).max(0.0);
            value += (active * active - self.mu[j] * self.mu[j]) / (2.0 * self.rho);
            for k in 0..n {
                grad[k] -= active * g.grad[k];
            }
        }
        Ok((value, grad, point))
    }

    /// The Hessian of the free block, by central differences of the exact gradient.
    fn hessian(&self, x: &[f64], free: &[usize], bounds: &[(f64, f64)]) -> Result<Vec<f64>, P::Error> {
        let m = free.len();
        let mut h = vec![0.0; m * m];
        for (c, &k) in free.iter().enumerate() {
            let delta = 1e-6 * x[k].abs().max(1.0);
            let mut up = x.to_vec();
            let mut down = x.to_vec();
            up[k] = (x[k] + delta).min(bounds[k].1);
            down[k] = (x[k] - delta).max(bounds[k].0);
            let span = up[k] - down[k];
            if span == 0.0 {
                h[c * m + c] = 1.0;
                continue;
            }
            let (_, gu, _) = self.at(&up)?;
            let (_, gd, _) = self.at(&down)?;
            for (r, &j) in free.iter().enumerate() {
                h[r * m + c] = (gu[j] - gd[j]) / span;
            }
        }
        // The true Hessian is symmetric; differencing is not, and the asymmetry is pure noise.
        for r in 0..m {
            for c in (r + 1)..m {
                let mean = 0.5 * (h[r * m + c] + h[c * m + r]);
                h[r * m + c] = mean;
                h[c * m + r] = mean;
            }
        }
        Ok(h)
    }
}

/// Is this variable free to move, or pinned at a bound the gradient pushes it through?
fn is_free(x: f64, grad: f64, (lower, upper): (f64, f64)) -> bool {
    let at_lower = x <= lower && grad > 0.0;
    let at_upper = x >= upper && grad < 0.0;
    !at_lower && !at_upper
}

/// The stationarity measure: the gradient, with the pinned directions taken out.
fn free_norm(x: &[f64], grad: &[f64], bounds: &[(f64, f64)]) -> f64 {
    (0..x.len())
        .filter(|&k| is_free(x[k], grad[k], bounds[k]))
        .map(|k| grad[k].abs())
        .fold(0.0, f64::max)
}
