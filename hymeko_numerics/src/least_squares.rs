//! **Nonlinear least squares on a manifold**: Gauss--Newton and Levenberg--Marquardt over a problem
//! that states its own normal equations and retraction (plan 2026-10-19-numerics-optimizers, O1 and
//! O2).
//!
//! ```text
//! min_x  ½ Σ_k w_k ‖r_k(x)‖²,      H δ = −Jᵀ W r,  H = Jᵀ W J,      x ← x ⊕ δ
//! ```
//!
//! The loop knows nothing of poses, meshes or HyMeKo: a [`LeastSquares`] problem assembles its
//! normal equations at a point under given block weights and retracts an increment; an
//! [`IncrementSolver`] solves the normal equations. Moved here from `nagare_slam`'s SE(2) pose
//! graph, whose loop it was; the pose graph is now one implementation of the trait. The robust
//! reweighting around it is [`crate::robust`].
//!
//! [`levenberg_marquardt`] asks nothing more of a problem: its cost `½ Σ w_k ‖r_k‖²` comes from
//! [`LeastSquares::residual_norms`], and a trial point is a copy of the point retracted.

use core::borrow::{Borrow, BorrowMut};

use crate::pcg::{pcg, Jacobi};
use crate::{Operator, Report};

/// **Normal equations** `H δ = rhs` in the layout the symmetric solvers read: `H`'s diagonal and its
/// strict upper triangle as `(row, column, value)`, `row < column`.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalSystem {
    /// `H_ii`.
    pub diagonal: Vec<f64>,
    /// `H_ij`, `i < j`.
    pub upper: Vec<(u32, u32, f64)>,
    /// The right-hand side, `−Jᵀ W r` for a Gauss--Newton step.
    pub rhs: Vec<f64>,
}

impl Operator for NormalSystem {
    fn dim(&self) -> usize {
        self.diagonal.len()
    }

    /// `y = H x`, the upper triangle applied to both sides, in its stored order.
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for ((yi, d), xi) in y.iter_mut().zip(&self.diagonal).zip(x) {
            *yi = d * xi;
        }
        for &(i, j, v) in &self.upper {
            let (i, j) = (i as usize, j as usize);
            y[i] += v * x[j];
            y[j] += v * x[i];
        }
    }
}

/// **A least-squares problem**: residual blocks over a point of a manifold.
pub trait LeastSquares {
    /// The point: a vector, a trajectory of poses, a field of transports; a slice is allowed.
    type Point: ?Sized;

    /// The tangent dimension: an increment's length.
    fn dim(&self) -> usize;

    /// Residual blocks: the weights' length.
    fn blocks(&self) -> usize;

    /// The Gauss--Newton normal equations at `x` under block weights `weights`.
    ///
    /// # Preconditions
    /// `weights.len() == self.blocks()`.
    fn normal_system(&self, x: &Self::Point, weights: &[f64]) -> NormalSystem;

    /// Each block's residual norm `‖r_k(x)‖`, unweighted, into `out`.
    ///
    /// # Preconditions
    /// `out.len() == self.blocks()`.
    fn residual_norms(&self, x: &Self::Point, out: &mut [f64]);

    /// `x ← x ⊕ delta`; returns `‖delta‖∞` as applied.
    ///
    /// # Preconditions
    /// `delta.len() == self.dim()`.
    fn retract(&self, x: &mut Self::Point, delta: &[f64]) -> f64;
}

/// **A solver of the normal equations**: an increment from a [`NormalSystem`].
pub trait IncrementSolver {
    /// Why a solve is refused; [`core::convert::Infallible`] for a solver that always answers.
    type Error;

    /// `δ` with `H δ = rhs`.
    ///
    /// # Errors
    /// The solver's refusal.
    fn solve(&mut self, system: &NormalSystem) -> Result<Vec<f64>, Self::Error>;
}

/// The increment by Jacobi-preconditioned conjugate gradients from zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgIncrement {
    /// Relative residual the solve stops at.
    pub tolerance: f64,
    /// Iterations allowed.
    pub max_iterations: usize,
}

impl IncrementSolver for PcgIncrement {
    /// The solve's report, when its residual is not finite (a breakdown, not a slow solve).
    type Error = Report;

    fn solve(&mut self, system: &NormalSystem) -> Result<Vec<f64>, Report> {
        let mut delta = vec![0.0; system.dim()];
        let report: Report = pcg(
            system,
            &system.rhs,
            &mut delta,
            &Jacobi::of(&system.diagonal),
            self.tolerance,
            self.max_iterations,
            &mut |_| {},
        );
        if report.residual.is_finite() {
            Ok(delta)
        } else {
            Err(report)
        }
    }
}

/// Gauss--Newton's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussNewton {
    /// Iterations allowed.
    pub max_iterations: usize,
    /// The run stops after the first step whose `‖δ‖∞` is below this.
    pub step_tolerance: f64,
}

/// What a Gauss--Newton run did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussNewtonReport {
    /// Iterations taken, the stopping one included.
    pub iterations: usize,
    /// The last applied step's `‖δ‖∞`.
    pub last_step: f64,
    /// Whether a step fell below the tolerance.
    pub converged: bool,
}

/// **Gauss--Newton** on `problem` from `x`: each iteration assembles the normal equations under
/// `weights`, solves the increment with `solver` and retracts, stopping after the first step whose
/// `‖δ‖∞` is below the tolerance.
///
/// # Errors
/// What `solver` refuses.
///
/// # Preconditions
/// `weights.len() == problem.blocks()`.
pub fn gauss_newton<P: LeastSquares, S: IncrementSolver + ?Sized>(
    problem: &P,
    x: &mut P::Point,
    weights: &[f64],
    solver: &mut S,
    settings: GaussNewton,
) -> Result<GaussNewtonReport, S::Error> {
    debug_assert_eq!(weights.len(), problem.blocks());
    let mut last_step = f64::INFINITY;
    for iteration in 0..settings.max_iterations {
        let system = problem.normal_system(x, weights);
        let delta = solver.solve(&system)?;
        last_step = problem.retract(x, &delta);
        if last_step < settings.step_tolerance {
            return Ok(GaussNewtonReport {
                iterations: iteration + 1,
                last_step,
                converged: true,
            });
        }
    }
    Ok(GaussNewtonReport {
        iterations: settings.max_iterations,
        last_step,
        converged: false,
    })
}

/// Levenberg--Marquardt's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevenbergMarquardt {
    /// Trial steps allowed, accepted or rejected.
    pub max_iterations: usize,
    /// The run stops at a trial step whose `‖δ‖∞` is below this, applied if it lowers the cost:
    /// a step that small cannot be measured, accepted or not (Madsen, Nielsen and Tingleff's rule).
    pub step_tolerance: f64,
    /// The run stops when the gradient's `‖Jᵀ W r‖∞` is below this.
    pub gradient_tolerance: f64,
    /// The starting damping `λ₀`, dimensionless under Marquardt's scaling.
    pub initial_damping: f64,
}

/// What a Levenberg--Marquardt run did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmReport {
    /// Trial steps taken.
    pub iterations: usize,
    /// Of them, accepted.
    pub accepted: usize,
    /// `½ Σ w_k ‖r_k‖²` at the returned point.
    pub cost: f64,
    /// The damping `λ` at the end.
    pub damping: f64,
    /// Whether a tolerance stopped the run (not the iteration budget, not a non-finite damping).
    pub converged: bool,
}

/// **Levenberg--Marquardt** on `problem` from `x` under `weights`: each trial solves
/// `(H + λ diag H) δ = −g` with `solver`, evaluates `x ⊕ δ` on a copy, and accepts it iff the cost
/// falls; `λ` follows the gain ratio `ρ` (Nielsen 1999: `λ ← λ max(⅓, 1 − (2ρ − 1)³)` on
/// acceptance, `λ ← νλ, ν ← 2ν` on rejection). The accepted costs therefore never rise. The run
/// stops on the gradient tolerance, on a trial step below the step tolerance, or when `λ` stops
/// being finite (no representable step lowers the cost).
///
/// # Errors
/// What `solver` refuses.
///
/// # Preconditions
/// `weights.len() == problem.blocks()`; `H`'s diagonal is positive (Marquardt's scaling damps each
/// coordinate by its own curvature, so a zero entry is undamped).
pub fn levenberg_marquardt<P, S>(
    problem: &P,
    x: &mut P::Point,
    weights: &[f64],
    solver: &mut S,
    settings: LevenbergMarquardt,
) -> Result<LmReport, S::Error>
where
    P: LeastSquares,
    P::Point: ToOwned,
    <P::Point as ToOwned>::Owned: BorrowMut<P::Point>,
    S: IncrementSolver + ?Sized,
{
    debug_assert_eq!(weights.len(), problem.blocks());
    let mut norms = vec![0.0; problem.blocks()];
    let mut cost = weighted_cost(problem, x, weights, &mut norms);
    let mut system = problem.normal_system(x, weights);
    let (mut damping, mut growth) = (settings.initial_damping, 2.0);
    let (mut iterations, mut accepted, mut converged) = (0, 0, false);
    while iterations < settings.max_iterations {
        if infinity_norm(&system.rhs) < settings.gradient_tolerance {
            converged = true;
            break;
        }
        iterations += 1;
        let delta = solver.solve(&damped(&system, damping))?;
        let predicted = 0.5
            * delta
                .iter()
                .zip(&system.diagonal)
                .zip(&system.rhs)
                .map(|((d, h), g)| d * (damping * h * d + g))
                .sum::<f64>();
        let mut trial = x.to_owned();
        let step = problem.retract(trial.borrow_mut(), &delta);
        let trial_cost = weighted_cost(problem, trial.borrow(), weights, &mut norms);
        let gain = (cost - trial_cost) / predicted;
        let lowers = predicted > 0.0 && gain > 0.0;
        if lowers {
            problem.retract(x, &delta);
            cost = trial_cost;
            accepted += 1;
        }
        if step < settings.step_tolerance {
            converged = true;
            break;
        }
        if lowers {
            let c = 2.0 * gain - 1.0;
            damping *= (1.0 / 3.0f64).max(1.0 - c * c * c);
            growth = 2.0;
            system = problem.normal_system(x, weights);
        } else {
            damping *= growth;
            growth *= 2.0;
            if !damping.is_finite() {
                break;
            }
        }
    }
    Ok(LmReport {
        iterations,
        accepted,
        cost,
        damping,
        converged,
    })
}

/// `½ Σ w_k ‖r_k(x)‖²`, the norms written into `norms`, summed in block order.
fn weighted_cost<P: LeastSquares>(
    problem: &P,
    x: &P::Point,
    weights: &[f64],
    norms: &mut [f64],
) -> f64 {
    problem.residual_norms(x, norms);
    0.5 * norms
        .iter()
        .zip(weights)
        .map(|(r, w)| w * r * r)
        .sum::<f64>()
}

/// The system with its diagonal scaled by `1 + λ`: `H + λ diag H`.
fn damped(system: &NormalSystem, damping: f64) -> NormalSystem {
    NormalSystem {
        diagonal: system
            .diagonal
            .iter()
            .map(|h| h * (1.0 + damping))
            .collect(),
        upper: system.upper.clone(),
        rhs: system.rhs.clone(),
    }
}

/// `max |v_i|`.
fn infinity_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}
