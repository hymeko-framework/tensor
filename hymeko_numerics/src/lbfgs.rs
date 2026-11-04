//! **Limited-memory BFGS** on a smooth objective over a manifold (plan
//! 2026-10-19-numerics-optimizers, O3).
//!
//! ```text
//! d = −H g  (two-loop recursion over the last m pairs (s, y), H₀ = γ I, γ = sᵀy / yᵀy)
//! x ← x ⊕ t d,   t by a weak-Wolfe line search:  f(x ⊕ t d) ≤ f + c₁ t gᵀd,   g(t)ᵀd ≥ c₂ gᵀd
//! ```
//!
//! An [`Objective`] states its value and gradient in a tangent space, a retraction, and how a
//! tangent vector is carried to the retracted point. The default transport is the identity: exact
//! in `ℝⁿ`, and the standard choice in a Lie group whose updates are left exponentials, where every
//! tangent vector is already written in the algebra. Trial points are evaluated on copies, so `x`
//! only ever takes accepted steps.
//!
//! # The precision floor
//! The line search compares values of `f`. Near a minimum the Armijo decrease `c₁ t gᵀd` shrinks
//! like `‖g‖²` and eventually falls below the resolution of `f` itself; the search then fails by
//! rounding and the run ends unconverged at the best point it reached. Measured on a quadratic with
//! `κ = 10⁴` and `|f| ≈ 1.3`: the floor sits near `‖g‖ ≈ 3·10⁻⁷`. A gradient tolerance belongs above
//! it.

use core::borrow::{Borrow, BorrowMut};
use std::collections::VecDeque;

use crate::dot;

/// **A smooth objective** over a point of a manifold.
pub trait Objective {
    /// The point; a slice is allowed.
    type Point: ?Sized;

    /// The tangent dimension.
    fn dim(&self) -> usize;

    /// `f(x)`, with `∇f(x)` written into `gradient`.
    ///
    /// # Preconditions
    /// `gradient.len() == self.dim()`.
    fn value_and_gradient(&self, x: &Self::Point, gradient: &mut [f64]) -> f64;

    /// `x ← x ⊕ step`; returns `‖step‖∞` as applied.
    fn retract(&self, x: &mut Self::Point, step: &[f64]) -> f64;

    /// Carry the tangent vector `vector` at `x` to `x ⊕ step`. The identity by default.
    fn transport(&self, _x: &Self::Point, _step: &[f64], _vector: &mut [f64]) {}
}

/// L-BFGS's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lbfgs {
    /// Stored pairs `m`.
    pub memory: usize,
    /// Iterations (accepted steps) allowed.
    pub max_iterations: usize,
    /// The run stops when `‖∇f‖∞` is below this.
    pub gradient_tolerance: f64,
    /// Armijo's sufficient-decrease constant `c₁`.
    pub armijo: f64,
    /// Wolfe's curvature constant `c₂`, `c₁ < c₂ < 1`.
    pub curvature: f64,
    /// Trial points one line search may evaluate.
    pub max_line_search: usize,
}

/// What an L-BFGS run did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsReport {
    /// Accepted steps.
    pub iterations: usize,
    /// Objective evaluations, the first included.
    pub evaluations: usize,
    /// `f` at the returned point.
    pub value: f64,
    /// `‖∇f‖∞` there.
    pub gradient_norm: f64,
    /// Whether the gradient tolerance stopped the run (not the budget, not a failed line search).
    pub converged: bool,
}

/// One stored curvature pair: `s`, `y`, `1 / sᵀy`.
struct Pair {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
}

/// **L-BFGS** on `objective` from `x`.
///
/// # Preconditions
/// `0 < armijo < curvature < 1`; `memory > 0`.
pub fn lbfgs<O>(objective: &O, x: &mut O::Point, settings: Lbfgs) -> LbfgsReport
where
    O: Objective,
    O::Point: ToOwned,
    <O::Point as ToOwned>::Owned: BorrowMut<O::Point>,
{
    debug_assert!(0.0 < settings.armijo && settings.armijo < settings.curvature);
    debug_assert!(settings.curvature < 1.0 && settings.memory > 0);
    let n = objective.dim();
    let mut gradient = vec![0.0; n];
    let mut value = objective.value_and_gradient(x, &mut gradient);
    let mut evaluations = 1;
    let mut pairs: VecDeque<Pair> = VecDeque::with_capacity(settings.memory);
    let (mut iterations, mut converged) = (0, false);
    while iterations < settings.max_iterations {
        if infinity_norm(&gradient) < settings.gradient_tolerance {
            converged = true;
            break;
        }
        let mut direction = two_loop(&pairs, &gradient);
        let mut slope = dot(&gradient, &direction);
        if slope >= 0.0 {
            // Not a descent direction (a pair gone stale on a curved manifold): restart.
            pairs.clear();
            direction = two_loop(&pairs, &gradient);
            slope = dot(&gradient, &direction);
        }
        let search = LineSearch {
            objective,
            value,
            slope,
            direction: &direction,
            settings,
        };
        let Some(accepted) = search.run(x, &mut evaluations) else {
            break;
        };
        let mut previous = gradient;
        objective.transport(x, &accepted.step, &mut previous);
        for pair in &mut pairs {
            objective.transport(x, &accepted.step, &mut pair.s);
            objective.transport(x, &accepted.step, &mut pair.y);
        }
        let mut s = accepted.step.clone();
        objective.transport(x, &accepted.step, &mut s);
        objective.retract(x, &accepted.step);
        let y: Vec<f64> = accepted
            .gradient
            .iter()
            .zip(&previous)
            .map(|(a, b)| a - b)
            .collect();
        let sy = dot(&s, &y);
        if sy > 0.0 {
            if pairs.len() == settings.memory {
                pairs.pop_front();
            }
            pairs.push_back(Pair {
                s,
                y,
                rho: 1.0 / sy,
            });
        }
        gradient = accepted.gradient;
        value = accepted.value;
        iterations += 1;
    }
    LbfgsReport {
        iterations,
        evaluations,
        value,
        gradient_norm: infinity_norm(&gradient),
        converged,
    }
}

/// `d = −H g` by the two-loop recursion; with no pairs, the steepest descent scaled to a unit
/// Euclidean length when longer.
fn two_loop(pairs: &VecDeque<Pair>, gradient: &[f64]) -> Vec<f64> {
    let mut q = gradient.to_vec();
    let Some(last) = pairs.back() else {
        let scale = 1.0 / dot(gradient, gradient).sqrt().max(1.0);
        return q.iter().map(|g| -scale * g).collect();
    };
    let mut alphas = Vec::with_capacity(pairs.len());
    for pair in pairs.iter().rev() {
        let alpha = pair.rho * dot(&pair.s, &q);
        for (qi, yi) in q.iter_mut().zip(&pair.y) {
            *qi -= alpha * yi;
        }
        alphas.push(alpha);
    }
    let gamma = dot(&last.s, &last.y) / dot(&last.y, &last.y);
    for qi in &mut q {
        *qi *= gamma;
    }
    for (pair, alpha) in pairs.iter().zip(alphas.iter().rev()) {
        let beta = pair.rho * dot(&pair.y, &q);
        for (qi, si) in q.iter_mut().zip(&pair.s) {
            *qi += (alpha - beta) * si;
        }
    }
    q.iter().map(|v| -v).collect()
}

/// An accepted line-search step: the tangent step `t d`, and `f` and `∇f` at `x ⊕ t d`.
struct Accepted {
    step: Vec<f64>,
    value: f64,
    gradient: Vec<f64>,
}

/// One weak-Wolfe line search along `direction` from a point of value `value` and slope
/// `slope = gᵀd < 0`, by bisection of the bracket `[lo, hi]` (Lewis and Overton).
struct LineSearch<'a, O> {
    objective: &'a O,
    value: f64,
    slope: f64,
    direction: &'a [f64],
    settings: Lbfgs,
}

impl<O> LineSearch<'_, O>
where
    O: Objective,
    O::Point: ToOwned,
    <O::Point as ToOwned>::Owned: BorrowMut<O::Point>,
{
    /// The first step meeting both Wolfe conditions, or `None` within the trial budget.
    fn run(&self, x: &O::Point, evaluations: &mut usize) -> Option<Accepted> {
        let (mut lo, mut hi, mut t) = (0.0f64, f64::INFINITY, 1.0f64);
        let mut gradient = vec![0.0; self.direction.len()];
        for _ in 0..self.settings.max_line_search {
            let step: Vec<f64> = self.direction.iter().map(|d| t * d).collect();
            let mut trial = x.to_owned();
            self.objective.retract(trial.borrow_mut(), &step);
            let value = self
                .objective
                .value_and_gradient(trial.borrow(), &mut gradient);
            *evaluations += 1;
            let sufficient = value <= self.value + self.settings.armijo * t * self.slope;
            // A non-finite value is not sufficient and shrinks the step.
            if !sufficient {
                hi = t;
            } else {
                let mut carried = self.direction.to_vec();
                self.objective.transport(x, &step, &mut carried);
                if dot(&gradient, &carried) < self.settings.curvature * self.slope {
                    lo = t;
                } else {
                    return Some(Accepted {
                        step,
                        value,
                        gradient,
                    });
                }
            }
            t = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                2.0 * lo
            };
        }
        None
    }
}

/// `max |v_i|`.
fn infinity_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}
