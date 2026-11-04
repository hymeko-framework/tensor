//! **Trust-region Newton with Steihaug's truncated conjugate gradients** (plan
//! 2026-10-19-numerics-optimizers, O4).
//!
//! ```text
//! min_p  m(p) = gᵀp + ½ pᵀHp   subject to  ‖p‖_M ≤ Δ        (steihaug)
//! ρ = (f(x) − f(x ⊕ p)) / (−m(p));   Δ shrinks below ρ = ¼, grows at a boundary step above ¾
//! ```
//!
//! [`steihaug`] runs preconditioned CG from `p = 0` on `H p = −g` and stops at the first of: the
//! forcing tolerance (an interior step), the trust-region boundary, or a direction of non-positive
//! curvature -- which CG cannot handle and the trust region can: the step follows it to the
//! boundary. The `M`-norms come from the recurrences of Conn, Gould and Toint (Algorithm 7.5.1),
//! so only `M⁻¹` -- the [`Preconditioner`] -- is ever applied.
//!
//! A [`TrustRegionModel`] gives the value alone at trial points and its gradient and Hessian (or an
//! approximation) once per accepted point; [`GaussNewtonModel`] is any [`LeastSquares`] problem as
//! one, `H = Jᵀ W J`.

use core::borrow::{Borrow, BorrowMut};

use crate::least_squares::{LeastSquares, NormalSystem};
use crate::pcg::{Jacobi, Preconditioner};
use crate::{dot, norm, Operator};

/// How a Steihaug step ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepEnding {
    /// CG met its forcing tolerance inside the region.
    Interior,
    /// The next CG iterate would have left the region; the step is on its boundary.
    Boundary,
    /// A direction of non-positive curvature, followed to the boundary.
    NegativeCurvature,
    /// The CG budget ran out inside the region.
    Budget,
}

/// A Steihaug step.
#[derive(Clone, Debug, PartialEq)]
pub struct SteihaugStep {
    /// The step `p`.
    pub step: Vec<f64>,
    /// `‖p‖_M`.
    pub m_norm: f64,
    /// CG iterations taken.
    pub iterations: usize,
    /// Why CG stopped.
    pub ending: StepEnding,
}

/// **Steihaug's truncated CG** for `min gᵀp + ½ pᵀHp`, `‖p‖_M ≤ radius`, preconditioned by
/// `preconditioner` (`M⁻¹`); CG stops once `‖r‖ ≤ tolerance`.
///
/// # Preconditions
/// `gradient.len() == h.dim()`; `radius > 0`; the preconditioner is symmetric positive definite.
pub fn steihaug<H: Operator + ?Sized>(
    h: &H,
    gradient: &[f64],
    radius: f64,
    preconditioner: &dyn Preconditioner,
    tolerance: f64,
    max_iterations: usize,
) -> SteihaugStep {
    let n = gradient.len();
    let mut p = vec![0.0; n];
    let mut r = gradient.to_vec();
    let mut y = vec![0.0; n];
    preconditioner.apply(&r, &mut y);
    let mut d: Vec<f64> = y.iter().map(|v| -v).collect();
    let mut ry = dot(&r, &y);
    // ‖p‖²_M, pᵀMd, dᵀMd: with p = 0 and d = −M⁻¹r, dᵀMd = rᵀM⁻¹r.
    let (mut pmp, mut pmd, mut dmd) = (0.0, 0.0, ry);
    let mut hd = vec![0.0; n];
    let finish = |p: Vec<f64>, pmp: f64, iterations, ending| SteihaugStep {
        step: p,
        m_norm: pmp.max(0.0).sqrt(),
        iterations,
        ending,
    };
    if norm(&r) <= tolerance {
        return finish(p, pmp, 0, StepEnding::Interior);
    }
    for iteration in 1..=max_iterations {
        h.apply(&d, &mut hd);
        let curvature = dot(&d, &hd);
        if curvature <= 0.0 {
            let tau = to_boundary(pmp, pmd, dmd, radius);
            axpy(&mut p, tau, &d);
            return finish(p, radius * radius, iteration, StepEnding::NegativeCurvature);
        }
        let alpha = ry / curvature;
        let next_pmp = pmp + 2.0 * alpha * pmd + alpha * alpha * dmd;
        if next_pmp >= radius * radius {
            let tau = to_boundary(pmp, pmd, dmd, radius);
            axpy(&mut p, tau, &d);
            return finish(p, radius * radius, iteration, StepEnding::Boundary);
        }
        axpy(&mut p, alpha, &d);
        axpy(&mut r, alpha, &hd);
        pmp = next_pmp;
        if norm(&r) <= tolerance {
            return finish(p, pmp, iteration, StepEnding::Interior);
        }
        preconditioner.apply(&r, &mut y);
        let next_ry = dot(&r, &y);
        let beta = next_ry / ry;
        ry = next_ry;
        pmd = beta * (pmd + alpha * dmd);
        dmd = ry + beta * beta * dmd;
        for (di, yi) in d.iter_mut().zip(&y) {
            *di = -yi + beta * *di;
        }
    }
    finish(p, pmp, max_iterations, StepEnding::Budget)
}

/// The `τ ≥ 0` with `‖p + τ d‖_M = radius`, from `‖p‖²_M`, `pᵀMd`, `dᵀMd`.
fn to_boundary(pmp: f64, pmd: f64, dmd: f64, radius: f64) -> f64 {
    let discriminant = pmd * pmd + dmd * (radius * radius - pmp);
    (-pmd + discriminant.max(0.0).sqrt()) / dmd
}

/// `y ← y + a x`.
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// **A trust-region model** of an objective over a point of a manifold.
pub trait TrustRegionModel {
    /// The point; a slice is allowed.
    type Point: ?Sized;
    /// The Hessian, or its approximation, at a point.
    type Hessian: Operator;
    /// Its preconditioner.
    type Preconditioner: Preconditioner;

    /// The tangent dimension.
    fn dim(&self) -> usize;
    /// `f(x)`.
    fn value(&self, x: &Self::Point) -> f64;
    /// `∇f(x)` into `gradient`, and the Hessian there.
    fn model(&self, x: &Self::Point, gradient: &mut [f64]) -> Self::Hessian;
    /// The preconditioner of a Hessian.
    fn preconditioner(&self, hessian: &Self::Hessian) -> Self::Preconditioner;
    /// `x ← x ⊕ step`; returns `‖step‖∞` as applied.
    fn retract(&self, x: &mut Self::Point, step: &[f64]) -> f64;
}

/// **A least-squares problem as a trust-region model**: `f = ½ Σ w_k ‖r_k‖²`, `∇f = Jᵀ W r`,
/// `H = Jᵀ W J` from its normal equations, Jacobi-preconditioned.
pub struct GaussNewtonModel<'a, P> {
    /// The problem.
    pub problem: &'a P,
    /// The block weights.
    pub weights: &'a [f64],
}

impl<P: LeastSquares> TrustRegionModel for GaussNewtonModel<'_, P> {
    type Point = P::Point;
    type Hessian = NormalSystem;
    type Preconditioner = Jacobi;

    fn dim(&self) -> usize {
        self.problem.dim()
    }

    fn value(&self, x: &P::Point) -> f64 {
        let mut norms = vec![0.0; self.problem.blocks()];
        self.problem.residual_norms(x, &mut norms);
        0.5 * norms
            .iter()
            .zip(self.weights)
            .map(|(r, w)| w * r * r)
            .sum::<f64>()
    }

    fn model(&self, x: &P::Point, gradient: &mut [f64]) -> NormalSystem {
        let system = self.problem.normal_system(x, self.weights);
        for (g, b) in gradient.iter_mut().zip(&system.rhs) {
            *g = -b;
        }
        system
    }

    fn preconditioner(&self, hessian: &NormalSystem) -> Jacobi {
        Jacobi::of(&hessian.diagonal)
    }

    fn retract(&self, x: &mut P::Point, step: &[f64]) -> f64 {
        self.problem.retract(x, step)
    }
}

/// The trust-region method's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrustRegion {
    /// Trial steps allowed, accepted or rejected.
    pub max_iterations: usize,
    /// The run stops when `‖∇f‖∞` is below this.
    pub gradient_tolerance: f64,
    /// The run stops at a trial step whose `‖p‖∞` is below this (as O2's Levenberg--Marquardt).
    pub step_tolerance: f64,
    /// `Δ₀`, in the preconditioner's norm.
    pub initial_radius: f64,
    /// `Δ_max`.
    pub max_radius: f64,
    /// A step is accepted iff `ρ` exceeds this, in `[0, ¼)`.
    pub acceptance: f64,
    /// CG iterations one step may take.
    pub cg_max_iterations: usize,
    /// `η_max` of the forcing tolerance `min(η_max, √‖g‖) ‖g‖`: Nocedal and Wright use ½. The
    /// `√‖g‖` term is absolute, so a problem whose gradient stays far above 1 -- a pose graph --
    /// lives at `η_max` and needs a tighter cap to take Newton-like steps.
    pub max_forcing: f64,
}

/// What a trust-region run did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrustRegionReport {
    /// Trial steps taken.
    pub iterations: usize,
    /// Of them, accepted.
    pub accepted: usize,
    /// CG iterations over all steps.
    pub cg_iterations: usize,
    /// `f` at the returned point.
    pub value: f64,
    /// `Δ` at the end.
    pub radius: f64,
    /// Whether a tolerance stopped the run.
    pub converged: bool,
}

/// **The trust-region method** on `model` from `x` (Nocedal and Wright, Algorithm 4.1, with
/// [`steihaug`] steps and the forcing tolerance `min(η_max, √‖g‖) ‖g‖`).
///
/// # Preconditions
/// `0 ≤ acceptance < ¼`; `0 < initial_radius ≤ max_radius`.
pub fn trust_region<M>(model: &M, x: &mut M::Point, settings: TrustRegion) -> TrustRegionReport
where
    M: TrustRegionModel,
    M::Point: ToOwned,
    <M::Point as ToOwned>::Owned: BorrowMut<M::Point>,
{
    let mut gradient = vec![0.0; model.dim()];
    let mut value = model.value(x);
    let mut hessian = model.model(x, &mut gradient);
    let mut product = vec![0.0; gradient.len()];
    let mut radius = settings.initial_radius;
    let (mut iterations, mut accepted, mut cg_iterations) = (0, 0, 0);
    let mut converged = false;
    while iterations < settings.max_iterations {
        let g_norm = norm(&gradient);
        if gradient.iter().fold(0.0f64, |m, g| m.max(g.abs())) < settings.gradient_tolerance {
            converged = true;
            break;
        }
        iterations += 1;
        let forcing = settings.max_forcing.min(g_norm.sqrt()) * g_norm;
        let step = steihaug(
            &hessian,
            &gradient,
            radius,
            &model.preconditioner(&hessian),
            forcing,
            settings.cg_max_iterations,
        );
        cg_iterations += step.iterations;
        hessian.apply(&step.step, &mut product);
        let predicted = -(dot(&gradient, &step.step) + 0.5 * dot(&step.step, &product));
        let mut trial = x.to_owned();
        let size = model.retract(trial.borrow_mut(), &step.step);
        let trial_value = model.value(trial.borrow());
        let rho = (value - trial_value) / predicted;
        let lowers = predicted > 0.0 && rho > settings.acceptance;
        if lowers {
            model.retract(x, &step.step);
            value = trial_value;
            accepted += 1;
        }
        if size < settings.step_tolerance {
            converged = true;
            break;
        }
        // A non-finite ratio (a trial value that is not finite) shrinks the region.
        let good = rho >= 0.25;
        if !good {
            radius = 0.25 * step.m_norm.min(radius);
        } else if rho > 0.75 && step.ending != StepEnding::Interior {
            radius = (2.0 * radius).min(settings.max_radius);
        }
        if lowers {
            hessian = model.model(x, &mut gradient);
        }
    }
    TrustRegionReport {
        iterations,
        accepted,
        cg_iterations,
        value,
        radius,
        converged,
    }
}
