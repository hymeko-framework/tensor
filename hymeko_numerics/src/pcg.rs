//! **Preconditioned conjugate gradients** on a symmetric positive (semi)definite [`Operator`].
//!
//! The crate's other Krylov solvers serve the shifted skew-symmetric systems of the step; this one
//! serves the symmetric ones a least-squares problem produces -- normal equations, Hessians of
//! defect losses -- shaped by a [`Preconditioner`]. Moved here 2026-10-19 from
//! `nagare_autoholonomy::relax`, where it was written because no `f64` preconditioned CG existed in
//! the workspace (plan 2026-10-19-numerics-optimizers, O1).
//!
//! # Contract
//! The operator is symmetric and positive definite on the Krylov space the iteration explores; a
//! consistent positive-semidefinite system converges as well. The preconditioner is symmetric
//! positive definite. Every sum runs in slice order, single-threaded: the iterates are a function of
//! the inputs alone.

use crate::{dot, norm, Operator, Report};

/// `z = M⁻¹ r`, symmetric positive definite.
pub trait Preconditioner {
    /// Write `M⁻¹ residual` into `out`.
    ///
    /// # Preconditions
    /// `residual.len() == out.len()`, the operator's dimension.
    fn apply(&self, residual: &[f64], out: &mut [f64]);
}

/// No preconditioning: `z = r`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Preconditioner for Identity {
    fn apply(&self, residual: &[f64], out: &mut [f64]) {
        out.copy_from_slice(residual);
    }
}

/// The diagonal (Jacobi) preconditioner: `z = D⁻¹ r`; a non-positive diagonal entry leaves its
/// coordinate still.
#[derive(Clone, Debug, PartialEq)]
pub struct Jacobi {
    inverse: Vec<f64>,
}

impl Jacobi {
    /// `D⁻¹` of a diagonal.
    #[must_use]
    pub fn of(diagonal: &[f64]) -> Self {
        Self {
            inverse: diagonal
                .iter()
                .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
                .collect(),
        }
    }
}

impl Preconditioner for Jacobi {
    fn apply(&self, residual: &[f64], out: &mut [f64]) {
        for ((z, r), d) in out.iter_mut().zip(residual).zip(&self.inverse) {
            *z = r * d;
        }
    }
}

/// Two preconditioners added, `z = A r + B r`: a smoother and a coarse correction, BPX's form.
#[derive(Clone, Debug)]
pub struct Additive<A, B>(pub A, pub B);

impl<A: Preconditioner, B: Preconditioner> Preconditioner for Additive<A, B> {
    fn apply(&self, residual: &[f64], out: &mut [f64]) {
        let mut second = vec![0.0; out.len()];
        self.0.apply(residual, out);
        self.1.apply(residual, &mut second);
        for (z, s) in out.iter_mut().zip(&second) {
            *z += s;
        }
    }
}

/// What one PCG iteration reports to an observer: the iteration and the recursive residual's
/// relative norm `‖r‖ / ‖b‖`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgProgress {
    /// Iterations completed.
    pub iteration: usize,
    /// `‖r‖ / ‖b‖`, the recursive residual.
    pub relative_residual: f64,
}

/// **Solve `A x = b`** from `x` by preconditioned conjugate gradients, to `‖r‖ / ‖b‖ ≤ tol` or
/// `max_iterations`; `observe` sees every iteration's recursive residual.
///
/// # Preconditions
/// `b.len() == x.len() == a.dim()`; see the module's contract on `a` and `m`.
///
/// # Postconditions
/// The report's residual is the true one, `‖b − A x‖ / ‖b‖`, recomputed at the end.
pub fn pcg<O: Operator>(
    a: &O,
    b: &[f64],
    x: &mut [f64],
    m: &dyn Preconditioner,
    tol: f64,
    max_iterations: usize,
    observe: &mut dyn FnMut(&PcgProgress),
) -> Report {
    let n = a.dim();
    debug_assert!(b.len() == n && x.len() == n);
    let scale = norm(b).max(f64::MIN_POSITIVE);
    let (mut residual, mut z, mut product) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    a.apply(x, &mut product);
    for ((r, bi), ax) in residual.iter_mut().zip(b).zip(&product) {
        *r = bi - ax;
    }
    m.apply(&residual, &mut z);
    let mut direction = z.clone();
    let mut rho = dot(&residual, &z);
    let mut relative = norm(&residual) / scale;
    let mut iterations = 0;
    while relative > tol && iterations < max_iterations {
        a.apply(&direction, &mut product);
        let curvature = dot(&direction, &product);
        if curvature <= 0.0 {
            break;
        }
        let alpha = rho / curvature;
        for (((xi, r), p), ap) in x
            .iter_mut()
            .zip(residual.iter_mut())
            .zip(&direction)
            .zip(&product)
        {
            *xi += alpha * p;
            *r -= alpha * ap;
        }
        iterations += 1;
        relative = norm(&residual) / scale;
        if relative <= tol {
            // The recursive residual drifts from the true one; stop on the true one.
            a.apply(x, &mut product);
            relative = true_relative(b, &product, scale);
        }
        observe(&PcgProgress {
            iteration: iterations,
            relative_residual: relative,
        });
        m.apply(&residual, &mut z);
        let next = dot(&residual, &z);
        let beta = next / rho;
        rho = next;
        for (p, zi) in direction.iter_mut().zip(&z) {
            *p = zi + beta * *p;
        }
    }
    a.apply(x, &mut product);
    let true_residual = true_relative(b, &product, scale);
    Report {
        iterations,
        residual: true_residual,
        converged: true_residual <= tol,
    }
}

/// `‖b − A x‖ / scale`, from `A x` in `product`, summed in slice order.
fn true_relative(b: &[f64], product: &[f64], scale: f64) -> f64 {
    b.iter()
        .zip(product)
        .map(|(bi, ax)| (bi - ax) * (bi - ax))
        .sum::<f64>()
        .sqrt()
        / scale
}
