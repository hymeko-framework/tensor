//! Forward-mode dual numbers, carrying a value and its gradient with respect to the decision
//! variables.
//!
//! # Why forward mode and not finite differences
//!
//! A constrained solver wants gradients of the objective and of every constraint. Taking them by
//! finite differences costs one extra evaluation per variable and, worse, is inexact by an amount
//! nobody can bound: too large a step and the truncation shows, too small and the cancellation does.
//! For a design problem whose equality constraint is meant to hold to `1e-9`, a gradient good to
//! `1e-6` is the thing that stops the solve converging.
//!
//! Forward mode carries the derivative alongside the value through every operation, so the gradient
//! is exact to rounding and costs one pass. With a handful of decision variables — which is what an
//! inverse-design problem has — it is also cheaper than reverse mode and far simpler: no tape, no
//! second pass, and no allocation beyond the gradient vector itself.
//!
//! It goes through a linear solve too: a consumer can run Gaussian elimination over these, so the
//! sensitivity of a solution to a parameter comes out of the same arithmetic that produced the
//! solution, rather than from a separately derived formula that could disagree with it.
//!
//! Moved unchanged from transformation's `hymeko_optimize::dual` (2026-11-01), whose projection
//! and network solve consume it; [`crate::optimize`] is the constrained solver built on it.

/// A value and its gradient with respect to the decision variables.
#[derive(Debug, Clone, PartialEq)]
pub struct Dual {
    pub value: f64,
    /// One entry per decision variable, in the problem's own order.
    pub grad: Vec<f64>,
}

impl Dual {
    /// A constant: no dependence on any decision variable.
    pub fn constant(value: f64, n: usize) -> Self {
        Self {
            value,
            grad: vec![0.0; n],
        }
    }

    /// Decision variable `k` of `n`, at `value`.
    pub fn variable(value: f64, k: usize, n: usize) -> Self {
        let mut grad = vec![0.0; n];
        grad[k] = 1.0;
        Self { value, grad }
    }

    pub fn zero(n: usize) -> Self {
        Self::constant(0.0, n)
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value + rhs.value,
            grad: self
                .grad
                .iter()
                .zip(&rhs.grad)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self {
            value: self.value - rhs.value,
            grad: self
                .grad
                .iter()
                .zip(&rhs.grad)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// `(uv)' = u'v + uv'`.
    pub fn mul(&self, rhs: &Self) -> Self {
        Self {
            value: self.value * rhs.value,
            grad: self
                .grad
                .iter()
                .zip(&rhs.grad)
                .map(|(a, b)| a * rhs.value + self.value * b)
                .collect(),
        }
    }

    /// `(u/v)' = (u'v - uv') / v²`.
    pub fn div(&self, rhs: &Self) -> Self {
        let denominator = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            grad: self
                .grad
                .iter()
                .zip(&rhs.grad)
                .map(|(a, b)| (a * rhs.value - self.value * b) / denominator)
                .collect(),
        }
    }

    pub fn scale(&self, k: f64) -> Self {
        Self {
            value: self.value * k,
            grad: self.grad.iter().map(|g| g * k).collect(),
        }
    }

    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// How many decision variables the gradient covers.
    pub fn width(&self) -> usize {
        self.grad.len()
    }
}
