//! Phase 2 — reverse-mode (adjoint) differentiation for rotor chains.
//!
//! A central-difference gradient over `k` angles costs `2k` forward evaluations,
//! each of which applies `k` rotors: `O(k²)` rotor applications, measured at
//! **63.9× the forward pass** for `k = 32`
//! (`nagare/reports/2026-08-16-nagare-cpu-memory-benchmark.md` §6h). The reverse
//! sweep below costs `2k` rotor applications — one forward, one backward —
//! and yields every partial derivative, plus the gradient with respect to the
//! input vector at no extra cost.
//!
//! ## The recurrence
//!
//! With `x₀ = v`, `xᵢ = Rᵢ xᵢ₋₁ R̃ᵢ`, `w = x_k`, and each rotor a rotation of
//! `θᵢ` about the unit axis `ωᵢ`:
//!
//! ```text
//!     ∂xᵢ/∂θᵢ    = ωᵢ × xᵢ
//!     ∂xᵢ/∂xᵢ₋₁  = Rᵢ (·) R̃ᵢ
//! ```
//!
//! Seeding `ā_k = ∂L/∂w` and sweeping backwards:
//!
//! ```text
//!     ∂L/∂θᵢ = āᵢ · (ωᵢ × xᵢ)
//!     āᵢ₋₁   = R̃ᵢ āᵢ Rᵢ
//! ```
//!
//! and `ā₀ = ∂L/∂v` falls out of the last step.
//!
//! ## Why the metric does not appear
//!
//! The angles are coordinates on a product of one-parameter subgroups, and the
//! bi-invariant metric is the identity in that basis — a unit change in `θᵢ` is a
//! unit geodesic step, measured for every link of an 8-chain
//! (`nagare/scripts/dev/probe_2026-08-16_rotor_metric.py`). So the vector
//! returned here **is** the Riemannian (natural) gradient with respect to the
//! group metric; there is no metric tensor to invert. The Fisher metric induced
//! by an output distribution is a different object and is *not* the identity —
//! see §6i of the report before reaching for it.
//!
//! Rotors are built directly as `cos(θ/2) + sin(θ/2)·B` rather than through
//! [`crate::cayley_to_unit_quat`], whose Cayley parameterisation diverges at
//! `θ → ±π`.

use core::fmt;

use crate::algebra::{Multivector, Signature};

/// Dimension of the algebra these chains live in: `Cl(3,0)`.
const N: usize = 3;

/// A caller error in the chain API.
///
/// Returned rather than panicked: both variants are wrong *input*, and §6.4
/// forbids swallowing an error path or turning it into a silent default.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainError {
    /// An axis was not unit length. Rotors built from a non-unit axis are not
    /// rotations, and every downstream invariant would be quietly wrong.
    NonUnitAxis {
        /// Position of the offending axis.
        index: usize,
        /// Its measured Euclidean norm.
        norm: f64,
    },
    /// The angle slice does not match the number of links.
    AngleCount {
        /// Links in the chain.
        expected: usize,
        /// Angles supplied.
        got: usize,
    },
    /// A supplied value was not finite.
    NotFinite {
        /// What was being validated.
        what: &'static str,
    },
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainError::NonUnitAxis { index, norm } => write!(
                f,
                "axis {index} has norm {norm}, expected unit length within 1e-9"
            ),
            ChainError::AngleCount { expected, got } => {
                write!(f, "chain has {expected} links but {got} angles were given")
            }
            ChainError::NotFinite { what } => write!(f, "{what} contained a non-finite value"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Output of one forward-plus-adjoint sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct Adjoint {
    /// The transported vector `w = x_k`.
    pub value: [f64; 3],
    /// `∂L/∂θᵢ`, one entry per link, in chain order.
    pub grad_angles: Vec<f64>,
    /// `∂L/∂v` — produced by the same backward sweep, at no extra cost.
    pub grad_input: [f64; 3],
}

/// A chain of rotations about fixed unit axes, one angle per link.
///
/// Link `0` is applied first. This is the shape a parallel-transport pass, a
/// kinematic chain, or a stacked rotor layer has.
#[derive(Debug, Clone, PartialEq)]
pub struct RotorChain {
    axes: Vec<[f64; 3]>,
    sig: Signature,
}

impl RotorChain {
    /// Build a chain from unit axes.
    ///
    /// # Preconditions
    /// Every axis is finite and unit length to within `1e-9`.
    ///
    /// # Errors
    /// [`ChainError::NonUnitAxis`] or [`ChainError::NotFinite`].
    pub fn new(axes: Vec<[f64; 3]>) -> Result<Self, ChainError> {
        for (index, a) in axes.iter().enumerate() {
            if !a.iter().all(|c| c.is_finite()) {
                return Err(ChainError::NotFinite { what: "axis" });
            }
            let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if (norm - 1.0).abs() > 1e-9 {
                return Err(ChainError::NonUnitAxis { index, norm });
            }
        }
        Ok(Self {
            axes,
            sig: Signature::euclidean(N),
        })
    }

    /// Number of links.
    pub fn len(&self) -> usize {
        self.axes.len()
    }

    /// Whether the chain has no links (the identity map).
    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    /// The axes, in chain order.
    pub fn axes(&self) -> &[[f64; 3]] {
        &self.axes
    }

    /// Rotor for link `i` at angle `theta`: `cos(θ/2) + sin(θ/2)·B`, where `B`
    /// is the bivector dual to the axis (`e₂e₃ ← x`, `e₃e₁ ← y`, `e₁e₂ ← z`).
    fn rotor(&self, i: usize, theta: f64) -> Multivector {
        let axis = self.axes[i];
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let mut r = Multivector::zero(N);
        r.components[0] = c;
        r.components[0b110] = -s * axis[0];
        r.components[0b101] = s * axis[1];
        r.components[0b011] = -s * axis[2];
        r
    }

    fn to_mv(v: [f64; 3]) -> Multivector {
        let mut mv = Multivector::zero(N);
        mv.components[0b001] = v[0];
        mv.components[0b010] = v[1];
        mv.components[0b100] = v[2];
        mv
    }

    fn from_mv(mv: &Multivector) -> [f64; 3] {
        [
            mv.components[0b001],
            mv.components[0b010],
            mv.components[0b100],
        ]
    }

    /// `R x R̃` — the rotor acting on a vector.
    fn rotate(&self, r: &Multivector, x: &Multivector) -> Multivector {
        r.geo(x, &self.sig).geo(&r.reverse(), &self.sig)
    }

    /// `R̃ x R` — the inverse action, used by the backward sweep.
    fn rotate_inverse(&self, r: &Multivector, x: &Multivector) -> Multivector {
        r.reverse().geo(x, &self.sig).geo(r, &self.sig)
    }

    fn check_angles(&self, angles: &[f64]) -> Result<(), ChainError> {
        if angles.len() != self.axes.len() {
            return Err(ChainError::AngleCount {
                expected: self.axes.len(),
                got: angles.len(),
            });
        }
        if !angles.iter().all(|t| t.is_finite()) {
            return Err(ChainError::NotFinite { what: "angles" });
        }
        Ok(())
    }

    /// Apply the whole chain to `v`.
    ///
    /// # Postconditions
    /// The result has the same Euclidean norm as `v` to floating-point
    /// tolerance — every link is a rotation.
    ///
    /// # Errors
    /// [`ChainError::AngleCount`] or [`ChainError::NotFinite`].
    pub fn apply(&self, angles: &[f64], v: [f64; 3]) -> Result<[f64; 3], ChainError> {
        self.check_angles(angles)?;
        let mut x = Self::to_mv(v);
        for (i, &theta) in angles.iter().enumerate() {
            x = self.rotate(&self.rotor(i, theta), &x);
        }
        Ok(Self::from_mv(&x))
    }

    /// Forward sweep plus reverse sweep: every `∂L/∂θᵢ` and `∂L/∂v` in
    /// **`O(k)`**, given the upstream gradient `∂L/∂w`.
    ///
    /// # Preconditions
    /// `angles.len() == self.len()`; all inputs finite.
    ///
    /// # Postconditions
    /// `grad_angles.len() == self.len()`. `value` equals [`Self::apply`] on the
    /// same inputs. Cost is `2k` rotor applications and `k` cross products —
    /// independent of how many angles are being differentiated, which is the
    /// property the finite difference does not have.
    ///
    /// # Errors
    /// [`ChainError::AngleCount`] or [`ChainError::NotFinite`].
    pub fn adjoint(
        &self,
        angles: &[f64],
        v: [f64; 3],
        grad_out: [f64; 3],
    ) -> Result<Adjoint, ChainError> {
        self.check_angles(angles)?;
        if !grad_out.iter().all(|c| c.is_finite()) {
            return Err(ChainError::NotFinite { what: "grad_out" });
        }
        let k = self.axes.len();

        // Forward: keep every intermediate. This is the O(k) memory the reverse
        // sweep trades for its O(k) time.
        let mut trace: Vec<Multivector> = Vec::with_capacity(k + 1);
        trace.push(Self::to_mv(v));
        for (i, &theta) in angles.iter().enumerate() {
            let next = self.rotate(&self.rotor(i, theta), &trace[i]);
            trace.push(next);
        }

        // Backward: one pass, last link first.
        let mut adj = Self::to_mv(grad_out);
        let mut grad_angles = vec![0.0f64; k];
        for i in (0..k).rev() {
            let x = Self::from_mv(&trace[i + 1]);
            let a = Self::from_mv(&adj);
            let w = self.axes[i];
            // ∂L/∂θᵢ = ā · (ω × x)
            grad_angles[i] = a[0] * (w[1] * x[2] - w[2] * x[1])
                + a[1] * (w[2] * x[0] - w[0] * x[2])
                + a[2] * (w[0] * x[1] - w[1] * x[0]);
            adj = self.rotate_inverse(&self.rotor(i, angles[i]), &adj);
        }

        Ok(Adjoint {
            value: Self::from_mv(&trace[k]),
            grad_angles,
            grad_input: Self::from_mv(&adj),
        })
    }

    /// Central-difference gradient with respect to the angles — the reference
    /// the adjoint is checked against, and the tool a caller uses to
    /// gradient-check its own objective.
    ///
    /// Second-order accurate, and therefore `O(k²)` in rotor applications: `2k`
    /// forward sweeps of `k` links each. It ships beside [`Self::adjoint`] so
    /// the two cannot drift apart unnoticed.
    ///
    /// # Preconditions
    /// `h > 0` and finite. A step near `√ε` (≈1e-8 for `f64`) balances
    /// truncation against cancellation; `1e-5` is a safe default for angles of
    /// order 1.
    ///
    /// # Errors
    /// [`ChainError::AngleCount`], [`ChainError::NotFinite`].
    pub fn finite_difference_gradient(
        &self,
        angles: &[f64],
        v: [f64; 3],
        grad_out: [f64; 3],
        h: f64,
    ) -> Result<Vec<f64>, ChainError> {
        self.check_angles(angles)?;
        if !(h.is_finite() && h > 0.0) {
            return Err(ChainError::NotFinite {
                what: "step size h",
            });
        }
        let k = self.axes.len();
        let mut probe = angles.to_vec();
        let mut grad = vec![0.0f64; k];
        let dot = |a: [f64; 3]| a[0] * grad_out[0] + a[1] * grad_out[1] + a[2] * grad_out[2];
        for i in 0..k {
            let keep = probe[i];
            probe[i] = keep + h;
            let up = dot(self.apply(&probe, v)?);
            probe[i] = keep - h;
            let down = dot(self.apply(&probe, v)?);
            probe[i] = keep;
            grad[i] = (up - down) / (2.0 * h);
        }
        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    #[test]
    fn single_link_adjoint_matches_the_closed_form() {
        // dL/dθ = ā · (ω × w) for L = ā · w.
        let axis = unit([0.4, -0.7, 0.6]);
        let chain = RotorChain::new(vec![axis]).expect("unit axis");
        let v = [0.3, -0.5, 0.8];
        let g = [1.0, 0.0, 0.0];
        for &theta in &[0.0, 0.4, 1.7, -0.9] {
            let out = chain.adjoint(&[theta], v, g).expect("adjoint");
            let w = out.value;
            let closed = g[0] * (axis[1] * w[2] - axis[2] * w[1])
                + g[1] * (axis[2] * w[0] - axis[0] * w[2])
                + g[2] * (axis[0] * w[1] - axis[1] * w[0]);
            assert!(
                (out.grad_angles[0] - closed).abs() < 1e-12,
                "θ={theta}: adjoint {} vs closed form {closed}",
                out.grad_angles[0]
            );
        }
    }

    #[test]
    fn non_unit_axis_is_rejected() {
        let err = RotorChain::new(vec![[1.0, 1.0, 0.0]]).unwrap_err();
        assert!(matches!(err, ChainError::NonUnitAxis { index: 0, .. }));
    }

    #[test]
    fn angle_count_mismatch_is_rejected() {
        let chain = RotorChain::new(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).expect("unit axes");
        let err = chain.apply(&[0.1], [1.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(
            err,
            ChainError::AngleCount {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        let chain = RotorChain::new(vec![[1.0, 0.0, 0.0]]).expect("unit axis");
        assert!(matches!(
            chain.apply(&[f64::NAN], [1.0, 0.0, 0.0]),
            Err(ChainError::NotFinite { .. })
        ));
        assert!(matches!(
            chain.finite_difference_gradient(&[0.1], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0),
            Err(ChainError::NotFinite { .. })
        ));
    }

    #[test]
    fn empty_chain_is_the_identity_with_no_gradients() {
        let chain = RotorChain::new(Vec::new()).expect("empty chain");
        assert!(chain.is_empty());
        let out = chain
            .adjoint(&[], [0.3, -0.5, 0.8], [1.0, 2.0, 3.0])
            .expect("adjoint");
        assert_eq!(out.value, [0.3, -0.5, 0.8]);
        assert!(out.grad_angles.is_empty());
        // With no links the input gradient is the seed, unchanged.
        for k in 0..3 {
            assert!((out.grad_input[k] - [1.0, 2.0, 3.0][k]).abs() < 1e-15);
        }
    }

    #[test]
    fn apply_preserves_norm() {
        let chain = RotorChain::new(vec![
            unit([0.4, -0.7, 0.6]),
            unit([-0.2, 0.9, 0.35]),
            unit([1.0, 0.1, -0.3]),
        ])
        .expect("unit axes");
        let v = [0.3f64, -0.5, 0.8];
        let before = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let w = chain.apply(&[0.3, -1.1, 2.0], v).expect("apply");
        let after = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        assert!((after - before).abs() < 1e-12, "{before} → {after}");
    }
}
