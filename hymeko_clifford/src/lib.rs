//! # `hymeko_clifford`
//!
//! Clifford-algebra autograd backend for G-SPHF over signed-incidence
//! hypergraphs. This crate is intentionally self-contained — it has no
//! dependency on `hymeko_core` (per the design invariant in the
//! companion plan) so that the algebraic primitives can be reused
//! independently and verified in isolation.
//!
//! ## Phase 1 (this revision)
//!
//! - [`algebra::Signature`]: $(p, q)$ metric signature.
//! - [`algebra::Multivector`] dense representation indexed by blade
//!   bitmask.
//! - [`algebra::blade_product`]: the load-bearing function that maps a
//!   pair of basis-blade bitmasks to `(result_bitmask, sign)` under
//!   the canonical-reorder convention.
//! - Unit tests covering anticommutativity, basis squares, and an
//!   exhaustive sweep of `canonical_reorder_sign` for $N \leq 4$.
//!
//! ## Phase 2 (2026-08-16)
//!
//! - [`autodiff`]: reverse-mode (adjoint) gradients for rotor chains —
//!   `O(k)` where a central difference is `O(k²)`. Plan and measurement in
//!   `docs/plans/2026-08-16-rotor-adjoint/`.
//!
//! Subsequent phases (outer/inner/scalar products, a general autograd
//! tape, G-SPHF Laplacian, gradient flow) populate the modules listed
//! at `docs/plans/plans_20260429/hymeko_clifford_plan.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod algebra;
pub mod autodiff;

pub use algebra::{
    ArticulatedInertia, Dim2, Dim3, EuclideanDim, Multivector, Se, Se2, Se3, SeFixed, Signature,
    SpatialInertia, TreeStep, Twist, Wrench, blade_product, canonical_reorder_sign,
    cayley_rotor_backward, cayley_rotor_forward, cayley_to_unit_quat, closed_form_curvature,
    cotree_residuals, cross, curvature_energy, q_at, quat_conjugate, quat_mul, quat_rotate,
    rotor_angle, rotor_holonomy_backward, rotor_holonomy_forward, tree_gauge_frames,
};
pub use autodiff::{Adjoint, Backend, ChainError, RotorChain};
