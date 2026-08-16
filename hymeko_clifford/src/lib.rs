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
//! Subsequent phases (geometric/outer/inner/scalar products, autograd
//! tape, G-SPHF Laplacian, gradient flow) populate the modules listed
//! at `docs/plans/plans_20260429/hymeko_clifford_plan.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod algebra;

pub use algebra::{
    Multivector, Signature, blade_product, canonical_reorder_sign, cayley_to_unit_quat,
    quat_conjugate, quat_mul, quat_rotate,
};
