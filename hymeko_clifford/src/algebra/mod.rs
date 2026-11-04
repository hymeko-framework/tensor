//! Phase 1 — Clifford algebra foundation.
//!
//! Currently exposes:
//! - [`Signature`] — $(p, q)$ metric;
//! - [`Multivector`] — dense $2^N$-component representation;
//! - [`canonical_reorder_sign`] — the load-bearing parity function;
//! - [`blade_product`] — basis-blade product with metric and sign.
//!
//! Geometric / outer / inner products and grade operations land in
//! `algebra/products.rs` (next), gated by an exhaustive test suite of
//! `canonical_reorder_sign` against a brute-force reference.

mod blade;
mod connection;
mod lanes;
mod multivector;
mod rotor;
mod se;
mod se_fixed;
mod signature;
mod spatial;
mod vector;

pub use blade::{blade_product, canonical_reorder_sign};
pub use connection::{
    TreeStep, closed_form_curvature, cotree_residuals, curvature_energy, rotor_angle,
    tree_gauge_frames,
};
pub use lanes::{quat_mul_lanes, quat_norm_sq_lanes};
pub use multivector::Multivector;
pub use rotor::{
    cayley_rotor_backward, cayley_rotor_forward, cayley_to_unit_quat, q_at, quat_conjugate,
    quat_mul, quat_rotate, rotor_holonomy_backward, rotor_holonomy_forward,
};
pub use se::Se;
pub use se_fixed::{Dim2, Dim3, EuclideanDim, Se2, Se3, SeFixed};
pub use signature::Signature;
pub use spatial::{ArticulatedInertia, SpatialInertia, Twist, Wrench};
pub use vector::cross;
