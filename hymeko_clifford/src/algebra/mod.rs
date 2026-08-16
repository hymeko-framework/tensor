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
mod multivector;
mod rotor;
mod signature;

pub use blade::{blade_product, canonical_reorder_sign};
pub use multivector::Multivector;
pub use rotor::{cayley_to_unit_quat, quat_conjugate, quat_mul, quat_rotate};
pub use signature::Signature;
