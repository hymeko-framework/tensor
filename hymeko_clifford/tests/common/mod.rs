//! Shared helpers for the `hymeko_clifford` scenario tests.
//!
//! Deliberately thin: every scenario builds its own geometry from the public
//! API, so this module holds only the operations that would otherwise be
//! retyped in each file (CLAUDE.md §6.1).
//!
//! Nothing here depends on any crate but `hymeko_clifford` and `std` — the
//! scenarios exercise the Clifford package in isolation, with no graph, runtime,
//! or learner in the loop.

#![allow(dead_code)] // each scenario file uses a subset

use hymeko_clifford::{Multivector, Signature};

/// Absolute tolerance for `f64` geometric-product chains of the depth used here.
pub const EPS: f64 = 1e-12;

/// Build a vector `Σ coords[i] · e_{i+1}` in an `n`-dimensional algebra.
///
/// # Preconditions
/// `coords.len() <= n`.
pub fn vector(coords: &[f64], n: usize) -> Multivector {
    assert!(coords.len() <= n, "more coordinates than basis vectors");
    let mut mv = Multivector::zero(n);
    for (i, &c) in coords.iter().enumerate() {
        mv.components[1 << i] = c;
    }
    mv
}

/// Extract the grade-1 coordinates of `mv` as `[e_1, …, e_n]`.
pub fn coords_of(mv: &Multivector, n: usize) -> Vec<f64> {
    (0..n).map(|i| mv.components[1 << i]).collect()
}

/// The sandwich product `R x R̃` — the rotor/versor action on `x`.
pub fn sandwich(r: &Multivector, x: &Multivector, sig: &Signature) -> Multivector {
    r.geo(x, sig).geo(&r.reverse(), sig)
}

/// Squared magnitude of the scalar part of `A Ã` — the versor norm test.
pub fn versor_norm_sq(a: &Multivector, sig: &Signature) -> f64 {
    a.geo(&a.reverse(), sig).components[0]
}

/// Sum of squares over every component. Used to assert that a multivector is
/// (or is not) the identity, without assuming which grades are populated.
pub fn total_sq(mv: &Multivector) -> f64 {
    mv.components.iter().map(|c| c * c).sum()
}

/// Assert two multivectors agree componentwise.
pub fn assert_mv_close(got: &Multivector, want: &Multivector, tol: f64, what: &str) {
    assert_eq!(got.n, want.n, "{what}: dimension mismatch");
    for (i, (g, w)) in got
        .components
        .iter()
        .zip(want.components.iter())
        .enumerate()
    {
        assert!(
            (g - w).abs() < tol,
            "{what}: blade {i:#06b} differs — got {g}, want {w} (tol {tol})"
        );
    }
}

/// Assert two coordinate slices agree.
pub fn assert_coords_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < tol,
            "{what}: coordinate {i} differs — got {g}, want {w} (tol {tol})"
        );
    }
}
