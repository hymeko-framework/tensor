//! **Scenario: the three metric sectors, each doing what only it can do.**
//!
//! `Signature` admits `(p, q, r)` — positive, negative, and degenerate basis
//! vectors — and each sector buys a different geometry. A benchmark that only
//! ever runs `Signature::euclidean` measures none of that. These scenarios put
//! one irreducible capability of each sector under test, with a conserved
//! quantity to check against:
//!
//! | sector | algebra | capability | invariant |
//! |---|---|---|---|
//! | positive | `Cl(3,0)` | rotation | Euclidean length |
//! | negative | `Cl(3,1)` | Lorentz boost | spacetime interval |
//! | degenerate | `Cl(3,0,1)` (PGA) | translation | versor norm, with a null bivector |
//!
//! Entirely inside `hymeko_clifford`; `std` only.

mod common;

use common::{assert_coords_close, coords_of, sandwich, vector, versor_norm_sq};
use hymeko_clifford::{Multivector, Signature};

// ── Negative sector: Cl(3,1) boosts preserve the spacetime interval ───────────

/// In `Cl(3,1)` the first three basis vectors square to `+1` and `e₄` to `−1`,
/// so `(e₁e₄)² = −e₁²e₄² = +1`: a **hyperbolic** generator. The boost versor is
/// `B = cosh(φ/2) + sinh(φ/2) e₁e₄`, the hyperbolic analogue of a rotor.
fn boost(rapidity: f64) -> Multivector {
    let mut b = Multivector::zero(4);
    b.components[0] = (rapidity / 2.0).cosh();
    b.components[0b1001] = (rapidity / 2.0).sinh(); // e₁ ∧ e₄
    b
}

/// Minkowski interval `x₁² + x₂² + x₃² − t²` for `x = x₁e₁ + x₂e₂ + x₃e₃ + t e₄`.
fn interval(mv: &Multivector) -> f64 {
    let c = coords_of(mv, 4);
    c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - c[3] * c[3]
}

#[test]
fn lorentzian_boost_preserves_the_spacetime_interval() {
    let sig = Signature::lorentzian(3, 1);

    for &rapidity in &[0.0, 0.25, 0.9, 1.7, -1.1] {
        let b = boost(rapidity);

        // A boost versor is unit under the reverse, exactly as a rotor is:
        // cosh² − sinh² = 1.
        let norm = versor_norm_sq(&b, &sig);
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "boost versor at rapidity {rapidity} is not unit: {norm}"
        );

        for event in [
            vector(&[1.0, 0.0, 0.0, 0.0], 4),  // spacelike
            vector(&[0.0, 0.0, 0.0, 1.0], 4),  // timelike
            vector(&[1.0, 0.0, 0.0, 1.0], 4),  // lightlike
            vector(&[0.3, -0.7, 0.2, 1.4], 4), // general
        ] {
            let moved = sandwich(&b, &event, &sig);
            let before = interval(&event);
            let after = interval(&moved);
            assert!(
                (after - before).abs() < 1e-10,
                "boost at rapidity {rapidity} changed the interval: {before} → {after}"
            );
        }
    }
}

#[test]
fn a_lightlike_event_stays_lightlike_and_still_moves() {
    // The discriminating half of the previous test: interval preservation alone
    // would also be satisfied by doing nothing. A null vector must be rescaled,
    // not fixed.
    let sig = Signature::lorentzian(3, 1);
    let light = vector(&[1.0, 0.0, 0.0, 1.0], 4);
    let moved = sandwich(&boost(0.9), &light, &sig);

    assert!(
        interval(&moved).abs() < 1e-10,
        "lightlike event left the light cone: interval {}",
        interval(&moved)
    );
    let before = coords_of(&light, 4);
    let after = coords_of(&moved, 4);
    let moved_by: f64 = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        moved_by > 0.5,
        "the boost left the null vector essentially unchanged (Δ = {moved_by}); \
         interval preservation was vacuous"
    );
}

// ── Degenerate sector: Cl(3,0,1) admits null bivectors ───────────────────────

#[test]
fn pga_translator_is_unit_because_its_bivector_is_null() {
    // `Signature::pga3()` = Cl(3,0,1): e₁,e₂,e₃ square to +1 and the fourth
    // basis vector e₀ squares to 0. A translator is `T = 1 + (d/2)·e₀e₁`.
    //
    // `(e₀e₁)² = −e₀²e₁² = 0`, so `T T̃ = 1 − (d/2)²·(e₀e₁)² = 1` for **every**
    // displacement d. That is what makes translation a versor here rather than
    // an affine special case.
    let sig = Signature::pga3();
    let n = 4;
    let e0_index = 1 << 3; // the degenerate basis vector is the fourth
    let e1_index = 1 << 0;
    let e01_index = e0_index | e1_index;

    for &d in &[0.0, 0.5, 3.0, 250.0] {
        let mut t = Multivector::zero(n);
        t.components[0] = 1.0;
        t.components[e01_index] = d / 2.0;

        let norm = versor_norm_sq(&t, &sig);
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "PGA translator for d={d} is not unit: {norm} — the degenerate basis \
             vector is not squaring to zero"
        );
    }
}

#[test]
fn the_same_construction_is_not_unit_in_the_euclidean_sector() {
    // The discriminating counterpart. In Cl(4,0) the bivector squares to −1, so
    // `T T̃ = 1 + (d/2)²` and grows without bound. If this test ever passes with
    // a unit norm, `Signature` has stopped distinguishing its sectors and the
    // PGA result above is meaningless.
    let sig = Signature::euclidean(4);
    let n = 4;
    let e01_index = (1 << 3) | (1 << 0);

    for &d in &[0.5, 3.0, 250.0] {
        let mut t = Multivector::zero(n);
        t.components[0] = 1.0;
        t.components[e01_index] = d / 2.0;

        let norm = versor_norm_sq(&t, &sig);
        let expected = 1.0 + (d / 2.0) * (d / 2.0);
        assert!(
            (norm - expected).abs() < 1e-9,
            "Euclidean Cl(4,0): expected 1 + (d/2)² = {expected}, got {norm}"
        );
        assert!(
            (norm - 1.0).abs() > 1e-6,
            "Euclidean bivector behaved like a null one at d={d}"
        );
    }
}

#[test]
fn degenerate_basis_vector_squares_to_zero_under_the_geometric_product() {
    // The primitive the whole degenerate sector rests on, asserted directly
    // rather than inferred from the translator.
    let sig = Signature::pga3();
    let e0 = vector(&[0.0, 0.0, 0.0, 1.0], 4);
    let sq = e0.geo(&e0, &sig);
    for (i, c) in sq.components.iter().enumerate() {
        assert!(
            c.abs() < 1e-15,
            "e₀² has a nonzero blade {i:#06b} = {c}; the degenerate sector is broken"
        );
    }

    // …while the non-degenerate ones do not.
    let e1 = vector(&[1.0, 0.0, 0.0, 0.0], 4);
    let sq1 = e1.geo(&e1, &sig);
    assert_coords_close(
        &[sq1.components[0]],
        &[1.0],
        1e-15,
        "e₁² in Cl(3,0,1) must be the scalar 1",
    );
}
