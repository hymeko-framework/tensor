//! **Scenario: parallel transport around a closed loop on the sphere.**
//!
//! The textbook holonomy result, run entirely inside `hymeko_clifford`: transport
//! a tangent frame around a closed spherical polygon and the frame comes back
//! rotated by the enclosed **solid angle**. For the octant triangle
//! `e₁ → e₂ → e₃ → e₁` — three great-circle quarter-arcs meeting at right
//! angles — the enclosed solid angle is `π/2`, so the holonomy is a quarter turn
//! about the starting axis.
//!
//! This is the geometric claim Nagare's name is built on, reduced to the algebra
//! that has to support it, with a closed-form answer to check against. Nothing
//! outside this crate is involved: no graph, no runtime, no learner.
//!
//! Transport along a great-circle arc from unit `a` to unit `b` is the minimal
//! rotation carrying `a` to `b`, which in `Cl(3,0)` is the rotor `R = m a` where
//! `m` is the unit bisector of `a` and `b` — the composition of two reflections.

mod common;

use common::{assert_coords_close, coords_of, sandwich, vector, versor_norm_sq};
use hymeko_clifford::{Multivector, Signature};

const N: usize = 3;

/// Rotor carrying unit vector `a` to unit vector `b` along the shorter great arc.
///
/// `R = m a` with `m = (a + b)/‖a + b‖`; the double-reflection through `a` then
/// `m` rotates by twice the angle between them, i.e. exactly from `a` to `b`.
///
/// # Preconditions
/// `a` and `b` are unit grade-1 multivectors and are not antipodal.
fn transport_rotor(a: &Multivector, b: &Multivector, sig: &Signature) -> Multivector {
    let sum = a.add(b);
    let norm = coords_of(&sum, N).iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!(
        norm > 1e-9,
        "antipodal endpoints have no unique transport arc"
    );
    let m = sum.scale(1.0 / norm);
    m.geo(a, sig)
}

#[test]
fn octant_loop_holonomy_is_a_quarter_turn_about_the_start_axis() {
    let sig = Signature::euclidean(N);
    let e1 = vector(&[1.0, 0.0, 0.0], N);
    let e2 = vector(&[0.0, 1.0, 0.0], N);
    let e3 = vector(&[0.0, 0.0, 1.0], N);

    // Three quarter-arcs closing the octant: e1 → e2 → e3 → e1.
    let leg_a = transport_rotor(&e1, &e2, &sig);
    let leg_b = transport_rotor(&e2, &e3, &sig);
    let leg_c = transport_rotor(&e3, &e1, &sig);

    // Rotors compose by the geometric product, last leg on the left.
    let holonomy = leg_c.geo(&leg_b, &sig).geo(&leg_a, &sig);

    // A rotor stays unit: R R̃ = 1.
    assert!(
        (versor_norm_sq(&holonomy, &sig) - 1.0).abs() < 1e-12,
        "holonomy rotor lost unit norm: {}",
        versor_norm_sq(&holonomy, &sig)
    );

    // The loop starts and ends at e1, so e1 is the rotation axis — fixed.
    let axis_image = sandwich(&holonomy, &e1, &sig);
    assert_coords_close(
        &coords_of(&axis_image, N),
        &[1.0, 0.0, 0.0],
        1e-12,
        "the loop's base point must be the holonomy axis",
    );

    // Solid angle of the octant is π/2 ⇒ a quarter turn in the e₂e₃ plane.
    // A quarter turn about e₁ sends e₂ → e₃ and e₃ → −e₂.
    let e2_image = sandwich(&holonomy, &e2, &sig);
    let e3_image = sandwich(&holonomy, &e3, &sig);
    assert_coords_close(
        &coords_of(&e2_image, N),
        &[0.0, 0.0, 1.0],
        1e-12,
        "e₂ under octant holonomy",
    );
    assert_coords_close(
        &coords_of(&e3_image, N),
        &[0.0, -1.0, 0.0],
        1e-12,
        "e₃ under octant holonomy",
    );
}

#[test]
fn transport_around_a_degenerate_loop_is_the_identity() {
    // Out and back along the same arc encloses zero area ⇒ zero holonomy.
    // The discriminating companion to the octant case: it separates "the
    // algebra rotates things" from "the algebra measures enclosed area".
    let sig = Signature::euclidean(N);
    let e1 = vector(&[1.0, 0.0, 0.0], N);
    let e2 = vector(&[0.0, 1.0, 0.0], N);

    let out = transport_rotor(&e1, &e2, &sig);
    let back = transport_rotor(&e2, &e1, &sig);
    let loop_rotor = back.geo(&out, &sig);

    for probe in [
        vector(&[1.0, 0.0, 0.0], N),
        vector(&[0.0, 1.0, 0.0], N),
        vector(&[0.0, 0.0, 1.0], N),
        vector(&[0.3, -0.5, 0.8], N),
    ] {
        let image = sandwich(&loop_rotor, &probe, &sig);
        assert_coords_close(
            &coords_of(&image, N),
            &coords_of(&probe, N),
            1e-12,
            "zero-area loop must not rotate anything",
        );
    }
}

#[test]
fn holonomy_grows_with_enclosed_area_not_with_path_length() {
    // Subdivide one leg of the octant into many small arcs. Path length rises;
    // enclosed area does not — so the holonomy must be unchanged. This is the
    // property that separates a genuine curvature effect from accumulated
    // numerical drift, and it is the reason the test is worth having.
    let sig = Signature::euclidean(N);
    let e1 = vector(&[1.0, 0.0, 0.0], N);
    let e2 = vector(&[0.0, 1.0, 0.0], N);
    let e3 = vector(&[0.0, 0.0, 1.0], N);

    let coarse = transport_rotor(&e3, &e1, &sig)
        .geo(&transport_rotor(&e2, &e3, &sig), &sig)
        .geo(&transport_rotor(&e1, &e2, &sig), &sig);

    // Walk e1 → e2 through 64 intermediate points on the same great circle.
    const STEPS: usize = 64;
    let mut fine = Multivector::one(N);
    let mut prev = e1.clone();
    for s in 1..=STEPS {
        let t = (s as f64) * (std::f64::consts::FRAC_PI_2) / (STEPS as f64);
        let next = vector(&[t.cos(), t.sin(), 0.0], N);
        fine = transport_rotor(&prev, &next, &sig).geo(&fine, &sig);
        prev = next;
    }
    fine = transport_rotor(&e3, &e1, &sig)
        .geo(&transport_rotor(&e2, &e3, &sig), &sig)
        .geo(&fine, &sig);

    let probe = vector(&[0.0, 1.0, 0.0], N);
    let coarse_image = coords_of(&sandwich(&coarse, &probe, &sig), N);
    let fine_image = coords_of(&sandwich(&fine, &probe, &sig), N);
    assert_coords_close(
        &fine_image,
        &coarse_image,
        1e-10,
        "subdividing a leg changed the holonomy — it must depend on area, not on \
         how finely the same path is walked",
    );
}

#[test]
fn rotor_chain_conditioning_over_ten_thousand_products() {
    // §5 (numerical stability): document the conditioning of a long rotor chain
    // rather than assume it. 10 000 small rotations about a fixed axis compose to
    // one large rotation; the accumulated drift in the unit-norm invariant is the
    // quantity of interest, and it is reported, not merely bounded.
    let sig = Signature::euclidean(N);
    const STEPS: usize = 10_000;
    let theta = std::f64::consts::TAU / (STEPS as f64);

    // Rotor for a rotation by `theta` in the e₁e₂ plane: cos(θ/2) − sin(θ/2) e₁e₂.
    let mut step = Multivector::zero(N);
    step.components[0] = (theta / 2.0).cos();
    step.components[0b011] = -(theta / 2.0).sin();

    let mut acc = Multivector::one(N);
    let mut worst_norm_err: f64 = 0.0;
    for _ in 0..STEPS {
        acc = step.geo(&acc, &sig);
        worst_norm_err = worst_norm_err.max((versor_norm_sq(&acc, &sig) - 1.0).abs());
    }

    // A full turn returns every vector to itself.
    let probe = vector(&[0.6, -0.8, 0.0], N);
    let image = coords_of(&sandwich(&acc, &probe, &sig), N);
    let drift: f64 = image
        .iter()
        .zip(coords_of(&probe, N).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    eprintln!(
        "rotor chain conditioning: {STEPS} products — worst ‖RR̃−1‖ = {worst_norm_err:.3e}, \
         full-turn vector drift = {drift:.3e}"
    );

    // Budgets: f64 rotor chains of this depth should hold both to ~1e-12.
    assert!(
        worst_norm_err < 1e-12,
        "unit-norm invariant drifted to {worst_norm_err:e} over {STEPS} products"
    );
    assert!(
        drift < 1e-12,
        "a full turn failed to return the probe vector: drift {drift:e}"
    );
}
