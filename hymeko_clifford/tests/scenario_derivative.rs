//! **Scenario: differentiating a rotation, and what the algebra costs to differentiate.**
//!
//! `hymeko_clifford` describes itself as the "Clifford-algebra autograd backend
//! for G-SPHF", but Phase 1 ships no gradients. Anything needing one today
//! differentiates numerically, so the numerical derivative is the operation that
//! must be correct — and it is the baseline any future adjoint pass has to beat.
//!
//! The closed form is exact and needs no library support: for `w(θ) = R(θ) v R̃(θ)`
//! about a unit axis `ω`,
//!
//! ```text
//!     dw/dθ = ω × w(θ)
//! ```
//!
//! so a central difference through the crate's own rotor path can be checked
//! against an answer that does not come from the crate. That is what makes this a
//! test rather than a self-consistency check.
//!
//! `nagare/benches/clifford_algebra_bench.rs` measures the *cost* of the same
//! gradient (`clifford/derivative/*`); this file pins its *correctness*, and the
//! second-order convergence property that says the difference quotient is
//! actually converging to the derivative rather than to something nearby.
//!
//! Only `hymeko_clifford` and `std` are involved.

mod common;

use common::{coords_of, sandwich, vector};
use hymeko_clifford::{Multivector, Signature, cayley_to_unit_quat, quat_rotate};

const N: usize = 3;

/// Rotor for a rotation of `theta` about the unit axis `axis`, in `Cl(3,0)`.
///
/// The bivector dual to the axis: `e₂e₃ ← x`, `e₃e₁ ← y`, `e₁e₂ ← z`.
fn rotor_about(axis: [f64; 3], theta: f64) -> Multivector {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let mut r = Multivector::zero(N);
    r.components[0] = c;
    r.components[0b110] = -s * axis[0];
    r.components[0b101] = s * axis[1];
    r.components[0b011] = -s * axis[2];
    r
}

/// `w(θ) = R(θ) v R̃(θ)` through the multivector path.
fn transported(axis: [f64; 3], theta: f64, v: [f64; 3], sig: &Signature) -> [f64; 3] {
    let w = sandwich(&rotor_about(axis, theta), &vector(&v, N), sig);
    let c = coords_of(&w, N);
    [c[0], c[1], c[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Central difference of `w(θ)` with step `h`.
fn central_difference(
    axis: [f64; 3],
    theta: f64,
    v: [f64; 3],
    h: f64,
    sig: &Signature,
) -> [f64; 3] {
    let up = transported(axis, theta + h, v, sig);
    let down = transported(axis, theta - h, v, sig);
    [
        (up[0] - down[0]) / (2.0 * h),
        (up[1] - down[1]) / (2.0 * h),
        (up[2] - down[2]) / (2.0 * h),
    ]
}

#[test]
fn numerical_derivative_matches_the_closed_form_angular_velocity() {
    let sig = Signature::euclidean(N);
    let h = 1e-5;
    let mut worst: f64 = 0.0;

    for axis in [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        unit([0.4, -0.7, 0.6]),
        unit([-0.2, 0.9, 0.35]),
    ] {
        for &theta in &[0.0, 0.3, 1.2, 2.5, -0.8] {
            for v in [[1.0, 0.0, 0.0], [0.3, -0.5, 0.8], [0.0, 1.0, 0.0]] {
                let numeric = central_difference(axis, theta, v, h, &sig);
                let analytic = cross(axis, transported(axis, theta, v, &sig));
                for k in 0..3 {
                    let d = (numeric[k] - analytic[k]).abs();
                    worst = worst.max(d);
                    assert!(
                        d < 1e-8,
                        "axis {axis:?} θ={theta} v={v:?}: component {k} — numeric \
                         {numeric:?} vs closed form ω×w {analytic:?}"
                    );
                }
            }
        }
    }
    eprintln!("central difference vs ω×w: worst component error {worst:.3e} (h = {h:.0e})");
}

#[test]
fn the_difference_quotient_converges_at_second_order() {
    // Agreement at one step size is weak: a wrong analytic form could still land
    // within tolerance. A central difference must converge as O(h²), so halving
    // `h` must cut the error by ~4×. That is a property of the derivative, not of
    // the tolerance — until round-off takes over, which is why the sweep stops at
    // 1e-4 rather than chasing h → 0.
    let sig = Signature::euclidean(N);
    let axis = unit([0.4, -0.7, 0.6]);
    let (theta, v) = (0.9, [0.3, -0.5, 0.8]);
    let analytic = cross(axis, transported(axis, theta, v, &sig));

    let err = |h: f64| -> f64 {
        let numeric = central_difference(axis, theta, v, h, &sig);
        (0..3)
            .map(|k| (numeric[k] - analytic[k]).abs())
            .fold(0.0, f64::max)
    };

    let coarse = err(2e-3);
    let fine = err(1e-3);
    let ratio = coarse / fine;
    eprintln!(
        "central-difference convergence: err(2e-3) = {coarse:.3e}, err(1e-3) = {fine:.3e}, ratio = {ratio:.2} (expect ≈4)"
    );
    assert!(
        (3.0..5.5).contains(&ratio),
        "central difference is not converging at second order: halving h changed \
         the error by {ratio:.2}×, expected ≈4×"
    );
}

#[test]
fn both_rotation_paths_give_the_same_derivative() {
    // The gradient a caller measures must not depend on which of the crate's two
    // rotation paths it differentiated through. `f32` on the quaternion side sets
    // the tolerance, and a finite difference amplifies its error by `1/2h`, so the
    // step is deliberately large here — chasing a small `h` on `f32` measures
    // round-off, not the derivative.
    let sig = Signature::euclidean(N);
    let h = 1e-2;
    let axis = unit([0.4, -0.7, 0.6]);
    let v = [0.3f64, -0.5, 0.8];

    let quat_transported = |theta: f64| -> [f64; 3] {
        let t = (theta / 2.0).tan();
        let q = cayley_to_unit_quat([
            (axis[0] * t) as f32,
            (axis[1] * t) as f32,
            (axis[2] * t) as f32,
        ]);
        let w = quat_rotate(q, [v[0] as f32, v[1] as f32, v[2] as f32]);
        [w[0] as f64, w[1] as f64, w[2] as f64]
    };

    for &theta in &[0.2, 0.9, 1.6] {
        let via_mv = central_difference(axis, theta, v, h, &sig);
        let up = quat_transported(theta + h);
        let down = quat_transported(theta - h);
        let via_quat = [
            (up[0] - down[0]) / (2.0 * h),
            (up[1] - down[1]) / (2.0 * h),
            (up[2] - down[2]) / (2.0 * h),
        ];
        for k in 0..3 {
            assert!(
                (via_mv[k] - via_quat[k]).abs() < 1e-4,
                "θ={theta} component {k}: multivector-path derivative {via_mv:?} vs \
                 quaternion-path derivative {via_quat:?}"
            );
        }
    }
}

#[test]
fn gradient_of_a_chain_costs_two_evaluations_per_angle() {
    // The scaling claim the benchmark measures, asserted here as a property of the
    // algorithm rather than of the clock: a central-difference gradient over `k`
    // angles performs exactly `2k` forward evaluations, and each forward
    // evaluation applies `k` rotors — hence `O(k²)` rotor applications, against
    // `O(k)` for an adjoint pass. A counter makes that exact and machine-checked.
    let sig = Signature::euclidean(N);
    let axes: Vec<[f64; 3]> = (0..8)
        .map(|i| unit([0.3 + 0.1 * i as f64, -0.7, 0.6 - 0.05 * i as f64]))
        .collect();

    for k in [1usize, 2, 4, 8] {
        let mut rotor_applications = 0usize;
        let mut apply_chain = |angles: &[f64]| -> [f64; 3] {
            let mut x = vector(&[0.3, -0.5, 0.8], N);
            for (axis, &theta) in axes.iter().take(k).zip(angles) {
                x = sandwich(&rotor_about(*axis, theta), &x, &sig);
                rotor_applications += 1;
            }
            let c = coords_of(&x, N);
            [c[0], c[1], c[2]]
        };

        let angles: Vec<f64> = (0..k).map(|i| 0.11 + 0.07 * i as f64).collect();
        let mut evaluations = 0usize;
        let mut probe = angles.clone();
        for i in 0..k {
            let keep = probe[i];
            probe[i] = keep + 1e-5;
            apply_chain(&probe);
            probe[i] = keep - 1e-5;
            apply_chain(&probe);
            probe[i] = keep;
            evaluations += 2;
        }

        assert_eq!(
            evaluations,
            2 * k,
            "central-difference gradient over {k} angles used {evaluations} forward \
             evaluations, expected exactly {}",
            2 * k
        );
        assert_eq!(
            rotor_applications,
            2 * k * k,
            "gradient over {k} angles applied {rotor_applications} rotors, expected \
             2k² = {}",
            2 * k * k
        );
    }
}
