//! **Scenario: the crate's two rotation paths must agree.**
//!
//! `hymeko_clifford` rotates in two independent ways:
//!
//! * the general path — a `Cl(3,0)` rotor built as a geometric product of unit
//!   vectors, applied by the sandwich `R x R̃` through `Multivector::geo`;
//! * the fast path — `quat_mul` / `quat_rotate` / `cayley_to_unit_quat`, which
//!   is what the callers on hot paths actually use.
//!
//! They share no code. A sign or ordering convention can drift in one without
//! any existing unit test noticing, because each half is tested only against
//! itself. These scenarios cross-validate them on the same rotations, which is
//! the check that catches convention drift.
//!
//! `f32` on the quaternion side against `f64` on the multivector side, so the
//! tolerance is set by `f32`, not by the algebra.
//!
//! Only `hymeko_clifford` and `std` are involved.

mod common;

use common::{coords_of, sandwich, vector};
use hymeko_clifford::{
    Multivector, Signature, cayley_to_unit_quat, quat_conjugate, quat_mul, quat_rotate,
};

const N: usize = 3;
/// Set by the `f32` quaternion path, not by the `f64` geometric product.
const TOL: f64 = 2e-6;

/// Rotor for a rotation of `angle` in the plane of the two given axes, built the
/// general way: the product of two unit vectors separated by `angle / 2`.
fn rotor_in_plane(angle: f64, sig: &Signature) -> Multivector {
    let a = vector(&[1.0, 0.0, 0.0], N);
    let half = angle / 2.0;
    let b = vector(&[half.cos(), half.sin(), 0.0], N);
    b.geo(&a, sig)
}

/// The same rotation as a unit quaternion: `angle` about `+e₃`.
///
/// `cayley_to_unit_quat` takes a Cayley (Gibbs) vector `tan(θ/2)·axis`.
fn quat_about_z(angle: f64) -> [f32; 4] {
    cayley_to_unit_quat([0.0, 0.0, (angle / 2.0).tan() as f32])
}

#[test]
fn rotor_sandwich_and_quat_rotate_agree_on_a_sweep() {
    let sig = Signature::euclidean(N);
    // Avoid ±π, where the Cayley parameterisation diverges (tan(θ/2) → ∞).
    let angles: Vec<f64> = (1..24).map(|i| (i as f64) * 0.25 - 2.75).collect();

    let probes = [
        [1.0f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.3, -0.5, 0.8],
        [-0.9, 0.1, 0.4],
    ];

    let mut worst: f64 = 0.0;
    for &angle in &angles {
        let r = rotor_in_plane(angle, &sig);
        let q = quat_about_z(angle);
        for p in probes {
            let via_rotor = coords_of(&sandwich(&r, &vector(&p, N), &sig), N);
            let via_quat = quat_rotate(q, [p[0] as f32, p[1] as f32, p[2] as f32]);
            for k in 0..3 {
                let diff = (via_rotor[k] - via_quat[k] as f64).abs();
                worst = worst.max(diff);
                assert!(
                    diff < TOL,
                    "angle {angle}: rotor and quaternion disagree on axis {k} for \
                     probe {p:?} — rotor {:?}, quat {:?}",
                    via_rotor,
                    via_quat
                );
            }
        }
    }
    eprintln!("rotor ↔ quaternion agreement: worst component difference {worst:.3e}");
}

#[test]
fn composition_order_matches_between_the_two_paths() {
    // Agreeing on single rotations is weaker than agreeing on composition: the
    // two paths could share a convention for one rotor and disagree on which
    // side a second one multiplies. Rotations about a common axis commute, so
    // this uses two rotations in *different* planes, where order is observable.
    let sig = Signature::euclidean(N);

    // 90° in the e₁e₂ plane (about +e₃), then 90° in the e₂e₃ plane (about +e₁).
    let r_z = rotor_in_plane(std::f64::consts::FRAC_PI_2, &sig);
    let mut r_x = Multivector::zero(N);
    let half = std::f64::consts::FRAC_PI_4;
    r_x.components[0] = half.cos();
    r_x.components[0b110] = -half.sin(); // e₂ ∧ e₃

    let composed = r_x.geo(&r_z, &sig);

    let q_z = quat_about_z(std::f64::consts::FRAC_PI_2);
    let q_x = cayley_to_unit_quat([(half).tan() as f32, 0.0, 0.0]);

    let probe = [0.37f64, -0.62, 0.81];
    let via_rotor = coords_of(&sandwich(&composed, &vector(&probe, N), &sig), N);

    // Apply the quaternions in the same order: z first, then x.
    let p32 = [probe[0] as f32, probe[1] as f32, probe[2] as f32];
    let via_quat_sequential = quat_rotate(q_x, quat_rotate(q_z, p32));
    // …and as a single composed quaternion, which must match the sequential form.
    let via_quat_composed = quat_rotate(quat_mul(q_x, q_z), p32);

    for k in 0..3 {
        assert!(
            (via_quat_sequential[k] - via_quat_composed[k]).abs() < 1e-5,
            "quat_mul composition disagrees with sequential rotation on axis {k}: \
             {via_quat_sequential:?} vs {via_quat_composed:?}"
        );
        assert!(
            (via_rotor[k] - via_quat_composed[k] as f64).abs() < TOL,
            "rotor composition disagrees with quaternion composition on axis {k}: \
             rotor {via_rotor:?}, quat {via_quat_composed:?}"
        );
    }

    // And order matters at all — otherwise the test proves nothing about order.
    let reversed = quat_rotate(quat_mul(q_z, q_x), p32);
    let order_gap: f32 = (0..3)
        .map(|k| (reversed[k] - via_quat_composed[k]).abs())
        .fold(0.0, f32::max);
    assert!(
        order_gap > 0.1,
        "swapping the rotation order changed nothing ({order_gap}); the composition \
         test is vacuous"
    );
}

#[test]
fn rotor_reverse_and_quat_conjugate_are_the_same_inverse() {
    // `R̃` undoes `R`; `q*` undoes `q`. Both crate halves must mean the same
    // thing by "inverse rotation", or a round trip through one and back through
    // the other silently fails.
    let sig = Signature::euclidean(N);
    let angle = 0.83;
    let r = rotor_in_plane(angle, &sig);
    let q = quat_about_z(angle);

    let probe = [0.21f64, 0.94, -0.37];
    let p32 = [probe[0] as f32, probe[1] as f32, probe[2] as f32];

    // Forward through the rotor, back through the conjugate quaternion.
    let forward = coords_of(&sandwich(&r, &vector(&probe, N), &sig), N);
    let back = quat_rotate(
        quat_conjugate(q),
        [forward[0] as f32, forward[1] as f32, forward[2] as f32],
    );
    for k in 0..3 {
        assert!(
            (back[k] as f64 - probe[k]).abs() < TOL,
            "rotor-forward / quaternion-inverse round trip failed on axis {k}: \
             {back:?} vs {probe:?}"
        );
    }

    // Forward through the quaternion, back through the reversed rotor.
    let forward_q = quat_rotate(q, p32);
    let back_r = coords_of(
        &sandwich(
            &r.reverse(),
            &vector(
                &[
                    forward_q[0] as f64,
                    forward_q[1] as f64,
                    forward_q[2] as f64,
                ],
                N,
            ),
            &sig,
        ),
        N,
    );
    for k in 0..3 {
        assert!(
            (back_r[k] - probe[k]).abs() < TOL,
            "quaternion-forward / rotor-inverse round trip failed on axis {k}: \
             {back_r:?} vs {probe:?}"
        );
    }
}
