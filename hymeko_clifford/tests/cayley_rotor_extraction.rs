//! Certification for the 2026-08-28 Cayley-rotor extraction out of Nagare.
//!
//! The reference below is a **verbatim transcription of the pre-extraction Nagare body**
//! (`nagare/src/ops/cayley_rotor.rs`), written independently and calling its own local copies of
//! the scalar primitives rather than the core implementations. Nothing here calls the core to
//! produce an expected value.
//!
//! Two things are kept apart deliberately: **finite-domain bit identity**, which is what the
//! extraction has to preserve, and the **NaN/Inf assertion behaviour**, which is a contract of the
//! scalar primitives and is asserted separately at the end.

use hymeko_clifford::{cayley_rotor_backward, cayley_rotor_forward};

// ---------------------------------------------------------------------------------------------
// Verbatim reference — transcribed from the pre-move Nagare body, self-contained.
// ---------------------------------------------------------------------------------------------

fn ref_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn ref_cayley_to_unit_quat(b: [f32; 3]) -> [f32; 4] {
    let norm = (1.0 + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    [1.0 / norm, b[0] / norm, b[1] / norm, b[2] / norm]
}

fn ref_quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let w = q[0];
    let u = [q[1], q[2], q[3]];
    let uxv = ref_cross(u, v);
    let u_cross_uxv = ref_cross(u, uxv);
    [
        v[0] + 2.0 * w * uxv[0] + 2.0 * u_cross_uxv[0],
        v[1] + 2.0 * w * uxv[1] + 2.0 * u_cross_uxv[1],
        v[2] + 2.0 * w * uxv[2] + 2.0 * u_cross_uxv[2],
    ]
}

fn ref_quat_conjugate(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn ref_dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn ref_grad_quat_from_rotate(q: [f32; 4], v: [f32; 3], grad_r: [f32; 3]) -> [f32; 4] {
    let [w, x, y, z] = q;
    let [vx, vy, vz] = v;
    let [gx, gy, gz] = grad_r;

    let grad_w = gx * (2.0 * (y * vz - z * vy))
        + gy * (2.0 * (z * vx - x * vz))
        + gz * (2.0 * (x * vy - y * vx));

    let grad_x = gx * (2.0 * y * vy + 2.0 * z * vz)
        + gy * (2.0 * y * vx - 4.0 * x * vy - 2.0 * w * vz)
        + gz * (2.0 * z * vx + 2.0 * w * vy - 4.0 * x * vz);

    let grad_y = gx * (-4.0 * y * vx + 2.0 * x * vy + 2.0 * w * vz)
        + gy * (2.0 * x * vx + 2.0 * z * vz)
        + gz * (-2.0 * w * vx + 2.0 * z * vy - 4.0 * y * vz);

    let grad_z = gx * (-4.0 * z * vx - 2.0 * w * vy + 2.0 * x * vz)
        + gy * (2.0 * w * vx - 4.0 * z * vy + 2.0 * y * vz)
        + gz * (2.0 * x * vx + 2.0 * y * vy);

    [grad_w, grad_x, grad_y, grad_z]
}

fn ref_forward(bivec: &[f32], v: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(bivec.len(), n * 3);
    assert_eq!(v.len(), n * 3);
    let mut rotated = vec![0.0; n * 3];
    let mut quats = vec![0.0; n * 4];
    for i in 0..n {
        let b = [bivec[i * 3], bivec[i * 3 + 1], bivec[i * 3 + 2]];
        let vi = [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]];
        let q = ref_cayley_to_unit_quat(b);
        let r = ref_quat_rotate(q, vi);
        rotated[i * 3..i * 3 + 3].copy_from_slice(&r);
        quats[i * 4..i * 4 + 4].copy_from_slice(&q);
    }
    (rotated, quats)
}

fn ref_backward(
    bivec: &[f32],
    v: &[f32],
    quats: &[f32],
    grad_r: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut grad_bivec = vec![0.0; n * 3];
    let mut grad_v = vec![0.0; n * 3];
    for i in 0..n {
        let b = [bivec[i * 3], bivec[i * 3 + 1], bivec[i * 3 + 2]];
        let vi = [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]];
        let q = [
            quats[i * 4],
            quats[i * 4 + 1],
            quats[i * 4 + 2],
            quats[i * 4 + 3],
        ];
        let gr = [grad_r[i * 3], grad_r[i * 3 + 1], grad_r[i * 3 + 2]];

        let gv = ref_quat_rotate(ref_quat_conjugate(q), gr);
        grad_v[i * 3..i * 3 + 3].copy_from_slice(&gv);

        let grad_q = ref_grad_quat_from_rotate(q, vi, gr);
        let l = (1.0 + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let q_dot_grad = ref_dot4(q, grad_q);
        grad_bivec[i * 3] = (grad_q[1] - q[1] * q_dot_grad) / l;
        grad_bivec[i * 3 + 1] = (grad_q[2] - q[2] * q_dot_grad) / l;
        grad_bivec[i * 3 + 2] = (grad_q[3] - q[3] * q_dot_grad) / l;
    }
    (grad_bivec, grad_v)
}

// ---------------------------------------------------------------------------------------------

/// Deterministic LCG, so the sweep is reproducible from this source alone.
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn case(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Rng(seed);
    let bivec: Vec<f32> = (0..n * 3).map(|_| r.f()).collect();
    let v: Vec<f32> = (0..n * 3).map(|_| r.f() * 5.0).collect();
    let grad_r: Vec<f32> = (0..n * 3).map(|_| r.f()).collect();
    (bivec, v, grad_r)
}

/// Forward: the saved quaternions and the rotated vectors are bit-identical to the pre-move body.
#[test]
fn the_forward_is_bit_identical_to_the_pre_extraction_body() {
    for n in [1usize, 3, 17] {
        for seed in [1u64, 42, 2026, 7] {
            let (bivec, v, _) = case(seed, n);
            let (rot, quats) = cayley_rotor_forward(&bivec, &v, n);
            let (ref_rot, ref_quats) = ref_forward(&bivec, &v, n);
            for i in 0..rot.len() {
                assert_eq!(
                    rot[i].to_bits(),
                    ref_rot[i].to_bits(),
                    "n={n} seed={seed} rotated[{i}]"
                );
            }
            for i in 0..quats.len() {
                assert_eq!(
                    quats[i].to_bits(),
                    ref_quats[i].to_bits(),
                    "n={n} seed={seed} quats[{i}]"
                );
            }
        }
    }
}

/// Backward: both gradient buffers are bit-identical to the pre-move body.
#[test]
fn the_backward_is_bit_identical_to_the_pre_extraction_body() {
    for n in [1usize, 3, 17] {
        for seed in [1u64, 42, 2026, 7] {
            let (bivec, v, grad_r) = case(seed, n);
            let (_, quats) = cayley_rotor_forward(&bivec, &v, n);
            let (gb, gv) = cayley_rotor_backward(&bivec, &v, &quats, &grad_r, n);
            let (ref_gb, ref_gv) = ref_backward(&bivec, &v, &quats, &grad_r, n);
            for i in 0..gb.len() {
                assert_eq!(
                    gb[i].to_bits(),
                    ref_gb[i].to_bits(),
                    "n={n} seed={seed} grad_bivec[{i}]"
                );
            }
            for i in 0..gv.len() {
                assert_eq!(
                    gv[i].to_bits(),
                    ref_gv[i].to_bits(),
                    "n={n} seed={seed} grad_v[{i}]"
                );
            }
        }
    }
}

/// The adjoint against composed central differences in `f64`.
///
/// `h = 1e-3` is the **measured** best cell of the D0 h-sweep, not an invented tolerance: at
/// `h = 1e-2` the difference is truncation-dominated, and at `h = 1e-4` perturbing an `f32` by an
/// amount near its own resolution dominates instead. The bound asserted here is the worst relative
/// error that sweep actually reported, with headroom, and the sweep is kept so a regression in the
/// Jacobian moves it.
#[test]
fn the_adjoint_matches_finite_differences_at_the_measured_floor() {
    fn ref_forward_f64(b: [f64; 3], v: [f64; 3]) -> [f64; 3] {
        let norm = (1.0 + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let q = [1.0 / norm, b[0] / norm, b[1] / norm, b[2] / norm];
        let (w, u) = (q[0], [q[1], q[2], q[3]]);
        let cross = |a: [f64; 3], c: [f64; 3]| {
            [
                a[1] * c[2] - a[2] * c[1],
                a[2] * c[0] - a[0] * c[2],
                a[0] * c[1] - a[1] * c[0],
            ]
        };
        let uxv = cross(u, v);
        let uuxv = cross(u, uxv);
        [
            v[0] + 2.0 * w * uxv[0] + 2.0 * uuxv[0],
            v[1] + 2.0 * w * uxv[1] + 2.0 * uuxv[1],
            v[2] + 2.0 * w * uxv[2] + 2.0 * uuxv[2],
        ]
    }

    let h = 1e-3f64;
    for n in [2usize, 5] {
        for seed in [1u64, 42, 2026] {
            let (bivec, v, grad_r) = case(seed, n);
            let (_, quats) = cayley_rotor_forward(&bivec, &v, n);
            let (gb, gv) = cayley_rotor_backward(&bivec, &v, &quats, &grad_r, n);
            let obj = |bb: &[f32], vv: &[f32]| -> f64 {
                (0..n)
                    .map(|i| {
                        let b = [bb[i * 3] as f64, bb[i * 3 + 1] as f64, bb[i * 3 + 2] as f64];
                        let x = [vv[i * 3] as f64, vv[i * 3 + 1] as f64, vv[i * 3 + 2] as f64];
                        let rr = ref_forward_f64(b, x);
                        (0..3)
                            .map(|k| rr[k] * grad_r[i * 3 + k] as f64)
                            .sum::<f64>()
                    })
                    .sum()
            };
            for idx in 0..n * 3 {
                let mut up = bivec.clone();
                let mut dn = bivec.clone();
                up[idx] = (up[idx] as f64 + h) as f32;
                dn[idx] = (dn[idx] as f64 - h) as f32;
                let fd = (obj(&up, &v) - obj(&dn, &v)) / (2.0 * h);
                let den = (gb[idx] as f64).abs().max(fd.abs()).max(1e-9);
                let rel = (gb[idx] as f64 - fd).abs() / den;
                assert!(
                    rel < 1e-3,
                    "n={n} seed={seed} grad_bivec[{idx}] rel={rel:.3e}"
                );

                let mut up = v.clone();
                let mut dn = v.clone();
                up[idx] = (up[idx] as f64 + h) as f32;
                dn[idx] = (dn[idx] as f64 - h) as f32;
                let fd = (obj(&bivec, &up) - obj(&bivec, &dn)) / (2.0 * h);
                let den = (gv[idx] as f64).abs().max(fd.abs()).max(1e-9);
                let rel = (gv[idx] as f64 - fd).abs() / den;
                assert!(rel < 1e-3, "n={n} seed={seed} grad_v[{idx}] rel={rel:.3e}");
            }
        }
    }
}

/// Shape preconditions are enforced by the operator, independently of the arithmetic above.
#[test]
#[should_panic(expected = "assertion")]
fn a_mismatched_bivec_length_is_rejected() {
    let _ = cayley_rotor_forward(&[0.0; 5], &[0.0; 6], 2);
}

/// The NaN/Inf contract is the **scalar primitives'**, not the batched operator's, and is asserted
/// here separately from the finite-domain bit identity above.
#[test]
#[should_panic(expected = "assertion")]
fn a_non_finite_bivector_is_rejected_by_the_scalar_contract() {
    let _ = cayley_rotor_forward(&[f32::NAN, 0.0, 0.0], &[1.0, 0.0, 0.0], 1);
}
