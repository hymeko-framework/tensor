//! Fixed-size quaternion rotor primitives.
//!
//! These helpers intentionally use `[f32; N]` arrays rather than the dense
//! [`crate::Multivector`] representation. A 3D rotor is always four scalars,
//! so routing through a `2^n` multivector would add the wrong abstraction and
//! unnecessary storage.

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Cayley map from Rodrigues parameters to a unit quaternion `[w, x, y, z]`.
///
/// # Args
/// - `b`: Rodrigues/bivector parameters `[bx, by, bz]`.
///
/// # Preconditions
/// `b` must contain finite values.
///
/// # Postconditions
/// Returns a unit quaternion to floating-point tolerance. The scalar part is
/// non-negative because the raw quaternion is `[1, bx, by, bz]`.
///
/// # Panics
/// Panics if any component of `b` is not finite.
pub fn cayley_to_unit_quat(b: [f32; 3]) -> [f32; 4] {
    assert!(b.iter().all(|x| x.is_finite()));
    let norm = (1.0 + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    [1.0 / norm, b[0] / norm, b[1] / norm, b[2] / norm]
}

/// Rotate a 3-vector by a unit quaternion using the sandwich cross-product.
///
/// # Args
/// - `q`: Unit quaternion `[w, x, y, z]`.
/// - `v`: Vector `[vx, vy, vz]`.
///
/// # Preconditions
/// `q` and `v` must contain finite values. `q` is expected to be unit length.
///
/// # Postconditions
/// If `q` is unit length, the returned vector has the same norm as `v` to
/// floating-point tolerance.
///
/// # Panics
/// Panics if any input component is not finite.
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    assert!(q.iter().all(|x| x.is_finite()));
    assert!(v.iter().all(|x| x.is_finite()));
    let w = q[0];
    let u = [q[1], q[2], q[3]];
    let uxv = cross(u, v);
    let u_cross_uxv = cross(u, uxv);
    [
        v[0] + 2.0 * w * uxv[0] + 2.0 * u_cross_uxv[0],
        v[1] + 2.0 * w * uxv[1] + 2.0 * u_cross_uxv[1],
        v[2] + 2.0 * w * uxv[2] + 2.0 * u_cross_uxv[2],
    ]
}

/// Conjugate a quaternion, `[w, x, y, z] -> [w, -x, -y, -z]`.
///
/// # Args
/// - `q`: Quaternion `[w, x, y, z]`.
///
/// # Preconditions
/// `q` must contain finite values.
///
/// # Postconditions
/// For unit `q`, the result is the inverse rotor.
///
/// # Panics
/// Panics if any input component is not finite.
pub fn quat_conjugate(q: [f32; 4]) -> [f32; 4] {
    assert!(q.iter().all(|x| x.is_finite()));
    [q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product `a * b` for quaternions stored as `[w, x, y, z]`.
///
/// # Args
/// - `a`: Left quaternion.
/// - `b`: Right quaternion.
///
/// # Preconditions
/// Both inputs must contain finite values.
///
/// # Postconditions
/// Returns the canonical Hamilton product. `[1, 0, 0, 0]` is the identity.
///
/// # Panics
/// Panics if any input component is not finite.
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    assert!(a.iter().all(|x| x.is_finite()));
    assert!(b.iter().all(|x| x.is_finite()));
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

// ---------------------------------------------------------------------------------------------
// Rotor holonomy — the ordered running rotor product over a cycle.
//
// Moved here from `nagare/src/ops/rotor_holonomy.rs` on 2026-08-25. It is general rotor algebra:
// an ordered product of unit quaternions and its adjoint, carrying no signed-link, dataset or
// model concept, so `hymeko_clifford` owns it and Nagare re-exports it.
//
// The arithmetic is unchanged. It now calls this crate's `quat_mul` / `quat_conjugate`, which
// `tests/quat_primitive_bit_identity.rs` proves bit-identical to the retired `qmul` / `qconj`
// expressions over a deterministic finite sweep including signed zeros.
// ---------------------------------------------------------------------------------------------

/// The identity rotor.
const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// Read quaternion `i` out of a flat 4-strided buffer.
#[inline]
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// Rotor-holonomy forward: ordered quaternion product per cycle.
///
/// # Preconditions
/// `edge_quats.len() == n_cycles * k * 4`, `k >= 1`.
///
/// # Postconditions
/// Returns `(holo, prefixes)` with `holo.len() == n_cycles * 4` (the holonomy per cycle) and
/// `prefixes.len() == n_cycles * k * 4` (the saved prefix products `P_0 … P_{k-1}`).
///
/// # Finite-input contract
/// Every element of `edge_quats` must be finite. `quat_mul` asserts this, so a non-finite
/// input panics rather than silently propagating a `NaN` — a deliberate tightening over the
/// retired Nagare expressions, certified separately from the finite-domain bit-identity.
///
/// # Panics
/// If `edge_quats.len() != n_cycles * k * 4` or `k == 0`.
pub fn rotor_holonomy_forward(
    edge_quats: &[f32],
    n_cycles: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(k >= 1);
    assert_eq!(edge_quats.len(), n_cycles * k * 4);
    let mut holo = vec![0.0f32; n_cycles * 4];
    let mut prefixes = vec![0.0f32; n_cycles * k * 4];
    for c in 0..n_cycles {
        let base = c * k * 4;
        let mut p = q_at(edge_quats, c * k); // P_0 = q_0
        prefixes[base..base + 4].copy_from_slice(&p);
        for i in 1..k {
            p = quat_mul(q_at(edge_quats, c * k + i), p); // P_i = q_i · P_{i-1}
            prefixes[base + i * 4..base + i * 4 + 4].copy_from_slice(&p);
        }
        holo[c * 4..c * 4 + 4].copy_from_slice(&p);
    }
    (holo, prefixes)
}

/// Rotor-holonomy backward. Given `grad_holo = ∂L/∂H`, returns `∂L/∂edge_quats`.
///
/// # Preconditions
/// `edge_quats.len() == prefixes.len() == n_cycles * k * 4`, `grad_holo.len() == n_cycles * 4`, `k >= 1`.
///
/// # Finite-input contract
/// Every element of `edge_quats` must be finite. `quat_mul` asserts this, so a non-finite
/// input panics rather than silently propagating a `NaN` — a deliberate tightening over the
/// retired Nagare expressions, certified separately from the finite-domain bit-identity.
///
/// # Panics
/// If the length preconditions do not hold.
pub fn rotor_holonomy_backward(
    edge_quats: &[f32],
    prefixes: &[f32],
    grad_holo: &[f32],
    n_cycles: usize,
    k: usize,
) -> Vec<f32> {
    assert!(k >= 1);
    assert_eq!(edge_quats.len(), n_cycles * k * 4);
    assert_eq!(prefixes.len(), n_cycles * k * 4);
    assert_eq!(grad_holo.len(), n_cycles * 4);
    let mut grad = vec![0.0f32; n_cycles * k * 4];
    for c in 0..n_cycles {
        let base = c * k * 4;
        let grad_h = q_at(grad_holo, c);
        let mut suffix = IDENT; // S_{k-1} = 1
        for i in (0..k).rev() {
            let p_prev = if i == 0 {
                IDENT
            } else {
                q_at(prefixes, c * k + i - 1)
            };
            let gqi = quat_mul(
                quat_mul(quat_conjugate(suffix), grad_h),
                quat_conjugate(p_prev),
            );
            grad[base + i * 4..base + i * 4 + 4].copy_from_slice(&gqi);
            suffix = quat_mul(suffix, q_at(edge_quats, c * k + i)); // S_{i-1} = S_i · q_i
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn norm4(q: [f32; 4]) -> f32 {
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "a={a} b={b}");
    }

    #[test]
    fn cayley_outputs_unit_quaternions() {
        for b in [
            [0.0, 0.0, 0.0],
            [0.2, -0.3, 0.4],
            [2.0, -1.0, 0.5],
            [-3.0, 1.5, 0.25],
        ] {
            let q = cayley_to_unit_quat(b);
            assert_close(norm4(q), 1.0, 1e-6);
            assert!(q[0] >= 0.0);
        }
    }

    #[test]
    fn zero_bivector_is_identity_quaternion() {
        assert_eq!(cayley_to_unit_quat([0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn identity_rotation_leaves_vector_unchanged() {
        let v = [0.25, -0.5, 2.0];
        assert_eq!(quat_rotate([1.0, 0.0, 0.0, 0.0], v), v);
    }

    #[test]
    fn unit_rotation_preserves_vector_norm() {
        let q = cayley_to_unit_quat([0.2, -0.3, 0.4]);
        let v = [0.7, -1.1, 0.6];
        assert_close(norm3(quat_rotate(q, v)), norm3(v), 1e-5);
    }

    #[test]
    fn conjugate_rotation_round_trips() {
        let q = cayley_to_unit_quat([0.2, -0.3, 0.4]);
        let v = [0.7, -1.1, 0.6];
        let rotated = quat_rotate(q, v);
        let round_trip = quat_rotate(quat_conjugate(q), rotated);
        for i in 0..3 {
            assert_close(round_trip[i], v[i], 1e-5);
        }
    }

    #[test]
    fn cayley_half_turn_about_z_rotates_x_to_y() {
        let q = cayley_to_unit_quat([0.0, 0.0, 1.0]);
        let out = quat_rotate(q, [1.0, 0.0, 0.0]);
        assert_close(out[0], 0.0, 1e-6);
        assert_close(out[1], 1.0, 1e-6);
        assert_close(out[2], 0.0, 1e-6);
    }

    #[test]
    fn hamilton_identity_is_neutral() {
        let q = cayley_to_unit_quat([0.2, -0.3, 0.4]);
        assert_eq!(quat_mul([1.0, 0.0, 0.0, 0.0], q), q);
        assert_eq!(quat_mul(q, [1.0, 0.0, 0.0, 0.0]), q);
    }
}
