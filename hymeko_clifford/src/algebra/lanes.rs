//! Quaternions as lanes: the `[w, x, y, z]` components each a contiguous plane over many quaternions, so that a
//! kernel over the lanes reads every component contiguously (plan nagare 2026-11-03-monogenic-cells, M1).
//!
//! An array of `[f32; 4]` puts each component at a 16-byte stride; a kernel reading such an array next to plane-laid
//! data is left scalar by the vectoriser's cost model (Nagare's bounded cells, 2026-11-03). Each function here is the
//! lane-wise counterpart of a by-value function in [`super::rotor`], with the same expressions in the same order, so a
//! lane equals the by-value result bit for bit on finite inputs. Unlike [`quat_mul`](super::quat_mul), nothing here
//! asserts finiteness per lane: a non-finite component propagates as IEEE arithmetic does.
//!
//! Every output plane is its own `&mut` parameter: the vectoriser sees that no written plane aliases a read one only
//! for reference parameters, not for references packed into a struct or tuple.

/// Lanes `0..n`, `n` the length of the first output: every plane cut to it, element by element.
#[inline(always)]
fn cut(planes: [&[f32]; 4], n: usize) -> [&[f32]; 4] {
    let [w, x, y, z] = planes;
    [&w[..n], &x[..n], &y[..n], &z[..n]]
}

/// The Hamilton product `a·b` per lane, written into `(w, x, y, z)`; with `CONJ_A` the left factor is `ā`, with `CONJ_B`
/// the right factor is `b̄` (the pullback's factors: for `c = a·b`, `∂L/∂a = c̄'·b̄` and `∂L/∂b = ā·c̄'`).
///
/// # Preconditions
/// Every plane of `a` and `b` is at least as long as `w`, and `x`, `y`, `z` are as long as `w`.
///
/// # Postconditions
/// Lane `i` equals `quat_mul(a_i', b_i')` bit for bit for finite components, where `a_i'` is `quat_conjugate(a_i)`
/// under `CONJ_A` and `a_i` otherwise (likewise `b`).
///
/// # Panics
/// When a plane is shorter than `w`.
#[inline]
pub fn quat_mul_lanes<const CONJ_A: bool, const CONJ_B: bool>(
    a: [&[f32]; 4],
    b: [&[f32]; 4],
    w: &mut [f32],
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
) {
    let n = w.len();
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (cut(a, n), cut(b, n));
    let (x, y, z) = (&mut x[..n], &mut y[..n], &mut z[..n]);
    let (sa, sb) = (
        if CONJ_A { -1.0f32 } else { 1.0 },
        if CONJ_B { -1.0f32 } else { 1.0 },
    );
    for i in 0..n {
        // A conjugate negates the vector part; multiplying by the constant sign is exact and folds away when it is 1.
        let (p0, p1, p2, p3) = (a0[i], sa * a1[i], sa * a2[i], sa * a3[i]);
        let (q0, q1, q2, q3) = (b0[i], sb * b1[i], sb * b2[i], sb * b3[i]);
        w[i] = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3;
        x[i] = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2;
        y[i] = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1;
        z[i] = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0;
    }
}

/// The squared norm `w² + x² + y² + z²` per lane, into `out`.
///
/// # Preconditions
/// Every plane is at least as long as `out`.
///
/// # Postconditions
/// `out[i] = ((w_i² + x_i²) + y_i²) + z_i²`, summed in component order.
///
/// # Panics
/// When a plane is shorter than `out`.
#[inline]
pub fn quat_norm_sq_lanes(q: [&[f32]; 4], out: &mut [f32]) {
    let [w, x, y, z] = cut(q, out.len());
    for (i, o) in out.iter_mut().enumerate() {
        *o = w[i] * w[i] + x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
    }
}
