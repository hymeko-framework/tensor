//! Kochanek-Bartels (TCB) cubic spline op with closed-form backward.
//!
//! Port of `hymeko_neuro/hyperedge/splines.py::_kb_eval`. A per-channel cubic Hermite
//! spline on `[-1,1]` whose endpoint tangents are shaped by learnable **tension /
//! continuity / bias** `(t,c,b) ∈ (-1,1)` (via `tanh`). At `t=c=b=0` it reduces exactly to
//! Catmull-Rom (`ops::catmull_rom`); non-zero tension flattens/steepens the segment
//! (the extrapolation lever near the ±1 boundary).
//!
//! Parameters: control points `coef (channels, grid)` + `tcb_raw (channels, grid, 3)`
//! (unconstrained; `tanh`-mapped). All gradients are explicit.

/// Cubic Hermite basis `[h00,h10,h01,h11]` at segment coordinate `s ∈ [0,1]`.
fn hermite(s: f32) -> [f32; 4] {
    let (s2, s3) = (s * s, s * s * s);
    [
        2.0 * s3 - 3.0 * s2 + 1.0,
        s3 - 2.0 * s2 + s,
        -2.0 * s3 + 3.0 * s2,
        s3 - s2,
    ]
}

/// `d/ds` of the cubic Hermite basis.
fn hermite_deriv(s: f32) -> [f32; 4] {
    let s2 = s * s;
    [
        6.0 * s2 - 6.0 * s,
        3.0 * s2 - 4.0 * s + 1.0,
        -6.0 * s2 + 6.0 * s,
        3.0 * s2 - 2.0 * s,
    ]
}

fn control_indices(i: usize, grid: usize) -> [usize; 4] {
    [
        i.saturating_sub(1).min(grid - 1),
        i.min(grid - 1),
        (i + 1).min(grid - 1),
        (i + 2).min(grid - 1),
    ]
}

/// `tanh`-mapped `(t,c,b)` at channel `ch`, grid index `g`.
fn tcb_at(tcb_raw: &[f32], ch: usize, g: usize, grid: usize) -> (f32, f32, f32) {
    let base = (ch * grid + g) * 3;
    (
        tcb_raw[base].tanh(),
        tcb_raw[base + 1].tanh(),
        tcb_raw[base + 2].tanh(),
    )
}

/// Cache from the KB forward for the closed-form backward.
#[derive(Debug, Clone)]
pub struct KbCache {
    /// Whether each raw input lay inside the closed interval `[-1, 1]`.
    ///
    /// Private, like the rest of this cache. See [`crate::catmull_rom::CatmullRomCache::interior`]
    /// for why `x_clamped` cannot answer this question.
    interior: Vec<bool>,
    indices: Vec<usize>,
    s: Vec<f32>,
    n: usize,
    channels: usize,
    grid: usize,
}

/// Result of the KB backward.
#[derive(Debug, Clone)]
pub struct KbBackward {
    /// Gradient w.r.t. control points, flat `(channels, grid)`.
    pub grad_coef: Vec<f32>,
    /// Gradient w.r.t. TCB params, flat `(channels, grid, 3)`.
    pub grad_tcb: Vec<f32>,
    /// Gradient w.r.t. input, flat `(n, channels)`.
    pub grad_x: Vec<f32>,
}

/// KB out-tangent weights at `P_i`: `(wL, wR)`.
fn out_weights(t: f32, c: f32, b: f32) -> (f32, f32) {
    (
        (1.0 - t) * (1.0 + c) * (1.0 + b) * 0.5,
        (1.0 - t) * (1.0 - c) * (1.0 - b) * 0.5,
    )
}

/// KB in-tangent weights at `P_{i+1}`: `(wL, wR)`.
fn in_weights(t: f32, c: f32, b: f32) -> (f32, f32) {
    (
        (1.0 - t) * (1.0 - c) * (1.0 + b) * 0.5,
        (1.0 - t) * (1.0 + c) * (1.0 - b) * 0.5,
    )
}

/// Forward KB spline.
///
/// # Preconditions
/// `grid >= 4`; `coef.len() == channels·grid`; `tcb_raw.len() == channels·grid·3`;
/// `x.len() == n·channels`.
///
/// # Postconditions
/// Returns output `(n, channels)` and a backward cache; inputs clamped to `[-1,1]`.
///
/// # Panics
/// Panics if any buffer length is inconsistent.
pub fn kb_forward(
    coef: &[f32],
    tcb_raw: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
) -> (Vec<f32>, KbCache) {
    assert!(grid >= 4);
    assert_eq!(coef.len(), channels * grid);
    assert_eq!(tcb_raw.len(), channels * grid * 3);
    assert_eq!(x.len(), n * channels);
    let mut out = vec![0.0f32; x.len()];
    let mut interior = vec![false; x.len()];
    let mut indices = vec![0usize; x.len()];
    let mut ss = vec![0.0f32; x.len()];
    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let raw = x[idx];
            let xc = raw.clamp(-1.0, 1.0);
            let u = (xc + 1.0) * 0.5 * (grid - 1) as f32;
            let i = (u.floor() as usize).min(grid - 2);
            let s = u - i as f32;
            let ctrl = control_indices(i, grid);
            let base = ch * grid;
            let p: [f32; 4] = [
                coef[base + ctrl[0]],
                coef[base + ctrl[1]],
                coef[base + ctrl[2]],
                coef[base + ctrl[3]],
            ];
            let (ti, ci, bi) = tcb_at(tcb_raw, ch, i, grid);
            let (tp, cp, bp) = tcb_at(tcb_raw, ch, ctrl[2], grid);
            let (wl0, wr0) = out_weights(ti, ci, bi);
            let (wl1, wr1) = in_weights(tp, cp, bp);
            let d0 = wl0 * (p[1] - p[0]) + wr0 * (p[2] - p[1]);
            let d1 = wl1 * (p[2] - p[1]) + wr1 * (p[3] - p[2]);
            let h = hermite(s);
            out[idx] = h[0] * p[1] + h[1] * d0 + h[2] * p[2] + h[3] * d1;
            // Closed interval: `±1` is interior, matching the guard this replaces.
            interior[idx] = (-1.0..=1.0).contains(&raw);
            indices[idx] = i;
            ss[idx] = s;
        }
    }
    (
        out,
        KbCache {
            interior,
            indices,
            s: ss,
            n,
            channels,
            grid,
        },
    )
}

/// Backward KB spline → gradients for control points, TCB params, and input.
///
/// # Panics
/// Panics if buffer lengths are inconsistent with `cache`.
pub fn kb_backward(coef: &[f32], tcb_raw: &[f32], cache: &KbCache, grad_y: &[f32]) -> KbBackward {
    let (n, channels, grid) = (cache.n, cache.channels, cache.grid);
    assert_eq!(grad_y.len(), n * channels);
    let mut grad_coef = vec![0.0f32; channels * grid];
    let mut grad_tcb = vec![0.0f32; channels * grid * 3];
    let mut grad_x = vec![0.0f32; n * channels];
    let du_dx = 0.5 * (grid - 1) as f32;

    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let gy = grad_y[idx];
            let i = cache.indices[idx];
            let s = cache.s[idx];
            let ctrl = control_indices(i, grid);
            let base = ch * grid;
            let p: [f32; 4] = [
                coef[base + ctrl[0]],
                coef[base + ctrl[1]],
                coef[base + ctrl[2]],
                coef[base + ctrl[3]],
            ];
            let (ti, ci, bi) = tcb_at(tcb_raw, ch, i, grid);
            let (tp, cp, bp) = tcb_at(tcb_raw, ch, ctrl[2], grid);
            let (wl0, wr0) = out_weights(ti, ci, bi);
            let (wl1, wr1) = in_weights(tp, cp, bp);
            let h = hermite(s);

            // grad w.r.t. the four gathered control points.
            let dv_dp = [
                h[1] * (-wl0),
                h[0] + h[1] * (wl0 - wr0) + h[3] * (-wl1),
                h[2] + h[1] * wr0 + h[3] * (wl1 - wr1),
                h[3] * wr1,
            ];
            for (k, &g) in dv_dp.iter().enumerate() {
                grad_coef[base + ctrl[k]] += gy * g;
            }

            // grad w.r.t. TCB at i (out-tangent) and i+1 (in-tangent).
            let dv_dwl0 = gy * h[1] * (p[1] - p[0]);
            let dv_dwr0 = gy * h[1] * (p[2] - p[1]);
            accum_tcb_out(
                &mut grad_tcb,
                base_tcb(ch, i, grid),
                ti,
                ci,
                bi,
                dv_dwl0,
                dv_dwr0,
            );
            let dv_dwl1 = gy * h[3] * (p[2] - p[1]);
            let dv_dwr1 = gy * h[3] * (p[3] - p[2]);
            accum_tcb_in(
                &mut grad_tcb,
                base_tcb(ch, ctrl[2], grid),
                tp,
                cp,
                bp,
                dv_dwl1,
                dv_dwr1,
            );

            // grad w.r.t. input via the segment coordinate.
            // The clamp's own Jacobian; read from the recorded `interior` mask rather than from
            // `x_clamped`, which is always inside by construction.
            if cache.interior[idx] {
                let d0 = wl0 * (p[1] - p[0]) + wr0 * (p[2] - p[1]);
                let d1 = wl1 * (p[2] - p[1]) + wr1 * (p[3] - p[2]);
                let hd = hermite_deriv(s);
                let dv_ds = hd[0] * p[1] + hd[1] * d0 + hd[2] * p[2] + hd[3] * d1;
                grad_x[idx] = gy * dv_ds * du_dx;
            }
        }
    }
    KbBackward {
        grad_coef,
        grad_tcb,
        grad_x,
    }
}

fn base_tcb(ch: usize, g: usize, grid: usize) -> usize {
    (ch * grid + g) * 3
}

/// Accumulate the out-tangent TCB gradient (through `tanh`) at `base`.
fn accum_tcb_out(grad: &mut [f32], base: usize, t: f32, c: f32, b: f32, dwl: f32, dwr: f32) {
    // wL = (1-t)(1+c)(1+b)/2 ; wR = (1-t)(1-c)(1-b)/2
    let gt = dwl * (-(1.0 + c) * (1.0 + b) * 0.5) + dwr * (-(1.0 - c) * (1.0 - b) * 0.5);
    let gc = dwl * ((1.0 - t) * (1.0 + b) * 0.5) + dwr * (-(1.0 - t) * (1.0 - b) * 0.5);
    let gb = dwl * ((1.0 - t) * (1.0 + c) * 0.5) + dwr * (-(1.0 - t) * (1.0 - c) * 0.5);
    grad[base] += gt * (1.0 - t * t);
    grad[base + 1] += gc * (1.0 - c * c);
    grad[base + 2] += gb * (1.0 - b * b);
}

/// Accumulate the in-tangent TCB gradient (through `tanh`) at `base`.
fn accum_tcb_in(grad: &mut [f32], base: usize, t: f32, c: f32, b: f32, dwl: f32, dwr: f32) {
    // wL = (1-t)(1-c)(1+b)/2 ; wR = (1-t)(1+c)(1-b)/2
    let gt = dwl * (-(1.0 - c) * (1.0 + b) * 0.5) + dwr * (-(1.0 + c) * (1.0 - b) * 0.5);
    let gc = dwl * (-(1.0 - t) * (1.0 + b) * 0.5) + dwr * ((1.0 - t) * (1.0 - b) * 0.5);
    let gb = dwl * ((1.0 - t) * (1.0 - c) * 0.5) + dwr * (-(1.0 - t) * (1.0 + c) * 0.5);
    grad[base] += gt * (1.0 - t * t);
    grad[base + 1] += gc * (1.0 - c * c);
    grad[base + 2] += gb * (1.0 - b * b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catmull_rom::catmull_rom_forward;

    #[test]
    fn zero_tcb_equals_catmull_rom() {
        // t=c=b=0 (tcb_raw=0 → tanh=0) must reduce to Catmull-Rom exactly.
        let (channels, grid, n) = (2, 5, 4);
        let coef = vec![0.1, -0.2, 0.3, 0.7, -0.1, 0.5, -0.4, 0.2, 0.1, 0.9];
        let tcb = vec![0.0f32; channels * grid * 3];
        let x = vec![-0.7, -0.2, 0.15, 0.4, 0.72, -0.55, 0.9, -0.9];
        let (kb, _) = kb_forward(&coef, &tcb, &x, n, channels, grid);
        let (cr, _) = catmull_rom_forward(&coef, &x, n, channels, grid);
        assert!(kb.iter().zip(&cr).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn backward_matches_finite_difference() {
        let (channels, grid, n) = (2, 5, 3);
        let coef = vec![0.1, -0.2, 0.3, 0.7, -0.1, 0.5, -0.4, 0.2, 0.1, 0.9];
        let tcb: Vec<f32> = (0..channels * grid * 3)
            .map(|i| 0.3 * ((i as f32 * 0.9).sin()))
            .collect();
        let x = vec![-0.7, -0.2, 0.15, 0.4, 0.72, -0.55];
        let (_, cache) = kb_forward(&coef, &tcb, &x, n, channels, grid);
        let grad_y = vec![1.0f32; x.len()];
        let g = kb_backward(&coef, &tcb, &cache, &grad_y);
        let eps = 1e-3;
        let loss = |c: &[f32], tc: &[f32], xf: &[f32]| -> f32 {
            kb_forward(c, tc, xf, n, channels, grid).0.iter().sum()
        };
        for (idx, &gc) in g.grad_coef.iter().enumerate() {
            let mut cp = coef.clone();
            cp[idx] += eps;
            let mut cm = coef.clone();
            cm[idx] -= eps;
            let num = (loss(&cp, &tcb, &x) - loss(&cm, &tcb, &x)) / (2.0 * eps);
            assert!((gc - num).abs() < 1e-2, "grad_coef[{idx}] {gc} vs {num}");
        }
        for (idx, &gt) in g.grad_tcb.iter().enumerate() {
            let mut tp = tcb.clone();
            tp[idx] += eps;
            let mut tm = tcb.clone();
            tm[idx] -= eps;
            let num = (loss(&coef, &tp, &x) - loss(&coef, &tm, &x)) / (2.0 * eps);
            assert!((gt - num).abs() < 1e-2, "grad_tcb[{idx}] {gt} vs {num}");
        }
        for (idx, &gx) in g.grad_x.iter().enumerate() {
            let mut xp = x.clone();
            xp[idx] += eps;
            let mut xm = x.clone();
            xm[idx] -= eps;
            let num = (loss(&coef, &tcb, &xp) - loss(&coef, &tcb, &xm)) / (2.0 * eps);
            assert!((gx - num).abs() < 1e-2, "grad_x[{idx}] {gx} vs {num}");
        }
    }
}
