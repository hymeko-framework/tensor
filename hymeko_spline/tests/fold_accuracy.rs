//! **The coefficient gradient's accuracy, folded once per call against once per block** (Nagare plan
//! 2026-10-24-computed-parameters-once, Amendments 1 and 2: the decomposition of E1 after its A/B).
//!
//! The control-point gradient is linear in the output cotangent: `∂L/∂q_k = Σ_e gy_e w_k(t_e)`. The reference sums
//! each term in `f64`, its weights `w_k(t_e)` read from the forward with a unit control point (the forward is linear
//! in the control points). The order the kernel replaced -- each row block folded through the leaf basis, rounded,
//! and the blocks added in `f32` -- is the kernel run over each block's rows on its own (one block each) and the
//! results added in block order. Both orders are measured against the reference: the change must not make the
//! gradient less accurate, nor bias it. The blocks' moments added in `f32` before the fold made it about twice as
//! inexact past a thousand rows (3.8e-7 against 1.9e-7 at 2048 rows), since the fold cancels between moments that
//! the sum grows; added in `f64` it is at least as exact as the per-block fold (measured 2026-10-24).

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, Knots};

/// `n` values in `[-1.2, 1.2]` from `seed`: some outside the clamp interval, as the cells' inputs are.
fn field(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 2401) as f32 / 1000.0 - 1.2
        })
        .collect()
}

/// The reference gradient, `[channels, grid]`, every term summed in `f64`.
fn reference(x: &[f32], gy: &[f32], (n, channels, grid): (usize, usize, usize)) -> Vec<f64> {
    let knots = Knots::uniform(grid);
    let mut grad = vec![0.0f64; channels * grid];
    for k in 0..grid {
        let mut unit = vec![0.0f32; channels * grid];
        for c in 0..channels {
            unit[c * grid + k] = 1.0;
        }
        let (w, _) = catmull_rom_forward(&unit, x, n, channels, &knots);
        for (e, (&w, &g)) in w.iter().zip(gy).enumerate() {
            grad[(e % channels) * grid + k] += f64::from(w) * f64::from(g);
        }
    }
    grad
}

/// The kernel's rows per block for `n` rows: the partition `block_stride` makes.
fn rows_per_block(n: usize) -> usize {
    n.div_ceil(64).max(32)
}

/// The order replaced: each block's rows folded on their own, the blocks added in `f32` in block order. A chunk of more
/// than 32 rows is itself split by the kernel, so past 2048 rows this is the old order at 32-row blocks, which is at
/// least as accurate as the old 64-row ones.
fn per_block(
    coef: &[f32],
    x: &[f32],
    gy: &[f32],
    (n, channels, grid): (usize, usize, usize),
) -> Vec<f32> {
    let knots = Knots::uniform(grid);
    let per = rows_per_block(n);
    let mut grad = vec![0.0f32; channels * grid];
    for start in (0..n).step_by(per) {
        let rows = per.min(n - start);
        let span = start * channels..(start + rows) * channels;
        let (_, cache) = catmull_rom_forward(coef, &x[span.clone()], rows, channels, &knots);
        let back = catmull_rom_backward(coef, &cache, &gy[span]);
        for (g, b) in grad.iter_mut().zip(back.grad_coef) {
            *g += b;
        }
    }
    grad
}

/// `(relative RMS error, normalised mean signed error)` of `got` against `exact`.
fn errors(got: &[f32], exact: &[f64]) -> (f64, f64) {
    let norm = exact.iter().map(|v| v * v).sum::<f64>().sqrt();
    let diff: Vec<f64> = got
        .iter()
        .zip(exact)
        .map(|(&g, &e)| f64::from(g) - e)
        .collect();
    let rms = diff.iter().map(|d| d * d).sum::<f64>().sqrt() / norm;
    let bias = diff.iter().sum::<f64>() / norm;
    (rms, bias)
}

/// At the cells' shapes (rows of 32 channels, grid 6, from one block to the 64-block cap), the gradient folded once
/// per call is as accurate as the per-block fold it replaced, and neither is biased beyond rounding.
#[test]
fn folding_once_per_call_is_as_accurate_as_folding_per_block() {
    let (channels, grid) = (32usize, 6usize);
    for (seed, n) in [64usize, 256, 1024, 2048, 4096].into_iter().enumerate() {
        let shape = (n, channels, grid);
        let (x, coef, gy) = (
            field(n * channels, seed as u64),
            field(channels * grid, 100 + seed as u64),
            field(n * channels, 200 + seed as u64),
        );
        let exact = reference(&x, &gy, shape);
        let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, &Knots::uniform(grid));
        let once = catmull_rom_backward(&coef, &cache, &gy).grad_coef;
        let blocks = per_block(&coef, &x, &gy, shape);
        let ((rms_once, bias_once), (rms_blocks, bias_blocks)) =
            (errors(&once, &exact), errors(&blocks, &exact));
        eprintln!(
            "n {n:5}: once per call rms {rms_once:.2e} bias {bias_once:+.2e} | per block rms {rms_blocks:.2e} bias {bias_blocks:+.2e}"
        );
        assert!(rms_once <= 1e-5 && rms_blocks <= 1e-5, "rounding level");
        assert!(
            rms_once <= 1.1 * rms_blocks + 2e-8,
            "{n}: once {rms_once:.2e} against blocks {rms_blocks:.2e}"
        );
        assert!(bias_once.abs() <= 1e-5, "{n}: biased {bias_once:.2e}");
    }
}
