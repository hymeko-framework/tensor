//! Scenario: the analytic gradients are the gradients of the forward pass.
//!
//! This crate's whole premise is closed-form (tape-free) differentiation — the backward is
//! hand-derived, not produced by an autograd engine. That makes a finite-difference check the
//! load-bearing correctness test: nothing else would catch a sign slip or a missing chain factor,
//! and the inline unit tests exercise the kernels rather than the public API end to end.
//!
//! Central differences, `h = 1e-3` on `f32`. The tolerance is loose (2e-2 relative) because f32
//! central differencing on a spline near a knot is itself only good to ~1e-3; the test is
//! discriminating for structural errors (wrong index, wrong sign, dropped term), not for the last bits.

use hymeko_spline::{
    catmull_rom_backward, catmull_rom_forward, chebyshev_cr_backward, chebyshev_cr_forward,
};

/// Deterministic LCG — no `rand` dependency; this crate is dependency-free with `parallel` off
/// and a test-only dependency would undo that.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs() / (p.abs().max(q.abs()).max(1e-3)))
        .fold(0.0f32, f32::max)
}

#[test]
fn catmull_rom_grad_coef_matches_central_differences() {
    let (n, channels, grid) = (24usize, 3usize, 6usize);
    let mut r = Lcg(0x5eed);
    let coef: Vec<f32> = (0..channels * grid).map(|_| r.next_f32()).collect();
    let x: Vec<f32> = (0..n * channels).map(|_| r.next_f32()).collect();
    let grad_y: Vec<f32> = (0..n * channels).map(|_| r.next_f32()).collect();

    let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
    let analytic = catmull_rom_backward(&coef, &cache, &grad_y).grad_coef;

    // dL/dcoef_i for L = <grad_y, forward(coef)>, by central difference on coef_i.
    let h = 1e-3f32;
    let mut numeric = vec![0.0f32; coef.len()];
    for i in 0..coef.len() {
        let mut cp = coef.clone();
        cp[i] = coef[i] + h;
        let (yp, _) = catmull_rom_forward(&cp, &x, n, channels, grid);
        cp[i] = coef[i] - h;
        let (ym, _) = catmull_rom_forward(&cp, &x, n, channels, grid);
        numeric[i] = yp
            .iter()
            .zip(&ym)
            .zip(&grad_y)
            .map(|((p, m), g)| g * (p - m) / (2.0 * h))
            .sum();
    }
    let e = max_rel_err(&analytic, &numeric);
    assert!(
        e < 2e-2,
        "grad_coef max rel err {e:.4}\nanalytic {analytic:?}\nnumeric {numeric:?}"
    );
}

#[test]
fn catmull_rom_grad_x_matches_central_differences() {
    let (n, channels, grid) = (16usize, 2usize, 6usize);
    let mut r = Lcg(0xc0ffee);
    let coef: Vec<f32> = (0..channels * grid).map(|_| r.next_f32()).collect();
    // Kept inside (-1, 1) and away from the clamp so the derivative is not one-sided.
    let x: Vec<f32> = (0..n * channels).map(|_| r.next_f32() * 0.8).collect();
    let grad_y: Vec<f32> = (0..n * channels).map(|_| r.next_f32()).collect();

    let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
    let analytic = catmull_rom_backward(&coef, &cache, &grad_y).grad_x;

    let h = 1e-3f32;
    let mut numeric = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        let mut xp = x.clone();
        xp[i] = x[i] + h;
        let (yp, _) = catmull_rom_forward(&coef, &xp, n, channels, grid);
        xp[i] = x[i] - h;
        let (ym, _) = catmull_rom_forward(&coef, &xp, n, channels, grid);
        // Only element i of the output row changes; the dot product picks it up regardless.
        numeric[i] = yp
            .iter()
            .zip(&ym)
            .zip(&grad_y)
            .map(|((p, m), g)| g * (p - m) / (2.0 * h))
            .sum();
    }
    let e = max_rel_err(&analytic, &numeric);
    assert!(e < 2e-2, "grad_x max rel err {e:.4}");
}

#[test]
fn chebyshev_cr_grad_coef_matches_central_differences() {
    let (n, channels, grid, k) = (20usize, 2usize, 6usize, 6usize);
    let mut r = Lcg(0xabcdef);
    let coef: Vec<f32> = (0..channels * k).map(|_| r.next_f32()).collect();
    let x: Vec<f32> = (0..n * channels).map(|_| r.next_f32() * 0.9).collect();
    let grad_y: Vec<f32> = (0..n * channels).map(|_| r.next_f32()).collect();

    let (_, cache, control, basis) = chebyshev_cr_forward(&coef, &x, n, channels, grid, k);
    let analytic = chebyshev_cr_backward(&control, &basis, &cache, &grad_y, k).grad_coef;

    let h = 1e-3f32;
    let mut numeric = vec![0.0f32; coef.len()];
    for i in 0..coef.len() {
        let mut cp = coef.clone();
        cp[i] = coef[i] + h;
        let (yp, _, _, _) = chebyshev_cr_forward(&cp, &x, n, channels, grid, k);
        cp[i] = coef[i] - h;
        let (ym, _, _, _) = chebyshev_cr_forward(&cp, &x, n, channels, grid, k);
        numeric[i] = yp
            .iter()
            .zip(&ym)
            .zip(&grad_y)
            .map(|((p, m), g)| g * (p - m) / (2.0 * h))
            .sum();
    }
    let e = max_rel_err(&analytic, &numeric);
    assert!(e < 2e-2, "chebyshev-CR grad_coef max rel err {e:.4}");
}
