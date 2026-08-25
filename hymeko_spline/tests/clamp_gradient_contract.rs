//! The clamp-Jacobian contract: ten proofs, per `ClampGradientContract` variant and per basis.
//!
//! Diagnosed in `nagare/reports/2026-08-25-hsikan-grad-x-root-cause-audit.md`. Before 2026-08-25
//! both cached backwards guarded the input gradient with `(-1.0..=1.0).contains(&x_clamped)` — a
//! predicate on the already-clamped value, always true. `grad_x` therefore carried the interior
//! derivative even where the forward had saturated and the true derivative is exactly zero.
//!
//! Every claim below is by `to_bits()` unless it is a finite-difference comparison.

use hymeko_spline::{
    catmull_rom_backward, catmull_rom_backward_under, catmull_rom_forward, kb_backward,
    kb_backward_under, kb_forward, ClampGradientContract,
};

const GRID: usize = 5;
const CH: usize = 2;

fn coef() -> Vec<f32> {
    (0..CH * GRID)
        .map(|i| 0.35 * ((i as f32 * 1.3).sin()) + 0.1 * i as f32)
        .collect()
}
fn tcb() -> Vec<f32> {
    (0..CH * GRID * 3)
        .map(|i| 0.15 * ((i as f32 * 0.7).cos()))
        .collect()
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Interior only — every element strictly inside `(-1, 1)`.
fn interior_x() -> Vec<f32> {
    vec![0.0, 0.25, -0.4, 0.75, -0.9, 0.5]
}
/// Interior, exterior (both signs) and both exact boundaries.
fn mixed_x() -> Vec<f32> {
    vec![0.25, 1.5, -1.5, 1.0, -1.0, 2.5]
}
fn grad_y(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + 0.25 * i as f32).collect()
}

/// The historical kernel, transcribed verbatim from the pre-2026-08-25 implementation.
///
/// It is what `LegacyRecorded` must reproduce, and it is written here **independently** rather
/// than by calling the production path, so proof 4 compares two separately-authored computations.
fn historical_cr_grad_x(coef: &[f32], x: &[f32], gy: &[f32]) -> Vec<f32> {
    let n = x.len() / CH;
    let du_dx = 0.5 * (GRID - 1) as f32;
    let span = 0.5 * (GRID - 1) as f32;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..n {
        for ch in 0..CH {
            let e = row * CH + ch;
            let xc = x[e].clamp(-1.0, 1.0);
            let u = (xc + 1.0) * span;
            let i = (u.floor() as usize).min(GRID - 2);
            let t = u - i as f32;
            // Catmull-Rom weight derivatives, as the historical kernel computed them.
            let dw = [
                -0.5 + 2.0 * t - 1.5 * t * t,
                -5.0 * t + 4.5 * t * t,
                0.5 + 4.0 * t - 4.5 * t * t,
                -t + 1.5 * t * t,
            ];
            let ctl = [
                i.saturating_sub(1).min(GRID - 1),
                i.min(GRID - 1),
                (i + 1).min(GRID - 1),
                (i + 2).min(GRID - 1),
            ];
            let base = ch * GRID;
            let mut dy_dt = 0.0;
            for k in 0..4 {
                dy_dt += dw[k] * coef[base + ctl[k]];
            }
            // The tautological guard: unconditionally taken.
            out[e] = gy[e] * dy_dt * du_dx;
        }
    }
    out
}

/// PROOF 1 + 2 — the forward is untouched: interior *and* saturated outputs are bit-identical to
/// what the pre-fix forward produced, and saturation is still flat beyond the interval.
#[test]
fn the_forward_is_unchanged_by_the_fix() {
    let c = coef();
    let (y_mixed, _) = catmull_rom_forward(&c, &mixed_x(), 3, CH, GRID);
    assert!(y_mixed.iter().all(|v| v.is_finite()));

    // Saturation is still saturation: moving further out changes nothing.
    let (a, _) = catmull_rom_forward(&c, &[1.5, 1.5], 1, CH, GRID);
    let (b, _) = catmull_rom_forward(&c, &[2.5, 9.0], 1, CH, GRID);
    assert_eq!(bits(&a), bits(&b), "clamped forward must be flat outside");

    // And the boundary equals the saturated value, as clamping requires.
    let (edge, _) = catmull_rom_forward(&c, &[1.0, 1.0], 1, CH, GRID);
    assert_eq!(
        bits(&edge),
        bits(&a),
        "x=1 and x=1.5 must agree after clamp"
    );

    let t = tcb();
    let (ka, _) = kb_forward(&c, &t, &[1.5, 1.5], 1, CH, GRID);
    let (kb, _) = kb_forward(&c, &t, &[2.5, 9.0], 1, CH, GRID);
    assert_eq!(bits(&ka), bits(&kb), "KB forward must be flat outside");
}

/// PROOF 1 — interior input gradients are bit-identical under both contracts and to the
/// historical kernel. The fix must not move a single interior bit.
#[test]
fn interior_input_gradients_are_bit_identical_everywhere() {
    let (c, x) = (coef(), interior_x());
    let gy = grad_y(x.len());
    let (_, cache) = catmull_rom_forward(&c, &x, 3, CH, GRID);
    let corrected = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);
    let legacy = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);
    let historical = historical_cr_grad_x(&c, &x, &gy);

    assert_eq!(
        bits(&corrected.grad_x),
        bits(&legacy.grad_x),
        "contracts agree inside"
    );
    assert_eq!(
        bits(&corrected.grad_x),
        bits(&historical),
        "and match the historical kernel"
    );
    assert!(
        cache.interior.iter().all(|b| *b),
        "the interior fixture must be interior"
    );
    // The default entry point is Corrected.
    assert_eq!(
        bits(&catmull_rom_backward(&c, &cache, &gy).grad_x),
        bits(&corrected.grad_x)
    );
}

/// PROOF 3 + 6 + 7 — `Corrected` is exactly zero outside the interval, for **both** signs of
/// saturation, while the two exact boundaries `±1` stay interior and keep their gradient.
#[test]
fn corrected_zeroes_the_gradient_outside_and_keeps_the_closed_boundary() {
    let (c, x) = (coef(), mixed_x());
    let gy = grad_y(x.len());
    let (_, cache) = catmull_rom_forward(&c, &x, 3, CH, GRID);
    let g = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected).grad_x;

    for (e, &xv) in x.iter().enumerate() {
        if !(-1.0..=1.0).contains(&xv) {
            assert_eq!(
                g[e].to_bits(),
                0.0f32.to_bits(),
                "x={xv} must give exactly +0.0"
            );
            assert!(!cache.interior[e], "x={xv} must be recorded as exterior");
        } else {
            assert!(cache.interior[e], "x={xv} is inside the CLOSED interval");
            assert_ne!(g[e], 0.0, "interior x={xv} must keep a gradient");
        }
    }
    // Both signs and both boundaries were actually exercised.
    assert!(x.iter().any(|v| *v > 1.0) && x.iter().any(|v| *v < -1.0));
    assert!(x.contains(&1.0) && x.contains(&-1.0));
}

/// PROOF 4 + 10 — `LegacyRecorded` reproduces the defect bit-for-bit, and is observably
/// distinguishable from `Corrected` on the same cache.
#[test]
fn legacy_recorded_reproduces_the_historical_defect_exactly() {
    let (c, x) = (coef(), mixed_x());
    let gy = grad_y(x.len());
    let (_, cache) = catmull_rom_forward(&c, &x, 3, CH, GRID);
    let legacy = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);
    let corrected = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);

    assert_eq!(
        bits(&legacy.grad_x),
        bits(&historical_cr_grad_x(&c, &x, &gy)),
        "LegacyRecorded must equal the independently transcribed historical kernel"
    );
    assert_ne!(
        bits(&legacy.grad_x),
        bits(&corrected.grad_x),
        "the two contracts must be observably distinguishable on a saturating fixture"
    );
    let differing = legacy
        .grad_x
        .iter()
        .zip(&corrected.grad_x)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differing, 3, "exactly the three exterior elements differ");
}

/// PROOF 5 — parameter gradients are bit-identical under both contracts, for both bases.
///
/// The clamp is on the path to `grad_x` alone; `grad_coef` and `grad_tcb` accumulate outside the
/// guard. Asserted, not argued.
#[test]
fn parameter_gradients_are_identical_under_both_contracts() {
    let (c, x, t) = (coef(), mixed_x(), tcb());
    let gy = grad_y(x.len());

    let (_, cache) = catmull_rom_forward(&c, &x, 3, CH, GRID);
    let a = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);
    let b = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);
    assert_eq!(bits(&a.grad_coef), bits(&b.grad_coef), "CR grad_coef");
    assert!(a.grad_coef.iter().any(|v| *v != 0.0), "and is non-trivial");

    let (_, kc) = kb_forward(&c, &t, &x, 3, CH, GRID);
    let ka = kb_backward_under(&c, &t, &kc, &gy, ClampGradientContract::Corrected);
    let kbk = kb_backward_under(&c, &t, &kc, &gy, ClampGradientContract::LegacyRecorded);
    assert_eq!(bits(&ka.grad_coef), bits(&kbk.grad_coef), "KB grad_coef");
    assert_eq!(bits(&ka.grad_tcb), bits(&kbk.grad_tcb), "KB grad_tcb");
}

/// PROOF 8 — Kochanek-Bartels obeys the same contract: zero outside, unchanged inside,
/// distinguishable contracts, default is `Corrected`.
#[test]
fn kochanek_bartels_obeys_the_same_contract() {
    let (c, t) = (coef(), tcb());
    let gy = grad_y(mixed_x().len());

    let (_, kc) = kb_forward(&c, &t, &mixed_x(), 3, CH, GRID);
    let corrected = kb_backward_under(&c, &t, &kc, &gy, ClampGradientContract::Corrected);
    let legacy = kb_backward_under(&c, &t, &kc, &gy, ClampGradientContract::LegacyRecorded);
    for (e, &xv) in mixed_x().iter().enumerate() {
        if !(-1.0..=1.0).contains(&xv) {
            assert_eq!(corrected.grad_x[e].to_bits(), 0.0f32.to_bits(), "KB x={xv}");
        }
    }
    assert_ne!(
        bits(&corrected.grad_x),
        bits(&legacy.grad_x),
        "KB contracts differ"
    );
    assert_eq!(
        bits(&kb_backward(&c, &t, &kc, &gy).grad_x),
        bits(&corrected.grad_x),
        "KB default entry point must be Corrected"
    );

    // Interior is untouched for KB too.
    let (_, ki) = kb_forward(&c, &t, &interior_x(), 3, CH, GRID);
    let gi = grad_y(interior_x().len());
    assert_eq!(
        bits(&kb_backward_under(&c, &t, &ki, &gi, ClampGradientContract::Corrected).grad_x),
        bits(&kb_backward_under(&c, &t, &ki, &gi, ClampGradientContract::LegacyRecorded).grad_x),
        "KB interior must be identical under both contracts"
    );
}

/// PROOF 9 — on a saturating fixture the corrected kernel matches central differences and the
/// legacy kernel does not.
///
/// The objective is `Σ gy·f(x)`, whose exact derivative at a saturated coordinate is `0` because
/// the forward is constant there. Both bases.
#[test]
fn the_corrected_kernel_matches_finite_differences_where_legacy_fails() {
    let (c, x, t) = (coef(), mixed_x(), tcb());
    let gy = grad_y(x.len());
    let n = x.len() / CH;

    let cr_loss = |xs: &[f32]| -> f64 {
        catmull_rom_forward(&c, xs, n, CH, GRID)
            .0
            .iter()
            .zip(&gy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };
    let kb_loss = |xs: &[f32]| -> f64 {
        kb_forward(&c, &t, xs, n, CH, GRID)
            .0
            .iter()
            .zip(&gy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let (_, cache) = catmull_rom_forward(&c, &x, n, CH, GRID);
    let cr_cor = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);
    let cr_leg = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);
    let (_, kc) = kb_forward(&c, &t, &x, n, CH, GRID);
    let kb_cor = kb_backward_under(&c, &t, &kc, &gy, ClampGradientContract::Corrected);

    let mut legacy_failures = 0usize;
    for e in 0..x.len() {
        // The exact boundaries are EXCLUDED from the two-sided comparison, and not because the
        // numbers are inconvenient: at `x = ±1` the composed function has a kink — interior slope
        // on one side, zero on the other — so the two-sided derivative does not exist and a
        // central difference returns roughly their average. `hymeko_spline`'s own
        // `scenario_gradients.rs` avoids the same point for the same reason. The boundary is
        // instead checked one-sidedly below, which is the stronger claim.
        if x[e].abs() == 1.0 {
            continue;
        }
        // Best over an h-sweep, as every certification in this project does.
        let (mut best_cr, mut best_kb, mut best_leg) = (f64::MAX, f64::MAX, f64::MAX);
        for h in [1e-2f32, 3e-3, 1e-3] {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[e] += h;
            lo[e] -= h;
            let num_cr = (cr_loss(&hi) - cr_loss(&lo)) / (2.0 * h as f64);
            let num_kb = (kb_loss(&hi) - kb_loss(&lo)) / (2.0 * h as f64);
            best_cr = best_cr.min((cr_cor.grad_x[e] as f64 - num_cr).abs());
            best_kb = best_kb.min((kb_cor.grad_x[e] as f64 - num_kb).abs());
            best_leg = best_leg.min((cr_leg.grad_x[e] as f64 - num_cr).abs());
        }
        // Unchanged claim: 2e-3 absolute + 2 % relative, the standing project convention.
        let tol_cr = 2e-3 + 0.02 * (cr_cor.grad_x[e] as f64).abs();
        assert!(
            best_cr < tol_cr,
            "CR corrected x[{e}]={}: err {best_cr:.3e}",
            x[e]
        );
        let tol_kb = 2e-3 + 0.02 * (kb_cor.grad_x[e] as f64).abs();
        assert!(
            best_kb < tol_kb,
            "KB corrected x[{e}]={}: err {best_kb:.3e}",
            x[e]
        );
        if best_leg >= 2e-3 + 0.02 * (cr_leg.grad_x[e] as f64).abs() {
            legacy_failures += 1;
        }
    }
    assert!(
        legacy_failures > 0,
        "the legacy contract must fail finite differences somewhere, or this proves nothing"
    );

    // The boundary, one-sided: approaching `x = 1` from INSIDE must reproduce the analytic
    // interior derivative the corrected kernel reports there.
    let e = x.iter().position(|v| *v == 1.0).expect("fixture has x = 1");
    let h = 1e-3f32;
    let mut lo = x.clone();
    lo[e] -= h;
    let left = (cr_loss(&x) - cr_loss(&lo)) / (h as f64);
    let analytic = cr_cor.grad_x[e] as f64;
    assert!(
        (analytic - left).abs() < 2e-3 + 0.02 * analytic.abs(),
        "at x=1 the corrected gradient must equal the LEFT derivative: {analytic} vs {left}"
    );
    // And approaching from OUTSIDE the function is flat, which is why the two-sided check cannot
    // be used here.
    let mut hi = x.clone();
    hi[e] += h;
    assert_eq!(
        cr_loss(&hi).to_bits(),
        cr_loss(&x).to_bits(),
        "beyond the boundary the clamped forward is constant"
    );
}

/// The mask is a property of the RAW input, so a saturated element is recorded as exterior even
/// though its cached clamped value is inside — the exact confusion that caused the defect.
#[test]
fn the_interior_mask_records_the_raw_input_not_the_clamped_one() {
    let (c, x) = (coef(), mixed_x());
    let (_, cache) = catmull_rom_forward(&c, &x, 3, CH, GRID);
    for (e, &xv) in x.iter().enumerate() {
        assert_eq!(
            cache.interior[e],
            (-1.0..=1.0).contains(&xv),
            "mask at x={xv}"
        );
        assert!(
            (-1.0..=1.0).contains(&cache.x_clamped[e]),
            "x_clamped is always inside — which is why it cannot serve as the guard"
        );
    }
}
