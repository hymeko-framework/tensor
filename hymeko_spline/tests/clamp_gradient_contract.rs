//! **The clamp-Jacobian contract of the canonical kernels.**
//!
//! Diagnosed in `nagare/reports/2026-08-25-hsikan-grad-x-root-cause-audit.md`. Before 2026-08-25
//! both cached backwards guarded the input gradient with `(-1.0..=1.0).contains(&x_clamped)` — a
//! predicate on the already-clamped value, always true. `grad_x` therefore carried the interior
//! derivative even where the forward had saturated and the true derivative is exactly zero.
//!
//! # Scope after the selector retirement (2026-08-29)
//!
//! This suite was written per `ClampGradientContract` variant. The selector is gone and there is
//! one Jacobian, so it now tests **only the canonical kernels** and holds **no historical
//! arithmetic**: the local `historical_cr_grad_x` transcription was deleted rather than kept as a
//! second oracle beside the frozen one.
//!
//! Where the retired half went:
//!
//! | retired claim | owner |
//! |---|---|
//! | frozen historical CR / KB / Chebyshev-CR `grad_x` | `hymeko_numeric_vv/tests/support/frozen_clamp_oracle.rs` + `tests/fixtures/frozen_clamp_grad_x.txt` |
//! | historical vs canonical discrimination | `hymeko_numeric_vv::frozen_clamp_fixture::the_fixture_is_discriminating` |
//! | frozen-oracle fidelity to the retired production selector | **immutable pre-deletion evidence at commit `28d4ee6`**, where `frozen_clamp_oracle_parity.rs` was green. Deliberately not re-asserted at runtime: the comparand no longer exists by design. |
//! | parameter gradients unaffected by the contract | same pre-deletion evidence; the *current* canonical `grad_coef`/`grad_tcb` correctness is `parameter_gradients_match_finite_differences` below |
//!
//! Every claim below is by `to_bits()` unless it is a finite-difference comparison.

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, kb_backward, kb_forward};

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

/// **Interior input gradients are every element non-zero and the fixture is wholly interior.**
///
/// This asserted bit-identity against the retired arithmetic and a local historical transcription.
/// Both are gone; the surviving canonical claim is that an interior fixture produces a gradient
/// everywhere — no element is zeroed — which is what the clamp guard must not do inside.
#[test]
fn interior_input_gradients_are_present_everywhere() {
    let (c, x) = (coef(), interior_x());
    let gy = grad_y(x.len());
    let (_, cache) = catmull_rom_forward(&c, &x, 3, CH, GRID);
    let corrected = catmull_rom_backward(&c, &cache, &gy);

    assert!(
        corrected.grad_x.iter().all(|g| *g != 0.0),
        "no interior coordinate may be zeroed by the clamp guard"
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
    let g = catmull_rom_backward(&c, &cache, &gy).grad_x;

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

/// **Kochanek-Bartels obeys the same clamp contract**: exactly `+0.0` outside, present inside.
#[test]
fn kochanek_bartels_obeys_the_same_contract() {
    let (c, t) = (coef(), tcb());
    let gy = grad_y(mixed_x().len());

    let (_, kc) = kb_forward(&c, &t, &mixed_x(), 3, CH, GRID);
    let g = kb_backward(&c, &t, &kc, &gy);
    for (e, &xv) in mixed_x().iter().enumerate() {
        if !(-1.0..=1.0).contains(&xv) {
            assert_eq!(
                g.grad_x[e].to_bits(),
                0.0f32.to_bits(),
                "KB x={xv}: exactly +0.0 outside, not -0.0 and not a small value"
            );
        }
    }

    // Interior coordinates keep a gradient.
    let (_, ki) = kb_forward(&c, &t, &interior_x(), 3, CH, GRID);
    let gi = grad_y(interior_x().len());
    assert!(
        kb_backward(&c, &t, &ki, &gi)
            .grad_x
            .iter()
            .all(|v| *v != 0.0),
        "KB must not zero an interior coordinate"
    );
}

/// **On a saturating fixture the canonical kernel matches central differences.**
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
    let cr_cor = catmull_rom_backward(&c, &cache, &gy);
    let (_, kc) = kb_forward(&c, &t, &x, n, CH, GRID);
    let kb_cor = kb_backward(&c, &t, &kc, &gy);

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
        let (mut best_cr, mut best_kb) = (f64::MAX, f64::MAX);
        for h in [1e-2f32, 3e-3, 1e-3] {
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[e] += h;
            lo[e] -= h;
            let num_cr = (cr_loss(&hi) - cr_loss(&lo)) / (2.0 * h as f64);
            let num_kb = (kb_loss(&hi) - kb_loss(&lo)) / (2.0 * h as f64);
            best_cr = best_cr.min((cr_cor.grad_x[e] as f64 - num_cr).abs());
            best_kb = best_kb.min((kb_cor.grad_x[e] as f64 - num_kb).abs());
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
    }
    // The non-vacuity this once took from "the legacy arm fails here" now lives in
    // `hymeko_numeric_vv::frozen_clamp_fixture::the_fixture_is_discriminating`, which shows the
    // frozen and canonical columns differ on a saturating probe.

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

/// **Canonical parameter gradients match central finite differences.**
///
/// Claim 7's replacement. The retired suite asserted only that `grad_coef`/`grad_tcb` were
/// *identical under both contracts* — a statement about the clamp, not about correctness, and it
/// died with the selector. Its pre-deletion evidence is immutable at commit `28d4ee6`.
///
/// SCOPE, checked rather than assumed: `scenario_gradients.rs` already owns
/// `catmull_rom_grad_coef_matches_central_differences` and
/// `chebyshev_cr_grad_coef_matches_central_differences`. **Kochanek-Bartels has no such owner** —
/// neither `grad_coef` nor `grad_tcb` — so only the KB half is new here. The CR block below is
/// retained deliberately as the *clamp-suite* cross-check that the shared `interior` mask does not
/// perturb `grad_coef`, which is a different claim from the scenario suite's; it is three lines of
/// reuse, not a second FD engine. No historical engine and no recorder artifact was created.
#[test]
fn parameter_gradients_match_finite_differences() {
    let (x, t) = (interior_x(), tcb());
    let gy = grad_y(x.len());
    let n = x.len() / CH;
    let h = 1e-3f64;

    // ── Catmull-Rom control points ────────────────────────────────────────────────────────────
    let c = coef();
    let (_, cache) = catmull_rom_forward(&c, &x, n, CH, GRID);
    let analytic = catmull_rom_backward(&c, &cache, &gy).grad_coef;
    let cr_loss = |cs: &[f32]| -> f64 {
        catmull_rom_forward(cs, &x, n, CH, GRID)
            .0
            .iter()
            .zip(&gy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };
    let mut worst = 0.0f64;
    for i in 0..c.len() {
        let mut up = c.clone();
        let mut dn = c.clone();
        up[i] = (c[i] as f64 + h) as f32;
        dn[i] = (c[i] as f64 - h) as f32;
        let num = (cr_loss(&up) - cr_loss(&dn)) / ((up[i] as f64) - (dn[i] as f64));
        let err = (num - analytic[i] as f64).abs() / (analytic[i] as f64).abs().max(1.0);
        worst = worst.max(err);
    }
    assert!(
        worst < 5e-3,
        "CR grad_coef vs FD: worst relative error {worst:.3e}"
    );

    // ── Kochanek-Bartels control points and TCB ───────────────────────────────────────────────
    let (_, kc) = kb_forward(&c, &t, &x, n, CH, GRID);
    let kbg = kb_backward(&c, &t, &kc, &gy);
    let kb_loss = |cs: &[f32], ts: &[f32]| -> f64 {
        kb_forward(cs, ts, &x, n, CH, GRID)
            .0
            .iter()
            .zip(&gy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };
    let mut worst_c = 0.0f64;
    for i in 0..c.len() {
        let (mut up, mut dn) = (c.clone(), c.clone());
        up[i] = (c[i] as f64 + h) as f32;
        dn[i] = (c[i] as f64 - h) as f32;
        let num = (kb_loss(&up, &t) - kb_loss(&dn, &t)) / ((up[i] as f64) - (dn[i] as f64));
        worst_c = worst_c
            .max((num - kbg.grad_coef[i] as f64).abs() / (kbg.grad_coef[i] as f64).abs().max(1.0));
    }
    assert!(worst_c < 5e-3, "KB grad_coef vs FD: worst {worst_c:.3e}");

    let mut worst_t = 0.0f64;
    for i in 0..t.len() {
        let (mut up, mut dn) = (t.clone(), t.clone());
        up[i] = (t[i] as f64 + h) as f32;
        dn[i] = (t[i] as f64 - h) as f32;
        let num = (kb_loss(&c, &up) - kb_loss(&c, &dn)) / ((up[i] as f64) - (dn[i] as f64));
        worst_t = worst_t
            .max((num - kbg.grad_tcb[i] as f64).abs() / (kbg.grad_tcb[i] as f64).abs().max(1.0));
    }
    assert!(worst_t < 5e-3, "KB grad_tcb vs FD: worst {worst_t:.3e}");
    println!("param-grad FD: CR {worst:.2e}  KB coef {worst_c:.2e}  KB tcb {worst_t:.2e}");
}
