//! **Capture-before-deletion evidence for the spline clamp contract.**
//!
//! This suite is the precondition on mission Phase H: `ClampGradientContract` and its
//! `*_under` selectors may not leave `hymeko_spline` until the historical arithmetic they select
//! has been captured somewhere that outlives them. That capture is
//! `tests/support/frozen_clamp_oracle.rs`, and this file is the proof that the capture is
//! faithful — bit-for-bit, while the production selector is still here to compare against.
//!
//! # The four claims, kept apart
//!
//! 1. **The defect is real and the probe exercises it.** The two contracts genuinely differ, and
//!    they differ exactly on the saturated elements — nowhere else.
//! 2. **The canonical contract is the mathematics.** Zero input gradient outside the raw interval;
//!    the interior derivative certified against central finite differences.
//! 3. **The frozen oracle reproduces the retired arithmetic**, bit-for-bit, for CR, Chebyshev-CR
//!    and KB, on interior-only and saturating inputs alike.
//! 4. **The parameter gradients were never affected.** `grad_coef` and `grad_tcb` are identical
//!    under both contracts, so nothing recorded about them is at risk from the retirement.
//!
//! Claim 3 is the one that is deleted-and-replaced in Phase H (there will be no production
//! selector left to compare against); claims 1, 2 and 4 survive, and `frozen_clamp_fixture.rs`
//! carries claim 3's numbers forward on disk.

mod support;

use hymeko_numeric_vv::fd;
use hymeko_spline::{
    catmull_rom_backward, catmull_rom_backward_under, catmull_rom_forward,
    chebyshev_cr_backward_under, chebyshev_cr_forward, kb_backward, kb_backward_under, kb_forward,
    ClampGradientContract,
};
use support::frozen_clamp_oracle as frozen;

// ── deterministic fixtures ────────────────────────────────────────────────────────────────────
// No RNG: a fixture the reader can evaluate by hand is worth more here than a seeded one.

const N: usize = 7;
const CH: usize = 3;
const GRID: usize = 5;
const KB_GRID: usize = 5;

fn coef() -> Vec<f32> {
    (0..CH * GRID)
        .map(|i| 0.35 * (i as f32) - 0.9 + 0.2 * ((i % 3) as f32))
        .collect()
}

fn tcb_raw() -> Vec<f32> {
    (0..CH * KB_GRID * 3)
        .map(|i| 0.11 * (i as f32 % 7.0) - 0.4)
        .collect()
}

fn grad_y() -> Vec<f32> {
    (0..N * CH)
        .map(|i| 0.5 - 0.13 * (i as f32) + 0.07 * ((i % 5) as f32))
        .collect()
}

/// Inputs that stay strictly inside `[-1, 1]`.
fn interior_x() -> Vec<f32> {
    (0..N * CH)
        .map(|i| -0.9 + 1.8 * (i as f32) / ((N * CH - 1) as f32))
        .collect()
}

/// Inputs that saturate on both sides and land on both endpoints.
fn saturating_x() -> Vec<f32> {
    frozen::saturating_probe(N, CH)
}

// ── claim 1: the defect is real, and only on the saturated elements ───────────────────────────

#[test]
fn the_probe_actually_saturates_on_both_sides_and_hits_both_endpoints() {
    let x = saturating_x();
    assert!(
        x.iter().any(|&v| v < -1.0),
        "no low saturation in the probe"
    );
    assert!(
        x.iter().any(|&v| v > 1.0),
        "no high saturation in the probe"
    );
    assert!(x.contains(&-1.0), "the closed lower endpoint is not probed");
    assert!(x.contains(&1.0), "the closed upper endpoint is not probed");
    assert!(
        x.iter()
            .any(|&v| (-1.0..=1.0).contains(&v) && v != -1.0 && v != 1.0),
        "the probe has no strictly-interior element to hold fixed"
    );
}

#[test]
fn the_two_contracts_differ_exactly_on_the_saturated_elements() {
    let (c, gy, x) = (coef(), grad_y(), saturating_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let corrected = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);
    let legacy = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);

    let mut differed = 0usize;
    for (i, &raw) in x.iter().enumerate() {
        let interior = (-1.0..=1.0).contains(&raw);
        if interior {
            assert_eq!(
                corrected.grad_x[i].to_bits(),
                legacy.grad_x[i].to_bits(),
                "element {i} (raw {raw}) is interior; the contracts must agree there"
            );
        } else {
            assert_eq!(
                corrected.grad_x[i], 0.0,
                "element {i} (raw {raw}) saturated; the corrected gradient must be exactly zero"
            );
            if legacy.grad_x[i] != 0.0 {
                differed += 1;
            }
        }
    }
    assert!(
        differed >= 4,
        "the probe must exercise the defect on several elements, it differed on {differed}"
    );
}

#[test]
fn on_a_wholly_interior_input_the_two_contracts_are_bit_identical() {
    let (c, gy, x) = (coef(), grad_y(), interior_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let a = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);
    let b = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);
    assert_eq!(bits(&a.grad_x), bits(&b.grad_x));
}

// ── claim 2: the canonical contract is the mathematics ────────────────────────────────────────

#[test]
fn the_canonical_input_gradient_is_exactly_zero_outside_the_raw_interval() {
    let (c, gy, x) = (coef(), grad_y(), saturating_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let g = catmull_rom_backward(&c, &cache, &gy).grad_x;
    for (i, &raw) in x.iter().enumerate() {
        if !(-1.0..=1.0).contains(&raw) {
            assert_eq!(g[i], 0.0, "element {i} (raw {raw})");
        }
    }
}

#[test]
fn the_canonical_kb_input_gradient_is_exactly_zero_outside_the_raw_interval() {
    let (c, t, gy, x) = (coef(), tcb_raw(), grad_y(), saturating_x());
    let (_, cache) = kb_forward(&c, &t, &x, N, CH, KB_GRID);
    let g = kb_backward(&c, &t, &cache, &gy).grad_x;
    for (i, &raw) in x.iter().enumerate() {
        if !(-1.0..=1.0).contains(&raw) {
            assert_eq!(g[i], 0.0, "element {i} (raw {raw})");
        }
    }
}

#[test]
fn the_canonical_cr_input_gradient_matches_central_finite_differences() {
    let (c, gy) = (coef(), grad_y());
    // Strictly interior, and away from the knots: the CR input derivative is piecewise cubic and
    // continuous, but a probe straddling a knot would measure the average of two segments.
    let x: Vec<f32> = (0..N * CH)
        .map(|i| -0.72 + 0.13 * (i as f32) % 1.44 - 0.0)
        .map(|v| v.clamp(-0.85, 0.85))
        .collect();

    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let analytic: Vec<f64> = catmull_rom_backward(&c, &cache, &gy)
        .grad_x
        .iter()
        .map(|&v| v as f64)
        .collect();

    let loss = |probe: &[f32]| {
        let (y, _) = catmull_rom_forward(&c, probe, N, CH, GRID);
        y.iter()
            .zip(&gy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };
    let numeric = fd::gradient(loss, &x, 1e-3);

    let err = fd::max_rel_error(&numeric, &analytic);
    assert!(
        err < 3e-2,
        "FD/adjoint disagreement {err}\nfd={numeric:?}\nad={analytic:?}"
    );
}

#[test]
fn the_canonical_kb_input_gradient_matches_central_finite_differences() {
    let (c, t, gy) = (coef(), tcb_raw(), grad_y());
    let x: Vec<f32> = (0..N * CH)
        .map(|i| (-0.7 + 0.11 * (i as f32)).clamp(-0.85, 0.85))
        .collect();

    let (_, cache) = kb_forward(&c, &t, &x, N, CH, KB_GRID);
    let analytic: Vec<f64> = kb_backward(&c, &t, &cache, &gy)
        .grad_x
        .iter()
        .map(|&v| v as f64)
        .collect();

    let loss = |probe: &[f32]| {
        let (y, _) = kb_forward(&c, &t, probe, N, CH, KB_GRID);
        y.iter()
            .zip(&gy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };
    let numeric = fd::gradient(loss, &x, 1e-3);

    let err = fd::max_rel_error(&numeric, &analytic);
    assert!(
        err < 3e-2,
        "FD/adjoint disagreement {err}\nfd={numeric:?}\nad={analytic:?}"
    );
}

// ── claim 3: the frozen oracle reproduces the retired arithmetic, bit-for-bit ─────────────────

#[test]
fn frozen_oracle_reproduces_legacy_cr_grad_x_on_a_saturating_input() {
    let (c, gy, x) = (coef(), grad_y(), saturating_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let production =
        catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded).grad_x;
    let oracle = frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy);
    assert_eq!(bits(&oracle), bits(&production));
}

#[test]
fn frozen_oracle_reproduces_legacy_cr_grad_x_on_an_interior_input() {
    let (c, gy, x) = (coef(), grad_y(), interior_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let production =
        catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded).grad_x;
    let oracle = frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy);
    assert_eq!(bits(&oracle), bits(&production));
}

#[test]
fn frozen_oracle_reproduces_legacy_kb_grad_x() {
    let (c, t, gy) = (coef(), tcb_raw(), grad_y());
    for x in [saturating_x(), interior_x()] {
        let (_, cache) = kb_forward(&c, &t, &x, N, CH, KB_GRID);
        let production =
            kb_backward_under(&c, &t, &cache, &gy, ClampGradientContract::LegacyRecorded).grad_x;
        let oracle = frozen::kb_legacy_grad_x(&c, &t, &x, N, CH, KB_GRID, &gy);
        assert_eq!(bits(&oracle), bits(&production));
    }
}

#[test]
fn frozen_oracle_reproduces_legacy_chebyshev_cr_grad_x() {
    const K: usize = 4;
    let cheb_coef: Vec<f32> = (0..CH * K).map(|i| 0.4 - 0.17 * (i as f32)).collect();
    let (gy, x) = (grad_y(), saturating_x());

    let (_, cache, control, basis) = chebyshev_cr_forward(&cheb_coef, &x, N, CH, GRID, K);
    let production = chebyshev_cr_backward_under(
        &control,
        &basis,
        &cache,
        &gy,
        K,
        ClampGradientContract::LegacyRecorded,
    )
    .grad_x;

    // The Chebyshev-CR input gradient IS the CR one, evaluated at the generated control points —
    // the crate documents the contract as reaching `grad_x` only. The oracle therefore needs no
    // Chebyshev-specific arm, and this test is what certifies that claim rather than assuming it.
    let oracle = frozen::cr_legacy_grad_x(&control, &x, N, CH, GRID, &gy);
    assert_eq!(bits(&oracle), bits(&production));
}

#[test]
fn the_oracle_is_not_vacuous_it_differs_from_the_canonical_result() {
    let (c, gy, x) = (coef(), grad_y(), saturating_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let canonical = catmull_rom_backward(&c, &cache, &gy).grad_x;
    let oracle = frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy);
    assert_ne!(
        bits(&oracle),
        bits(&canonical),
        "an oracle that equals the canonical result on a saturating probe proves nothing"
    );
}

// ── claim 4: the parameter gradients were never affected ──────────────────────────────────────

#[test]
fn cr_parameter_gradients_are_identical_under_both_contracts() {
    let (c, gy, x) = (coef(), grad_y(), saturating_x());
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let a = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::Corrected);
    let b = catmull_rom_backward_under(&c, &cache, &gy, ClampGradientContract::LegacyRecorded);
    assert_eq!(bits(&a.grad_coef), bits(&b.grad_coef));
}

#[test]
fn kb_parameter_gradients_are_identical_under_both_contracts() {
    let (c, t, gy, x) = (coef(), tcb_raw(), grad_y(), saturating_x());
    let (_, cache) = kb_forward(&c, &t, &x, N, CH, KB_GRID);
    let a = kb_backward_under(&c, &t, &cache, &gy, ClampGradientContract::Corrected);
    let b = kb_backward_under(&c, &t, &cache, &gy, ClampGradientContract::LegacyRecorded);
    assert_eq!(bits(&a.grad_coef), bits(&b.grad_coef));
    assert_eq!(bits(&a.grad_tcb), bits(&b.grad_tcb));
}

// ── helper ────────────────────────────────────────────────────────────────────────────────────

/// Bit patterns, so `-0.0`/`0.0` and any NaN payload are distinguished rather than compared with
/// float equality.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
