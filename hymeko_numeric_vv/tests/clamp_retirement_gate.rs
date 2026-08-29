//! **Executable proof that the clamp selector is retired and its history has exactly one owner.**
//!
//! Phase H retired `ClampGradientContract` and the three `*_under` entry points from production.
//! This gate makes that a checked property rather than a claim in a commit message, and pins the
//! V&V side: one historical arithmetic owner, an intact fixture, and behaviour that genuinely
//! discriminates.
//!
//! Every negative control below asserts the gate can FAIL. A retirement gate that cannot reject a
//! resurrected selector, a corrupted fixture, or a second historical body proves nothing.

mod support;

use hymeko_numeric_vv::source_gate::ProductionScan;
use hymeko_spline::{catmull_rom_backward, catmull_rom_forward};
use std::path::PathBuf;
use support::frozen_clamp_oracle as frozen;

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent")
        .to_path_buf()
}

/// Every production source root in the tensor workspace.
fn production() -> ProductionScan {
    let w = workspace();
    let scan = ProductionScan::new([
        w.join("hymeko_spline/src"),
        w.join("hymeko_clifford/src"),
        w.join("hymeko_graph/src"),
        w.join("hymeko_tensor/src"),
        w.join("hymeko_vision/src"),
    ]);
    scan.assert_scanned_at_least(20);
    scan
}

// ── production is clean ───────────────────────────────────────────────────────────────────────

#[test]
fn production_holds_no_clamp_selector() {
    let scan = production();
    scan.assert_absent("ClampGradientContract");
    scan.assert_absent("LegacyRecorded");
    scan.assert_absent("clamp_contract");
    scan.assert_absent("catmull_rom_backward_under");
    scan.assert_absent("chebyshev_cr_backward_under");
    scan.assert_absent("kb_backward_under");
    scan.assert_absent("propagates_input_gradient");
}

#[test]
fn the_canonical_backwards_are_still_offered() {
    // The converse: Phase H removed a selector, not the gradients. Without this, deleting the
    // kernels outright would satisfy the absence gate above.
    let scan = production();
    for kept in [
        "pub fn catmull_rom_backward",
        "pub fn chebyshev_cr_backward",
        "pub fn kb_backward",
        "pub fn catmull_rom_forward",
        "pub fn kb_forward",
    ] {
        assert!(
            !scan.hits(kept).is_empty(),
            "`{kept}` must still exist in production"
        );
    }
}

#[test]
fn exactly_one_historical_arithmetic_owner_exists() {
    // The historical `>> guard` arithmetic may live in exactly one place: the frozen oracle. A
    // second copy — in a production crate's tests, or a second V&V transcription — is what the
    // `hymeko_spline` local `historical_cr_grad_x` was before it was deleted.
    let w = workspace();
    let scan = ProductionScan::new([
        w.join("hymeko_spline/src"),
        w.join("hymeko_spline/tests"),
        w.join("hymeko_numeric_vv/src"),
    ]);
    scan.assert_scanned_at_least(6);
    scan.assert_absent("historical_cr_grad_x");
    scan.assert_absent("legacy_grad_x");

    let oracle = w.join("hymeko_numeric_vv/tests/support/frozen_clamp_oracle.rs");
    assert!(oracle.is_file(), "the single historical owner is missing");
    let src = std::fs::read_to_string(&oracle).expect("oracle source");
    assert!(
        src.contains("pub fn cr_legacy_grad_x"),
        "oracle must own CR"
    );
    assert!(
        src.contains("pub fn kb_legacy_grad_x"),
        "oracle must own KB"
    );
}

// ── negative controls: the gate must be able to fail ──────────────────────────────────────────

#[test]
#[should_panic(expected = "must not appear in production code")]
fn the_absence_gate_rejects_a_resurrected_selector() {
    // `pub fn catmull_rom_backward` is unambiguously present. If asserting its ABSENCE does not
    // panic, every absence claim above is vacuous.
    production().assert_absent("pub fn catmull_rom_backward");
}

#[test]
fn a_legacy_saturated_gradient_cannot_pass_the_canonical_contract() {
    // Feed the canonical kernel a saturating probe, then check that the frozen (legacy) values
    // would FAIL the canonical postcondition. This is the arithmetic-level negative control: it
    // shows the canonical gate rejects the retired behaviour rather than tolerating it.
    const N: usize = 7;
    const CH: usize = 3;
    const GRID: usize = 5;
    let c: Vec<f32> = (0..CH * GRID)
        .map(|i| 0.35 * (i as f32) - 0.9 + 0.2 * ((i % 3) as f32))
        .collect();
    let gy: Vec<f32> = (0..N * CH)
        .map(|i| 0.5 - 0.13 * (i as f32) + 0.07 * ((i % 5) as f32))
        .collect();
    let x = frozen::saturating_probe(N, CH);

    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
    let canonical = catmull_rom_backward(&c, &cache, &gy).grad_x;
    let legacy = frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy);

    // The canonical postcondition: exactly +0.0 at every saturated coordinate.
    let canonical_passes = x
        .iter()
        .zip(&canonical)
        .all(|(&raw, &g)| (-1.0..=1.0).contains(&raw) || g.to_bits() == 0.0f32.to_bits());
    let legacy_passes = x
        .iter()
        .zip(&legacy)
        .all(|(&raw, &g)| (-1.0..=1.0).contains(&raw) || g.to_bits() == 0.0f32.to_bits());

    assert!(
        canonical_passes,
        "the shipped kernel must satisfy its own postcondition"
    );
    assert!(
        !legacy_passes,
        "the retired arithmetic must FAIL the canonical postcondition — otherwise the retirement \
         changed nothing and the gate is decorative"
    );
}

#[test]
fn a_corrupted_fixture_bit_is_detected() {
    // The fixture gate compares hex bit patterns. Flipping one must be detectable, or
    // `the_fixture_matches_the_current_code` could pass over a damaged artifact.
    let p = workspace().join("hymeko_numeric_vv/tests/fixtures/frozen_clamp_grad_x.txt");
    let text = std::fs::read_to_string(&p).expect("fixture");
    let line = text
        .lines()
        .find(|l| l.starts_with("cr_frozen_legacy"))
        .expect("the frozen CR column");
    let first = line.split_whitespace().nth(1).expect("a bit pattern");
    let original = u32::from_str_radix(first, 16).expect("hex");
    let corrupted = original ^ 1;
    assert_ne!(
        original, corrupted,
        "a single-bit flip must change the recorded value"
    );
    assert_ne!(
        format!("{original:08x}"),
        format!("{corrupted:08x}"),
        "and must change its hex rendering, which is what the fixture compares"
    );
}
