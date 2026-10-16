//! **The historical clamp numbers, on disk.**
//!
//! `frozen_clamp_oracle_parity.rs` proves the frozen oracle equals the production
//! `LegacyRecorded` selector *while that selector still exists*. This file is what remains
//! afterwards: the oracle's outputs recorded as bytes, so the pre-2026-08-25 arithmetic stays
//! re-derivable and re-checkable once `ClampGradientContract` is gone from production entirely.
//!
//! Two directions are covered, and they are different claims:
//!
//! * the **frozen** column pins the retired arithmetic — it must never change;
//! * the **canonical** column pins today's correct arithmetic — if a future change to
//!   `hymeko_spline` moves it, this fixture fails and the change has to be argued for rather than
//!   absorbed.
//!
//! # Re-recording
//!
//! `cargo test -p hymeko_numeric_vv --test frozen_clamp_fixture -- --ignored record_` rewrites the
//! fixture. That is a deliberate two-step: a fixture that blesses itself on mismatch is not a
//! fixture. Re-recording the **frozen** column is a statement that the historical capture was
//! wrong, and needs a report entry saying why.

mod support;

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, kb_backward, kb_forward};
use std::path::PathBuf;
use support::frozen_clamp_oracle as frozen;

const N: usize = 7;
const CH: usize = 3;
const GRID: usize = 5;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/frozen_clamp_grad_x.txt")
}

fn coef() -> Vec<f32> {
    (0..CH * GRID)
        .map(|i| 0.35 * (i as f32) - 0.9 + 0.2 * ((i % 3) as f32))
        .collect()
}

fn tcb_raw() -> Vec<f32> {
    (0..CH * GRID * 3)
        .map(|i| 0.11 * (i as f32 % 7.0) - 0.4)
        .collect()
}

fn grad_y() -> Vec<f32> {
    (0..N * CH)
        .map(|i| 0.5 - 0.13 * (i as f32) + 0.07 * ((i % 5) as f32))
        .collect()
}

/// The four columns the fixture records, each `(label, values)`.
///
/// Built from the **public** entry points only — `catmull_rom_forward`/`_backward`,
/// `kb_forward`/`_backward` and the frozen oracle — so this function keeps working after Phase H
/// removes the contract selector.
fn columns() -> Vec<(&'static str, Vec<f32>)> {
    let (c, t, gy) = (coef(), tcb_raw(), grad_y());
    let x = frozen::saturating_probe(N, CH);

    let (_, cr_cache) = catmull_rom_forward(&c, &x, N, CH, &hymeko_spline::Knots::uniform(GRID));
    let cr_canonical = catmull_rom_backward(&c, &cr_cache, &gy).grad_x;

    let (_, kb_cache) = kb_forward(&c, &t, &x, N, CH, GRID);
    let kb_canonical = kb_backward(&c, &t, &kb_cache, &gy).grad_x;

    vec![
        (
            "cr_frozen_legacy",
            frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy),
        ),
        ("cr_canonical", cr_canonical),
        (
            "kb_frozen_legacy",
            frozen::kb_legacy_grad_x(&c, &t, &x, N, CH, GRID, &gy),
        ),
        ("kb_canonical", kb_canonical),
    ]
}

fn render() -> String {
    let mut s = String::new();
    s.push_str(
        "# Frozen spline clamp-gradient fixture.\n\
         # Each column is grad_x, flat (n=7, channels=3), as hex f32 bit patterns.\n\
         # `*_frozen_legacy` is the pre-2026-08-25 arithmetic captured by\n\
         # tests/support/frozen_clamp_oracle.rs; `*_canonical` is today's correct arithmetic.\n\
         # Probe: frozen_clamp_oracle::saturating_probe(7, 3); coef/tcb/grad_y are the\n\
         # deterministic fixtures in tests/frozen_clamp_fixture.rs.\n",
    );
    s.push_str(&format!("shape n={N} channels={CH} grid={GRID}\n"));
    for (label, values) in columns() {
        s.push_str(label);
        for v in values {
            s.push(' ');
            s.push_str(&format!("{:08x}", v.to_bits()));
        }
        s.push('\n');
    }
    s
}

fn parse(text: &str) -> Vec<(String, Vec<u32>)> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("shape") && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let label = it.next().expect("a column label").to_string();
            let vals = it
                .map(|h| u32::from_str_radix(h, 16).expect("hex bit pattern"))
                .collect();
            (label, vals)
        })
        .collect()
}

#[test]
fn the_fixture_matches_the_current_code() {
    let path = fixture_path();
    let recorded = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is missing ({e}); re-record with --ignored record_",
            path.display()
        )
    });
    let expected = parse(&recorded);
    let actual = parse(&render());

    assert_eq!(
        expected.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
        actual.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
        "the fixture's columns and the code's columns disagree"
    );
    for ((label, want), (_, got)) in expected.iter().zip(&actual) {
        assert_eq!(
            want,
            got,
            "column `{label}` drifted from the recorded fixture at {}",
            path.display()
        );
    }
}

#[test]
fn the_fixture_is_discriminating() {
    // A fixture whose frozen and canonical columns are equal would pass forever without proving
    // anything about the defect.
    let cols = columns();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let get = |name: &str| bits(&cols.iter().find(|(l, _)| *l == name).expect(name).1);
    assert_ne!(get("cr_frozen_legacy"), get("cr_canonical"));
    assert_ne!(get("kb_frozen_legacy"), get("kb_canonical"));
}

#[test]
fn the_canonical_columns_are_zero_wherever_the_probe_saturates() {
    let x = frozen::saturating_probe(N, CH);
    let cols = columns();
    for name in ["cr_canonical", "kb_canonical"] {
        let v = &cols.iter().find(|(l, _)| *l == name).expect(name).1;
        for (i, &raw) in x.iter().enumerate() {
            if !(-1.0..=1.0).contains(&raw) {
                assert_eq!(v[i], 0.0, "{name} element {i} (raw {raw})");
            }
        }
    }
}

#[test]
#[ignore = "recorder: rewrites the on-disk fixture, run deliberately"]
fn record_the_fixture() {
    let path = fixture_path();
    std::fs::create_dir_all(path.parent().expect("fixtures dir")).expect("create fixtures dir");
    std::fs::write(&path, render()).expect("write fixture");
    eprintln!("recorded {}", path.display());
}

// ── claims inherited from the retired `frozen_clamp_oracle_parity.rs` (2026-08-29) ────────────
//
// That suite's purpose was capture-before-deletion: it compared the frozen oracle against the
// production `*_under(LegacyRecorded)` selector while that selector still existed. The selector is
// gone, so fidelity-to-the-selector is now **immutable pre-deletion evidence at commit `28d4ee6`**,
// where the suite was green — deliberately not re-asserted at runtime, because the comparand no
// longer exists by design.
//
// What did NOT depend on the selector is inherited here. The frozen oracle uses only plain
// production entry points, so these claims remain executable:

/// The probe must genuinely saturate on both sides and land on both closed endpoints, or every
/// claim built on it is vacuous.
#[test]
fn the_probe_actually_saturates_on_both_sides_and_hits_both_endpoints() {
    let x = frozen::saturating_probe(N, CH);
    assert!(x.iter().any(|&v| v < -1.0), "no low saturation");
    assert!(x.iter().any(|&v| v > 1.0), "no high saturation");
    assert!(x.contains(&-1.0), "closed lower endpoint not probed");
    assert!(x.contains(&1.0), "closed upper endpoint not probed");
    assert!(
        x.iter()
            .any(|&v| (-1.0..=1.0).contains(&v) && v != -1.0 && v != 1.0),
        "no strictly-interior element to hold fixed"
    );
}

/// The frozen and canonical arithmetics differ **exactly** on the saturated elements — nowhere
/// else. Previously expressed as contract-vs-contract; now oracle-vs-canonical, which is the same
/// comparison without the retired selector.
#[test]
fn frozen_and_canonical_differ_exactly_on_the_saturated_elements() {
    let (c, gy) = (coef(), grad_y());
    let x = frozen::saturating_probe(N, CH);
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, &hymeko_spline::Knots::uniform(GRID));
    let canonical = catmull_rom_backward(&c, &cache, &gy).grad_x;
    let frozen_gx = frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy);

    // Exact characterisation, not a floor: bit-identical at every interior coordinate, differing
    // at EVERY saturated one. Counted by `to_bits()`, so `+0.0` vs `-0.0` cannot pass as equal.
    let (mut interior, mut saturated, mut sat_differ) = (0usize, 0usize, 0usize);
    for (i, &raw) in x.iter().enumerate() {
        if (-1.0..=1.0).contains(&raw) {
            interior += 1;
            assert_eq!(
                canonical[i].to_bits(),
                frozen_gx[i].to_bits(),
                "element {i} (raw {raw}) is interior; the two must agree bit-for-bit"
            );
        } else {
            saturated += 1;
            assert_eq!(
                canonical[i].to_bits(),
                0.0f32.to_bits(),
                "element {i} (raw {raw}): canonical must be exactly +0.0"
            );
            if canonical[i].to_bits() != frozen_gx[i].to_bits() {
                sat_differ += 1;
            }
        }
    }
    assert_eq!(x.len(), 21, "probe width");
    assert_eq!(interior, 11, "interior coordinates on this probe");
    assert_eq!(saturated, 10, "saturated coordinates on this probe");
    assert_eq!(
        sat_differ, 10,
        "the two arithmetics must differ at EVERY saturated coordinate, not merely somewhere"
    );
}

/// The same exact characterisation for Kochanek-Bartels.
#[test]
fn frozen_and_canonical_differ_at_every_saturated_kb_coordinate() {
    let (c, t, gy) = (coef(), tcb_raw(), grad_y());
    let x = frozen::saturating_probe(N, CH);
    let (_, cache) = kb_forward(&c, &t, &x, N, CH, GRID);
    let canonical = kb_backward(&c, &t, &cache, &gy).grad_x;
    let frozen_gx = frozen::kb_legacy_grad_x(&c, &t, &x, N, CH, GRID, &gy);

    let mut sat_differ = 0usize;
    for (i, &raw) in x.iter().enumerate() {
        if (-1.0..=1.0).contains(&raw) {
            assert_eq!(
                canonical[i].to_bits(),
                frozen_gx[i].to_bits(),
                "KB interior {i}"
            );
        } else {
            assert_eq!(canonical[i].to_bits(), 0.0f32.to_bits(), "KB saturated {i}");
            if canonical[i].to_bits() != frozen_gx[i].to_bits() {
                sat_differ += 1;
            }
        }
    }
    assert_eq!(
        sat_differ, 10,
        "KB must differ at every saturated coordinate"
    );
}

/// On a wholly interior input the frozen and canonical arithmetics are bit-identical.
#[test]
fn on_a_wholly_interior_input_frozen_and_canonical_agree() {
    let (c, gy) = (coef(), grad_y());
    let x: Vec<f32> = (0..N * CH)
        .map(|i| -0.9 + 1.8 * (i as f32) / ((N * CH - 1) as f32))
        .collect();
    let (_, cache) = catmull_rom_forward(&c, &x, N, CH, &hymeko_spline::Knots::uniform(GRID));
    let canonical = catmull_rom_backward(&c, &cache, &gy).grad_x;
    let frozen_gx = frozen::cr_legacy_grad_x(&c, &x, N, CH, GRID, &gy);
    assert_eq!(
        canonical.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        frozen_gx.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
