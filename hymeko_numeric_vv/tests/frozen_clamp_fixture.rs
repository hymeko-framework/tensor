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

    let (_, cr_cache) = catmull_rom_forward(&c, &x, N, CH, GRID);
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
