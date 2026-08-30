//! The architectural boundary, asserted rather than intended.
//!
//! Two claims, both mechanical:
//!
//! * **Nothing in the `tensor` workspace depends on `hymeko_numeric_vv`** — not in
//!   `[dependencies]`, not in `[dev-dependencies]`, not in `[build-dependencies]`. A frozen oracle
//!   that a production build graph can reach is not frozen.
//! * **No frozen oracle lives in a library target.** The oracle sources are under `tests/`, so
//!   they are compiled only into this package's own test binaries.
//!
//! If a later slice needs a V&V helper in production, the answer is to design a production owner
//! for it, not to reverse this edge.

use hymeko_numeric_vv::manifest_gate;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate directory has a parent")
        .to_path_buf()
}

#[test]
fn no_production_crate_depends_on_the_v_and_v_package() {
    manifest_gate::assert_v_and_v_is_downstream_only(&workspace_root(), "hymeko_numeric_vv");
}

#[test]
fn the_gate_actually_saw_the_workspace() {
    // Guard against the vacuous pass: a root that stopped resolving would make every absence
    // claim above trivially true.
    let manifests = manifest_gate::workspace_manifests(&workspace_root());
    let names: Vec<String> = manifests.iter().filter_map(|m| m.package_name()).collect();
    for expected in [
        "hymeko_clifford",
        // `hymeko_graph` is deliberately absent: it was retired from this workspace on
        // 2026-08-30 in favour of the canonical owner in the `hymeko/graph` repository. Leaving
        // it here would make this anti-vacuity guard fail for the right reason but the wrong
        // cause, and removing the guard entirely would let a genuinely empty scan pass.
        "hymeko_spline",
        "hymeko_tensor",
        "hymeko_vision",
        "hymeko_numeric_vv",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "manifest scan missed `{expected}`; found {names:?}"
        );
    }
}

#[test]
fn the_v_and_v_package_is_a_workspace_member() {
    let root = workspace_root().join("Cargo.toml");
    let text = std::fs::read_to_string(&root).expect("workspace manifest");
    assert!(
        text.contains("\"hymeko_numeric_vv\""),
        "{} does not list hymeko_numeric_vv as a member; its tests would never run under the \
         canonical gate",
        root.display()
    );
}

#[test]
fn frozen_oracles_live_only_under_tests() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = crate_dir.join("src");
    let scan = hymeko_numeric_vv::source_gate::ProductionScan::new([src]);
    scan.assert_scanned_at_least(5);
    // The oracle names the retired selector; the library must not.
    scan.assert_absent("LegacyRecorded");
    scan.assert_absent("frozen");

    let oracle = crate_dir.join("tests/support/frozen_clamp_oracle.rs");
    assert!(oracle.is_file(), "{} is missing", oracle.display());
}
