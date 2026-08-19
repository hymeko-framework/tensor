//! The `+sse4.1` build contract — the test this crate's header said did not exist.
//!
//! `catmull_rom`'s forward calls `f32::floor`. On baseline `x86-64` that lowers to a **libm
//! call**; with `sse4.1` it is a single `roundss`. `tensor/.cargo/config.toml` sets the target
//! feature, and `src/lib.rs` records the measured cost of losing it: **~10 % on KAN and ~5.6 %
//! on HSiKAN** (3.71 % / 4.36 % of profile samples spent in the libm call).
//!
//! Until 2026-08-19 that config file was **untracked in git** and, in the crate header's own
//! words, *"no test detects it"*. A `git clean` could therefore delete the contract and the only
//! symptom would be that everything got ~10 % slower with every test still green.
//!
//! `target_feature` is a **compile-time** cfg, so this costs nothing at runtime: if the rustflag
//! is absent when the test binary is built, the assertion is compiled in and fails.

/// Fails when the `+sse4.1` rustflag is missing on x86-64.
///
/// If this test fails, `.cargo/config.toml` has been lost or overridden — restore it rather than
/// deleting this test. On non-x86-64 targets the contract does not apply and the test is inert.
#[test]
// `assertions_on_constants`: intentional. `target_feature` is a compile-time cfg, so the assertion
// IS constant — that is what makes this test cost nothing at runtime and catch the missing rustflag
// at build time. Clippy suggests a `const {}` block; deliberately not taken, because that would make
// the crate fail to COMPILE without the flag rather than fail a test, breaking baseline and
// cross-target builds for people who are not chasing the KAN/HSiKAN hot path.
#[allow(clippy::assertions_on_constants)]
fn sse41_target_feature_is_enabled_on_x86_64() {
    #[cfg(target_arch = "x86_64")]
    assert!(
        cfg!(target_feature = "sse4.1"),
        "the `+sse4.1` build contract is NOT active.\n\
         `f32::floor` in catmull_rom's forward is compiling to a libm call instead of `roundss`, \
         which measured ~10% on KAN and ~5.6% on HSiKAN.\n\
         Restore `[target.'cfg(target_arch = \"x86_64\")'] rustflags = [\"-C\", \
         \"target-feature=+sse4.1\"]` in `tensor/.cargo/config.toml`."
    );
}

/// Documents the non-x86-64 case explicitly so the contract's scope is not mistaken for universal.
#[test]
#[allow(clippy::assertions_on_constants)] // same rationale as the sibling test above
fn contract_scope_is_x86_64_only() {
    if cfg!(target_arch = "x86_64") {
        assert!(cfg!(target_feature = "sse4.1"), "see the sibling test");
    }
    // On aarch64 etc. `floor` already lowers to a single instruction; nothing to enforce.
}
