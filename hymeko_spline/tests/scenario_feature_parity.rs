//! Scenario: `parallel` changes **who does the work, never what is computed**.
//!
//! `src/lib.rs` claims: *"Results are byte-identical with `parallel` on or off, and across 1, 8
//! or 16 threads."* The inline tests cover the **second half** (thread counts, within a parallel
//! build) but cannot cover the first — a Cargo feature is compile-time, so one binary cannot run
//! both paths.
//!
//! This closes it with **golden bit patterns**. The same values must be produced whether the crate
//! is built with `--features parallel` or without, so running this suite in both configurations
//! enforces cross-build parity. `f32::to_bits` is used rather than an epsilon: the claim is
//! byte-identity, and an approximate assertion would pass on exactly the drift it exists to catch.
//!
//! If this fails after a kernel change, do not update the goldens reflexively — first establish
//! which path moved and whether the sequential/parallel decomposition still matches.

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward};

/// Deterministic input generator, shared with the inline suite's approach (no `rand` dependency —
/// this crate is dependency-free with `parallel` off and a test-only dep would undo that).
fn pseudo_input(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// FNV-1a over the raw bits — one number that changes if any element does.
fn bits_hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |acc, x| {
        (acc ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
    })
}

/// `n = 4099` is deliberately prime and above `PAR_MIN_ELEMS`, so the parallel path is taken and
/// the final block is ragged — the case where a decomposition bug would show.
const N: usize = 4099;
const CHANNELS: usize = 8;
const GRID: usize = 6;

#[test]
fn forward_bits_are_identical_across_feature_configurations() {
    let x = pseudo_input(N * CHANNELS, 6);
    let coef = pseudo_input(CHANNELS * GRID, 7);
    let (y, _) = catmull_rom_forward(&coef, &x, N, CHANNELS, &hymeko_spline::Knots::uniform(GRID));
    assert_eq!(
        bits_hash(&y),
        GOLDEN_FORWARD,
        "forward output bits differ from the golden value. Either a kernel changed, or the \
         `parallel` path no longer matches the sequential one — which would break the crate's \
         stated byte-identity contract."
    );
}

#[test]
fn backward_bits_are_identical_across_feature_configurations() {
    let x = pseudo_input(N * CHANNELS, 6);
    let coef = pseudo_input(CHANNELS * GRID, 7);
    let grad_y = pseudo_input(N * CHANNELS, 8);
    let (_, cache) =
        catmull_rom_forward(&coef, &x, N, CHANNELS, &hymeko_spline::Knots::uniform(GRID));
    let bwd = catmull_rom_backward(&coef, &cache, &grad_y);
    assert_eq!(
        bits_hash(&bwd.grad_coef),
        GOLDEN_GRAD_COEF,
        "grad_coef bits differ"
    );
    assert_eq!(bits_hash(&bwd.grad_x), GOLDEN_GRAD_X, "grad_x bits differ");
}

// Goldens captured 2026-08-19 on x86-64 with `+sse4.1`, verified identical with and without
// `--features parallel`. They are a property of the arithmetic, not of the machine's core count.
// The coefficient gradient re-pinned 2026-10-09 (user decision "AVX2 és CR újrapinnelés"): its
// terms are now summed per segment and control slot before they reach a coefficient, so the order
// and the last bits moved; the same with and without `--features parallel`.
// All three re-pinned 2026-10-23 (user decision "Do it"; Nagare plan 2026-10-23-incidence-partition,
// Amendment 7): the power form evaluates each leaf's cubic by Horner from coefficients computed once
// per call and folds per-leaf moments through the leaf basis, so the forward, the input gradient and
// the coefficient gradient each round differently; the same with and without `--features parallel`.
// The coefficient gradient re-pinned 2026-10-24 (user approval "Approved"; Nagare plan
// 2026-10-24-computed-parameters-once, Amendment 1): the blocks' moments are added in `f64`, then folded once.
const GOLDEN_FORWARD: u64 = 11120622793391595860;
const GOLDEN_GRAD_COEF: u64 = 1360101236142475722;
const GOLDEN_GRAD_X: u64 = 11327949383195490707;
