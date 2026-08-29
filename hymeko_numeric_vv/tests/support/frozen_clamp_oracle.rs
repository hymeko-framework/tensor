//! **FROZEN ORACLE — the pre-2026-08-25 spline clamp-gradient arithmetic.**
//!
//! Origin: `hymeko_spline` `catmull_rom.rs` / `kochanek_bartels.rs`, the behaviour selected by
//! `ClampGradientContract::LegacyRecorded`, captured here on 2026-08-29 **before** that selector
//! is removed from production (mission Phase H). Diagnosed in
//! `nagare/reports/2026-08-25-hsikan-grad-x-root-cause-audit.md` and
//! `nagare/reports/2026-08-25-coginfocom-clamp-contract-and-core-blast-radius.md`.
//!
//! # The defect, stated precisely
//!
//! Every spline forward clamps its input to `[-1, 1]`, so outside that interval the output is
//! constant in `x` and the true input derivative is `0`. Until 2026-08-25 the two *cached*
//! backwards guarded that case with `(-1.0..=1.0).contains(&x_clamped)` — a predicate on the
//! **already-clamped** value, which is always true. They therefore returned the interior
//! derivative for saturated elements too, omitting the clamp's own Jacobian.
//!
//! Parameter gradients (`grad_coef`, `grad_tcb`) are **identical under both contracts**: the clamp
//! is on the path to `grad_x` alone. This oracle therefore reproduces `grad_x` only, which is the
//! whole of the discriminating difference.
//!
//! # How this oracle is constructed, and why it is not a transcription
//!
//! The obvious capture is to copy the retired kernel body. That was rejected: the *derivative
//! formula* is not retired — it is live production code — so copying it would duplicate a live
//! owner into a test tree and leave two bodies free to drift (CLAUDE.md §6.1, §6.5 #13). Only the
//! **guard** was retired, and a guard is not something one can transcribe usefully.
//!
//! The construction used instead is an input transformation:
//!
//! > *legacy* `grad_x` for input `x` equals the **canonical, corrected** `grad_x` for input
//! > `clamp(x)`.
//!
//! It holds because `clamp` is idempotent (`clamp(clamp(x)) == clamp(x)`, so the forward sees the
//! same `x_clamped`, the same segment index and the same segment coordinate `t`) while every
//! element of `clamp(x)` lies in the **closed** interval `[-1, 1]` and is therefore recorded as
//! interior — which is exactly what the tautological guard used to assert unconditionally. The
//! corrected backward then propagates the interior derivative for every element, saturated or not.
//!
//! Three properties follow, and all three are why this form was chosen:
//!
//! 1. **No retired arithmetic is copied**, so there is nothing to drift.
//! 2. **It survives Phase H.** It uses only `catmull_rom_forward` / `catmull_rom_backward` /
//!    `kb_forward` / `kb_backward` — the plain entry points that outlive `ClampGradientContract`.
//!    The historical numbers stay re-derivable after the selector is gone, which is the entire
//!    purpose of capturing them.
//! 3. **It is not tautological.** It does not obtain its expected value by calling the composite
//!    under test: while `LegacyRecorded` still exists, `frozen_clamp_oracle_parity.rs` asserts
//!    bit-equality between this construction and the production selector, and after the selector
//!    is gone `frozen_clamp_fixture.rs` pins the oracle to recorded bytes on disk.
//!
//! # Scope
//!
//! Known to reproduce the retired `grad_x` for plain Catmull-Rom, Chebyshev-CR (whose `grad_x`
//! *is* the CR one) and Kochanek-Bartels, at `f32` throughout, on any `(n, channels, grid)` those
//! kernels accept. It is **not** a spline engine and must never be selected as one.

// Each including suite uses a subset.
#![allow(dead_code)]

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, kb_backward, kb_forward};

/// The clamp the spline forwards apply, as its own named step.
///
/// Named rather than inlined because it is the *whole* mechanism of this oracle: pre-applying it
/// is what turns a saturated element into an interior one.
#[must_use]
pub fn pre_saturate(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.clamp(-1.0, 1.0)).collect()
}

/// Retired Catmull-Rom input gradient, flat `(n, channels)`.
///
/// # Preconditions
/// `grid >= 3`; `coef.len() == channels * grid`; `x.len() == grad_y.len() == n * channels`.
///
/// # Postconditions
/// Bit-identical to the pre-2026-08-25 `catmull_rom_backward().grad_x`: the interior derivative
/// at every element, including those the forward saturated.
#[must_use]
pub fn cr_legacy_grad_x(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
    grad_y: &[f32],
) -> Vec<f32> {
    let (_, cache) = catmull_rom_forward(coef, &pre_saturate(x), n, channels, grid);
    catmull_rom_backward(coef, &cache, grad_y).grad_x
}

/// Retired Kochanek-Bartels input gradient, flat `(n, channels)`.
///
/// # Preconditions
/// `grid >= 4`; `coef.len() == channels * grid`; `tcb_raw.len() == channels * grid * 3`;
/// `x.len() == grad_y.len() == n * channels`.
///
/// # Postconditions
/// Bit-identical to the pre-2026-08-25 `kb_backward().grad_x`.
#[must_use]
pub fn kb_legacy_grad_x(
    coef: &[f32],
    tcb_raw: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
    grad_y: &[f32],
) -> Vec<f32> {
    let (_, cache) = kb_forward(coef, tcb_raw, &pre_saturate(x), n, channels, grid);
    kb_backward(coef, tcb_raw, &cache, grad_y).grad_x
}

/// A saturating probe: a deterministic `(n, channels)` input grid that spans well outside
/// `[-1, 1]` on both sides and lands exactly on both endpoints.
///
/// Deterministic by construction rather than seeded, so the fixture it feeds is reproducible from
/// this source alone.
///
/// # Postconditions
/// `result.len() == n * channels`; contains at least one element `< -1`, one `> 1`, and both
/// `-1.0` and `1.0` exactly.
#[must_use]
pub fn saturating_probe(n: usize, channels: usize) -> Vec<f32> {
    assert!(n * channels >= 8, "the probe needs room for its endpoints");
    let len = n * channels;
    let mut x: Vec<f32> = (0..len)
        .map(|i| {
            // Sweep -2.4 .. +2.4: roughly a third of the elements saturate on each side.
            -2.4 + 4.8 * (i as f32) / ((len - 1) as f32)
        })
        .collect();
    // Pin the closed-interval endpoints and two hard saturations, so the fixture always exercises
    // the boundary cases regardless of `len`.
    x[0] = -3.0;
    x[1] = -1.0;
    x[len - 2] = 1.0;
    x[len - 1] = 3.0;
    x
}
