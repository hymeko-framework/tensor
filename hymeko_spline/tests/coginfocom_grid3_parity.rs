//! Grid-3 Catmull–Rom parity against the recovered CogInfoCom PyTorch evaluator.
//!
//! The paper's HSiKAN configures its KAN activations with **three** control points. This suite
//! proves that `hymeko_spline`'s canonical operator reproduces the recovered
//! `hymeko_neuro.hyperedge.splines._catmull_rom_eval` at that grid — forward, input gradient and
//! coefficient gradient — under the canonical clamp Jacobian, which is the arithmetic the
//! recovered evaluator implements (it clamps `x` with `torch.clamp` and lets autograd carry the
//! zero derivative outside `[-1, 1]`).
//!
//! # Evidence, not regeneration
//! `tests/fixtures/coginfocom_catmull_rom_grid3.json` is **immutable evidence**, produced once by
//! `tools/gen_coginfocom_grid3_fixture.py` against the recovered source
//! (SHA-256 `8c03a1e5…3553a`) and byte-identical across two independent runs. No Python or torch
//! code is imported or executed by this test, by Cargo, or by any production path.
//!
//! The fixture is parsed here by a purpose-built scanner rather than a JSON crate: `hymeko_spline`
//! has no dev-dependencies, and adding one would be a core dependency change.

use hymeko_spline::catmull_rom::{catmull_rom_backward, catmull_rom_forward};

const FIXTURE: &str = include_str!("fixtures/coginfocom_catmull_rom_grid3.json");

/// One recorded PyTorch evaluation: inputs, outputs and both autograd gradients.
struct Case {
    name: String,
    grid: usize,
    channels: usize,
    batch: usize,
    coef: Vec<f32>,
    x: Vec<f32>,
    grad_y: Vec<f32>,
    y: Vec<f32>,
    grad_x: Vec<f32>,
    grad_coef: Vec<f32>,
}

impl Case {
    fn coords(&self) -> usize {
        self.batch * self.channels
    }
}

/// Minimal fixture scanner.
///
/// The generator writes `json.dump(..., indent=1, sort_keys=True)`, so within every case object the
/// keys appear in a fixed alphabetical order and every value is either a number, a string, or a flat
/// array of numbers. `"batch"` sorts first, so each of its occurrences starts one record.
mod scan {
    pub fn number_after(hay: &str, key: &str, from: usize) -> Option<(f64, usize)> {
        let pat = format!("\"{key}\":");
        let at = hay[from..].find(&pat)? + from + pat.len();
        let rest = &hay[at..];
        let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
        rest[..end]
            .trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v, at + end))
    }

    pub fn string_after(hay: &str, key: &str, from: usize) -> Option<(String, usize)> {
        let pat = format!("\"{key}\":");
        let at = hay[from..].find(&pat)? + from + pat.len();
        let open = hay[at..].find('"')? + at + 1;
        let close = hay[open..].find('"')? + open;
        Some((hay[open..close].to_string(), close))
    }

    pub fn array_after(hay: &str, key: &str, from: usize) -> Option<(Vec<f32>, usize)> {
        let pat = format!("\"{key}\":");
        let at = hay[from..].find(&pat)? + from + pat.len();
        let open = hay[at..].find('[')? + at;
        let close = hay[open..].find(']')? + open;
        let vals = hay[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f32>()
                    .expect("fixture array holds finite decimals")
            })
            .collect();
        Some((vals, close))
    }
}

fn cases() -> Vec<Case> {
    let start = FIXTURE
        .find("\"cases\":")
        .expect("fixture has a cases array");
    let mut out = Vec::new();
    let mut cursor = start;
    while let Some((batch, next)) = scan::number_after(FIXTURE, "batch", cursor) {
        let (channels, _) = scan::number_after(FIXTURE, "channels", next).expect("channels");
        let (coef, _) = scan::array_after(FIXTURE, "coef", next).expect("coef");
        let (grad_coef, _) = scan::array_after(FIXTURE, "grad_coef", next).expect("grad_coef");
        let (grad_x, _) = scan::array_after(FIXTURE, "grad_x", next).expect("grad_x");
        let (grad_y, _) = scan::array_after(FIXTURE, "grad_y", next).expect("grad_y");
        let (grid, _) = scan::number_after(FIXTURE, "grid", next).expect("grid");
        let (name, _) = scan::string_after(FIXTURE, "name", next).expect("name");
        let (x, _) = scan::array_after(FIXTURE, "x", next).expect("x");
        let (y, after_y) = scan::array_after(FIXTURE, "y", next).expect("y");
        out.push(Case {
            name,
            grid: grid as usize,
            channels: channels as usize,
            batch: batch as usize,
            coef,
            x,
            grad_y,
            y,
            grad_x,
            grad_coef,
        });
        cursor = after_y;
    }
    out
}

/// Machine-asserted provenance.
///
/// The fixture is immutable evidence, so its lineage is checked by the test suite rather than left
/// to prose. Everything here is read out of the committed file itself — no path under `/mnt/hymeko`
/// is opened during `cargo test`.
///
/// The generator's own SHA recorded inside the fixture is the byte-image that produced these
/// numbers. The committed generator was afterwards corrected for a stale usage line naming the old
/// filename; re-running it reproduces every numeric field **byte-identically** (only `generated_by`
/// and `generator_sha256` move), which is why the numeric evidence was not rewritten.
#[test]
fn the_fixture_carries_the_expected_provenance() {
    const RECOVERED_SHA: &str = "8c03a1e5298e23007e7979921b891605f2787b6c327cec8ab7d8b3b94973553a";
    const GENERATOR_SHA_AT_GENERATION: &str =
        "9ef5ff2eadea535569931dae2e2278a4a94eaf2fa8eb4a62d38bf4855f584dd8";
    const ORACLE_TORCH: &str = "2.13.0+cu130";

    for (key, want) in [
        ("recovered_source_sha256", RECOVERED_SHA),
        ("generator_sha256", GENERATOR_SHA_AT_GENERATION),
    ] {
        let (got, _) = scan::string_after(FIXTURE, key, 0).expect(key);
        assert_eq!(got, want, "fixture provenance field {key}");
    }
    let (symbol, _) = scan::string_after(FIXTURE, "recovered_symbol", 0).expect("symbol");
    assert_eq!(symbol, "hymeko_neuro.hyperedge.splines._catmull_rom_eval");
    let (torch, _) = scan::string_after(FIXTURE, "torch", 0).expect("torch");
    assert_eq!(
        torch, ORACLE_TORCH,
        "the oracle ran under a different torch than the paper's 2.11.0+cu130; that gap is \
         recorded provenance, not an equivalence claim"
    );
    let (dtype, _) = scan::string_after(FIXTURE, "dtype", 0).expect("dtype");
    assert_eq!(dtype, "float32");
    let (compile, _) = scan::string_after(FIXTURE, "torch_compile", 0).expect("compile");
    assert!(compile.contains("disabled"), "eager evaluation required");
}

/// The fixture must actually carry what the suite claims to test.
#[test]
fn the_fixture_is_well_formed_and_discriminating() {
    let cs = cases();
    assert_eq!(cs.len(), 5, "expected five recorded cases");
    let total: usize = cs.iter().map(Case::coords).sum();
    assert_eq!(total, 105, "expected 105 recorded coordinates");

    for c in &cs {
        assert_eq!(c.grid, 3, "{}: fixture must be grid 3", c.name);
        assert_eq!(c.coef.len(), c.channels * c.grid, "{}: coef shape", c.name);
        assert_eq!(c.x.len(), c.coords(), "{}: x shape", c.name);
        assert_eq!(c.y.len(), c.coords(), "{}: y shape", c.name);
        assert_eq!(c.grad_x.len(), c.coords(), "{}: grad_x shape", c.name);
        assert_eq!(c.grad_y.len(), c.coords(), "{}: grad_y shape", c.name);
        assert_eq!(
            c.grad_coef.len(),
            c.coef.len(),
            "{}: grad_coef shape",
            c.name
        );
    }

    // Both channel counts, both batch regimes.
    assert!(cs.iter().any(|c| c.channels == 1));
    assert!(cs.iter().any(|c| c.channels > 1));
    assert!(cs.iter().any(|c| c.batch == 1));
    assert!(cs.iter().any(|c| c.batch > 1));

    let all_x: Vec<f32> = cs.iter().flat_map(|c| c.x.iter().copied()).collect();
    assert!(all_x.contains(&-1.0), "left endpoint present");
    assert!(all_x.contains(&1.0), "right endpoint present");
    assert!(all_x.iter().any(|&v| v < -1.0), "saturated below present");
    assert!(all_x.iter().any(|&v| v > 1.0), "saturated above present");
    // Signed zero: both bit patterns must appear, and they are not equal as bits.
    assert!(all_x.iter().any(|&v| v.to_bits() == 0.0f32.to_bits()));
    assert!(all_x.iter().any(|&v| v.to_bits() == (-0.0f32).to_bits()));
    // Both segments of the grid-3 stencil.
    assert!(all_x.iter().any(|&v| (-1.0..0.0).contains(&v)), "segment 0");
    assert!(all_x.iter().any(|&v| (0.0..=1.0).contains(&v)), "segment 1");
    // Non-symmetric coefficients: a reversed control order must be observable.
    let c0 = &cs[0].coef;
    assert!(
        c0.iter().zip(c0.iter().rev()).any(|(a, b)| a != b),
        "coefficients must not be palindromic"
    );
    // Non-constant upstream gradient.
    let g = &cs[1].grad_y;
    assert!(g.iter().any(|v| *v != g[0]), "grad_y must not be constant");
}

/// Forward, `grad_x` and `grad_coef` against the recorded PyTorch values, every coordinate.
#[test]
fn grid3_matches_the_recovered_pytorch_evaluator() {
    let cs = cases();
    let mut checked_out = 0usize;
    let mut checked_gx = 0usize;
    let mut checked_gc = 0usize;
    let mut worst_out = 0i64;
    let mut worst_gx = 0.0f64;
    let mut worst_gc = 0.0f64;

    for c in &cs {
        let (out, cache) = catmull_rom_forward(
            &c.coef,
            &c.x,
            c.batch,
            c.channels,
            &hymeko_spline::Knots::uniform(c.grid),
        );
        let back = catmull_rom_backward(&c.coef, &cache, &c.grad_y);

        // Forward: within FORWARD_ULP_BOUND (bit identity until the power form, 2026-10-23).
        for (k, (&got, &want)) in out.iter().zip(&c.y).enumerate() {
            let u = ulps(got, want);
            worst_out = worst_out.max(u);
            assert!(
                u <= FORWARD_ULP_BOUND,
                "{}: forward lane {k} — {u} ulp (rust {got:e} vs torch {want:e})",
                c.name
            );
            checked_out += 1;
        }
        for (k, (&got, &want)) in back.grad_x.iter().zip(&c.grad_x).enumerate() {
            let u = ulps(got, want);
            worst_gx = worst_gx.max(u as f64);
            assert!(
                u <= GRAD_ULP_BOUND,
                "{}: grad_x lane {k} — {u} ulp (rust {got:e} vs torch {want:e})",
                c.name
            );
            checked_gx += 1;
        }
        for (k, (&got, &want)) in back.grad_coef.iter().zip(&c.grad_coef).enumerate() {
            let u = ulps(got, want);
            worst_gc = worst_gc.max(u as f64);
            assert!(
                u <= GRAD_ULP_BOUND,
                "{}: grad_coef lane {k} — {u} ulp (rust {got:e} vs torch {want:e})",
                c.name
            );
            checked_gc += 1;
        }
    }

    // Asserted counts: no group can be silently skipped.
    assert_eq!(checked_out, 105, "forward coordinates compared");
    assert_eq!(checked_gx, 105, "grad_x coordinates compared");
    assert_eq!(checked_gc, 36, "grad_coef coordinates compared");
    println!(
        "grid3 parity: worst forward={worst_out} ulp, grad_x={worst_gx} ulp, grad_coef={worst_gc} ulp"
    );
}

/// The `Corrected` contract, read straight off the recorded autograd values: every saturated raw
/// input has an **exactly zero** input gradient, and no interior one does by accident.
#[test]
fn saturated_inputs_have_exactly_zero_corrected_input_gradient() {
    let cs = cases();
    let mut saturated = 0usize;
    for c in &cs {
        let (_, cache) = catmull_rom_forward(
            &c.coef,
            &c.x,
            c.batch,
            c.channels,
            &hymeko_spline::Knots::uniform(c.grid),
        );
        let back = catmull_rom_backward(&c.coef, &cache, &c.grad_y);
        for (k, &raw) in c.x.iter().enumerate() {
            if !(-1.0..=1.0).contains(&raw) {
                assert_eq!(
                    back.grad_x[k].to_bits(),
                    0.0f32.to_bits(),
                    "{}: saturated lane {k} (x={raw}) must have zero corrected grad_x",
                    c.name
                );
                assert_eq!(c.grad_x[k], 0.0, "{}: torch agrees at lane {k}", c.name);
                saturated += 1;
            }
        }
    }
    assert_eq!(saturated, 24, "expected 24 saturated coordinates");
}

/// Endpoint and knot behaviour: `x = ±1` are interior under the closed-interval contract, so their
/// input gradient is live, and the interior knot `x = 0` is reached from both sides.
#[test]
fn endpoints_are_interior_and_the_knot_is_approached_from_both_sides() {
    let cs = cases();
    let mut endpoints = 0usize;
    for c in &cs {
        let (_, cache) = catmull_rom_forward(
            &c.coef,
            &c.x,
            c.batch,
            c.channels,
            &hymeko_spline::Knots::uniform(c.grid),
        );
        let back = catmull_rom_backward(&c.coef, &cache, &c.grad_y);
        for (k, &raw) in c.x.iter().enumerate() {
            if raw == -1.0 || raw == 1.0 {
                assert!(
                    ulps(back.grad_x[k], c.grad_x[k]) <= GRAD_ULP_BOUND,
                    "{}: endpoint lane {k}",
                    c.name
                );
                endpoints += 1;
            }
        }
    }
    assert!(endpoints >= 2, "both endpoints must be exercised");
}

/// Monotonic IEEE-754 ULP distance.
///
/// `f32::to_bits` yields a `u32`, so the sign bit lands in the high bit of a *non-negative* integer
/// and the raw bit pattern is **not** monotonic across zero. The bits are therefore reinterpreted as
/// `i32` first — which makes every negative float negative — and the negative half is reflected about
/// `i32::MIN`, giving a total order in which `-0.0` and `+0.0` share key `0` and adjacent floats are
/// one apart in both directions.
fn ulps(a: f32, b: f32) -> i64 {
    let key = |v: f32| -> i64 {
        let bits = v.to_bits() as i32;
        if bits < 0 {
            i64::from(i32::MIN) - i64::from(bits)
        } else {
            i64::from(bits)
        }
    };
    (key(a) - key(b)).abs()
}

#[test]
fn the_ulp_metric_is_monotonic_across_zero() {
    // The defect this replaces: with `to_bits() as i64` the negative branch was unreachable, so
    // negative floats compared as huge positives and every negative-lane distance was wrong.
    assert_eq!(ulps(0.0, -0.0), 0, "signed zeros are the same point");
    assert_eq!(ulps(1.0, f32::from_bits(1.0f32.to_bits() + 1)), 1);
    assert_eq!(ulps(-1.0, f32::from_bits((-1.0f32).to_bits() + 1)), 1);
    assert_eq!(ulps(-1.0, 1.0), ulps(1.0, -1.0), "symmetric");
    // Straddling zero: the two smallest subnormals either side are two steps apart.
    let tiny = f32::from_bits(1);
    assert_eq!(ulps(-tiny, tiny), 2);
    // A negative pair must not report a huge distance.
    assert!(
        ulps(-2.5648756, -2.5648792) < 100,
        "adjacent negatives are close"
    );
}

/// Gradient agreement bound, **derived from the measurement below, not chosen in advance**.
///
/// The gradients do not agree bit for bit, and the reason is structural rather than sloppy: the
/// Catmull–Rom derivative weights sum to zero (the spline reproduces constants exactly), so `grad_x`
/// is a *cancelling* four-term sum. Cancellation amplifies the relative round-off of the summands,
/// and PyTorch reaches the same quantity through a different operation graph — autograd through
/// `clamp`, `floor`, `sub` and four separate gathers — than the closed-form derivative Rust
/// evaluates. Measured worst case: **15 ulp** on `grad_x`, **2 ulp** on `grad_coef` (2026-08); with the power form
/// (2026-10-23) **14 ulp** and **8 ulp**.
///
/// The bound is the next power of two above the measured worst case. `measure_error_distribution`
/// reports the live distribution, and `a_materially_wrong_implementation_is_rejected` shows the
/// bound is far too tight to admit a wrong implementation.
const GRAD_ULP_BOUND: i64 = 16;

/// Forward agreement bound. Until 2026-10-23 the forward agreed **bit for bit** on all 105 coordinates and no
/// tolerance was granted. The power form (user decision "Do it", Nagare plan 2026-10-23-incidence-partition,
/// Amendment 7; the 10-21 rule that bit identity is a means, judged by tolerance where it blocks a structural speed-up)
/// evaluates each leaf's cubic by Horner from `a = Bᵀ q`, which rounds differently from the four-term weighted sum
/// torch evaluates. Measured: 64 of 105 coordinates still bit-identical, worst **7 ulp** (absolute 4.8e-7). The bound
/// is the next power of two above it; the controls below break by orders of magnitude more.
const FORWARD_ULP_BOUND: i64 = 8;

#[test]
fn measure_error_distribution() {
    for (label, sel) in [("forward", 0usize), ("grad_x", 1), ("grad_coef", 2)] {
        let mut exact = 0usize;
        let mut tot = 0usize;
        let mut worst_ulp = 0i64;
        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        let mut opposite = 0usize;
        for c in cases() {
            let (out, cache) = catmull_rom_forward(
                &c.coef,
                &c.x,
                c.batch,
                c.channels,
                &hymeko_spline::Knots::uniform(c.grid),
            );
            let back = catmull_rom_backward(&c.coef, &cache, &c.grad_y);
            let (g, w): (Vec<f32>, Vec<f32>) = match sel {
                0 => (out, c.y.clone()),
                1 => (back.grad_x, c.grad_x.clone()),
                _ => (back.grad_coef, c.grad_coef.clone()),
            };
            for (&a, &b) in g.iter().zip(&w) {
                tot += 1;
                if a.to_bits() == b.to_bits() {
                    exact += 1;
                    continue;
                }
                if a.is_sign_negative() != b.is_sign_negative() {
                    opposite += 1;
                }
                let d = (f64::from(a) - f64::from(b)).abs();
                let sc = f64::from(a).abs().max(f64::from(b).abs());
                worst_ulp = worst_ulp.max(ulps(a, b));
                worst_abs = worst_abs.max(d);
                if sc > 0.0 {
                    worst_rel = worst_rel.max(d / sc);
                }
            }
        }
        println!(
            "MEASURE {label}: exact={exact}/{tot} worst_ulp={worst_ulp} \
             worst_abs={worst_abs:.4e} worst_rel={worst_rel:.4e} opposite_sign_pairs={opposite}"
        );
        // The corrected ULP metric left the parity numbers unchanged. This is why: every compared
        // pair shares a sign, and for same-sign pairs the old (broken) key differed from the correct
        // one by a constant that cancelled in the subtraction. The bug was real, but it could only
        // corrupt opposite-sign comparisons — which the controls do contain and the parity data does
        // not. Measured, not argued.
        assert_eq!(
            opposite, 0,
            "{label}: a sign-straddling pair would have been mismeasured by the old ULP metric"
        );
    }
}

// ---------------------------------------------------------------------------
// Non-vacuity controls: the bound above must reject anything materially wrong
// ---------------------------------------------------------------------------

/// How the stencil obtains control points outside `[0, grid-1]`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Padding {
    /// What the production operator and the recovered PyTorch both do.
    Replicate,
    /// A plausible-looking alternative that must NOT reproduce the fixture.
    Wrap,
}

/// Independent Catmull–Rom reference, written here from the closed form so the controls below
/// compare the operator against something other than itself.
fn reference_cr(coef: &[f32], x: f32, grid: usize, padding: Padding, reversed: bool) -> f32 {
    let xc = x.clamp(-1.0, 1.0);
    let u = (xc + 1.0) * 0.5 * (grid - 1) as f32;
    let i = (u.floor() as usize).min(grid - 2);
    let t = u - i as f32;
    let (t2, t3) = (t * t, t * t * t);
    let w = [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ];
    let g = grid as isize;
    let pick = |off: isize| -> usize {
        let raw = i as isize + off;
        match padding {
            Padding::Replicate => raw.clamp(0, g - 1) as usize,
            Padding::Wrap => raw.rem_euclid(g) as usize,
        }
    };
    let mut p = [pick(-1), pick(0), pick(1), pick(2)];
    if reversed {
        p.reverse();
    }
    (0..4).map(|k| w[k] * coef[p[k]]).sum()
}

/// The reference itself must be right, or the controls below prove nothing: it reproduces the operator within the
/// forward bound (bit for bit until the power form, 2026-10-23).
#[test]
fn the_reference_reproduces_the_operator_under_replication() {
    for c in cases() {
        let (out, _) = catmull_rom_forward(
            &c.coef,
            &c.x,
            c.batch,
            c.channels,
            &hymeko_spline::Knots::uniform(c.grid),
        );
        for row in 0..c.batch {
            for ch in 0..c.channels {
                let k = row * c.channels + ch;
                let got = reference_cr(
                    &c.coef[ch * c.grid..(ch + 1) * c.grid],
                    c.x[k],
                    c.grid,
                    Padding::Replicate,
                    false,
                );
                assert!(
                    ulps(got, out[k]) <= FORWARD_ULP_BOUND,
                    "{}: lane {k}: {got:e} vs {:e}",
                    c.name,
                    out[k]
                );
            }
        }
    }
}

/// Reversed control order and a different padding rule must both move the output far outside the
/// parity bound — otherwise the fixture could not discriminate an implementation error.
#[test]
fn a_materially_wrong_implementation_is_rejected() {
    let mut reversed_bad = 0usize;
    let mut wrap_bad = 0usize;
    let mut worst_reversed = 0i64;
    let mut worst_wrap = 0i64;

    for c in cases() {
        for row in 0..c.batch {
            for ch in 0..c.channels {
                let k = row * c.channels + ch;
                let slice = &c.coef[ch * c.grid..(ch + 1) * c.grid];
                let want = c.y[k];

                let rev = reference_cr(slice, c.x[k], c.grid, Padding::Replicate, true);
                let u = ulps(rev, want);
                if u > GRAD_ULP_BOUND {
                    reversed_bad += 1;
                    worst_reversed = worst_reversed.max(u);
                }

                let wrap = reference_cr(slice, c.x[k], c.grid, Padding::Wrap, false);
                let u = ulps(wrap, want);
                if u > GRAD_ULP_BOUND {
                    wrap_bad += 1;
                    worst_wrap = worst_wrap.max(u);
                }
            }
        }
    }

    assert!(
        reversed_bad > 50,
        "reversing the control order must break most coordinates, broke {reversed_bad}"
    );
    assert!(
        wrap_bad > 20,
        "wrap padding must break many coordinates, broke {wrap_bad}"
    );
    println!(
        "control: reversed order breaks {reversed_bad}/105 (worst {worst_reversed} ulp); \
         wrap padding breaks {wrap_bad}/105 (worst {worst_wrap} ulp)"
    );
}

// ── RETIRED with the clamp selector (2026-08-29) ──────────────────────────────────────────────
//
// `the_legacy_contract_remains_selectable_and_differs_on_saturated_inputs` compared the two clamp
// arms on this fixture and established that the retired arithmetic propagates a gradient at
// saturated lanes where the recovered PyTorch evaluator returns exactly zero, while agreeing
// bit-for-bit at interior lanes.
//
// With one arithmetic left it cannot run. Named owners for its three claims:
//
//   retired-vs-canonical difference at saturated lanes, equality at interior lanes
//     -> hymeko_numeric_vv::frozen_clamp_fixture::frozen_and_canonical_differ_exactly_on_the_saturated_elements
//     -> hymeko_numeric_vv::frozen_clamp_fixture::on_a_wholly_interior_input_frozen_and_canonical_agree
//
//   grad_coef unaffected by the clamp
//     -> hymeko_spline::clamp_gradient_contract::parameter_gradients_match_finite_differences
//        (a stronger claim: the shipped parameter gradients are the TRUE ones, not merely
//         unchanged between two arms); the historical "identical under both contracts" statement
//         is immutable pre-deletion evidence at commit 28d4ee6.
//
//   canonical arm matching the recovered evaluator at saturated lanes
//     -> the surviving grad_x assertions in this file, which compare the shipped kernel against
//        c.grad_x directly.

/// Perturbing one control point must move both its forward value and its own gradient.
#[test]
fn perturbing_a_single_coefficient_moves_forward_and_its_gradient() {
    let c = &cases()[4];
    let (base_out, base_cache) = catmull_rom_forward(
        &c.coef,
        &c.x,
        c.batch,
        c.channels,
        &hymeko_spline::Knots::uniform(c.grid),
    );
    let base = catmull_rom_backward(&c.coef, &base_cache, &c.grad_y);

    let mut bumped = c.coef.clone();
    let target = 1usize; // interior control point of channel 0
    bumped[target] += 0.5;
    let (out, cache) = catmull_rom_forward(
        &bumped,
        &c.x,
        c.batch,
        c.channels,
        &hymeko_spline::Knots::uniform(c.grid),
    );
    let back = catmull_rom_backward(&bumped, &cache, &c.grad_y);

    assert!(
        out.iter()
            .zip(&base_out)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "a bumped control point must change the forward output"
    );
    // The coefficient gradient is linear in the weights, so it is unchanged; the INPUT gradient is
    // what carries the perturbation, because it multiplies the control values.
    assert!(
        back.grad_x
            .iter()
            .zip(&base.grad_x)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "a bumped control point must change the measured input gradient"
    );
}

/// Grid 2 remains **out of contract**. This asserts the supported minimum is 3 — the configuration
/// this study requires — and makes no claim that grid 2 is mathematically impossible.
#[test]
#[should_panic(expected = "Catmull--Rom needs at least 3 control points")]
fn grid_two_is_still_rejected() {
    let coef = [0.5f32, -0.25];
    let x = [0.3f32];
    let _ = catmull_rom_forward(&coef, &x, 1, 1, &hymeko_spline::Knots::uniform(2));
}
