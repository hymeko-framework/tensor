//! **Gate for the rotor-holonomy extraction (2026-08-25).**
//!
//! Before `rotor_holonomy_{forward,backward}` may be rehomed onto this crate's `quat_mul` /
//! `quat_conjugate`, those primitives must be proven **bit-identical** on the finite domain to the
//! expressions the Nagare operator is being retired from. Otherwise the move would silently change
//! every recorded holonomy number.
//!
//! The references below are transcribed **verbatim** from `nagare/src/ops/rotor_holonomy.rs`
//! (`qmul`, `qconj`) and are written independently here, so this compares two separately-authored
//! computations rather than a function against itself.
//!
//! # Scope of the claim
//! Bit-identity is asserted on **finite** inputs only. `quat_mul` / `quat_conjugate` additionally
//! assert `is_finite`, which the retired expressions did not — a **contract tightening**, not an
//! arithmetic difference. That is deliberately kept separate: see
//! `the_finiteness_assertion_is_a_contract_tightening_not_an_arithmetic_difference`.

use hymeko_clifford::{quat_conjugate, quat_mul};

/// Verbatim transcription of the retired `qmul` from `nagare/src/ops/rotor_holonomy.rs`.
fn reference_qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Verbatim transcription of the retired `qconj`.
fn reference_qconj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn bits(q: [f32; 4]) -> [u32; 4] {
    [
        q[0].to_bits(),
        q[1].to_bits(),
        q[2].to_bits(),
        q[3].to_bits(),
    ]
}

/// A deterministic finite generator spanning ordinary, tiny, huge and signed-zero magnitudes.
fn sample(i: usize) -> [f32; 4] {
    const SPECIAL: [f32; 8] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        3.402_823e38,
        -3.402_823e38,
    ];
    let mut s = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    let mut out = [0.0f32; 4];
    for slot in out.iter_mut() {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let pick = (s >> 59) as usize;
        *slot = if pick < SPECIAL.len() {
            SPECIAL[pick]
        } else {
            // Ordinary finite values across several decades, both signs.
            let m = ((s >> 20) as f32 / (1u64 << 43) as f32) * 2.0 - 1.0;
            let scale = [1e-6f32, 1e-3, 1.0, 1e3][((s >> 5) & 3) as usize];
            m * scale
        };
    }
    out
}

/// **The gate.** `quat_mul` is bit-identical to the retired `qmul` over a deterministic
/// many-sample finite sweep, including extremes and signed zeros.
#[test]
fn quat_mul_is_bit_identical_to_the_retired_expression_on_finite_inputs() {
    let mut checked = 0usize;
    for i in 0..20_000usize {
        let (a, b) = (sample(i), sample(i.wrapping_add(7_919)));
        if !a.iter().chain(&b).all(|x| x.is_finite()) {
            continue;
        }
        let got = quat_mul(a, b);
        let want = reference_qmul(a, b);
        assert_eq!(
            bits(got),
            bits(want),
            "quat_mul diverged at i={i}: a={a:?} b={b:?}"
        );
        checked += 1;
    }
    assert!(
        checked > 15_000,
        "the sweep must actually exercise inputs, got {checked}"
    );
}

/// `quat_conjugate` is bit-identical to the retired `qconj`, signed zeros included.
///
/// `-0.0` matters: `to_bits()` distinguishes it from `+0.0`, so a conjugate implemented as
/// `0.0 - q[i]` rather than `-q[i]` would be caught here.
#[test]
fn quat_conjugate_is_bit_identical_to_the_retired_expression() {
    for i in 0..20_000usize {
        let q = sample(i);
        if !q.iter().all(|x| x.is_finite()) {
            continue;
        }
        assert_eq!(bits(quat_conjugate(q)), bits(reference_qconj(q)), "i={i}");
    }
    // Signed zero, explicitly.
    assert_eq!(
        bits(quat_conjugate([0.0, 0.0, -0.0, 0.0])),
        bits(reference_qconj([0.0, 0.0, -0.0, 0.0]))
    );
}

/// Identity and non-commuting controls — the sweep must not be passing on trivial inputs alone.
#[test]
fn identity_and_noncommuting_controls_agree_and_are_discriminating() {
    const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    let x = [0.0f32, 1.0, 0.0, 0.0];
    let y = [0.0f32, 0.0, 1.0, 0.0];

    assert_eq!(bits(quat_mul(IDENT, x)), bits(x), "identity is a left unit");
    assert_eq!(
        bits(quat_mul(x, IDENT)),
        bits(x),
        "identity is a right unit"
    );
    assert_eq!(bits(quat_mul(x, y)), bits(reference_qmul(x, y)));
    assert_eq!(bits(quat_mul(y, x)), bits(reference_qmul(y, x)));

    // The control is only meaningful if these genuinely differ.
    assert_ne!(
        bits(quat_mul(x, y)),
        bits(quat_mul(y, x)),
        "i·j != j·i — a control that cannot distinguish order proves nothing"
    );
}

/// The `is_finite` assertions are a **contract tightening**, not an arithmetic change.
///
/// The retired expressions propagated NaN/Inf silently; the core primitives reject them. This is
/// asserted separately so it can never be confused with the finite-domain bit-identity above, and
/// so the tightening is visible to any caller migrating onto these primitives.
#[test]
fn the_finiteness_assertion_is_a_contract_tightening_not_an_arithmetic_difference() {
    const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    for bad in [
        [f32::NAN, 0.0, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0, 0.0],
        [0.0, f32::NEG_INFINITY, 0.0, 0.0],
    ] {
        assert!(
            std::panic::catch_unwind(|| quat_mul(bad, IDENT)).is_err(),
            "quat_mul must reject the non-finite input {bad:?}"
        );
        assert!(
            std::panic::catch_unwind(|| quat_conjugate(bad)).is_err(),
            "quat_conjugate must reject the non-finite input {bad:?}"
        );
        // The retired expression, by contrast, propagates it without complaint.
        assert!(reference_qmul(bad, IDENT).iter().any(|v| !v.is_finite()));
    }
}
