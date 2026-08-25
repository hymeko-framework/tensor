//! Certification for the 2026-08-25 rotor-holonomy extraction out of Nagare.
//!
//! The reference below is a **verbatim transcription of the pre-extraction Nagare body**
//! (`nagare/src/ops/rotor_holonomy.rs`), written independently and calling its own `qmul`/`qconj`
//! rather than the core implementation. Nothing here calls the core to produce an expected value.

use hymeko_clifford::{rotor_holonomy_backward, rotor_holonomy_forward};

const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn qconj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// Verbatim reference forward.
fn ref_forward(edge_quats: &[f32], n_cycles: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut holo = vec![0.0f32; n_cycles * 4];
    let mut prefixes = vec![0.0f32; n_cycles * k * 4];
    for c in 0..n_cycles {
        let base = c * k * 4;
        let mut p = q_at(edge_quats, c * k);
        prefixes[base..base + 4].copy_from_slice(&p);
        for i in 1..k {
            p = qmul(q_at(edge_quats, c * k + i), p);
            prefixes[base + i * 4..base + i * 4 + 4].copy_from_slice(&p);
        }
        holo[c * 4..c * 4 + 4].copy_from_slice(&p);
    }
    (holo, prefixes)
}

/// Verbatim reference backward.
fn ref_backward(
    edge_quats: &[f32],
    prefixes: &[f32],
    grad_holo: &[f32],
    n_cycles: usize,
    k: usize,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; n_cycles * k * 4];
    for c in 0..n_cycles {
        let base = c * k * 4;
        let grad_h = q_at(grad_holo, c);
        let mut suffix = IDENT;
        for i in (0..k).rev() {
            let p_prev = if i == 0 {
                IDENT
            } else {
                q_at(prefixes, c * k + i - 1)
            };
            let gqi = qmul(qmul(qconj(suffix), grad_h), qconj(p_prev));
            grad[base + i * 4..base + i * 4 + 4].copy_from_slice(&gqi);
            suffix = qmul(suffix, q_at(edge_quats, c * k + i));
        }
    }
    grad
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Deterministic finite quaternion buffer; `unit` normalises each rotor.
fn buffer(n_cycles: usize, k: usize, seed: u64, unit: bool) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut out = vec![0.0f32; n_cycles * k * 4];
    for q in out.chunks_mut(4) {
        for slot in q.iter_mut() {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *slot = ((s >> 20) as f32 / (1u64 << 43) as f32) * 2.0 - 1.0;
        }
        if unit {
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            let n = if n > 0.0 { n } else { 1.0 };
            for slot in q.iter_mut() {
                *slot /= n;
            }
        }
    }
    out
}

/// PHASE 4 — output, full prefix buffer and every backward component are bit-identical to the
/// independent transcription, across `k`, cycle counts, unit/non-unit rotors and several seeds.
#[test]
fn the_extracted_operator_is_bit_identical_to_the_retired_nagare_body() {
    let mut cases = 0usize;
    for k in [1usize, 2, 3, 8] {
        for n_cycles in [1usize, 5] {
            for seed in [1u64, 42, 7919] {
                for unit in [true, false] {
                    let q = buffer(n_cycles, k, seed, unit);
                    let (holo, pre) = rotor_holonomy_forward(&q, n_cycles, k);
                    let (rholo, rpre) = ref_forward(&q, n_cycles, k);
                    assert_eq!(
                        bits(&holo),
                        bits(&rholo),
                        "k={k} n={n_cycles} seed={seed} unit={unit}: holonomy"
                    );
                    assert_eq!(
                        bits(&pre),
                        bits(&rpre),
                        "k={k} n={n_cycles} seed={seed} unit={unit}: prefixes"
                    );

                    let g = buffer(n_cycles, 1, seed ^ 0x5bf0, false);
                    let gq = rotor_holonomy_backward(&q, &pre, &g, n_cycles, k);
                    let rgq = ref_backward(&q, &rpre, &g, n_cycles, k);
                    assert_eq!(
                        bits(&gq),
                        bits(&rgq),
                        "k={k} n={n_cycles} seed={seed} unit={unit}: grad"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 48, "every (k, cycles, seed, unit) cell must run");
}

/// PHASE 5.4 — `k = 1` is exactly the identity map: `H = q_0`, and the prefix buffer is the input.
#[test]
fn k_of_one_is_an_exact_identity_map() {
    let q = buffer(3, 1, 11, true);
    let (holo, pre) = rotor_holonomy_forward(&q, 3, 1);
    assert_eq!(bits(&holo), bits(&q));
    assert_eq!(bits(&pre), bits(&q));
}

/// PHASE 5.3 — positive control: reversing the composition order changes the result on
/// non-commuting rotors. A control that cannot distinguish order would prove nothing.
#[test]
fn reversing_the_composition_order_changes_the_result() {
    let i = [0.0f32, 1.0, 0.0, 0.0];
    let j = [0.0f32, 0.0, 1.0, 0.0];
    let fwd: Vec<f32> = i.iter().chain(&j).copied().collect(); // q0 = i, q1 = j
    let rev: Vec<f32> = j.iter().chain(&i).copied().collect(); // q0 = j, q1 = i
    let (a, _) = rotor_holonomy_forward(&fwd, 1, 2);
    let (b, _) = rotor_holonomy_forward(&rev, 1, 2);
    assert_ne!(bits(&a), bits(&b), "j·i must differ from i·j");
    // And the recorded order is q_{k-1}·…·q_0, i.e. j·i for this input.
    assert_eq!(bits(&a), bits(&qmul(j, i)));
}

/// PHASE 5.6 — negative controls: a wrong prefix order and a wrong suffix order both change the
/// gradient, so the recorded orders are genuinely load-bearing.
#[test]
fn wrong_prefix_or_suffix_order_changes_the_gradient() {
    let (n, k) = (2usize, 3usize);
    let q = buffer(n, k, 5, true);
    let (_, pre) = rotor_holonomy_forward(&q, n, k);
    let g = buffer(n, 1, 77, false);
    let good = rotor_holonomy_backward(&q, &pre, &g, n, k);

    // Wrong prefix order: feed reversed prefixes per cycle.
    let mut bad_pre = pre.clone();
    for c in 0..n {
        let base = c * k * 4;
        let block: Vec<f32> = pre[base..base + k * 4].to_vec();
        for i in 0..k {
            let src = (k - 1 - i) * 4;
            bad_pre[base + i * 4..base + i * 4 + 4].copy_from_slice(&block[src..src + 4]);
        }
    }
    let bad = rotor_holonomy_backward(&q, &bad_pre, &g, n, k);
    assert_ne!(bits(&good), bits(&bad), "prefix order must be load-bearing");

    // Wrong suffix order: reverse the edge quaternions the suffix accumulates.
    let mut q_rev = q.clone();
    for c in 0..n {
        let base = c * k * 4;
        let block: Vec<f32> = q[base..base + k * 4].to_vec();
        for i in 0..k {
            let src = (k - 1 - i) * 4;
            q_rev[base + i * 4..base + i * 4 + 4].copy_from_slice(&block[src..src + 4]);
        }
    }
    let bad2 = rotor_holonomy_backward(&q_rev, &pre, &g, n, k);
    assert_ne!(
        bits(&good),
        bits(&bad2),
        "suffix order must be load-bearing"
    );
}

/// PHASE 5.5 — double cover, stated **correctly**.
///
/// Negating one input rotor negates every prefix from that index onward and negates the holonomy.
/// It does **not** leave the quaternion invariant — `q` and `−q` are the same rotation in SO(3) but
/// different quaternions, so asserting raw invariance would be false.
#[test]
fn negating_one_rotor_negates_the_dependent_prefixes_and_the_holonomy() {
    let (n, k) = (1usize, 4usize);
    let q = buffer(n, k, 909, true);
    let (holo, pre) = rotor_holonomy_forward(&q, n, k);
    for flip in 0..k {
        let mut qf = q.clone();
        for slot in qf[flip * 4..flip * 4 + 4].iter_mut() {
            *slot = -*slot;
        }
        let (holo_f, pre_f) = rotor_holonomy_forward(&qf, n, k);
        assert_eq!(
            bits(&holo_f),
            bits(&holo.iter().map(|v| -v).collect::<Vec<_>>())
        );
        for i in 0..k {
            let got = &pre_f[i * 4..i * 4 + 4];
            let base = &pre[i * 4..i * 4 + 4];
            let want: Vec<f32> = if i >= flip {
                base.iter().map(|v| -v).collect()
            } else {
                base.to_vec()
            };
            assert_eq!(bits(got), bits(&want), "flip={flip} prefix {i}");
        }
    }
}

/// PHASE 5.1 — the adjoint identity, on every quaternion component.
///
/// `<grad_in, dq> == <grad_holo, dH>` for a directional perturbation, which is what makes the
/// backward the true adjoint of the forward rather than merely a plausible expression.
#[test]
fn the_backward_is_the_adjoint_of_the_forward_on_every_component() {
    let (n, k) = (2usize, 3usize);
    let q = buffer(n, k, 31, true);
    let (_, pre) = rotor_holonomy_forward(&q, n, k);
    let g = buffer(n, 1, 131, false);
    let grad = rotor_holonomy_backward(&q, &pre, &g, n, k);

    let h = 1e-3f32;
    for idx in 0..n * k * 4 {
        let mut hi = q.clone();
        hi[idx] += h;
        let mut lo = q.clone();
        lo[idx] -= h;
        let (fh, _) = rotor_holonomy_forward(&hi, n, k);
        let (fl, _) = rotor_holonomy_forward(&lo, n, k);
        // <grad_holo, dH/dq_idx>
        let rhs: f64 = fh
            .iter()
            .zip(&fl)
            .zip(&g)
            .map(|((a, b), gg)| ((*a - *b) as f64 / (2.0 * h as f64)) * (*gg as f64))
            .sum();
        let lhs = grad[idx] as f64;
        assert!(
            (lhs - rhs).abs() < 2e-3 + 0.02 * lhs.abs(),
            "adjoint mismatch at {idx}: analytic {lhs} vs directional {rhs}"
        );
    }
}

/// PHASE 5.7 — the finite-input contract, kept separate from the finite-domain bit-identity.
#[test]
fn non_finite_input_panics_rather_than_propagating() {
    let mut q = buffer(1, 2, 3, true);
    q[0] = f32::NAN;
    assert!(std::panic::catch_unwind(move || rotor_holonomy_forward(&q, 1, 2)).is_err());
}
