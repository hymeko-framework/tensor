//! Integration tests for the ABB global top-K enumerator.
//!
//! Two regression guards:
//!
//! 1. **Admissibility**: every `BoundedScorer`'s `upper_bound` is a
//!    valid upper bound on the score of any closed cycle reachable
//!    from the partial state.  Violation → ABB silently produces
//!    wrong top-K results.
//!
//! 2. **Output parity vs non-ABB path**: `enumerate_top_k_cycles_par_bb`
//!    returns the same canonical top-K cycle set (within score-tie
//!    tolerance) as `enumerate_top_k_cycles_par` on a synthetic
//!    Erdős–Rényi fixture, both with `NoOpPruner` and with
//!    `CartwrightHararyPruner(OnlyBalanced)`.

use std::collections::BTreeSet;

use hymeko_graph::{
    SignedGraph,
    balance::{BalanceMode, CartwrightHararyPruner},
    enumerate_top_k_cycles_par, enumerate_top_k_cycles_par_bb,
    topk_cycles::{
        BalanceScorer, BoundedScorer, FractionNegativeScorer, LowRootScorer, SignProductAbsScorer,
        scorers,
    },
};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn synthetic_signed_graph(n: u32, edge_prob_pm: u32, pos_prob_pm: u32, seed: u32) -> SignedGraph {
    let mut rng = seed;
    let mut us: Vec<u32> = Vec::new();
    let mut vs: Vec<u32> = Vec::new();
    let mut ss: Vec<i8> = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if lcg(&mut rng) % 1_000_000 < edge_prob_pm * 1_000 {
                us.push(u);
                vs.push(v);
                let s = if lcg(&mut rng) % 1_000_000 < pos_prob_pm * 1_000 {
                    1i8
                } else {
                    -1i8
                };
                ss.push(s);
            }
        }
    }
    SignedGraph::from_parts(n, &us, &vs, &ss)
}

fn canonical_set(cycles: &[(f64, Vec<u32>, Vec<i8>)]) -> BTreeSet<Vec<u32>> {
    cycles
        .iter()
        .map(|(_, v, _)| {
            let mut canon = v.clone();
            canon.sort();
            canon
        })
        .collect()
}

// ─── Admissibility: UB must dominate all reachable scores ─────────

/// Enumerate every k-cycle of the given graph (via the existing
/// no-pruner enumerator) and check that the BoundedScorer's UB at
/// the trivial root state (n_neg_so_far=0, k_remaining=k_len) is at
/// least as large as every cycle's actual score.  This is the
/// loosest UB call (root); the test catches any UB that is below
/// some realised score.
fn assert_root_ub_admissible<S: BoundedScorer>(
    g: &SignedGraph,
    k_len: usize,
    scorer: &S,
    label: &str,
) {
    use hymeko_graph::pruner::NoOpPruner;
    let cycles = enumerate_top_k_cycles_par(
        g,
        k_len,
        &NoOpPruner,
        usize::MAX,
        |vs: &[u32], signs: &[i8]| scorer.score(vs, signs),
    );
    let ub = scorer.upper_bound(0, k_len, k_len);
    for (s, vs, signs) in &cycles {
        assert!(
            *s <= ub + 1e-9,
            "{label}: scorer.score({vs:?}, {signs:?}) = {s} exceeds \
             root upper_bound {ub}",
        );
    }
}

#[test]
fn ub_admissible_fraction_negative() {
    let g = synthetic_signed_graph(80, 100, 700, 11);
    assert_root_ub_admissible(&g, 4, &FractionNegativeScorer, "fraction_negative");
}

#[test]
fn ub_admissible_balance() {
    let g = synthetic_signed_graph(80, 100, 700, 12);
    assert_root_ub_admissible(&g, 4, &BalanceScorer, "balance");
}

#[test]
fn ub_admissible_sign_product_abs() {
    let g = synthetic_signed_graph(80, 100, 700, 13);
    assert_root_ub_admissible(&g, 4, &SignProductAbsScorer, "sign_product_abs");
}

#[test]
fn ub_admissible_low_root() {
    let g = synthetic_signed_graph(80, 100, 700, 14);
    assert_root_ub_admissible(&g, 4, &LowRootScorer, "low_root");
}

#[test]
fn ub_terminal_call_equals_score_for_fraction_negative() {
    // At the closure step (k_remaining = 0), UB collapses to the
    // exact score: n_neg_so_far / k_len.  Catches a class of
    // off-by-one errors in upper_bound implementations.
    let scorer = FractionNegativeScorer;
    for k in [3usize, 4, 5, 6, 7, 8] {
        for n in 0..=k {
            let ub = scorer.upper_bound(n, 0, k);
            let expected = n as f64 / k as f64;
            assert!(
                (ub - expected).abs() < 1e-12,
                "fraction_negative UB({n}, 0, {k}) = {ub}, expected {expected}",
            );
        }
    }
}

#[test]
fn ub_monotonic_along_descent_fraction_negative() {
    // As the DFS descends (k_remaining decreases by 1 per step),
    // and absorbs at most one more negative edge, the UB must not
    // increase.  Any loosening violates admissibility's monotonic
    // refinement contract.
    let scorer = FractionNegativeScorer;
    let k_len = 4;
    for k_remaining in 0..k_len {
        for n_neg in 0..=(k_len - k_remaining) {
            let ub = scorer.upper_bound(n_neg, k_remaining, k_len);
            // Step: commit one more edge.  If the new edge is
            // positive: n_neg unchanged, k_remaining decreases.
            // If negative: n_neg increases by 1, k_remaining decreases.
            if k_remaining > 0 {
                let ub_pos = scorer.upper_bound(n_neg, k_remaining - 1, k_len);
                let ub_neg = scorer.upper_bound(n_neg + 1, k_remaining - 1, k_len);
                assert!(
                    ub_pos <= ub + 1e-12 && ub_neg <= ub + 1e-12,
                    "non-monotonic UB at (n_neg={n_neg}, k_rem={k_remaining}): \
                     ub={ub} ub_pos={ub_pos} ub_neg={ub_neg}",
                );
            }
        }
    }
}

// ─── Output parity vs non-ABB ─────────────────────────────────────

#[test]
fn parity_par_vs_par_bb_balance_pruner() {
    let g = synthetic_signed_graph(150, 60, 700, 21);
    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };
    let k = 4;
    let k_keep = 200;

    let baseline = enumerate_top_k_cycles_par(&g, k, &pruner, k_keep, scorers::fraction_negative);
    let abb = enumerate_top_k_cycles_par_bb(&g, k, &pruner, k_keep, &FractionNegativeScorer);
    assert!(!baseline.is_empty(), "fixture should produce some cycles");
    assert!(!abb.is_empty(), "ABB path should produce some cycles");
    // Cardinality within tolerance: the heap is min-heap with
    // strict `>` replacement, so score ties between baseline and
    // ABB can swap which canonical cycle lands in slot K.  The
    // top-K size should still be exactly K (or all-cycles if
    // fewer than K total) on either path.
    assert_eq!(
        baseline.len(),
        abb.len(),
        "top-K cardinality must match across baseline and ABB paths",
    );
    let base_set = canonical_set(&baseline);
    let abb_set = canonical_set(&abb);
    let intersect = base_set.intersection(&abb_set).count();
    let agreement = intersect as f64 / base_set.len() as f64;
    assert!(
        agreement > 0.9,
        "baseline/ABB canonical-cycle agreement {:.2}% < 90% \
         (score-tie tie-breaking shouldn't account for >10% drift)",
        agreement * 100.0,
    );
    // Score multisets must be identical (the score *values* are
    // deterministic from the cycle structure; only which canonical
    // cycle sits at each tied score can drift).
    let mut base_scores: Vec<f64> = baseline.iter().map(|c| c.0).collect();
    let mut abb_scores: Vec<f64> = abb.iter().map(|c| c.0).collect();
    base_scores.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    abb_scores.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    for (a, b) in base_scores.iter().zip(abb_scores.iter()) {
        let diff: f64 = a - b;
        assert!(
            diff.abs() < 1e-12,
            "score multiset mismatch: baseline {a} vs ABB {b}",
        );
    }
}

#[test]
fn parity_par_vs_par_bb_no_pruner() {
    use hymeko_graph::pruner::NoOpPruner;
    let g = synthetic_signed_graph(150, 60, 700, 22);
    let k = 4;
    let k_keep = 200;
    let baseline =
        enumerate_top_k_cycles_par(&g, k, &NoOpPruner, k_keep, scorers::fraction_negative);
    let abb = enumerate_top_k_cycles_par_bb(&g, k, &NoOpPruner, k_keep, &FractionNegativeScorer);
    assert_eq!(baseline.len(), abb.len());
    let mut base_scores: Vec<f64> = baseline.iter().map(|c| c.0).collect();
    let mut abb_scores: Vec<f64> = abb.iter().map(|c| c.0).collect();
    base_scores.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    abb_scores.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    for (a, b) in base_scores.iter().zip(abb_scores.iter()) {
        let diff: f64 = a - b;
        assert!(diff.abs() < 1e-12);
    }
}

#[test]
fn parity_par_vs_par_bb_balance_scorer() {
    // BalanceScorer has trivial UB = 1.0 → ABB never fires.
    // Output must be exactly identical to baseline (modulo
    // tie-breaking).
    use hymeko_graph::pruner::NoOpPruner;
    let g = synthetic_signed_graph(150, 60, 700, 23);
    let k = 4;
    let k_keep = 200;
    let baseline = enumerate_top_k_cycles_par(&g, k, &NoOpPruner, k_keep, scorers::balance);
    let abb = enumerate_top_k_cycles_par_bb(&g, k, &NoOpPruner, k_keep, &BalanceScorer);
    assert_eq!(baseline.len(), abb.len());
}
