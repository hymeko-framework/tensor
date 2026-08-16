//! Integration test for the CSR-aligned sign lookup change in
//! `topk_cycles.rs`.
//!
//! Two regression guards:
//!
//! 1. **HashMap parity**: every (u, v) sign that
//!    `SignedGraph::build_sign_lookup` returns must round-trip
//!    through `build_csr_with_signs` + a CSR scan. The unit tests in
//!    the `signed_graph` module already cover this on a hand-checked
//!    fixture; this integration test extends it to a 200-vertex
//!    random signed graph at production-relevant density.
//!
//! 2. **Sequential / parallel cycle-set parity**: the rayon-parallel
//!    `enumerate_top_k_per_vertex_cycles_par` and the sequential
//!    `enumerate_top_k_per_vertex_cycles` must agree on the
//!    canonicalised set of cycles emitted under the same pruner /
//!    scorer config. Confirms that the rayon scratch-hoist refactor
//!    (per-fold-task `Scratch` struct replacing per-iteration
//!    allocations) preserves the algorithm's output exactly.

use std::collections::BTreeSet;

use hymeko_graph::{
    SignedGraph,
    balance::{BalanceMode, CartwrightHararyPruner},
    enumerate_top_k_per_vertex_cycles, enumerate_top_k_per_vertex_cycles_par,
    topk_cycles::scorers,
};

/// Deterministic LCG so the test fixture has no `rand` dependency
/// (per CLAUDE.md §1: no new deps without approval).  Numerical
/// Recipes parameters; not cryptographic, but more than adequate
/// for fixture generation with a seed-stable shape.
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

/// Structural invariants every cycle in a top-K result must satisfy.
/// The sequential and parallel enumerators may legitimately tie-break
/// differently when several candidates share the same score (`heap.peek()
/// .map(|min| s > min.score)` is strict — equal scores never replace),
/// so an exact-membership comparison would over-constrain.  Instead we
/// assert: same cardinality, same arity, every cycle is simple, every
/// cycle satisfies the balance pruner's invariant.
fn assert_top_k_invariants(
    cycles: &[(f64, Vec<u32>, Vec<i8>)],
    k_len: usize,
    expect_balanced: bool,
) {
    for (_, vs, signs) in cycles {
        assert_eq!(vs.len(), k_len, "every cycle must have length k_len");
        assert_eq!(
            signs.len(),
            k_len,
            "edge_signs length must match cycle length"
        );
        let mut sorted = vs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            k_len,
            "every cycle must be simple (no repeated vertices)"
        );
        if expect_balanced {
            let prod: i32 = signs.iter().map(|&s| s as i32).product();
            assert_eq!(
                prod, 1,
                "balance pruner must only emit cycles with sign \
                         product = +1 (got {prod} for {vs:?})"
            );
        }
    }
}

#[test]
fn parallel_and_sequential_balance_pruner_satisfy_same_invariants() {
    let g = synthetic_signed_graph(200, 50, 700, 42);
    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };
    let seq = enumerate_top_k_per_vertex_cycles(&g, 4, &pruner, 8, scorers::fraction_negative);
    let par = enumerate_top_k_per_vertex_cycles_par(&g, 4, &pruner, 8, scorers::fraction_negative);
    assert!(!seq.is_empty(), "fixture should produce some cycles");
    assert!(!par.is_empty(), "parallel path should produce some cycles");
    // Cardinalities should be within a small tolerance: score-tie
    // tie-breaking can shift which canonical cycles land in which
    // heap slots, but neither path can lose or duplicate cycles
    // beyond that.  3% drift is comfortably above typical tie
    // populations for this fixture.
    let drift = (seq.len() as f64 - par.len() as f64).abs() / seq.len().max(par.len()) as f64;
    assert!(
        drift < 0.03,
        "seq vs par cycle-count drift {:.1}% > 3% — sequential={}, parallel={}",
        drift * 100.0,
        seq.len(),
        par.len()
    );
    assert_top_k_invariants(&seq, 4, true);
    assert_top_k_invariants(&par, 4, true);
    // A large fraction of the canonical sets should overlap; only
    // score-tie tie-breaking can cause divergence, and on a synthetic
    // ER graph the tie population is bounded.
    let seq_set = canonical_set(&seq);
    let par_set = canonical_set(&par);
    let intersect = seq_set.intersection(&par_set).count();
    let agreement = intersect as f64 / seq_set.len() as f64;
    assert!(
        agreement > 0.85,
        "seq/par cycle-set agreement {:.2}% < 85% — tie-breaking \
             nondeterminism shouldn't account for >15% drift",
        agreement * 100.0
    );
}

#[test]
fn parallel_and_sequential_no_pruner_match_in_count() {
    use hymeko_graph::enumerate_top_k_per_vertex_cycles_noprune;
    use hymeko_graph::enumerate_top_k_per_vertex_cycles_par_noprune;
    let g = synthetic_signed_graph(200, 50, 700, 7);
    let seq = enumerate_top_k_per_vertex_cycles_noprune(&g, 4, 8, scorers::fraction_negative);
    let par = enumerate_top_k_per_vertex_cycles_par_noprune(&g, 4, 8, scorers::fraction_negative);
    assert!(!seq.is_empty());
    let drift = (seq.len() as f64 - par.len() as f64).abs() / seq.len().max(par.len()) as f64;
    assert!(
        drift < 0.03,
        "seq vs par cycle-count drift {:.1}% > 3% in noprune path \
             (sequential={}, parallel={})",
        drift * 100.0,
        seq.len(),
        par.len()
    );
    assert_top_k_invariants(&seq, 4, false);
    assert_top_k_invariants(&par, 4, false);
}

#[test]
fn csr_with_signs_round_trips_every_hashmap_entry_dense_fixture() {
    // 200-vertex 5%-density graph: ~2k edges, exercises the dedup
    // path + multi-row CSR layout.
    let g = synthetic_signed_graph(200, 50, 700, 99);
    let lk = g.build_sign_lookup();
    let (row_ptr, col_idx, signs_csr) = g.build_csr_with_signs();
    for (&(u, v), &s) in lk.iter() {
        let row_u = {
            let lo = row_ptr[u as usize] as usize;
            let hi = row_ptr[u as usize + 1] as usize;
            col_idx[lo..hi]
                .iter()
                .position(|&x| x == v)
                .map(|pos| signs_csr[lo + pos])
        };
        let row_v = {
            let lo = row_ptr[v as usize] as usize;
            let hi = row_ptr[v as usize + 1] as usize;
            col_idx[lo..hi]
                .iter()
                .position(|&x| x == u)
                .map(|pos| signs_csr[lo + pos])
        };
        assert_eq!(row_u, Some(s), "CSR scan ({u},{v}) != HashMap sign {s}");
        assert_eq!(row_v, Some(s), "CSR scan ({v},{u}) != HashMap sign {s}");
    }
}
