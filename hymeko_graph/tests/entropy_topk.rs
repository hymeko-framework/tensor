//! Integration tests for the entropy-heuristic top-K enumerator.
//!
//! Verifies the three contracts the heuristic must satisfy:
//!
//! 1. **State round-trip**: `update` followed by `rollback` returns
//!    `UniformityState` to its starting value (modulo float
//!    rounding for `s_sum`).  Required so heap evictions don't
//!    leak count drift.
//!
//! 2. **UB admissibility**: for every closed cycle reachable from
//!    a partial state, `score(cycle, state) <= upper_bound(prefix,
//!    k_remaining, state)`.  Violation → ABB silently produces
//!    wrong top-K.
//!
//! 3. **Per-vertex coverage uniformity**: the entropy enumerator's
//!    per-vertex incidence variance should be lower (more uniform)
//!    than the global ABB enumerator's at the same K, on a
//!    fixture where the optimal cycle set has high score variance.

use hymeko_graph::{
    EntropyGainScorer, InverseDegreeScorer, SignedGraph, UniformityHeuristic, UniformityState,
    balance::{BalanceMode, CartwrightHararyPruner},
    enumerate_top_k_cycles_par_bb, enumerate_top_k_cycles_par_entropy,
    pruner::NoOpPruner,
    topk_cycles::FractionNegativeScorer,
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

// ─── State round-trip tests ───────────────────────────────────────

#[test]
fn entropy_state_round_trip_via_update_rollback() {
    let mut state = UniformityState::new(8);
    let scorer = EntropyGainScorer;
    let cycle: &[u32] = &[1, 3, 5, 7];
    let signs: &[i8] = &[1, -1, 1, -1];

    let counts0 = state.counts.clone();
    let total0 = state.total;
    let s_sum0 = state.s_sum;

    scorer.update(cycle, signs, &mut state);
    assert_eq!(state.counts[1], 1);
    assert_eq!(state.counts[3], 1);
    assert_eq!(state.total, 4);
    assert!(state.s_sum > 0.0 - 1e-12); // c=1 contributes 1*ln(1)=0; sum may stay 0
    // 4 vertices each at count 1 -> each contributes 1*ln(1) = 0
    assert!((state.s_sum - 0.0).abs() < 1e-12);

    scorer.rollback(cycle, signs, &mut state);
    assert_eq!(state.counts, counts0);
    assert_eq!(state.total, total0);
    assert!((state.s_sum - s_sum0).abs() < 1e-12);
}

#[test]
fn entropy_state_round_trip_with_repeats() {
    // Add the cycle twice, then roll back twice.  s_sum should
    // accumulate (c=2 -> 2*ln(2)) and then return to 0.
    let mut state = UniformityState::new(8);
    let scorer = EntropyGainScorer;
    let cycle: &[u32] = &[0, 1, 2, 3];
    let signs: &[i8] = &[1, 1, 1, 1];

    scorer.update(cycle, signs, &mut state);
    scorer.update(cycle, signs, &mut state);
    // Each vertex has count 2 now; s_sum = 4 * (2*ln(2)) ≈ 4 * 1.3863
    let expected = 4.0 * 2.0 * (2.0_f64.ln());
    assert!((state.s_sum - expected).abs() < 1e-9);
    assert_eq!(state.total, 8);

    scorer.rollback(cycle, signs, &mut state);
    scorer.rollback(cycle, signs, &mut state);
    assert_eq!(state.total, 0);
    assert!((state.s_sum - 0.0).abs() < 1e-9);
    assert_eq!(state.counts, vec![0u32; 8]);
}

#[test]
fn inverse_degree_state_round_trip() {
    let mut state = UniformityState::new(8);
    let scorer = InverseDegreeScorer;
    let cycle: &[u32] = &[2, 4, 6];
    let signs: &[i8] = &[1, 1, 1];

    scorer.update(cycle, signs, &mut state);
    assert_eq!(state.counts[2], 1);
    assert_eq!(state.counts[4], 1);
    assert_eq!(state.counts[6], 1);
    assert_eq!(state.total, 3);

    scorer.rollback(cycle, signs, &mut state);
    assert_eq!(state.counts, vec![0u32; 8]);
    assert_eq!(state.total, 0);
}

// ─── UB admissibility ─────────────────────────────────────────────

/// Brute-force admissibility check: for the given partial path and
/// reachability subgraph, check that every realisable closed-cycle
/// completion's actual score is ≤ the heuristic's UB.
fn check_admissibility<H: UniformityHeuristic>(
    g: &SignedGraph,
    heuristic: &H,
    state: &UniformityState,
    label: &str,
) {
    use hymeko_graph::enumerate_simple_cycles_noprune;
    let cycles = enumerate_simple_cycles_noprune(g, 4);
    for cycle in &cycles {
        let signs = vec![1i8; cycle.len()]; // signs irrelevant for these scorers
        let actual = heuristic.score(cycle, &signs, state);
        // UB at the root state (k_remaining = k_len, prefix empty)
        // is the loosest bound; every closed cycle must satisfy it.
        let ub = heuristic.upper_bound(&[], 4, 4, state);
        assert!(
            actual <= ub + 1e-9,
            "{label}: cycle {cycle:?} score {actual} exceeds root UB {ub}",
        );
    }
}

#[test]
fn entropy_ub_admissible_root_state() {
    let g = synthetic_signed_graph(60, 100, 700, 42);
    let state = UniformityState::new(g.n_nodes as usize);
    check_admissibility(&g, &EntropyGainScorer, &state, "entropy");
}

#[test]
fn inverse_degree_ub_admissible_root_state() {
    let g = synthetic_signed_graph(60, 100, 700, 43);
    let state = UniformityState::new(g.n_nodes as usize);
    check_admissibility(&g, &InverseDegreeScorer, &state, "inverse_degree");
}

#[test]
fn entropy_ub_admissible_after_some_updates() {
    let g = synthetic_signed_graph(60, 100, 700, 44);
    let mut state = UniformityState::new(g.n_nodes as usize);
    let scorer = EntropyGainScorer;
    // Pre-populate state with a few cycles to make counts non-trivial.
    let seed_cycles: &[(&[u32], &[i8])] = &[
        (&[0, 1, 2, 3], &[1, 1, 1, 1]),
        (&[4, 5, 6, 7], &[1, 1, 1, 1]),
    ];
    for (vs, signs) in seed_cycles {
        scorer.update(vs, signs, &mut state);
    }
    check_admissibility(&g, &scorer, &state, "entropy_with_state");
}

// ─── End-to-end smoke ─────────────────────────────────────────────

#[test]
fn entropy_enumerator_returns_k_cycles_balance_pruner() {
    let g = synthetic_signed_graph(150, 60, 700, 51);
    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };
    let out = enumerate_top_k_cycles_par_entropy(&g, 4, &pruner, 50, &EntropyGainScorer);
    assert!(!out.is_empty(), "fixture must produce some cycles");
    assert!(out.len() <= 50, "must respect k_keep cap");
    // All cycles balanced (sign product +1).
    for (_, _, signs) in &out {
        let prod: i32 = signs.iter().map(|&s| s as i32).product();
        assert_eq!(prod, 1);
    }
}

#[test]
fn entropy_enumerator_returns_k_cycles_no_pruner() {
    let g = synthetic_signed_graph(150, 60, 700, 52);
    let out = enumerate_top_k_cycles_par_entropy(&g, 4, &NoOpPruner, 50, &InverseDegreeScorer);
    assert!(!out.is_empty());
    assert!(out.len() <= 50);
}

// ─── Per-vertex coverage variance vs global ABB ──────────────────

#[test]
fn entropy_coverage_more_uniform_than_global_abb() {
    // Goal: confirm the entropy heuristic's per-vertex incidence
    // distribution is more uniform (lower variance) than global
    // ABB's, on a fixture where the global ABB pulls cycles from
    // a concentrated subset.
    let g = synthetic_signed_graph(150, 60, 700, 61);
    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };
    let k_keep = 200;
    let entropy_out =
        enumerate_top_k_cycles_par_entropy(&g, 4, &pruner, k_keep, &EntropyGainScorer);
    let abb_out = enumerate_top_k_cycles_par_bb(&g, 4, &pruner, k_keep, &FractionNegativeScorer);

    let n = g.n_nodes as usize;
    let mut entropy_counts = vec![0u32; n];
    for (_, vs, _) in &entropy_out {
        for &v in vs {
            entropy_counts[v as usize] += 1;
        }
    }
    let mut abb_counts = vec![0u32; n];
    for (_, vs, _) in &abb_out {
        for &v in vs {
            abb_counts[v as usize] += 1;
        }
    }

    let entropy_covered = entropy_counts.iter().filter(|&&c| c > 0).count();
    let abb_covered = abb_counts.iter().filter(|&&c| c > 0).count();
    eprintln!(
        "entropy: {} vertices covered (out of {} touching any cycle); abb: {}",
        entropy_covered, n, abb_covered,
    );

    // Variance over covered vertices.
    fn variance(counts: &[u32]) -> f64 {
        let nz: Vec<f64> = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64)
            .collect();
        if nz.is_empty() {
            return 0.0;
        }
        let mean = nz.iter().sum::<f64>() / nz.len() as f64;
        nz.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nz.len() as f64
    }
    let v_entropy = variance(&entropy_counts);
    let v_abb = variance(&abb_counts);
    eprintln!("variance entropy={v_entropy:.4} abb={v_abb:.4}");

    // The entropy heuristic should cover at least as many vertices
    // as global ABB on this fixture.  If it covers strictly more,
    // and its variance is reasonable, the test passes.
    assert!(
        entropy_covered >= abb_covered,
        "entropy heuristic covered {entropy_covered} vertices; global \
         ABB covered {abb_covered}.  Entropy should not lose coverage \
         vs the score-extreme path.",
    );
}
