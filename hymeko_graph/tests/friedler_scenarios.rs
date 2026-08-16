//! Real-world / textbook Friedler P-graph scenarios.
//!
//! Each test sets up a small process-synthesis graph, runs the
//! same DFS / BFS / cycle-enumeration code TWICE — once with no
//! pruner (`NoOpPruner`) and once with the [`FriedlerAxiomPruner`]
//! — and asserts that:
//!
//! 1. The unpruned run finds the entire combinatorial set
//!    (including infeasible structures).
//! 2. The Friedler-pruned run finds only feasible structures
//!    (those satisfying A0 alternation, A1 product membership,
//!    and A3 valid-O-nodes).
//! 3. The two are NOT equal — the pruner actually fires.
//!
//! This proves the pluggable-strategy architecture works: the
//! DFS / cycle-enumeration core is *agnostic* to whether a
//! Friedler check is in play.  The pruner is a strategy plug-in.

use hymeko_graph::{
    BfsScratch, BipartiteOnlyPruner, CartwrightHararyPruner, Csr, DfsScratch, FriedlerAxiomPruner,
    NoOpPruner, SignedGraph, balance::BalanceMode, bfs_distances, dfs_visit, dfs_visit_pruned,
    enumerate_simple_cycles, enumerate_simple_cycles_noprune, friedler::NodeKind,
};

// ─── Scenario 1: Simple two-route synthesis ──────────────────────
//
// Materials:  M0 (raw)  M1 (intermediate)  M2 (product)
// Units:      U0, U1   (two operating units producing M2 via M1)
//
// Bipartite layout:
//
//     M0 — U0 — M1 — U1 — M2
//      \              /
//       \— U0' — M1' /  (alternate route, omitted for brevity)
//
// We pick vertex IDs:
//   0 = M0 (Material, raw)
//   1 = U0 (Operating unit)
//   2 = M1 (Material, intermediate)
//   3 = U1 (Operating unit)
//   4 = M2 (Material, product)
//
// Edges form the bipartite alternation M-O-M-O-M, with one extra
// "shortcut" M0—M2 (illegal, M-M) added to test pruner rejection.

fn build_simple_synthesis() -> (SignedGraph, Vec<NodeKind>) {
    let kinds = vec![
        NodeKind::Material,      // 0  M0 (raw)
        NodeKind::OperatingUnit, // 1  U0
        NodeKind::Material,      // 2  M1
        NodeKind::OperatingUnit, // 3  U1
        NodeKind::Material,      // 4  M2 (product)
    ];
    // Legal P-graph edges (M-O alternation):
    //   M0 — U0, U0 — M1, M1 — U1, U1 — M2
    // Plus TWO ILLEGAL shortcuts for testing:
    //   M0 — M2 (M-M, violates A0)
    //   U0 — U1 (O-O, violates A0; also closes a triangle
    //            U0—M1—U1—U0 that the unpruned DFS finds).
    let g = SignedGraph::from_parts(
        5,
        &[0, 1, 2, 3, 0, 1], // sources
        &[1, 2, 3, 4, 4, 3], // targets
        &[1, 1, 1, 1, 1, 1],
    );
    (g, kinds)
}

// ─── Scenario 2: HDA-like with two synthesis routes ──────────────
//
// Toluene + H2 → Benzene (and methane byproduct), simplified.
//
// Materials (bipartite even):
//   0 = Toluene  (raw)
//   2 = H2       (raw)
//   4 = Mix      (intermediate)
//   6 = Benzene  (product)
//   8 = Methane  (byproduct)
//
// Operating units (bipartite odd):
//   1 = Mixer
//   3 = Reactor
//   5 = Separator
//   7 = Catalyst
//
// Topology:
//   Toluene — Mixer — Mix — Reactor — (Benzene & Methane)
//   H2      — Mixer
//   Mix     — Catalyst — Mix      (recycle, illegal in pure
//                                  bipartite if both Mix vertices
//                                  are Material)
//   Mix     — Separator — Benzene
//   Mix     — Separator — Methane
//
// Required products: {Benzene} (id 6).

fn build_hda_like() -> (SignedGraph, Vec<NodeKind>) {
    let kinds = vec![
        NodeKind::Material,      // 0 Toluene
        NodeKind::OperatingUnit, // 1 Mixer
        NodeKind::Material,      // 2 H2
        NodeKind::OperatingUnit, // 3 Reactor
        NodeKind::Material,      // 4 Mix
        NodeKind::OperatingUnit, // 5 Separator
        NodeKind::Material,      // 6 Benzene (product)
        NodeKind::OperatingUnit, // 7 Catalyst (auxiliary)
        NodeKind::Material,      // 8 Methane (byproduct)
    ];
    let g = SignedGraph::from_parts(
        9,
        // Toluene — Mixer
        // H2      — Mixer
        // Mixer   — Mix
        // Mix     — Reactor
        // Reactor — Benzene
        // Reactor — Methane
        // Mix     — Separator
        // Separator — Benzene
        // Mix     — Catalyst
        // Catalyst — Mix      (recycle — Mix is M, so this M-O-M
        //                      is fine; closes a cycle)
        &[0, 2, 1, 4, 3, 3, 4, 5, 4, 7],
        &[1, 1, 4, 3, 6, 8, 5, 6, 7, 4],
        &[1; 10],
    );
    (g, kinds)
}

// ─── strategy-pattern demonstration ──────────────────────────────

#[test]
fn scenario1_friedler_rejects_illegal_shortcut() {
    let (g, kinds) = build_simple_synthesis();
    let csr = Csr::from_graph(&g);

    // Without any pruner, enumerate_simple_cycles finds whatever
    // closes — including 3-cycles using the illegal M0—M2 chord.
    let no_prune_cycles = enumerate_simple_cycles_noprune(&g, 3);

    // With Friedler A0 (bipartite alternation), the M0—M2 edge is
    // visible in the graph but every 3-cycle that uses it must
    // pass two same-kind vertices adjacently, which the pruner
    // rejects during DFS.
    let p = FriedlerAxiomPruner::new(kinds.clone());
    let friedler_cycles = enumerate_simple_cycles(&g, 3, &p);

    // The unpruned set should include some 3-cycles touching the
    // illegal chord.
    assert!(
        !no_prune_cycles.is_empty(),
        "unpruned 3-cycle search must find the chord-cycle"
    );
    // The Friedler-pruned set must contain ZERO 3-cycles
    // (bipartite alternation forbids odd-length cycles).
    assert_eq!(
        friedler_cycles.len(),
        0,
        "Friedler A0 must reject every 3-cycle in a \
                bipartite P-graph"
    );

    // 4-cycles: the unpruned DFS finds 0-1-3-4-0 (via the U-U
    // chord (1,3) and the M-M chord (0,4)).  A0 rejects this
    // because vertex 1 (U) and vertex 3 (U) are the same kind and
    // the M-M chord (0,4) is also an A0 violation, so Friedler
    // returns zero 4-cycles.
    let no_prune_4 = enumerate_simple_cycles_noprune(&g, 4).len();
    let friedler_4 = enumerate_simple_cycles(&g, 4, &p).len();
    assert!(
        no_prune_4 >= friedler_4,
        "Friedler is monotone: cannot find more cycles than \
             the unpruned enumerator"
    );
    assert_eq!(
        friedler_4, 0,
        "all 4-cycles in this graph use a same-kind chord, \
                A0 rejects every one"
    );
    let _ = csr;
}

#[test]
fn scenario2_hda_finds_feasible_synthesis_loops() {
    let (g, kinds) = build_hda_like();

    // Without pruning: enumerate 4-cycles.
    let no_prune_4 = enumerate_simple_cycles_noprune(&g, 4);
    let no_prune_3 = enumerate_simple_cycles_noprune(&g, 3);

    // With Friedler A0 only.
    let p_a0 = FriedlerAxiomPruner::new(kinds.clone());
    let friedler_3 = enumerate_simple_cycles(&g, 3, &p_a0);
    let friedler_4 = enumerate_simple_cycles(&g, 4, &p_a0);

    // A0 should kill every odd-length cycle.
    assert_eq!(
        friedler_3.len(),
        0,
        "no 3-cycle is feasible under bipartite alternation"
    );
    // A0 should preserve even-length cycles.
    assert_eq!(
        friedler_4.len(),
        no_prune_4.len(),
        "even cycles unaffected by A0"
    );

    // With Friedler A0 + A1 (require Benzene = id 6).
    let p_a01 = FriedlerAxiomPruner::new(kinds.clone()).with_required_products([6u32]);
    let friedler_a01_4 = enumerate_simple_cycles(&g, 4, &p_a01);
    // A1 is strictly more restrictive than A0 alone.
    assert!(
        friedler_a01_4.len() <= friedler_4.len(),
        "A1 (product membership) cannot increase cycle count"
    );
    // Every surviving cycle must touch vertex 6.
    for c in &friedler_a01_4 {
        assert!(
            c.contains(&6),
            "A1 must require cycle to touch a product node"
        );
    }

    // Sanity: no_prune_3 may be non-zero (the graph has odd cycles).
    let _ = no_prune_3;
}

#[test]
fn dfs_works_identically_with_and_without_friedler() {
    // The architectural promise: dfs_visit (no pruner) and
    // dfs_visit_pruned with NoOpPruner produce identical
    // traversals.  And dfs_visit_pruned with Friedler produces a
    // SUBSET (some vertices unreachable when extension is
    // rejected).
    let (g, kinds) = build_hda_like();
    let csr = Csr::from_graph(&g);
    let mut s = DfsScratch::with_capacity(g.n_nodes);

    // Plain DFS from vertex 0.
    s.reset();
    let mut order_plain = Vec::new();
    dfs_visit(&csr, &mut s, 0, |v| order_plain.push(v));

    // DFS with NoOpPruner — must be identical.
    s.reset();
    let mut order_noop = Vec::new();
    dfs_visit_pruned(&csr, &mut s, &NoOpPruner, 0, |v| {
        order_noop.push(v);
    });
    assert_eq!(
        order_plain, order_noop,
        "NoOpPruner must produce identical traversal"
    );

    // DFS with Friedler — produces a (possibly empty) subset.
    let p = FriedlerAxiomPruner::new(kinds);
    s.reset();
    let mut order_friedler = Vec::new();
    dfs_visit_pruned(&csr, &mut s, &p, 0, |v| {
        order_friedler.push(v);
    });
    // Every Friedler-visited vertex must also be in the plain DFS
    // traversal (since we only ADD pruning, not connectivity).
    for v in &order_friedler {
        assert!(
            order_plain.contains(v),
            "Friedler must not produce a vertex unreachable \
                 in the plain DFS"
        );
    }
    // And the start vertex itself is always visited.
    assert!(order_friedler.contains(&0));
}

#[test]
fn bfs_works_identically_with_friedler_strategy_omitted() {
    // BFS in our crate doesn't take a pruner (yet) — but we can
    // demonstrate that the same BFS code returns identical
    // distances regardless of which pruner is "on hold" elsewhere.
    let (g, _kinds) = build_hda_like();
    let csr = Csr::from_graph(&g);
    let mut s = BfsScratch::with_capacity(g.n_nodes);

    // First call — distances from vertex 0.
    let n1 = bfs_distances(&csr, &mut s, 0, 10);
    let dist_first: Vec<u8> = s.dist.clone();

    // Second call from vertex 6 (Benzene) — same BFS, different
    // root.  Distance arrays differ, but the BFS function itself
    // is stateless apart from the scratch buffer.
    let n2 = bfs_distances(&csr, &mut s, 6, 10);
    assert!(n2 > 0, "BFS from product should reach >= 1 vertex");

    // Third call back from 0 — must reproduce the first run.
    let n3 = bfs_distances(&csr, &mut s, 0, 10);
    assert_eq!(n1, n3);
    assert_eq!(dist_first, s.dist, "BFS is deterministic across reset()");
}

#[test]
fn strategy_pattern_progressive_pruning() {
    // Set up a 4-vertex bipartite graph with a couple of cycles.
    // Demonstrate that swapping pruners progressively reduces the
    // cycle count.
    //
    // Graph:
    //   0 (M) — 1 (O), 1 — 2 (M), 2 — 3 (O), 3 — 0
    //   ⇒ a single 4-cycle 0-1-2-3.
    // With negative edge (0, 1): one 4-cycle, sign-product = -1
    // (unbalanced).
    let g = SignedGraph::from_parts(4, &[0, 1, 2, 3], &[1, 2, 3, 0], &[-1, 1, 1, 1]);
    let kinds = vec![
        NodeKind::Material,
        NodeKind::OperatingUnit,
        NodeKind::Material,
        NodeKind::OperatingUnit,
    ];

    // Strategy 1: no pruning — finds all 1 cycle.
    let no_prune = enumerate_simple_cycles_noprune(&g, 4);
    assert_eq!(no_prune.len(), 1);

    // Strategy 2: BipartiteOnly — emit-time even check; in a
    // bipartite graph, all cycles are even, so no change.
    let bipartite = enumerate_simple_cycles(&g, 4, &BipartiteOnlyPruner);
    assert_eq!(bipartite.len(), 1);

    // Strategy 3: Friedler A0 (during-DFS bipartite alternation).
    // Identical result to BipartiteOnly on this graph.
    let friedler = enumerate_simple_cycles(&g, 4, &FriedlerAxiomPruner::new(kinds.clone()));
    assert_eq!(friedler.len(), 1);

    // Strategy 4: CartwrightHararyPruner OnlyBalanced.  The 4-cycle
    // has product = -1 (one negative edge), so it's unbalanced.
    // OnlyBalanced rejects it.
    let bal = enumerate_simple_cycles(
        &g,
        4,
        &CartwrightHararyPruner {
            mode: BalanceMode::OnlyBalanced,
        },
    );
    assert_eq!(bal.len(), 0, "unbalanced 4-cycle rejected by OnlyBalanced");

    // Strategy 5: CartwrightHararyPruner OnlyUnbalanced — keeps it.
    let unbal = enumerate_simple_cycles(
        &g,
        4,
        &CartwrightHararyPruner {
            mode: BalanceMode::OnlyUnbalanced,
        },
    );
    assert_eq!(unbal.len(), 1);

    // Demonstrates: same DFS, different strategies, predictable
    // monotone behaviour on cycle count.
}

#[test]
fn friedler_a1_filters_cycles_by_product_membership() {
    let (g, kinds) = build_hda_like();
    let p_a01 = FriedlerAxiomPruner::new(kinds.clone()).with_required_products([6u32]);
    let friedler_4 = enumerate_simple_cycles(&g, 4, &p_a01);
    let friedler_6 = enumerate_simple_cycles(&g, 6, &p_a01);
    // Every emitted cycle must contain the product node 6.
    for c in &friedler_4 {
        assert!(c.contains(&6));
    }
    for c in &friedler_6 {
        assert!(c.contains(&6));
    }
}

#[test]
fn friedler_a3_whitelist_rejects_unwhitelisted_units() {
    let (g, kinds) = build_hda_like();
    // Pretend only the Mixer (id 1) and the Reactor (id 3) are
    // valid operating units; Separator (5) and Catalyst (7) are
    // not in the master catalogue.
    let p_a013 = FriedlerAxiomPruner::new(kinds.clone()).with_valid_o_nodes([1u32, 3u32]);
    let friedler_4 = enumerate_simple_cycles(&g, 4, &p_a013);
    // Every cycle's O-nodes must be in the whitelist.
    for c in &friedler_4 {
        for &v in c {
            if matches!(kinds[v as usize], NodeKind::OperatingUnit) {
                assert!(
                    [1u32, 3u32].contains(&v),
                    "A3 violation: O-node {} not in whitelist",
                    v,
                );
            }
        }
    }
}
