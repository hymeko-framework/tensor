//! Strategy-pattern demo: same DFS / cycle-enumeration routine,
//! different pruners.  Run with:
//!
//! ```bash
//! cargo run --example strategy_pattern -p hymeko_graph
//! ```

use hymeko_graph::{
    BfsScratch, BipartiteOnlyPruner, CartwrightHararyPruner, Csr, DavisWeakBalancePruner,
    DfsScratch, FriedlerAxiomPruner, NoOpPruner, SignedGraph, balance::BalanceMode, bfs_distances,
    bidirectional_bfs, dfs_visit, dfs_visit_pruned, enumerate_simple_cycles,
    enumerate_simple_cycles_noprune, friedler::NodeKind,
};

fn main() {
    println!("hymeko_graph — strategy-pattern demo\n");

    // ── 1. Build a small bipartite P-graph: 4 Materials, 4 Units,
    // forming a Möbius-like cycle structure.
    //
    //   M0 — U0 — M1 — U1 — M2 — U2 — M3 — U3 — M0
    //
    let n = 8;
    let mut eu = Vec::new();
    let mut ev = Vec::new();
    for i in 0..n {
        eu.push(i);
        ev.push((i + 1) % n);
    }
    // Mixed signs: 3 positive, 1 negative ⇒ unbalanced 8-cycle.
    let signs = vec![1, 1, 1, -1, 1, 1, 1, 1];
    let g = SignedGraph::from_parts(n, &eu, &ev, &signs);

    let kinds = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                NodeKind::Material
            } else {
                NodeKind::OperatingUnit
            }
        })
        .collect::<Vec<_>>();

    // ── 2. Enumerate cycles under several strategies.
    // type-complexity allow: the strategies vector deliberately
    // mixes heterogeneous closures that capture local fixtures by
    // reference; factoring into a type alias would require a named
    // lifetime that isn't stable in 2024-edition example code.
    #[allow(clippy::type_complexity)]
    let strategies: Vec<(&str, Box<dyn Fn() -> Vec<Vec<u32>>>)> = vec![
        (
            "NoOpPruner",
            Box::new(|| enumerate_simple_cycles_noprune(&g, 8)),
        ),
        (
            "BipartiteOnly",
            Box::new(|| enumerate_simple_cycles(&g, 8, &BipartiteOnlyPruner)),
        ),
        (
            "CartwrightHarary OnlyBalanced",
            Box::new(|| {
                enumerate_simple_cycles(
                    &g,
                    8,
                    &CartwrightHararyPruner {
                        mode: BalanceMode::OnlyBalanced,
                    },
                )
            }),
        ),
        (
            "CartwrightHarary OnlyUnbalanced",
            Box::new(|| {
                enumerate_simple_cycles(
                    &g,
                    8,
                    &CartwrightHararyPruner {
                        mode: BalanceMode::OnlyUnbalanced,
                    },
                )
            }),
        ),
        (
            "DavisWeakBalance (no all-neg triads)",
            Box::new(|| enumerate_simple_cycles(&g, 8, &DavisWeakBalancePruner)),
        ),
        (
            "Friedler A0 (bipartite alternation)",
            Box::new(|| enumerate_simple_cycles(&g, 8, &FriedlerAxiomPruner::new(kinds.clone()))),
        ),
    ];

    println!(
        "8-cycle enumeration on the {n}-vertex bipartite ring,\n\
              one negative edge (3,4); cycle product = -1\n"
    );
    println!("  {:<42}  8-cycles found", "strategy");
    println!("  {:<42}  {}", "—".repeat(42), "—".repeat(15));
    for (name, run) in &strategies {
        let cycles = run();
        println!("  {:<42}  {}", name, cycles.len());
    }
    println!();

    // ── 3. DFS strategy-pattern: dfs_visit vs dfs_visit_pruned.
    let csr = Csr::from_graph(&g);
    let mut s = DfsScratch::with_capacity(g.n_nodes);

    let mut order_plain: Vec<u32> = Vec::new();
    s.reset();
    dfs_visit(&csr, &mut s, 0, |v| order_plain.push(v));

    let mut order_noop: Vec<u32> = Vec::new();
    s.reset();
    dfs_visit_pruned(&csr, &mut s, &NoOpPruner, 0, |v| order_noop.push(v));

    let mut order_friedler: Vec<u32> = Vec::new();
    s.reset();
    dfs_visit_pruned(
        &csr,
        &mut s,
        &FriedlerAxiomPruner::new(kinds.clone()),
        0,
        |v| order_friedler.push(v),
    );

    println!("DFS traversal from vertex 0 (Material):");
    println!("  plain DFS         : {:?}", order_plain);
    println!("  DFS + NoOpPruner  : {:?}", order_noop);
    println!("  DFS + Friedler A0 : {:?}", order_friedler);
    println!();
    assert_eq!(
        order_plain, order_noop,
        "plain DFS must equal DFS + NoOpPruner"
    );

    // ── 4. BFS demo (no pruner — BFS is strategy-independent for
    // now; the strategy pattern enters once we add the
    // BFS-with-pruner variant).
    let mut bs = BfsScratch::with_capacity(g.n_nodes);
    let n_reached = bfs_distances(&csr, &mut bs, 0, 16);
    println!("BFS distances from vertex 0:");
    for v in 0..g.n_nodes {
        println!("  v={:>2}  dist={}", v, bs.dist[v as usize]);
    }
    println!("  reached: {} / {}", n_reached, g.n_nodes);

    // ── 5. Bidirectional BFS shortest-path query.
    let path_len = bidirectional_bfs(&csr, 0, 4, 8);
    println!("\nbi-BFS shortest path 0 → 4: {:?}", path_len);
}
