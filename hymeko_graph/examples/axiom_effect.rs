//! Per-axiom effect on cycle enumeration: build a fixed graph,
//! run with progressively more pruners stacked, and report
//! (cycles found, rejection counts, wall time) per configuration.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example axiom_effect -p hymeko_graph
//! ```

use std::time::Instant;

use hymeko_graph::{
    CompositePruner, NoOpPruner, SignedGraph,
    balance::{BalanceMode, BipartiteOnlyPruner, CartwrightHararyPruner, DavisWeakBalancePruner},
    enumerate_simple_cycles,
    friedler::{FriedlerAxiomPruner, NodeKind},
};

/// Build a moderately-sized bipartite ring with chords — picks up
/// many 6-cycles that exercise all bipartite/balance axioms.
fn build_test_graph(n: u32, k_chord: u32) -> (SignedGraph, Vec<NodeKind>) {
    let mut state: u64 = 0xC0FFEE;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(2685821657736338717)
    };
    let mut eu = Vec::new();
    let mut ev = Vec::new();
    for i in 0..n {
        eu.push(i);
        ev.push((i + 1) % n);
    }
    // Random M-O chords (even → odd) and a sprinkling of M-M, O-O
    // illegal chords so the bipartite pruner has work to do.
    for _ in 0..k_chord {
        let u = (next() % n as u64) as u32;
        let v = (next() % n as u64) as u32;
        if u != v {
            eu.push(u);
            ev.push(v);
        }
    }
    let signs: Vec<i8> = (0..eu.len())
        .map(|i| if i & 5 == 0 { -1 } else { 1 })
        .collect();
    let kinds = (0..n)
        .map(|i| {
            if i & 1 == 0 {
                NodeKind::Material
            } else {
                NodeKind::OperatingUnit
            }
        })
        .collect();
    let g = SignedGraph::from_parts(n, &eu, &ev, &signs);
    (g, kinds)
}

fn main() {
    println!("axiom_effect: per-axiom impact on cycle enumeration\n");

    // Two graphs at different scales.
    for &(n, k_chord, k_len) in &[
        (16u32, 6u32, 4usize),
        (24u32, 12u32, 4usize),
        (16u32, 6u32, 6usize),
    ] {
        let (g, kinds) = build_test_graph(n, k_chord);
        println!("─── graph n={n}  chords={k_chord}  k-cycle={k_len} ───");
        println!("    edges: {}", g.n_edges());
        println!();

        // ── 1. Baseline (NoOp).
        let t0 = Instant::now();
        let cycles = enumerate_simple_cycles(&g, k_len, &NoOpPruner);
        let dt_noop = t0.elapsed();
        let n_noop = cycles.len();
        println!(
            "  {:<26}  cycles={:>6}  ext_rej={:>6}  emit_rej={:>6}  time={:>10.3?}",
            "NoOpPruner", n_noop, 0u64, 0u64, dt_noop,
        );

        // ── 2-N. Single-axiom configurations.
        // type-complexity allow: per-row closures capture local
        // fixtures; a type alias would need a named lifetime that
        // doesn't read well at this call site.
        #[allow(clippy::type_complexity)]
        let configs: Vec<(&str, Box<dyn Fn() -> CompositePruner>)> = vec![
            (
                "Bipartite-only (A0 emit)",
                Box::new(|| CompositePruner::new().with("A0_emit", Box::new(BipartiteOnlyPruner))),
            ),
            (
                "Friedler-A0 (during DFS)",
                Box::new(|| {
                    CompositePruner::new().with(
                        "Friedler_A0",
                        Box::new(FriedlerAxiomPruner::new(kinds.clone())),
                    )
                }),
            ),
            (
                "Cartwright-Harary balanced",
                Box::new(|| {
                    CompositePruner::new().with(
                        "CH_balanced",
                        Box::new(CartwrightHararyPruner {
                            mode: BalanceMode::OnlyBalanced,
                        }),
                    )
                }),
            ),
            (
                "Davis weak-balance",
                Box::new(|| CompositePruner::new().with("Davis", Box::new(DavisWeakBalancePruner))),
            ),
            (
                "A0 + CH balanced",
                Box::new(|| {
                    CompositePruner::new()
                        .with(
                            "Friedler_A0",
                            Box::new(FriedlerAxiomPruner::new(kinds.clone())),
                        )
                        .with(
                            "CH_balanced",
                            Box::new(CartwrightHararyPruner {
                                mode: BalanceMode::OnlyBalanced,
                            }),
                        )
                }),
            ),
            (
                "A0 + Davis",
                Box::new(|| {
                    CompositePruner::new()
                        .with(
                            "Friedler_A0",
                            Box::new(FriedlerAxiomPruner::new(kinds.clone())),
                        )
                        .with("Davis", Box::new(DavisWeakBalancePruner))
                }),
            ),
        ];

        for (name, build) in configs {
            let p = build();
            let t0 = Instant::now();
            let cycles = enumerate_simple_cycles(&g, k_len, &p);
            let dt = t0.elapsed();
            let stats = p.child_stats();
            let total_ext_rej: u64 = stats.iter().map(|(_, s)| s.extend_rejects).sum();
            let total_emit_rej: u64 = stats.iter().map(|(_, s)| s.emit_rejects).sum();
            println!(
                "  {:<26}  cycles={:>6}  ext_rej={:>6}  emit_rej={:>6}  time={:>10.3?}",
                name,
                cycles.len(),
                total_ext_rej,
                total_emit_rej,
                dt,
            );
            // Per-child breakdown for composites with > 1 child.
            if stats.len() > 1 {
                for (cname, s) in &stats {
                    println!(
                        "      └─ {:<22}  ext={}/{}  emit={}/{}",
                        cname, s.extend_rejects, s.extend_calls, s.emit_rejects, s.emit_calls,
                    );
                }
            }
        }
        println!();
    }
}
