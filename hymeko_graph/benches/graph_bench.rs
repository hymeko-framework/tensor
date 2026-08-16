//! Microbenches for hymeko_graph — DFS, BFS, k-cycle enumeration
//! and top-k cycle enumeration across graph sizes and pruning
//! strategies.
//!
//! Run all:        `cargo bench -p hymeko_graph`
//! One group:      `cargo bench -p hymeko_graph -- dfs`
//! Compare full vs top-k: `cargo bench -p hymeko_graph -- topk`
//!
//! The graph generators below are tuned so that even the largest
//! configuration completes a single bench iteration in well under a
//! second on a 2020-class laptop. Real-world graphs (Slashdot,
//! Epinions) are not bundled — Slashdot lives in
//! `hymeko_py/src/cycles.rs` benches.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use hymeko_graph::{
    NoOpPruner, SignedGraph,
    balance::{BalanceMode, BipartiteOnlyPruner, CartwrightHararyPruner},
    cycle_enum::enumerate_simple_cycles_noprune,
    enumerate_simple_cycles, enumerate_top_k_cycles_noprune,
    friedler::{FriedlerAxiomPruner, NodeKind},
    topk_cycles::scorers,
    traversal::{
        BfsScratch, Csr, DfsScratch, bfs_distances, bidirectional_bfs, dfs_visit, dfs_visit_pruned,
    },
    traversal_heuristic::{AstarScratch, ZeroHeuristic, astar},
};

// ─── Graph generators ───────────────────────────────────────────────

/// Dense Erdős-Rényi-like signed graph: every pair of distinct
/// vertices is connected with probability `p` (deterministic LCG so
/// repeated runs are byte-identical).
fn gen_random(n: u32, p_pct: u32, seed: u64) -> SignedGraph {
    let mut state = seed | 1;
    let mut next = || {
        // xorshift64*
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(2685821657736338717)
    };
    let mut eu = Vec::new();
    let mut ev = Vec::new();
    let mut signs = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            let r = (next() % 100) as u32;
            if r < p_pct {
                eu.push(u);
                ev.push(v);
                let s = if (next() & 1) == 0 { 1i8 } else { -1 };
                signs.push(s);
            }
        }
    }
    SignedGraph::from_parts(n, &eu, &ev, &signs)
}

/// Bipartite ring (P-graph-shaped): even vertices = M, odd = O,
/// edges link consecutive vertices plus k_chord random M-O chords.
fn gen_bipartite_ring(n: u32, k_chord: u32, seed: u64) -> SignedGraph {
    let mut state = seed | 1;
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
    for _ in 0..k_chord {
        // Random M (even) → O (odd) chord.
        let m = ((next() % (n as u64 / 2)) * 2) as u32 % n;
        let o = ((next() % (n as u64 / 2)) * 2 + 1) as u32 % n;
        if m != o {
            eu.push(m);
            ev.push(o);
        }
    }
    let signs: Vec<i8> = (0..eu.len())
        .map(|i| if i & 7 == 0 { -1 } else { 1 })
        .collect();
    SignedGraph::from_parts(n, &eu, &ev, &signs)
}

fn gen_kinds_alternating(n: u32) -> Vec<NodeKind> {
    (0..n)
        .map(|i| {
            if i & 1 == 0 {
                NodeKind::Material
            } else {
                NodeKind::OperatingUnit
            }
        })
        .collect()
}

// ─── DFS ────────────────────────────────────────────────────────────

fn bench_dfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("dfs");
    for &n in &[64u32, 256, 1024, 4096, 16384] {
        let g = gen_random(n, 6, 0xDEADBEEF);
        let csr = Csr::from_graph(&g);
        let kinds = gen_kinds_alternating(n);
        group.throughput(Throughput::Elements(n as u64));

        // 1. Plain DFS — no pruner machinery.
        group.bench_with_input(BenchmarkId::new("plain", n), &n, |b, _| {
            let mut s = DfsScratch::with_capacity(n);
            b.iter(|| {
                s.reset();
                let mut visited = 0u64;
                dfs_visit(&csr, &mut s, 0, |_| visited += 1);
                black_box(visited)
            })
        });

        // 2. DFS + NoOpPruner — measures the pruner-trait overhead.
        group.bench_with_input(BenchmarkId::new("noop_pruner", n), &n, |b, _| {
            let mut s = DfsScratch::with_capacity(n);
            b.iter(|| {
                s.reset();
                let mut visited = 0u64;
                dfs_visit_pruned(&csr, &mut s, &NoOpPruner, 0, |_| visited += 1);
                black_box(visited)
            })
        });

        // 3. DFS + Friedler pruner — full axiom check at every step.
        group.bench_with_input(BenchmarkId::new("friedler_pruner", n), &n, |b, _| {
            let mut s = DfsScratch::with_capacity(n);
            let p = FriedlerAxiomPruner::new(kinds.clone());
            b.iter(|| {
                s.reset();
                let mut visited = 0u64;
                dfs_visit_pruned(&csr, &mut s, &p, 0, |_| visited += 1);
                black_box(visited)
            })
        });
    }
    group.finish();
}

// ─── BFS ────────────────────────────────────────────────────────────

fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs");
    for &n in &[64u32, 256, 1024, 4096, 16384] {
        let g = gen_random(n, 6, 0xDEADBEEF);
        let csr = Csr::from_graph(&g);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("distances", n), &n, |b, _| {
            let mut s = BfsScratch::with_capacity(n);
            b.iter(|| {
                let r = bfs_distances(&csr, &mut s, 0, 32);
                black_box(r)
            })
        });

        group.bench_with_input(BenchmarkId::new("bi_bfs", n), &n, |b, _| {
            let target = (n - 1) % n;
            b.iter(|| {
                let r = bidirectional_bfs(&csr, 0, target, 32);
                black_box(r)
            })
        });
    }
    group.finish();
}

// ─── k-cycle enumeration ───────────────────────────────────────────

fn bench_cycles(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_cycles");
    // Bipartite-ring instance: cycle counts grow with chord count.
    for &n in &[16u32, 32, 64] {
        let g = gen_bipartite_ring(n, n / 2, 0x1234);
        let kinds = gen_kinds_alternating(n);
        group.throughput(Throughput::Elements(n as u64));

        // k=4 cycles, three pruning strategies.
        for k in [4usize, 6] {
            group.bench_with_input(BenchmarkId::new(format!("k{k}/no_prune"), n), &n, |b, _| {
                b.iter(|| {
                    let cs = enumerate_simple_cycles_noprune(&g, k);
                    black_box(cs.len())
                })
            });

            group.bench_with_input(
                BenchmarkId::new(format!("k{k}/bipartite"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let cs = enumerate_simple_cycles(&g, k, &BipartiteOnlyPruner);
                        black_box(cs.len())
                    })
                },
            );

            group.bench_with_input(BenchmarkId::new(format!("k{k}/friedler"), n), &n, |b, _| {
                let p = FriedlerAxiomPruner::new(kinds.clone());
                b.iter(|| {
                    let cs = enumerate_simple_cycles(&g, k, &p);
                    black_box(cs.len())
                })
            });

            group.bench_with_input(
                BenchmarkId::new(format!("k{k}/cartwright_harary"), n),
                &n,
                |b, _| {
                    let p = CartwrightHararyPruner {
                        mode: BalanceMode::OnlyBalanced,
                    };
                    b.iter(|| {
                        let cs = enumerate_simple_cycles(&g, k, &p);
                        black_box(cs.len())
                    })
                },
            );
        }
    }
    group.finish();
}

// ─── Top-k vs full ─────────────────────────────────────────────────

fn bench_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("topk_vs_full");
    // Use a moderately dense ER graph so the full cycle set is large
    // enough to make the top-k memory win visible.
    for &n in &[24u32, 32, 40] {
        let g = gen_random(n, 25, 0xC0FFEE);
        group.throughput(Throughput::Elements(n as u64));

        // Full enumeration of all 4-cycles.
        group.bench_with_input(BenchmarkId::new("full_k4", n), &n, |b, _| {
            b.iter(|| {
                let cs = enumerate_simple_cycles_noprune(&g, 4);
                black_box(cs.len())
            })
        });

        // Top-100 4-cycles by balance.
        group.bench_with_input(BenchmarkId::new("top100_k4_balance", n), &n, |b, _| {
            b.iter(|| {
                let cs = enumerate_top_k_cycles_noprune(&g, 4, 100, scorers::balance);
                black_box(cs.len())
            })
        });

        // Top-10 — heap stays smaller, but DFS work is identical.
        group.bench_with_input(BenchmarkId::new("top10_k4_balance", n), &n, |b, _| {
            b.iter(|| {
                let cs = enumerate_top_k_cycles_noprune(&g, 4, 10, scorers::balance);
                black_box(cs.len())
            })
        });
    }
    group.finish();
}

// ─── A* with zero heuristic (Dijkstra/BFS via priority queue) ──────

fn bench_astar(c: &mut Criterion) {
    let mut group = c.benchmark_group("astar");
    for &n in &[256u32, 1024, 4096, 16384] {
        let g = gen_random(n, 6, 0xDEADBEEF);
        let csr = Csr::from_graph(&g);
        let goal = n - 1;
        group.throughput(Throughput::Elements(n as u64));

        // Plain BFS distances — baseline.
        group.bench_with_input(BenchmarkId::new("bfs_baseline", n), &n, |b, _| {
            let mut s = BfsScratch::with_capacity(n);
            b.iter(|| {
                let r = bfs_distances(&csr, &mut s, 0, 64);
                black_box(r)
            })
        });

        // A* with zero heuristic.
        group.bench_with_input(BenchmarkId::new("zero_heuristic", n), &n, |b, _| {
            let mut s = AstarScratch::with_capacity(n);
            b.iter(|| {
                let r = astar(&csr, &mut s, 0, goal, &ZeroHeuristic, 64);
                black_box(r)
            })
        });
    }
    group.finish();
}

// ─── Larger cycle instances ────────────────────────────────────────

fn bench_cycles_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_cycles_large");
    group.sample_size(10);
    for &n in &[128u32, 256] {
        let g = gen_bipartite_ring(n, n / 2, 0xC0FFEE);
        let kinds = gen_kinds_alternating(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("k4/no_prune", n), &n, |b, _| {
            b.iter(|| {
                let cs = enumerate_simple_cycles_noprune(&g, 4);
                black_box(cs.len())
            })
        });
        group.bench_with_input(BenchmarkId::new("k4/friedler", n), &n, |b, _| {
            let p = FriedlerAxiomPruner::new(kinds.clone());
            b.iter(|| {
                let cs = enumerate_simple_cycles(&g, 4, &p);
                black_box(cs.len())
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)        // keep wall-time reasonable
        .warm_up_time(std::time::Duration::from_millis(300))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_dfs, bench_bfs, bench_astar, bench_cycles,
              bench_cycles_large, bench_topk
}
criterion_main!(benches);
