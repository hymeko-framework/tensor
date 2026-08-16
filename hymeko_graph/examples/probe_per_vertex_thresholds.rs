//! Per-vertex threshold-distribution probe for the Epinions
//! per_vertex top-K config.  Decides whether per-vertex ABB is
//! viable on this workload.
//!
//! Method: run `enumerate_top_k_per_vertex_cycles_par` at the
//! production config (m_per_vertex=128, balance pruner,
//! fraction_negative scorer), then reconstruct each vertex's heap
//! threshold by bucketing the output cycle list by each cycle's
//! k vertices.  For vertex v, the threshold = score of the
//! m_per_vertex-th-best cycle touching v, or "not full" if fewer
//! cycles touch v.
//!
//! Output: a histogram of per-vertex thresholds.  If most vertices
//! have threshold = 0 (heap never filled), per-vertex ABB is
//! impossible without changing m_per_vertex.  If most vertices
//! have threshold ≥ 0.5, per-vertex ABB has real prune potential.
//!
//! Usage:
//! ```
//! ./target/release/examples/probe_per_vertex_thresholds \
//!     hymeko_neuro/assets/data/epinions.txt 4 128
//! ```

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use hymeko_graph::{
    SignedGraph,
    balance::{BalanceMode, CartwrightHararyPruner},
    enumerate_top_k_per_vertex_cycles_par,
    topk_cycles::scorers,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: probe_per_vertex_thresholds <edge-file> <k_len> <m_per_vertex>");
        std::process::exit(1);
    }
    let path = &args[1];
    let k_len: usize = args[2].parse().expect("k_len");
    let m: usize = args[3].parse().expect("m_per_vertex");

    let g = load_signed_graph(path);
    let n = g.n_nodes as usize;
    eprintln!("|V|={n} |E|={}", g.edges.len());

    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };
    eprintln!("enumerating k={k_len} m_per_vertex={m} balance pruner ...");
    let t0 = Instant::now();
    let cycles =
        enumerate_top_k_per_vertex_cycles_par(&g, k_len, &pruner, m, scorers::fraction_negative);
    eprintln!("enumerated {} cycles in {:.1?}", cycles.len(), t0.elapsed());

    // ── Per-vertex threshold reconstruction ────────────────────────
    //
    // The output cycles are the deduplicated UNION of per-vertex
    // top-m sets.  Each cycle of length k touches k vertices and
    // was originally pushed into k heap slots.  To recover what
    // each per-vertex heap converged to, we re-bucket the output
    // cycles by vertex and recompute the top-m per vertex.
    //
    // Approximation: tie-breaking on score equality between the
    // original DFS execution and this post-hoc reconstruction can
    // differ on low-tens of cycles per vertex.  The threshold
    // value at slot m is robust to that drift.
    eprintln!("reconstructing per-vertex thresholds ...");
    let t0 = Instant::now();
    let mut per_vertex_scores: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (score, vs, _) in &cycles {
        for &v in vs {
            per_vertex_scores[v as usize].push(*score);
        }
    }

    let mut full = 0usize;
    let mut empty = 0usize;
    let mut partial = 0usize;
    let mut threshold_hist: [u64; 5] = [0; 5]; // [0, 0.25, 0.5, 0.75, 1.0]
    let mut threshold_sum_full = 0.0f64;
    let mut threshold_sum_full_n = 0u64;
    for scores in per_vertex_scores.iter_mut() {
        if scores.is_empty() {
            empty += 1;
            continue;
        }
        if scores.len() < m {
            partial += 1;
            continue;
        }
        full += 1;
        // m-th highest score = threshold once heap fills.
        // Sort descending; take element at index m-1.
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let t = scores[m - 1];
        threshold_sum_full += t;
        threshold_sum_full_n += 1;
        // Bin into 5 buckets aligned to the k=4 fraction_negative
        // domain {0, 0.25, 0.5, 0.75, 1.0}.
        let bin = if t >= 0.99 {
            4
        } else if t >= 0.74 {
            3
        } else if t >= 0.49 {
            2
        } else if t >= 0.24 {
            1
        } else {
            0
        };
        threshold_hist[bin] += 1;
    }
    eprintln!("reconstruction took {:.1?}", t0.elapsed());

    // ── Report ─────────────────────────────────────────────────────
    println!();
    println!("─── Per-vertex heap-fill status (n_vertices={n}, m={m}) ───");
    println!(
        "  empty (touched no cycles):          {:>7}  ({:>5.2}%)",
        empty,
        100.0 * empty as f64 / n as f64,
    );
    println!(
        "  partial (heap.len() < m, threshold = 0): {:>7}  ({:>5.2}%)",
        partial,
        100.0 * partial as f64 / n as f64,
    );
    println!(
        "  full (heap.len() == m, threshold ≥ 0):   {:>7}  ({:>5.2}%)",
        full,
        100.0 * full as f64 / n as f64,
    );

    println!();
    println!(
        "─── Threshold distribution among the {} FULL heaps ───",
        full
    );
    let labels = [
        "[0.00, 0.24]",
        "[0.25, 0.49]",
        "[0.50, 0.74]",
        "[0.75, 0.99]",
        "[1.00, 1.00]",
    ];
    for (bin, label) in labels.iter().enumerate() {
        let count = threshold_hist[bin];
        let pct = if full > 0 {
            100.0 * count as f64 / full as f64
        } else {
            0.0
        };
        println!("  threshold {label:<14}: {:>7}  ({:>5.2}%)", count, pct);
    }
    if threshold_sum_full_n > 0 {
        println!(
            "  mean threshold (full heaps only): {:.4}",
            threshold_sum_full / threshold_sum_full_n as f64,
        );
    }

    // ── ABB-feasibility verdict ────────────────────────────────────
    println!();
    println!("─── Per-vertex ABB feasibility ───");
    let frac_full = full as f64 / n as f64;
    let frac_thresh_high = (threshold_hist[3] + threshold_hist[4]) as f64 / full.max(1) as f64;
    println!(
        "  Fraction of vertices with full heap: {:.1}%",
        100.0 * frac_full,
    );
    println!(
        "  Of those, fraction with threshold ≥ 0.75: {:.1}%",
        100.0 * frac_thresh_high,
    );

    // The ABB prune fires only if EVERY vertex on a candidate
    // cycle has a full heap whose threshold ≥ UB.  Approximate
    // probability that all k vertices of a random cycle have a
    // full heap = frac_full^k (uniform-vertex assumption — loose
    // since cycles preferentially touch high-degree hubs which
    // fill faster, but useful as a lower bound).
    let p_all_full = frac_full.powi(k_len as i32);
    println!(
        "  P(all {} cycle vertices have full heap)*: {:.2}%",
        k_len,
        100.0 * p_all_full,
    );
    println!("  *uniform-vertex lower bound; actual is higher because hubs over-represent");
    println!();
    if frac_full < 0.10 {
        println!("  → Per-vertex ABB is NOT viable: fewer than 10% of vertices fill.");
        println!("    Almost any cycle touches a non-full vertex; ABB can never fire.");
        println!("    Action: lower m_per_vertex (degree-adaptive), or pivot to global top-K.");
    } else if frac_full > 0.50 && frac_thresh_high > 0.50 {
        println!("  → Per-vertex ABB IS viable: most vertices fill with high thresholds.");
        println!("    Action: implement per-vertex ABB with min-over-cycle-vertices threshold.");
    } else {
        println!("  → Per-vertex ABB is MARGINAL: thresholds vary across vertices.");
        println!("    Action: degree-adaptive m_per_vertex would lift threshold uniformly.");
    }
}

fn load_signed_graph(path: &str) -> SignedGraph {
    let f = File::open(path).expect("open edge file");
    let r = BufReader::new(f);
    let mut us: Vec<u32> = Vec::new();
    let mut vs: Vec<u32> = Vec::new();
    let mut ss: Vec<i8> = Vec::new();
    let mut max_v: u32 = 0;
    for line in r.lines() {
        let line = line.expect("read line");
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let mut parts = s.split([' ', '\t', ',']);
        let u: u32 = match parts.next().and_then(|x| x.parse().ok()) {
            Some(x) => x,
            None => continue,
        };
        let v: u32 = match parts.next().and_then(|x| x.parse().ok()) {
            Some(x) => x,
            None => continue,
        };
        let sign: i8 = match parts.next().and_then(|x| x.parse().ok()) {
            Some(x) => x,
            None => continue,
        };
        if u == v {
            continue;
        }
        max_v = max_v.max(u).max(v);
        us.push(u);
        vs.push(v);
        ss.push(sign);
    }
    SignedGraph::from_parts(max_v + 1, &us, &vs, &ss)
}
