//! ABB-feasibility probe: how fast would the global top-K heap
//! threshold rise on Epinions, and what upper-bound prune rate
//! does that imply?
//!
//! Method (no DFS instrumentation): enumerate the full balanced
//! cycle space at k=4 on Epinions via the existing per-vertex
//! top-K path (m_per_vertex large enough to be effectively
//! unrestricted on this workload), bin the resulting cycles by
//! `fraction_negative` score, and from the score distribution
//! compute:
//!
//!   1. The threshold a global top-K heap would settle at for
//!      various K (the K-th highest score in the dataset).
//!   2. For partial paths at depth d ∈ {1, 2, 3} with n_neg ∈ {0..d}
//!      negatives, the upper-bound score `(n_neg + k - d) / k` and
//!      whether it survives at each candidate threshold.
//!
//! Output: a table the user can read directly to decide whether
//! ABB is worth the implementation cost.
//!
//! Usage:
//! ```
//! ./target/release/examples/probe_abb_threshold hymeko_neuro/assets/data/epinions.txt 4
//! ```

use std::collections::HashMap;
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
    if args.len() < 3 {
        eprintln!("usage: probe_abb_threshold <edge-file> <k_len>");
        std::process::exit(1);
    }
    let path = &args[1];
    let k_len: usize = args[2].parse().expect("k_len must be an integer");

    let g = load_signed_graph(path);
    eprintln!("|V|={} |E|={}", g.n_nodes, g.edges.len());

    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };

    // m_per_vertex large enough to capture every balanced cycle on
    // Epinions at k=4 — production used 128, but for a probe we
    // want the full distribution.  256 caps per vertex but is
    // generous.
    let m: usize = 256;

    eprintln!("enumerating k={k_len} m_per_vertex={m} balance pruner ...");
    let t0 = Instant::now();
    let cycles =
        enumerate_top_k_per_vertex_cycles_par(&g, k_len, &pruner, m, scorers::fraction_negative);
    let dt = t0.elapsed();
    eprintln!("enumerated {} cycles in {:.1?}", cycles.len(), dt);

    // ── Score histogram ────────────────────────────────────────────
    // For balanced k_len cycles, n_neg is even.  For k=4, valid bins
    // are 0, 2, 4 → scores 0.0, 0.5, 1.0.  For general k they are
    // 0, 2, ..., k → scores 0.0, 2/k, 4/k, ..., 1.0.
    let mut bin_counts: HashMap<usize, u64> = HashMap::new();
    for (_, _, signs) in &cycles {
        let n_neg = signs.iter().filter(|&&s| s < 0).count();
        *bin_counts.entry(n_neg).or_insert(0) += 1;
    }
    let mut bins: Vec<(usize, u64)> = bin_counts.into_iter().collect();
    bins.sort_by_key(|(n, _)| *n);
    let total: u64 = bins.iter().map(|(_, c)| *c).sum();

    println!();
    println!("─── Score histogram (balanced k={k_len} cycles, fraction_negative scorer) ───");
    println!(
        "  {:>5}  {:>10}  {:>7}  {:>8}",
        "n_neg", "count", "score", "%"
    );
    for (n_neg, count) in &bins {
        let score = *n_neg as f64 / k_len as f64;
        println!(
            "  {:>5}  {:>10}  {:>7.4}  {:>7.2}%",
            n_neg,
            count,
            score,
            100.0 * *count as f64 / total as f64,
        );
    }
    println!("  total: {}", total);

    // ── Top-K threshold trajectory ─────────────────────────────────
    // Compute the K-th highest score for varying K.  This is the
    // value the global top-K heap will settle on once it fills.
    println!();
    println!("─── Global top-K heap threshold (the K-th highest score) ───");
    let mut all_scores: Vec<f64> = cycles.iter().map(|c| c.0).collect();
    all_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    println!("  {:>10}  {:>9}  interpretation", "K", "threshold");
    for &k_target in &[1usize, 100, 1_000, 10_000, 100_000, 1_000_000] {
        if k_target > all_scores.len() {
            continue;
        }
        let t = all_scores[k_target - 1];
        let interp = if t >= 0.99 {
            "all-negative cycles only"
        } else if t >= 0.49 {
            "≥ 2-negative balanced (score ≥ 0.5)"
        } else {
            "any balanced cycle"
        };
        println!("  {:>10}  {:>9.4}  {}", k_target, t, interp);
    }

    // ── ABB prune-rate estimate ────────────────────────────────────
    // For a partial path of length d edges with n_neg negative edges
    // accumulated, the upper bound on the closed cycle's score is
    // (n_neg + (k - d)) / k.  At threshold T the branch is pruned
    // iff UB ≤ T.  We tabulate the matrix; the "fires" cells show
    // configurations where ABB is effective.
    println!();
    println!("─── Upper-bound prune matrix (UB = (n_neg + k-d) / k) ───");
    println!("  Branch is pruned iff UB ≤ threshold T (heap full, cycle can't beat min).");
    println!();
    let thresholds = [0.0f64, 0.25, 0.5, 0.75, 1.0];
    print!("  {:>5} {:>5}  {:>6}", "d", "n_neg", "UB");
    for t in thresholds {
        print!("   T≥{t:.2}");
    }
    println!();
    for d in 1..k_len {
        for n_neg in 0..=d {
            let ub = (n_neg + k_len - d) as f64 / k_len as f64;
            print!("  {:>5} {:>5}  {:>6.4}", d, n_neg, ub);
            for t in thresholds {
                let cut = ub <= t;
                print!("   {}", if cut { "PRUNE " } else { "  -   " });
            }
            println!();
        }
    }

    // ── Summary verdict for global top-K K=10_000 ──────────────────
    println!();
    if all_scores.len() >= 10_000 {
        let t10k = all_scores[9_999];
        println!("─── Verdict ───");
        println!("  Global top-K=10_000 threshold settles at T = {t10k:.4}.");
        let mut prune_count = 0;
        let mut total_branches = 0;
        for d in 1..k_len {
            for n_neg in 0..=d {
                total_branches += 1;
                let ub = (n_neg + k_len - d) as f64 / k_len as f64;
                if ub <= t10k {
                    prune_count += 1;
                }
            }
        }
        println!(
            "  At this T, {} of {} (depth, n_neg) configurations get pruned. ABB potential = {:.0}% of partial paths cut.",
            prune_count,
            total_branches,
            100.0 * prune_count as f64 / total_branches as f64,
        );
        if prune_count == 0 {
            println!(
                "  → ABB does not fire at K=10_000 for this distribution. Different K or scorer needed."
            );
        } else if prune_count >= total_branches / 2 {
            println!("  → ABB fires on >= half of (d, n_neg) cells. Likely worth implementing.");
        } else {
            println!(
                "  → ABB fires on a minority of cells. Marginal payoff; consider per-vertex case carefully before committing."
            );
        }
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
