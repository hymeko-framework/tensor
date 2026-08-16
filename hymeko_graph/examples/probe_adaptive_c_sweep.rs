//! CPU-only c-sweep probe for the degree-adaptive m_v enumerator.
//!
//! Runs the per-vertex top-K enumeration on a real signed graph at
//! varying values of the slope `c`, recording for each:
//!   - total cycle count emitted
//!   - per-vertex coverage histogram
//!   - score histogram (how the cycle's score distribution shifts)
//!   - full-heap rate
//!   - wall time
//!
//! Used as a pre-training prediction for which c values to test at
//! GPU-time (the actual training smoke).  CPU-only, so safe to run
//! concurrently with an in-flight 5-seed validation that uses the
//! GPU.
//!
//! Usage:
//! ```bash
//! cargo run --release --example probe_adaptive_c_sweep -p hymeko_graph -- \
//!     hymeko_neuro/assets/data/epinions.txt 4 128 0.25 0.5 1.0 2.0 4.0 8.0
//! ```
//!
//! Args: `<edge-file> <k_len> <m_max> <c_1> <c_2> ...`

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use hymeko_graph::{
    SignedGraph,
    balance::{BalanceMode, CartwrightHararyPruner},
    degree_adaptive_m_v, enumerate_top_k_per_vertex_cycles_par_adaptive,
    topk_cycles::scorers,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: probe_adaptive_c_sweep <edge-file> <k_len> <m_max> <c_1> [<c_2> ...]"
        );
        std::process::exit(1);
    }
    let path = &args[1];
    let k_len: usize = args[2].parse().expect("k_len");
    let m_max: u32 = args[3].parse().expect("m_max");
    let cs: Vec<f64> = args[4..]
        .iter()
        .map(|s| s.parse::<f64>().expect("c values"))
        .collect();

    let g = load_signed_graph(path);
    let n = g.n_nodes as usize;
    eprintln!("|V|={n} |E|={}", g.edges.len());
    eprintln!("k_len={k_len}  m_max={m_max}  c values: {cs:?}");

    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };

    println!();
    println!("─── Adaptive c-sweep on Epinions k={k_len} balance pruner ───");
    println!(
        "  {:>5}  {:>6}  {:>9}  {:>9}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}",
        "c", "wall_s", "cycles", "covered_v", "full_rate", "score_0", "score_0.5", "score_1.0", "mean_score",
    );

    for &c in &cs {
        let m_v = degree_adaptive_m_v(&g, 1, m_max, c);

        let t0 = Instant::now();
        let out = enumerate_top_k_per_vertex_cycles_par_adaptive(
            &g,
            k_len,
            &pruner,
            &m_v,
            scorers::fraction_negative,
        );
        let wall = t0.elapsed().as_secs_f64();

        // Per-vertex coverage reconstruction (matches the
        // probe_per_vertex_thresholds.rs technique).
        let mut counts: HashMap<u32, u32> = HashMap::new();
        let mut score_bins = [0u64; 3]; // [score=0, 0<score<1, score=1]
        let mut score_sum = 0.0f64;
        for (score, vs, _) in &out {
            for &v in vs {
                *counts.entry(v).or_insert(0) += 1;
            }
            if *score >= 0.99 {
                score_bins[2] += 1;
            } else if *score <= 0.01 {
                score_bins[0] += 1;
            } else {
                score_bins[1] += 1;
            }
            score_sum += *score;
        }
        let covered = counts.len();
        // Full-heap rate: vertex's contribution count >= m_v[v].
        let mut full = 0usize;
        for v in 0..n as u32 {
            let cap = m_v[v as usize] as usize;
            if cap == 0 {
                continue;
            }
            let len = counts.get(&v).copied().unwrap_or(0) as usize;
            if len >= cap {
                full += 1;
            }
        }
        let full_rate = full as f64 / n as f64;
        let mean_score = if out.is_empty() {
            0.0
        } else {
            score_sum / out.len() as f64
        };

        println!(
            "  {:>5.2}  {:>6.1}  {:>9}  {:>9}  {:>10.4}  {:>9}  {:>9}  {:>9}  {:>9.4}",
            c,
            wall,
            out.len(),
            covered,
            full_rate,
            score_bins[0],
            score_bins[1],
            score_bins[2],
            mean_score,
        );
    }

    println!();
    println!("Diagnostic: smaller c → smaller m_v → fewer cycles overall;");
    println!("higher full-heap rate; mean score should rise (only top-fraction-negative");
    println!("cycles per vertex retained, vs the bottom-saturated fixed-m cycles).");
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
