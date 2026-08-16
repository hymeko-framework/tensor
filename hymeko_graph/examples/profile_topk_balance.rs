//! Profiling harness for the per-vertex top-K cycle enumeration with
//! the Cartwright-Harary balance pruner — exactly the path exercised
//! by `hymeko_neuro/experiments/run_epinions_balance_5seed_2026_05_10.sh`
//! at the Rust side.
//!
//! Usage:
//! ```bash
//! cargo flamegraph -p hymeko_graph --example profile_topk_balance --release \
//!     -- hymeko_neuro/assets/data/epinions.txt 4 128
//! # → flamegraph.svg in CWD
//! ```
//!
//! Args: `<edge-file> <k_len> <m_per_vertex>`.
//!
//! No new dependencies; loads the same edge format as
//! `examples/cycle_stats.rs` (tab-separated `u v sign`, optional
//! header lines starting with `#`).
//!
//! Output: prints wall-time per phase (load, build, enumerate) plus
//! the cycle count.  The flamegraph captured by the wrapping
//! `cargo flamegraph` tool is the actual artefact for analysis.

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
        eprintln!("usage: profile_topk_balance <edge-file> <k_len> <m_per_vertex>");
        std::process::exit(1);
    }
    let path = &args[1];
    let k_len: usize = args[2].parse().expect("k_len must be an integer");
    let m_per_vertex: usize = args[3].parse().expect("m_per_vertex must be an integer");

    let t0 = Instant::now();
    let (g, n_neg) = load_signed_graph(path);
    eprintln!(
        "load: {:>9.3?}  |V|={}  |E|={}  neg={} ({:.1}%)",
        t0.elapsed(),
        g.n_nodes,
        g.edges.len(),
        n_neg,
        100.0 * n_neg as f64 / g.edges.len() as f64,
    );

    // Match the experiment's pruner + scorer config exactly.
    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };

    eprintln!(
        "enumerate: k={} m={} pruner=balance scorer=fraction_negative",
        k_len, m_per_vertex
    );
    let t_enum = Instant::now();
    let cycles = enumerate_top_k_per_vertex_cycles_par(
        &g,
        k_len,
        &pruner,
        m_per_vertex,
        scorers::fraction_negative,
    );
    let dt_enum = t_enum.elapsed();
    eprintln!("enumerate: {:>9.3?}  cycles={}", dt_enum, cycles.len(),);
}

fn load_signed_graph(path: &str) -> (SignedGraph, usize) {
    let f = File::open(path).expect("open edge file");
    let r = BufReader::new(f);
    let mut us: Vec<u32> = Vec::new();
    let mut vs: Vec<u32> = Vec::new();
    let mut ss: Vec<i8> = Vec::new();
    let mut max_v: u32 = 0;
    let mut n_neg = 0usize;
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
        if sign < 0 {
            n_neg += 1;
        }
    }
    let g = SignedGraph::from_parts(max_v + 1, &us, &vs, &ss);
    (g, n_neg)
}
