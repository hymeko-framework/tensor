//! Wall-time + flamegraph harness for the ABB global top-K
//! enumerator.  Runs both the baseline `enumerate_top_k_cycles_par`
//! and the new `enumerate_top_k_cycles_par_bb` on the same Epinions
//! input and reports the speedup.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use hymeko_graph::{
    SignedGraph,
    balance::{BalanceMode, CartwrightHararyPruner},
    enumerate_top_k_cycles_par, enumerate_top_k_cycles_par_bb,
    topk_cycles::{FractionNegativeScorer, scorers},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: profile_topk_global_bb <edge-file> <k_len> <K_keep> [mode]");
        eprintln!("  mode: 'baseline' | 'abb' | 'both' (default both)");
        std::process::exit(1);
    }
    let path = &args[1];
    let k_len: usize = args[2].parse().expect("k_len");
    let k_keep: usize = args[3].parse().expect("K_keep");
    let mode = args.get(4).map(|s| s.as_str()).unwrap_or("both");

    let g = load_signed_graph(path);
    eprintln!("|V|={} |E|={}", g.n_nodes, g.edges.len());

    let pruner = CartwrightHararyPruner {
        mode: BalanceMode::OnlyBalanced,
    };

    if mode == "baseline" || mode == "both" {
        eprintln!("baseline (enumerate_top_k_cycles_par) ...");
        let t = Instant::now();
        let cycles =
            enumerate_top_k_cycles_par(&g, k_len, &pruner, k_keep, scorers::fraction_negative);
        let dt = t.elapsed();
        eprintln!("  baseline: {} cycles in {:.3?}", cycles.len(), dt);
    }
    if mode == "abb" || mode == "both" {
        eprintln!("ABB (enumerate_top_k_cycles_par_bb) ...");
        let t = Instant::now();
        let cycles =
            enumerate_top_k_cycles_par_bb(&g, k_len, &pruner, k_keep, &FractionNegativeScorer);
        let dt = t.elapsed();
        eprintln!("  ABB:      {} cycles in {:.3?}", cycles.len(), dt);
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
