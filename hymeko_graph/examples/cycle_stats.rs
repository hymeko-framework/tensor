//! Slashdot/Epinions cycle-statistics analysis: load a real signed
//! edge list, induce a subgraph on the top-N highest-degree vertices,
//! and quantify the *informational* effect of axiom-based pruning
//! and top-$K$ scoring.
//!
//! Why a subgraph? Full Slashdot/Epinions cycle enumeration takes
//! minutes (Slashdot k=4 = 55.5M cycles). A top-degree induced
//! subgraph captures the structurally interesting region (hubs are
//! where signed prediction signal lives) and stays tractable in
//! seconds, making this an actionable analysis tool for the question
//! "does top-K cycle filtering keep enough information for HSiKAN?".
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example cycle_stats -p hymeko_graph -- \
//!   hymeko_neuro/assets/data/slashdot.txt 600 3
//! cargo run --release --example cycle_stats -p hymeko_graph -- \
//!   hymeko_neuro/assets/data/epinions.txt 600 3
//! ```
//!
//! Args: `<edge-file> <top_n_vertices> <k_len>`.

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use hymeko_graph::{
    CompositePruner, SignedGraph,
    balance::{BalanceMode, CartwrightHararyPruner},
    enumerate_simple_cycles, enumerate_simple_cycles_noprune, enumerate_top_k_cycles_noprune,
    enumerate_top_k_per_vertex_cycles_noprune,
    topk_cycles::scorers,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: cycle_stats <edge-file> <top_n_vertices> <k_len>");
        std::process::exit(1);
    }
    let path = &args[1];
    let top_n: u32 = args[2].parse().expect("top_n_vertices");
    let k_len: usize = args[3].parse().expect("k_len");

    println!("loading edges from {} …", path);
    let (raw_u, raw_v, raw_s, total_edges, raw_n_neg) = load_edges(path);
    println!(
        "  total edges loaded: {}    raw negatives: {} ({:.1}%)",
        total_edges,
        raw_n_neg,
        100.0 * raw_n_neg as f64 / total_edges as f64,
    );

    // ── Induce subgraph on the top_n highest-degree vertices.
    println!("inducing subgraph on top-{top_n} highest-degree vertices …");
    let (g, vertex_kept, n_kept_edges, n_kept_neg) =
        induce_top_degree(&raw_u, &raw_v, &raw_s, top_n);
    println!(
        "  subgraph: |V|={}  |E|={}  negatives={} ({:.1}%)",
        vertex_kept.len(),
        n_kept_edges,
        n_kept_neg,
        100.0 * n_kept_neg as f64 / n_kept_edges as f64,
    );
    println!();

    // ── Full enumeration: count cycles, balanced vs unbalanced.
    println!(
        "─── full {}-cycle enumeration ───────────────────────",
        k_len
    );
    let t0 = Instant::now();
    let cycles = enumerate_simple_cycles_noprune(&g, k_len);
    let dt_full = t0.elapsed();
    let n_full = cycles.len();
    println!(
        "  full count:      {:>10}  time: {:>10.3?}",
        n_full, dt_full,
    );

    if n_full == 0 {
        println!("  (no cycles at this k — try smaller k or larger top_n)");
        return;
    }

    // Sign-product analysis: how many balanced vs unbalanced.
    let sign_lookup = g.build_sign_lookup();
    let (n_bal, n_unbal) = classify_cycles(&cycles, &sign_lookup, k_len);
    println!(
        "    balanced:      {:>10}  ({:.1}%)",
        n_bal,
        100.0 * n_bal as f64 / n_full as f64,
    );
    println!(
        "    unbalanced:    {:>10}  ({:.1}%)",
        n_unbal,
        100.0 * n_unbal as f64 / n_full as f64,
    );
    println!();

    // ── Cartwright-Harary pruning: how much DFS work do we save?
    println!("─── pruning effect (DFS work saved) ─────────────────────");
    for (mode_name, mode) in &[
        ("OnlyBalanced", BalanceMode::OnlyBalanced),
        ("OnlyUnbalanced", BalanceMode::OnlyUnbalanced),
    ] {
        let p = CompositePruner::new().with("CH", Box::new(CartwrightHararyPruner { mode: *mode }));
        let t0 = Instant::now();
        let cs = enumerate_simple_cycles(&g, k_len, &p);
        let dt = t0.elapsed();
        let stats = p.child_stats();
        let s = &stats[0].1;
        println!(
            "  {:<16}  kept={:>9}  emit_rej={:>9}  time={:>10.3?}",
            mode_name,
            cs.len(),
            s.emit_rejects,
            dt,
        );
    }
    println!();

    // ── Top-K analysis under TWO scoring strategies.
    let n_v = vertex_kept.len() as u32;
    let n_e = n_kept_edges;
    type ScorerEntry = (&'static str, fn(&[u32], &[i8]) -> f64);
    let scorer_configs: Vec<ScorerEntry> = vec![
        ("top-K by balance", scorers::balance),
        ("top-K by fraction_negative", scorers::fraction_negative),
    ];
    for (scorer_name, scorer) in scorer_configs {
        println!("─── {} ────────────────────────────", scorer_name);
        println!(
            "  {:<10}  {:<8}  {:<10}  {:<10}  {:<8}  {:>10}",
            "K", "kept", "%vertex", "%edges", "bal%", "time",
        );
        for &k in &[10usize, 100, 1_000, 10_000, 100_000] {
            if k > n_full * 10 {
                continue;
            }
            let t0 = Instant::now();
            let topk = enumerate_top_k_cycles_noprune(&g, k_len, k, scorer);
            let dt = t0.elapsed();
            let n = topk.len();
            let mut touched: Vec<bool> = vec![false; n_v as usize];
            let mut edges_used: BTreeMap<(u32, u32), bool> = BTreeMap::new();
            let mut bal = 0u64;
            for (_, vs, signs) in &topk {
                for &v in vs {
                    touched[v as usize] = true;
                }
                for j in 0..vs.len() {
                    let u = vs[j];
                    let w = vs[(j + 1) % vs.len()];
                    edges_used.insert((u.min(w), u.max(w)), true);
                }
                let prod: i32 = signs.iter().map(|&s| s as i32).product();
                if prod > 0 {
                    bal += 1;
                }
            }
            let n_touched = touched.iter().filter(|&&b| b).count();
            println!(
                "  {:<10}  {:<8}  {:<10.1}  {:<10.1}  {:<8.1}  {:>10.3?}",
                k,
                n,
                100.0 * n_touched as f64 / n_v as f64,
                100.0 * edges_used.len() as f64 / n_e as f64,
                if n > 0 {
                    100.0 * bal as f64 / n as f64
                } else {
                    0.0
                },
                dt,
            );
        }
        println!();
    }

    // ── Vertex-stratified top-K: per-vertex top-m + global dedup.
    println!("─── vertex-stratified top-m by fraction_negative ────────");
    println!(
        "  {:<10}  {:<8}  {:<10}  {:<10}  {:<8}  {:>10}",
        "m/v", "kept", "%vertex", "%edges", "bal%", "time",
    );
    for &m in &[1usize, 4, 16, 64, 256] {
        let t0 = Instant::now();
        let strat =
            enumerate_top_k_per_vertex_cycles_noprune(&g, k_len, m, scorers::fraction_negative);
        let dt = t0.elapsed();
        let n = strat.len();
        let mut touched: Vec<bool> = vec![false; n_v as usize];
        let mut edges_used: BTreeMap<(u32, u32), bool> = BTreeMap::new();
        let mut bal = 0u64;
        for (_, vs, signs) in &strat {
            for &v in vs {
                touched[v as usize] = true;
            }
            for j in 0..vs.len() {
                let u = vs[j];
                let w = vs[(j + 1) % vs.len()];
                edges_used.insert((u.min(w), u.max(w)), true);
            }
            let prod: i32 = signs.iter().map(|&s| s as i32).product();
            if prod > 0 {
                bal += 1;
            }
        }
        let n_touched = touched.iter().filter(|&&b| b).count();
        println!(
            "  {:<10}  {:<8}  {:<10.1}  {:<10.1}  {:<8.1}  {:>10.3?}",
            m,
            n,
            100.0 * n_touched as f64 / n_v as f64,
            100.0 * edges_used.len() as f64 / n_e as f64,
            if n > 0 {
                100.0 * bal as f64 / n as f64
            } else {
                0.0
            },
            dt,
        );
    }
    println!();

    // ── Random-K (uniform sample) baseline: same |M_e|, no bias.
    println!("─── random-K (uniform shuffle) baseline ─────────────────");
    println!(
        "  {:<10}  {:<8}  {:<10}  {:<10}",
        "K", "kept", "%vertex", "%edges",
    );
    use_random_k(&cycles, &vertex_kept, n_kept_edges, k_len);

    println!();
    println!(
        "Implication for HSiKAN: top-K by balance keeps >>X% of unique\n\
         vertices and edges → if the M_e construction depends on\n\
         vertex-touching coverage, top-K is *informationally* nearly\n\
         lossless.  If it depends on cycle multiplicity through the\n\
         same edge, random-K is the unbiased comparator. The next\n\
         step is to wire either selector into the Python training\n\
         pipeline and re-measure AUC.",
    );
}

// ────────────────────────────────────────────────────────────────────

fn load_edges(path: &str) -> (Vec<u32>, Vec<u32>, Vec<i8>, usize, usize) {
    let f = File::open(path).expect("open edge file");
    let r = BufReader::new(f);
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut s = Vec::new();
    let mut n_neg = 0usize;
    for line in r.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let a: u32 = match parts[0].parse() {
            Ok(x) => x,
            Err(_) => continue,
        };
        let b: u32 = match parts[1].parse() {
            Ok(x) => x,
            Err(_) => continue,
        };
        let sign: i32 = match parts[2].parse() {
            Ok(x) => x,
            Err(_) => continue,
        };
        let sign: i8 = if sign < 0 { -1 } else { 1 };
        if sign < 0 {
            n_neg += 1;
        }
        if a != b {
            u.push(a);
            v.push(b);
            s.push(sign);
        }
    }
    let total = u.len();
    (u, v, s, total, n_neg)
}

fn induce_top_degree(
    raw_u: &[u32],
    raw_v: &[u32],
    raw_s: &[i8],
    top_n: u32,
) -> (SignedGraph, Vec<u32>, usize, usize) {
    // Compute degree.
    let mut deg: BTreeMap<u32, u32> = BTreeMap::new();
    for &x in raw_u {
        *deg.entry(x).or_default() += 1;
    }
    for &x in raw_v {
        *deg.entry(x).or_default() += 1;
    }
    let mut by_deg: Vec<(u32, u32)> = deg.into_iter().collect();
    by_deg.sort_by(|a, b| b.1.cmp(&a.1));
    let kept_set: std::collections::HashSet<u32> = by_deg
        .iter()
        .take(top_n as usize)
        .map(|(v, _)| *v)
        .collect();
    // Re-index kept vertices to 0..top_n.
    let mut sorted_kept: Vec<u32> = kept_set.iter().copied().collect();
    sorted_kept.sort();
    let remap: BTreeMap<u32, u32> = sorted_kept
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i as u32))
        .collect();
    // Filter edges + dedup undirected pairs.
    let mut seen: BTreeMap<(u32, u32), i8> = BTreeMap::new();
    for ((u, v), sg) in raw_u.iter().zip(raw_v.iter()).zip(raw_s.iter()) {
        if let (Some(&ru), Some(&rv)) = (remap.get(u), remap.get(v)) {
            let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
            // First edge wins; skip duplicates with conflicting signs.
            seen.entry((lo, hi)).or_insert(*sg);
        }
    }
    let mut eu = Vec::with_capacity(seen.len());
    let mut ev = Vec::with_capacity(seen.len());
    let mut es = Vec::with_capacity(seen.len());
    let mut n_neg = 0usize;
    for ((u, v), s) in seen {
        eu.push(u);
        ev.push(v);
        es.push(s);
        if s < 0 {
            n_neg += 1;
        }
    }
    let n_kept_edges = eu.len();
    let g = SignedGraph::from_parts(top_n, &eu, &ev, &es);
    (g, sorted_kept, n_kept_edges, n_neg)
}

fn classify_cycles(
    cycles: &[Vec<u32>],
    sign_lookup: &std::collections::HashMap<(u32, u32), i8>,
    k_len: usize,
) -> (u64, u64) {
    let mut bal = 0u64;
    let mut unbal = 0u64;
    for cyc in cycles {
        let mut prod: i32 = 1;
        for j in 0..k_len {
            let u = cyc[j];
            let v = cyc[(j + 1) % k_len];
            let key = (u.min(v), u.max(v));
            prod *= *sign_lookup.get(&key).unwrap_or(&1) as i32;
        }
        if prod > 0 {
            bal += 1;
        } else {
            unbal += 1;
        }
    }
    (bal, unbal)
}

fn use_random_k(cycles: &[Vec<u32>], vertex_kept: &[u32], n_kept_edges: usize, k_len: usize) {
    use std::collections::HashSet;
    // Reservoir-style selection without RNG — just a deterministic
    // stride.  For this analysis we want reproducible numbers, not a
    // true uniform sample.
    let n = cycles.len();
    if n == 0 {
        return;
    }
    let n_v = vertex_kept.len();
    for &k in &[10usize, 100, 1_000, 10_000, 100_000] {
        if k > n {
            continue;
        }
        let stride = (n / k).max(1);
        let mut touched = vec![false; n_v];
        let mut edges = HashSet::new();
        let mut count = 0;
        let mut i = 0;
        while count < k && i < n {
            let cyc = &cycles[i];
            for &v in cyc {
                touched[v as usize] = true;
            }
            for j in 0..k_len {
                let u = cyc[j];
                let w = cyc[(j + 1) % k_len];
                edges.insert((u.min(w), u.max(w)));
            }
            count += 1;
            i += stride;
        }
        let n_t = touched.iter().filter(|&&b| b).count();
        println!(
            "  {:<10}  {:<8}  {:<10.1}  {:<10.1}",
            k,
            count,
            100.0 * n_t as f64 / n_v as f64,
            100.0 * edges.len() as f64 / n_kept_edges as f64,
        );
    }
}
