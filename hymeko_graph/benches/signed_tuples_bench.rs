//! Resolver-representation measurement for `signed_tuples`.
//!
//! `SignedGraph::build_csr_with_signs` exists because profiling (Epinions k=4, m=128, balance
//! pruner) put `build_sign_lookup` plus its per-edge HashMap hits at ~39 % of CPU cycles. A new
//! multi-edge resolver that reaches for a `HashMap` is therefore reintroducing a *measured* core
//! bottleneck, and must not be chosen without measuring it.
//!
//! Two representations of the same directed `(from, to) -> sign` question, over the same graph and
//! the same query stream:
//!
//! - **HashMap** — `HashMap<(u32, u32), i8>`, SipHash, one probe per lookup.
//! - **Sorted CSR** — per-row `(col, sign)` sorted by column, binary search per lookup. The same
//!   shape `build_csr_with_signs` produces, but carrying this module's resolution contract instead
//!   of last-write-unordered.

use std::collections::HashMap;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use hymeko_graph::SignedGraph;
use hymeko_graph::rand_lcg::Lcg;

/// A multigraph with parallel edges, so resolution is exercised rather than bypassed.
fn fixture(n_nodes: u32, n_edges: usize) -> SignedGraph {
    let mut rng = Lcg::new(0xC0FFEE);
    let mut us = Vec::with_capacity(n_edges);
    let mut vs = Vec::with_capacity(n_edges);
    let mut signs = Vec::with_capacity(n_edges);
    for i in 0..n_edges {
        let u = rng.next_in_range(n_nodes);
        let mut v = rng.next_in_range(n_nodes);
        if u == v {
            v = (v + 1) % n_nodes;
        }
        us.push(u);
        vs.push(v);
        signs.push(if i % 7 == 0 { -1i8 } else { 1 });
    }
    SignedGraph::from_parts(n_nodes, &us, &vs, &signs)
}

/// First-occurrence forward-then-reverse, as a hash table.
fn build_hash(g: &SignedGraph) -> HashMap<(u32, u32), i8> {
    let mut m: HashMap<(u32, u32), i8> = HashMap::with_capacity(g.edges.len() * 2);
    for (i, &(u, v)) in g.edges.iter().enumerate() {
        m.entry((u, v)).or_insert(g.signs[i]);
    }
    for (i, &(u, v)) in g.edges.iter().enumerate() {
        m.entry((v, u)).or_insert(g.signs[i]);
    }
    m
}

/// The same contract as sorted CSR: `row_ptr` over `from`, each row sorted by `to`.
fn build_csr(g: &SignedGraph) -> (Vec<u32>, Vec<u32>, Vec<i8>) {
    let n = g.n_nodes as usize;
    // (from, to, priority, sign): forward entries first, then reverse, lowest priority wins.
    let mut rows: Vec<Vec<(u32, usize, i8)>> = vec![Vec::new(); n];
    for (i, &(u, v)) in g.edges.iter().enumerate() {
        rows[u as usize].push((v, i, g.signs[i]));
    }
    let e = g.edges.len();
    for (i, &(u, v)) in g.edges.iter().enumerate() {
        rows[v as usize].push((u, e + i, g.signs[i]));
    }
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col = Vec::new();
    let mut sign = Vec::new();
    row_ptr.push(0u32);
    for row in rows.iter_mut() {
        row.sort_unstable_by_key(|&(c, p, _)| (c, p));
        let mut j = 0;
        while j < row.len() {
            let c = row[j].0;
            col.push(c);
            sign.push(row[j].2); // lowest priority in the group = first occurrence
            while j < row.len() && row[j].0 == c {
                j += 1;
            }
        }
        row_ptr.push(col.len() as u32);
    }
    (row_ptr, col, sign)
}

#[inline]
fn csr_lookup(csr: &(Vec<u32>, Vec<u32>, Vec<i8>), from: u32, to: u32) -> Option<i8> {
    let (row_ptr, col, sign) = csr;
    let s = row_ptr[from as usize] as usize;
    let e = row_ptr[from as usize + 1] as usize;
    col[s..e].binary_search(&to).ok().map(|k| sign[s + k])
}

/// A query stream shaped like a traversal: consecutive pairs of real edges.
fn queries(g: &SignedGraph, n: usize) -> Vec<(u32, u32)> {
    let mut rng = Lcg::new(7);
    (0..n)
        .map(|_| {
            let e = rng.next_in_range(g.edges.len() as u32) as usize;
            let (u, v) = g.edges[e];
            if e % 2 == 0 { (u, v) } else { (v, u) }
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    // Production-shaped: comparable to the signed trust graphs this module is enumerated over.
    let g = fixture(6_000, 36_000);
    let q = queries(&g, 500_000);

    let mut build = c.benchmark_group("resolver_build");
    build.bench_function("hashmap", |b| {
        b.iter_batched(|| &g, |g| build_hash(g), BatchSize::SmallInput)
    });
    build.bench_function("sorted_csr", |b| {
        b.iter_batched(|| &g, |g| build_csr(g), BatchSize::SmallInput)
    });
    build.finish();

    let hash = build_hash(&g);
    let csr = build_csr(&g);
    // Agreement is a precondition of the comparison being meaningful.
    for &(u, v) in q.iter().take(10_000) {
        assert_eq!(
            hash.get(&(u, v)).copied(),
            csr_lookup(&csr, u, v),
            "({u},{v})"
        );
    }

    let mut lookup = c.benchmark_group("resolver_lookup_500k");
    lookup.bench_function("hashmap", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for &(u, v) in &q {
                acc += hash.get(&(u, v)).copied().unwrap_or(0) as i64;
            }
            acc
        })
    });
    lookup.bench_function("sorted_csr", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for &(u, v) in &q {
                acc += csr_lookup(&csr, u, v).unwrap_or(0) as i64;
            }
            acc
        })
    });
    lookup.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
