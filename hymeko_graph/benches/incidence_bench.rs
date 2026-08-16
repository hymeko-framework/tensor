//! Criterion microbench: serial vs parallel `build_edge_incidence`
//! over a Slashdot-class fixture (200 vertices, 3500 tuples, ~half-T
//! queries, k=2 self-edges).  Smaller-than-real-Slashdot to keep CI
//! green, but the per-edge work shape is identical to the
//! production-scale hot loop in Stage D-3-BREAK Phase 7+.
//!
//! Run:           `cargo bench -p hymeko_graph -- incidence`
//! Save baseline: `cargo bench -p hymeko_graph -- incidence --save-baseline serial`
//! Compare:       `cargo bench -p hymeko_graph -- incidence --baseline serial`

use std::collections::HashSet;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use hymeko_graph::incidence::{
    build_edge_incidence, BuildOpts, IncidenceOutput,
};
use hymeko_graph::rand_lcg::Lcg;

struct Fixture {
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    csr_row_ptr: Vec<u32>,
    csr_col_idx: Vec<u32>,
    self_keys_u: Vec<u32>,
    self_keys_v: Vec<u32>,
    self_tuple_idx: Vec<u32>,
}

fn build_fixture(seed: u64, n_vertices: u32, n_tuples: u32) -> Fixture {
    let mut rng = Lcg::new(seed);
    let mut tuples: Vec<(u32, u32)> = Vec::with_capacity(n_tuples as usize);
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    while (tuples.len() as u32) < n_tuples {
        let a = rng.next_in_range(n_vertices);
        let b = rng.next_in_range(n_vertices);
        if a == b { continue; }
        let key = if a < b { (a, b) } else { (b, a) };
        if seen.insert(key) { tuples.push((a, b)); }
    }

    let mut flat: Vec<(u32, u32)> = Vec::with_capacity(tuples.len() * 2);
    for (t_idx, &(a, b)) in tuples.iter().enumerate() {
        flat.push((a, t_idx as u32));
        flat.push((b, t_idx as u32));
    }
    flat.sort();
    let mut csr_row_ptr: Vec<u32> = vec![0; (n_vertices + 1) as usize];
    let mut csr_col_idx: Vec<u32> = Vec::with_capacity(flat.len());
    let mut cur_v: u32 = 0;
    for &(v, t) in &flat {
        while cur_v < v {
            cur_v += 1;
            csr_row_ptr[cur_v as usize] = csr_col_idx.len() as u32;
        }
        let start = csr_row_ptr[cur_v as usize];
        if csr_col_idx.len() as u32 > start && *csr_col_idx.last().unwrap() == t {
            continue;
        }
        csr_col_idx.push(t);
    }
    while (cur_v as usize) < n_vertices as usize {
        cur_v += 1;
        csr_row_ptr[cur_v as usize] = csr_col_idx.len() as u32;
    }

    let n_queries = n_tuples / 2;
    let mut edges_u: Vec<u32> = Vec::with_capacity(n_queries as usize);
    let mut edges_v: Vec<u32> = Vec::with_capacity(n_queries as usize);
    let mut q_seen: HashSet<(u32, u32)> = HashSet::new();
    while (edges_u.len() as u32) < n_queries {
        let u = rng.next_in_range(n_vertices);
        let v = rng.next_in_range(n_vertices);
        if u == v { continue; }
        let key = if u < v { (u, v) } else { (v, u) };
        if q_seen.insert(key) {
            edges_u.push(u);
            edges_v.push(v);
        }
    }

    let mut self_keys_u: Vec<u32> = Vec::with_capacity(tuples.len());
    let mut self_keys_v: Vec<u32> = Vec::with_capacity(tuples.len());
    let mut self_tuple_idx: Vec<u32> = Vec::with_capacity(tuples.len());
    for (t_idx, &(a, b)) in tuples.iter().enumerate() {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self_keys_u.push(lo);
        self_keys_v.push(hi);
        self_tuple_idx.push(t_idx as u32);
    }

    Fixture {
        edges_u,
        edges_v,
        csr_row_ptr,
        csr_col_idx,
        self_keys_u,
        self_keys_v,
        self_tuple_idx,
    }
}

fn bench_one_size(c: &mut Criterion, label: &str, n_vertices: u32, n_tuples: u32) {
    let fix = build_fixture(0x5EEDED, n_vertices, n_tuples);
    let mut grp = c.benchmark_group(format!("incidence_build_{label}"));
    grp.sample_size(20);

    grp.bench_function("serial", |b| {
        b.iter(|| {
            let r = build_edge_incidence(
                black_box(&fix.edges_u),
                black_box(&fix.edges_v),
                black_box(&fix.csr_row_ptr),
                black_box(&fix.csr_col_idx),
                black_box(&fix.self_keys_u),
                black_box(&fix.self_keys_v),
                black_box(&fix.self_tuple_idx),
                BuildOpts::default(),
            ).unwrap();
            black_box(r)
        });
    });

    grp.bench_function("parallel", |b| {
        b.iter(|| {
            let r = build_edge_incidence(
                black_box(&fix.edges_u),
                black_box(&fix.edges_v),
                black_box(&fix.csr_row_ptr),
                black_box(&fix.csr_col_idx),
                black_box(&fix.self_keys_u),
                black_box(&fix.self_keys_v),
                black_box(&fix.self_tuple_idx),
                BuildOpts { parallel: true, ..BuildOpts::default() },
            ).unwrap();
            black_box(r)
        });
    });

    // Bitset path: gate at slightly above n_tuples so dispatch fires.
    let bt = (n_tuples + 64).max(64);
    grp.bench_function("bitset_serial", |b| {
        b.iter(|| {
            let r = build_edge_incidence(
                black_box(&fix.edges_u),
                black_box(&fix.edges_v),
                black_box(&fix.csr_row_ptr),
                black_box(&fix.csr_col_idx),
                black_box(&fix.self_keys_u),
                black_box(&fix.self_keys_v),
                black_box(&fix.self_tuple_idx),
                BuildOpts {
                    bitset_threshold: bt,
                    ..BuildOpts::default()
                },
            ).unwrap();
            black_box(r)
        });
    });

    grp.bench_function("bitset_parallel", |b| {
        b.iter(|| {
            let r = build_edge_incidence(
                black_box(&fix.edges_u),
                black_box(&fix.edges_v),
                black_box(&fix.csr_row_ptr),
                black_box(&fix.csr_col_idx),
                black_box(&fix.self_keys_u),
                black_box(&fix.self_keys_v),
                black_box(&fix.self_tuple_idx),
                BuildOpts {
                    parallel: true,
                    bitset_threshold: bt,
                    ..BuildOpts::default()
                },
            ).unwrap();
            black_box(r)
        });
    });

    grp.bench_function("parallel_csr", |b| {
        b.iter(|| {
            let r = build_edge_incidence(
                black_box(&fix.edges_u),
                black_box(&fix.edges_v),
                black_box(&fix.csr_row_ptr),
                black_box(&fix.csr_col_idx),
                black_box(&fix.self_keys_u),
                black_box(&fix.self_keys_v),
                black_box(&fix.self_tuple_idx),
                BuildOpts {
                    parallel: true,
                    output: IncidenceOutput::Csr,
                    ..BuildOpts::default()
                },
            ).unwrap();
            black_box(r)
        });
    });

    grp.finish();
}

fn bench_incidence(c: &mut Criterion) {
    // Three scales:
    //   small  — 200 V × 3.5k T  (Bitcoin Alpha k=4 shape)
    //   mid    — 1k V × 25k T    (Bitcoin OTC k=4)
    //   large  — 4k V × 100k T   (Slashdot k=4 shape, scaled to fit CI)
    bench_one_size(c, "small",  200,   3_500);
    bench_one_size(c, "mid",   1_000, 25_000);
    bench_one_size(c, "large", 4_000, 100_000);
}

criterion_group!(benches, bench_incidence);
criterion_main!(benches);
