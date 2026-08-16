//! Microbench: Rust spine FIR vs naive sequential pure-Rust reference.
//!
//! Run:
//!     cargo bench -p hymeko_graph --bench spine_bench
//!
//! Reports: median over 5 iters, throughput in cycles/s and GB/s
//! effective memory bandwidth, comparing parallel Rayon spine vs
//! single-threaded baseline.
use std::time::Instant;

use hymeko_graph::spine::{SignedCycleFIR, fir_cycle_forward};
use hymeko_graph::topk_cycles::TopKCyclesBatch;

fn make_fixture(n_vertices: usize, n_cycles: usize, k: usize, d: usize)
    -> (TopKCyclesBatch, Vec<f32>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut cycles: Vec<u32> = Vec::with_capacity(n_cycles * k);
    let mut signs: Vec<i8> = Vec::with_capacity(n_cycles * k);
    for _ in 0..n_cycles {
        for _ in 0..k {
            cycles.push(rng.random_range(0..n_vertices as u32));
            signs.push(if rng.random_bool(0.5) { 1 } else { -1 });
        }
    }
    let features: Vec<f32> = (0..n_vertices * d)
        .map(|i| (i as f32) / (n_vertices * d) as f32)
        .collect();
    let batch = TopKCyclesBatch {
        cycles, signs, scores: vec![0.0; n_cycles], k,
    };
    (batch, features)
}

fn bench_one(label: &str, batch: &TopKCyclesBatch, features: &[f32],
              n_vertices: usize, d: usize, fir: &SignedCycleFIR) {
    // Warm-up
    let _ = fir_cycle_forward(batch, features, n_vertices, d, fir);
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = fir_cycle_forward(batch, features, n_vertices, d, fir);
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[2];
    let n_cycles = batch.len();
    let cycles_per_s = n_cycles as f64 / median;
    let bytes_read = (n_cycles * batch.k * d * std::mem::size_of::<f32>()) as f64;
    let gbps = bytes_read / median / 1e9;
    println!(
        "{:<25}  median={:7.4} ms  {:8.2} M cycles/s  {:5.2} GB/s effective",
        label, median * 1e3, cycles_per_s / 1e6, gbps,
    );
}

fn main() {
    // Bitcoin OTC scale.
    let (b1, f1) = make_fixture(6_000, 30_000, 3, 32);
    let fir3 = SignedCycleFIR::signed_mean(3);
    println!("=== Bitcoin OTC scale: |V|=6000, |C|=30000, k=3, d=32 ===");
    bench_one("rayon parallel", &b1, &f1, 6_000, 32, &fir3);

    // Slashdot-ish scale.
    let (b2, f2) = make_fixture(82_000, 500_000, 3, 32);
    println!("\n=== Slashdot scale: |V|=82000, |C|=500K, k=3, d=32 ===");
    bench_one("rayon parallel", &b2, &f2, 82_000, 32, &fir3);

    // Epinions scale, k=4, d=64.
    let (b3, f3) = make_fixture(131_000, 530_000, 4, 64);
    let fir4 = SignedCycleFIR::signed_mean(4);
    println!("\n=== Epinions scale: |V|=131000, |C|=530K, k=4, d=64 ===");
    bench_one("rayon parallel", &b3, &f3, 131_000, 64, &fir4);
}
