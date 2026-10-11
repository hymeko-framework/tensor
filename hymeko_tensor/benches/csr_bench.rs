//! CSR SpMV / SpMM benchmarks.
//!
//! Workloads come from `hymeko_hnn::workloads` (deterministic, seeded, shared
//! with the hypergraph benches and the regression tests) so a saved baseline
//! stays comparable across sessions and hosts.
//!
//! Measurement discipline (CLAUDE.md §3):
//!   * matrix and input vectors are built **outside** the timed closure;
//!   * every input crosses `black_box` so LLVM cannot hoist the call;
//!   * `Throughput::Elements` is declared per group — nnz for SpMV,
//!     `nnz * feature_dim` for SpMM — so Criterion reports elem/s next to the
//!     wall time and the scales are directly comparable;
//!   * the large cells drop to `sample_size(10)`; that reduction is explicit
//!     here rather than silently applied.
//!
//! Coverage note: the SpMM feature-dimension sweep is the full
//! `{1, 8, 32, 128}` at small and medium scale, but only `{1, 32}` at large
//! scale. The large × 128 cell costs ~0.5 s per iteration and adds no
//! information the medium × 128 cell does not already carry. Stating the cap
//! rather than quietly omitting it.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use hymeko_tensor::representations::tensor_csr::TensorCsr;
use hymeko_tensor::sparse_workload::{DegreeShape, SparseSpec};

/// The three measured scales, with a label used in benchmark ids.
fn scales() -> [(&'static str, SparseSpec); 3] {
    [
        ("small", SparseSpec::small()),
        ("medium", SparseSpec::medium()),
        ("large", SparseSpec::large()),
    ]
}

/// Feature dimensions swept per scale. See the coverage note above.
fn feature_dims(scale: &str) -> &'static [usize] {
    match scale {
        "large" => &[1, 32],
        _ => &[1, 8, 32, 128],
    }
}

fn nnz(csr: &TensorCsr<f32>) -> usize {
    csr.val.len()
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_spmv");

    for (label, spec) in scales() {
        let csr = spec.build();
        let x = spec.dense_vector();

        group.throughput(Throughput::Elements(nnz(&csr) as u64));
        if label == "large" {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(8));
        } else {
            group
                .sample_size(100)
                .measurement_time(Duration::from_secs(5));
        }

        group.bench_function(BenchmarkId::new("alloc", label), |b| {
            b.iter(|| black_box(black_box(&csr).spmv(black_box(&x))));
        });
    }

    group.finish();
}

fn bench_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_spmm");

    for (label, spec) in scales() {
        let csr = spec.build();
        let nnz = nnz(&csr);

        for &fd in feature_dims(label) {
            let x = spec.dense_block(fd);

            group.throughput(Throughput::Elements((nnz * fd) as u64));
            if label == "large" || fd >= 128 {
                group
                    .sample_size(10)
                    .measurement_time(Duration::from_secs(8));
            } else {
                group
                    .sample_size(50)
                    .measurement_time(Duration::from_secs(5));
            }

            group.bench_function(BenchmarkId::new("alloc", format!("{label}/fd{fd}")), |b| {
                b.iter(|| black_box(black_box(&csr).spmm(black_box(&x), black_box(fd))));
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Path comparison: explicit serial vs explicit parallel.
//
// This is what fixes `TensorCsr::PAR_WORK_THRESHOLD`. The constant claims to be
// the measured crossover, so it has to actually be measured: the sweep walks
// the work count across three orders of magnitude around the candidate value
// and reports both paths at each point. The threshold is then read off the
// crossing, not guessed.
//
// Work is `nnz` for SpMV and `nnz * feature_dim` for SpMM, matching the
// dispatch rule in `spmv_into` / `spmm_into`.
// ---------------------------------------------------------------------------

/// Work counts bracketing the candidate threshold (65_536).
const WORK_SWEEP: [usize; 7] = [1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304];

/// Build a matrix with approximately `work` non-zeros at a fixed row degree.
fn spec_for_nnz(nnz: usize) -> SparseSpec {
    const DEG: usize = 8;
    let rows = (nnz / DEG).max(1);
    SparseSpec {
        rows,
        cols: rows,
        avg_nnz_per_row: DEG,
        shape: DegreeShape::Uniform,
        seed: 0x5EED_7777,
    }
}

fn bench_spmv_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_spmv_paths");
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(5));

    for work in WORK_SWEEP {
        let spec = spec_for_nnz(work);
        let csr = spec.build();
        let x = spec.dense_vector();
        let mut y = vec![0.0f32; csr.num_rows];

        group.throughput(Throughput::Elements(nnz(&csr) as u64));

        group.bench_function(BenchmarkId::new("serial", work), |b| {
            b.iter(|| black_box(&csr).spmv_into_serial(black_box(&x), black_box(&mut y)));
        });
        group.bench_function(BenchmarkId::new("parallel", work), |b| {
            b.iter(|| black_box(&csr).spmv_into_parallel(black_box(&x), black_box(&mut y)));
        });
    }

    group.finish();
}

fn bench_spmm_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_spmm_paths");
    group
        .sample_size(30)
        .measurement_time(Duration::from_secs(5));

    // Hold nnz fixed and sweep feature_dim: this checks that the *work*-based
    // rule (nnz x fd) is the right dispatch variable, not nnz alone.
    let spec = spec_for_nnz(16_384);
    let csr = spec.build();

    for fd in [1usize, 4, 8, 32, 128] {
        let x = spec.dense_block(fd);
        let mut y = vec![0.0f32; csr.num_rows * fd];

        group.throughput(Throughput::Elements((nnz(&csr) * fd) as u64));

        group.bench_function(BenchmarkId::new("serial", format!("nnz16k/fd{fd}")), |b| {
            b.iter(|| {
                black_box(&csr).spmm_into_serial(black_box(&x), black_box(fd), black_box(&mut y))
            });
        });
        group.bench_function(
            BenchmarkId::new("parallel", format!("nnz16k/fd{fd}")),
            |b| {
                b.iter(|| {
                    black_box(&csr).spmm_into_parallel(
                        black_box(&x),
                        black_box(fd),
                        black_box(&mut y),
                    )
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Allocation cost: the allocating wrapper vs the *_into form on a hoisted
// buffer. Reported separately from kernel speed so "allocation removed" is
// never mistaken for "kernel got faster" (a risk the plan called out).
// ---------------------------------------------------------------------------

fn bench_alloc_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_alloc_overhead");

    for (label, spec) in scales() {
        let csr = spec.build();
        let x = spec.dense_vector();
        let mut y = vec![0.0f32; csr.num_rows];

        group.throughput(Throughput::Elements(nnz(&csr) as u64));
        if label == "large" {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(8));
        } else {
            group
                .sample_size(100)
                .measurement_time(Duration::from_secs(5));
        }

        group.bench_function(BenchmarkId::new("spmv_alloc", label), |b| {
            b.iter(|| black_box(black_box(&csr).spmv(black_box(&x))));
        });
        group.bench_function(BenchmarkId::new("spmv_into", label), |b| {
            b.iter(|| black_box(&csr).spmv_into(black_box(&x), black_box(&mut y)));
        });
    }

    group.finish();
}
criterion_group!(
    benches,
    bench_spmv,
    bench_spmm,
    bench_spmv_paths,
    bench_spmm_paths,
    bench_alloc_overhead
);
criterion_main!(benches);
