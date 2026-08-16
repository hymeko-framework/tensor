//! Generic parallel sampler driver for k-cycle enumeration.
//!
//! Polymorphism + reusability per CLAUDE.md §6.5 + paradigm hierarchy
//! (trait/struct-based programming). Three concrete cycle samplers
//! (color-coding, path-closure, …) share one parallel/dedup orchestrator;
//! previously these were three near-identical 80-LOC free-function bodies.
//!
//! The `CycleSampler` trait defines the per-worker batch step. The
//! `enumerate_par` driver owns:
//!   * rayon thread-pool selection
//!   * per-thread scratch buffers
//!   * lock-free DashSet dedup of canonical cycles
//!   * early-stop on hitting the target count
//!
//! Concrete samplers live in their own modules (color-coding, path-closure)
//! and only implement the algorithm-specific inner step.
//!
//! `SamplerScratch` fields are self-explanatory buffer slots; rustdoc
//! duplication would mirror the struct field names verbatim.

#![allow(missing_docs)]

use dashmap::DashSet;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::rand_lcg::Lcg;

/// Thread-local scratch for a sampler. Avoids per-attempt heap churn.
pub struct SamplerScratch {
    pub visited: Vec<bool>,
    pub path: Vec<u32>,
    pub local_out: Vec<Vec<u32>>,
}

impl SamplerScratch {
    pub fn new(n_nodes: usize, k: usize) -> Self {
        Self {
            visited: vec![false; n_nodes],
            path: Vec::with_capacity(k),
            local_out: Vec::new(),
        }
    }
}

/// One concrete cycle-sampling strategy.
///
/// `run_batch` is invoked many times in parallel (one per rayon work-unit).
/// Each call should produce zero or more candidate cycles, pushed into
/// `scratch.local_out` in canonical form. The driver handles dedup and
/// the global cycle-count cap.
///
/// Implementors are responsible for canonicalising emitted cycles (so
/// dedup actually deduplicates rotations / orientations).
pub trait CycleSampler: Sync {
    /// Per-thread inner work. `batch_idx` is the unique worker index
    /// (rayon job number); used to derive an independent RNG stream.
    fn run_batch(
        &self,
        batch_idx: usize,
        seed_base: u64,
        scratch: &mut SamplerScratch,
        target_reached: &AtomicUsize,
        target_cycles: usize,
    );
}

/// Drive a `CycleSampler` to produce a flat `Vec<u32>` of up to
/// `target_cycles` unique canonical cycles, returned as length `k * n_out`.
///
/// `n_batches` is the maximum number of rayon work units to dispatch
/// (e.g. number of colorings, or `ceil(max_attempts / chunk_size)`).
pub fn enumerate_par<S: CycleSampler>(
    sampler: &S,
    n_nodes: usize,
    k: usize,
    target_cycles: usize,
    n_batches: usize,
    seed: u64,
    n_threads: Option<usize>,
) -> Vec<u32> {
    let pool = n_threads.and_then(|nt| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt.max(1))
            .build()
            .ok()
    });

    let dedup: DashSet<Vec<u32>> = DashSet::with_capacity(target_cycles * 2);
    let total_kept = AtomicUsize::new(0);

    let do_work = || {
        (0..n_batches).into_par_iter().for_each(|batch_idx| {
            if total_kept.load(Ordering::Relaxed) >= target_cycles {
                return;
            }
            let mut scratch = SamplerScratch::new(n_nodes, k);
            sampler.run_batch(
                batch_idx, seed, &mut scratch, &total_kept, target_cycles,
            );
            for cyc in scratch.local_out.drain(..) {
                if total_kept.load(Ordering::Relaxed) >= target_cycles {
                    break;
                }
                if dedup.insert(cyc) {
                    total_kept.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    };
    if let Some(p) = pool {
        p.install(do_work);
    } else {
        do_work();
    }

    let mut flat: Vec<u32> = Vec::with_capacity(target_cycles * k);
    for cyc in dedup.into_iter().take(target_cycles) {
        flat.extend_from_slice(&cyc);
    }
    flat
}

/// Canonical form of an undirected k-cycle: rotate so the smallest vertex
/// is first, then choose the lex-smaller of forward / reverse traversal.
/// Stable across the dedup set — used by every sampler.
pub fn canonical_cycle(path: &[u32]) -> Vec<u32> {
    let k = path.len();
    let min_pos = (0..k).min_by_key(|&i| path[i]).expect("non-empty");
    let forward: Vec<u32> = (0..k).map(|i| path[(min_pos + i) % k]).collect();
    let reverse: Vec<u32> = std::iter::once(forward[0])
        .chain((1..k).map(|i| forward[k - i]))
        .collect();
    if forward < reverse { forward } else { reverse }
}

/// Convenience: derive an independent per-batch RNG seed.
///
/// Uses the golden-ratio mix used by both legacy samplers to spread
/// `batch_idx` across the `u64` state space, then re-mixes with `Lcg::new`'s
/// kickstart.
#[inline]
pub fn lcg_for_batch(seed_base: u64, batch_idx: usize) -> Lcg {
    Lcg::new(
        seed_base
            .wrapping_add(batch_idx as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_picks_min_rotation_lex_smallest_direction() {
        // Two rotations + two directions of the same triangle should
        // canonicalise to the same vector.
        let a = canonical_cycle(&[2, 0, 1]);
        let b = canonical_cycle(&[0, 1, 2]);
        let c = canonical_cycle(&[1, 2, 0]);
        let d = canonical_cycle(&[2, 1, 0]); // reverse
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(c, d);
        assert_eq!(a, vec![0, 1, 2]);
    }
}
