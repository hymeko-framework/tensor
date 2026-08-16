//! Path-closure sampler for undirected k-cycles.
//!
//! `PathClosureSampler` implements `CycleSampler`. Each batch is a chunk
//! of `chunk_size` random-walk attempts; each successful closure (final
//! vertex shares an edge with the start) produces one canonical cycle.
//!
//! Trade-off vs color-coding (in `crate::color_coding`):
//!   * scales to ANY k (no `k <= 16` cap)
//!   * biased toward low-degree-vertex cycles
//!   * fastest single-cycle generator on dense, low-arity graphs
//!
//! Moved out of `hymeko_py/src/cycles.rs` 2026-05-11 per CLAUDE.md §6.5 #2.
//!
//! Field-level docs live on the `CycleSampler` impl in PyO3 glue for now.

#![allow(missing_docs)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cycle_sampler::{
    canonical_cycle, enumerate_par, lcg_for_batch, CycleSampler, SamplerScratch,
};
use crate::rand_lcg::Lcg;
use crate::unsigned_cycles::{build_csr, has_edge, neighbours};

/// One walk-and-test attempt batch.
pub struct PathClosureSampler<'a> {
    pub row_ptr: &'a [u32],
    pub col_idx: &'a [u32],
    pub n_nodes: usize,
    pub k: usize,
    pub chunk_size: usize,
    pub max_attempts: usize,
}

impl<'a> CycleSampler for PathClosureSampler<'a> {
    fn run_batch(
        &self,
        batch_idx: usize,
        seed_base: u64,
        scratch: &mut SamplerScratch,
        target_reached: &AtomicUsize,
        target_cycles: usize,
    ) {
        let mut rng = lcg_for_batch(seed_base, batch_idx);
        let lo = batch_idx * self.chunk_size;
        let hi = ((batch_idx + 1) * self.chunk_size).min(self.max_attempts);
        for _ in lo..hi {
            if target_reached.load(Ordering::Relaxed) >= target_cycles {
                break;
            }
            if let Some(cyc) = try_one_walk(
                self.row_ptr,
                self.col_idx,
                self.n_nodes,
                self.k,
                &mut rng,
                &mut scratch.visited,
                &mut scratch.path,
            ) {
                scratch.local_out.push(cyc);
            }
        }
    }
}

/// Top-level: build CSR, drive the sampler, return flat buf.
#[allow(clippy::too_many_arguments)]
pub fn enumerate_path_closure(
    edges_u: &[u32],
    edges_v: &[u32],
    n_nodes: usize,
    k: usize,
    target_cycles: usize,
    seed: u64,
    max_attempts: Option<usize>,
    n_threads: Option<usize>,
) -> Vec<u32> {
    let edges: Vec<(u32, u32)> =
        edges_u.iter().copied().zip(edges_v.iter().copied()).collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, false);

    let max_att = max_attempts
        .unwrap_or(target_cycles * 50)
        .max(target_cycles);

    let chunk_size = 1024usize;
    let n_batches = max_att.div_ceil(chunk_size);

    let sampler = PathClosureSampler {
        row_ptr: &row_ptr,
        col_idx: &col_idx,
        n_nodes,
        k,
        chunk_size,
        max_attempts: max_att,
    };
    enumerate_par(&sampler, n_nodes, k, target_cycles, n_batches, seed, n_threads)
}

/// One walk attempt — Some(canonical) on closure, None on dead-end / revisit
/// stuck / non-closing.
fn try_one_walk(
    row_ptr: &[u32],
    col_idx: &[u32],
    n_nodes: usize,
    k: usize,
    rng: &mut Lcg,
    visited: &mut [bool],
    path: &mut Vec<u32>,
) -> Option<Vec<u32>> {
    debug_assert!(visited.iter().all(|&b| !b));
    debug_assert!(path.is_empty());
    let start = rng.next_in_range(n_nodes as u32);
    path.push(start);
    visited[start as usize] = true;

    for _ in 1..k {
        let tail = *path.last().expect("path non-empty");
        let nbrs = neighbours(row_ptr, col_idx, tail);
        if nbrs.is_empty() {
            reset(visited, path);
            return None;
        }
        let mut chosen: Option<u32> = None;
        for _ in 0..8 {
            let idx = rng.next_in_range(nbrs.len() as u32);
            let cand = nbrs[idx as usize];
            if !visited[cand as usize] {
                chosen = Some(cand);
                break;
            }
        }
        let nxt = match chosen {
            Some(v) => v,
            None => {
                reset(visited, path);
                return None;
            }
        };
        path.push(nxt);
        visited[nxt as usize] = true;
    }
    let last = *path.last().expect("path non-empty");
    let cyc = if has_edge(row_ptr, col_idx, last, start) {
        Some(canonical_cycle(path))
    } else {
        None
    };
    reset(visited, path);
    cyc
}

#[inline]
fn reset(visited: &mut [bool], path: &mut Vec<u32>) {
    for &v in path.iter() {
        visited[v as usize] = false;
    }
    path.clear();
}
