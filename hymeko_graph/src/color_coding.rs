//! Color-coding sampler for undirected k-cycles.
//!
//! `ColorCodingSampler` implements `CycleSampler`. Each "batch" is one
//! coloring; the inner DFS only extends to vertices with previously-unused
//! colors, which by Alon-Yuster-Zwick yields each k-cycle with probability
//! `k! / k^k`. The driver runs many colorings in parallel via rayon and
//! collects unique canonical cycles.
//!
//! Moved out of `hymeko_py/src/cycles.rs` 2026-05-11 per CLAUDE.md §6.5 #2.
//!
//! Rustdoc for the sampler struct is deferred: the stable contract is
//! `CycleSampler` + the PyO3 Strategy wrappers.

#![allow(missing_docs)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cycle_sampler::{
    canonical_cycle, enumerate_par, lcg_for_batch, CycleSampler, SamplerScratch,
};
use crate::rand_lcg::Lcg;
use crate::unsigned_cycles::{build_csr, has_edge, neighbours};

pub struct ColorCodingSampler<'a> {
    pub row_ptr: &'a [u32],
    pub col_idx: &'a [u32],
    pub n_nodes: usize,
    pub k: usize,
}

impl<'a> CycleSampler for ColorCodingSampler<'a> {
    fn run_batch(
        &self,
        batch_idx: usize,
        seed_base: u64,
        scratch: &mut SamplerScratch,
        target_reached: &AtomicUsize,
        target_cycles: usize,
    ) {
        let mut rng = lcg_for_batch(seed_base, batch_idx);
        let colors = random_coloring(self.n_nodes, self.k, &mut rng);
        let mut used: u32 = 0;
        for start in 0..self.n_nodes as u32 {
            dfs_color_coded(
                self.row_ptr,
                self.col_idx,
                start,
                self.k,
                &colors,
                &mut scratch.visited,
                &mut scratch.path,
                &mut used,
                &mut scratch.local_out,
            );
            if start.is_multiple_of(512)
                && target_reached.load(Ordering::Relaxed) >= target_cycles
            {
                break;
            }
        }
    }
}

/// Top-level entry: build CSR, drive the sampler, return flat (n_kept * k) buf.
#[allow(clippy::too_many_arguments)]
pub fn enumerate_color_coded(
    edges_u: &[u32],
    edges_v: &[u32],
    n_nodes: usize,
    k: usize,
    target_cycles: usize,
    seed: u64,
    max_colorings: Option<usize>,
    n_threads: Option<usize>,
) -> Vec<u32> {
    let edges: Vec<(u32, u32)> =
        edges_u.iter().copied().zip(edges_v.iter().copied()).collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, false);

    // Default budget: ~5x coverage at the k!/k^k rainbow rate.
    let kf: u64 = (1..=k as u64).product();
    let kk: u64 = (k as u64).pow(k as u32);
    let cov_factor: u64 = (kk + kf - 1) / kf.max(1);
    let n_batches = max_colorings.unwrap_or((cov_factor as usize) * 5).max(1);

    let sampler = ColorCodingSampler {
        row_ptr: &row_ptr,
        col_idx: &col_idx,
        n_nodes,
        k,
    };
    enumerate_par(&sampler, n_nodes, k, target_cycles, n_batches, seed, n_threads)
}

// ─── internals ──────────────────────────────────────────────────────

fn random_coloring(n_nodes: usize, k: usize, rng: &mut Lcg) -> Vec<u8> {
    let k_u64 = k as u64;
    let mut out = vec![0u8; n_nodes];
    for v in out.iter_mut() {
        let r = rng.next_u64() >> 33;
        *v = (r % k_u64) as u8;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs_color_coded(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    colors: &[u8],
    visited: &mut [bool],
    path: &mut Vec<u32>,
    used_colors: &mut u32,
    out: &mut Vec<Vec<u32>>,
) {
    path.push(start);
    visited[start as usize] = true;
    *used_colors |= 1u32 << colors[start as usize];

    recurse(row_ptr, col_idx, start, k, colors, path, visited, used_colors, out);

    path.pop();
    visited[start as usize] = false;
    *used_colors &= !(1u32 << colors[start as usize]);
}

#[allow(clippy::too_many_arguments)]
fn recurse(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    colors: &[u8],
    path: &mut Vec<u32>,
    visited: &mut [bool],
    used_colors: &mut u32,
    out: &mut Vec<Vec<u32>>,
) {
    if path.len() == k {
        let last = *path.last().expect("path non-empty inside recurse");
        if has_edge(row_ptr, col_idx, last, start) {
            out.push(canonical_cycle(path));
        }
        return;
    }
    let tail = *path.last().expect("path non-empty inside recurse");
    for &nxt in neighbours(row_ptr, col_idx, tail) {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        let c = colors[nxt as usize];
        let bit = 1u32 << c;
        if *used_colors & bit != 0 {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        *used_colors |= bit;
        recurse(
            row_ptr, col_idx, start, k, colors, path, visited, used_colors, out,
        );
        path.pop();
        visited[nxt as usize] = false;
        *used_colors &= !bit;
    }
}
