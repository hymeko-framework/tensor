//! Single-threaded DFS over CSR adjacency for exact k-cycle enum.

use super::csr::{bs_clear, bs_get, bs_set, has_edge, neighbours};
use super::sink::Sink;

#[allow(clippy::too_many_arguments)]
// DFS hot path: flat CSR buffers match the legacy enumerator for zero-cost
// inlining; outer PyO3 entry points use config structs.
pub fn dfs_recurse(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    directed: bool,
    path: &mut Vec<u32>,
    visited: &mut [u64],
    dist: &[u8],
    sink: &mut Sink,
) -> bool {
    if path.len() == k {
        let last = *path.last().unwrap();
        if has_edge(row_ptr, col_idx, last, start) && (directed || path[1] < path[k - 1]) {
            return sink.offer(path);
        }
        return true;
    }
    let tail = *path.last().unwrap();
    let max_remaining = (k - path.len()) as u8;
    let prune = !dist.is_empty();
    for &nxt in neighbours(row_ptr, col_idx, tail) {
        if nxt < start { continue; }
        if bs_get(visited, nxt) { continue; }
        if prune && dist[nxt as usize] > max_remaining { continue; }
        path.push(nxt);
        bs_set(visited, nxt);
        let cont = dfs_recurse(row_ptr, col_idx, start, k, directed,
                                 path, visited, dist, sink);
        path.pop();
        bs_clear(visited, nxt);
        if !cont { return false; }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub fn dfs_from(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    directed: bool,
    visited: &mut [u64],
    path: &mut Vec<u32>,
    dist: &[u8],
    sink: &mut Sink,
) -> bool {
    // Returns false if the sink signalled "full, stop the DFS".
    //
    // UNDIRECTED: each cycle (v_0=start, v_1, ..., v_{k-1}) is enumerated
    // twice (forward + reverse from the same root). We deduplicate on the
    // fly by emitting only the orientation with v_1 < v_{k-1}.
    //
    // DIRECTED: each cycle has only one valid traversal direction (out-
    // edges only), so each cycle is emitted exactly once from its
    // smallest-vertex root. No tiebreak needed.
    path.push(start);
    bs_set(visited, start);
    let cont = dfs_recurse(row_ptr, col_idx, start, k, directed,
                             path, visited, dist, sink);
    path.pop();
    bs_clear(visited, start);
    cont
}

/// Run DFS from a (start, first_hop) pair — used as a parallel work unit
/// to expose intra-root parallelism. The outer level for-loop in
/// `enumerate_parallel` sweeps every (start, first_hop > start) so that
/// the heavy-root vertex-0 DFS gets split into deg(0) independent tasks
/// that rayon can distribute.
#[allow(clippy::too_many_arguments)]
pub fn dfs_from_pair(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    first_hop: u32,
    k: usize,
    directed: bool,
    visited: &mut [u64],
    path: &mut Vec<u32>,
    dist: &[u8],
    sink: &mut Sink,
) -> bool {
    path.push(start);
    bs_set(visited, start);
    path.push(first_hop);
    bs_set(visited, first_hop);

    let cont = if k == 2 {
        let last = *path.last().unwrap();
        if has_edge(row_ptr, col_idx, last, start) {
            if directed || path[1] < path[k - 1] {
                sink.offer(path)
            } else { true }
        } else { true }
    } else {
        dfs_recurse(row_ptr, col_idx, start, k, directed,
                      path, visited, dist, sink)
    };

    bs_clear(visited, first_hop);
    path.pop();
    bs_clear(visited, start);
    path.pop();
    cont
}

