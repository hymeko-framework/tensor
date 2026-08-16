//! BFS distance precompute (smallest-vertex-root domain).

use super::csr::neighbours;

// ============================================================================
// BFS distance precompute, smallest-vertex-root domain.
//
// For undirected enumeration the DFS only descends through vertices >= start.
// We do a single-source BFS from `start` restricted to that subgraph and
// store `dist[v]` (capped at u8::MAX = unreachable). The exact DFS then
// drops any candidate `nxt` with dist[nxt] > k - path.len(), because that
// candidate cannot reach `start` in the remaining hops to close a k-cycle.
//
// Only sound for the undirected case: a directed cycle needs an in-edge
// to start, so out-BFS from start is the wrong distance. The directed
// path skips BFS pruning (dist passed in as an empty slice).
//
// Caller-provided `dist` scratch buffer is reset to u8::MAX inside; reuse
// across starts within a worker avoids the per-start allocation that
// otherwise dominates allocator pressure on big graphs.
// ============================================================================

#[allow(clippy::too_many_arguments)]
// BFS scratch buffers are caller-owned for reuse across starts; flattening
// into a struct would only add indirection on the hot loop boundary.
pub fn bfs_distances_into(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    n_nodes: usize,
    max_dist: u8,
    dist: &mut [u8],
    frontier: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    debug_assert_eq!(dist.len(), n_nodes);
    dist.fill(u8::MAX);
    dist[start as usize] = 0;
    frontier.clear();
    next.clear();
    frontier.push(start);
    let mut d: u8 = 0;
    while !frontier.is_empty() && d < max_dist {
        for &u in frontier.iter() {
            for &w in neighbours(row_ptr, col_idx, u) {
                if w < start { continue; }
                if dist[w as usize] != u8::MAX { continue; }
                dist[w as usize] = d + 1;
                next.push(w);
            }
        }
        std::mem::swap(frontier, next);
        next.clear();
        d += 1;
    }
}

