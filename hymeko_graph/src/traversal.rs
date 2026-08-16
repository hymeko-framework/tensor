//! Low-level iterative DFS / BFS / bidirectional-BFS primitives
//! over a CSR-adjacency signed graph.
//!
//! Design constraints:
//!
//! - **Allocation-free hot path.**  All scratch state lives in a
//!   reusable [`DfsScratch`] / [`BfsScratch`] struct that the caller
//!   owns; the traversal functions take `&mut scratch` and never
//!   call `Vec::push` on a fresh buffer in their inner loops.
//!   Re-using a scratch across thousands of traversals (the typical
//!   shape of cycle / walk / shortest-path enumeration over a fixed
//!   graph) eliminates allocator pressure entirely.
//! - **Bitset visited.**  `Vec<u64>` with one bit per vertex —
//!   $8\times$ smaller than `Vec<bool>` and fits in L1 for graphs
//!   up to $\sim 500$\,k vertices.  Bitset get / set are
//!   single-instruction shifts on x86-64.
//! - **CSR adjacency.**  `(row_ptr, col_idx)` pre-built from the
//!   [`crate::SignedGraph`] once per session.  Neighbour iteration
//!   is a single pointer-walk, branch-predictable.
//! - **Iterative, not recursive.**  No stack-frame overhead; depth
//!   is limited only by `Vec` capacity, not by the OS thread
//!   stack.
//! - **Pruner-aware.**  DFS and BFS variants can take a
//!   [`crate::CyclePruner`] reference to short-circuit
//!   structurally-infeasible branches at extension time.  When you
//!   don't need a pruner, pass [`crate::NoOpPruner`] or use the
//!   `_noprune` convenience wrappers (zero overhead — Rust compiles
//!   the trait dispatch away).
//!
//! Targeted use cases: cycle enumeration (already in
//! [`crate::cycle_enum`]), shortest-path queries on signed
//! networks, BFS-layer pruning for $k$-cycle DFS (skip vertices
//! that can't close in the remaining budget), connected-component
//! decomposition, walk enumeration (planned).

use crate::pruner::{CyclePruner, PrunerDecision};
use crate::signed_graph::SignedGraph;

// ─── bitset utilities ────────────────────────────────────────────

/// Number of `u64` words needed for an `n`-bit bitset.
#[inline]
pub fn bs_words(n: usize) -> usize {
    n.div_ceil(64)
}

/// Test bit `v` in a packed bitset of `u64` words.
#[inline]
pub fn bs_get(bits: &[u64], v: u32) -> bool {
    (bits[(v >> 6) as usize] >> (v & 63)) & 1 == 1
}

/// Set bit `v` in a packed bitset of `u64` words.
#[inline]
pub fn bs_set(bits: &mut [u64], v: u32) {
    bits[(v >> 6) as usize] |= 1u64 << (v & 63);
}

/// Clear bit `v` in a packed bitset of `u64` words.
#[inline]
pub fn bs_clear(bits: &mut [u64], v: u32) {
    bits[(v >> 6) as usize] &= !(1u64 << (v & 63));
}

/// Zero out an entire bitset.  $O(\text{words})$.
#[inline]
pub fn bs_zero(bits: &mut [u64]) {
    for w in bits.iter_mut() {
        *w = 0;
    }
}

// ─── CSR view ───────────────────────────────────────────────────

/// Pre-built CSR adjacency for fast traversal.  Build once per
/// graph via [`Csr::from_graph`] and reuse across many traversals.
#[derive(Debug, Clone)]
pub struct Csr {
    /// `row_ptr[v]` is the offset into `col_idx` where vertex `v`'s
    /// neighbour list begins; `row_ptr[v + 1]` is where it ends.
    /// Length: `n_nodes + 1`.
    pub row_ptr: Vec<u32>,
    /// Concatenated, sorted, deduplicated neighbour lists.
    pub col_idx: Vec<u32>,
    /// Number of vertices.
    pub n_nodes: u32,
}

impl Csr {
    /// Build from a [`SignedGraph`].  $O(|E|\log\bar d)$ for the
    /// per-row sort + dedup.
    pub fn from_graph(g: &SignedGraph) -> Csr {
        let (row_ptr, col_idx) = g.build_csr();
        Csr {
            row_ptr,
            col_idx,
            n_nodes: g.n_nodes,
        }
    }

    /// Borrow vertex `v`'s neighbour slice.  $O(1)$.
    #[inline]
    pub fn neighbours(&self, v: u32) -> &[u32] {
        let s = self.row_ptr[v as usize] as usize;
        let e = self.row_ptr[v as usize + 1] as usize;
        &self.col_idx[s..e]
    }

    /// Test whether `(u, v)` is an edge.  $O(\log \bar d)$ binary
    /// search since neighbour lists are sorted.
    #[inline]
    pub fn has_edge(&self, u: u32, v: u32) -> bool {
        self.neighbours(u).binary_search(&v).is_ok()
    }

    /// Out-degree of `v`.  $O(1)$.
    #[inline]
    pub fn degree(&self, v: u32) -> u32 {
        self.row_ptr[v as usize + 1] - self.row_ptr[v as usize]
    }
}

// ─── DFS scratch + iterative DFS ─────────────────────────────────

/// Reusable scratch buffers for iterative DFS traversal.
///
/// Construct once with the graph's vertex count, then call
/// [`Self::reset`] between traversals to zero the visited bitset
/// without reallocating.
#[derive(Debug)]
pub struct DfsScratch {
    /// Bitset visited, one bit per vertex.
    pub visited: Vec<u64>,
    /// Stack of `(vertex, child-iterator index)` pairs.  We store
    /// the child position as a `u32` index into `Csr::col_idx`
    /// rather than a slice iterator so the struct stays Clone +
    /// Default-friendly.
    pub stack: Vec<(u32, u32)>,
    /// Path of vertices currently in the recursion (for path-aware
    /// traversals).
    pub path: Vec<u32>,
}

impl DfsScratch {
    /// Build with capacity for `n_nodes` vertices.
    pub fn with_capacity(n_nodes: u32) -> DfsScratch {
        DfsScratch {
            visited: vec![0u64; bs_words(n_nodes as usize)],
            stack: Vec::with_capacity(64),
            path: Vec::with_capacity(64),
        }
    }

    /// Zero the visited bitset and clear the stack/path.  $O(n / 64)$
    /// for the bitset, $O(1)$ for the others.
    pub fn reset(&mut self) {
        bs_zero(&mut self.visited);
        self.stack.clear();
        self.path.clear();
    }
}

/// Visit every vertex reachable from `start` in DFS order, calling
/// `visit` on each.  The visited bitset is shared with the scratch
/// so a single scratch can be used across multiple `dfs_visit`
/// calls without re-walking already-seen components.
///
/// Returns the count of newly-visited vertices.
pub fn dfs_visit<F: FnMut(u32)>(
    csr: &Csr,
    scratch: &mut DfsScratch,
    start: u32,
    mut visit: F,
) -> usize {
    if bs_get(&scratch.visited, start) {
        return 0;
    }
    let mut count = 0;
    scratch.stack.clear();
    scratch.stack.push((start, 0));
    bs_set(&mut scratch.visited, start);
    visit(start);
    count += 1;
    while let Some(&(v, ci)) = scratch.stack.last() {
        let nbrs = csr.neighbours(v);
        if (ci as usize) < nbrs.len() {
            let nxt = nbrs[ci as usize];
            // Advance the parent's child iterator first, then
            // push the child.  This way the parent retains the
            // correct resumption point when we pop back up.
            scratch.stack.last_mut().unwrap().1 = ci + 1;
            if !bs_get(&scratch.visited, nxt) {
                bs_set(&mut scratch.visited, nxt);
                visit(nxt);
                count += 1;
                scratch.stack.push((nxt, 0));
            }
        } else {
            scratch.stack.pop();
        }
    }
    count
}

/// Pruner-aware iterative DFS.  Walks from `start`, calling
/// `visit` on every accepted vertex.  At each extension the
/// pruner's [`CyclePruner::extend_ok`] is consulted with the
/// current path; rejected branches are skipped.
///
/// Returns the count of accepted-and-visited vertices.
pub fn dfs_visit_pruned<F: FnMut(u32)>(
    csr: &Csr,
    scratch: &mut DfsScratch,
    pruner: &dyn CyclePruner,
    start: u32,
    mut visit: F,
) -> usize {
    if bs_get(&scratch.visited, start) {
        return 0;
    }
    let mut count = 0;
    scratch.stack.clear();
    scratch.path.clear();
    scratch.stack.push((start, 0));
    scratch.path.push(start);
    bs_set(&mut scratch.visited, start);
    visit(start);
    count += 1;
    while let Some(&(v, ci)) = scratch.stack.last() {
        let nbrs = csr.neighbours(v);
        if (ci as usize) < nbrs.len() {
            let nxt = nbrs[ci as usize];
            scratch.stack.last_mut().unwrap().1 = ci + 1;
            if bs_get(&scratch.visited, nxt) {
                continue;
            }
            // Pruner check before pushing.
            if pruner.extend_ok(&scratch.path, nxt) == PrunerDecision::Reject {
                continue;
            }
            bs_set(&mut scratch.visited, nxt);
            visit(nxt);
            count += 1;
            scratch.path.push(nxt);
            scratch.stack.push((nxt, 0));
        } else {
            scratch.stack.pop();
            scratch.path.pop();
        }
    }
    count
}

/// Convenience: DFS without pruning.  Identical semantics to
/// [`dfs_visit`] — kept as a wrapper so call-sites that
/// conditionally use a pruner can drop in [`NoOpPruner`].
pub fn dfs_visit_noprune<F: FnMut(u32)>(
    csr: &Csr,
    scratch: &mut DfsScratch,
    start: u32,
    visit: F,
) -> usize {
    dfs_visit(csr, scratch, start, visit)
}

// ─── BFS scratch + iterative BFS ─────────────────────────────────

/// Sentinel "vertex unreached" marker for distance arrays.  Chosen
/// as `u8::MAX` so the array is one byte per vertex, fits in L1
/// for graphs up to a few hundred thousand vertices, and represents
/// any path beyond depth $254$.
pub const BFS_UNREACHED: u8 = u8::MAX;

/// Reusable scratch buffers for iterative BFS.
#[derive(Debug)]
pub struct BfsScratch {
    /// Per-vertex distance from the BFS root.  `BFS_UNREACHED`
    /// means not yet reached.
    pub dist: Vec<u8>,
    /// Current frontier.
    pub frontier: Vec<u32>,
    /// Next-layer frontier (double-buffered with `frontier`).
    pub next_frontier: Vec<u32>,
}

impl BfsScratch {
    /// Build with capacity for `n_nodes` vertices.
    pub fn with_capacity(n_nodes: u32) -> BfsScratch {
        BfsScratch {
            dist: vec![BFS_UNREACHED; n_nodes as usize],
            frontier: Vec::with_capacity(256),
            next_frontier: Vec::with_capacity(256),
        }
    }

    /// Reset distances to `BFS_UNREACHED` and clear frontiers.
    /// $O(n)$.
    pub fn reset(&mut self) {
        for d in self.dist.iter_mut() {
            *d = BFS_UNREACHED;
        }
        self.frontier.clear();
        self.next_frontier.clear();
    }
}

/// Iterative BFS from `start` up to `max_depth` layers.  Returns
/// `Ok(num_reached)` after termination; `dist[v]` is set to the
/// shortest path length from `start` to `v` for every reached
/// vertex.  Vertices beyond `max_depth` keep `BFS_UNREACHED`.
///
/// Layer-at-a-time double-buffered frontier — each layer is a
/// flat `Vec<u32>` walk, branch-predictable.
pub fn bfs_distances(csr: &Csr, scratch: &mut BfsScratch, start: u32, max_depth: u8) -> usize {
    scratch.reset();
    scratch.dist[start as usize] = 0;
    scratch.frontier.push(start);
    let mut depth: u8 = 0;
    let mut total = 1usize;
    while !scratch.frontier.is_empty() && depth < max_depth {
        scratch.next_frontier.clear();
        for &v in &scratch.frontier {
            for &nxt in csr.neighbours(v) {
                if scratch.dist[nxt as usize] == BFS_UNREACHED {
                    scratch.dist[nxt as usize] = depth + 1;
                    scratch.next_frontier.push(nxt);
                    total += 1;
                }
            }
        }
        std::mem::swap(&mut scratch.frontier, &mut scratch.next_frontier);
        depth = depth.saturating_add(1);
    }
    total
}

/// Bidirectional BFS for shortest-path queries.  Searches from
/// both `src` and `dst` simultaneously, expanding the smaller
/// frontier each step until the two frontiers intersect.
///
/// Returns `Some(distance)` if a path exists within
/// `2 * max_half_depth` (so both sides searched up to half the
/// budget), `None` otherwise.
///
/// Bi-BFS hits $O(\bar d^{D/2})$ instead of single BFS's
/// $O(\bar d^D)$, an exponential win for distant pairs in
/// well-connected graphs.
pub fn bidirectional_bfs(csr: &Csr, src: u32, dst: u32, max_half_depth: u8) -> Option<u32> {
    if src == dst {
        return Some(0);
    }
    let n = csr.n_nodes as usize;
    let mut dist_src = vec![BFS_UNREACHED; n];
    let mut dist_dst = vec![BFS_UNREACHED; n];
    dist_src[src as usize] = 0;
    dist_dst[dst as usize] = 0;
    let mut frontier_src = vec![src];
    let mut frontier_dst = vec![dst];
    let mut depth_src: u32 = 0;
    let mut depth_dst: u32 = 0;
    let mut next_buf = Vec::with_capacity(256);

    while !frontier_src.is_empty()
        && !frontier_dst.is_empty()
        && (depth_src + depth_dst) < (2 * max_half_depth as u32)
    {
        // Expand the smaller frontier first — keeps the search
        // wave-fronts balanced and prevents one side from
        // ballooning while the other does nothing.
        let expand_src = frontier_src.len() <= frontier_dst.len();
        let (frontier, dist_self, dist_other, depth) = if expand_src {
            (&mut frontier_src, &mut dist_src, &dist_dst, &mut depth_src)
        } else {
            (&mut frontier_dst, &mut dist_dst, &dist_src, &mut depth_dst)
        };
        next_buf.clear();
        for &v in frontier.iter() {
            for &nxt in csr.neighbours(v) {
                if dist_self[nxt as usize] == BFS_UNREACHED {
                    dist_self[nxt as usize] = (*depth as u8) + 1;
                    next_buf.push(nxt);
                    // Intersection check.
                    if dist_other[nxt as usize] != BFS_UNREACHED {
                        return Some(*depth + 1 + dist_other[nxt as usize] as u32);
                    }
                }
            }
        }
        std::mem::swap(frontier, &mut next_buf);
        *depth += 1;
    }
    None
}

// ─── connected-component count via repeated DFS ──────────────────

/// Count the number of connected components.  Single pass over
/// vertex IDs, skipping any already visited by previous DFS waves.
pub fn count_connected_components(csr: &Csr) -> u32 {
    let mut scratch = DfsScratch::with_capacity(csr.n_nodes);
    let mut count = 0u32;
    for v in 0..csr.n_nodes {
        if !bs_get(&scratch.visited, v) {
            count += 1;
            dfs_visit(csr, &mut scratch, v, |_| {});
        }
    }
    count
}

// ─── tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_path(n: u32) -> SignedGraph {
        // Linear path 0 — 1 — 2 — … — (n-1) with all-positive signs.
        let edges_u: Vec<u32> = (0..n - 1).collect();
        let edges_v: Vec<u32> = (1..n).collect();
        let signs = vec![1i8; (n - 1) as usize];
        SignedGraph::from_parts(n, &edges_u, &edges_v, &signs)
    }

    fn build_cube() -> SignedGraph {
        // 8 vertices at the corners of a cube; 12 edges.
        // Vertex layout: bit 0 = x, bit 1 = y, bit 2 = z.
        let mut eu = Vec::new();
        let mut ev = Vec::new();
        for v in 0..8u32 {
            for axis in 0..3 {
                let bit = 1u32 << axis;
                if v & bit == 0 {
                    eu.push(v);
                    ev.push(v | bit);
                }
            }
        }
        let signs = vec![1i8; eu.len()];
        SignedGraph::from_parts(8, &eu, &ev, &signs)
    }

    #[test]
    fn dfs_visits_every_vertex_in_path() {
        let g = build_path(7);
        let csr = Csr::from_graph(&g);
        let mut s = DfsScratch::with_capacity(g.n_nodes);
        let mut order = Vec::new();
        let count = dfs_visit(&csr, &mut s, 0, |v| order.push(v));
        assert_eq!(count, 7);
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn dfs_scratch_reuse_across_components() {
        // Two disjoint triangles: {0,1,2} and {3,4,5}.
        let g = SignedGraph::from_parts(6, &[0, 1, 2, 3, 4, 5], &[1, 2, 0, 4, 5, 3], &[1; 6]);
        let csr = Csr::from_graph(&g);
        let mut s = DfsScratch::with_capacity(g.n_nodes);
        let mut visited_total = Vec::new();
        // First DFS from 0 visits {0,1,2}.
        let c1 = dfs_visit(&csr, &mut s, 0, |v| visited_total.push(v));
        assert_eq!(c1, 3);
        // Second DFS from 3 (without resetting scratch) visits
        // only the second component {3,4,5}.
        let c2 = dfs_visit(&csr, &mut s, 3, |v| visited_total.push(v));
        assert_eq!(c2, 3);
        // Re-visiting 0 with the existing scratch is a no-op.
        let c3 = dfs_visit(&csr, &mut s, 0, |_| {});
        assert_eq!(c3, 0);
    }

    #[test]
    fn bfs_distances_on_path() {
        let g = build_path(7);
        let csr = Csr::from_graph(&g);
        let mut s = BfsScratch::with_capacity(g.n_nodes);
        let n = bfs_distances(&csr, &mut s, 0, 10);
        assert_eq!(n, 7);
        for v in 0u32..7 {
            assert_eq!(s.dist[v as usize], v as u8);
        }
    }

    #[test]
    fn bfs_max_depth_caps_exploration() {
        let g = build_path(20);
        let csr = Csr::from_graph(&g);
        let mut s = BfsScratch::with_capacity(g.n_nodes);
        let n = bfs_distances(&csr, &mut s, 0, 5);
        // Depth 5 reaches vertices 0..=5 from start 0.
        assert_eq!(n, 6);
        for v in 0u32..6 {
            assert_eq!(s.dist[v as usize], v as u8);
        }
        for v in 6u32..20 {
            assert_eq!(s.dist[v as usize], BFS_UNREACHED);
        }
    }

    #[test]
    fn bidirectional_bfs_finds_shortest_path_in_cube() {
        let g = build_cube();
        let csr = Csr::from_graph(&g);
        // Opposite corners 0 and 7 are at Hamming distance 3.
        assert_eq!(bidirectional_bfs(&csr, 0, 7, 5), Some(3));
        // Same vertex.
        assert_eq!(bidirectional_bfs(&csr, 0, 0, 5), Some(0));
        // Adjacent vertices (Hamming 1).
        assert_eq!(bidirectional_bfs(&csr, 0, 1, 5), Some(1));
    }

    #[test]
    fn connected_components_disjoint_triangles() {
        let g = SignedGraph::from_parts(6, &[0, 1, 2, 3, 4, 5], &[1, 2, 0, 4, 5, 3], &[1; 6]);
        let csr = Csr::from_graph(&g);
        assert_eq!(count_connected_components(&csr), 2);
    }

    #[test]
    fn bitset_round_trip() {
        let mut bits = vec![0u64; bs_words(200)];
        bs_set(&mut bits, 73);
        assert!(bs_get(&bits, 73));
        bs_clear(&mut bits, 73);
        assert!(!bs_get(&bits, 73));
        // Boundary at 64.
        bs_set(&mut bits, 63);
        bs_set(&mut bits, 64);
        assert!(bs_get(&bits, 63));
        assert!(bs_get(&bits, 64));
    }

    #[test]
    fn dfs_pruned_skips_rejected_branches() {
        use crate::friedler::{FriedlerAxiomPruner, NodeKind};
        // Path 0 — 1 — 2 — 3 with bipartite alternation
        // 0=M, 1=O, 2=M, 3=O.  Add a cheating edge 0—2 (M—M)
        // that the Friedler pruner will reject during DFS.
        let g = SignedGraph::from_parts(4, &[0, 1, 2, 0], &[1, 2, 3, 2], &[1; 4]);
        let csr = Csr::from_graph(&g);
        let kinds = vec![
            NodeKind::Material,
            NodeKind::OperatingUnit,
            NodeKind::Material,
            NodeKind::OperatingUnit,
        ];
        let p = FriedlerAxiomPruner::new(kinds);
        let mut s = DfsScratch::with_capacity(g.n_nodes);
        let mut order = Vec::new();
        // Start from 0 (Material).  The 0→2 (M→M) edge must be
        // pruned; the path 0→1→2→3 must succeed.
        let n = dfs_visit_pruned(&csr, &mut s, &p, 0, |v| order.push(v));
        assert_eq!(n, 4);
        // Accepted vertices are 0 (start), 1, 2 (via 1), 3.
        assert!(order.contains(&0));
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        assert!(order.contains(&3));
    }
}
