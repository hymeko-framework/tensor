//! Heuristic-driven traversal: A\* shortest path, best-first DFS,
//! and heuristic-ordered cycle enumeration.
//!
//! All three share the same idea: the **shape** of the search is the
//! same DFS / BFS skeleton you already have in [`crate::traversal`],
//! but at each branching decision a user-supplied [`Heuristic`]
//! reorders or prioritises the next candidates.  When the heuristic
//! is admissible (never overestimates the true cost), A\* returns
//! optimal paths; for cycle enumeration, the heuristic is purely an
//! ordering (it never *prunes* — that's [`crate::pruner`]'s job).
//!
//! ## Pieces
//!
//! - [`Heuristic`] — `h(v) -> f64`, lower is better.
//! - [`ZeroHeuristic`] — the "no info" baseline; A\* with this
//!   collapses to BFS / Dijkstra.
//! - [`DegreeHeuristic`] — high-degree-first (or low-, configurable).
//! - [`astar`] — classical A\* on the CSR.
//! - [`best_first_dfs`] — DFS ordered by `h`; visits the
//!   most-promising neighbour first.
//! - [`enumerate_cycles_ordered`] — cycle DFS with heuristic-ordered
//!   neighbour iteration.
//!
//! ## When ordering matters even though enumeration is total
//!
//! On instances where you want the *first* feasible cycle (or the
//! first time-to-incumbent in a branch-and-bound, à la ABB), the
//! ordering decides how deep DFS goes before it finds a witness.
//! Pair this with a top-$k$ enumerator that bounds by
//! incumbent-best-score and the win is significant — partial paths
//! known to be dominated by the heap minimum get pruned earlier.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::pruner::{CyclePruner, NoOpPruner, PrunerDecision};
use crate::signed_graph::SignedGraph;
use crate::traversal::Csr;

/// Heuristic on vertices: `h(v)` returns a non-negative score; lower
/// is better. Implementations are typically tiny structs that pack
/// per-vertex data (degree, coordinates, distance estimate).
pub trait Heuristic {
    /// The heuristic value at vertex `v`. Lower = more promising.
    fn h(&self, v: u32) -> f64;
}

/// A blanket impl so plain closures work as heuristics.
impl<F: Fn(u32) -> f64> Heuristic for F {
    fn h(&self, v: u32) -> f64 {
        (self)(v)
    }
}

/// "No information" — every vertex is equally good. A\* with this
/// heuristic is just Dijkstra (or BFS in unweighted graphs).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroHeuristic;

impl Heuristic for ZeroHeuristic {
    fn h(&self, _v: u32) -> f64 {
        0.0
    }
}

/// Degree-based heuristic: prefer high-degree (or low-degree)
/// vertices. High-degree-first tends to push DFS through hubs
/// quickly; low-degree-first pulls towards graph peripheries.
#[derive(Debug, Clone)]
pub struct DegreeHeuristic {
    deg: Vec<u32>,
    /// If `true`, prefer **high** degree (smaller `h` for higher deg).
    pub prefer_high: bool,
}

impl DegreeHeuristic {
    /// Build a [`DegreeHeuristic`] from a CSR.
    pub fn new(csr: &Csr, prefer_high: bool) -> Self {
        let deg: Vec<u32> = (0..csr.row_ptr.len() - 1)
            .map(|v| csr.row_ptr[v + 1] - csr.row_ptr[v])
            .collect();
        Self { deg, prefer_high }
    }
}

impl Heuristic for DegreeHeuristic {
    fn h(&self, v: u32) -> f64 {
        let d = self.deg.get(v as usize).copied().unwrap_or(0) as f64;
        if self.prefer_high { -d } else { d }
    }
}

// ─── A* ─────────────────────────────────────────────────────────────

/// Reusable scratch for [`astar`].
#[derive(Debug, Clone, Default)]
pub struct AstarScratch {
    g_score: Vec<f64>,
    came_from: Vec<u32>,
    closed: Vec<bool>,
}

impl AstarScratch {
    /// Construct with capacity for `n` vertices.
    pub fn with_capacity(n: u32) -> Self {
        Self {
            g_score: vec![f64::INFINITY; n as usize],
            came_from: vec![u32::MAX; n as usize],
            closed: vec![false; n as usize],
        }
    }

    /// Reset to `INFINITY` / `u32::MAX` / `false` so the buffers
    /// can be reused for another query without freshly allocating.
    pub fn reset(&mut self) {
        self.g_score.fill(f64::INFINITY);
        self.came_from.fill(u32::MAX);
        self.closed.fill(false);
    }
}

#[derive(Clone, Copy, Debug)]
struct AstarNode {
    f: f64,
    v: u32,
}
impl Eq for AstarNode {}
impl PartialEq for AstarNode {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f && self.v == o.v
    }
}
impl PartialOrd for AstarNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for AstarNode {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reverse on f so BinaryHeap (max-heap) becomes a min-heap.
        o.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal)
    }
}

/// A\* shortest path on an unweighted CSR (every edge has cost 1).
///
/// `h` should be admissible (never overestimate the remaining cost
/// to reach `goal`). With [`ZeroHeuristic`] this is BFS expressed
/// through a priority queue.
///
/// Returns the vertex sequence `start … goal` if reachable within
/// `max_depth`, else `None`.
pub fn astar<H: Heuristic>(
    csr: &Csr,
    s: &mut AstarScratch,
    start: u32,
    goal: u32,
    h: &H,
    max_depth: u32,
) -> Option<Vec<u32>> {
    let n = csr.row_ptr.len() - 1;
    if (start as usize) >= n || (goal as usize) >= n {
        return None;
    }
    s.reset();
    s.g_score[start as usize] = 0.0;
    let mut open: BinaryHeap<AstarNode> = BinaryHeap::new();
    open.push(AstarNode {
        f: h.h(start),
        v: start,
    });

    while let Some(AstarNode { v, .. }) = open.pop() {
        if v == goal {
            // Reconstruct.
            let mut path = vec![goal];
            let mut cur = goal;
            while let Some(prev) = s.came_from.get(cur as usize).copied() {
                if prev == u32::MAX {
                    break;
                }
                path.push(prev);
                cur = prev;
            }
            path.reverse();
            return Some(path);
        }
        if s.closed[v as usize] {
            continue;
        }
        s.closed[v as usize] = true;

        let g = s.g_score[v as usize];
        if g as u32 >= max_depth {
            continue;
        }
        let st = csr.row_ptr[v as usize] as usize;
        let en = csr.row_ptr[v as usize + 1] as usize;
        for &nxt in &csr.col_idx[st..en] {
            if s.closed[nxt as usize] {
                continue;
            }
            let tentative = g + 1.0;
            if tentative < s.g_score[nxt as usize] {
                s.g_score[nxt as usize] = tentative;
                s.came_from[nxt as usize] = v;
                let f = tentative + h.h(nxt);
                open.push(AstarNode { f, v: nxt });
            }
        }
    }
    None
}

// ─── Best-first DFS ─────────────────────────────────────────────────

/// DFS that visits each vertex, but at every branch sorts the
/// outgoing neighbours by `h` (ascending — lowest first).
///
/// Same observable contract as
/// [`crate::traversal::dfs_visit`] (every reachable vertex is
/// visited exactly once); only the **order** changes. The `visit`
/// closure is called once per discovered vertex.
pub fn best_first_dfs<H: Heuristic, F: FnMut(u32)>(
    csr: &Csr,
    visited: &mut [bool],
    start: u32,
    h: &H,
    mut visit: F,
) {
    let n = visited.len();
    if (start as usize) >= n {
        return;
    }
    let mut stack: Vec<u32> = vec![start];
    while let Some(v) = stack.pop() {
        if visited[v as usize] {
            continue;
        }
        visited[v as usize] = true;
        visit(v);
        let st = csr.row_ptr[v as usize] as usize;
        let en = csr.row_ptr[v as usize + 1] as usize;
        // Collect neighbours, sort by descending h (because the
        // stack pops the top first, we want the smallest h on
        // top).
        let mut nb: Vec<u32> = csr.col_idx[st..en]
            .iter()
            .copied()
            .filter(|&u| !visited[u as usize])
            .collect();
        nb.sort_by(|&a, &b| h.h(b).partial_cmp(&h.h(a)).unwrap_or(Ordering::Equal));
        for u in nb {
            if !visited[u as usize] {
                stack.push(u);
            }
        }
    }
}

// ─── Heuristic-ordered cycle enumeration ────────────────────────────

/// Same contract as
/// [`crate::cycle_enum::enumerate_simple_cycles`], but at every DFS
/// branch, candidate next-vertices are tried in `h`-ascending
/// order. The set of cycles is identical (the heuristic only orders
/// — it never prunes); the time-to-first-cycle is what changes.
///
/// Useful for early-stop variants and for top-$k$ enumeration
/// where the heuristic is correlated with the cycle score (then
/// the heap incumbent improves faster).
pub fn enumerate_cycles_ordered<P: CyclePruner, H: Heuristic>(
    graph: &SignedGraph,
    k: usize,
    pruner: &P,
    h: &H,
) -> Vec<Vec<u32>> {
    if k < 3 {
        return Vec::new();
    }
    let (row_ptr, col_idx) = graph.build_csr();
    let sign_lookup = graph.build_sign_lookup();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(k);
    let mut out: Vec<Vec<u32>> = Vec::new();

    for start in 0..(n as u32) {
        path.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs_ordered(
            start,
            &row_ptr,
            &col_idx,
            &sign_lookup,
            k,
            pruner,
            h,
            &mut path,
            &mut visited,
            &mut out,
        );
        visited[start as usize] = false;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs_ordered<P: CyclePruner, H: Heuristic>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    sign_lookup: &std::collections::HashMap<(u32, u32), i8>,
    k: usize,
    pruner: &P,
    h: &H,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    out: &mut Vec<Vec<u32>>,
) {
    if path.len() == k {
        let last = *path.last().unwrap();
        let key = (last.min(start), last.max(start));
        if !sign_lookup.contains_key(&key) {
            return;
        }
        if path.len() >= 3 && path[1] >= path[k - 1] {
            return;
        }
        let mut signs: Vec<i8> = Vec::with_capacity(k);
        for j in 0..k {
            let u = path[j];
            let v = path[(j + 1) % k];
            let kk = (u.min(v), u.max(v));
            signs.push(*sign_lookup.get(&kk).expect("edge present"));
        }
        if pruner.emit_ok(path, &signs) == PrunerDecision::Accept {
            out.push(path.clone());
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    // Collect candidates, sort by h ascending.
    let mut nb: Vec<u32> = col_idx[st..en]
        .iter()
        .copied()
        .filter(|&nxt| nxt >= start && !visited[nxt as usize])
        .collect();
    nb.sort_by(|&a, &b| h.h(a).partial_cmp(&h.h(b)).unwrap_or(Ordering::Equal));
    for nxt in nb {
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs_ordered(
            start,
            row_ptr,
            col_idx,
            sign_lookup,
            k,
            pruner,
            h,
            path,
            visited,
            out,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Cycle enumeration with a closure-based heuristic and no pruner.
pub fn enumerate_cycles_ordered_noprune<H: Heuristic>(
    graph: &SignedGraph,
    k: usize,
    h: &H,
) -> Vec<Vec<u32>> {
    enumerate_cycles_ordered(graph, k, &NoOpPruner, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_grid(rows: u32, cols: u32) -> (SignedGraph, u32) {
        // 4-neighbour grid, vertices indexed row-major.
        let n = rows * cols;
        let mut eu = Vec::new();
        let mut ev = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    eu.push(v);
                    ev.push(v + 1);
                }
                if r + 1 < rows {
                    eu.push(v);
                    ev.push(v + cols);
                }
            }
        }
        let signs: Vec<i8> = (0..eu.len()).map(|_| 1).collect();
        (SignedGraph::from_parts(n, &eu, &ev, &signs), n)
    }

    #[test]
    fn astar_zero_heuristic_matches_bfs_distance() {
        // 5×5 grid, start = (0, 0), goal = (4, 4) — Manhattan
        // distance 8.
        let (g, _) = build_grid(5, 5);
        let csr = Csr::from_graph(&g);
        let mut s = AstarScratch::with_capacity(g.n_nodes);
        let path = astar(&csr, &mut s, 0, 24, &ZeroHeuristic, 32).expect("path exists");
        assert_eq!(path.len() - 1, 8);
    }

    #[test]
    fn astar_with_admissible_heuristic_finds_optimal() {
        let (g, _) = build_grid(8, 8);
        let csr = Csr::from_graph(&g);
        let cols: u32 = 8;
        let h = |v: u32| -> f64 {
            // Manhattan to target = 63 = (7, 7).
            let r = (v / cols) as i32;
            let c = (v % cols) as i32;
            ((7 - r).abs() + (7 - c).abs()) as f64
        };
        let mut s = AstarScratch::with_capacity(g.n_nodes);
        let path = astar(&csr, &mut s, 0, 63, &h, 64).unwrap();
        assert_eq!(path.len() - 1, 14);
    }

    #[test]
    fn astar_returns_none_for_unreachable() {
        // Two disjoint K2's.
        let g = SignedGraph::from_parts(4, &[0, 2], &[1, 3], &[1, 1]);
        let csr = Csr::from_graph(&g);
        let mut s = AstarScratch::with_capacity(g.n_nodes);
        assert!(astar(&csr, &mut s, 0, 3, &ZeroHeuristic, 100).is_none());
    }

    #[test]
    fn best_first_dfs_visits_every_reachable_vertex() {
        let (g, n) = build_grid(4, 4);
        let csr = Csr::from_graph(&g);
        let mut visited = vec![false; n as usize];
        let mut count = 0u32;
        let h = DegreeHeuristic::new(&csr, true);
        best_first_dfs(&csr, &mut visited, 0, &h, |_| count += 1);
        assert_eq!(count, n);
    }

    #[test]
    fn ordered_cycle_enum_finds_same_set_as_unordered() {
        // A graph with a few triangles + a quad.
        let g = SignedGraph::from_parts(
            6,
            &[0, 1, 2, 0, 1, 3, 4, 5],
            &[1, 2, 0, 3, 3, 4, 5, 3],
            &[1; 8],
        );
        let unordered = crate::enumerate_simple_cycles_noprune(&g, 3);
        let ordered = enumerate_cycles_ordered_noprune(&g, 3, &ZeroHeuristic);
        // Both must agree on the *set* of cycles (order may differ).
        let mut a: Vec<Vec<u32>> = unordered
            .into_iter()
            .map(|mut c| {
                c.sort();
                c
            })
            .collect();
        let mut b: Vec<Vec<u32>> = ordered
            .into_iter()
            .map(|mut c| {
                c.sort();
                c
            })
            .collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
}
