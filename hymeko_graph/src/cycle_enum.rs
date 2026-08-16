//! DFS-based simple-cycle enumeration with pluggable pruning.
//!
//! This module is the canonical cycle-enumeration entry point of
//! the crate.  The full-featured Rust+rayon enumerator that ships
//! with the `hymeko_py` PyO3 wrapper (color-coding, reservoir
//! sampling, atomic early-stop) will migrate here in a later
//! refactor pass; for now this module provides:
//!
//! 1. A serial reference implementation
//!    [`enumerate_simple_cycles`] that consumes any
//!    [`CyclePruner`].
//! 2. The Friedler-axiom benchmark
//!    [`tests::friedler_pruner_skips_odd_lengths`] that
//!    demonstrates the structural-pruning speed-up: on a
//!    bipartite graph, the Friedler A0 pruner skips every
//!    odd-length DFS branch *before* materialisation.
//!
//! The serial path is deliberately simple — the pruner hook is
//! the architectural contribution, not the DFS algorithm itself.
//! Once the API is validated, the parallel + sampling variants
//! will be ported wholesale from `hymeko_py/src/cycles.rs`.

use crate::pruner::{CyclePruner, NoOpPruner, PrunerDecision};
use crate::signed_graph::SignedGraph;

/// Enumerate every simple closed $k$-cycle in `graph` whose
/// canonical form is rooted at the smallest vertex and whose
/// "first hop" is less than its "last hop" (so each cycle is
/// emitted exactly once even though both rotation and reflection
/// would otherwise produce duplicates).
///
/// Each cycle is checked against the pruner at two points:
///
/// 1. *During DFS*: [`CyclePruner::extend_ok`] vetoes a candidate
///    extension before it is pushed onto the path.
/// 2. *On closure*: [`CyclePruner::emit_ok`] vetoes the cycle
///    after the closing edge is found and the edge-sign sequence
///    is built.
///
/// Returns the accepted cycles as a `Vec<Vec<u32>>` of length-$k$
/// vertex sequences.  `O(|V| \cdot \bar d^k)` worst case; the
/// pruner should bring that down dramatically on structured
/// graphs.
pub fn enumerate_simple_cycles(
    graph: &SignedGraph,
    k: usize,
    pruner: &dyn CyclePruner,
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
        dfs(
            start,
            &row_ptr,
            &col_idx,
            &sign_lookup,
            k,
            pruner,
            &mut path,
            &mut visited,
            &mut out,
        );
        visited[start as usize] = false;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    sign_lookup: &std::collections::HashMap<(u32, u32), i8>,
    k: usize,
    pruner: &dyn CyclePruner,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    out: &mut Vec<Vec<u32>>,
) {
    if path.len() == k {
        // Closing-edge check.
        let last = *path.last().unwrap();
        let key = (last.min(start), last.max(start));
        if !sign_lookup.contains_key(&key) {
            return;
        }
        // Emit-time canonicalisation: keep only orientations with
        // path[1] < path[k-1] (avoids both-direction duplicates).
        if path.len() >= 3 && path[1] >= path[k - 1] {
            return;
        }
        // Build the cycle's edge-sign sequence in canonical order.
        let mut signs: Vec<i8> = Vec::with_capacity(k);
        for j in 0..k {
            let u = path[j];
            let v = path[(j + 1) % k];
            let key = (u.min(v), u.max(v));
            signs.push(*sign_lookup.get(&key).expect("edge-key present"));
        }
        if pruner.emit_ok(path, &signs) == PrunerDecision::Accept {
            out.push(path.clone());
        }
        return;
    }
    let tail = *path.last().unwrap();
    let s = row_ptr[tail as usize] as usize;
    let e = row_ptr[tail as usize + 1] as usize;
    for &nxt in &col_idx[s..e] {
        // Smallest-vertex root canonicalisation: only extend to
        // vertices >= start.
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        // Pruner pre-check.
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs(
            start,
            row_ptr,
            col_idx,
            sign_lookup,
            k,
            pruner,
            path,
            visited,
            out,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Convenience wrapper: enumerate without any pruning.  Equivalent
/// to passing [`NoOpPruner`].
pub fn enumerate_simple_cycles_noprune(graph: &SignedGraph, k: usize) -> Vec<Vec<u32>> {
    enumerate_simple_cycles(graph, k, &NoOpPruner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balance::{BalanceMode, BipartiteOnlyPruner, CartwrightHararyPruner};
    use crate::friedler::{FriedlerAxiomPruner, NodeKind};

    /// 4-cycle 0-1-2-3 + chord 0-2.  Has triangles 0-1-2 and
    /// 0-2-3, plus the 4-cycle 0-1-2-3.
    fn build_chord_graph() -> SignedGraph {
        SignedGraph::from_parts(4, &[0, 1, 2, 3, 0], &[1, 2, 3, 0, 2], &[1, -1, 1, -1, 1])
    }

    /// 4-cycle 0-1-2-3, signed so the cycle is balanced
    /// (product of signs = +1).
    fn build_balanced_quad() -> SignedGraph {
        SignedGraph::from_parts(4, &[0, 1, 2, 3], &[1, 2, 3, 0], &[1, 1, 1, 1])
    }

    #[test]
    fn enumerate_finds_all_triangles_in_chord_graph() {
        let g = build_chord_graph();
        let cycles = enumerate_simple_cycles_noprune(&g, 3);
        // Triangles: 0-1-2 and 0-2-3.
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn enumerate_finds_the_quad() {
        let g = build_balanced_quad();
        let cycles = enumerate_simple_cycles_noprune(&g, 4);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0], vec![0, 1, 2, 3]);
    }

    /// Demonstrates the headline Friedler-pruning result: on a
    /// bipartite graph, A0 alternation rejects every odd-length
    /// candidate during DFS, *before* materialisation.  The
    /// 3-cycle that exists in the unsigned chord graph (0-1-2)
    /// is structurally infeasible if vertices 0 and 2 are both
    /// Material — which would put two same-kind vertices adjacently.
    #[test]
    fn friedler_pruner_skips_odd_lengths() {
        // Make 0, 2 Material; 1, 3 Operating.  Edge (0, 2) is
        // M-M which is structurally infeasible in P-graph terms,
        // so Friedler A0 will refuse to extend through it.
        let g = build_chord_graph();
        let kinds = vec![
            NodeKind::Material,
            NodeKind::OperatingUnit,
            NodeKind::Material,
            NodeKind::OperatingUnit,
        ];
        let p = FriedlerAxiomPruner::new(kinds);
        let cycles3 = enumerate_simple_cycles(&g, 3, &p);
        // No triangle survives bipartite alternation.
        assert_eq!(cycles3.len(), 0);
        // The 4-cycle 0-1-2-3 survives (M-O-M-O alternation).
        let cycles4 = enumerate_simple_cycles(&g, 4, &p);
        assert_eq!(cycles4.len(), 1);
        assert_eq!(cycles4[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn cartwright_harary_filters_balanced() {
        // Balanced quad — sign product = +1.
        let g = build_balanced_quad();
        let p_bal = CartwrightHararyPruner {
            mode: BalanceMode::OnlyBalanced,
        };
        assert_eq!(
            enumerate_simple_cycles(&g, 4, &p_bal).len(),
            1,
            "balanced quad survives the OnlyBalanced filter",
        );
        let p_unbal = CartwrightHararyPruner {
            mode: BalanceMode::OnlyUnbalanced,
        };
        assert_eq!(
            enumerate_simple_cycles(&g, 4, &p_unbal).len(),
            0,
            "balanced quad rejected by the OnlyUnbalanced filter",
        );
    }

    #[test]
    fn bipartite_only_kills_odd_emissions() {
        // 5-vertex cycle (odd).
        let g = SignedGraph::from_parts(5, &[0, 1, 2, 3, 4], &[1, 2, 3, 4, 0], &[1; 5]);
        // Without pruner: 1 cycle.
        assert_eq!(enumerate_simple_cycles_noprune(&g, 5).len(), 1);
        // With BipartiteOnlyPruner: 0 cycles (5 is odd).
        let p = BipartiteOnlyPruner;
        assert_eq!(enumerate_simple_cycles(&g, 5, &p).len(), 0);
    }
}
