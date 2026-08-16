//! Generic signed-graph types — vertex IDs, edges, signs, CSR adjacency.
//!
//! Decoupled from the Python / numpy conversion layer in
//! `hymeko_py` so this crate is pure-Rust and can be imported
//! anywhere.

use std::collections::HashMap;

/// Edge sign in a signed graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Positive edge ($+1$).
    Pos,
    /// Negative edge ($-1$).
    Neg,
}

impl Sign {
    /// Map to $\pm 1$ as `i8` for arithmetic over edge-sign products.
    #[inline]
    pub fn as_i8(self) -> i8 {
        match self {
            Sign::Pos => 1,
            Sign::Neg => -1,
        }
    }

    /// Construct from a non-zero `i8`; panics on `0`.
    #[inline]
    pub fn from_i8(s: i8) -> Sign {
        match s {
            1 => Sign::Pos,
            -1 => Sign::Neg,
            _ => panic!("Sign::from_i8: expected +1 or -1, got {s}"),
        }
    }
}

/// Compact signed graph: edges stored as `(u, v, sign)` triples
/// plus the vertex count.  Build a CSR adjacency from this via
/// [`SignedGraph::build_csr`] for fast neighbour queries.
#[derive(Debug, Clone)]
pub struct SignedGraph {
    /// Number of vertices.  Vertex IDs are `[0, n_nodes)`.
    pub n_nodes: u32,
    /// Edge endpoints, length $|E|$.  Each entry is `(u, v)` with
    /// `u != v`; we don't enforce `u < v`, callers may pass either
    /// orientation.
    pub edges: Vec<(u32, u32)>,
    /// Per-edge signs in `[-1, +1]`, parallel to `edges`.
    pub signs: Vec<i8>,
}

impl SignedGraph {
    /// Build the graph from a parallel-arrays representation.
    /// Panics if `edges_u` and `edges_v` differ in length.
    pub fn from_parts(n_nodes: u32, edges_u: &[u32], edges_v: &[u32], signs: &[i8]) -> SignedGraph {
        assert_eq!(
            edges_u.len(),
            edges_v.len(),
            "edges_u and edges_v differ in length"
        );
        assert_eq!(
            edges_u.len(),
            signs.len(),
            "edges and signs differ in length"
        );
        let edges: Vec<(u32, u32)> = edges_u
            .iter()
            .zip(edges_v.iter())
            .map(|(&u, &v)| (u, v))
            .collect();
        SignedGraph {
            n_nodes,
            edges,
            signs: signs.to_vec(),
        }
    }

    /// Number of edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Build undirected CSR adjacency (`row_ptr`, `col_idx`) for
    /// fast neighbour iteration during DFS.  Symmetric: each edge
    /// `(u, v)` produces both `u → v` and `v → u` entries.
    /// Neighbour lists are sorted and deduplicated.
    pub fn build_csr(&self) -> (Vec<u32>, Vec<u32>) {
        let n = self.n_nodes as usize;
        let mut deg = vec![0u32; n];
        for &(u, v) in &self.edges {
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
        let mut raw_ptr = vec![0u32; n + 1];
        for i in 0..n {
            raw_ptr[i + 1] = raw_ptr[i] + deg[i];
        }
        let total = raw_ptr[n] as usize;
        let mut raw_col = vec![0u32; total];
        let mut cursor = raw_ptr.clone();
        for &(u, v) in &self.edges {
            let pu = cursor[u as usize] as usize;
            raw_col[pu] = v;
            cursor[u as usize] += 1;
            let pv = cursor[v as usize] as usize;
            raw_col[pv] = u;
            cursor[v as usize] += 1;
        }
        let mut col_idx: Vec<u32> = Vec::with_capacity(total);
        let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        for i in 0..n {
            let s = raw_ptr[i] as usize;
            let e = raw_ptr[i + 1] as usize;
            raw_col[s..e].sort_unstable();
            let mut prev: i64 = -1;
            for &v in &raw_col[s..e] {
                if v as i64 != prev {
                    col_idx.push(v);
                    prev = v as i64;
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        (row_ptr, col_idx)
    }

    /// Build an `(u, v) → sign` lookup table.  Treats the graph as
    /// undirected: `(u, v)` and `(v, u)` map to the same sign.
    /// Used by cycle pruners that need sign products.
    pub fn build_sign_lookup(&self) -> HashMap<(u32, u32), i8> {
        let mut out = HashMap::with_capacity(self.edges.len() * 2);
        for (i, &(u, v)) in self.edges.iter().enumerate() {
            let s = self.signs[i];
            let key = (u.min(v), u.max(v));
            out.insert(key, s);
        }
        out
    }

    /// Build undirected CSR adjacency together with a per-edge sign
    /// array aligned to `col_idx`.  `signs_csr[k]` is the sign of the
    /// directed edge ending at `col_idx[k]`, so the cycle DFS can
    /// recover the sign of `(tail, neighbour)` with a single array
    /// index — no HashMap, no SipHash, no allocator pressure.
    ///
    /// # Postconditions
    /// - `row_ptr.len() == n_nodes + 1`
    /// - `col_idx.len() == signs_csr.len()`
    /// - For every undirected input edge `(u, v, s)` with `u != v`,
    ///   both `(u → v, s)` and `(v → u, s)` appear in the CSR.
    /// - Within each row, neighbour ids are sorted and deduplicated;
    ///   on duplicate `(u, v)` input edges the **first** sign in
    ///   sorted-by-column order survives (matches the behaviour of
    ///   [`Self::build_csr`] on the column side).
    ///
    /// Profiling (Epinions $k{=}4$, $m{=}128$, balance pruner)
    /// showed `build_sign_lookup` + per-edge HashMap hits at
    /// ~39% of CPU cycles.  This helper exists so the top-K cycle
    /// enumerators can drop the HashMap entirely.
    pub fn build_csr_with_signs(&self) -> (Vec<u32>, Vec<u32>, Vec<i8>) {
        let n = self.n_nodes as usize;
        let mut deg = vec![0u32; n];
        for &(u, v) in &self.edges {
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
        let mut raw_ptr = vec![0u32; n + 1];
        for i in 0..n {
            raw_ptr[i + 1] = raw_ptr[i] + deg[i];
        }
        let total = raw_ptr[n] as usize;
        let mut raw_col = vec![0u32; total];
        let mut raw_signs = vec![0i8; total];
        let mut cursor = raw_ptr.clone();
        for (i, &(u, v)) in self.edges.iter().enumerate() {
            let s = self.signs[i];
            let pu = cursor[u as usize] as usize;
            raw_col[pu] = v;
            raw_signs[pu] = s;
            cursor[u as usize] += 1;
            let pv = cursor[v as usize] as usize;
            raw_col[pv] = u;
            raw_signs[pv] = s;
            cursor[v as usize] += 1;
        }
        let mut col_idx: Vec<u32> = Vec::with_capacity(total);
        let mut signs_csr: Vec<i8> = Vec::with_capacity(total);
        let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        let mut perm: Vec<usize> = Vec::with_capacity(64);
        for i in 0..n {
            let s = raw_ptr[i] as usize;
            let e = raw_ptr[i + 1] as usize;
            // Stable sort indices [s..e) by `raw_col` only.  Within each
            // duplicate-column group input order is preserved, so taking
            // the LAST entry per group matches `build_sign_lookup`'s
            // `HashMap::insert` last-write-wins semantics on parallel
            // edges with conflicting signs.
            perm.clear();
            perm.extend(s..e);
            perm.sort_by_key(|&k| raw_col[k]);
            let mut j = 0;
            while j < perm.len() {
                let v = raw_col[perm[j]];
                let mut g = j + 1;
                while g < perm.len() && raw_col[perm[g]] == v {
                    g += 1;
                }
                col_idx.push(v);
                signs_csr.push(raw_signs[perm[g - 1]]);
                j = g;
            }
            row_ptr.push(col_idx.len() as u32);
        }
        debug_assert_eq!(col_idx.len(), signs_csr.len());
        debug_assert_eq!(row_ptr.len(), n + 1);
        (row_ptr, col_idx, signs_csr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_round_trip_small_triangle() {
        // Triangle 0-1-2 with signs +, -, +.
        let g = SignedGraph::from_parts(3, &[0, 1, 2], &[1, 2, 0], &[1, -1, 1]);
        let (row_ptr, col_idx) = g.build_csr();
        assert_eq!(row_ptr.len(), 4);
        assert_eq!(col_idx.len(), 6);
        // Each vertex has 2 unique undirected neighbours.
        for v in 0..3 {
            let s = row_ptr[v] as usize;
            let e = row_ptr[v + 1] as usize;
            assert_eq!(e - s, 2);
        }
    }

    #[test]
    fn sign_lookup_canonicalises_undirected() {
        let g = SignedGraph::from_parts(3, &[0, 1], &[1, 2], &[1, -1]);
        let lk = g.build_sign_lookup();
        assert_eq!(lk.get(&(0, 1)), Some(&1));
        assert_eq!(lk.get(&(1, 2)), Some(&-1));
        // Reversed-orientation lookup also works (canonicalised).
        assert_eq!(
            lk.get(&(2, 1)).copied().or(lk.get(&(1, 2)).copied()),
            Some(-1)
        );
    }

    /// Helper: scan the CSR sign array for the directed edge `(u, v)`.
    /// Returns `None` if `v` is not a neighbour of `u`.
    fn csr_lookup(
        row_ptr: &[u32],
        col_idx: &[u32],
        signs_csr: &[i8],
        u: u32,
        v: u32,
    ) -> Option<i8> {
        let s = row_ptr[u as usize] as usize;
        let e = row_ptr[u as usize + 1] as usize;
        col_idx[s..e]
            .iter()
            .position(|&x| x == v)
            .map(|pos| signs_csr[s + pos])
    }

    #[test]
    fn csr_with_signs_postcondition_small_triangle() {
        // Triangle 0-1-2 with signs +, -, +.
        let g = SignedGraph::from_parts(3, &[0, 1, 2], &[1, 2, 0], &[1, -1, 1]);
        let (row_ptr, col_idx, signs_csr) = g.build_csr_with_signs();
        assert_eq!(row_ptr.len(), 4);
        assert_eq!(col_idx.len(), signs_csr.len());
        // 3 undirected edges → 6 CSR entries.
        assert_eq!(col_idx.len(), 6);
        for v in 0..3 {
            let s = row_ptr[v] as usize;
            let e = row_ptr[v + 1] as usize;
            assert_eq!(e - s, 2);
        }
    }

    #[test]
    fn csr_with_signs_matches_build_sign_lookup() {
        // 5-vertex graph with a mix of positive/negative edges,
        // asserting that every key in the HashMap returns the same
        // sign via the CSR-aligned lookup helper.  Regression guard
        // for the topk_cycles HashMap → CSR migration.
        let g = SignedGraph::from_parts(
            5,
            &[0, 0, 1, 1, 2, 3],
            &[1, 2, 2, 3, 4, 4],
            &[1, -1, 1, -1, 1, -1],
        );
        let lk = g.build_sign_lookup();
        let (row_ptr, col_idx, signs_csr) = g.build_csr_with_signs();
        for (&(u, v), &s) in lk.iter() {
            assert_eq!(
                csr_lookup(&row_ptr, &col_idx, &signs_csr, u, v),
                Some(s),
                "CSR lookup ({u}, {v}) disagrees with HashMap sign {s}",
            );
            // Reverse direction must also resolve to the same sign.
            assert_eq!(
                csr_lookup(&row_ptr, &col_idx, &signs_csr, v, u),
                Some(s),
                "CSR lookup ({v}, {u}) disagrees with HashMap sign {s}",
            );
        }
    }

    #[test]
    fn csr_with_signs_parallel_edge_matches_hashmap_last_write_wins() {
        // Regression guard: when the same canonical edge (u,v) appears
        // twice with different signs, build_sign_lookup keeps the
        // LAST one (HashMap::insert overwrites).  build_csr_with_signs
        // must agree, otherwise the topk_cycles refactor changes
        // emitted cycle counts on real data (Epinions has duplicates).
        let g = SignedGraph::from_parts(
            3,
            // Same canonical (0,1) appears twice: first as positive,
            // then as negative.  HashMap last-write-wins → -1.
            &[0, 1, 0],
            &[1, 2, 1],
            &[1, -1, -1],
        );
        let lk = g.build_sign_lookup();
        let (row_ptr, col_idx, signs_csr) = g.build_csr_with_signs();
        for (&(u, v), &s) in lk.iter() {
            assert_eq!(
                csr_lookup(&row_ptr, &col_idx, &signs_csr, u, v),
                Some(s),
                "parallel-edge dedup mismatch on ({u}, {v})",
            );
            assert_eq!(
                csr_lookup(&row_ptr, &col_idx, &signs_csr, v, u),
                Some(s),
                "parallel-edge dedup mismatch on ({v}, {u})",
            );
        }
        // Specifically verify the HashMap-overwritten edge.
        assert_eq!(lk.get(&(0, 1)), Some(&-1));
        assert_eq!(csr_lookup(&row_ptr, &col_idx, &signs_csr, 0, 1), Some(-1),);
    }

    #[test]
    fn csr_with_signs_no_self_lookup_for_missing_edge() {
        // Pure failure-case test for the CSR helper: a non-edge
        // returns None from the CSR scan, never a stale sign.
        let g = SignedGraph::from_parts(3, &[0], &[1], &[1]);
        let (row_ptr, col_idx, signs_csr) = g.build_csr_with_signs();
        assert_eq!(csr_lookup(&row_ptr, &col_idx, &signs_csr, 0, 2), None);
        assert_eq!(csr_lookup(&row_ptr, &col_idx, &signs_csr, 2, 0), None);
        assert_eq!(csr_lookup(&row_ptr, &col_idx, &signs_csr, 0, 1), Some(1));
    }
}
