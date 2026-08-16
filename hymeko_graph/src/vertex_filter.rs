//! Vertex pre-filter for the per-vertex top-K cycle enumerator.
//!
//! Identifies which vertices are worth iterating from as DFS roots.
//! Implements the v1 building blocks from
//! `docs/plans/2026-05-11-vertex-prefilter/`.
//!
//! Cheap filters (Degree, Triangle) are O(|E|) or O(|E|·d̄) and run in
//! <2s on Epinions (131k vertices / 841k edges).  More expensive
//! filters (Clustering, LocalSignEntropy, CliqueSize,
//! BalancedReachable) ship in v2 of the plan.
//!
//! All filters return `Vec<u32>` of vertex ids sorted ascending.

use crate::signed_graph::SignedGraph;

/// Trait for vertex pre-filters used by the per-vertex top-K cycle
/// enumerator.  Returns the set of vertex ids that should be DFS
/// roots.  Empty Vec is a valid result.
pub trait VertexFilter: Send + Sync {
    /// Compute the keep-set on this graph.  Result must be sorted
    /// ascending so downstream rayon iteration is deterministic
    /// across threads.
    fn keep_set(&self, g: &SignedGraph) -> Vec<u32>;

    /// Human-readable filter name (for logs / reports).
    fn name(&self) -> &'static str;
}

/// Pass-through: every vertex is a DFS root.  Bit-equivalent to the
/// pre-filter-introduction baseline.
#[derive(Default, Clone, Copy)]
pub struct NoFilter;

impl VertexFilter for NoFilter {
    fn keep_set(&self, g: &SignedGraph) -> Vec<u32> {
        (0..g.n_nodes).collect()
    }
    fn name(&self) -> &'static str {
        "none"
    }
}

/// Keep vertices with degree ≥ `min_degree`.  Default `min_degree=2`
/// drops leaves and isolated vertices.  O(|E|) via bincount.
#[derive(Clone, Copy)]
pub struct DegreeFilter {
    /// Minimum degree to keep a vertex.  `min_degree=2` drops leaves
    /// + isolated; `min_degree=1` keeps everything reachable.
    pub min_degree: u32,
}

impl Default for DegreeFilter {
    fn default() -> Self {
        DegreeFilter { min_degree: 2 }
    }
}

impl VertexFilter for DegreeFilter {
    fn keep_set(&self, g: &SignedGraph) -> Vec<u32> {
        let n = g.n_nodes as usize;
        let mut deg = vec![0u32; n];
        for &(u, v) in &g.edges {
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
        let mut out: Vec<u32> = (0..n as u32)
            .filter(|&v| deg[v as usize] >= self.min_degree)
            .collect();
        out.sort_unstable();
        out
    }
    fn name(&self) -> &'static str {
        "degree"
    }
}

/// Keep vertices that participate in at least one triangle.  Uses
/// the existing CSR adjacency to find triangles via neighbour-list
/// intersection.  O(|E|·d̄) total — a few seconds on Epinions.
#[derive(Default, Clone, Copy)]
pub struct TriangleFilter;

impl VertexFilter for TriangleFilter {
    fn keep_set(&self, g: &SignedGraph) -> Vec<u32> {
        let (row_ptr, col_idx) = g.build_csr();
        let n = g.n_nodes as usize;
        let mut in_triangle = vec![false; n];

        // Walk each undirected edge (u, v) with u < v; check if any
        // common neighbour exists.  If yes, all three vertices flip.
        for u in 0..n {
            let su = row_ptr[u] as usize;
            let eu = row_ptr[u + 1] as usize;
            for &v in &col_idx[su..eu] {
                let v_us = v as usize;
                if v_us <= u {
                    continue;
                }
                // Sorted-merge intersection of N(u) and N(v).
                let sv = row_ptr[v_us] as usize;
                let ev = row_ptr[v_us + 1] as usize;
                let (mut i, mut j) = (su, sv);
                while i < eu && j < ev {
                    let ni = col_idx[i];
                    let nj = col_idx[j];
                    match ni.cmp(&nj) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            in_triangle[u] = true;
                            in_triangle[v_us] = true;
                            in_triangle[ni as usize] = true;
                            i += 1;
                            j += 1;
                        }
                    }
                }
            }
        }
        (0..n as u32)
            .filter(|&v| in_triangle[v as usize])
            .collect()
    }
    fn name(&self) -> &'static str {
        "triangle"
    }
}

/// AND-composition of multiple filters.  Returns the intersection of
/// each filter's keep-set.  Filters are evaluated in sequence and the
/// intersection is computed at the end (each filter's keep-set is
/// independent of the others — this is intentional so we can cache
/// individual filter outputs).
pub struct AndFilter(pub Vec<Box<dyn VertexFilter>>);

impl VertexFilter for AndFilter {
    fn keep_set(&self, g: &SignedGraph) -> Vec<u32> {
        if self.0.is_empty() {
            return (0..g.n_nodes).collect();
        }
        let n = g.n_nodes as usize;
        let mut bitset = vec![true; n];
        for f in &self.0 {
            let ks = f.keep_set(g);
            let mut next = vec![false; n];
            for v in ks {
                next[v as usize] = bitset[v as usize];
            }
            bitset = next;
        }
        (0..n as u32).filter(|&v| bitset[v as usize]).collect()
    }
    fn name(&self) -> &'static str {
        "and"
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny fixture: triangle (0,1,2) + leaf 3 attached to 0 +
    /// isolated 4.
    /// Edge sign doesn't matter for these filters.
    fn fixture() -> SignedGraph {
        SignedGraph {
            n_nodes: 5,
            edges: vec![(0, 1), (1, 2), (0, 2), (0, 3)],
            signs: vec![1, 1, 1, 1],
        }
    }

    #[test]
    fn no_filter_keeps_all() {
        let g = fixture();
        let f = NoFilter;
        assert_eq!(f.keep_set(&g), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn degree_filter_drops_leaves_and_isolated() {
        let g = fixture();
        // deg: 0→3, 1→2, 2→2, 3→1, 4→0
        // min_degree=2: keep {0, 1, 2}
        let f = DegreeFilter { min_degree: 2 };
        assert_eq!(f.keep_set(&g), vec![0, 1, 2]);
    }

    #[test]
    fn degree_filter_min_1_drops_only_isolated() {
        let g = fixture();
        let f = DegreeFilter { min_degree: 1 };
        assert_eq!(f.keep_set(&g), vec![0, 1, 2, 3]);
    }

    #[test]
    fn triangle_filter_keeps_only_triangle_vertices() {
        let g = fixture();
        let f = TriangleFilter;
        // Triangle (0,1,2) is the only triangle; 3 and 4 not in any.
        assert_eq!(f.keep_set(&g), vec![0, 1, 2]);
    }

    #[test]
    fn and_filter_is_intersection() {
        let g = fixture();
        // degree>=1 keeps {0,1,2,3}; triangle keeps {0,1,2};
        // AND -> {0,1,2}
        let f = AndFilter(vec![
            Box::new(DegreeFilter { min_degree: 1 }),
            Box::new(TriangleFilter),
        ]);
        assert_eq!(f.keep_set(&g), vec![0, 1, 2]);
    }

    #[test]
    fn and_filter_empty_is_identity() {
        let g = fixture();
        let f = AndFilter(vec![]);
        assert_eq!(f.keep_set(&g), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn triangle_filter_handles_isolated_vertex() {
        let g = SignedGraph {
            n_nodes: 3,
            edges: vec![(0, 1)],
            signs: vec![1],
        };
        let f = TriangleFilter;
        // No triangles at all — empty keep set.
        assert!(f.keep_set(&g).is_empty());
    }

    #[test]
    fn keep_set_is_sorted_ascending() {
        let g = fixture();
        for v in [DegreeFilter::default().keep_set(&g),
                   TriangleFilter.keep_set(&g),
                   NoFilter.keep_set(&g)] {
            for w in v.windows(2) {
                assert!(w[0] < w[1], "keep_set must be sorted ascending");
            }
        }
    }

    #[test]
    fn empty_graph_returns_empty_keep_set() {
        let g = SignedGraph {
            n_nodes: 0,
            edges: vec![],
            signs: vec![],
        };
        assert_eq!(NoFilter.keep_set(&g), Vec::<u32>::new());
        assert_eq!(DegreeFilter::default().keep_set(&g), Vec::<u32>::new());
        assert_eq!(TriangleFilter.keep_set(&g), Vec::<u32>::new());
    }
}
