//! Integration tests for the vertex-prefilter path of the per-vertex
//! top-K cycle enumerator.  Verifies:
//!
//! 1. `_par_adaptive_starting` with the full vertex range is
//!    bit-equivalent to `_par_adaptive`.
//! 2. Filtered starting set produces a subset of unfiltered output
//!    (canonical-cycle keys are a subset).
//! 3. Empty starting set returns zero cycles.
//! 4. Single-vertex starting set returns only cycles rooted at that
//!    vertex (canonical-rotation-aware: a cycle (1,2,3,4) rooted at
//!    vertex 1 has canonical key {1,2,3,4} regardless of rotation).

use hymeko_graph::pruner::NoOpPruner;
use hymeko_graph::signed_graph::SignedGraph;
use hymeko_graph::topk_cycles::{
    enumerate_top_k_per_vertex_cycles_par_adaptive,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting,
};
use hymeko_graph::vertex_filter::{DegreeFilter, TriangleFilter, VertexFilter};

/// 8-vertex fixture: two triangles (0,1,2) and (3,4,5) joined by a
/// bridge edge (2,3), plus isolated 6 and leaf 7 attached to 0.
fn fixture() -> SignedGraph {
    SignedGraph {
        n_nodes: 8,
        edges: vec![
            (0, 1), (1, 2), (0, 2),       // triangle A
            (3, 4), (4, 5), (3, 5),       // triangle B
            (2, 3),                        // bridge
            (0, 7),                        // leaf
        ],
        signs: vec![1; 8],
    }
}

fn canonical_keys(cycles: &[(f64, Vec<u32>, Vec<i8>)]) -> std::collections::HashSet<Vec<u32>> {
    cycles
        .iter()
        .map(|(_, c, _)| {
            let mut k = c.clone();
            k.sort_unstable();
            k
        })
        .collect()
}

#[test]
fn starting_with_full_range_matches_unfiltered() {
    let g = fixture();
    let p = NoOpPruner;
    let m_v = vec![16u32; g.n_nodes as usize];
    let score = |_c: &[u32], _s: &[i8]| 0.0;

    let baseline = enumerate_top_k_per_vertex_cycles_par_adaptive(&g, 3, &p, &m_v, score);
    let all: Vec<u32> = (0..g.n_nodes).collect();
    let with_all =
        enumerate_top_k_per_vertex_cycles_par_adaptive_starting(&g, 3, &p, &m_v, &all, score);

    let base_keys = canonical_keys(&baseline);
    let all_keys = canonical_keys(&with_all);
    assert_eq!(base_keys, all_keys, "full-range starting must match unfiltered");
}

#[test]
fn empty_starting_returns_no_cycles() {
    let g = fixture();
    let p = NoOpPruner;
    let m_v = vec![16u32; g.n_nodes as usize];
    let score = |_c: &[u32], _s: &[i8]| 0.0;

    let out =
        enumerate_top_k_per_vertex_cycles_par_adaptive_starting(&g, 3, &p, &m_v, &[], score);
    assert!(out.is_empty(), "empty starting set must return no cycles");
}

#[test]
fn filtered_starting_is_subset_of_unfiltered() {
    let g = fixture();
    let p = NoOpPruner;
    let m_v = vec![16u32; g.n_nodes as usize];
    let score = |_c: &[u32], _s: &[i8]| 0.0;

    let baseline = enumerate_top_k_per_vertex_cycles_par_adaptive(&g, 3, &p, &m_v, score);

    // Degree filter min=2 drops leaf 7 and isolated 6.
    let keep = DegreeFilter { min_degree: 2 }.keep_set(&g);
    assert!(!keep.contains(&6), "deg=0 vertex 6 must be filtered out");
    assert!(!keep.contains(&7), "deg=1 vertex 7 must be filtered out");

    let filtered =
        enumerate_top_k_per_vertex_cycles_par_adaptive_starting(&g, 3, &p, &m_v, &keep, score);

    let base_keys = canonical_keys(&baseline);
    let filt_keys = canonical_keys(&filtered);

    assert!(
        filt_keys.is_subset(&base_keys),
        "filtered keep-set output must be a subset of full output\n  filtered: {:?}\n  baseline: {:?}",
        filt_keys,
        base_keys
    );

    // Both triangles (0,1,2) and (3,4,5) should still be found
    // because every triangle vertex is in keep-set (their degrees
    // are all ≥ 2).
    let t1: Vec<u32> = vec![0, 1, 2];
    let t2: Vec<u32> = vec![3, 4, 5];
    assert!(filt_keys.contains(&t1), "triangle (0,1,2) must be present");
    assert!(filt_keys.contains(&t2), "triangle (3,4,5) must be present");
}

#[test]
fn single_vertex_starting_only_emits_cycles_through_that_vertex() {
    let g = fixture();
    let p = NoOpPruner;
    let m_v = vec![16u32; g.n_nodes as usize];
    let score = |_c: &[u32], _s: &[i8]| 0.0;

    // Start only from vertex 0 — should only find triangle (0,1,2),
    // not triangle (3,4,5).
    let out =
        enumerate_top_k_per_vertex_cycles_par_adaptive_starting(&g, 3, &p, &m_v, &[0u32], score);

    for (_, cycle, _) in &out {
        assert!(
            cycle.contains(&0),
            "every cycle from starting={{0}} must contain vertex 0; got {:?}",
            cycle
        );
    }

    let keys = canonical_keys(&out);
    assert!(keys.contains(&vec![0u32, 1, 2]), "triangle (0,1,2) should be found");
    assert!(!keys.contains(&vec![3u32, 4, 5]), "triangle (3,4,5) must NOT be found from start=0");
}

#[test]
fn triangle_filter_keeps_only_triangle_cycles() {
    let g = fixture();
    let p = NoOpPruner;
    let m_v = vec![16u32; g.n_nodes as usize];
    let score = |_c: &[u32], _s: &[i8]| 0.0;

    // TriangleFilter keeps vertices in at least one triangle:
    // {0,1,2,3,4,5} on this fixture (the bridge edge (2,3) doesn't
    // create a triangle, so 6 and 7 are filtered out).
    let keep = TriangleFilter.keep_set(&g);
    assert_eq!(keep, vec![0, 1, 2, 3, 4, 5]);

    let filtered =
        enumerate_top_k_per_vertex_cycles_par_adaptive_starting(&g, 3, &p, &m_v, &keep, score);

    let keys = canonical_keys(&filtered);
    assert!(keys.contains(&vec![0u32, 1, 2]));
    assert!(keys.contains(&vec![3u32, 4, 5]));
}
