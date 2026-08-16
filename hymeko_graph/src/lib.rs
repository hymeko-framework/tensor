//! # `hymeko_graph` — generic signed-graph cycle / walk enumeration
//! with axiomatic pruning
//!
//! This crate lifts the Rust DFS, color-coding, reservoir and
//! open-walk enumerators that currently live inside `hymeko_py`
//! into a stand-alone library so they can be used outside the
//! Python / KAN pipeline.
//!
//! ## Why a separate crate
//!
//! The cycle / walk enumeration algorithms are valuable on their
//! own — `hymeko.enumerate_k_cycles_rs` already runs $20$–$50\times$
//! faster than serial Python on Slashdot-class graphs, and the
//! open-walk variant lands canonical-form-correct output bit-for-bit.
//! There is no reason to gate that on the rest of the HyMeKo
//! framework being installed.  The refactor also separates concerns:
//!
//! - `hymeko_graph` :  *generic* signed-graph algorithms
//!   ([`SignedGraph`], CSR construction, k-cycle / k-walk DFS, the
//!   [`pruner::CyclePruner`] trait below)
//! - `hymeko_pgraph` :  P-graph schema + Friedler A1–A5 axiom
//!   checker (already in the workspace)
//! - `hymeko_py` :  PyO3 wrappers + numpy boundary crossing
//!
//! Down-stream code that only needs cycle enumeration imports
//! `hymeko_graph` directly and pays no Python / IR cost.
//!
//! ## Friedler axioms as cycle pruners
//!
//! The pivot point of this crate is the [`pruner::CyclePruner`] trait:
//! during DFS, the enumerator consults a pruner at every partial
//! path to decide whether to continue, and at every closed cycle
//! to decide whether to emit.  Concrete pruners include:
//!
//! - [`balance::CartwrightHararyPruner`] — emit only balanced
//!   ($\prod s = +1$) cycles, or only unbalanced.
//! - [`balance::DavisWeakBalancePruner`] — exclude all-negative
//!   triangles (Davis 1967).
//! - [`balance::BipartiteOnlyPruner`] — for bipartite graphs (e.g.
//!   star-expansion of a hypergraph), prune odd-length cycles
//!   before they materialise.
//! - [`friedler::FriedlerAxiomPruner`] — given a Material /
//!   OperatingUnit kind map à la `hymeko_pgraph`, prune cycles
//!   that violate the P-graph alternation invariant
//!   (axiom A0 in the Friedler 1992 framework: bipartite
//!   M-O-M-O-…) plus the deeper A1–A5 reachability and degree
//!   constraints, so cycle enumeration only emits *feasible*
//!   process loops.
//!
//! In every case the pruner kicks in *during* the DFS, not after
//! materialisation, so the enumeration cost is proportional to
//! the size of the *feasible* cycle set, not the full set.  On
//! P-graph-shaped instances this can be orders of magnitude
//! faster than enumerate-then-filter.
//!
//! ## Status
//!
//! - **2026-05-04**: initial scaffold — types, pruner trait,
//!   Cartwright–Harary / Davis / bipartite stock pruners, Friedler
//!   pruner skeleton.  Cycle enumeration body still lives in
//!   `hymeko_py/src/cycles.rs` and will migrate over in the next
//!   refactor pass.

// SIMD spine kernel (spine::fir_one_cycle_avx2) needs unsafe for
// `_mm256_*` intrinsics; the rest of the crate is `deny(unsafe_code)`
// and the unsafe is scoped via `#[allow]` only on that function.
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod balance;
pub mod color_coding;
pub mod community;
pub mod cycle_enum;
pub mod cycle_sampler;
pub mod friedler;
pub mod incidence;
pub mod path_closure;
pub mod pruner;
pub mod quadtree;
pub mod rand_lcg;
pub mod signed_graph;
pub mod spine;
pub mod topk_cycles;
pub mod topk_walks;
pub mod traversal;
pub mod traversal_heuristic;
pub mod unsigned_cycles;
pub mod vertex_filter;
pub mod walks_unsigned;

pub use balance::{BipartiteOnlyPruner, CartwrightHararyPruner, DavisWeakBalancePruner};
pub use cycle_enum::{enumerate_simple_cycles, enumerate_simple_cycles_noprune};
pub use friedler::FriedlerAxiomPruner;
pub use incidence::{
    build_edge_incidence, build_edge_incidence_vertex_adj, BuildOpts as IncidenceBuildOpts,
    IncidenceCoo, IncidenceOutput, IncidenceResult,
};
pub use quadtree::{build_quadtree, forman_kappa_4conn, QuadtreeAnchors};
pub use vertex_filter::{AndFilter, DegreeFilter, NoFilter, TriangleFilter, VertexFilter};
pub use pruner::{
    CompositePruner, CountingPruner, CyclePruner, NoOpPruner, PrunerDecision, PrunerStats,
    StatsSnapshot,
};
pub use signed_graph::{Sign, SignedGraph};
pub use spine::{
    CliffordFIR, SignedCycleFIR, clifford_fir_backward, clifford_fir_forward,
    fir_cycle_forward, fir_cycle_scatter_mean,
};
pub use topk_cycles::{tiered_m_v_by_degree, WeightedSumScorer};
pub use topk_walks::{
    enumerate_top_k_walks, enumerate_top_k_walks_batch, TopKWalk, TopKWalksBatch,
};
pub use topk_cycles::{
    EntropyGainScorer, HybridScorer, InverseDegreeScorer, TopKBuilder, TopKCycle,
    UniformityHeuristic, UniformityState, degree_adaptive_m_v, enumerate_top_k_cycles,
    enumerate_top_k_cycles_bb, enumerate_top_k_cycles_noprune, enumerate_top_k_cycles_par,
    enumerate_top_k_cycles_par_bb, enumerate_top_k_cycles_par_entropy,
    enumerate_top_k_cycles_par_noprune, enumerate_top_k_per_vertex_cycles,
    enumerate_top_k_per_vertex_cycles_adaptive, enumerate_top_k_per_vertex_cycles_noprune,
    enumerate_top_k_cycles_par_batched,
    enumerate_top_k_cycles_par_bb_batched,
    enumerate_top_k_cycles_par_entropy_batched,
    enumerate_top_k_per_vertex_cycles_par, enumerate_top_k_per_vertex_cycles_par_adaptive,
    enumerate_top_k_per_vertex_cycles_par_adaptive_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched,
    enumerate_top_k_per_vertex_cycles_par_batched,
    enumerate_top_k_per_vertex_cycles_par_noprune,
    TopKCyclesBatch,
};
pub use traversal::{
    BFS_UNREACHED, BfsScratch, Csr, DfsScratch, bfs_distances, bidirectional_bfs, bs_clear, bs_get,
    bs_set, bs_words, bs_zero, count_connected_components, dfs_visit, dfs_visit_noprune,
    dfs_visit_pruned,
};
pub use traversal_heuristic::{
    AstarScratch, DegreeHeuristic, Heuristic, ZeroHeuristic, astar, best_first_dfs,
    enumerate_cycles_ordered, enumerate_cycles_ordered_noprune,
};
