//! # hymeko_tensor — the Tensor pillar's math + streamline
//!
//! Two `std`-only, dependency-free foundations extracted from the Nagare learner so they can be reused and evolved
//! independently of any concrete network:
//!
//! - [`renyi`] — matrix-based Rényi-α=2 information estimators (Giraldo, Rao & Principe 2015): Gram matrices,
//!   `S₂` entropy, mutual information and its gradient, Chebyshev feature maps, and the stable rank. All computed
//!   from traces only, on flat row-major slices.
//! - [`stream`] — the streamed HyMeKo IR format: a network's parameters in a single flat arena addressed by
//!   precomputed offsets, with a bit-exact encode/decode round-trip. The learner ↔ stream bridge stays with the
//!   learner; this crate owns only the format.
//!
//! Co-located in the `tensor` workspace with the geometric-algebra (`hymeko_clifford`) and hypergraph-incidence
//! (`hymeko_graph`) foundations of the Tensor pillar.

pub mod linalg;
pub mod renyi;
pub mod stream;
