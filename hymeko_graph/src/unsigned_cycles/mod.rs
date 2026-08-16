//! Unsigned k-cycle enumeration (basic DFS over CSR adjacency).
//!
//! Decomposed from the original 545-LOC `unsigned_cycles.rs` 2026-05-11
//! per CLAUDE.md §6.5 #4 + the user's no-monstrosity rule. All five
//! sub-modules are ≤200 LOC and have a single cohesive concern.
//!
//! Public items here are primarily algorithm-internal helpers shared with
//! `hymeko_py`; full rustdoc lives on the PyO3 Strategy surface.

#![allow(missing_docs)]

mod bfs;
mod csr;
mod dfs;
mod parallel;
mod sink;

pub use bfs::bfs_distances_into;
pub use csr::{bs_clear, bs_get, bs_set, bs_words, build_csr, has_edge, neighbours};
pub use dfs::{dfs_from, dfs_from_pair, dfs_recurse};
pub use parallel::{enumerate_parallel, make_identity_sink, make_thread_sink, merge_sinks};
pub use sink::Sink;
