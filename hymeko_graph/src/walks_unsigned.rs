//! Simple `walk_len`-walk enumeration (open paths, no vertex revisits).
//!
//! Sibling of `unsigned_cycles` — same DFS skeleton, no closure check,
//! canonical-form dedup by smallest endpoint. Used by Walk-HSiKAN.
//!
//! Moved out of `hymeko_py/src/cycles.rs` 2026-05-11 per CLAUDE.md §6.5 #2.

use crate::unsigned_cycles::{
    bs_clear, bs_get, bs_set, bs_words, build_csr, neighbours, Sink,
};

/// Enumerate all simple length-`walk_len` walks in an undirected graph,
/// dedup'd by canonical orientation (`path[0] <= path[walk_len]`).
///
/// Returns the flat `(n_walks * (walk_len + 1))` u32 buffer. With
/// `max_walks` set, samples uniformly via reservoir.
pub fn enumerate_walks(
    edges_u: &[u32],
    edges_v: &[u32],
    n_nodes: usize,
    walk_len: usize,
    max_walks: Option<usize>,
    seed: u64,
) -> Vec<u32> {
    if walk_len == 0 {
        return Vec::new();
    }
    let edges: Vec<(u32, u32)> =
        edges_u.iter().copied().zip(edges_v.iter().copied()).collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, false);

    let mut sink = match max_walks {
        Some(cap) => Sink::new_reservoir(cap, seed),
        None => Sink::new_full(),
    };
    let mut visited: Vec<u64> = vec![0u64; bs_words(n_nodes)];
    let mut path: Vec<u32> = Vec::with_capacity(walk_len + 1);
    for start in 0..n_nodes as u32 {
        let cont = dfs_walks_from(
            &row_ptr,
            &col_idx,
            start,
            walk_len,
            &mut visited,
            &mut path,
            &mut sink,
        );
        if !cont {
            break;
        }
    }
    sink.into_flat()
}

// ─── internals ──────────────────────────────────────────────────────

fn dfs_walks_recurse(
    row_ptr: &[u32],
    col_idx: &[u32],
    walk_len: usize,
    path: &mut Vec<u32>,
    visited: &mut [u64],
    sink: &mut Sink,
) -> bool {
    if path.len() == walk_len + 1 {
        if path[0] <= path[walk_len] {
            return sink.offer(path);
        }
        return true;
    }
    let tail = *path.last().expect("path non-empty");
    for &nxt in neighbours(row_ptr, col_idx, tail) {
        if bs_get(visited, nxt) {
            continue;
        }
        path.push(nxt);
        bs_set(visited, nxt);
        let cont = dfs_walks_recurse(row_ptr, col_idx, walk_len, path, visited, sink);
        path.pop();
        bs_clear(visited, nxt);
        if !cont {
            return false;
        }
    }
    true
}

fn dfs_walks_from(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    walk_len: usize,
    visited: &mut [u64],
    path: &mut Vec<u32>,
    sink: &mut Sink,
) -> bool {
    path.push(start);
    bs_set(visited, start);
    let cont = dfs_walks_recurse(row_ptr, col_idx, walk_len, path, visited, sink);
    path.pop();
    bs_clear(visited, start);
    cont
}
