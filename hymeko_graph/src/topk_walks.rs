//! Top-$K$ signed-walk enumeration with admissible-upper-bound pruning.
//!
//! Walk analogue of [`crate::topk_cycles::enumerate_top_k_cycles`].
//! Differences from the cycle path:
//! - Open structure: no closure check at depth `k_len`; the walk has
//!   `k_len` edges and `k_len + 1` vertices.
//! - Canonical-form filter `walk[0] <= walk[walk_len]` halves the
//!   emitted count (matches the convention in `walks_unsigned.rs`).
//! - The `BoundedScorer` trait used here is the same one defined in
//!   [`crate::topk_cycles`]; the admissible upper bound's contract
//!   carries through identically because the score function takes
//!   only `(vs, signs)` and the bound takes only
//!   `(n_neg_so_far, k_remaining, k_len)`.
//!
//! Promoted to library API on 2026-06-03 to close the framework story
//! the Python reference at
//! [`hymeko_neuro::src::core::abb_walks::abb_enumerate_walks`] sketches.
//! The Python path stays the correctness specification; this Rust path
//! is the production runner that drops the per-walk Python object
//! allocation cost.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::signed_graph::SignedGraph;
use crate::topk_cycles::BoundedScorer;

/// Output of [`enumerate_top_k_walks`]: $K$ best walks in descending
/// score order, each as `(score, vertices, edge-signs)`. Vertex count
/// is `walk_len + 1`; sign count is `walk_len`.
pub type TopKWalk = (f64, Vec<u32>, Vec<i8>);

/// Structure-of-Arrays (SoA) output, mirroring
/// [`crate::topk_cycles::TopKCyclesBatch`]. Suitable for zero-copy
/// PyO3 conversion via `into_pyarray()`.
#[derive(Debug, Clone)]
pub struct TopKWalksBatch {
    /// Row-major flat walks, shape (N, walk_len + 1). Length =
    /// N * (walk_len + 1).
    pub walks: Vec<u32>,
    /// Row-major flat signs, shape (N, walk_len). Length = N * walk_len.
    pub signs: Vec<i8>,
    /// Per-walk scores. Length = N.
    pub scores: Vec<f64>,
    /// Walk length (number of edges).
    pub walk_len: usize,
}

impl TopKWalksBatch {
    /// Number of walks retained.
    pub fn len(&self) -> usize {
        self.scores.len()
    }
    /// True iff the batch holds no walks.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

const MAX_INLINE_WALK_LEN: usize = 16;

/// Heap entry: `[walk_len + 1]` vertices + `[walk_len]` signs inline
/// to avoid per-entry `Vec` allocation. The heap is ordered such that
/// the least-preferred entry is at the top (next eviction candidate);
/// this matches the cycle pattern and lets us call `peek()` to compare
/// against the current top-$K$ threshold.
#[derive(Clone, Debug)]
struct HeapEntry {
    score: f64,
    walk_len: u8, // edges; vertices = walk_len + 1
    walk: [u32; MAX_INLINE_WALK_LEN + 1],
    signs: [i8; MAX_INLINE_WALK_LEN],
}

impl HeapEntry {
    fn from_slices(score: f64, walk: &[u32], signs: &[i8]) -> HeapEntry {
        debug_assert_eq!(walk.len(), signs.len() + 1);
        debug_assert!(walk.len() <= MAX_INLINE_WALK_LEN + 1);
        let mut w = [0u32; MAX_INLINE_WALK_LEN + 1];
        let mut s = [0i8; MAX_INLINE_WALK_LEN];
        w[..walk.len()].copy_from_slice(walk);
        s[..signs.len()].copy_from_slice(signs);
        HeapEntry { score, walk_len: signs.len() as u8, walk: w, signs: s }
    }

    fn walk_slice(&self) -> &[u32] {
        &self.walk[..(self.walk_len as usize + 1)]
    }
    fn signs_slice(&self) -> &[i8] {
        &self.signs[..self.walk_len as usize]
    }

    /// Score-then-lex preference; replaces NaN with -inf so admissibility
    /// violations on input don't crash the heap order.
    fn cmp_preference(&self, other: &Self) -> Ordering {
        let a = if self.score.is_nan() { f64::NEG_INFINITY } else { self.score };
        let b = if other.score.is_nan() { f64::NEG_INFINITY } else { other.score };
        a.partial_cmp(&b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.walk_slice().cmp(other.walk_slice()))
    }
}

impl Eq for HeapEntry {}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_preference(other) == Ordering::Equal
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on preference: least-preferred at top.
        other.cmp_preference(self)
    }
}

/// Enumerate the top-$K$ open simple walks of length `walk_len` by
/// `scorer.score`, with admissible-upper-bound DFS pruning.
///
/// Canonical form: emit `walk` only if `walk[0] <= walk[walk_len]`
/// (halves the count over both directions). Simple walks: no vertex
/// revisit within a walk.
///
/// Returns the surviving walks in descending score order.
pub fn enumerate_top_k_walks<S: BoundedScorer>(
    graph: &SignedGraph,
    walk_len: usize,
    top_k: usize,
    scorer: &S,
) -> Vec<TopKWalk> {
    if walk_len == 0 || top_k == 0 {
        return Vec::new();
    }
    if walk_len > MAX_INLINE_WALK_LEN {
        // Fall back to Vec-backed entries would change the API; for
        // the production HSiKAN range walk_len <= 5 we never hit this.
        // A future refactor could lift the inline cap if needed.
        return Vec::new();
    }
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(walk_len + 1);
    let mut signs: Vec<i8> = Vec::with_capacity(walk_len);
    let mut heap: BinaryHeap<HeapEntry> =
        BinaryHeap::with_capacity(top_k + 1);

    for start in 0..(n as u32) {
        path.clear();
        signs.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs(
            start, &row_ptr, &col_idx, &signs_csr,
            walk_len, top_k, scorer,
            &mut path, &mut signs, &mut visited, &mut heap,
        );
        visited[start as usize] = false;
    }

    let mut out: Vec<TopKWalk> = heap
        .into_iter()
        .map(|e| (e.score, e.walk_slice().to_vec(), e.signs_slice().to_vec()))
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// SoA-output variant of [`enumerate_top_k_walks`] for zero-copy PyO3
/// conversion. Functionally equivalent (same walks, same order); the
/// difference is the return layout.
pub fn enumerate_top_k_walks_batch<S: BoundedScorer>(
    graph: &SignedGraph,
    walk_len: usize,
    top_k: usize,
    scorer: &S,
) -> TopKWalksBatch {
    let v = enumerate_top_k_walks(graph, walk_len, top_k, scorer);
    let n = v.len();
    let mut walks: Vec<u32> = Vec::with_capacity(n * (walk_len + 1));
    let mut signs: Vec<i8> = Vec::with_capacity(n * walk_len);
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    for (sc, vs, ss) in v {
        walks.extend_from_slice(&vs);
        signs.extend_from_slice(&ss);
        scores.push(sc);
    }
    TopKWalksBatch { walks, signs, scores, walk_len }
}

#[allow(clippy::too_many_arguments)]
fn dfs<S: BoundedScorer>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    walk_len: usize,
    top_k: usize,
    scorer: &S,
    path: &mut Vec<u32>,
    signs: &mut Vec<i8>,
    visited: &mut [bool],
    heap: &mut BinaryHeap<HeapEntry>,
) {
    if signs.len() == walk_len {
        // Complete walk; apply canonical filter then offer.
        let last = *path.last().unwrap();
        if start <= last {
            let sc = scorer.score(path, signs);
            if heap.len() < top_k {
                heap.push(HeapEntry::from_slices(sc, path, signs));
            } else if let Some(worst) = heap.peek() {
                let candidate = HeapEntry::from_slices(sc, path, signs);
                // Push if strictly more preferred than the worst.
                if candidate.cmp_preference(worst) == Ordering::Greater {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }
        return;
    }

    // ABB upper-bound prune: skip the entire subtree if the
    // optimistic best completion of the current prefix cannot
    // displace the worst entry currently in the heap.
    if heap.len() >= top_k {
        let n_neg_so_far = signs.iter().filter(|&&s| s < 0).count();
        let k_remaining = walk_len - signs.len();
        let ub = scorer.upper_bound(n_neg_so_far, k_remaining, walk_len);
        if let Some(worst) = heap.peek() {
            if ub <= worst.score {
                return;
            }
        }
    }

    let tail = *path.last().unwrap();
    let start_idx = row_ptr[tail as usize] as usize;
    let end_idx = row_ptr[tail as usize + 1] as usize;
    for ei in start_idx..end_idx {
        let nxt = col_idx[ei];
        if visited[nxt as usize] {
            continue;
        }
        let edge_sign = signs_csr[ei];
        path.push(nxt);
        signs.push(edge_sign);
        visited[nxt as usize] = true;
        dfs(start, row_ptr, col_idx, signs_csr,
            walk_len, top_k, scorer,
            path, signs, visited, heap);
        path.pop();
        signs.pop();
        visited[nxt as usize] = false;
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_graph::SignedGraph;
    use crate::topk_cycles::{BalanceScorer, FractionNegativeScorer};

    fn toy_graph() -> SignedGraph {
        // Triangle with mixed signs + one extra edge to a 4th vertex.
        // Vertices: 0..4. Edges: (0,1)+, (1,2)-, (0,2)+, (2,3)-.
        // ``signs`` are i8 in ±1 (Pos / Neg) per the canonical layout.
        let eu: [u32; 4] = [0, 1, 0, 2];
        let ev: [u32; 4] = [1, 2, 2, 3];
        let es: [i8; 4]  = [1, -1, 1, -1];
        SignedGraph::from_parts(4, &eu, &ev, &es)
    }

    #[test]
    fn enumerate_top_k_walks_returns_at_most_k() {
        let g = toy_graph();
        let s = FractionNegativeScorer;
        let v = enumerate_top_k_walks(&g, 2, 3, &s);
        assert!(v.len() <= 3);
    }

    #[test]
    fn enumerate_top_k_walks_top_k_zero_is_empty() {
        let g = toy_graph();
        let s = BalanceScorer;
        let v = enumerate_top_k_walks(&g, 2, 0, &s);
        assert!(v.is_empty());
    }

    #[test]
    fn enumerate_top_k_walks_walk_len_zero_is_empty() {
        let g = toy_graph();
        let s = BalanceScorer;
        let v = enumerate_top_k_walks(&g, 0, 5, &s);
        assert!(v.is_empty());
    }

    #[test]
    fn enumerate_top_k_walks_descending_by_score() {
        let g = toy_graph();
        let s = FractionNegativeScorer;
        let v = enumerate_top_k_walks(&g, 2, 10, &s);
        for w in v.windows(2) {
            assert!(w[0].0 >= w[1].0,
                    "scores not in descending order: {:?}", v.iter().map(|x| x.0).collect::<Vec<_>>());
        }
    }

    #[test]
    fn enumerate_top_k_walks_canonical_form_holds() {
        let g = toy_graph();
        let s = BalanceScorer;
        let v = enumerate_top_k_walks(&g, 3, 100, &s);
        for (_score, walk, signs) in &v {
            assert_eq!(signs.len(), 3);
            assert_eq!(walk.len(), 4);
            assert!(walk[0] <= walk[walk.len() - 1],
                    "canonical form violated: walk={walk:?}");
        }
    }

    #[test]
    fn enumerate_top_k_walks_no_vertex_revisit() {
        let g = toy_graph();
        let s = BalanceScorer;
        let v = enumerate_top_k_walks(&g, 3, 100, &s);
        for (_score, walk, _signs) in &v {
            let mut sorted = walk.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), walk.len(),
                       "vertex revisited in walk: {walk:?}");
        }
    }

    #[test]
    fn enumerate_top_k_walks_batch_shape_consistent() {
        let g = toy_graph();
        let s = FractionNegativeScorer;
        let batch = enumerate_top_k_walks_batch(&g, 2, 5, &s);
        let n = batch.len();
        assert_eq!(batch.walks.len(), n * (batch.walk_len + 1));
        assert_eq!(batch.signs.len(), n * batch.walk_len);
        assert_eq!(batch.scores.len(), n);
    }

    /// Brute reference: enumerate every simple walk of length k from
    /// every start, score it, filter canonical, sort, take K. Used to
    /// verify the heap-bounded ABB result matches the exhaustive top-K.
    fn brute_top_k<S: BoundedScorer>(
        g: &SignedGraph, walk_len: usize, top_k: usize, scorer: &S,
    ) -> Vec<TopKWalk> {
        let (row_ptr, col_idx, signs_csr) = g.build_csr_with_signs();
        let n = g.n_nodes as usize;
        let mut all: Vec<TopKWalk> = Vec::new();

        fn rec(
            row_ptr: &[u32], col_idx: &[u32], signs_csr: &[i8],
            start: u32, walk_len: usize,
            path: &mut Vec<u32>, signs: &mut Vec<i8>,
            visited: &mut Vec<bool>, all: &mut Vec<TopKWalk>,
            scorer: &dyn crate::topk_cycles::BoundedScorer,
        ) {
            if signs.len() == walk_len {
                if start <= *path.last().unwrap() {
                    let sc = scorer.score(path, signs);
                    all.push((sc, path.clone(), signs.clone()));
                }
                return;
            }
            let tail = *path.last().unwrap();
            let s = row_ptr[tail as usize] as usize;
            let e = row_ptr[tail as usize + 1] as usize;
            for ei in s..e {
                let nxt = col_idx[ei];
                if visited[nxt as usize] { continue; }
                let sign = signs_csr[ei];
                path.push(nxt);
                signs.push(sign);
                visited[nxt as usize] = true;
                rec(row_ptr, col_idx, signs_csr, start, walk_len,
                    path, signs, visited, all, scorer);
                path.pop(); signs.pop();
                visited[nxt as usize] = false;
            }
        }

        let mut visited = vec![false; n];
        let mut path = Vec::new();
        let mut signs = Vec::new();
        for start in 0..(n as u32) {
            path.clear(); signs.clear();
            path.push(start);
            visited[start as usize] = true;
            rec(&row_ptr, &col_idx, &signs_csr, start, walk_len,
                &mut path, &mut signs, &mut visited, &mut all, scorer);
            visited[start as usize] = false;
        }
        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        all.truncate(top_k);
        all
    }

    #[test]
    fn enumerate_top_k_walks_matches_brute_force_scores() {
        // The ABB top-K MUST produce the same score multiset as a
        // brute-force enumerate+sort+take_k. Tied-score walks may
        // differ row-by-row; the multiset equality is the contract.
        let g = toy_graph();
        let s = FractionNegativeScorer;
        for walk_len in [2, 3] {
            for k in [1, 3, 7, 100] {
                let abb = enumerate_top_k_walks(&g, walk_len, k, &s);
                let brute = brute_top_k(&g, walk_len, k, &s);
                let abb_scores: Vec<f64> = abb.iter().map(|x| x.0).collect();
                let brute_scores: Vec<f64> = brute.iter().map(|x| x.0).collect();
                assert_eq!(abb_scores, brute_scores,
                    "ABB top-{k} (walk_len={walk_len}) scores differ from brute: {abb_scores:?} vs {brute_scores:?}");
            }
        }
    }
}
