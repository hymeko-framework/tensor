//! Top-$k$ cycle enumeration: keep only the $k$ highest-scoring
//! cycles, never materialise the full set.
//!
//! Same DFS skeleton as [`crate::cycle_enum::enumerate_simple_cycles`],
//! but the output is a bounded min-heap of size $k$. Once the heap is
//! full, every closed cycle whose score is $\le$ the current heap
//! minimum is discarded immediately. The full enumeration cost is
//! still paid in the worst case (no admissible upper-bound is
//! assumed on the score function), but the **memory** cost stays
//! $O(k)$ — and the score-comparison short-circuit lets the caller
//! drop millions of cycles without ever cloning the path.
//!
//! ## When to use this over the full enumerator
//!
//! - You only want a ranked top-$k$ (e.g. "the 100 highest-balance
//!   cycles in a Slashdot-class graph"), not the entire cycle set.
//! - The full set would be too large to materialise (Slashdot's k=4
//!   set is 55.5 M cycles; collecting all to disk took 4 minutes —
//!   keeping only the top 1 000 takes the same DFS time but
//!   $\approx 50\,000\times$ less RAM).
//! - You want a quick heuristic-best for a downstream algorithm
//!   (cycle-aware MSG, balance-of-cycles GNN feature) that only
//!   reads the highest-scoring cycles.
//!
//! ## Score functions
//!
//! Several pre-built scorers live in [`scorers`]; pass any
//! `Fn(&[u32], &[i8]) -> f64` to [`enumerate_top_k_cycles`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::pruner::{CyclePruner, NoOpPruner, PrunerDecision};
use crate::signed_graph::SignedGraph;

/// Sentinel value for "unreachable" / "not yet visited" in BFS-distance buffers.
const DIST_INF: u8 = u8::MAX;

/// CSR-aligned sign lookup: scan `col_idx[row_ptr[u]..row_ptr[u+1]]` for
/// `v` and return `signs_csr[that_index]`.  Replaces the
/// `HashMap<(u32, u32), i8>` lookup that
/// [`crate::SignedGraph::build_sign_lookup`] returns; the array path
/// drops ~39% of CPU cycles on the Epinions $k{=}4$, $m{=}128$ workload
/// (profile attached to `docs/plans/2026-05-10-csr-sign-lookup`).
///
/// # Preconditions
/// - `row_ptr.len() >= u as usize + 2`
/// - `col_idx.len() == signs_csr.len()`
///
/// Returns `None` when `v` is not a neighbour of `u` in the CSR.
#[inline]
fn csr_sign_of(row_ptr: &[u32], col_idx: &[u32], signs_csr: &[i8], u: u32, v: u32) -> Option<i8> {
    let s = row_ptr[u as usize] as usize;
    let e = row_ptr[u as usize + 1] as usize;
    col_idx[s..e]
        .iter()
        .position(|&x| x == v)
        .map(|pos| signs_csr[s + pos])
}

/// BFS from `start` over the CSR, writing the minimum number of hops
/// from `start` to each vertex into `dist` (capped at `k_len`, since
/// any vertex farther than that can never close a $k$-cycle through
/// `start`).  Reuses provided scratch buffers — caller is responsible
/// for sizing them to `n_nodes`.
#[inline]
fn bfs_distances_capped(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k_len: usize,
    dist: &mut [u8],
    frontier_a: &mut Vec<u32>,
    frontier_b: &mut Vec<u32>,
) {
    for d in dist.iter_mut() {
        *d = DIST_INF;
    }
    let cap = (k_len as u8).saturating_sub(1);
    dist[start as usize] = 0;
    frontier_a.clear();
    frontier_a.push(start);
    let mut depth: u8 = 0;
    while !frontier_a.is_empty() && depth < cap {
        depth += 1;
        frontier_b.clear();
        for &v in frontier_a.iter() {
            let s = row_ptr[v as usize] as usize;
            let e = row_ptr[v as usize + 1] as usize;
            for &nxt in &col_idx[s..e] {
                if dist[nxt as usize] == DIST_INF {
                    dist[nxt as usize] = depth;
                    frontier_b.push(nxt);
                }
            }
        }
        std::mem::swap(frontier_a, frontier_b);
    }
}

/// Largest cycle length supported by the inline `HeapEntry`
/// fixed-size buffers.  The signed-graph workloads in this
/// repository run at $k \in \{3, 4, 5\}$ for cycles and at most
/// length-5 walks; a hard ceiling of 8 leaves headroom while
/// keeping the per-entry footprint inside one cache line plus a
/// pad slot.  A `debug_assert!` in the DFS guards
/// `k_len <= MAX_INLINE_K`.
const MAX_INLINE_K: usize = 8;

/// Safe [`BinaryHeap::with_capacity`] hint for top-$K$ cycle storage.
///
/// Tests pass [`usize::MAX`] to mean "unbounded" enumeration; `k_keep + 1`
/// overflows in debug builds. The heap may grow beyond this hint when
/// pushes exceed it.
#[inline]
fn heap_capacity_hint(k_keep: usize) -> usize {
    const MAX_HINT: usize = 1 << 22;
    k_keep.saturating_add(1).clamp(4, MAX_HINT)
}

/// One entry in the bounded heap.
///
/// `cycle` and `signs` are stored as fixed-size stack arrays
/// (`[u32; MAX_INLINE_K]` / `[i8; MAX_INLINE_K]`) instead of `Vec`,
/// because the per-cycle heap-push pattern allocated and dropped
/// two `Vec`s per push.  Profile (27% sample loss): allocator was
/// 14% of cycles, `Vec::clone` 2%, almost all of it from this
/// path.  Inline storage drops the allocations to zero.
///
/// We use the `Reverse`-of-score trick to turn `BinaryHeap`
/// (a max-heap) into a min-heap on score.
#[derive(Clone, Debug)]
struct HeapEntry {
    score: f64,
    /// Length of valid prefix in `cycle` / `signs`.  Always
    /// $\le$ `MAX_INLINE_K`.
    len: u8,
    cycle: [u32; MAX_INLINE_K],
    signs: [i8; MAX_INLINE_K],
}

impl HeapEntry {
    /// Build a `HeapEntry` from slices.  Caller guarantees
    /// `cycle.len() == signs.len() <= MAX_INLINE_K`.
    #[inline]
    fn from_slices(score: f64, cycle: &[u32], signs: &[i8]) -> HeapEntry {
        debug_assert_eq!(cycle.len(), signs.len());
        debug_assert!(cycle.len() <= MAX_INLINE_K);
        let mut c = [0u32; MAX_INLINE_K];
        let mut s = [0i8; MAX_INLINE_K];
        c[..cycle.len()].copy_from_slice(cycle);
        s[..signs.len()].copy_from_slice(signs);
        HeapEntry {
            score,
            len: cycle.len() as u8,
            cycle: c,
            signs: s,
        }
    }

    #[inline]
    fn cycle_slice(&self) -> &[u32] {
        &self.cycle[..self.len as usize]
    }

    #[inline]
    fn signs_slice(&self) -> &[i8] {
        &self.signs[..self.len as usize]
    }

    /// Total preference order: higher score is more preferred; on
    /// score ties, lex-larger cycle slice is more preferred.  Used
    /// at heap-boundary "should this displace the min?" checks and
    /// as the basis of [`Ord`] (reversed) so the heap top is the
    /// least-preferred entry / next eviction candidate.
    ///
    /// The tiebreaker is what makes parallel reduce deterministic:
    /// without it, equal-score cycles never displace each other and
    /// "which one survives" depends on rayon's non-deterministic
    /// merge order.
    fn cmp_preference(&self, other: &Self) -> Ordering {
        let a = if self.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            other.score
        };
        a.partial_cmp(&b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.cycle_slice().cmp(other.cycle_slice()))
    }

    /// Same total order as [`cmp_preference`] but the right-hand side
    /// is a raw `(score, cycle_slice)` pair, avoiding a `HeapEntry`
    /// construction in the dfs boundary-check hot loop.
    fn cmp_preference_vs_slice(&self, other_score: f64, other_slice: &[u32]) -> Ordering {
        let a = if self.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other_score.is_nan() {
            f64::NEG_INFINITY
        } else {
            other_score
        };
        a.partial_cmp(&b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.cycle_slice().cmp(other_slice))
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
        // BinaryHeap pops the *largest by Ord*, so reverse the
        // preference order: least-preferred sits at top and is the
        // next eviction candidate.
        other.cmp_preference(self)
    }
}

/// Output of [`enumerate_top_k_cycles`]: the $k$ best cycles in
/// descending score order, each as `(score, vertices, edge-signs)`.
pub type TopKCycle = (f64, Vec<u32>, Vec<i8>);

/// Struct-of-Arrays (SoA) output for the per-vertex top-K cycle
/// enumerator.  Designed to be returned from
/// `_par_adaptive_starting_batched` and consumed by PyO3 bindings
/// via zero-copy `into_pyarray()`.
///
/// **Layout**: `cycles` and `signs` are row-major `(N, k)` flat
/// buffers; `scores` is length-`N`.  All arrays share the same
/// row index, so row `i` is
/// `(scores[i], cycles[i*k..(i+1)*k], signs[i*k..(i+1)*k])`.
///
/// **Why SoA**: the legacy `Vec<TopKCycle>` allocates 2 small `Vec`s
/// per cycle (vertices + signs).  On Epinions $k{=}4$ at production
/// scale that's $\sim 3.7$M $\times 2 = 7.4$M small heap allocations
/// just for the output, plus a third allocation per dedup-rejected
/// candidate.  SoA replaces these with three `Vec::extend_from_slice`
/// appends into pre-allocated flat buffers --- one allocation
/// amortized over millions of cycles, no fragmentation.
///
/// **PyO3 conversion** is then zero-copy via
/// `cycles.into_pyarray(py)` and `scores.into_pyarray(py)` ---
/// numpy takes ownership of the `Vec`'s heap buffer without copy.
#[derive(Debug, Clone)]
pub struct TopKCyclesBatch {
    /// Row-major flat cycles, shape (N, k).  Length = N × k.
    pub cycles: Vec<u32>,
    /// Row-major flat signs, shape (N, k).  Length = N × k.
    pub signs: Vec<i8>,
    /// Per-cycle scores.  Length = N.
    pub scores: Vec<f64>,
    /// Cycle arity (k).  Multiplied by row index to slice flat arrays.
    pub k: usize,
}

impl TopKCyclesBatch {
    /// Build an empty batch with the given arity.
    #[inline]
    pub fn new(k: usize) -> Self {
        Self {
            cycles: Vec::new(),
            signs: Vec::new(),
            scores: Vec::new(),
            k,
        }
    }

    /// Build with reserved capacity (rough upper bound on output N).
    pub fn with_capacity(k: usize, n_hint: usize) -> Self {
        Self {
            cycles: Vec::with_capacity(n_hint * k),
            signs: Vec::with_capacity(n_hint * k),
            scores: Vec::with_capacity(n_hint),
            k,
        }
    }

    /// Number of cycles in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// True if no cycles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Push a cycle.  `cycle.len() == signs.len() == self.k`.
    #[inline]
    pub fn push(&mut self, score: f64, cycle: &[u32], signs: &[i8]) {
        debug_assert_eq!(cycle.len(), self.k, "cycle.len() must equal self.k");
        debug_assert_eq!(signs.len(), self.k, "signs.len() must equal self.k");
        self.scores.push(score);
        self.cycles.extend_from_slice(cycle);
        self.signs.extend_from_slice(signs);
    }

    /// Sort all rows by score descending (in place, stable across
    /// scores+cycles+signs via permutation).
    pub fn sort_by_score_desc(&mut self) {
        let n = self.len();
        if n <= 1 {
            return;
        }
        let mut indices: Vec<usize> = (0..n).collect();
        let scores = &self.scores;
        indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal)
        });
        // Permute scores, cycles, signs by indices.
        let new_scores: Vec<f64> = indices.iter().map(|&i| self.scores[i]).collect();
        let mut new_cycles: Vec<u32> = Vec::with_capacity(n * self.k);
        let mut new_signs: Vec<i8> = Vec::with_capacity(n * self.k);
        for &i in &indices {
            let s = i * self.k;
            let e = s + self.k;
            new_cycles.extend_from_slice(&self.cycles[s..e]);
            new_signs.extend_from_slice(&self.signs[s..e]);
        }
        self.scores = new_scores;
        self.cycles = new_cycles;
        self.signs = new_signs;
    }

    /// Convert to the legacy `Vec<TopKCycle>` representation for
    /// back-compatibility.  Pays the same per-cycle allocation cost
    /// as the legacy code path --- use only when an existing caller
    /// requires that shape.
    pub fn into_vec_topkcycle(self) -> Vec<TopKCycle> {
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = i * self.k;
            let e = s + self.k;
            out.push((
                self.scores[i],
                self.cycles[s..e].to_vec(),
                self.signs[s..e].to_vec(),
            ));
        }
        out
    }
}

/// DFS-enumerate cycles of length `k_len`, keep the `k_keep`
/// highest-scoring ones according to `score`.
///
/// `score(vertices, edge_signs) -> f64`. Higher = better.
///
/// Returns the surviving cycles sorted by score descending.
pub fn enumerate_top_k_cycles<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(k_len);
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(heap_capacity_hint(k_keep));
    let mut dist: Vec<u8> = vec![DIST_INF; n];
    let mut bfs_a: Vec<u32> = Vec::new();
    let mut bfs_b: Vec<u32> = Vec::new();

    for start in 0..(n as u32) {
        bfs_distances_capped(
            &row_ptr, &col_idx, start, k_len, &mut dist, &mut bfs_a, &mut bfs_b,
        );
        path.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs(
            start,
            &row_ptr,
            &col_idx,
            &signs_csr,
            k_len,
            pruner,
            k_keep,
            &score,
            &mut path,
            &mut visited,
            &mut heap,
            &dist,
        );
        visited[start as usize] = false;
    }

    let mut out: Vec<TopKCycle> = heap
        .into_iter()
        .map(|e| (e.score, e.cycle_slice().to_vec(), e.signs_slice().to_vec()))
        .collect();
    // Heap iteration order isn't sorted; sort descending by score.
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    heap: &mut BinaryHeap<HeapEntry>,
    dist: &[u8],
) where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if path.len() == k_len {
        // Closing-edge check via CSR scan; bail if (last, start) isn't
        // an edge.  Materialises the closing edge's sign in the same
        // pass — reused below to fill `signs[k_len - 1]`.
        let last = *path.last().unwrap();
        let closing_sign = match csr_sign_of(row_ptr, col_idx, signs_csr, last, start) {
            Some(s) => s,
            None => return,
        };
        // Same canonicalisation rule as the full enumerator.
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        // Materialise the edge-sign sequence via CSR scans into a
        // stack-allocated buffer — `Vec::with_capacity(k_len)` here
        // was 3.7M small allocations on Epinions k=4.
        debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
        let mut signs_buf = [0i8; MAX_INLINE_K];
        for j in 0..(k_len - 1) {
            let u = path[j];
            let v = path[j + 1];
            // Invariant: every consecutive pair in `path` is an edge
            // because `dfs` only extends along `col_idx` neighbours.
            signs_buf[j] =
                csr_sign_of(row_ptr, col_idx, signs_csr, u, v).expect("interior edge present");
        }
        signs_buf[k_len - 1] = closing_sign;
        let signs: &[i8] = &signs_buf[..k_len];
        if pruner.emit_ok(path, signs) != PrunerDecision::Accept {
            return;
        }
        // Score and push if competitive.
        let s = score(path, signs);
        if heap.len() < k_keep {
            heap.push(HeapEntry::from_slices(s, path, signs));
        } else {
            // Heap min — under our Ord, peek() returns the smallest
            // score (because we inverted the comparator).
            let beat = heap.peek().map(|min| min.cmp_preference_vs_slice(s, path).is_lt()).unwrap_or(true);
            if beat {
                heap.pop();
                heap.push(HeapEntry::from_slices(s, path, signs));
            }
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    // Remaining edges from `nxt` back to `start` (closing inclusive).
    // After pushing nxt at current path.len()=d, depth becomes d+1.
    // We still owe (k_len - d) edges: (k_len - d - 1) interior +
    // 1 closing.  BFS distance from start to nxt is a lower bound
    // on those edges, so reject if dist[nxt] > k_len - d.
    let remaining_after = (k_len - path.len()) as u8;
    for &nxt in &col_idx[st..en] {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        // BFS-distance pruning: nxt must be ≤ remaining_after hops
        // away from start, otherwise no possible close.
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs(
            start, row_ptr, col_idx, signs_csr, k_len, pruner, k_keep, score, path, visited, heap,
            dist,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Convenience: top-$k$ with no pruner.
pub fn enumerate_top_k_cycles_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64,
{
    enumerate_top_k_cycles(graph, k_len, &NoOpPruner, k_keep, score)
}

// ─── Vertex-stratified top-K ────────────────────────────────────────

/// Vertex-stratified top-$m$ cycle enumeration: for **every** vertex
/// $v$, keep the $m$ highest-scoring cycles that pass through $v$.
///
/// The total cycle set returned is the *union* of those per-vertex
/// top-$m$ sets, with duplicates removed (a cycle that touches $k$
/// vertices appears in $k$ candidate heaps but is emitted once).
///
/// Bound: $|M_e| \le |V| \cdot m$ and **every** vertex is covered as
/// long as it sits on at least one cycle. This is the variant
/// that unblocks Epinions/Slashdot HSiKAN training: it caps the
/// cycle hyperedge incidence matrix per row instead of globally,
/// preserving vertex-uniform information density.
///
/// `score(vertices, edge_signs) -> f64`. Higher = better.
pub fn enumerate_top_k_per_vertex_cycles<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    let n = graph.n_nodes as usize;
    let m_v = vec![m_per_vertex as u32; n];
    enumerate_top_k_per_vertex_cycles_adaptive(graph, k_len, pruner, &m_v, score)
}

/// Per-vertex top-$m_v$ with a per-vertex cap vector.  Same as
/// [`enumerate_top_k_per_vertex_cycles`] but the cap is a slice
/// `m_v[v]` instead of a scalar — every vertex gets its own
/// configured retention size.
///
/// `m_v.len()` must equal `graph.n_nodes`.  Setting `m_v[v] = 0`
/// excludes vertex `v` from the per-vertex heap pool (it still
/// participates in cycles passing through other vertices).
///
/// See [`degree_adaptive_m_v`] for the canonical construction
/// `m_v[v] = min(m_max, max(m_min, c · deg(v)))`.
pub fn enumerate_top_k_per_vertex_cycles_adaptive<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if k_len < 3 || m_v.iter().all(|&c| c == 0) {
        return Vec::new();
    }
    assert_eq!(
        m_v.len(),
        graph.n_nodes as usize,
        "m_v.len() must equal graph.n_nodes",
    );
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(k_len);
    // Per-vertex heaps sized by `m_v[v]`.  Capacity hint = m_v[v]+1
    // (the +1 absorbs the brief overflow during pop+push replacement).
    let mut per_vertex: Vec<BinaryHeap<HeapEntry>> = m_v
        .iter()
        .map(|&cap| BinaryHeap::with_capacity((cap as usize).saturating_add(1)))
        .collect();
    let mut dist: Vec<u8> = vec![DIST_INF; n];
    let mut bfs_a: Vec<u32> = Vec::new();
    let mut bfs_b: Vec<u32> = Vec::new();

    for start in 0..(n as u32) {
        bfs_distances_capped(
            &row_ptr, &col_idx, start, k_len, &mut dist, &mut bfs_a, &mut bfs_b,
        );
        path.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs_per_vertex(
            start,
            &row_ptr,
            &col_idx,
            &signs_csr,
            k_len,
            pruner,
            m_v,
            &score,
            &mut path,
            &mut visited,
            &mut per_vertex,
            &dist,
        );
        visited[start as usize] = false;
    }

    // Union the per-vertex heaps and deduplicate by canonical cycle
    // representation (sorted tuple of vertices).  Uses stack-allocated
    // [u32; MAX_INLINE_K] for the dedup key to avoid one Vec<u32>
    // allocation per cycle (3.7M+ on Epinions c4).
    let mut seen: std::collections::HashSet<[u32; MAX_INLINE_K]> =
        std::collections::HashSet::new();
    let mut out: Vec<TopKCycle> = Vec::new();
    for heap in per_vertex {
        for entry in heap {
            let slice = entry.cycle_slice();
            let mut canon = [0u32; MAX_INLINE_K];
            canon[..slice.len()].copy_from_slice(slice);
            canon[..slice.len()].sort_unstable();
            if seen.insert(canon) {
                out.push((
                    entry.score,
                    entry.cycle_slice().to_vec(),
                    entry.signs_slice().to_vec(),
                ));
            }
        }
    }
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// Construct a degree-adaptive `m_v` vector:
/// `m_v[v] = min(m_max, max(m_min, ⌈c · deg(v)⌉))`.
///
/// `c = 0.0` produces a flat `m_v = m_min` for every vertex (useful
/// for bypassing degree-adaptation in tests).  `c >= 1.0` introduces
/// degree-proportional caps; low-degree vertices get small caps they
/// can actually fill, raising the per-vertex full-heap rate and
/// preserving per-vertex selection structure on the long tail.
///
/// Degree is the undirected vertex degree (each undirected edge
/// contributes 1 to each endpoint's count).  Self-loops, if any,
/// count once.
pub fn degree_adaptive_m_v(graph: &SignedGraph, m_min: u32, m_max: u32, c: f64) -> Vec<u32> {
    assert!(m_min <= m_max, "m_min ({m_min}) > m_max ({m_max})");
    assert!(c >= 0.0, "c must be non-negative; got {c}");
    let n = graph.n_nodes as usize;
    let mut deg = vec![0u32; n];
    for &(u, v) in &graph.edges {
        deg[u as usize] = deg[u as usize].saturating_add(1);
        deg[v as usize] = deg[v as usize].saturating_add(1);
    }
    deg.iter()
        .map(|&d| {
            let scaled = (c * d as f64).ceil() as u64;
            let raw = scaled.max(m_min as u64).min(m_max as u64);
            raw as u32
        })
        .collect()
}

/// **Concentric Pyramid Graph (CPG)** tiered per-vertex cap, formerly
/// "FPN-style".  The metaphor: vertices are sorted by descending
/// degree and bucketed into concentric rings; the innermost ring
/// (hubs, top percentile) gets the largest cap, and each outward ring
/// gets a progressively smaller cap, ending with the leaves at cap 0
/// or a small floor.  This is the cycle-enumeration analogue of a
/// Feature-Pyramid Network's coarse-to-fine resolution hierarchy ---
/// CPG distributes cycle budget the same way an FPN distributes
/// resolution.  v2 of the vertex-prefilter
/// plan (`docs/plans/2026-05-11-vertex-prefilter/`).
///
/// Vertices are binned by an ascending-sorted **centrality score**
/// (currently degree); each bin gets a fixed cap, with the highest
/// tier (top-percentile hubs) receiving the largest cap.  The
/// `tiers` argument is a list of `(percentile_from_top, m_v)` pairs
/// sorted ascending by `percentile_from_top`:
///
/// ```ignore
/// let tiers = vec![
///     (0.1, 1024),   // top 0.1% hubs get 1024 cycles
///     (1.0,  256),   // top 1% (cumulative) get 256
///     (5.0,   64),   // top 5% (cumulative) get 64
///     (20.0,  16),   // top 20% (cumulative) get 16
///     (100.0,  0),   // remaining 80% are skipped
/// ];
/// ```
///
/// Returns a Vec<u32> of per-vertex caps, indexable by vertex id.
/// Vertices with cap=0 are functionally skipped — the per-vertex
/// enumerator will allocate a zero-capacity heap and never push.
///
/// **Cost**: O(n log n) for the sort.  On Epinions (n=131k) this is
/// well under 100 ms.
///
/// **Why this beats degree-adaptive linear**: linear `c * deg(v)`
/// gives every vertex some allocation proportional to its degree.
/// Tiered caps concentrate budget on the top-fraction of vertices
/// by centrality, leaving the long tail (most of the vertex set)
/// with small caps that fill quickly --- raising per-vertex
/// full-heap rate, which unlocks per-vertex ABB.  See the lift
/// study (`reports/2026-05-10-epinions-lift-studies.md`) for the
/// signal-density-vs-cap relationship.
pub fn tiered_m_v_by_degree(graph: &SignedGraph, tiers: &[(f32, u32)]) -> Vec<u32> {
    // Validate input.
    assert!(!tiers.is_empty(), "tiers must be non-empty");
    for w in tiers.windows(2) {
        assert!(
            w[0].0 <= w[1].0,
            "tiers must be sorted ascending by percentile; got {} > {}",
            w[0].0,
            w[1].0
        );
    }
    let n = graph.n_nodes as usize;
    if n == 0 {
        return Vec::new();
    }

    // Compute degrees.
    let mut deg = vec![0u32; n];
    for &(u, v) in &graph.edges {
        deg[u as usize] = deg[u as usize].saturating_add(1);
        deg[v as usize] = deg[v as usize].saturating_add(1);
    }

    // Sort degrees descending to compute percentile thresholds
    // (top-X% means the X% highest degrees).
    let mut sorted_desc: Vec<u32> = deg.clone();
    sorted_desc.sort_unstable_by(|a, b| b.cmp(a));

    // For each tier `(pct, _)`, compute the degree threshold such
    // that vertices with degree >= threshold are in the top `pct%`.
    // `idx = ceil(pct/100 * n) - 1` (clipped).
    let tier_thresholds: Vec<u32> = tiers
        .iter()
        .map(|&(pct, _)| {
            let pct_f = pct.clamp(0.0, 100.0) as f64;
            let idx = ((pct_f / 100.0) * n as f64).ceil() as usize;
            let idx = idx.saturating_sub(1).min(n.saturating_sub(1));
            sorted_desc[idx]
        })
        .collect();

    // Assign each vertex to its tier.  Vertex degree d falls into
    // the FIRST tier (smallest percentile) whose threshold it
    // meets.  We iterate tiers in ascending-pct order, so the
    // top-percentile (smallest pct) check fires first.
    deg.iter()
        .map(|&d| {
            for (i, &thr) in tier_thresholds.iter().enumerate() {
                if d >= thr {
                    return tiers[i].1;
                }
            }
            // Vertex didn't make any tier (shouldn't happen if
            // last tier is 100.0%; defensive fallback).
            tiers.last().map(|t| t.1).unwrap_or(0)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn dfs_per_vertex<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    score: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    per_vertex: &mut [BinaryHeap<HeapEntry>],
    dist: &[u8],
) where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if path.len() == k_len {
        let last = *path.last().unwrap();
        // Closing-edge check via CSR scan; bail if (last, start) isn't
        // an edge.  Keep the resolved sign for use below.
        let closing_sign = match csr_sign_of(row_ptr, col_idx, signs_csr, last, start) {
            Some(s) => s,
            None => return,
        };
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        // Stack-allocated edge-sign buffer; same rationale as in
        // `dfs` above (this hot path was the dominant allocator
        // pressure source pre-fix).
        debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
        let mut signs_buf = [0i8; MAX_INLINE_K];
        for j in 0..(k_len - 1) {
            let u = path[j];
            let v = path[j + 1];
            // Invariant: every consecutive pair in `path` is an edge
            // because the DFS only extends along `col_idx` neighbours.
            signs_buf[j] =
                csr_sign_of(row_ptr, col_idx, signs_csr, u, v).expect("interior edge present");
        }
        signs_buf[k_len - 1] = closing_sign;
        let signs: &[i8] = &signs_buf[..k_len];
        if pruner.emit_ok(path, signs) != PrunerDecision::Accept {
            return;
        }
        let s = score(path, signs);
        // Push into every vertex's heap; HeapEntry stores its cycle /
        // signs inline so each push is a fixed-size memcpy, no Vec
        // allocations and no per-entry heap chunks.
        //
        // Cap is per-vertex (`m_v[v]`) so that low-degree vertices
        // can have small heaps that fill quickly — see plan
        // `docs/plans/2026-05-10-degree-adaptive-mv/`.
        for &v in path.iter() {
            let cap = m_v[v as usize] as usize;
            if cap == 0 {
                // Vertex configured to keep zero cycles — skip.
                continue;
            }
            let heap = &mut per_vertex[v as usize];
            if heap.len() < cap {
                heap.push(HeapEntry::from_slices(s, path, signs));
            } else {
                let beat = heap.peek().map(|min| min.cmp_preference_vs_slice(s, path).is_lt()).unwrap_or(true);
                if beat {
                    heap.pop();
                    heap.push(HeapEntry::from_slices(s, path, signs));
                }
            }
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    let remaining_after = (k_len - path.len()) as u8;
    for &nxt in &col_idx[st..en] {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs_per_vertex(
            start, row_ptr, col_idx, signs_csr, k_len, pruner, m_v, score, path, visited,
            per_vertex, dist,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Convenience: vertex-stratified top-$m$ with no pruner.
pub fn enumerate_top_k_per_vertex_cycles_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64,
{
    enumerate_top_k_per_vertex_cycles(graph, k_len, &NoOpPruner, m_per_vertex, score)
}

// ─── Rayon-parallel variants ────────────────────────────────────────

/// Parallel vertex-stratified top-$m$.  Each rayon thread takes a
/// disjoint slice of starts and accumulates its own per-vertex heap
/// array; at the end the per-thread heaps are merged into one heap
/// per vertex and the union is dedup'd.
///
/// Memory cost is `O(n_threads × n_vertices × m)` for the per-thread
/// heap arrays — fine on graphs up to a few million vertices with a
/// handful of cores. Lock-free, so contention is zero.
pub fn enumerate_top_k_per_vertex_cycles_par<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    let n = graph.n_nodes as usize;
    let m_v = vec![m_per_vertex as u32; n];
    enumerate_top_k_per_vertex_cycles_par_adaptive(graph, k_len, pruner, &m_v, score)
}

/// SoA variant of [`enumerate_top_k_per_vertex_cycles_par`].
/// Returns [`TopKCyclesBatch`] directly (no per-cycle Vec allocs).
pub fn enumerate_top_k_per_vertex_cycles_par_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_per_vertex: usize,
    score: S,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    let n = graph.n_nodes as usize;
    let m_v = vec![m_per_vertex as u32; n];
    enumerate_top_k_per_vertex_cycles_par_adaptive_batched(
        graph, k_len, pruner, &m_v, score,
    )
}

/// Parallel per-vertex top-$m_v$ with a per-vertex cap vector.  Same
/// as [`enumerate_top_k_per_vertex_cycles_par`] but uses `m_v[v]`
/// per vertex instead of a scalar.
///
/// See [`degree_adaptive_m_v`] for the canonical construction.
pub fn enumerate_top_k_per_vertex_cycles_par_adaptive<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    // Iterate from every vertex.  For a filtered starting set,
    // use [`enumerate_top_k_per_vertex_cycles_par_adaptive_starting`].
    let starting: Vec<u32> = (0..graph.n_nodes).collect();
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting(
        graph, k_len, pruner, m_v, &starting, score,
    )
}

/// SoA variant of [`enumerate_top_k_per_vertex_cycles_par_adaptive`].
pub fn enumerate_top_k_per_vertex_cycles_par_adaptive_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    score: S,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    let starting: Vec<u32> = (0..graph.n_nodes).collect();
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
        graph, k_len, pruner, m_v, &starting, score,
    )
}

/// Parallel per-vertex top-$m_v$ enumerator with a caller-supplied
/// `starting_vertices` slice.  The DFS roots iterate over exactly
/// these vertex ids (must be sorted ascending for deterministic
/// rayon scheduling; duplicates are allowed but pointless).
///
/// Useful for v1 of the vertex-prefilter plan
/// (`docs/plans/2026-05-11-vertex-prefilter/`): the caller computes
/// a `VertexFilter::keep_set` then passes it here, skipping
/// non-productive vertices entirely.
///
/// Same per-vertex output semantics as
/// [`enumerate_top_k_per_vertex_cycles_par_adaptive`]: cycles
/// retained from a vertex `v` are at most `m_v[v]` ranked by
/// `score`.  Vertices NOT in `starting_vertices` contribute zero
/// cycles (they may still appear as non-starting members of cycles
/// rooted at other vertices --- the per-vertex cap is per-DFS-root,
/// not per-cycle-participant).
pub fn enumerate_top_k_per_vertex_cycles_par_adaptive_starting<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    starting_vertices: &[u32],
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    if k_len < 3 || m_v.iter().all(|&c| c == 0) {
        return Vec::new();
    }
    assert_eq!(
        m_v.len(),
        graph.n_nodes as usize,
        "m_v.len() must equal graph.n_nodes",
    );
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    // Per-fold-task scratch.  Allocated once when rayon kicks off a
    // task (typically num_threads × O(splits) ≈ a few hundred for any
    // realistic graph), reused across the thousands of starting
    // vertices the task processes.  The previous version allocated
    // `visited` and `dist` (~262 KB at Epinions scale) on every
    // single iteration → 131 828 × 262 KB ≈ 34 GB of allocator churn.
    struct Scratch {
        per_vertex: Vec<BinaryHeap<HeapEntry>>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }
    impl Scratch {
        fn new(n: usize, k_len: usize) -> Scratch {
            Scratch {
                per_vertex: vec![BinaryHeap::<HeapEntry>::new(); n],
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            }
        }
    }

    let final_heaps = starting_vertices
        .par_iter()
        .copied()
        .fold(
            || Scratch::new(n, k_len),
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr,
                    &col_idx,
                    start,
                    k_len,
                    &mut s.dist,
                    &mut s.bfs_a,
                    &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_per_vertex(
                    start,
                    &row_ptr,
                    &col_idx,
                    &signs_csr,
                    k_len,
                    pruner,
                    m_v,
                    &score,
                    &mut s.path,
                    &mut s.visited,
                    &mut s.per_vertex,
                    &s.dist,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        // Reduce: only the per-vertex heaps need merging; the rest of
        // each task's scratch is dropped here.
        .map(|s| s.per_vertex)
        .reduce(
            || vec![BinaryHeap::<HeapEntry>::new(); n],
            |mut a, b| {
                for (idx, (av, bv)) in a.iter_mut().zip(b.into_iter()).enumerate() {
                    let cap = m_v[idx] as usize;
                    if cap == 0 {
                        continue;
                    }
                    for entry in bv {
                        if av.len() < cap {
                            av.push(entry);
                        } else {
                            let beat = av.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                            if beat {
                                av.pop();
                                av.push(entry);
                            }
                        }
                    }
                }
                a
            },
        );

    // Dedup union by canonical vertex tuple.  Stack-allocated
    // [u32; MAX_INLINE_K] key avoids one Vec<u32> alloc per cycle.
    let mut seen: std::collections::HashSet<[u32; MAX_INLINE_K]> =
        std::collections::HashSet::new();
    let mut out: Vec<TopKCycle> = Vec::new();
    for heap in final_heaps {
        for entry in heap {
            let slice = entry.cycle_slice();
            let mut canon = [0u32; MAX_INLINE_K];
            canon[..slice.len()].copy_from_slice(slice);
            canon[..slice.len()].sort_unstable();
            if seen.insert(canon) {
                out.push((
                    entry.score,
                    entry.cycle_slice().to_vec(),
                    entry.signs_slice().to_vec(),
                ));
            }
        }
    }
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// Struct-of-Arrays variant of
/// [`enumerate_top_k_per_vertex_cycles_par_adaptive_starting`].
/// Returns a [`TopKCyclesBatch`] directly, eliminating $\sim 2N$
/// small heap allocations (one `Vec<u32>` + one `Vec<i8>` per
/// accepted cycle) that the legacy `Vec<TopKCycle>` output path
/// produces.  PyO3 bindings consuming this can then zero-copy
/// `cycles.into_pyarray()` and `scores.into_pyarray()` rather than
/// rebuilding flat arrays in a copy loop.
///
/// Output is bit-identical to the legacy variant.  Same input
/// parameters; same enumeration semantics; same final score-descending
/// sort.
pub fn enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    starting_vertices: &[u32],
    score: S,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    if k_len < 3 || m_v.iter().all(|&c| c == 0) {
        return TopKCyclesBatch::new(k_len);
    }
    assert_eq!(
        m_v.len(),
        graph.n_nodes as usize,
        "m_v.len() must equal graph.n_nodes",
    );
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    struct Scratch {
        per_vertex: Vec<BinaryHeap<HeapEntry>>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }
    impl Scratch {
        fn new(n: usize, k_len: usize) -> Scratch {
            Scratch {
                per_vertex: vec![BinaryHeap::<HeapEntry>::new(); n],
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            }
        }
    }

    let final_heaps = starting_vertices
        .par_iter()
        .copied()
        .fold(
            || Scratch::new(n, k_len),
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut s.dist, &mut s.bfs_a, &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_per_vertex(
                    start, &row_ptr, &col_idx, &signs_csr, k_len,
                    pruner, m_v, &score,
                    &mut s.path, &mut s.visited,
                    &mut s.per_vertex, &s.dist,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.per_vertex)
        .reduce(
            || vec![BinaryHeap::<HeapEntry>::new(); n],
            |mut a, b| {
                for (idx, (av, bv)) in a.iter_mut().zip(b.into_iter()).enumerate() {
                    let cap = m_v[idx] as usize;
                    if cap == 0 {
                        continue;
                    }
                    for entry in bv {
                        if av.len() < cap {
                            av.push(entry);
                        } else {
                            let beat = av.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                            if beat {
                                av.pop();
                                av.push(entry);
                            }
                        }
                    }
                }
                a
            },
        );

    // Dedup directly into SoA batch — no per-cycle Vec<u32>/Vec<i8>
    // allocations.  Capacity estimate: sum of all heaps' lens
    // (upper bound on output N before dedup).
    let cap_estimate: usize = final_heaps.iter().map(|h| h.len()).sum();
    let mut batch = TopKCyclesBatch::with_capacity(k_len, cap_estimate);
    let mut seen: std::collections::HashSet<[u32; MAX_INLINE_K]> =
        std::collections::HashSet::new();
    for heap in final_heaps {
        for entry in heap {
            let slice = entry.cycle_slice();
            let mut canon = [0u32; MAX_INLINE_K];
            canon[..slice.len()].copy_from_slice(slice);
            canon[..slice.len()].sort_unstable();
            if seen.insert(canon) {
                batch.push(entry.score, slice, entry.signs_slice());
            }
        }
    }
    batch.sort_by_score_desc();
    batch
}

/// Per-vertex DFS with **score upper-bound branch-and-bound (ABB)**.
///
/// Trade-off vs [`dfs_per_vertex`]:
///   - ABB is checked at the **start vertex's heap threshold** only.
///   - If start's heap is full and UB ≤ start.peek().score, the
///     branch is pruned --- even though the cycle MIGHT have been
///     useful for some other cycle-vertex's heap.
///   - In exchange, ABB-fired branches save the recursive DFS work
///     into the subtree.
///
/// Net effect: the ABB variant emits a SUBSET of the cycles that
/// the non-ABB variant would.  For paired comparisons across
/// configs (e.g. v2 tiered × baseline) the bias is consistent so
/// paired statistics remain valid; absolute cycle counts may
/// differ from the non-ABB output.
///
/// Most effective when per-vertex heaps fill quickly (small `m_v`,
/// tiered configurations, dense graphs).  When heaps don't fill,
/// the threshold is `-∞` and ABB is a no-op (correct, just slower
/// by the cost of the UB check).
#[allow(clippy::too_many_arguments)]
fn dfs_per_vertex_bb<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    scorer: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    per_vertex: &mut [BinaryHeap<HeapEntry>],
    dist: &[u8],
    n_neg_in_path: usize,
) where
    P: CyclePruner,
    S: BoundedScorer,
{
    if path.len() == k_len {
        let last = *path.last().unwrap();
        let closing_sign = match csr_sign_of(row_ptr, col_idx, signs_csr, last, start) {
            Some(s) => s,
            None => return,
        };
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
        let mut signs_buf = [0i8; MAX_INLINE_K];
        for j in 0..(k_len - 1) {
            let u = path[j];
            let v = path[j + 1];
            signs_buf[j] =
                csr_sign_of(row_ptr, col_idx, signs_csr, u, v).expect("interior edge present");
        }
        signs_buf[k_len - 1] = closing_sign;
        let signs: &[i8] = &signs_buf[..k_len];
        if pruner.emit_ok(path, signs) != PrunerDecision::Accept {
            return;
        }
        let s = scorer.score(path, signs);
        // Push to every cycle vertex's heap as in dfs_per_vertex.
        for &v in path.iter() {
            let cap = m_v[v as usize] as usize;
            if cap == 0 {
                continue;
            }
            let heap = &mut per_vertex[v as usize];
            if heap.len() < cap {
                heap.push(HeapEntry::from_slices(s, path, signs));
            } else {
                let beat = heap.peek().map(|min| min.cmp_preference_vs_slice(s, path).is_lt()).unwrap_or(true);
                if beat {
                    heap.pop();
                    heap.push(HeapEntry::from_slices(s, path, signs));
                }
            }
        }
        return;
    }

    // Hoist start's heap threshold out of the inner loop (single
    // peek per DFS recursion level instead of per neighbour).
    let start_cap = m_v[start as usize] as usize;
    let start_threshold = if start_cap > 0
        && per_vertex[start as usize].len() == start_cap
    {
        per_vertex[start as usize]
            .peek()
            .map(|e| e.score)
            .unwrap_or(f64::NEG_INFINITY)
    } else {
        f64::NEG_INFINITY
    };

    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    let remaining_after = (k_len - path.len()) as u8;
    for (slot, &nxt) in col_idx[st..en].iter().enumerate() {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }

        // ABB: extension's sign + UB on best possible cycle.
        let nxt_sign = signs_csr[st + slot];
        let new_n_neg = n_neg_in_path + (nxt_sign < 0) as usize;
        let k_remaining = k_len - (path.len() + 1);
        let ub = scorer.upper_bound(new_n_neg, k_remaining, k_len);
        if ub <= start_threshold {
            continue;
        }

        path.push(nxt);
        visited[nxt as usize] = true;
        dfs_per_vertex_bb(
            start, row_ptr, col_idx, signs_csr, k_len, pruner, m_v, scorer,
            path, visited, per_vertex, dist, new_n_neg,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Parallel per-vertex top-$m_v$ enumeration with **ABB** (score
/// upper-bound branch-and-bound).  SoA output.
///
/// See [`dfs_per_vertex_bb`] for the ABB-prune semantics
/// (start-vertex-threshold conservative).  Combines naturally with
/// v2 tiered caps from
/// [`tiered_m_v_by_degree`] where heaps fill quickly and ABB
/// fires aggressively.
///
/// Same param surface as
/// [`enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched`]
/// except `scorer` is a [`BoundedScorer`] (must implement
/// `upper_bound()` in addition to `score()`).
pub fn enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    starting_vertices: &[u32],
    scorer: &S,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: BoundedScorer + Sync,
{
    if k_len < 3 || m_v.iter().all(|&c| c == 0) {
        return TopKCyclesBatch::new(k_len);
    }
    assert_eq!(
        m_v.len(),
        graph.n_nodes as usize,
        "m_v.len() must equal graph.n_nodes",
    );
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    struct ScratchBb {
        per_vertex: Vec<BinaryHeap<HeapEntry>>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }
    impl ScratchBb {
        fn new(n: usize, k_len: usize) -> ScratchBb {
            ScratchBb {
                per_vertex: vec![BinaryHeap::<HeapEntry>::new(); n],
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            }
        }
    }

    let final_heaps = starting_vertices
        .par_iter()
        .copied()
        .fold(
            || ScratchBb::new(n, k_len),
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut s.dist, &mut s.bfs_a, &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_per_vertex_bb(
                    start, &row_ptr, &col_idx, &signs_csr, k_len,
                    pruner, m_v, scorer,
                    &mut s.path, &mut s.visited, &mut s.per_vertex, &s.dist, 0,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.per_vertex)
        .reduce(
            || vec![BinaryHeap::<HeapEntry>::new(); n],
            |mut a, b| {
                for (idx, (av, bv)) in a.iter_mut().zip(b.into_iter()).enumerate() {
                    let cap = m_v[idx] as usize;
                    if cap == 0 {
                        continue;
                    }
                    for entry in bv {
                        if av.len() < cap {
                            av.push(entry);
                        } else {
                            let beat = av.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                            if beat {
                                av.pop();
                                av.push(entry);
                            }
                        }
                    }
                }
                a
            },
        );

    // SoA collection with stack-array dedup keys (same as the
    // non-ABB batched variant).
    let cap_estimate: usize = final_heaps.iter().map(|h| h.len()).sum();
    let mut batch = TopKCyclesBatch::with_capacity(k_len, cap_estimate);
    let mut seen: std::collections::HashSet<[u32; MAX_INLINE_K]> =
        std::collections::HashSet::new();
    for heap in final_heaps {
        for entry in heap {
            let slice = entry.cycle_slice();
            let mut canon = [0u32; MAX_INLINE_K];
            canon[..slice.len()].copy_from_slice(slice);
            canon[..slice.len()].sort_unstable();
            if seen.insert(canon) {
                batch.push(entry.score, slice, entry.signs_slice());
            }
        }
    }
    batch.sort_by_score_desc();
    batch
}

/// Global-min ABB: the threshold is the MIN heap-min across all
/// FULL heaps, shared via `AtomicU64` (encoding f64 bits) across
/// rayon tasks.  Monotonically non-increasing for conservative
/// correctness: once any task observes a tighter min, global goes
/// down; we never raise it (raising could let us prune a cycle that
/// was already in some heap, but we don't need to — the heap will
/// reject it on its own).
///
/// Adaptive gating (Approach 5): ABB fires only once the fraction
/// of full heaps reaches `abb_fullness_gate` (0.0 = always fire,
/// 1.0 = never fire).  Default 0.25 means ABB doesn't kick in until
/// at least 25% of vertex heaps are at capacity.  Cheap heuristic:
/// fewer wasted UB-checks early when threshold is loose.
#[allow(clippy::too_many_arguments)]
fn dfs_per_vertex_bb_global<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    scorer: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    per_vertex: &mut [BinaryHeap<HeapEntry>],
    dist: &[u8],
    n_neg_in_path: usize,
    global_min: &std::sync::atomic::AtomicU64,
    n_full_heaps: &std::sync::atomic::AtomicUsize,
    vertex_seen_full: &[std::sync::atomic::AtomicBool],
    fullness_gate_count: usize,
) where
    P: CyclePruner,
    S: BoundedScorer,
{
    if path.len() == k_len {
        let last = *path.last().unwrap();
        let closing_sign = match csr_sign_of(row_ptr, col_idx, signs_csr, last, start) {
            Some(s) => s,
            None => return,
        };
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
        let mut signs_buf = [0i8; MAX_INLINE_K];
        for j in 0..(k_len - 1) {
            let u = path[j];
            let v = path[j + 1];
            signs_buf[j] =
                csr_sign_of(row_ptr, col_idx, signs_csr, u, v).expect("interior edge present");
        }
        signs_buf[k_len - 1] = closing_sign;
        let signs: &[i8] = &signs_buf[..k_len];
        if pruner.emit_ok(path, signs) != PrunerDecision::Accept {
            return;
        }
        let s = scorer.score(path, signs);
        // Push to every cycle vertex's heap.  Track when a heap
        // newly fills (n_full_heaps++) and when its min changes
        // (global_min decreases).
        for &v in path.iter() {
            let cap = m_v[v as usize] as usize;
            if cap == 0 {
                continue;
            }
            let heap = &mut per_vertex[v as usize];
            let was_full = heap.len() == cap;
            if heap.len() < cap {
                heap.push(HeapEntry::from_slices(s, path, signs));
                if heap.len() == cap {
                    // This task's heap[v] just newly filled.  Use the
                    // shared `vertex_seen_full[v]` flag to make sure
                    // we increment n_full_heaps AT MOST ONCE per
                    // distinct vertex (regardless of how many rayon
                    // tasks see their local heap[v] fill).  Without
                    // this dedup the counter inflates by ~W× where
                    // W is the rayon worker count, and the fullness
                    // gate fires far earlier than intended.
                    let already_seen = vertex_seen_full[v as usize]
                        .swap(true, std::sync::atomic::Ordering::Relaxed);
                    if !already_seen {
                        n_full_heaps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    if let Some(min_entry) = heap.peek() {
                        atomic_min_f64(global_min, min_entry.score);
                    }
                }
            } else {
                let beat = heap.peek().map(|min| min.cmp_preference_vs_slice(s, path).is_lt()).unwrap_or(true);
                if beat {
                    heap.pop();
                    heap.push(HeapEntry::from_slices(s, path, signs));
                    if was_full {
                        // Heap min may have changed; conservatively
                        // refresh global_min if new peek is smaller.
                        if let Some(min_entry) = heap.peek() {
                            atomic_min_f64(global_min, min_entry.score);
                        }
                    }
                }
            }
        }
        return;
    }

    // Read fullness + global_min ONCE per recursion level (hoist out
    // of inner neighbour loop).  ABB requires BOTH (a) the fullness
    // gate satisfied AND (b) at least one heap full, otherwise
    // global_min is still +INFINITY and we would prune everything.
    let nfull = n_full_heaps.load(std::sync::atomic::Ordering::Relaxed);
    let gate_satisfied = nfull >= fullness_gate_count;
    let any_full = nfull > 0;
    let abb_on = gate_satisfied && any_full;
    let threshold = if abb_on {
        f64::from_bits(global_min.load(std::sync::atomic::Ordering::Relaxed))
    } else {
        f64::NEG_INFINITY
    };

    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    let remaining_after = (k_len - path.len()) as u8;
    for (slot, &nxt) in col_idx[st..en].iter().enumerate() {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }

        if abb_on {
            let nxt_sign = signs_csr[st + slot];
            let new_n_neg = n_neg_in_path + (nxt_sign < 0) as usize;
            let k_remaining = k_len - (path.len() + 1);
            let ub = scorer.upper_bound(new_n_neg, k_remaining, k_len);
            if ub <= threshold {
                continue;
            }
            path.push(nxt);
            visited[nxt as usize] = true;
            dfs_per_vertex_bb_global(
                start, row_ptr, col_idx, signs_csr, k_len, pruner, m_v, scorer,
                path, visited, per_vertex, dist, new_n_neg,
                global_min, n_full_heaps, vertex_seen_full, fullness_gate_count,
            );
            path.pop();
            visited[nxt as usize] = false;
        } else {
            // ABB not yet active; descend with the running n_neg
            // accumulator but skip the UB check.
            let nxt_sign = signs_csr[st + slot];
            let new_n_neg = n_neg_in_path + (nxt_sign < 0) as usize;
            path.push(nxt);
            visited[nxt as usize] = true;
            dfs_per_vertex_bb_global(
                start, row_ptr, col_idx, signs_csr, k_len, pruner, m_v, scorer,
                path, visited, per_vertex, dist, new_n_neg,
                global_min, n_full_heaps, vertex_seen_full, fullness_gate_count,
            );
            path.pop();
            visited[nxt as usize] = false;
        }
    }
}

/// Atomic compare-and-swap loop: `target = min(target, candidate)`.
/// Uses bit-cast f64 ↔ u64 (well-defined for non-NaN floats; we
/// expect cycle scores in [0, 1] so safe).
#[inline]
fn atomic_min_f64(target: &std::sync::atomic::AtomicU64, candidate: f64) {
    let mut cur = target.load(std::sync::atomic::Ordering::Relaxed);
    loop {
        let cur_f = f64::from_bits(cur);
        if candidate >= cur_f {
            return;
        }
        match target.compare_exchange_weak(
            cur,
            candidate.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// Per-vertex top-$m_v$ enumeration with **global-min ABB** and
/// **fullness-gated activation** (Approach 1 + Approach 5 of the
/// 2026-05-11 ABB-improvement discussion).
///
/// Trade-off vs the start-vertex-threshold variant
/// ([`enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched`]):
///   - Threshold is the MIN heap-min across all FULL heaps in the
///     graph, shared across rayon tasks via `AtomicU64`.
///   - ABB fires only once `fullness_gate * n_nodes` heaps are full
///     (default 0.25): no wasted UB checks during the warm-up phase.
///   - **Correctness-preserving in the limit**: when global-min ≥ UB,
///     no full heap could accept the cycle, so pruning loses no
///     useful cycle.  Cycles destined for non-full heaps are
///     unconditionally retained.
///   - **Less aggressive speedup** than start-only ABB (because
///     global-min ≤ start-min by construction) but AUC-preserving.
pub fn enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_v: &[u32],
    starting_vertices: &[u32],
    scorer: &S,
    fullness_gate: f64,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: BoundedScorer + Sync,
{
    if k_len < 3 || m_v.iter().all(|&c| c == 0) {
        return TopKCyclesBatch::new(k_len);
    }
    assert_eq!(
        m_v.len(),
        graph.n_nodes as usize,
        "m_v.len() must equal graph.n_nodes",
    );
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let gate = fullness_gate.clamp(0.0, 1.0);
    // Count vertices with *non-zero* cap as the gate denominator.
    // CPG ladders zero out leaves (bottom-80% gets cap=0); those
    // vertices never increment `n_full_heaps`, so a naive
    // `gate * n_nodes` denominator means ABB at gate=1.0 NEVER fires
    // on CPG configs (the 80% zero-cap vertices can never count
    // toward the threshold). Using `n_active` lets gate=1.0 mean
    // "all non-zero-cap heaps full" --- the natural CPG semantics.
    let n_active = m_v.iter().filter(|&&c| c > 0).count();
    let fullness_gate_count =
        (gate * n_active as f64).ceil() as usize;
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
    let global_min = AtomicU64::new(f64::INFINITY.to_bits());
    let n_full_heaps = AtomicUsize::new(0);
    // De-dup fullness counting: a vertex only contributes once to
    // n_full_heaps the first time ANY rayon worker's local heap[v]
    // hits cap.  Without this, n_full_heaps is inflated by ~W× and
    // gate=1.0 fires too eagerly (~8% spurious pruning on Slashdot k=4
    // m=128, measured 2026-05-11).
    let vertex_seen_full: Vec<AtomicBool> =
        (0..n).map(|_| AtomicBool::new(false)).collect();

    struct ScratchBbG {
        per_vertex: Vec<BinaryHeap<HeapEntry>>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }
    impl ScratchBbG {
        fn new(n: usize, k_len: usize) -> ScratchBbG {
            ScratchBbG {
                per_vertex: vec![BinaryHeap::<HeapEntry>::new(); n],
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            }
        }
    }

    let final_heaps = starting_vertices
        .par_iter()
        .copied()
        .fold(
            || ScratchBbG::new(n, k_len),
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut s.dist, &mut s.bfs_a, &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_per_vertex_bb_global(
                    start, &row_ptr, &col_idx, &signs_csr, k_len,
                    pruner, m_v, scorer,
                    &mut s.path, &mut s.visited, &mut s.per_vertex, &s.dist,
                    0,
                    &global_min, &n_full_heaps, &vertex_seen_full,
                    fullness_gate_count,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.per_vertex)
        .reduce(
            || vec![BinaryHeap::<HeapEntry>::new(); n],
            |mut a, b| {
                for (idx, (av, bv)) in a.iter_mut().zip(b.into_iter()).enumerate() {
                    let cap = m_v[idx] as usize;
                    if cap == 0 {
                        continue;
                    }
                    for entry in bv {
                        if av.len() < cap {
                            av.push(entry);
                        } else {
                            let beat = av.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                            if beat {
                                av.pop();
                                av.push(entry);
                            }
                        }
                    }
                }
                a
            },
        );

    let cap_estimate: usize = final_heaps.iter().map(|h| h.len()).sum();
    let mut batch = TopKCyclesBatch::with_capacity(k_len, cap_estimate);
    let mut seen: std::collections::HashSet<[u32; MAX_INLINE_K]> =
        std::collections::HashSet::new();
    for heap in final_heaps {
        for entry in heap {
            let slice = entry.cycle_slice();
            let mut canon = [0u32; MAX_INLINE_K];
            canon[..slice.len()].copy_from_slice(slice);
            canon[..slice.len()].sort_unstable();
            if seen.insert(canon) {
                batch.push(entry.score, slice, entry.signs_slice());
            }
        }
    }
    batch.sort_by_score_desc();
    batch
}

/// Convenience: parallel vertex-stratified top-$m$ with no pruner.
pub fn enumerate_top_k_per_vertex_cycles_par_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    enumerate_top_k_per_vertex_cycles_par(graph, k_len, &NoOpPruner, m_per_vertex, score)
}

/// Parallel global top-$K$.  Each thread keeps a local min-heap of
/// size $K$; at the end the per-thread heaps are merged.
pub fn enumerate_top_k_cycles_par<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    // Per-fold-task scratch — same hoist rationale as
    // `enumerate_top_k_per_vertex_cycles_par`.
    struct ScratchGlobal {
        heap: BinaryHeap<HeapEntry>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || ScratchGlobal {
                heap: BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            },
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr,
                    &col_idx,
                    start,
                    k_len,
                    &mut s.dist,
                    &mut s.bfs_a,
                    &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs(
                    start,
                    &row_ptr,
                    &col_idx,
                    &signs_csr,
                    k_len,
                    pruner,
                    k_keep,
                    &score,
                    &mut s.path,
                    &mut s.visited,
                    &mut s.heap,
                    &s.dist,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.heap)
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
            |mut a, b| {
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    let mut out: Vec<TopKCycle> = final_heap
        .into_iter()
        .map(|e| (e.score, e.cycle_slice().to_vec(), e.signs_slice().to_vec()))
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// SoA variant of [`enumerate_top_k_cycles_par`].  Returns
/// [`TopKCyclesBatch`] directly — no per-cycle Vec allocations.
pub fn enumerate_top_k_cycles_par_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: S,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    if k_len < 3 || k_keep == 0 {
        return TopKCyclesBatch::new(k_len);
    }
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    struct ScratchGlobal {
        heap: BinaryHeap<HeapEntry>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || ScratchGlobal {
                heap: BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            },
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut s.dist, &mut s.bfs_a, &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs(
                    start, &row_ptr, &col_idx, &signs_csr, k_len,
                    pruner, k_keep, &score,
                    &mut s.path, &mut s.visited, &mut s.heap, &s.dist,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.heap)
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
            |mut a, b| {
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    // SoA collection — no per-cycle Vec<u32>/Vec<i8> allocations.
    let n_out = final_heap.len();
    let mut batch = TopKCyclesBatch::with_capacity(k_len, n_out);
    for entry in final_heap {
        batch.push(entry.score, entry.cycle_slice(), entry.signs_slice());
    }
    batch.sort_by_score_desc();
    batch
}

/// Convenience: parallel global top-$K$ with no pruner.
pub fn enumerate_top_k_cycles_par_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    enumerate_top_k_cycles_par(graph, k_len, &NoOpPruner, k_keep, score)
}

// ─── Pre-built scorers ──────────────────────────────────────────────

/// Stock heuristics that take `(vertices, edge_signs)` and return
/// $f64$. All are pure functions — pass them directly to
/// [`enumerate_top_k_cycles`].
pub mod scorers {
    /// Sign-product magnitude: $\bigl|\prod_i s_i\bigr|$.
    /// Always $1$ on signed graphs; not useful in isolation, but
    /// included for completeness.
    #[inline]
    pub fn sign_product_abs(_vs: &[u32], signs: &[i8]) -> f64 {
        signs.iter().map(|&s| s as f64).product::<f64>().abs()
    }

    /// Cartwright–Harary balance: $+1$ if cycle is balanced,
    /// $-1$ if unbalanced. Top-$k$ with this scorer surfaces the
    /// balanced cycles first; combine with $-x$ for unbalanced.
    #[inline]
    pub fn balance(_vs: &[u32], signs: &[i8]) -> f64 {
        signs.iter().map(|&s| s as f64).product::<f64>()
    }

    /// Number of negative edges in the cycle, normalised by $k$.
    /// High score = mostly-negative cycle (Heider's "all-enemy"
    /// triad in the limit).
    #[inline]
    pub fn fraction_negative(_vs: &[u32], signs: &[i8]) -> f64 {
        if signs.is_empty() {
            return 0.0;
        }
        let n_neg = signs.iter().filter(|&&s| s < 0).count() as f64;
        n_neg / signs.len() as f64
    }

    /// "Lowest-vertex" heuristic: prefer cycles whose canonical
    /// rotation starts at a small index (pulls cycles touching
    /// the densely-connected hubs in many real graphs).
    #[inline]
    pub fn low_root(vs: &[u32], _signs: &[i8]) -> f64 {
        vs.first().map(|v| -(*v as f64)).unwrap_or(0.0)
    }

    /// Returns a closure that scores by the negation of the sum of
    /// per-vertex weights — top-$k$ then picks the cycles whose
    /// vertex set has the *lowest* total weight (e.g. lowest cost
    /// when weights are vertex costs).
    pub fn min_vertex_weight(weights: Vec<f64>) -> impl Fn(&[u32], &[i8]) -> f64 {
        move |vs: &[u32], _signs: &[i8]| {
            let s: f64 = vs
                .iter()
                .map(|&v| weights.get(v as usize).copied().unwrap_or(0.0))
                .sum();
            -s
        }
    }
}

// ─── ABB: score upper-bound branch-and-bound for global top-K ─────────
//
// Profile (probe at examples/probe_abb_threshold.rs): on Epinions
// k=4 with CartwrightHararyPruner(OnlyBalanced) + fraction_negative
// scorer, 22% of the balanced 4-cycle space is all-negative
// (score = 1.0).  The global top-K heap threshold settles at T=1.0
// for any K up to ~1.4M, which means UB(d, n_neg) ≤ T cuts every
// partial path with a positive committed edge.  See
// docs/plans/2026-05-10-abb-global-topk/plan.{tex,pdf,tikz,mmd}.

/// A scorer paired with an admissible upper bound on the closed
/// cycle's score given a partial path's running negative-edge count.
/// ABB descent uses the bound to prune branches whose best possible
/// closed cycle cannot beat the heap's current threshold.
///
/// # Admissibility postcondition
///
/// For any partial path of `k_len - k_remaining` edges with
/// `n_neg_so_far` negatives, every closed cycle reachable from this
/// state must satisfy
/// ```text
///     score(closed_cycle) ≤ upper_bound(n_neg_so_far, k_remaining, k_len)
/// ```
/// Implementations that violate admissibility silently produce a
/// **wrong** top-K result.  The integration test
/// `tests/abb_global_topk.rs::ub_admissible_*` enforces it on
/// random fixtures.
pub trait BoundedScorer: Sync + Send {
    /// Score a closed cycle.  Same contract as the
    /// `Fn(&[u32], &[i8]) -> f64` scorers in the [`scorers`] module.
    fn score(&self, vs: &[u32], signs: &[i8]) -> f64;

    /// Upper bound on the closed cycle's score given the partial-path
    /// state.  See trait-level admissibility postcondition.
    fn upper_bound(&self, n_neg_so_far: usize, k_remaining: usize, k_len: usize) -> f64;
}

/// `fraction_negative` scorer with admissible UB
/// `(n_neg_so_far + k_remaining) / k_len` (best case: every remaining
/// edge is negative).
#[derive(Debug, Default, Clone, Copy)]
pub struct FractionNegativeScorer;

impl BoundedScorer for FractionNegativeScorer {
    #[inline]
    fn score(&self, _vs: &[u32], signs: &[i8]) -> f64 {
        if signs.is_empty() {
            return 0.0;
        }
        let n_neg = signs.iter().filter(|&&s| s < 0).count() as f64;
        n_neg / signs.len() as f64
    }

    #[inline]
    fn upper_bound(&self, n_neg_so_far: usize, k_remaining: usize, k_len: usize) -> f64 {
        if k_len == 0 {
            return 0.0;
        }
        let total_neg = n_neg_so_far + k_remaining;
        total_neg as f64 / k_len as f64
    }
}

/// Cartwright–Harary balance scorer (sign product) with the trivial
/// UB = 1.0 (sign product is always ±1; closure can flip parity to
/// either via the closing edge's sign).
#[derive(Debug, Default, Clone, Copy)]
pub struct BalanceScorer;

impl BoundedScorer for BalanceScorer {
    #[inline]
    fn score(&self, _vs: &[u32], signs: &[i8]) -> f64 {
        signs.iter().map(|&s| s as f64).product::<f64>()
    }

    #[inline]
    fn upper_bound(&self, _n_neg_so_far: usize, _k_remaining: usize, _k_len: usize) -> f64 {
        // Sign product can take +1 or -1; UB = +1.
        1.0
    }
}

/// `sign_product_abs` scorer (always 1 on signed graphs) with UB 1.0.
#[derive(Debug, Default, Clone, Copy)]
pub struct SignProductAbsScorer;

impl BoundedScorer for SignProductAbsScorer {
    #[inline]
    fn score(&self, _vs: &[u32], signs: &[i8]) -> f64 {
        signs.iter().map(|&s| s as f64).product::<f64>().abs()
    }

    #[inline]
    fn upper_bound(&self, _n_neg_so_far: usize, _k_remaining: usize, _k_len: usize) -> f64 {
        1.0
    }
}

/// `low_root` scorer with the trivial UB 0.0 (the score is always
/// `-vs[0]` for cycles in canonical rotation, where `vs[0] = start`
/// is the smallest vertex of the cycle; UB = max over all start
/// vertices = -0 = 0).  The bound is admissible but loose: ABB on
/// `low_root` will only fire late when many cycles starting at
/// vertex 0 have already populated the heap.
#[derive(Debug, Default, Clone, Copy)]
pub struct LowRootScorer;

impl BoundedScorer for LowRootScorer {
    #[inline]
    fn score(&self, vs: &[u32], _signs: &[i8]) -> f64 {
        vs.first().map(|v| -(*v as f64)).unwrap_or(0.0)
    }

    #[inline]
    fn upper_bound(&self, _n_neg_so_far: usize, _k_remaining: usize, _k_len: usize) -> f64 {
        0.0
    }
}

/// Weighted sum of two `BoundedScorer`s --- the building block for
/// **multi-criteria ABB** (Approach 2, 2026-05-11).  Both weights
/// `a` and `b` MUST be non-negative so the composite upper bound
/// stays admissible:
///
/// $$\mathrm{UB}_{\mathrm{comp}} = a \cdot \mathrm{UB}_{s_1} + b \cdot \mathrm{UB}_{s_2}
///   \;\ge\; a \cdot s_1 + b \cdot s_2 \;=\; \mathrm{score}_{\mathrm{comp}}.$$
///
/// For weighted *differences* or signed combinations, fall back to
/// the single-scorer path --- admissibility of the sum requires
/// admissibility of each part with the same sign.
///
/// Nestable: `WeightedSum<WeightedSum<A,B>, C>` gives a triple, etc.
#[derive(Debug, Default, Clone, Copy)]
pub struct WeightedSumScorer<S1, S2> {
    /// Weight on `s1.score` and `s1.upper_bound` (must be `>= 0`).
    pub a: f64,
    /// First scorer.
    pub s1: S1,
    /// Weight on `s2.score` and `s2.upper_bound` (must be `>= 0`).
    pub b: f64,
    /// Second scorer.
    pub s2: S2,
}

impl<S1, S2> BoundedScorer for WeightedSumScorer<S1, S2>
where
    S1: BoundedScorer,
    S2: BoundedScorer,
{
    #[inline]
    fn score(&self, vs: &[u32], signs: &[i8]) -> f64 {
        self.a * self.s1.score(vs, signs) + self.b * self.s2.score(vs, signs)
    }

    #[inline]
    fn upper_bound(&self, n_neg_so_far: usize, k_remaining: usize, k_len: usize) -> f64 {
        debug_assert!(
            self.a >= 0.0 && self.b >= 0.0,
            "WeightedSumScorer weights must be non-negative for admissible UB"
        );
        self.a * self.s1.upper_bound(n_neg_so_far, k_remaining, k_len)
            + self.b * self.s2.upper_bound(n_neg_so_far, k_remaining, k_len)
    }
}

/// Sequential global top-$K$ with ABB descent.
///
/// Same observable output as [`enumerate_top_k_cycles`] (modulo
/// score-tie tie-breaking — the heap's `s > min` rule is unchanged),
/// but the descent is augmented with the score upper-bound check.
/// When the heap is full at threshold `T` and the partial path's
/// best possible closed-cycle score (per
/// [`BoundedScorer::upper_bound`]) is `≤ T`, the entire subtree is
/// dropped.
pub fn enumerate_top_k_cycles_bb<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    scorer: &S,
) -> Vec<TopKCycle>
where
    P: CyclePruner,
    S: BoundedScorer,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(k_len);
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(heap_capacity_hint(k_keep));
    let mut dist: Vec<u8> = vec![DIST_INF; n];
    let mut bfs_a: Vec<u32> = Vec::new();
    let mut bfs_b: Vec<u32> = Vec::new();

    for start in 0..(n as u32) {
        bfs_distances_capped(
            &row_ptr, &col_idx, start, k_len, &mut dist, &mut bfs_a, &mut bfs_b,
        );
        path.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs_bb(
            start,
            &row_ptr,
            &col_idx,
            &signs_csr,
            k_len,
            pruner,
            k_keep,
            scorer,
            &mut path,
            &mut visited,
            &mut heap,
            &dist,
            0,
        );
        visited[start as usize] = false;
    }

    let mut out: Vec<TopKCycle> = heap
        .into_iter()
        .map(|e| (e.score, e.cycle_slice().to_vec(), e.signs_slice().to_vec()))
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// Parallel global top-$K$ with ABB descent.  Per-rayon-fold-task
/// local heap (same shape as [`enumerate_top_k_cycles_par`]); ABB
/// fires per local heap, so a thread whose heap fills slowly does
/// not benefit from a peer thread's higher threshold.  A
/// cross-thread atomic threshold could tighten further; deferred as
/// a follow-up pending v1 measurements.
pub fn enumerate_top_k_cycles_par_bb<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    scorer: &S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: BoundedScorer,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    // Per-fold-task scratch (rationale identical to the non-ABB
    // parallel variant).
    struct ScratchBb {
        heap: BinaryHeap<HeapEntry>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || ScratchBb {
                heap: BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            },
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr,
                    &col_idx,
                    start,
                    k_len,
                    &mut s.dist,
                    &mut s.bfs_a,
                    &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_bb(
                    start,
                    &row_ptr,
                    &col_idx,
                    &signs_csr,
                    k_len,
                    pruner,
                    k_keep,
                    scorer,
                    &mut s.path,
                    &mut s.visited,
                    &mut s.heap,
                    &s.dist,
                    0,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.heap)
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
            |mut a, b| {
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    let mut out: Vec<TopKCycle> = final_heap
        .into_iter()
        .map(|e| (e.score, e.cycle_slice().to_vec(), e.signs_slice().to_vec()))
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// SoA variant of [`enumerate_top_k_cycles_par_bb`].  ABB DFS with
/// batched output — no per-cycle `Vec<u32>`/`Vec<i8>` allocations.
pub fn enumerate_top_k_cycles_par_bb_batched<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    scorer: &S,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    S: BoundedScorer,
{
    if k_len < 3 || k_keep == 0 {
        return TopKCyclesBatch::new(k_len);
    }
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    struct ScratchBb {
        heap: BinaryHeap<HeapEntry>,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || ScratchBb {
                heap: BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            },
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut s.dist, &mut s.bfs_a, &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_bb(
                    start, &row_ptr, &col_idx, &signs_csr, k_len,
                    pruner, k_keep, scorer,
                    &mut s.path, &mut s.visited, &mut s.heap, &s.dist, 0,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.heap)
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
            |mut a, b| {
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a.peek().map(|min| entry.cmp_preference(min) == Ordering::Greater).unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    let n_out = final_heap.len();
    let mut batch = TopKCyclesBatch::with_capacity(k_len, n_out);
    for entry in final_heap {
        batch.push(entry.score, entry.cycle_slice(), entry.signs_slice());
    }
    batch.sort_by_score_desc();
    batch
}

#[allow(clippy::too_many_arguments)]
fn dfs_bb<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    scorer: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    heap: &mut BinaryHeap<HeapEntry>,
    dist: &[u8],
    n_neg_in_path: usize,
) where
    P: CyclePruner,
    S: BoundedScorer,
{
    if path.len() == k_len {
        let last = *path.last().unwrap();
        let closing_sign = match csr_sign_of(row_ptr, col_idx, signs_csr, last, start) {
            Some(s) => s,
            None => return,
        };
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        let mut signs_buf = [0i8; MAX_INLINE_K];
        for j in 0..(k_len - 1) {
            let u = path[j];
            let v = path[j + 1];
            signs_buf[j] =
                csr_sign_of(row_ptr, col_idx, signs_csr, u, v).expect("interior edge present");
        }
        signs_buf[k_len - 1] = closing_sign;
        let signs: &[i8] = &signs_buf[..k_len];
        if pruner.emit_ok(path, signs) != PrunerDecision::Accept {
            return;
        }
        let s = scorer.score(path, signs);
        if heap.len() < k_keep {
            heap.push(HeapEntry::from_slices(s, path, signs));
        } else {
            let beat = heap.peek().map(|min| min.cmp_preference_vs_slice(s, path).is_lt()).unwrap_or(true);
            if beat {
                heap.pop();
                heap.push(HeapEntry::from_slices(s, path, signs));
            }
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    let remaining_after = (k_len - path.len()) as u8;
    // Heap threshold is meaningful only once the heap is full;
    // before that, ABB is a no-op (every cycle is provisionally
    // accepted).  Hoisting the read out of the inner loop avoids
    // a heap.peek() per candidate neighbour.
    let threshold = if heap.len() == k_keep {
        heap.peek().map(|e| e.score).unwrap_or(f64::NEG_INFINITY)
    } else {
        f64::NEG_INFINITY
    };
    for (slot, &nxt) in col_idx[st..en].iter().enumerate() {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        // ABB: look up the prospective edge's sign and check whether
        // any continuation can possibly beat the heap threshold.
        // The CSR slot index of `nxt` in `tail`'s neighbour list is
        // `st + slot`; the sign is co-located in `signs_csr`.
        let nxt_sign = signs_csr[st + slot];
        let new_n_neg = n_neg_in_path + (nxt_sign < 0) as usize;
        let k_remaining = k_len - (path.len() + 1);
        let ub = scorer.upper_bound(new_n_neg, k_remaining, k_len);
        if ub <= threshold {
            continue; // provably dominated by the heap minimum
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs_bb(
            start, row_ptr, col_idx, signs_csr, k_len, pruner, k_keep, scorer, path, visited, heap,
            dist, new_n_neg,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

// ─── Entropy-heuristic top-K (vertex-uniform cycle selection) ─────
//
// The global ABB enumerator above pulls cycles by extreme score
// (e.g. all-negative cycles when scoring by `fraction_negative`).
// On Epinions HSiKAN training that biases the resulting M_e and
// drops AUC -6.7 pp vs the per-vertex baseline (smoke test in
// `reports/2026-05-10-abb-hsikan-smoke-and-builder.md`).
//
// This section adds an alternative top-K family that selects cycles
// to maximise the **per-vertex incidence-distribution entropy** of
// the kept set.  Vertices that are already well-covered contribute
// less to the next cycle's score; rare vertices contribute more.
// The result is a vertex-uniform M_e by construction with the
// rayon-parallel speed of ABB.  See
// `docs/plans/2026-05-10-entropy-vertex-uniform-cycles/plan.{tex,pdf,tikz,mmd}`.

/// Per-fold-task state for the entropy-heuristic enumerator.
///
/// `counts[v]` is the number of times vertex `v` appears in the
/// cycles currently in the heap.  `total = sum(counts)`.  For the
/// entropy-gain scorer we additionally track
/// `s_sum = sum(c * ln(c) for c in counts if c > 0)`, which lets
/// the entropy `H = ln(total) - s_sum / total` be computed in O(1)
/// and the entropy gain of a candidate cycle in O(k).
pub struct UniformityState {
    /// Per-vertex incidence count over cycles in the heap.
    pub counts: Vec<u32>,
    /// `sum(counts)`.
    pub total: u64,
    /// `sum(c * ln(c) for c in counts if c > 0)`.
    /// Maintained incrementally; only [`EntropyGainScorer`] reads it.
    pub s_sum: f64,
}

impl UniformityState {
    /// Build a zeroed state for `n_vertices`.
    #[inline]
    pub fn new(n_vertices: usize) -> UniformityState {
        UniformityState {
            counts: vec![0u32; n_vertices],
            total: 0,
            s_sum: 0.0,
        }
    }

    /// Apply `delta ∈ {+1, -1}` to vertex `v`'s count, updating
    /// `total` and `s_sum`.  Used by [`EntropyGainScorer::update`]
    /// and [`EntropyGainScorer::rollback`].
    #[inline]
    fn shift_count(&mut self, v: u32, delta: i32) {
        let idx = v as usize;
        let old = self.counts[idx] as f64;
        let new_count = (self.counts[idx] as i64 + delta as i64) as u32;
        let new_f = new_count as f64;
        // s_sum delta = new*ln(new) - old*ln(old), with 0*ln(0) := 0.
        let old_term = if old > 0.0 { old * old.ln() } else { 0.0 };
        let new_term = if new_f > 0.0 { new_f * new_f.ln() } else { 0.0 };
        self.s_sum += new_term - old_term;
        self.counts[idx] = new_count;
        self.total = (self.total as i64 + delta as i64) as u64;
    }
}

/// Strategy interface for scorers that maintain a per-task state
/// (per-vertex incidence counts) and pick cycles to maximise some
/// function of that state.  The state is owned by the DFS / fold
/// task; this trait is the read-only-but-state-aware view.
///
/// # Why a trait instead of a free function
///
/// The two concrete scorers below ([`EntropyGainScorer`],
/// [`InverseDegreeScorer`]) share the same DFS skeleton but differ
/// in:
/// - what part of [`UniformityState`] they read (`counts` only vs
///   `counts + total + s_sum`);
/// - their upper-bound function for ABB pruning;
/// - the magnitude of state updates per accepted cycle.
///
/// The trait factors those out so the DFS body is generic in the
/// strategy, picking up the right monomorphisation per call site.
///
/// # Greedy max-coverage caveat
///
/// Both concrete scorers' "score" depends on the heap's current
/// membership (via the counts).  The heap's `s > min` rule keeps
/// the cycle that scored highest *at the time of admission* — a
/// greedy approximation, not a global optimum.  For monotone
/// submodular score functions (entropy gain is *approximately*
/// submodular until saturation) the greedy yields a
/// $(1 - 1/e)$ approximation.
pub trait UniformityHeuristic: Sync {
    /// Score a closed cycle given the current per-task state.
    /// Higher = more diversity contribution.
    fn score(&self, vs: &[u32], signs: &[i8], state: &UniformityState) -> f64;

    /// Update the state to incorporate `vs` (cycle just admitted
    /// to the heap).
    fn update(&self, vs: &[u32], signs: &[i8], state: &mut UniformityState);

    /// Reverse the effect of [`Self::update`] for a cycle that was
    /// later evicted from the heap (a higher-scoring cycle replaced
    /// it).  Required to keep the state consistent with heap
    /// membership; without rollback the counts drift and future
    /// scores become wrong.
    fn rollback(&self, vs: &[u32], signs: &[i8], state: &mut UniformityState);

    /// Admissible upper bound on the score of any closed cycle
    /// reachable by extending the partial path
    /// `prefix_vs` (already-committed vertices, length
    /// `k_len - k_remaining`) by `k_remaining` more vertices.
    ///
    /// Implementations must satisfy: every closed cycle reachable
    /// from this partial state has `score(...) <= upper_bound(...)`.
    /// Loose bounds are safe; tight bounds let ABB prune more.
    fn upper_bound(
        &self,
        prefix_vs: &[u32],
        k_remaining: usize,
        k_len: usize,
        state: &UniformityState,
    ) -> f64;
}

/// State-dependent entropy-gain scorer.
///
/// The score is the marginal entropy gain $\Delta H = H(C \cup
/// \{c\}) - H(C)$ where $H(C) = \ln T - S/T$ with $T = \sum_v c_v$
/// and $S = \sum_v c_v \ln c_v$.  Computed in O(k) per cycle from
/// the running [`UniformityState`].
#[derive(Debug, Default, Clone, Copy)]
pub struct EntropyGainScorer;

impl EntropyGainScorer {
    /// Compute the post-cycle entropy as if `vs` were added to the
    /// state, without modifying the state.  $T' = T + k$ and
    /// $S' = S + \sum_{v \in vs}[(c_v + 1)\ln(c_v + 1) - c_v\ln c_v]$.
    /// $H' = \ln T' - S'/T'$ with the convention $H(\emptyset) = 0$.
    #[inline]
    fn entropy_after(&self, vs: &[u32], state: &UniformityState) -> f64 {
        let k = vs.len() as u64;
        let t_new = state.total + k;
        if t_new == 0 {
            return 0.0;
        }
        let mut s_new = state.s_sum;
        for &v in vs {
            let c_old = state.counts[v as usize] as f64;
            let c_new = c_old + 1.0;
            let old_term = if c_old > 0.0 { c_old * c_old.ln() } else { 0.0 };
            let new_term = c_new * c_new.ln(); // c_new >= 1 so ln defined
            s_new += new_term - old_term;
        }
        let t_new_f = t_new as f64;
        t_new_f.ln() - s_new / t_new_f
    }

    #[inline]
    fn entropy_now(state: &UniformityState) -> f64 {
        if state.total == 0 {
            return 0.0;
        }
        let t = state.total as f64;
        t.ln() - state.s_sum / t
    }
}

impl UniformityHeuristic for EntropyGainScorer {
    fn score(&self, vs: &[u32], _signs: &[i8], state: &UniformityState) -> f64 {
        self.entropy_after(vs, state) - Self::entropy_now(state)
    }

    fn update(&self, vs: &[u32], _signs: &[i8], state: &mut UniformityState) {
        for &v in vs {
            state.shift_count(v, 1);
        }
    }

    fn rollback(&self, vs: &[u32], _signs: &[i8], state: &mut UniformityState) {
        for &v in vs {
            state.shift_count(v, -1);
        }
    }

    fn upper_bound(
        &self,
        prefix_vs: &[u32],
        k_remaining: usize,
        k_len: usize,
        state: &UniformityState,
    ) -> f64 {
        // The closed cycle's $\Delta H$ is maximised when its
        // remaining (unknown) vertices have c_v = 0 (they contribute
        // zero to S').  Compute the contribution from the committed
        // prefix exactly, take c_v = 0 for the rest.
        let k = k_len as u64;
        let t_new = state.total + k;
        if t_new == 0 {
            return f64::INFINITY;
        }
        let mut s_new = state.s_sum;
        for &v in prefix_vs {
            let c_old = state.counts[v as usize] as f64;
            let c_new = c_old + 1.0;
            let old_term = if c_old > 0.0 { c_old * c_old.ln() } else { 0.0 };
            let new_term = c_new * c_new.ln();
            s_new += new_term - old_term;
        }
        // For each of `k_remaining` unknown future vertices, the
        // best (max-Δ) case is c_v = 0 -> contributes
        // (0+1)*ln(0+1) - 0 = 0 to s_new.  So s_new unchanged.
        let _ = k_remaining; // documented above; no numerical change
        let t_new_f = t_new as f64;
        let h_new = t_new_f.ln() - s_new / t_new_f;
        h_new - Self::entropy_now(state)
    }
}

/// State-independent rare-vertex preference scorer.
///
/// Score = $\sum_{v \in c} 1 / \sqrt{c_v + 1}$.  Cheaper than
/// [`EntropyGainScorer`] (no `ln`, no `total` bookkeeping) and
/// gives a similar bias toward under-covered vertices.  Doesn't
/// require `s_sum` updates; only `counts` and `total` are touched.
#[derive(Debug, Default, Clone, Copy)]
pub struct InverseDegreeScorer;

impl UniformityHeuristic for InverseDegreeScorer {
    fn score(&self, vs: &[u32], _signs: &[i8], state: &UniformityState) -> f64 {
        let mut s = 0.0;
        for &v in vs {
            let c = state.counts[v as usize] as f64;
            s += 1.0 / (c + 1.0).sqrt();
        }
        s
    }

    fn update(&self, vs: &[u32], _signs: &[i8], state: &mut UniformityState) {
        for &v in vs {
            state.counts[v as usize] += 1;
            state.total += 1;
        }
    }

    fn rollback(&self, vs: &[u32], _signs: &[i8], state: &mut UniformityState) {
        for &v in vs {
            debug_assert!(state.counts[v as usize] > 0);
            state.counts[v as usize] -= 1;
            state.total -= 1;
        }
    }

    fn upper_bound(
        &self,
        prefix_vs: &[u32],
        k_remaining: usize,
        _k_len: usize,
        state: &UniformityState,
    ) -> f64 {
        // Prefix contributes its actual sum; future vertices best
        // case have c_v = 0 -> contribute 1/sqrt(1) = 1 each.
        let mut s = 0.0;
        for &v in prefix_vs {
            let c = state.counts[v as usize] as f64;
            s += 1.0 / (c + 1.0).sqrt();
        }
        s + k_remaining as f64
    }
}

/// Parallel global top-$K$ with an entropy-style heuristic and ABB.
///
/// Each rayon fold task maintains its own [`UniformityState`] and
/// local heap; cycles are scored relative to that task's state and
/// retained greedily.  The reduce step takes the union of per-task
/// heaps and keeps the top-$K$ by stored score.  The result is a
/// $(1 - 1/e)$ approximation per task to the optimal vertex-uniform
/// cycle set.
///
/// Returns cycles sorted by score descending.
pub fn enumerate_top_k_cycles_par_entropy<P, H>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    heuristic: &H,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    H: UniformityHeuristic,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    struct ScratchEntropy {
        heap: BinaryHeap<HeapEntry>,
        state: UniformityState,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || ScratchEntropy {
                heap: BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
                state: UniformityState::new(n),
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            },
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr,
                    &col_idx,
                    start,
                    k_len,
                    &mut s.dist,
                    &mut s.bfs_a,
                    &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_entropy(
                    start,
                    &row_ptr,
                    &col_idx,
                    &signs_csr,
                    k_len,
                    pruner,
                    k_keep,
                    heuristic,
                    &mut s.path,
                    &mut s.visited,
                    &mut s.heap,
                    &mut s.state,
                    &s.dist,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.heap)
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
            |mut a, b| {
                // Cross-task merge: take the top-K by stored score.
                // The state inconsistency across tasks is intentional
                // (the heuristic was greedy per-task; merging trades
                // exactness for parallelism).
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a.peek().map(|m| entry.cmp_preference(m) == Ordering::Greater).unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    let mut out: Vec<TopKCycle> = final_heap
        .into_iter()
        .map(|e| (e.score, e.cycle_slice().to_vec(), e.signs_slice().to_vec()))
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// SoA variant of [`enumerate_top_k_cycles_par_entropy`].  Same
/// entropy-greedy enumerator, batched output.
pub fn enumerate_top_k_cycles_par_entropy_batched<P, H>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    heuristic: &H,
) -> TopKCyclesBatch
where
    P: CyclePruner + Sync,
    H: UniformityHeuristic,
{
    if k_len < 3 || k_keep == 0 {
        return TopKCyclesBatch::new(k_len);
    }
    debug_assert!(k_len <= MAX_INLINE_K, "k_len exceeds MAX_INLINE_K");
    let (row_ptr, col_idx, signs_csr) = graph.build_csr_with_signs();
    let n = graph.n_nodes as usize;

    struct ScratchEntropy {
        heap: BinaryHeap<HeapEntry>,
        state: UniformityState,
        visited: Vec<bool>,
        path: Vec<u32>,
        dist: Vec<u8>,
        bfs_a: Vec<u32>,
        bfs_b: Vec<u32>,
    }

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || ScratchEntropy {
                heap: BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
                state: UniformityState::new(n),
                visited: vec![false; n],
                path: Vec::with_capacity(k_len),
                dist: vec![DIST_INF; n],
                bfs_a: Vec::new(),
                bfs_b: Vec::new(),
            },
            |mut s, start| {
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut s.dist, &mut s.bfs_a, &mut s.bfs_b,
                );
                s.path.clear();
                s.path.push(start);
                s.visited[start as usize] = true;
                dfs_entropy(
                    start, &row_ptr, &col_idx, &signs_csr, k_len,
                    pruner, k_keep, heuristic,
                    &mut s.path, &mut s.visited, &mut s.heap, &mut s.state, &s.dist,
                );
                s.visited[start as usize] = false;
                s
            },
        )
        .map(|s| s.heap)
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(heap_capacity_hint(k_keep)),
            |mut a, b| {
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a.peek().map(|m| entry.cmp_preference(m) == Ordering::Greater).unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    let n_out = final_heap.len();
    let mut batch = TopKCyclesBatch::with_capacity(k_len, n_out);
    for entry in final_heap {
        batch.push(entry.score, entry.cycle_slice(), entry.signs_slice());
    }
    batch.sort_by_score_desc();
    batch
}

#[allow(clippy::too_many_arguments)]
fn dfs_entropy<P, H>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    signs_csr: &[i8],
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    heuristic: &H,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    heap: &mut BinaryHeap<HeapEntry>,
    state: &mut UniformityState,
    dist: &[u8],
) where
    P: CyclePruner,
    H: UniformityHeuristic,
{
    if path.len() == k_len {
        let last = *path.last().unwrap();
        let closing_sign = match csr_sign_of(row_ptr, col_idx, signs_csr, last, start) {
            Some(s) => s,
            None => return,
        };
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        let mut signs_buf = [0i8; MAX_INLINE_K];
        for j in 0..(k_len - 1) {
            signs_buf[j] = csr_sign_of(row_ptr, col_idx, signs_csr, path[j], path[j + 1])
                .expect("interior edge present");
        }
        signs_buf[k_len - 1] = closing_sign;
        let signs: &[i8] = &signs_buf[..k_len];
        if pruner.emit_ok(path, signs) != PrunerDecision::Accept {
            return;
        }
        let s = heuristic.score(path, signs, state);
        let mut accepted = false;
        if heap.len() < k_keep {
            heap.push(HeapEntry::from_slices(s, path, signs));
            accepted = true;
        } else {
            let beat = heap.peek().map(|m| m.cmp_preference_vs_slice(s, path).is_lt()).unwrap_or(true);
            if beat {
                let popped = heap.pop().expect("heap non-empty");
                // Roll back the evicted cycle's state contribution
                // before applying the new one — keeps state in sync
                // with heap membership.
                heuristic.rollback(popped.cycle_slice(), popped.signs_slice(), state);
                heap.push(HeapEntry::from_slices(s, path, signs));
                accepted = true;
            }
        }
        if accepted {
            heuristic.update(path, signs, state);
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    let remaining_after = (k_len - path.len()) as u8;
    let threshold = if heap.len() == k_keep {
        heap.peek().map(|e| e.score).unwrap_or(f64::NEG_INFINITY)
    } else {
        f64::NEG_INFINITY
    };
    for &nxt in &col_idx[st..en] {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        // ABB on the entropy / inverse-degree UB.  Build the
        // hypothetical new prefix vertex set (path + nxt) and ask
        // the heuristic for its bound.  `path` does not yet contain
        // `nxt` — pass them as the partial-prefix slice.
        let mut prefix_buf = [0u32; MAX_INLINE_K];
        let prefix_len = path.len() + 1;
        prefix_buf[..path.len()].copy_from_slice(path);
        prefix_buf[path.len()] = nxt;
        let prefix = &prefix_buf[..prefix_len];
        let k_remaining = k_len - prefix_len;
        let ub = heuristic.upper_bound(prefix, k_remaining, k_len, state);
        if ub <= threshold {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs_entropy(
            start, row_ptr, col_idx, signs_csr, k_len, pruner, k_keep, heuristic, path, visited,
            heap, state, dist,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

// ─── Hybrid α-blended scorer (signal × diversity Pareto frontier) ─
//
// `score(c, state) = α · signal(c) + (1 - α) · ΔH(c | C)` where
// `signal: BoundedScorer` is the existing global-ABB scorer family
// (e.g. fraction_negative) and `ΔH: UniformityHeuristic` is the
// existing entropy or inverse-degree heuristic.  α = 0 collapses
// to pure diversity; α = 1 collapses to pure signal (with the
// entropy DFS's greedy-with-rollback semantics — output may differ
// at score ties from the BoundedScorer-only path).
//
// Plan + admissibility derivation:
// `docs/plans/2026-05-10-hybrid-alpha-scorer/plan.{tex,pdf,tikz,mmd}`.

/// Linear blend of a signal scorer and a diversity heuristic.
///
/// # Score
///
/// `score = α · signal(c) + (1 - α) · diversity(c | state)`
///
/// # Admissibility (linearity-of-UB)
///
/// `UB_hybrid = α · UB_signal + (1 - α) · UB_div`.  Admissible by
/// linearity: every reachable closed cycle's score is at most the
/// linear combination of the component upper bounds.
///
/// # Signal UB call convention
///
/// `BoundedScorer::upper_bound` takes `(n_neg_so_far, k_remaining,
/// k_len)` but the `UniformityHeuristic::upper_bound` signature only
/// carries `prefix_vs` (no signs).  This impl passes
/// `n_neg_so_far = prefix_vs.len().saturating_sub(1)` — the
/// worst-case number of negative edges committed so far (every
/// prefix edge is negative).  This gives the **loosest** admissible
/// signal UB; a follow-up plan can thread the actual running n_neg
/// through `dfs_entropy` for a tighter bound.
pub struct HybridScorer<B, H>
where
    B: BoundedScorer,
    H: UniformityHeuristic,
{
    /// Stateless signal component (e.g. [`FractionNegativeScorer`]).
    pub signal: B,
    /// State-dependent diversity component (e.g.
    /// [`EntropyGainScorer`] or [`InverseDegreeScorer`]).
    pub diversity: H,
    /// Mixing weight in `[0, 1]`.  `α = 0` → pure diversity;
    /// `α = 1` → pure signal.
    pub alpha: f64,
}

impl<B, H> HybridScorer<B, H>
where
    B: BoundedScorer,
    H: UniformityHeuristic,
{
    /// Construct a hybrid scorer.  Panics in debug builds if `alpha`
    /// is outside `[0, 1]` (the linear combo's admissibility argument
    /// requires non-negative weights summing to 1).
    #[inline]
    pub fn new(signal: B, diversity: H, alpha: f64) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&alpha),
            "HybridScorer alpha must lie in [0, 1]; got {alpha}",
        );
        Self {
            signal,
            diversity,
            alpha,
        }
    }
}

impl<B, H> UniformityHeuristic for HybridScorer<B, H>
where
    B: BoundedScorer,
    H: UniformityHeuristic,
{
    fn score(&self, vs: &[u32], signs: &[i8], state: &UniformityState) -> f64 {
        let s_signal = self.signal.score(vs, signs);
        let s_div = self.diversity.score(vs, signs, state);
        self.alpha * s_signal + (1.0 - self.alpha) * s_div
    }

    fn update(&self, vs: &[u32], signs: &[i8], state: &mut UniformityState) {
        // Only the diversity component owns state; signal is stateless.
        self.diversity.update(vs, signs, state);
    }

    fn rollback(&self, vs: &[u32], signs: &[i8], state: &mut UniformityState) {
        self.diversity.rollback(vs, signs, state);
    }

    fn upper_bound(
        &self,
        prefix_vs: &[u32],
        k_remaining: usize,
        k_len: usize,
        state: &UniformityState,
    ) -> f64 {
        // Worst-case (admissible) n_neg_so_far = every prefix edge
        // is negative.  Prefix has prefix_vs.len() vertices and
        // prefix_vs.len() - 1 edges (the cycle hasn't closed yet).
        let n_neg_worst = prefix_vs.len().saturating_sub(1);
        let ub_signal = self.signal.upper_bound(n_neg_worst, k_remaining, k_len);
        let ub_div = self
            .diversity
            .upper_bound(prefix_vs, k_remaining, k_len, state);
        self.alpha * ub_signal + (1.0 - self.alpha) * ub_div
    }
}

// ─── Builder façade over the enumerate_top_k_cycles* family ───────
//
// Eight free entry points exist (global vs per-vertex × ABB vs
// not × sequential vs parallel × pruned vs noprune wrappers).
// `TopKBuilder` is a fluent terminal-method API that organizes
// the same algorithms behind a single configuration object, per
// CLAUDE.md §7's Builder pattern guidance — useful when the
// caller picks variants dynamically (e.g. a Python wheel
// dispatching by env-var).  All terminals use the rayon-parallel
// path internally; serial dispatch is accessible as a fold over
// `RAYON_NUM_THREADS=1`.  The free functions remain in place so
// existing call sites and rollback diffs are unaffected.

/// Fluent builder for the top-$K$ cycle enumeration family.
///
/// Construct with [`TopKBuilder::new`], then choose a terminal:
///
/// - [`TopKBuilder::global`] — global top-$K$ with a plain
///   `Fn(&[u32], &[i8]) -> f64` scorer, no ABB.
/// - [`TopKBuilder::global_bb`] — global top-$K$ with a
///   [`BoundedScorer`], score upper-bound branch-and-bound enabled.
/// - [`TopKBuilder::per_vertex`] — per-vertex top-$m$, no ABB
///   (the per-vertex path's threshold structure makes uniform-bound
///   ABB ineffective at small $m$ on long-tail graphs; see
///   `examples/probe_per_vertex_thresholds.rs`).
///
/// Each terminal calls into the corresponding `_par` free function
/// and returns the same `Vec<TopKCycle>` output.
pub struct TopKBuilder<'a, P: CyclePruner + Sync> {
    graph: &'a SignedGraph,
    k_len: usize,
    pruner: &'a P,
}

impl<'a, P: CyclePruner + Sync> TopKBuilder<'a, P> {
    /// Begin a builder for cycles of length `k_len` on `graph`,
    /// with `pruner` consulted at every DFS extension and every
    /// closed-cycle emission.
    #[inline]
    pub fn new(graph: &'a SignedGraph, k_len: usize, pruner: &'a P) -> Self {
        Self {
            graph,
            k_len,
            pruner,
        }
    }

    /// Global top-$K$ with a plain `Fn` scorer (no ABB).
    /// Equivalent to [`enumerate_top_k_cycles_par`].
    #[inline]
    pub fn global<S>(self, k_keep: usize, score: S) -> Vec<TopKCycle>
    where
        S: Fn(&[u32], &[i8]) -> f64 + Sync,
    {
        enumerate_top_k_cycles_par(self.graph, self.k_len, self.pruner, k_keep, score)
    }

    /// Global top-$K$ with a [`BoundedScorer`] — enables ABB
    /// (score-upper-bound branch-and-bound).  Equivalent to
    /// [`enumerate_top_k_cycles_par_bb`].
    ///
    /// On Epinions $k{=}4$, $K{=}10\,000$, balance pruner +
    /// `FractionNegativeScorer`, ABB delivers a 25× wall-time
    /// reduction (`reports/2026-05-10-abb-global-topk.md`).
    #[inline]
    pub fn global_bb<S>(self, k_keep: usize, scorer: &S) -> Vec<TopKCycle>
    where
        S: BoundedScorer,
    {
        enumerate_top_k_cycles_par_bb(self.graph, self.k_len, self.pruner, k_keep, scorer)
    }

    /// Per-vertex top-$m$ — keep the top-$m$ cycles passing through
    /// each vertex; output is the deduplicated union of the
    /// per-vertex sets.  Equivalent to
    /// [`enumerate_top_k_per_vertex_cycles_par`].
    #[inline]
    pub fn per_vertex<S>(self, m_per_vertex: usize, score: S) -> Vec<TopKCycle>
    where
        S: Fn(&[u32], &[i8]) -> f64 + Sync,
    {
        enumerate_top_k_per_vertex_cycles_par(
            self.graph,
            self.k_len,
            self.pruner,
            m_per_vertex,
            score,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balance::{BalanceMode, CartwrightHararyPruner};
    use crate::pruner::NoOpPruner;
    use crate::signed_graph::SignedGraph;

    #[test]
    fn tiered_m_v_by_degree_assigns_hub_to_top_tier() {
        // Star graph: vertex 0 connected to 1..9.  deg(0)=9; rest deg=1.
        // n=10 vertices, so top 10% (1 vertex) = vertex 0.
        let g = SignedGraph {
            n_nodes: 10,
            edges: vec![
                (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (0, 6), (0, 7), (0, 8), (0, 9),
            ],
            signs: vec![1; 9],
        };
        let tiers = vec![(10.0_f32, 1024_u32), (100.0_f32, 16_u32)];
        let m_v = tiered_m_v_by_degree(&g, &tiers);
        assert_eq!(m_v.len(), 10);
        assert_eq!(m_v[0], 1024, "hub vertex 0 must land in top tier");
        for (v, &cap) in m_v.iter().enumerate().skip(1) {
            assert_eq!(cap, 16, "leaf vertex {v} must land in bottom tier");
        }
    }

    #[test]
    fn tiered_m_v_by_degree_skips_when_top_tier_cap_zero() {
        // Bottom-tier cap=0 should give vertices zero cycles (skipped).
        let g = SignedGraph {
            n_nodes: 5,
            edges: vec![(0, 1), (0, 2), (1, 2), (3, 4)],
            signs: vec![1; 4],
        };
        // Top 20% (1 vertex of 5) gets cap=128; rest skipped.
        let tiers = vec![(20.0_f32, 128_u32), (100.0_f32, 0_u32)];
        let m_v = tiered_m_v_by_degree(&g, &tiers);
        assert_eq!(m_v.len(), 5);
        // Vertex 0 has deg=2 (highest tied with 1,2); top tier
        // captures top-20% by degree, which is at most 1 vertex.
        // The function picks based on ascending tier order so any
        // vertex with deg >= top-threshold is in top tier.
        let in_top: Vec<usize> =
            (0..5).filter(|&v| m_v[v] == 128).collect();
        let in_bot: Vec<usize> =
            (0..5).filter(|&v| m_v[v] == 0).collect();
        // At least one vertex in each tier; total = 5
        assert_eq!(in_top.len() + in_bot.len(), 5);
        assert!(!in_top.is_empty(), "top tier must be non-empty");
    }

    #[test]
    fn tiered_m_v_by_degree_empty_graph() {
        let g = SignedGraph {
            n_nodes: 0,
            edges: vec![],
            signs: vec![],
        };
        let tiers = vec![(100.0_f32, 16_u32)];
        let m_v = tiered_m_v_by_degree(&g, &tiers);
        assert!(m_v.is_empty());
    }

    #[test]
    fn tiered_m_v_by_degree_single_tier_uniform_cap() {
        // Single tier covering 100% = uniform cap for all vertices.
        let g = SignedGraph {
            n_nodes: 4,
            edges: vec![(0, 1), (1, 2), (2, 3)],
            signs: vec![1; 3],
        };
        let tiers = vec![(100.0_f32, 32_u32)];
        let m_v = tiered_m_v_by_degree(&g, &tiers);
        for (v, &cap) in m_v.iter().take(4).enumerate() {
            assert_eq!(cap, 32, "all vertices should get cap=32 (v={v})");
        }
    }

    #[test]
    fn weighted_sum_scorer_admissibility() {
        // For any fixture, WeightedSumScorer's UB must be >= its
        // score (admissibility postcondition of BoundedScorer).
        let comp = WeightedSumScorer {
            a: 0.7, s1: FractionNegativeScorer,
            b: 0.3, s2: SignProductAbsScorer,
        };
        let vs = [0u32, 1, 2];
        for signs in &[[1i8,-1,1], [-1,-1,-1], [1,1,1], [-1,1,-1]] {
            let s = comp.score(&vs, signs);
            let n_neg = signs.iter().filter(|&&x| x < 0).count();
            // Fully-known cycle: k_remaining=0, n_neg_so_far=n_neg.
            let ub = comp.upper_bound(n_neg, 0, signs.len());
            assert!(ub >= s - 1e-12,
                "UB {} < score {} for signs {:?}", ub, s, signs);
        }
    }

    #[test]
    fn weighted_sum_scorer_partial_path_ub() {
        // Partial-path UB must also dominate any closing score.
        // Take k=4, partial path of length 2 with 1 negative so far.
        let comp = WeightedSumScorer {
            a: 1.0, s1: FractionNegativeScorer,
            b: 0.5, s2: BalanceScorer,
        };
        let ub = comp.upper_bound(1, 2, 4);
        // For every possible completion (4 cycles with 1 neg already):
        let vs = [0u32, 1, 2, 3];
        for s2 in &[1i8, -1] {
            for s3 in &[1i8, -1] {
                let signs = [-1, 1, *s2, *s3];  // 1 neg upfront
                let s = comp.score(&vs, &signs);
                assert!(ub >= s - 1e-12,
                    "partial UB {} < score {} for signs {:?}", ub, s, signs);
            }
        }
    }

    #[test]
    fn per_vertex_bb_global_huge_caps_matches_non_bb() {
        // With huge caps, no heap ever fills → global_min stays at
        // +INFINITY → ABB never fires → output matches non-ABB.
        let g = SignedGraph {
            n_nodes: 6,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (2, 3),
            ],
            signs: vec![1, -1, 1, 1, 1, -1, 1],
        };
        let huge = vec![10_000u32; g.n_nodes as usize];
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let non_bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &huge, &starting,
            |c: &[u32], s: &[i8]| {
                let nn = s.iter().filter(|&&x| x < 0).count() as f64;
                nn / c.len() as f64
            },
        );
        let bb_g = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched(
            &g, 3, &NoOpPruner, &huge, &starting,
            &FractionNegativeScorer, 0.0,
        );
        assert_eq!(non_bb.len(), bb_g.len());
    }

    #[test]
    fn per_vertex_bb_global_subset_of_non_bb() {
        // Small caps + dense fixture → heaps fill → global ABB fires.
        // Must still produce a SUBSET of non-ABB.
        let g = SignedGraph {
            n_nodes: 8,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (0, 3), (1, 4), (2, 5),
                (6, 7),
            ],
            signs: vec![1, 1, -1, 1, -1, 1, -1, 1, 1, 1],
        };
        let m_v = vec![2u32; g.n_nodes as usize];  // tiny caps
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let non_bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            |c: &[u32], s: &[i8]| {
                let nn = s.iter().filter(|&&x| x < 0).count() as f64;
                nn / c.len() as f64
            },
        );
        let bb_g = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            &FractionNegativeScorer, 0.0,  // gate=0 → always on
        );
        use std::collections::HashSet;
        let canon = |b: &TopKCyclesBatch| -> HashSet<Vec<u32>> {
            (0..b.len()).map(|i| {
                let s = i * b.k;
                let e = s + b.k;
                let mut k = b.cycles[s..e].to_vec();
                k.sort_unstable();
                k
            }).collect()
        };
        let non_bb_set = canon(&non_bb);
        let bb_g_set = canon(&bb_g);
        assert!(
            bb_g_set.is_subset(&non_bb_set),
            "global-min ABB output ({}) must be a subset of non-ABB ({})",
            bb_g_set.len(), non_bb_set.len(),
        );
    }

    #[test]
    fn per_vertex_bb_global_gate_normalised_to_active_cap() {
        // CPG-style: half the vertices have cap=0 (zero-cap leaves).
        // With gate=1.0, the threshold should be the count of
        // non-zero-cap (active) vertices, NOT total n_nodes.  This
        // makes gate=1.0 fire when ALL active heaps are full.
        let g = SignedGraph {
            n_nodes: 8,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (0, 3), (1, 4), (2, 5),
                (6, 7),
            ],
            signs: vec![1, 1, -1, 1, -1, 1, -1, 1, 1, 1],
        };
        // Top 6 vertices: cap=2.  Bottom 2 (cap-0 vertices 6,7).
        let m_v = vec![2u32, 2, 2, 2, 2, 2, 0, 0];
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let non_bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            |c: &[u32], s: &[i8]| {
                let nn = s.iter().filter(|&&x| x < 0).count() as f64;
                nn / c.len() as f64
            },
        );
        // gate=1.0 with the fix: threshold = 6 (n_active), not 8.
        // ABB should still produce a SUBSET of non-ABB output.
        let bb_g = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            &FractionNegativeScorer, 1.0,
        );
        use std::collections::HashSet;
        let canon = |b: &TopKCyclesBatch| -> HashSet<Vec<u32>> {
            (0..b.len()).map(|i| {
                let s = i * b.k;
                let e = s + b.k;
                let mut k = b.cycles[s..e].to_vec();
                k.sort_unstable();
                k
            }).collect()
        };
        assert!(canon(&bb_g).is_subset(&canon(&non_bb)),
            "ABB output ({}) must be subset of non-ABB ({})",
            bb_g.len(), non_bb.len(),
        );
    }

    #[test]
    fn per_vertex_bb_global_with_full_gate_disables_abb() {
        // fullness_gate=1.0 → fullness_gate_count = n_nodes → ABB
        // never activates until ALL heaps full (rarely happens).
        // Output should match non-ABB in most fixtures.
        let g = SignedGraph {
            n_nodes: 6,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (2, 3),
            ],
            signs: vec![1, -1, 1, 1, 1, -1, 1],
        };
        let m_v = vec![2u32; g.n_nodes as usize];
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let non_bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            |c: &[u32], s: &[i8]| {
                let nn = s.iter().filter(|&&x| x < 0).count() as f64;
                nn / c.len() as f64
            },
        );
        let bb_g = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            &FractionNegativeScorer, 1.0,  // gate=1.0 → effectively off
        );
        // With gate=1.0, ABB never activates in typical fixtures.
        // The output should match non-ABB.
        assert_eq!(non_bb.len(), bb_g.len());
    }

    #[test]
    fn per_vertex_bb_batched_subset_of_non_bb() {
        // ABB-pruned output must be a SUBSET of the non-ABB output
        // (ABB sacrifices may-have-helped-other-heaps cycles for
        // speed; never invents new ones).
        let g = SignedGraph {
            n_nodes: 8,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (0, 3), (1, 4), (2, 5),
                (6, 7),
            ],
            signs: vec![1, 1, -1, 1, -1, 1, -1, 1, 1, 1],
        };
        let m_v = vec![4u32; g.n_nodes as usize];
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let non_bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            |c: &[u32], s: &[i8]| {
                let nn = s.iter().filter(|&&x| x < 0).count() as f64;
                nn / c.len() as f64
            },
        );
        let bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched(
            &g, 3, &NoOpPruner, &m_v, &starting,
            &FractionNegativeScorer,
        );
        use std::collections::HashSet;
        let canon = |b: &TopKCyclesBatch| -> HashSet<Vec<u32>> {
            (0..b.len())
                .map(|i| {
                    let s = i * b.k;
                    let e = s + b.k;
                    let mut k = b.cycles[s..e].to_vec();
                    k.sort_unstable();
                    k
                })
                .collect()
        };
        // non_bb is already a TopKCyclesBatch (SoA path).
        let non_bb_set = canon(&non_bb);
        let bb_set = canon(&bb);
        assert!(
            bb_set.is_subset(&non_bb_set),
            "ABB output ({}) must be a subset of non-ABB ({})",
            bb_set.len(), non_bb_set.len(),
        );
    }

    #[test]
    fn per_vertex_bb_batched_empty_starting_returns_empty() {
        let g = SignedGraph {
            n_nodes: 3,
            edges: vec![(0, 1), (1, 2), (0, 2)],
            signs: vec![1; 3],
        };
        let m_v = vec![16u32; 3];
        let bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched(
            &g, 3, &NoOpPruner, &m_v, &[], &FractionNegativeScorer,
        );
        assert!(bb.is_empty());
        assert_eq!(bb.k, 3);
    }

    #[test]
    fn per_vertex_bb_with_huge_caps_matches_non_bb() {
        // With huge per-vertex caps, ABB threshold stays at -∞ and
        // ABB is a no-op → output should match non-ABB bit-for-bit
        // on cycle SET (counts/scores).
        let g = SignedGraph {
            n_nodes: 6,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (2, 3),
            ],
            signs: vec![1, -1, 1, 1, 1, -1, 1],
        };
        let huge = vec![10_000u32; g.n_nodes as usize];
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let non_bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &huge, &starting,
            |c: &[u32], s: &[i8]| {
                let nn = s.iter().filter(|&&x| x < 0).count() as f64;
                nn / c.len() as f64
            },
        );
        let bb = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched(
            &g, 3, &NoOpPruner, &huge, &starting, &FractionNegativeScorer,
        );
        // At huge caps, heaps never fill → ABB threshold = -∞ → no pruning.
        assert_eq!(non_bb.len(), bb.len(),
                   "huge-cap ABB must match non-ABB cycle count");
    }

    #[test]
    fn batched_variant_produces_identical_cycle_set() {
        // Build a small fixture with 6 triangles + bridges.
        let g = SignedGraph {
            n_nodes: 10,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (6, 7), (7, 8), (6, 8),
                (2, 3), (5, 6),
                (0, 9),
            ],
            signs: vec![1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1],
        };
        let m_v = vec![16u32; g.n_nodes as usize];
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let score = |c: &[u32], s: &[i8]| {
            let n_neg = s.iter().filter(|&&x| x < 0).count() as f64;
            n_neg / c.len() as f64
        };

        let legacy = enumerate_top_k_per_vertex_cycles_par_adaptive_starting(
            &g, 3, &NoOpPruner, &m_v, &starting, score,
        );
        let batched = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &m_v, &starting, score,
        );

        // Same cycle count.
        assert_eq!(
            legacy.len(),
            batched.len(),
            "legacy and batched must return same cycle count"
        );

        // Same cycle set (compare as canonical-sorted vertex tuples).
        use std::collections::HashSet;
        let legacy_set: HashSet<Vec<u32>> = legacy
            .iter()
            .map(|(_, c, _)| {
                let mut k = c.clone();
                k.sort_unstable();
                k
            })
            .collect();
        let batched_set: HashSet<Vec<u32>> = (0..batched.len())
            .map(|i| {
                let s = i * batched.k;
                let e = s + batched.k;
                let mut k = batched.cycles[s..e].to_vec();
                k.sort_unstable();
                k
            })
            .collect();
        assert_eq!(
            legacy_set, batched_set,
            "legacy and batched cycle SETS must be identical"
        );

        // Scores must match for matching cycles (within fp tolerance).
        // (Both paths sort by score-desc so row i ↔ row i.)
        for (i, leg) in legacy.iter().enumerate() {
            assert!(
                (leg.0 - batched.scores[i]).abs() < 1e-12,
                "row {i} score mismatch: legacy={} batched={}",
                leg.0,
                batched.scores[i]
            );
        }
    }

    #[test]
    fn batched_empty_starting_returns_empty_batch() {
        let g = SignedGraph {
            n_nodes: 3,
            edges: vec![(0, 1), (1, 2), (0, 2)],
            signs: vec![1; 3],
        };
        let m_v = vec![16u32; 3];
        let score = |_c: &[u32], _s: &[i8]| 0.0;
        let batched = enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, 3, &NoOpPruner, &m_v, &[], score,
        );
        assert!(batched.is_empty());
        assert_eq!(batched.k, 3);
    }

    #[test]
    fn batch_into_vec_topkcycle_round_trip() {
        let mut batch = TopKCyclesBatch::new(3);
        batch.push(0.5, &[1u32, 2, 3], &[1i8, 1, -1]);
        batch.push(0.9, &[4u32, 5, 6], &[-1i8, 1, 1]);
        let cycles = batch.into_vec_topkcycle();
        assert_eq!(cycles.len(), 2);
        assert_eq!(cycles[0].0, 0.5);
        assert_eq!(cycles[0].1, vec![1u32, 2, 3]);
        assert_eq!(cycles[0].2, vec![1i8, 1, -1]);
        assert_eq!(cycles[1].0, 0.9);
        assert_eq!(cycles[1].1, vec![4u32, 5, 6]);
    }

    #[test]
    fn batch_sort_by_score_desc_permutes_all_arrays_consistently() {
        let mut batch = TopKCyclesBatch::new(2);
        batch.push(0.1, &[1u32, 2], &[1i8, 1]);
        batch.push(0.9, &[3u32, 4], &[-1i8, 1]);
        batch.push(0.5, &[5u32, 6], &[1i8, -1]);
        batch.sort_by_score_desc();
        assert_eq!(batch.scores, vec![0.9, 0.5, 0.1]);
        assert_eq!(batch.cycles, vec![3u32, 4, 5, 6, 1, 2]);
        assert_eq!(batch.signs, vec![-1i8, 1, 1, -1, 1, 1]);
    }

    #[test]
    fn tiered_m_v_by_degree_works_with_par_adaptive_starting() {
        // End-to-end: build tiered m_v, then enumerate.
        // 4 connected triangles + isolated vertex.
        let g = SignedGraph {
            n_nodes: 7,
            edges: vec![
                (0, 1), (1, 2), (0, 2),
                (3, 4), (4, 5), (3, 5),
                (0, 3),
            ],
            signs: vec![1; 7],
        };
        // 50% (top 3-4 vertices) get cap 16; rest get cap 4.
        let tiers = vec![(50.0_f32, 16_u32), (100.0_f32, 4_u32)];
        let m_v = tiered_m_v_by_degree(&g, &tiers);
        let starting: Vec<u32> = (0..g.n_nodes).collect();
        let score = |_c: &[u32], _s: &[i8]| 0.0;
        let out = enumerate_top_k_per_vertex_cycles_par_adaptive_starting(
            &g, 3, &NoOpPruner, &m_v, &starting, score,
        );
        // Should find both triangles.
        let has_t1 = out.iter().any(|(_, c, _)| {
            let mut k = c.clone();
            k.sort_unstable();
            k == vec![0, 1, 2]
        });
        let has_t2 = out.iter().any(|(_, c, _)| {
            let mut k = c.clone();
            k.sort_unstable();
            k == vec![3, 4, 5]
        });
        assert!(has_t1, "triangle (0,1,2) must be found");
        assert!(has_t2, "triangle (3,4,5) must be found");
    }

    /// Build a graph with three triangles of known balance:
    ///   Δ_pos:  signs (+, +, +) ⇒ balanced.
    ///   Δ_mix1: signs (+, +, -) ⇒ unbalanced.
    ///   Δ_mix2: signs (+, -, -) ⇒ balanced.
    fn build_three_triangles() -> SignedGraph {
        // Triangles: 0-1-2, 3-4-5, 6-7-8.
        SignedGraph::from_parts(
            9,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
            &[1, 2, 0, 4, 5, 3, 7, 8, 6],
            &[1, 1, 1, 1, 1, -1, 1, -1, -1],
        )
    }

    #[test]
    fn top_k_keeps_only_k_cycles() {
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles_noprune(&g, 3, 2, scorers::balance);
        assert_eq!(out.len(), 2);
        // Both balanced triangles should win.
        assert!(out.iter().all(|(s, _, _)| (*s - 1.0).abs() < 1e-9));
    }

    #[test]
    fn top_k_full_request_returns_all() {
        // Asking for k_keep = 5 when only 3 cycles exist returns all 3.
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles_noprune(&g, 3, 5, scorers::balance);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn top_k_descending_order() {
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles_noprune(&g, 3, 3, scorers::fraction_negative);
        assert_eq!(out.len(), 3);
        // fraction_negative: Δ_mix2 (2/3), Δ_mix1 (1/3), Δ_pos (0).
        assert!(out[0].0 >= out[1].0);
        assert!(out[1].0 >= out[2].0);
    }

    #[test]
    fn top_k_with_pruner_composes() {
        // Same graph, ask Cartwright-Harary OnlyBalanced + top-1 by
        // low_root: should return the lowest-rooted balanced
        // triangle (Δ_pos at vertex 0).
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles(
            &g,
            3,
            &CartwrightHararyPruner {
                mode: BalanceMode::OnlyBalanced,
            },
            1,
            scorers::low_root,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, vec![0, 1, 2]);
    }

    #[test]
    fn per_vertex_top_k_covers_every_vertex() {
        // Graph with two disjoint triangles 0-1-2 and 3-4-5.
        let g = SignedGraph::from_parts(6, &[0, 1, 2, 3, 4, 5], &[1, 2, 0, 4, 5, 3], &[1; 6]);
        let out = enumerate_top_k_per_vertex_cycles_noprune(&g, 3, 1, scorers::balance);
        // Two triangles, two unique cycles.
        assert_eq!(out.len(), 2);
        // Every vertex appears in at least one returned cycle.
        let mut touched = [false; 6];
        for (_, vs, _) in &out {
            for &v in vs {
                touched[v as usize] = true;
            }
        }
        assert!(
            touched.iter().all(|&b| b),
            "vertex-stratified top-K must cover every vertex"
        );
    }

    #[test]
    fn per_vertex_top_k_dedups_shared_cycles() {
        // A single triangle 0-1-2: every vertex's heap will hold the
        // same cycle, but the union must dedup to one entry.
        let g = SignedGraph::from_parts(3, &[0, 1, 2], &[1, 2, 0], &[1; 3]);
        let out = enumerate_top_k_per_vertex_cycles_noprune(&g, 3, 5, scorers::balance);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn top_k_min_vertex_weight_picks_cheapest() {
        // 4-vertex graph with two triangles 0-1-2 and 1-2-3.
        // Make vertex weights so 1-2-3 is cheaper: w = [10, 1, 1, 1].
        let g = SignedGraph::from_parts(4, &[0, 1, 2, 1, 2], &[1, 2, 0, 3, 3], &[1; 5]);
        let weights = vec![10.0, 1.0, 1.0, 1.0];
        let scorer = scorers::min_vertex_weight(weights);
        let out = enumerate_top_k_cycles(&g, 3, &NoOpPruner, 1, scorer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, vec![1, 2, 3]);
    }

    // ─── Builder façade tests ─────────────────────────────────────

    fn fixture_for_builder() -> SignedGraph {
        // 4 triangles (0-1-2), (0-2-3), (1-2-3), (0-1-3) on a
        // 4-clique with mixed signs.  Enough cycles to make the
        // top-K choice non-trivial.
        SignedGraph::from_parts(
            4,
            &[0, 0, 0, 1, 1, 2],
            &[1, 2, 3, 2, 3, 3],
            &[1, -1, 1, 1, -1, -1],
        )
    }

    #[test]
    fn builder_global_matches_free_function() {
        let g = fixture_for_builder();
        let pruner = NoOpPruner;
        let direct = enumerate_top_k_cycles_par(&g, 3, &pruner, 4, scorers::fraction_negative);
        let via_builder = TopKBuilder::new(&g, 3, &pruner).global(4, scorers::fraction_negative);
        assert_eq!(direct.len(), via_builder.len());
        // Identical (score, vertices, signs) multiset (the free
        // function is what the builder calls; tie-breaking is the
        // same).
        let mut direct_sorted: Vec<(u32, u32, u32)> = direct
            .iter()
            .map(|(_, vs, _)| {
                let mut v = vs.clone();
                v.sort();
                (v[0], v[1], v[2])
            })
            .collect();
        let mut builder_sorted: Vec<(u32, u32, u32)> = via_builder
            .iter()
            .map(|(_, vs, _)| {
                let mut v = vs.clone();
                v.sort();
                (v[0], v[1], v[2])
            })
            .collect();
        direct_sorted.sort();
        builder_sorted.sort();
        assert_eq!(direct_sorted, builder_sorted);
    }

    #[test]
    fn builder_global_bb_matches_free_function() {
        let g = fixture_for_builder();
        let pruner = CartwrightHararyPruner {
            mode: BalanceMode::OnlyBalanced,
        };
        let direct = enumerate_top_k_cycles_par_bb(&g, 3, &pruner, 4, &FractionNegativeScorer);
        let via_builder = TopKBuilder::new(&g, 3, &pruner).global_bb(4, &FractionNegativeScorer);
        assert_eq!(direct.len(), via_builder.len());
        let direct_scores: Vec<f64> = direct.iter().map(|c| c.0).collect();
        let builder_scores: Vec<f64> = via_builder.iter().map(|c| c.0).collect();
        assert_eq!(direct_scores, builder_scores);
    }

    #[test]
    fn builder_per_vertex_matches_free_function() {
        let g = fixture_for_builder();
        let pruner = NoOpPruner;
        let direct =
            enumerate_top_k_per_vertex_cycles_par(&g, 3, &pruner, 2, scorers::fraction_negative);
        let via_builder =
            TopKBuilder::new(&g, 3, &pruner).per_vertex(2, scorers::fraction_negative);
        assert_eq!(direct.len(), via_builder.len());
    }
}
