//! Adaptive quadtree with Forman-Ricci κ on the 4-connected same-depth
//! graph + variance-driven subdivision.
//!
//! Migrated from `hymeko_py/src/quadtree.rs` (2026-05-22 Phase 6.5
//! anti-pattern #2 cleanup): the per-depth subdivision state machine,
//! the Forman κ computation, and the unit tests are pure-Rust and
//! belong in this algorithm crate.  The PyO3 wrapper in `hymeko_py`
//! is now a thin layer that adapts a Python scoring callable to the
//! [`build_quadtree`] closure signature below.
//!
//! ## Background
//!
//! Used by the GömbSoma vision pipeline to drive adaptive spatial
//! decomposition of natural images: the per-depth frontier is
//! variance-scored (typically via `torchvision.ops.roi_align` on
//! the GPU caller-side); the curvature score is a closed-form Forman
//! κ over the 4-connected grid of same-depth anchors (no triangles in
//! a 4-conn grid, so `edge_κ = 2 − d_u − d_v` exactly).
//!
//! Plan: `docs/plans/2026-05-16-gomb-soma-quadtree-triton/plan.tex`.
//!
//! ## Public API
//!
//! [`build_quadtree`] takes a `score_fn` closure rather than a Python
//! callable so the algorithm can be used from any Rust caller without
//! a Python interpreter.  The closure is invoked once per depth with
//! the current frontier's (row, col) positions and patch sizes; it
//! must return a per-anchor variance score.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────
// Forman-Ricci κ on the 4-connected same-depth grid
// ─────────────────────────────────────────────────────────────────────

/// Compute Forman vertex κ on a 4-connected same-depth grid.
///
/// `frontier_positions[i] = (row, col)`; `frontier_size` is the patch
/// side length common to every frontier anchor at one depth.  Returns
/// the per-anchor vertex κ vector (length = `frontier.len()`), matching
/// the `FormanCurvatureHead.vertex_kappa` semantics in the Python
/// reference: mean over incident edges of `edge_κ = 2 − deg(u) − deg(v)`
/// (no triangles in a 4-conn grid).
///
/// ```text
/// vertex_kappa[v] = (1 / deg(v)) · Σ_{u ∈ N(v)} (2 − deg(v) − deg(u))
/// ```
/// when `deg(v) > 0`; `0` otherwise.
pub fn forman_kappa_4conn(
    frontier_positions: &[(i64, i64)],
    frontier_size: i64,
) -> Vec<f32> {
    let n = frontier_positions.len();
    if n == 0 {
        return Vec::new();
    }

    // Spatial-hash lookup: (row, col) → local frontier index.  All
    // anchors at the same depth share `frontier_size`.
    let mut pos_to_local: HashMap<(i64, i64), usize> =
        HashMap::with_capacity(n);
    for (li, &(r, c)) in frontier_positions.iter().enumerate() {
        pos_to_local.insert((r, c), li);
    }

    // Compute degree + neighbour list per anchor (≤ 4 each).
    let s = frontier_size;
    let mut neighbours: Vec<arrayvec_lite::ArrayVec4> =
        vec![arrayvec_lite::ArrayVec4::new(); n];
    let mut degree: Vec<u8> = vec![0; n];
    for (li, &(r, c)) in frontier_positions.iter().enumerate() {
        for (dr, dc) in [(s, 0i64), (-s, 0), (0, s), (0, -s)] {
            if let Some(&lj) = pos_to_local.get(&(r + dr, c + dc)) {
                neighbours[li].push(lj);
                degree[li] += 1;
            }
        }
    }

    // Vertex κ = mean over incident edges.
    let mut kappa: Vec<f32> = vec![0.0; n];
    for li in 0..n {
        let d_u = degree[li] as i32;
        if d_u == 0 {
            continue;
        }
        let mut acc: i32 = 0;
        for k in 0..(degree[li] as usize) {
            let lj = neighbours[li].get(k);
            let d_v = degree[lj] as i32;
            acc += 2 - d_u - d_v;
        }
        kappa[li] = acc as f32 / d_u as f32;
    }
    kappa
}

mod arrayvec_lite {
    //! Tiny fixed-capacity inline vector to avoid Vec allocations for
    //! the per-vertex neighbour list.  A 4-connected grid means at most
    //! 4 neighbours per vertex; full `ArrayVec` semantics are overkill.

    #[derive(Clone)]
    pub struct ArrayVec4 {
        data: [usize; 4],
        len: u8,
    }
    impl ArrayVec4 {
        pub fn new() -> Self {
            Self { data: [0; 4], len: 0 }
        }
        pub fn push(&mut self, v: usize) {
            debug_assert!(self.len < 4);
            self.data[self.len as usize] = v;
            self.len += 1;
        }
        pub fn get(&self, i: usize) -> usize {
            debug_assert!((i as u8) < self.len);
            self.data[i]
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Adaptive subdivision state machine
// ─────────────────────────────────────────────────────────────────────

/// Output of [`build_quadtree`].
///
/// Each vector is parallel-indexed: anchor `i` has position
/// `positions[i]`, side length `sizes[i]`, depth `scales[i]`, and
/// `parent_indices[i]` is the index of its parent anchor in this same
/// output (or `-1` for the scale-0 base tiling).
#[derive(Debug, Clone)]
pub struct QuadtreeAnchors {
    /// `(row, col)` top-left position of each anchor in image coordinates.
    pub positions:      Vec<(i64, i64)>,
    /// Side length (square anchors) in pixels.
    pub sizes:          Vec<i64>,
    /// Subdivision depth: depth `d` anchors have side `patch_size_initial / 2^d`.
    pub scales:         Vec<i64>,
    /// Index into this same set of vectors of the anchor that was
    /// subdivided to produce this one (or `-1` for the depth-0 base tiling).
    pub parent_indices: Vec<i64>,
}

impl QuadtreeAnchors {
    /// Number of anchors emitted across all depths.
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    /// True iff no anchors were emitted (e.g. zero-sized image).
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Build an adaptive quadtree by per-depth variance + curvature
/// scoring.  Returns parallel-indexed `positions / sizes / scales /
/// parent_indices` vectors wrapped in [`QuadtreeAnchors`].
///
/// # Arguments
///
/// - `image_h`, `image_w`: image dimensions.  Must be divisible by
///   `patch_size_initial`.
/// - `patch_size_initial`: side length of every scale-0 anchor.
/// - `patch_size_min`: subdivision stops when a child's side would
///   fall below this (so subdivision only happens while
///   `parent_size / 2 >= patch_size_min`).
/// - `max_depth`: hard cap on the subdivision depth.  At depth `d` the
///   anchor side length is `patch_size_initial / 2^d`.
/// - `max_anchors`: hard cap on the total number of anchors emitted.
///   When the budget is exhausted, the depth loop terminates early.
/// - `variance_weight`, `curvature_weight`: the combined score is
///   `variance_weight · score_fn_result + curvature_weight · |κ|`.
///   Either may be zero.
/// - `score_threshold`: anchors whose combined score does not strictly
///   exceed this are NOT subdivided.
/// - `score_fn`: closure invoked once per depth with
///   `(frontier_positions, frontier_sizes)`, expected to return a
///   `Vec<f32>` of length `frontier_positions.len()` containing the
///   per-anchor variance score.
///
/// # Preconditions
///
/// - `image_h % patch_size_initial == 0` and `image_w % patch_size_initial == 0`.
/// - `patch_size_min <= patch_size_initial` and both are positive.
/// - `score_fn` returns a vector of the right length on every call.
///
/// (The PyO3 wrapper validates these at the binding boundary; this
/// pure-Rust entry point is a no-op-on-empty function for trivial
/// inputs and only `debug_assert!`s the contract.)
pub fn build_quadtree<F>(
    image_h: i64,
    image_w: i64,
    patch_size_initial: i64,
    patch_size_min: i64,
    max_depth: i64,
    max_anchors: i64,
    variance_weight: f32,
    curvature_weight: f32,
    score_threshold: f32,
    mut score_fn: F,
) -> QuadtreeAnchors
where
    F: FnMut(&[(i64, i64)], &[i64]) -> Vec<f32>,
{
    let max_anchors_usize = max_anchors.max(0) as usize;
    let mut positions: Vec<(i64, i64)> = Vec::with_capacity(max_anchors_usize);
    let mut sizes:     Vec<i64>        = Vec::with_capacity(max_anchors_usize);
    let mut scales:    Vec<i64>        = Vec::with_capacity(max_anchors_usize);
    let mut parent_indices: Vec<i64>   = Vec::with_capacity(max_anchors_usize);

    // ---- Initial tiling at depth 0 (no scoring needed) ----
    let mut r: i64 = 0;
    while r < image_h {
        let mut c: i64 = 0;
        while c < image_w {
            positions.push((r, c));
            sizes.push(patch_size_initial);
            scales.push(0);
            parent_indices.push(-1);
            c += patch_size_initial;
        }
        r += patch_size_initial;
    }

    let mut frontier: Vec<usize> = (0..positions.len()).collect();
    let mut current_depth: i64 = 0;

    // ---- Depth loop ----
    while !frontier.is_empty() && current_depth < max_depth {
        let n_front = frontier.len();
        let mut frontier_positions: Vec<(i64, i64)> = Vec::with_capacity(n_front);
        let mut frontier_sizes:     Vec<i64>        = Vec::with_capacity(n_front);
        for &fi in &frontier {
            frontier_positions.push(positions[fi]);
            frontier_sizes.push(sizes[fi]);
        }

        // ---- Variance via caller-supplied closure ----
        let variances: Vec<f32> = score_fn(&frontier_positions, &frontier_sizes);
        debug_assert_eq!(
            variances.len(), n_front,
            "score_fn must return one score per frontier anchor",
        );

        // ---- Forman κ on the 4-conn same-depth graph ----
        // All anchors at one depth share the same patch size (the
        // builder enforces this: base tiling at depth 0 is uniform;
        // every subdivision halves the parent size for ALL its 4
        // children, so the depth-d frontier is uniform-size).
        let frontier_size = frontier_sizes[0];
        let kappa: Vec<f32> = if curvature_weight > 0.0 {
            forman_kappa_4conn(&frontier_positions, frontier_size)
        } else {
            vec![0.0; n_front]
        };

        // ---- Combined score = w_v·var + w_κ·|κ| ----
        let mut scores: Vec<f32> = Vec::with_capacity(n_front);
        for i in 0..n_front {
            let s = variance_weight * variances[i]
                  + curvature_weight * kappa[i].abs();
            scores.push(s);
        }

        // ---- Subdivide ----
        let mut new_frontier: Vec<usize> = Vec::new();
        for (local_i, &anchor_i) in frontier.iter().enumerate() {
            if scores[local_i] <= score_threshold {
                continue;
            }
            let parent_size = sizes[anchor_i];
            let child_size = parent_size / 2;
            if child_size < patch_size_min {
                continue;
            }
            // Budget cap: leave at least the 4 children we're about
            // to write.
            if positions.len() + 4 > max_anchors_usize {
                break;
            }
            let (pr, pc) = positions[anchor_i];
            for dr in [0i64, child_size] {
                for dc in [0i64, child_size] {
                    let new_idx = positions.len();
                    positions.push((pr + dr, pc + dc));
                    sizes.push(child_size);
                    scales.push(current_depth + 1);
                    parent_indices.push(anchor_i as i64);
                    new_frontier.push(new_idx);
                }
            }
        }

        frontier = new_frontier;
        current_depth += 1;
    }

    QuadtreeAnchors { positions, sizes, scales, parent_indices }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forman_kappa_single_anchor_is_zero() {
        let pos = vec![(0i64, 0i64)];
        let k = forman_kappa_4conn(&pos, 4);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0], 0.0);
    }

    #[test]
    fn forman_kappa_pair_horizontal_neighbours() {
        // Two anchors side-by-side, both with degree 1.
        // edge κ = 2 - 1 - 1 = 0; vertex κ = mean over 1 edge = 0.
        let pos = vec![(0i64, 0i64), (0, 4)];
        let k = forman_kappa_4conn(&pos, 4);
        assert_eq!(k.len(), 2);
        assert_eq!(k[0], 0.0);
        assert_eq!(k[1], 0.0);
    }

    #[test]
    fn forman_kappa_l_shape_three_anchors() {
        //   (0,0): neighbours (0,4) and (4,0) → deg 2
        //   (0,4): neighbour (0,0) only → deg 1
        //   (4,0): neighbour (0,0) only → deg 1
        // edge_κ:
        //   ( (0,0)-(0,4) ) = 2 - 2 - 1 = -1
        //   ( (0,0)-(4,0) ) = 2 - 2 - 1 = -1
        // vertex κ: -1 / -1 / -1
        let pos = vec![(0i64, 0i64), (0, 4), (4, 0)];
        let k = forman_kappa_4conn(&pos, 4);
        assert_eq!(k.len(), 3);
        for &v in &k {
            assert!((v - (-1.0)).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn forman_kappa_two_by_two_block() {
        // 2×2 block; each vertex has degree 2; every edge κ = -2.
        let pos = vec![(0i64, 0i64), (0, 4), (4, 0), (4, 4)];
        let k = forman_kappa_4conn(&pos, 4);
        assert_eq!(k.len(), 4);
        for &v in &k {
            assert!((v - (-2.0)).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn forman_kappa_center_of_3x3_has_degree_4() {
        // 3×3 grid; center (4,4) has 4 neighbours, each with degree 3.
        // edge κ = 2 - 4 - 3 = -5; vertex κ = mean of 4 × -5 = -5.
        let mut pos = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                pos.push((r * 4, c * 4));
            }
        }
        let k = forman_kappa_4conn(&pos, 4);
        assert!((k[4] - (-5.0)).abs() < 1e-6, "center got {}", k[4]);
    }

    // ---- build_quadtree state-machine tests ----

    #[test]
    fn build_quadtree_empty_image_returns_empty() {
        let anchors = build_quadtree(
            0, 0, 4, 1, 4, 1024, 1.0, 0.0, 0.0,
            |_, _| Vec::new(),
        );
        assert!(anchors.is_empty());
    }

    #[test]
    fn build_quadtree_uniform_low_score_no_subdivision() {
        // 16×16 image, root cells 8×8 → 4 cells at depth 0.
        // score_fn returns 0 everywhere; threshold 0.5 → no subdivision.
        let anchors = build_quadtree(
            16, 16, 8, 1, 4, 64,
            1.0, 0.0, 0.5,
            |pos, _| vec![0.0; pos.len()],
        );
        assert_eq!(anchors.len(), 4, "expected 4 root cells, no children");
        assert!(anchors.scales.iter().all(|&s| s == 0));
        assert!(anchors.parent_indices.iter().all(|&p| p == -1));
    }

    #[test]
    fn build_quadtree_high_score_subdivides_once() {
        // 16×16, root 8×8 → 4 root cells.  score_fn returns 1.0 always.
        // Children would be 4×4; patch_size_min=2 lets them be created.
        // After 1 depth: 4 root + 4×4 = 20 anchors.
        let anchors = build_quadtree(
            16, 16, 8, 2, 1, 64,
            1.0, 0.0, 0.5,
            |pos, _| vec![1.0; pos.len()],
        );
        assert_eq!(anchors.len(), 4 + 16);
        let n_root = anchors.scales.iter().filter(|&&s| s == 0).count();
        let n_depth_1 = anchors.scales.iter().filter(|&&s| s == 1).count();
        assert_eq!(n_root, 4);
        assert_eq!(n_depth_1, 16);
    }

    #[test]
    fn build_quadtree_respects_budget_cap() {
        let anchors = build_quadtree(
            16, 16, 8, 1, 4, 6, // max_anchors = 6
            1.0, 0.0, 0.5,
            |pos, _| vec![1.0; pos.len()],
        );
        assert!(anchors.len() <= 6, "got {}", anchors.len());
    }

    #[test]
    fn build_quadtree_score_fn_called_once_per_depth() {
        let mut n_calls = 0;
        // 16×16, root 8×8 → 4 root.  At depth 1 the children are 4×4.
        // patch_size_min=4 means depth-1 children would be 4 → still
        // >= min, so they're created.  Depth-2 children would be 2 <
        // min, so no further subdivision; loop terminates after the
        // depth-1 frontier produces no new frontier.  Therefore
        // score_fn is called ONCE at depth 0 (frontier = 4 root cells).
        // Note: depth-1 frontier IS evaluated but no subdivision since
        // child_size 2 < patch_size_min 4.
        let _ = build_quadtree(
            16, 16, 8, 4, 4, 1024,
            1.0, 0.0, 0.5,
            |pos, _| {
                n_calls += 1;
                vec![1.0; pos.len()]
            },
        );
        // Depth 0 frontier ⇒ subdivides into depth-1 frontier of 16.
        // Depth 1 frontier (16 cells of size 4) IS evaluated by score_fn,
        // but each child would be size 2 < min 4, so no new frontier.
        // So score_fn called at depth 0 AND depth 1 = 2 calls.
        assert_eq!(n_calls, 2, "score_fn should be called once per depth");
    }
}
