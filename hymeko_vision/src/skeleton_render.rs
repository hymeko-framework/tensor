//! **Skeleton rasterisation** — draw a planar stick figure into a square grid, and mask parts of it.
//!
//! Consolidated 2026-08-24 out of the four `nagare` `pose_*` experiment modules, which held four
//! copies of the same inner loop, then moved into this crate by authorisation the same day. The copies differed only in grid size, limb list, sample count and
//! occlusion policy; all four of those are arguments here.
//!
//! # The sample count is a parameter, not a constant
//! The four studies used **60, 72, 64 and 60** samples per segment. Folding them to one value would
//! move four recorded studies, so it is passed in and the caller states which it needs. This is the
//! same discipline as `nagare`'s `planar_scene::PlanarRng`, whose two named constructors differ by
//! a single shift.
//!
//! # Not `nagare`'s `neocognitron::render_bar`
//! That renders a *centred oriented bar* by testing every pixel against a distance field. This
//! walks a segment and stamps a 3×3 brush. Different construction, different anti-aliasing (none in
//! either case, but for different reasons), and neither is expressible as the other's special case.

/// A square rasterisation grid.
///
/// # Invariants
/// `size >= 1`. Pixel `(row, col)` is at flat index `row * size + col`, row-major, y-down — the
/// image convention the pose studies' coordinates are already in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterGrid {
    size: usize,
}

impl RasterGrid {
    /// A `size × size` grid.
    ///
    /// # Panics
    /// If `size == 0`.
    pub fn new(size: usize) -> Self {
        assert!(size >= 1, "a raster grid needs at least one pixel");
        Self { size }
    }

    /// Side length.
    pub fn size(self) -> usize {
        self.size
    }

    /// Pixel count, `size²`.
    pub fn area(self) -> usize {
        self.size * self.size
    }

    /// A zeroed image of this grid's shape.
    pub fn blank(self) -> Vec<f32> {
        vec![0.0f32; self.area()]
    }

    /// Whether integer pixel coordinates are inside.
    fn contains(self, col: i32, row: i32) -> bool {
        col >= 0 && row >= 0 && (col as usize) < self.size && (row as usize) < self.size
    }
}

/// Stamp a 3×3 brush along the segment `a → b`, at `samples + 1` evenly spaced points.
///
/// Pixels are **set to 1.0**, not accumulated, so overlapping strokes and repeated calls are
/// idempotent — which is what makes the drawing order irrelevant.
///
/// # Preconditions
/// `img.len() == grid.area()`; `samples >= 1`.
///
/// # Postconditions
/// Every written pixel is `1.0`; pixels outside the grid are silently skipped, so a segment leaving
/// the frame is clipped rather than wrapping or panicking.
///
/// # Panics
/// If `img` is not the grid's size, or `samples == 0`.
pub fn draw_segment(img: &mut [f32], grid: RasterGrid, a: [f32; 2], b: [f32; 2], samples: usize) {
    assert_eq!(img.len(), grid.area(), "image must match the grid");
    assert!(samples >= 1, "a segment needs at least one step");
    let n = grid.size();
    for s in 0..=samples {
        let t = s as f32 / samples as f32;
        let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
        let (ci, ri) = (x.round() as i32, y.round() as i32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cc, rr) = (ci + dx, ri + dy);
                if grid.contains(cc, rr) {
                    img[rr as usize * n + cc as usize] = 1.0;
                }
            }
        }
    }
}

/// Rasterise a whole skeleton: one segment per limb.
///
/// `skip_limb` drops one limb's evidence entirely — the *structural* occlusion, distinct from
/// masking a region of the finished image ([`occlude_box`]). A study needs one or the other, and
/// which one changes what the model can still infer.
///
/// # Preconditions
/// Every limb's endpoints index `joints`; `samples >= 1`.
///
/// # Postconditions
/// `out.len() == grid.area()`; every value is `0.0` or `1.0`. Independent of limb order.
///
/// # Panics
/// If a limb endpoint is out of range, or `samples == 0`.
pub fn rasterise_skeleton(
    joints: &[[f32; 2]],
    limbs: &[[usize; 2]],
    grid: RasterGrid,
    samples: usize,
    skip_limb: Option<usize>,
) -> Vec<f32> {
    let mut img = grid.blank();
    for (li, limb) in limbs.iter().enumerate() {
        if Some(li) == skip_limb {
            continue;
        }
        assert!(
            limb[0] < joints.len() && limb[1] < joints.len(),
            "limb {li} endpoint out of range"
        );
        draw_segment(&mut img, grid, joints[limb[0]], joints[limb[1]], samples);
    }
    img
}

/// An inclusive pixel rectangle, already clamped to a grid.
///
/// # Invariants
/// `x0, x1, y0, y1` are valid indices. A degenerate range (`x0 > x1`) is legal and masks nothing —
/// it is what a box entirely off-grid collapses to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelBox {
    /// Leftmost column, inclusive.
    pub x0: usize,
    /// Rightmost column, inclusive.
    pub x1: usize,
    /// Topmost row, inclusive.
    pub y0: usize,
    /// Bottommost row, inclusive.
    pub y1: usize,
}

impl PixelBox {
    /// A square of half-width `radius` around a point, clipped to the grid.
    ///
    /// The centre is **rounded**, then the radius is applied in integer pixels — matching the
    /// recorded per-pixel bounds check exactly, which is why this is not expressed as a float
    /// margin.
    ///
    /// # A square entirely off-grid masks nothing
    /// Clamping both ends of the span to the same edge would turn a box at `y = -8, radius = 6`
    /// into a one-pixel-tall box on row 0 — masking a row the retired per-pixel check never
    /// touched. The span is therefore returned **empty** in that case. The equivalence test caught
    /// exactly this; see [`PixelBox::bounding`] for the opposite behaviour, which is also recorded.
    ///
    /// # Preconditions
    /// `radius >= 0`.
    ///
    /// # Postconditions
    /// Empty (`x0 > x1` or `y0 > y1`) iff the square does not intersect the grid.
    pub fn centred(centre: [f32; 2], radius: i32, grid: RasterGrid) -> Self {
        debug_assert!(radius >= 0, "a radius is not negative");
        let last = grid.size() as i32 - 1;
        let (cx, cy) = (centre[0].round() as i32, centre[1].round() as i32);
        let clip = |lo: i32, hi: i32| -> (usize, usize) {
            if hi < 0 || lo > last {
                (1, 0) // an empty inclusive range
            } else {
                (lo.max(0) as usize, hi.min(last) as usize)
            }
        };
        let (x0, x1) = clip(cx - radius, cx + radius);
        let (y0, y1) = clip(cy - radius, cy + radius);
        Self { x0, x1, y0, y1 }
    }

    /// The axis-aligned bounding box of `points`, grown by `margin` and clamped.
    ///
    /// The clamp is `max(0.0)` / `min(size - 1)` followed by an `as usize` **truncation**, which is
    /// the recorded arithmetic: a lower bound of `2.9` becomes column 2, not 3.
    ///
    /// # It does NOT clip an off-grid box to nothing
    /// Unlike [`PixelBox::centred`], a box entirely off the negative side still masks the corner
    /// pixel: Rust's `f32 as usize` saturates a negative value to `0`, so both ends collapse to
    /// column 0 rather than to an empty span. That is the retired body's behaviour, bit-for-bit,
    /// and it is preserved rather than "fixed" — the two recorded studies that use it never place a
    /// box off-grid, and changing it would move their numbers for no measured reason.
    ///
    /// # Preconditions
    /// `points` is non-empty.
    ///
    /// # Panics
    /// If `points` is empty — an empty bounding box has no defined position, and silently returning
    /// the whole grid or nothing would both be wrong.
    pub fn bounding(points: &[[f32; 2]], margin: f32, grid: RasterGrid) -> Self {
        assert!(!points.is_empty(), "an empty point set has no bounding box");
        let last = (grid.size() - 1) as f32;
        let (mut lo_x, mut hi_x) = (f32::MAX, f32::MIN);
        let (mut lo_y, mut hi_y) = (f32::MAX, f32::MIN);
        for p in points {
            lo_x = lo_x.min(p[0]);
            hi_x = hi_x.max(p[0]);
            lo_y = lo_y.min(p[1]);
            hi_y = hi_y.max(p[1]);
        }
        Self {
            x0: (lo_x - margin).max(0.0) as usize,
            x1: (hi_x + margin).min(last) as usize,
            y0: (lo_y - margin).max(0.0) as usize,
            y1: (hi_y + margin).min(last) as usize,
        }
    }
}

/// Zero every pixel inside `region`.
///
/// # Preconditions
/// `img.len() == grid.area()`; `region` was built for the same grid.
///
/// # Postconditions
/// Pixels outside the region are untouched. Idempotent.
///
/// # Panics
/// If `img` is not the grid's size.
pub fn occlude_box(img: &mut [f32], grid: RasterGrid, region: PixelBox) {
    assert_eq!(img.len(), grid.area(), "image must match the grid");
    let n = grid.size();
    for row in region.y0..=region.y1 {
        for col in region.x0..=region.x1 {
            img[row * n + col] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: usize = 28;

    /// `examples`-era body, transcribed from `experiments::pose_backbone::render` before the move:
    /// two limbs at 60 samples with a centred square mask.
    fn retired_backbone(joints: &[[f32; 2]; 3], occ: Option<[f32; 2]>) -> Vec<f32> {
        let mut img = vec![0.0f32; G * G];
        let limbs = [(0usize, 1usize), (1, 2)];
        for &(a, b) in &limbs {
            let (p, q) = (joints[a], joints[b]);
            for s in 0..=60 {
                let t = s as f32 / 60.0;
                let (x, y) = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t);
                let (ci, ri) = (x.round() as i32, y.round() as i32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (cc, rr) = (ci + dx, ri + dy);
                        if cc >= 0 && rr >= 0 && (cc as usize) < G && (rr as usize) < G {
                            img[rr as usize * G + cc as usize] = 1.0;
                        }
                    }
                }
            }
        }
        if let Some(c) = occ {
            let rad = 6i32;
            let (cx, cy) = (c[0].round() as i32, c[1].round() as i32);
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (cc, rr) = (cx + dx, cy + dy);
                    if cc >= 0 && rr >= 0 && (cc as usize) < G && (rr as usize) < G {
                        img[rr as usize * G + cc as usize] = 0.0;
                    }
                }
            }
        }
        img
    }

    /// `experiments::pose_symmetry::render`'s occlusion, transcribed: a bounding box with margin 3.
    fn retired_bounding_mask(img: &mut [f32], size: usize, el: [f32; 2], ha: [f32; 2]) {
        let (x0, x1) = (
            (el[0].min(ha[0]) - 3.0).max(0.0) as usize,
            (el[0].max(ha[0]) + 3.0).min((size - 1) as f32) as usize,
        );
        let (y0, y1) = (
            (el[1].min(ha[1]) - 3.0).max(0.0) as usize,
            (el[1].max(ha[1]) + 3.0).min((size - 1) as f32) as usize,
        );
        for r in y0..=y1 {
            for c in x0..=x1 {
                img[r * size + c] = 0.0;
            }
        }
    }

    fn poses() -> Vec<[[f32; 2]; 3]> {
        vec![
            [[14.0, 23.0], [9.5, 15.2], [16.1, 9.4]],
            [[14.0, 23.0], [20.7, 18.3], [26.9, 22.1]], // runs off the right edge
            [[14.0, 23.0], [-3.0, 4.0], [1.0, -8.0]],   // runs off two edges
            [[14.0, 23.0], [14.0, 23.0], [14.0, 23.0]], // degenerate: zero-length limbs
        ]
    }

    #[test]
    fn rasterise_skeleton_is_bit_identical_to_the_retired_backbone_renderer() {
        let grid = RasterGrid::new(G);
        for joints in poses() {
            let ours = rasterise_skeleton(&joints, &[[0, 1], [1, 2]], grid, 60, None);
            assert_eq!(ours, retired_backbone(&joints, None), "clean {joints:?}");
        }
    }

    #[test]
    fn a_centred_box_is_bit_identical_to_the_retired_per_pixel_mask() {
        let grid = RasterGrid::new(G);
        for joints in poses() {
            for centre in [joints[1], joints[2], [0.0, 0.0], [27.0, 27.0], [-5.0, 40.0]] {
                let mut ours = rasterise_skeleton(&joints, &[[0, 1], [1, 2]], grid, 60, None);
                occlude_box(&mut ours, grid, PixelBox::centred(centre, 6, grid));
                assert_eq!(
                    ours,
                    retired_backbone(&joints, Some(centre)),
                    "occluded at {centre:?}"
                );
            }
        }
    }

    #[test]
    fn a_bounding_box_is_bit_identical_to_the_retired_two_point_mask() {
        let size = 32usize;
        let grid = RasterGrid::new(size);
        for (el, ha) in [
            ([9.0f32, 12.0], [15.5, 20.25]),
            ([0.4, 0.6], [1.2, 1.9]),
            ([28.0, 30.0], [31.9, 31.1]),
            ([20.0, 5.0], [3.0, 27.0]), // reversed order
        ] {
            let base: Vec<f32> = (0..size * size).map(|i| (i % 3) as f32 * 0.5).collect();
            let mut ours = base.clone();
            occlude_box(&mut ours, grid, PixelBox::bounding(&[el, ha], 3.0, grid));
            let mut theirs = base;
            retired_bounding_mask(&mut theirs, size, el, ha);
            assert_eq!(ours, theirs, "el {el:?} ha {ha:?}");
        }
    }

    /// The two box constructors disagree on an entirely off-grid region, and both behaviours are
    /// the recorded ones. Pinned so neither is "tidied" into the other.
    #[test]
    fn the_two_box_constructors_treat_an_off_grid_region_differently_by_record() {
        let grid = RasterGrid::new(8);
        let centred = PixelBox::centred([1.0, -8.0], 6, grid);
        assert!(
            centred.y0 > centred.y1,
            "a square fully above the grid is empty"
        );
        let bounding = PixelBox::bounding(&[[-9.0, -9.0], [-8.0, -8.0]], 3.0, grid);
        assert_eq!(
            (bounding.x0, bounding.x1, bounding.y0, bounding.y1),
            (0, 0, 0, 0),
            "the bounding form saturates to the corner pixel, as the retired body did"
        );
        let mut img = vec![1.0f32; grid.area()];
        occlude_box(&mut img, grid, centred);
        assert!(
            img.iter().all(|&v| v == 1.0),
            "the empty span masks nothing"
        );
    }

    #[test]
    fn skipping_a_limb_removes_only_that_limbs_strokes() {
        let grid = RasterGrid::new(G);
        let joints = poses()[0];
        let limbs = [[0usize, 1], [1, 2]];
        let full = rasterise_skeleton(&joints, &limbs, grid, 60, None);
        let without = rasterise_skeleton(&joints, &limbs, grid, 60, Some(1));
        let only_first = rasterise_skeleton(&joints, &limbs[..1], grid, 60, None);
        assert_eq!(
            without, only_first,
            "skipping limb 1 leaves limb 0's strokes"
        );
        assert!(
            full.iter().sum::<f32>() > without.iter().sum::<f32>(),
            "and it really removes ink"
        );
    }

    #[test]
    fn drawing_is_idempotent_and_order_independent() {
        let grid = RasterGrid::new(G);
        let joints = poses()[0];
        let a = rasterise_skeleton(&joints, &[[0, 1], [1, 2]], grid, 60, None);
        let b = rasterise_skeleton(&joints, &[[1, 2], [0, 1]], grid, 60, None);
        assert_eq!(a, b, "limb order must not matter");
        let mut twice = a.clone();
        draw_segment(&mut twice, grid, joints[0], joints[1], 60);
        assert_eq!(a, twice, "redrawing must change nothing");
        assert!(a.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn the_sample_count_is_load_bearing() {
        let grid = RasterGrid::new(G);
        let joints = [[2.0f32, 2.0], [25.0, 25.0], [25.0, 2.0]];
        let dense = rasterise_skeleton(&joints, &[[0, 1]], grid, 60, None);
        let sparse = rasterise_skeleton(&joints, &[[0, 1]], grid, 3, None);
        assert_ne!(
            dense, sparse,
            "a long segment at 3 samples must leave gaps a 60-sample one does not"
        );
        assert!(dense.iter().sum::<f32>() > sparse.iter().sum::<f32>());
    }

    #[test]
    fn an_off_grid_segment_is_clipped_not_wrapped() {
        let grid = RasterGrid::new(8);
        let img = rasterise_skeleton(&[[-40.0, -40.0], [-30.0, -30.0]], &[[0, 1]], grid, 20, None);
        assert!(img.iter().all(|&v| v == 0.0), "nothing should be drawn");
    }

    #[test]
    #[should_panic(expected = "empty point set")]
    fn an_empty_bounding_box_is_rejected() {
        let _ = PixelBox::bounding(&[], 1.0, RasterGrid::new(8));
    }

    #[test]
    #[should_panic(expected = "at least one pixel")]
    fn a_zero_sized_grid_is_rejected() {
        let _ = RasterGrid::new(0);
    }
}
