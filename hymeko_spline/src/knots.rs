//! The knots a Catmull–Rom spline interpolates over `[-1, 1]` (plan
//! `docs/plans/2026-10-16-cr-orthtree-knots/`): a uniform base grid of `grid` knots, each base segment
//! refined to its own dyadic depth -- `2^d` equal leaves, the complete orthtree of that cell -- so the
//! spline has resolution where its inputs and its curvature are, and stays coarse elsewhere.
//!
//! The uniform grid is the knots at depth zero everywhere, and the kernel evaluates it by its own
//! arithmetic, bit for bit. Refined knots take the non-uniform Catmull–Rom, the cubic Hermite whose
//! tangents are the neighbouring chords' slopes -- at a leaf of equally spaced neighbours the
//! uniform Catmull–Rom in exact arithmetic. The ends replicate their control points over a mirrored
//! spacing, as the uniform spline always has.
//!
//! A point finds its leaf without a search: its base segment by `floor`, its leaf within it by
//! `floor(s 2^d)`, a table read.

use std::sync::Arc;

/// The knots of a Catmull–Rom spline over `[-1, 1]`; cloning shares one table. Two knot sets are
/// equal, and hash alike, when their declarations are: the base grid and the depths, of which the
/// table is a function.
#[derive(Debug, Clone)]
pub struct Knots(Arc<KnotTable>);

impl PartialEq for Knots {
    fn eq(&self, other: &Self) -> bool {
        self.0.grid == other.0.grid && self.0.depths == other.0.depths
    }
}

impl Eq for Knots {}

impl std::hash::Hash for Knots {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.grid.hash(state);
        self.0.depths.hash(state);
    }
}

#[derive(Debug)]
struct KnotTable {
    /// Base knots, uniform over `[-1, 1]`.
    grid: usize,
    /// Per base segment its depth.
    depths: Vec<u8>,
    /// Per base segment its first leaf and its scale `2^d`, side by side: what an element reads to
    /// find its leaf, in one entry (callgrind 2026-10-16: the lookup was 27 % of a refined forward's
    /// instructions with the two in separate tables and the scale shifted per element).
    cells: Vec<(u32, f32)>,
    /// Every knot, sorted.
    positions: Vec<f32>,
    /// Per leaf -- the segment between two consecutive knots, in order -- the four control points
    /// it reads, the ends replicated.
    controls: Vec<[usize; 4]>,
    /// Per leaf `h / (x_{k+1} − x_{k−1})` and `h / (x_{k+2} − x_k)`, `h` its width: the tangents'
    /// shares.
    ratio: Vec<[f32; 2]>,
}

/// The deepest refinement a base segment takes: `2^8` leaves.
pub const MAX_DEPTH: u8 = 8;

impl Knots {
    /// `grid` knots spaced uniformly over `[-1, 1]`: every base segment at depth zero.
    ///
    /// # Panics
    /// When `grid < 3`.
    #[must_use]
    pub fn uniform(grid: usize) -> Self {
        assert!(
            grid >= 3,
            "Catmull--Rom needs at least 3 control points, got {grid}"
        );
        Self::refined(grid, &vec![0; grid - 1])
    }

    /// `grid` base knots over `[-1, 1]`, base segment `b` refined to `depths[b]`: `2^depths[b]` equal
    /// leaves.
    ///
    /// # Panics
    /// When `grid < 3`, `depths.len() != grid − 1` or a depth exceeds [`MAX_DEPTH`].
    #[must_use]
    pub fn refined(grid: usize, depths: &[u8]) -> Self {
        assert!(
            grid >= 3,
            "Catmull--Rom needs at least 3 control points, got {grid}"
        );
        assert_eq!(depths.len(), grid - 1, "one depth per base segment");
        assert!(
            depths.iter().all(|&d| d <= MAX_DEPTH),
            "a depth up to {MAX_DEPTH}"
        );
        // A base knot is `-1 + 2 b / (g - 1)` in `f32`, the expression a sampler of the uniform
        // knots evaluates, so a shape sampled at these knots is the uniform grid's, bit for bit.
        let base = |b: usize| -1.0f32 + 2.0f32 * b as f32 / (grid - 1) as f32;
        let mut owners: Vec<(usize, u32)> = Vec::new();
        let mut first_leaf = Vec::with_capacity(depths.len());
        for (b, &d) in depths.iter().enumerate() {
            first_leaf.push(owners.len() as u32);
            owners.extend((0..1u32 << d).map(|j| (b, j)));
        }
        let mut positions: Vec<f32> = owners
            .iter()
            .map(|&(b, j)| match j {
                0 => base(b),
                _ => {
                    let (lo, hi) = (base(b), base(b + 1));
                    lo + (hi - lo) * j as f32 / (1u32 << depths[b]) as f32
                }
            })
            .collect();
        positions.push(1.0);
        let n = positions.len();
        // Widths in base-segment units, exact: 2^-d.
        let width = |leaf: usize| -> f64 { 1.0 / f64::from(1u32 << depths[owners[leaf].0]) };
        let mut controls = Vec::with_capacity(owners.len());
        let mut ratio = Vec::with_capacity(owners.len());
        for k in 0..owners.len() {
            let h = width(k);
            // A missing neighbour mirrors the leaf's own width: the end replication.
            let before = if k == 0 { h } else { width(k - 1) };
            let after = if k + 1 == owners.len() {
                h
            } else {
                width(k + 1)
            };
            controls.push([
                k.saturating_sub(1),
                k,
                (k + 1).min(n - 1),
                (k + 2).min(n - 1),
            ]);
            ratio.push([(h / (before + h)) as f32, (h / (h + after)) as f32]);
        }
        let cells = first_leaf
            .iter()
            .zip(depths)
            .map(|(&first, &d)| (first, (1u32 << d) as f32))
            .collect();
        Self(Arc::new(KnotTable {
            grid,
            depths: depths.to_vec(),
            cells,
            positions,
            controls,
            ratio,
        }))
    }

    /// Base knots.
    #[must_use]
    pub fn grid(&self) -> usize {
        self.0.grid
    }

    /// Every base segment's depth.
    #[must_use]
    pub fn depths(&self) -> &[u8] {
        &self.0.depths
    }

    /// Every knot, sorted: one control point each.
    #[must_use]
    pub fn positions(&self) -> &[f32] {
        &self.0.positions
    }

    /// Knots: the control points per channel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.positions.len()
    }

    /// Never: a spline has at least three knots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.positions.is_empty()
    }

    /// Whether every base segment is at depth zero: the uniform grid.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.0.depths.iter().all(|&d| d == 0)
    }

    /// Per leaf its four control points.
    pub(crate) fn controls(&self) -> &[[usize; 4]] {
        &self.0.controls
    }

    /// Leaf `leaf`'s tangent shares.
    #[inline(always)]
    pub(crate) fn leaf_ratio(&self, leaf: usize) -> [f32; 2] {
        self.0.ratio[leaf]
    }

    /// The leaf a clamped input reads, its coordinate `t` and the leaf's scale `2^d` onto it: base
    /// segment `b = min(⌊u⌋, g − 2)` of `u = (x_c + 1) · span` (an input that is not a number takes
    /// segment 0, as the integer cast did), `s = u − b`, the leaf `c = min(⌊s 2^d⌋, 2^d − 1)` within
    /// it, `t = s 2^d − c`, every step exact in `f32` for a power-of-two scale. At depth zero this is
    /// the uniform kernel's segment and coordinate, bit for bit; the uniform knots take that kernel's
    /// own arithmetic and never call this.
    #[inline(always)]
    pub(crate) fn locate(&self, xc: f32, span: f32) -> (usize, f32, f32) {
        let last = (self.0.grid - 2) as f32;
        let u = (xc + 1.0) * span;
        let floor = u.floor();
        let b = if floor >= 0.0 { floor.min(last) } else { 0.0 };
        let s = u - b;
        let (first, cells) = self.0.cells[b as usize];
        let scaled = s * cells;
        let cell = scaled.floor().clamp(0.0, cells - 1.0);
        (first as usize + cell as usize, scaled - cell, cells)
    }

    /// The refinement whose leaves follow `samples`' mass: base segment `b` takes depth
    /// `min(max_depth, max(0, round(log2(share_b · budget))))`, `share_b` the fraction of the
    /// clamped samples in it and `budget` the leaves a base segment holding every sample would take.
    /// Samples that are not numbers are not counted.
    ///
    /// # Panics
    /// When `grid < 3`, `budget` is not positive or `max_depth` exceeds [`MAX_DEPTH`].
    #[must_use]
    pub fn refined_by_density(grid: usize, samples: &[f32], budget: f32, max_depth: u8) -> Self {
        assert!(budget > 0.0, "a positive leaf budget");
        assert!(max_depth <= MAX_DEPTH, "a depth up to {MAX_DEPTH}");
        let uniform = Self::uniform(grid);
        let span = 0.5 * (grid - 1) as f32;
        let mut counts = vec![0usize; grid - 1];
        let mut total = 0usize;
        for &x in samples.iter().filter(|x| !x.is_nan()) {
            let (leaf, _, _) = uniform.locate(x.clamp(-1.0, 1.0), span);
            counts[leaf] += 1;
            total += 1;
        }
        let depths: Vec<u8> = counts
            .iter()
            .map(|&c| {
                let mass = c as f32 / total.max(1) as f32 * budget;
                if mass <= 1.0 {
                    0
                } else {
                    (mass.log2().round() as u8).min(max_depth)
                }
            })
            .collect();
        Self::refined(grid, &depths)
    }

    /// The control points over `to` of the spline `coef` defines over these knots, `channels`
    /// splines of `self.len()` points each: each new control point is the spline's value at its
    /// knot, so the new spline interpolates the old at every knot of `to` -- at the old knots the
    /// old control points, at a dyadic midpoint of equal neighbours the 4-point rule
    /// `(−1, 9, 9, −1) / 16`.
    ///
    /// # Panics
    /// When `coef.len() != channels · self.len()`.
    #[must_use]
    pub fn transfer(&self, coef: &[f32], channels: usize, to: &Knots) -> Vec<f32> {
        assert_eq!(coef.len(), channels * self.len(), "one spline a channel");
        coef.chunks_exact(self.len())
            .flat_map(|channel| {
                crate::catmull_rom::catmull_rom_forward(channel, to.positions(), to.len(), 1, self)
                    .0
            })
            .collect()
    }
}
