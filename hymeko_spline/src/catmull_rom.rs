//! Native Catmull--Rom and Chebyshev-CR activation kernels.
//!
//! These operators mirror `hymeko_neuro.core.splines`: per-channel uniform CR on
//! `[-1, 1]`, plus Chebyshev-generated CR control points and direct Chebyshev
//! deploy evaluation. All gradients are explicit closed-form buffers.
//!
//! # Parallelism
//!
//! The forward and backward loop over `n` rows, and rows are independent, so both
//! are row-parallel above [`PAR_MIN_ELEMS`]. The row axis is the *only* useful one
//! here: `nagare`'s `ops::kan`'s outer loop is over `d_out`, which is `1` for
//! regression, so parallelising that instead would buy nothing at the shape that
//! matters. `kan.rs` is therefore left sequential and these kernels own the
//! parallelism — one region per call, never nested.
//!
//! # The invariant that makes this safe
//!
//! **Rayon changes only *who* does the work, never *what* is computed.** Both
//! paths block the rows identically and run the same kernel; the parallel branch
//! differs solely in using `par_chunks_mut` instead of `chunks_mut`. Concretely:
//!
//! * The forward, and the backward's `grad_x`, are pure per-row maps with disjoint
//!   writes — byte-identical on any thread count, and to the pre-parallel code.
//! * The backward's `grad_coef` is a reduction across rows, and float addition is
//!   not associative, so the sum is *blocked*. The block layout is a function of
//!   `n` and `channels` **only** — never the thread count — so the result is
//!   byte-identical on 1, 8 or 16 threads. It is *not* equal to the old flat
//!   sequential sum, which is the one deliberate numerical change here.
//!
//! Because of that invariant, [`use_parallel`] is free to consult the thread count
//! as well as the size: on a single-threaded host a rayon region is pure overhead
//! (~13 % measured) and buys nothing, so it is skipped without any effect on the
//! numbers. Had the sequential path kept the flat sum, this optimisation would
//! have made results differ between a 1-core and an 8-core machine.
//!
//! Below [`PAR_MIN_ELEMS`] no rayon region is entered at all. That threshold is not
//! decoration: the 2026-07-16 record has a dense layer going from 21.4 ms at one
//! thread to 133.3 ms at 18 — a 6x regression from splitting 120 rows across
//! workers. Small shapes must stay sequential.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::knots::Knots;

/// Row-parallelism is used only when `n * channels` reaches this many elements.
///
/// **Measured crossover, not a guess.** The `kan_cr_threshold` bench group run at
/// 8 threads with the parallel path forced on and forced off (epoch, µs):
///
/// | elements | sequential | parallel |
/// |---:|---:|---:|
/// | 2048 | **35.6** | 42.7 |
/// | 4096 | 69.6 | **54.9** |
/// | 16384 | 277.9 | **134.3** |
///
/// The crossing lies between 2048 and 4096, so 4096 is the first size at which
/// rayon pays for itself. iris (`120 * 4 = 480`) stays sequential; california
/// (`1600 * 8 = 12800`) does not. See `docs/plans/2026-08-17-cr-row-parallel/`.
pub const PAR_MIN_ELEMS: usize = 4096;

/// Whether a call of this many elements should use the rayon path.
///
/// Thread count may be consulted here **only** because the numerics are identical
/// on both paths: the forward is a pure map, and the backward always accumulates
/// `grad_coef` in the same fixed block order whether or not rayon is involved. If
/// the sequential path ever goes back to a flat unblocked sum, this function must
/// stop looking at the thread count, or results would differ between a 1-core and
/// an 8-core host.
#[cfg(feature = "parallel")]
#[inline]
fn use_parallel(elems: usize) -> bool {
    elems >= PAR_MIN_ELEMS && rayon::current_num_threads() > 1
}

/// Fixed number of row blocks the parallel paths split into.
///
/// Deliberately a constant and **not** `rayon::current_num_threads()`: the
/// `grad_coef` summation order — and therefore the exact float result — must not
/// depend on how many threads happen to be available, or a training run would be
/// irreproducible across machines. Same discipline as `ops::linear`'s
/// `GRAD_W_CHUNKS`.
///
/// It also has to be big enough to load-balance and small enough to amortise
/// rayon's per-item cost. The first implementation chunked the forward by a single
/// row — 8 elements per work item at california's width — and measured *slower*
/// than sequential on one thread; blocking many rows together fixed it.
const ROW_BLOCKS: usize = 64;

/// Floor on rows per block, so small inputs do not get shredded into blocks.
///
/// Without this, `ROW_BLOCKS` applies literally at every size: iris (`n = 120`)
/// produced 60 blocks for a problem with 24 gradient outputs, and the partials
/// buffer plus its fold cost **+21 %** on the backward. The floor collapses that
/// to a handful of blocks. It is a function of `n` alone — never the thread
/// count — so the summation order stays machine-independent.
const MIN_ROWS_PER_BLOCK: usize = 32;

/// Rows per parallel block, and the element stride that implies.
///
/// # Postconditions
/// The stride is a whole number of rows, so a block never straddles a row and the
/// per-channel indexing inside the kernels stays valid. Depends only on `n` and
/// `channels`, which is what makes `grad_coef` reproducible across machines.
#[inline]
fn block_stride(n: usize, channels: usize) -> usize {
    n.div_ceil(ROW_BLOCKS).max(MIN_ROWS_PER_BLOCK) * channels
}

/// Cache from Catmull--Rom forward needed by the closed-form backward.
///
/// A caller that runs the forward repeatedly may keep one cache and pass it to
/// [`catmull_rom_forward_into`], which refills it in place: its buffers keep their capacity, so a
/// training step stops allocating (and page-faulting) a fresh tape every call. `Default` is the
/// empty cache such a caller starts from.
///
/// It holds **only what the backward cannot recompute**: five bytes per element. The segment
/// index and the local coordinate are functions of the clamped input ([`segment`]), so the backward
/// recomputes them with the forward's own expression instead of reading them from a tape; the
/// result is bit-identical. Until 2026-09-25 both were stored as well, ten bytes per element.
#[derive(Debug, Clone, Default)]
pub struct CatmullRomCache {
    /// Clamped input values, flat `(n, channels)`. The segment of each element is
    /// [`segment`] of this value.
    pub x_clamped: Vec<f32>,
    /// Whether each raw input lay **inside the closed interval** `[-1, 1]`, flat `(n, channels)`.
    ///
    /// The forward clamps, so `x_clamped` cannot answer this: it is always inside by
    /// construction. Recording the fact here is what lets the backward apply the clamp's own
    /// Jacobian — the information the pre-2026-08-25 implementation destroyed. `bool` rather than
    /// the raw `x`: one byte instead of four, and it is exactly the predicate the guard needs.
    ///
    /// The interval is **closed**, so `x == ±1` is interior and keeps the interior derivative the
    /// historical implementation gave it.
    pub interior: Vec<bool>,
    /// Number of rows in the flat input.
    pub n: usize,
    /// Channel count.
    pub channels: usize,
    /// Number of CR control points per channel: the knots' count.
    pub grid: usize,
    /// The knots the forward evaluated over; the backward locates each element's leaf in them again.
    pub knots: Option<Knots>,
}

impl CatmullRomCache {
    /// The `f32` scalars this tape retains: the clamped inputs. The `interior` mask is `bool`, not
    /// an `f32`, so it is not counted.
    #[must_use]
    pub fn retained_floats(&self) -> usize {
        self.x_clamped.len()
    }
}

/// The CR segment of a clamped input over uniform knots: `(i, t)` with `u = (xc + 1) · span`,
/// `i = min(⌊u⌋, grid − 2)` and `t = u − i`, the index held as the float it is.
///
/// **The one expression** the uniform forward evaluates and its backward recomputes, which is what
/// makes the recomputed segment bit-identical to the forward's. Written in floats so a pass over a
/// row vectorises (Nagare plan 2026-10-08-orthtree-hypergraph-convolution, Amendment 1): `⌊u⌋` and
/// `grid − 2` are small whole numbers, so their minimum and `u − i` are the integer form's bit for
/// bit, and an input that is not a number takes segment `0`, as the saturating integer cast did.
/// Refined knots locate their leaf by [`Knots::locate`], the same steps with a dyadic one after.
///
/// # Preconditions
/// `xc ∈ [-1, 1]` or NaN, `span == 0.5 · (grid − 1)`, `last == (grid − 2) as f32`, `grid >= 3`.
#[inline(always)]
fn segment(xc: f32, span: f32, last: f32) -> (f32, f32) {
    let u = (xc + 1.0) * span;
    let floor = u.floor();
    let i = if floor >= 0.0 { floor.min(last) } else { 0.0 };
    (i, u - i)
}

/// Elements per run of the two-pass kernels: what the first pass writes for a run stays in L1.
const RUN: usize = 256;

/// The first pass's values for a run of up to [`RUN`] elements: each element's segment index, its four
/// weights and, for the pullback, their four derivatives. The first pass computes them for the
/// whole run at once and vectorises; the second reads them beside the per-channel tables, which
/// it indexes by channel and segment and which do not vectorise.
struct ElementPass {
    segment: [u32; RUN],
    weights: [[f32; RUN]; 4],
    dweights: [[f32; RUN]; 4],
    /// Each element's leaf scale `2^d` onto its `t` (refined knots; `1` over uniform ones).
    scale: [f32; RUN],
}

impl ElementPass {
    fn new() -> Self {
        Self {
            segment: [0; RUN],
            weights: [[0.0; RUN]; 4],
            dweights: [[0.0; RUN]; 4],
            scale: [1.0; RUN],
        }
    }
}

/// Each segment of each channel with what an element in it reads (Nagare plan
/// 2026-10-08-orthtree-hypergraph-convolution, Amendment 1): the four control values, gathered once
/// per call instead of once per element, and the four control indices the coefficient gradient
/// accumulates into.
///
/// # Invariants
/// `quads[ch · (grid − 1) + i][k] == coef[ch · grid + controls[i][k]]` and
/// `controls[i] == cr_control_indices(i, grid)` for every channel `ch` and segment `i ≤ grid − 2`:
/// the values and indices the per-element expressions read, so every result is unchanged bit for
/// bit.
struct Segments {
    quads: Vec<[f32; 4]>,
    controls: Vec<[usize; 4]>,
}

impl Segments {
    /// The table of `coef`, `channels × knots.len()`, by leaf.
    ///
    /// # Preconditions
    /// `coef.len() == channels · knots.len()`.
    fn of(coef: &[f32], channels: usize, knots: &Knots) -> Self {
        let controls: Vec<[usize; 4]> = knots.controls().to_vec();
        let quads = coef
            .chunks_exact(knots.len())
            .take(channels)
            .flat_map(|channel| controls.iter().map(move |c| c.map(|j| channel[j])))
            .collect();
        Self { quads, controls }
    }

    /// Channel `ch`'s four control values in segment `i`.
    fn quad(&self, ch: usize, i: usize) -> [f32; 4] {
        self.quads[ch * self.controls.len() + i]
    }
}

/// Result of a Catmull--Rom backward pass.
#[derive(Debug, Clone)]
pub struct CatmullRomBackward {
    /// Gradient w.r.t. input `x`, flat `(n, channels)`.
    pub grad_x: Vec<f32>,
    /// Gradient w.r.t. CR control points, flat `(channels, grid)`.
    pub grad_coef: Vec<f32>,
}

/// Result of Chebyshev-CR train-mode backward.
#[derive(Debug, Clone)]
pub struct ChebyshevCrBackward {
    /// Gradient w.r.t. input `x`, flat `(n, channels)`.
    pub grad_x: Vec<f32>,
    /// Gradient w.r.t. Chebyshev coefficients, flat `(channels, k)`.
    pub grad_coef: Vec<f32>,
}

/// The per-element buffers a forward fills for a block of rows: each bank's output, and the one tape.
///
/// Bundled into a struct so the shared row kernel takes four arguments instead of
/// seven (§6.5 #6) and so the sequential and parallel paths call *one*
/// implementation rather than each carrying a copy of the arithmetic (§6.1).
struct ForwardRow<'a, 'b> {
    /// Each bank's output slice over the block.
    outs: &'b mut [&'a mut [f32]],
    x_clamped: &'a mut [f32],
    interior: &'a mut [bool],
}

/// Evaluate every bank's CR spline for every element of a whole block of rows.
///
/// One pass over runs of elements in row-major order, the channel a counter that wraps (not
/// `element % channels`, which measured ~24 % slower): no per-row iterator setup, which at one
/// channel cost as much as the arithmetic. Each run is two passes: the clamp, the segment and the
/// weights of every element at once, then, bank by bank, the control values by channel and segment.
/// The first pass is the banks' one (tensor plan 2026-10-22-cr-banks): banks over one input share
/// their segments and weights, and write one tape.
///
/// # Preconditions
/// Every slice in `w` has the same length as `x_block`, which is a whole multiple
/// of `channels`; one output per bank in `banks`; each bank's table over `channels × knots.len()`
/// coefficients.
#[inline(always)]
fn forward_block<const UNIFORM: bool>(
    banks: &[Segments],
    knots: &Knots,
    channels: usize,
    x_block: &[f32],
    w: ForwardRow<'_, '_>,
) {
    debug_assert_eq!(x_block.len() % channels, 0, "block must be whole rows");
    debug_assert_eq!(banks.len(), w.outs.len(), "one output per bank");
    let (span, last) = (0.5 * (knots.grid() - 1) as f32, (knots.grid() - 2) as f32);
    let mut pass = ElementPass::new();
    let mut ch = 0;
    for start in (0..x_block.len()).step_by(RUN) {
        let end = (start + RUN).min(x_block.len());
        let n = end - start;
        let x = &x_block[start..end];
        let ElementPass {
            segment: seg,
            weights: [w0, w1, w2, w3],
            ..
        } = &mut pass;
        let (seg, w0, w1, w2, w3) = (
            &mut seg[..n],
            &mut w0[..n],
            &mut w1[..n],
            &mut w2[..n],
            &mut w3[..n],
        );
        let (x_clamped, interior) = (&mut w.x_clamped[start..end], &mut w.interior[start..end]);
        // First pass, every element of the run at once: the clamp, the segment and the weights.
        for e in 0..n {
            let raw = x[e];
            let xc = raw.clamp(-1.0, 1.0);
            let (leaf, weights) = if UNIFORM {
                let (i, t) = segment(xc, span, last);
                (i as u32, cr_weights(t))
            } else {
                let (leaf, t, _) = knots.locate(xc, span);
                (leaf as u32, refined_weights(knots, leaf, t))
            };
            (w0[e], w1[e], w2[e], w3[e]) = (weights[0], weights[1], weights[2], weights[3]);
            seg[e] = leaf;
            x_clamped[e] = xc;
            // Closed interval: `±1` is interior, matching the guard this replaces.
            interior[e] = (-1.0..=1.0).contains(&raw);
        }
        // Second pass, bank by bank: each element's four control values, by its channel and segment.
        let first = ch;
        for (segments, out) in banks.iter().zip(w.outs.iter_mut()) {
            let out = &mut out[start..end];
            ch = first;
            for e in 0..n {
                let c = segments.quad(ch, seg[e] as usize);
                out[e] = w0[e] * c[0] + w1[e] * c[1] + w2[e] * c[2] + w3[e] * c[3];
                ch += 1;
                if ch == channels {
                    ch = 0;
                }
            }
        }
    }
}

/// [`forward_block`] instantiated for the knots: uniform knots take the uniform segment arithmetic,
/// which vectorises; refined ones locate their leaves. One choice per call, never per element.
#[inline(always)]
fn forward_block_over(
    banks: &[Segments],
    knots: &Knots,
    channels: usize,
    x_block: &[f32],
    w: ForwardRow<'_, '_>,
) {
    if knots.is_uniform() {
        forward_block::<true>(banks, knots, channels, x_block, w);
    } else {
        forward_block::<false>(banks, knots, channels, x_block, w);
    }
}

/// Forward Catmull--Rom activation.
///
/// # Preconditions
/// - `x.len() == n * channels`
/// - `coef.len() == channels * knots.len()`: one control point per knot ([`Knots`]; the uniform
///   grid of `g` points is `Knots::uniform(g)`)
///
/// # Postconditions
/// Returns output flat `(n, channels)` and a cache for backward. Inputs are
/// clamped to `[-1, 1]`, matching the canonical Python evaluator.
///
/// Allocates both; [`catmull_rom_forward_into`] is the same forward into buffers the caller keeps.
pub fn catmull_rom_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    knots: &Knots,
) -> (Vec<f32>, CatmullRomCache) {
    let mut out = vec![0.0; x.len()];
    let mut cache = CatmullRomCache::default();
    let into = CatmullRomBuffers {
        out: &mut out,
        cache: &mut cache,
    };
    catmull_rom_forward_into(coef, x, n, channels, knots, into);
    (out, cache)
}

/// The two buffers a Catmull--Rom forward writes, owned by the caller: the output, flat
/// `(n, channels)`, and the tape the backward reads.
#[derive(Debug)]
pub struct CatmullRomBuffers<'a> {
    /// Output, flat `(n, channels)`; every element is written.
    pub out: &'a mut [f32],
    /// Refilled in place for the call's shape; its buffers keep their capacity.
    pub cache: &'a mut CatmullRomCache,
}

/// [`catmull_rom_forward`] into caller-owned buffers. Nothing either buffer held before is read.
///
/// # Preconditions
/// As [`catmull_rom_forward`], and `into.out.len() == n * channels`.
///
/// # Postconditions
/// `into.out` and `into.cache` hold exactly what [`catmull_rom_forward`] returns for the same
/// arguments, bit for bit.
pub fn catmull_rom_forward_into(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    knots: &Knots,
    into: CatmullRomBuffers<'_>,
) {
    let CatmullRomBuffers { out, cache } = into;
    let banks = CatmullRomBanksBuffers {
        outs: vec![out],
        cache,
    };
    catmull_rom_banks_forward_into(&[coef], x, n, channels, knots, banks);
}

/// The buffers a forward of several banks writes, owned by the caller: one output per bank, each flat
/// `(n, channels)`, and the one tape the banks share.
#[derive(Debug)]
pub struct CatmullRomBanksBuffers<'a> {
    /// One output per bank, flat `(n, channels)`; every element is written.
    pub outs: Vec<&'a mut [f32]>,
    /// Refilled in place for the call's shape; its buffers keep their capacity.
    pub cache: &'a mut CatmullRomCache,
}

/// Several banks of control points over one input (tensor plan 2026-10-22-cr-banks): bank `b`'s output
/// is [`catmull_rom_forward`] of `coefs[b]`, and the clamp, the segment and the weights of every element
/// are computed once for all of them, into one tape.
///
/// # Preconditions
/// At least one bank; every `coefs[b].len() == channels * knots.len()`; one output per bank, each
/// `n * channels`; `x.len() == n * channels`.
///
/// # Postconditions
/// Each `into.outs[b]` is bit for bit [`catmull_rom_forward`]'s output for `coefs[b]`, and `into.cache`
/// its cache (the same for every bank).
pub fn catmull_rom_banks_forward_into(
    coefs: &[&[f32]],
    x: &[f32],
    n: usize,
    channels: usize,
    knots: &Knots,
    into: CatmullRomBanksBuffers<'_>,
) {
    let CatmullRomBanksBuffers { mut outs, cache } = into;
    // Three is the smallest grid this operator **supports**: `i` ranges over `{0, 1}`, giving two
    // segments, and the outer stencil points are supplied by the endpoint replication
    // `cr_control_indices` already performs. The CogInfoCom HSiKAN configuration uses grid 3, which
    // is why the minimum moved from 4. Grid 2 is out of contract here — not asserted to be
    // mathematically impossible, simply not a configuration this operator is certified for.
    assert!(!coefs.is_empty(), "at least one bank");
    assert_eq!(x.len(), n * channels);
    assert_eq!(outs.len(), coefs.len(), "one output per bank");
    for (coef, out) in coefs.iter().zip(&outs) {
        assert_eq!(coef.len(), channels * knots.len());
        assert_eq!(out.len(), x.len(), "out must be (n, channels)");
    }

    // Refilled, not rebuilt: `resize` keeps the capacity, and the blocks below overwrite every
    // element, so nothing a previous call left is read.
    cache.x_clamped.resize(x.len(), 0.0);
    cache.interior.resize(x.len(), false);
    cache.n = n;
    cache.channels = channels;
    cache.grid = knots.len();
    cache.knots = Some(knots.clone());
    let (x_clamped, interior) = (&mut cache.x_clamped, &mut cache.interior);
    let banks: Vec<Segments> = coefs
        .iter()
        .map(|coef| Segments::of(coef, channels, knots))
        .collect();

    // Rows are independent and write disjoint slices, so this is a pure map and the
    // parallel result is bit-identical to the sequential one at any thread count.
    // Blocks are many rows wide: one row is far too little work per rayon item.
    #[cfg(feature = "parallel")]
    if use_parallel(x.len()) {
        let stride = block_stride(n, channels);
        // Every bank's slice of each block, block-major in one buffer: a block takes its banks'
        // slices as one chunk of it, and nothing is allocated per block.
        let bank_count = outs.len();
        let mut by_bank: Vec<_> = outs.into_iter().map(|out| out.chunks_mut(stride)).collect();
        let mut by_block: Vec<&mut [f32]> =
            Vec::with_capacity(x.len().div_ceil(stride) * bank_count);
        while let Some(first) = by_bank[0].next() {
            by_block.push(first);
            by_block.extend(by_bank[1..].iter_mut().filter_map(Iterator::next));
        }
        by_block
            .par_chunks_mut(bank_count)
            .zip(x_clamped.par_chunks_mut(stride))
            .zip(interior.par_chunks_mut(stride))
            .zip(x.par_chunks(stride))
            .for_each(|(((outs, xc_b), in_b), x_b)| {
                forward_block_over(
                    &banks,
                    knots,
                    channels,
                    x_b,
                    ForwardRow {
                        outs,
                        x_clamped: xc_b,
                        interior: in_b,
                    },
                );
            });
        return;
    }

    forward_block_over(
        &banks,
        knots,
        channels,
        x,
        ForwardRow {
            outs: &mut outs,
            x_clamped,
            interior,
        },
    );
}

/// One chunk's view of the backward's per-element inputs and outputs.
///
/// Every slice covers the same contiguous element range, a whole number of rows
/// long. Bundled for the same reason as [`ForwardRow`]: one shared kernel, and a
/// parameter list that stays under the §6.5 #6 limit.
struct BackwardChunk<'a> {
    grad_x: &'a mut [f32],
    /// Each bank's whole output cotangent; the chunk reads it from `offset`.
    grad_ys: &'a [&'a [f32]],
    /// The chunk's first element in the cotangents.
    offset: usize,
    /// Clamped inputs from the forward; the segment is recomputed from them ([`segment`]).
    x_clamped: &'a [f32],
    /// Per-element interior mask from the forward; see [`CatmullRomCache::interior`].
    interior: &'a [bool],
}

/// One bank over one run of the backward: its table, its cotangent over the run, the run's interior mask
/// and first pass, and the bank's accumulators.
struct BankRun<'a> {
    segments: &'a Segments,
    grad_y: &'a [f32],
    interior: &'a [bool],
    pass: &'a ElementPass,
    accumulators: &'a mut [[f32; 4]],
}

impl BankRun<'_> {
    /// The bank's second pass over the run: each element's `gy · w` into its channel's and segment's
    /// accumulator, and its input-gradient term written (`ASSIGN`) or added into `grad_x`. Returns the
    /// channel counter after the run.
    #[inline(always)]
    fn pull<const UNIFORM: bool, const ASSIGN: bool>(
        self,
        grad_x: &mut [f32],
        (mut ch, channels, segs): (usize, usize, usize),
        du_dx: f32,
    ) -> usize {
        let ElementPass {
            segment: seg,
            weights,
            dweights,
            scale,
        } = self.pass;
        for (e, (gx, &gy)) in grad_x.iter_mut().zip(self.grad_y).enumerate() {
            // The element's channel and segment: its accumulator and its control values.
            let at = ch * segs + seg[e] as usize;
            let slots = &mut self.accumulators[at];
            for k in 0..4 {
                slots[k] += gy * weights[k][e];
            }
            // The input gradient as before: the clamp's own Jacobian is `1` inside the closed
            // interval, `0` outside, read from the recorded `interior` mask (not from `x_clamped`,
            // which is inside by construction).
            let term = if self.interior[e] {
                let c = self.segments.quads[at];
                let mut dy_dt = 0.0;
                for k in 0..4 {
                    dy_dt += dweights[k][e] * c[k];
                }
                // A leaf at depth `d` is `2^d` times narrower in `u`: `dt/dx = span · 2^d`.
                if UNIFORM {
                    gy * dy_dt * du_dx
                } else {
                    gy * dy_dt * (du_dx * scale[e])
                }
            } else {
                0.0
            };
            if ASSIGN {
                *gx = term;
            } else {
                *gx += term;
            }
            ch += 1;
            if ch == channels {
                ch = 0;
            }
        }
        ch
    }
}

/// Accumulate one contiguous chunk of rows into `grad_x` and each bank's coefficient gradient.
///
/// # Preconditions
/// Every slice in `c` has the same length, a multiple of `channels`, and the chunk
/// starts on a row boundary — otherwise the channel counter is wrong. One cotangent per bank;
/// `grad_coefs` holds the banks' gradients one after another, `channels · knots.len()` each.
///
/// # Postconditions
/// `grad_x` is *written*: the first bank's term assigned, the others' added in bank order, so one bank
/// writes exactly the single-bank value. `grad_coefs` is *accumulated* into, so the caller owns its
/// initial value.
#[inline(always)]
fn backward_chunk<const UNIFORM: bool>(
    banks: &[Segments],
    channels: usize,
    knots: &Knots,
    c: BackwardChunk<'_>,
    grad_coefs: &mut [f32],
) {
    debug_assert_eq!(c.grad_x.len() % channels, 0, "chunk must be whole rows");
    debug_assert_eq!(banks.len(), c.grad_ys.len(), "one cotangent per bank");
    // `du/dx` is the forward's `span`: one value, named for each of its two uses.
    let du_dx = 0.5 * (knots.grid() - 1) as f32;
    let last = (knots.grid() - 2) as f32;
    let segs = knots.controls().len();
    // The coefficient gradient by segment (Nagare plan 2026-10-08-orthtree-hypergraph-convolution,
    // Amendments 3 and 6): one accumulator per bank, channel and segment, four control slots wide,
    // each element adding `gy · w` into its own segment's four slots at once; the chunk's sums are
    // folded into its coefficients, per channel, in segment then slot order at the end. Per
    // accumulator these are the additions of the segment-masked form, which added `0 · gy · w` into
    // every other segment, in the same row order without its zeros, so the bits are its bits for
    // finite values. The masked form's pass over every segment made the pullback slower, not
    // faster: 7x at one channel, 1.9x at 32 (measured 2026-10-09).
    let per_bank = channels * segs;
    let mut by_segment = vec![[0.0f32; 4]; banks.len() * per_bank];
    // Runs of elements in row-major order, the channel a counter that wraps, as the forward: the
    // segments, weights and derivatives of a run at once, then, bank by bank, its accumulators and
    // input gradients by channel and segment.
    let mut pass = ElementPass::new();
    let mut ch = 0;
    for start in (0..c.x_clamped.len()).step_by(RUN) {
        let end = (start + RUN).min(c.x_clamped.len());
        let n = end - start;
        let x_clamped = &c.x_clamped[start..end];
        let ElementPass {
            segment: seg,
            weights,
            dweights,
            scale,
        } = &mut pass;
        let seg = &mut seg[..n];
        for e in 0..n {
            let (w, dw) = if UNIFORM {
                let (i, t) = segment(x_clamped[e], du_dx, last);
                seg[e] = i as u32;
                (cr_weights(t), cr_weight_derivatives(t))
            } else {
                let (leaf, t, cells) = knots.locate(x_clamped[e], du_dx);
                seg[e] = leaf as u32;
                scale[e] = cells;
                (
                    refined_weights(knots, leaf, t),
                    refined_weight_derivatives(knots, leaf, t),
                )
            };
            for k in 0..4 {
                weights[k][e] = w[k];
                dweights[k][e] = dw[k];
            }
        }
        let (grad_x, interior) = (&mut c.grad_x[start..end], &c.interior[start..end]);
        let first = ch;
        for (b, (segments, grad_y)) in banks.iter().zip(c.grad_ys).enumerate() {
            let at = c.offset + start..c.offset + end;
            let run = BankRun {
                segments,
                grad_y: &grad_y[at],
                interior,
                pass: &pass,
                accumulators: &mut by_segment[b * per_bank..(b + 1) * per_bank],
            };
            ch = first;
            // The first bank writes the input gradient, the others add to it: one bank is the
            // single-bank kernel exactly, with no choice inside the element loop.
            ch = if b == 0 {
                run.pull::<UNIFORM, true>(grad_x, (ch, channels, segs), du_dx)
            } else {
                run.pull::<UNIFORM, false>(grad_x, (ch, channels, segs), du_dx)
            };
        }
    }
    let coefs = channels * knots.len();
    for (bank_grad, bank_accumulators) in grad_coefs
        .chunks_exact_mut(coefs)
        .zip(by_segment.chunks_exact(per_bank))
    {
        let accumulators = bank_accumulators.chunks_exact(segs);
        for (channel_grad, accumulators) in
            bank_grad.chunks_exact_mut(knots.len()).zip(accumulators)
        {
            for (slots, controls) in accumulators.iter().zip(&banks[0].controls) {
                for (&sum, &j) in slots.iter().zip(controls) {
                    channel_grad[j] += sum;
                }
            }
        }
    }
}

/// Backward Catmull--Rom activation.
///
/// # Preconditions
/// `grad_y` has the same flat shape as the forward input/output.
///
/// # Postconditions
/// Returns gradients for input and control points. The input gradient is zero
/// outside the clamp interval.
/// # History
/// Until 2026-08-25 this returned the interior derivative everywhere, because the guard tested the
/// already-clamped value and was a tautology. A selector kept that arithmetic reachable until
/// 2026-08-29; it is gone, and the retired behaviour is frozen — and still executable — in
/// `hymeko_numeric_vv/tests/support/frozen_clamp_oracle.rs`. `grad_coef` was never affected: it
/// accumulates outside the guard.
///
/// Allocates `grad_x`; [`catmull_rom_backward_into`] writes it into a buffer the caller keeps.
pub fn catmull_rom_backward(
    coef: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
) -> CatmullRomBackward {
    let mut grad_x = vec![0.0; grad_y.len()];
    let grad_coef = catmull_rom_backward_into(coef, cache, grad_y, &mut grad_x);
    CatmullRomBackward { grad_x, grad_coef }
}

/// [`catmull_rom_backward`] with the input gradient written into `grad_x`, every element of it;
/// returns the control-point gradient. Every element of `grad_x` is written, `0` outside the clamp
/// interval, whatever the buffer held before.
///
/// # Preconditions
/// As [`catmull_rom_backward`], and `grad_x.len() == grad_y.len()`.
///
/// # Postconditions
/// `grad_x` and the returned gradient are exactly [`catmull_rom_backward`]'s, bit for bit.
pub fn catmull_rom_backward_into(
    coef: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
    grad_x: &mut [f32],
) -> Vec<f32> {
    let mut banks = catmull_rom_banks_backward_into(&[coef], cache, &[grad_y], grad_x);
    banks.swap_remove(0)
}

/// The pullback of [`catmull_rom_banks_forward_into`] (tensor plan 2026-10-22-cr-banks): each bank's
/// control-point gradient, and the input gradient summed over the banks, written into `grad_x`.
///
/// # Preconditions
/// The banks and the cache of one banks forward; one cotangent per bank, each `n · channels`;
/// `grad_x.len() == n · channels`.
///
/// # Postconditions
/// Bank `b`'s gradient is bit for bit [`catmull_rom_backward_into`]'s for `coefs[b]` and `grad_ys[b]`;
/// `grad_x` is the banks' input gradients summed in bank order, every element written, `0` outside the
/// clamp interval.
pub fn catmull_rom_banks_backward_into(
    coefs: &[&[f32]],
    cache: &CatmullRomCache,
    grad_ys: &[&[f32]],
    grad_x: &mut [f32],
) -> Vec<Vec<f32>> {
    let elems = cache.n * cache.channels;
    let len = cache.channels * cache.grid;
    assert!(!coefs.is_empty(), "at least one bank");
    assert_eq!(grad_ys.len(), coefs.len(), "one cotangent per bank");
    for (coef, grad_y) in coefs.iter().zip(grad_ys) {
        assert_eq!(grad_y.len(), elems);
        assert_eq!(coef.len(), len);
    }
    assert_eq!(grad_x.len(), elems, "grad_x must be (n, channels)");
    let knots = cache
        .knots
        .as_ref()
        .expect("a forward wrote the cache, and recorded its knots");

    // Rows split into a FIXED number of blocks. Each block writes its own disjoint
    // `grad_x` slice (bit-exact) and accumulates into its own private gradient
    // slot per bank — 48 floats for california.
    //
    // **Both paths block identically.** Only the iteration differs, never the
    // arithmetic or the summation order, which is what lets `use_parallel` consult
    // the thread count: a 1-core and a 16-core host produce the same bytes. Doing
    // it the other way round — a flat sum when sequential — would have made the
    // result depend on the machine.
    //
    // The partials live in ONE flat buffer rather than a `Vec<Vec<f32>>`: the
    // per-block allocation cost is not hypothetical here, since a training run
    // calls this 300 times, and the 2026-08-17 fusion work established that the
    // allocator dominates this crate's kernels well before data traffic does.
    // Measured: the `Vec<Vec<f32>>` version cost 132.6 µs/epoch, this one 115.7.
    let stride = block_stride(cache.n, cache.channels);
    let blocks = grad_x.len().div_ceil(stride).max(1);
    let block_len = coefs.len() * len;
    let mut partials = vec![0.0f32; blocks * block_len];

    let banks: Vec<Segments> = coefs
        .iter()
        .map(|coef| Segments::of(coef, cache.channels, knots))
        .collect();
    let uniform = knots.is_uniform();
    let kernel = |(block, (gx, local)): (usize, (&mut [f32], &mut [f32]))| {
        let span = block * stride..(block * stride + gx.len());
        let chunk = BackwardChunk {
            grad_ys,
            offset: span.start,
            x_clamped: &cache.x_clamped[span.clone()],
            interior: &cache.interior[span],
            grad_x: gx,
        };
        // One choice per call: uniform knots take the uniform segment arithmetic.
        if uniform {
            backward_chunk::<true>(&banks, cache.channels, knots, chunk, local);
        } else {
            backward_chunk::<false>(&banks, cache.channels, knots, chunk, local);
        }
    };

    #[cfg(feature = "parallel")]
    if use_parallel(elems) {
        grad_x
            .par_chunks_mut(stride)
            .zip(partials.par_chunks_mut(block_len))
            .enumerate()
            .for_each(kernel);
    } else {
        grad_x
            .chunks_mut(stride)
            .zip(partials.chunks_mut(block_len))
            .enumerate()
            .for_each(kernel);
    }

    #[cfg(not(feature = "parallel"))]
    grad_x
        .chunks_mut(stride)
        .zip(partials.chunks_mut(block_len))
        .enumerate()
        .for_each(kernel);

    // Summed in block order, which is fixed by the buffer layout and not by the
    // scheduler — this is what makes every bank's gradient thread-count independent.
    let mut grad_coefs = vec![vec![0.0f32; len]; coefs.len()];
    for local in partials.chunks(block_len) {
        for (grad, values) in grad_coefs.iter_mut().zip(local.chunks(len)) {
            for (acc, value) in grad.iter_mut().zip(values) {
                *acc += value;
            }
        }
    }
    grad_coefs
}

/// Chebyshev basis evaluated at uniform CR knots in `[-1, 1]`.
///
/// Returns flat `(grid, k)`.
pub fn chebyshev_knot_basis(grid: usize, k: usize) -> Vec<f32> {
    assert!(grid >= 2);
    assert!(k >= 1);
    let mut out = vec![0.0; grid * k];
    for g in 0..grid {
        let x = -1.0 + 2.0 * g as f32 / (grid - 1) as f32;
        let terms = chebyshev_terms(x, k);
        out[g * k..(g + 1) * k].copy_from_slice(&terms);
    }
    out
}

/// Build Chebyshev-CR control points.
///
/// `coef` is flat `(channels, k)`, `basis` is flat `(grid, k)`, and the return
/// is flat `(channels, grid)`.
pub fn chebyshev_control_points(
    coef: &[f32],
    basis: &[f32],
    channels: usize,
    grid: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(coef.len(), channels * k);
    assert_eq!(basis.len(), grid * k);
    let mut out = vec![0.0; channels * grid];
    for ch in 0..channels {
        for g in 0..grid {
            let mut acc = 0.0;
            for term in 0..k {
                acc += coef[ch * k + term] * basis[g * k + term];
            }
            out[ch * grid + g] = acc;
        }
    }
    out
}

/// Forward Chebyshev-CR train path: generate CR control points, then run CR.
pub fn chebyshev_cr_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
    k: usize,
) -> (Vec<f32>, CatmullRomCache, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0; x.len()];
    let mut cache = CatmullRomCache::default();
    let into = CatmullRomBuffers {
        out: &mut y,
        cache: &mut cache,
    };
    let (control, basis) = chebyshev_cr_forward_into(coef, x, n, channels, grid, k, into);
    (y, cache, control, basis)
}

/// [`chebyshev_cr_forward`] into caller-owned buffers (see [`catmull_rom_forward_into`]); returns
/// the derived control points and the knot basis, `channels · grid` and `grid · k` long.
pub fn chebyshev_cr_forward_into(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
    k: usize,
    into: CatmullRomBuffers<'_>,
) -> (Vec<f32>, Vec<f32>) {
    let CatmullRomBuffers { out, cache } = into;
    let banks = CatmullRomBanksBuffers {
        outs: vec![out],
        cache,
    };
    let (mut controls, basis) =
        chebyshev_cr_banks_forward_into(&[coef], x, n, channels, (grid, k), banks);
    (controls.swap_remove(0), basis)
}

/// [`chebyshev_cr_forward_into`] for several banks of Chebyshev terms over one input (tensor plan
/// 2026-10-22-cr-banks): each bank's control points generated, then one banks forward
/// ([`catmull_rom_banks_forward_into`]). Returns each bank's control points and the knot basis.
///
/// # Preconditions
/// At least one bank, each `channels · k` terms; `grid >= 3`; the buffers as the banks forward's.
pub fn chebyshev_cr_banks_forward_into(
    coefs: &[&[f32]],
    x: &[f32],
    n: usize,
    channels: usize,
    (grid, k): (usize, usize),
    into: CatmullRomBanksBuffers<'_>,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let basis = chebyshev_knot_basis(grid, k);
    let controls: Vec<Vec<f32>> = coefs
        .iter()
        .map(|coef| chebyshev_control_points(coef, &basis, channels, grid, k))
        .collect();
    let banks: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
    // The Chebyshev terms are read at the uniform knots, so the spline they generate is uniform.
    catmull_rom_banks_forward_into(&banks, x, n, channels, &Knots::uniform(grid), into);
    (controls, basis)
}

/// Backward Chebyshev-CR train path.
/// The clamp reaches `grad_x` only; the Chebyshev coefficient gradient is projected from
/// `grad_coef`, which the clamp does not touch.
pub fn chebyshev_cr_backward(
    control_points: &[f32],
    basis: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
    k: usize,
) -> ChebyshevCrBackward {
    let mut grad_x = vec![0.0; grad_y.len()];
    let grad_coef =
        chebyshev_cr_backward_into(control_points, basis, cache, grad_y, k, &mut grad_x);
    ChebyshevCrBackward { grad_x, grad_coef }
}

/// [`chebyshev_cr_backward`] with the input gradient written into `grad_x` (see
/// [`catmull_rom_backward_into`]); returns the Chebyshev-coefficient gradient.
pub fn chebyshev_cr_backward_into(
    control_points: &[f32],
    basis: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
    k: usize,
    grad_x: &mut [f32],
) -> Vec<f32> {
    let mut banks =
        chebyshev_cr_banks_backward_into(&[control_points], basis, cache, &[grad_y], k, grad_x);
    banks.swap_remove(0)
}

/// The pullback of [`chebyshev_cr_banks_forward_into`]: each bank's Chebyshev-term gradient, projected
/// from its control-point gradient, and the input gradient summed over the banks into `grad_x`
/// ([`catmull_rom_banks_backward_into`]).
pub fn chebyshev_cr_banks_backward_into(
    controls: &[&[f32]],
    basis: &[f32],
    cache: &CatmullRomCache,
    grad_ys: &[&[f32]],
    k: usize,
    grad_x: &mut [f32],
) -> Vec<Vec<f32>> {
    catmull_rom_banks_backward_into(controls, cache, grad_ys, grad_x)
        .iter()
        .map(|cr_grad_coef| chebyshev_terms_gradient(cr_grad_coef, basis, cache, k))
        .collect()
}

/// The Chebyshev terms' gradient from their control points' (`(channels, grid)`): each term's the
/// control gradients weighted by the term at the knots.
fn chebyshev_terms_gradient(
    cr_grad_coef: &[f32],
    basis: &[f32],
    cache: &CatmullRomCache,
    k: usize,
) -> Vec<f32> {
    let mut grad_coef = vec![0.0; cache.channels * k];
    for ch in 0..cache.channels {
        for term in 0..k {
            let mut acc = 0.0;
            for g in 0..cache.grid {
                acc += cr_grad_coef[ch * cache.grid + g] * basis[g * k + term];
            }
            grad_coef[ch * k + term] = acc;
        }
    }
    grad_coef
}

/// Direct Chebyshev deploy forward, avoiding CR gather.
///
/// `coef` is flat `(channels, k)`, `x` is flat `(n, channels)`.
pub fn chebyshev_deploy_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(coef.len(), channels * k);
    assert_eq!(x.len(), n * channels);
    let mut out = vec![0.0; x.len()];
    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let xc = x[idx].clamp(-1.0, 1.0);
            out[idx] = chebyshev_eval_channel(coef, ch, k, xc);
        }
    }
    out
}

/// Direct Chebyshev deploy backward.
///
/// Returns gradients for input and Chebyshev coefficients.
pub fn chebyshev_deploy_backward(
    coef: &[f32],
    x: &[f32],
    grad_y: &[f32],
    n: usize,
    channels: usize,
    k: usize,
) -> ChebyshevCrBackward {
    assert_eq!(coef.len(), channels * k);
    assert_eq!(x.len(), n * channels);
    assert_eq!(grad_y.len(), x.len());
    let mut grad_x = vec![0.0; x.len()];
    let mut grad_coef = vec![0.0; coef.len()];
    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let xc = x[idx].clamp(-1.0, 1.0);
            let gy = grad_y[idx];
            accumulate_chebyshev_coef_grad(&mut grad_coef, ch, k, xc, gy);
            if (-1.0..=1.0).contains(&x[idx]) {
                grad_x[idx] = gy * chebyshev_eval_derivative_channel(coef, ch, k, xc);
            }
        }
    }
    ChebyshevCrBackward { grad_x, grad_coef }
}

fn chebyshev_eval_channel(coef: &[f32], ch: usize, k: usize, x: f32) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let base = ch * k;
    let mut acc = coef[base];
    if k == 1 {
        return acc;
    }
    let mut prev = 1.0;
    let mut curr = x;
    acc += curr * coef[base + 1];
    for term in 2..k {
        let next = 2.0 * x * curr - prev;
        acc += next * coef[base + term];
        prev = curr;
        curr = next;
    }
    acc
}

fn chebyshev_eval_derivative_channel(coef: &[f32], ch: usize, k: usize, x: f32) -> f32 {
    if k <= 1 {
        return 0.0;
    }
    let base = ch * k;
    let mut prev = 1.0;
    let mut curr = x;
    let mut d_prev = 0.0;
    let mut d_curr = 1.0;
    let mut acc = coef[base + 1];
    for term in 2..k {
        let next = 2.0 * x * curr - prev;
        let d_next = 2.0 * curr + 2.0 * x * d_curr - d_prev;
        acc += coef[base + term] * d_next;
        prev = curr;
        curr = next;
        d_prev = d_curr;
        d_curr = d_next;
    }
    acc
}

fn accumulate_chebyshev_coef_grad(grad_coef: &mut [f32], ch: usize, k: usize, x: f32, gy: f32) {
    if k == 0 {
        return;
    }
    let base = ch * k;
    grad_coef[base] += gy;
    if k == 1 {
        return;
    }
    let mut prev = 1.0;
    let mut curr = x;
    grad_coef[base + 1] += gy * curr;
    for term in 2..k {
        let next = 2.0 * x * curr - prev;
        grad_coef[base + term] += gy * next;
        prev = curr;
        curr = next;
    }
}

fn cr_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

fn cr_weight_derivatives(t: f32) -> [f32; 4] {
    let t2 = t * t;
    [
        0.5 * (-3.0 * t2 + 4.0 * t - 1.0),
        0.5 * (9.0 * t2 - 10.0 * t),
        0.5 * (-9.0 * t2 + 8.0 * t + 1.0),
        0.5 * (3.0 * t2 - 2.0 * t),
    ]
}

/// The weights of refined knots' leaf `leaf` at `t`: the non-uniform Catmull–Rom, the cubic Hermite
/// with tangents `(p_{k+1} − p_{k−1}) / (x_{k+1} − x_{k−1})`: `(−h₁₀ r₀, h₀₀ − h₁₁ r₁, h₀₁ + h₁₀ r₀,
/// h₁₁ r₁)` with the leaf's tangent shares -- the uniform weights in exact arithmetic where both
/// shares are ½. One form for every leaf, no branch per element (measured 2026-10-16: −7 % forward,
/// −6 % backward at level 4 against a branch to the uniform expressions).
#[inline(always)]
fn refined_weights(knots: &Knots, leaf: usize, t: f32) -> [f32; 4] {
    let (t2, [r0, r1]) = (t * t, knots.leaf_ratio(leaf));
    let t3 = t2 * t;
    let (h00, h10) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t);
    let (h01, h11) = (-2.0 * t3 + 3.0 * t2, t3 - t2);
    [-h10 * r0, h00 - h11 * r1, h01 + h10 * r0, h11 * r1]
}

/// The `t`-derivatives of [`refined_weights`].
#[inline(always)]
fn refined_weight_derivatives(knots: &Knots, leaf: usize, t: f32) -> [f32; 4] {
    let (t2, [r0, r1]) = (t * t, knots.leaf_ratio(leaf));
    let (d00, d10) = (6.0 * t2 - 6.0 * t, 3.0 * t2 - 4.0 * t + 1.0);
    let (d01, d11) = (-6.0 * t2 + 6.0 * t, 3.0 * t2 - 2.0 * t);
    [-d10 * r0, d00 - d11 * r1, d01 + d10 * r0, d11 * r1]
}

fn chebyshev_terms(x: f32, k: usize) -> Vec<f32> {
    let mut terms = vec![0.0; k];
    if k == 0 {
        return terms;
    }
    terms[0] = 1.0;
    if k > 1 {
        terms[1] = x;
    }
    for term in 2..k {
        terms[term] = 2.0 * x * terms[term - 1] - terms[term - 2];
    }
    terms
}

#[cfg(test)]
mod tests {
    /// The uniform grid's control indices, the ends replicated: the naive reference's own, as the
    /// kernel read them before the knots carried them.
    fn cr_control_indices(i: usize, grid: usize) -> [usize; 4] {
        [
            i.saturating_sub(1).min(grid - 1),
            i.min(grid - 1),
            (i + 1).min(grid - 1),
            (i + 2).min(grid - 1),
        ]
    }

    use super::*;

    fn loss_sum_cr(coef: &[f32], x: &[f32], n: usize, channels: usize, grid: usize) -> f32 {
        catmull_rom_forward(coef, x, n, channels, &crate::knots::Knots::uniform(grid))
            .0
            .iter()
            .sum()
    }

    // ── row-parallelism (2026-08-17) ────────────────────────────────────────
    //
    // These pin the two guarantees the parallel paths must keep: the pure maps are
    // bit-exact against an independent reference, and the `grad_coef` reduction is
    // identical across thread counts. The reference below is deliberately a
    // separate naive implementation rather than a call back into the production
    // code, so a bug in the shared row kernel cannot hide by cancelling itself.

    fn pseudo_input(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(salt.wrapping_mul(40_503));
                // Spans past ±1 on purpose, so the clamp and the dead-gradient
                // branch in the backward are both exercised.
                (h >> 8) as f32 / (1u32 << 24) as f32 * 2.6 - 1.3
            })
            .collect()
    }

    /// Independent sequential reference for the forward.
    #[allow(clippy::type_complexity)] // mirrors the production return shape
    fn naive_forward(
        coef: &[f32],
        x: &[f32],
        n: usize,
        channels: usize,
        grid: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<u8>, Vec<f32>) {
        let (mut out, mut xc, mut idx, mut ts) = (
            vec![0.0f32; x.len()],
            vec![0.0f32; x.len()],
            vec![0u8; x.len()],
            vec![0.0f32; x.len()],
        );
        for row in 0..n {
            for ch in 0..channels {
                let e = row * channels + ch;
                let clamped = x[e].clamp(-1.0, 1.0);
                let u = (clamped + 1.0) * 0.5 * (grid - 1) as f32;
                let i = (u.floor() as usize).min(grid - 2);
                let t = u - i as f32;
                let w = cr_weights(t);
                let control = cr_control_indices(i, grid);
                let base = ch * grid;
                out[e] = w[0] * coef[base + control[0]]
                    + w[1] * coef[base + control[1]]
                    + w[2] * coef[base + control[2]]
                    + w[3] * coef[base + control[3]];
                xc[e] = clamped;
                idx[e] = i as u8;
                ts[e] = t;
            }
        }
        (out, xc, idx, ts)
    }

    /// Independent sequential reference for the backward: a straightforward mirror of the blocked
    /// kernel.
    ///
    /// It reads the recorded `interior` mask exactly as the kernel does. This is the test's own
    /// reference implementation of the same mathematics, not a recorded artifact.
    fn naive_backward(
        coef: &[f32],
        cache: &CatmullRomCache,
        grad_y: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut grad_x = vec![0.0f32; grad_y.len()];
        let mut grad_coef = vec![0.0f32; coef.len()];
        let du_dx = 0.5 * (cache.grid - 1) as f32;
        for row in 0..cache.n {
            for ch in 0..cache.channels {
                let e = row * cache.channels + ch;
                let gy = grad_y[e];
                let u = (cache.x_clamped[e] + 1.0) * 0.5 * (cache.grid - 1) as f32;
                let i = (u.floor() as usize).min(cache.grid - 2);
                let t = u - i as f32;
                let w = cr_weights(t);
                let dw = cr_weight_derivatives(t);
                let control = cr_control_indices(i, cache.grid);
                let base = ch * cache.grid;
                for k in 0..4 {
                    grad_coef[base + control[k]] += gy * w[k];
                }
                if cache.interior[e] {
                    let mut dy_dt = 0.0;
                    for k in 0..4 {
                        dy_dt += dw[k] * coef[base + control[k]];
                    }
                    grad_x[e] = gy * dy_dt * du_dx;
                }
            }
        }
        (grad_x, grad_coef)
    }

    /// The forward is a pure map, so it must be **bit-exact** on both sides of the
    /// parallel threshold — including a row count that does not divide the chunk
    /// count, which is where an off-by-one in the chunking would show up.
    #[test]
    fn forward_is_bit_exact_against_the_reference_on_both_paths() {
        let (channels, grid) = (8usize, 6usize);
        // 511 * 8 = 4088 < PAR_MIN_ELEMS <= 513 * 8 = 4104: straddles the boundary,
        // and 1601 is coprime with the 64 chunks.
        for n in [1usize, 7, 511, 513, 1600, 1601, 4099] {
            let x = pseudo_input(n * channels, 1);
            let coef = pseudo_input(channels * grid, 2);
            let (out, cache) =
                catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid));
            let (want_out, want_xc, want_idx, want_t) = naive_forward(&coef, &x, n, channels, grid);
            assert_eq!(out, want_out, "n={n}: output must be bit-exact");
            assert_eq!(cache.x_clamped, want_xc, "n={n}: clamped input");
            // The cache no longer stores the segment; the one expression both passes evaluate over
            // uniform knots must still agree with the reference's, bit for bit.
            let span = 0.5 * (grid - 1) as f32;
            for e in 0..x.len() {
                let (i, t) = segment(cache.x_clamped[e], span, (grid - 2) as f32);
                assert_eq!(
                    i as usize, want_idx[e] as usize,
                    "n={n}: segment index at {e}"
                );
                assert_eq!(t.to_bits(), want_t[e].to_bits(), "n={n}: coordinate at {e}");
            }
            assert_eq!(
                cache.retained_floats(),
                n * channels,
                "n={n}: one f32 per element"
            );
        }
    }

    /// `grad_x` is a pure map and must be bit-exact. `grad_coef` is a reduction, so
    /// it is only required to be *close* to the sequential order — the strict
    /// requirement on it is thread-count independence, tested separately.
    #[test]
    fn backward_grad_x_is_bit_exact_and_grad_coef_matches_closely() {
        let (channels, grid) = (8usize, 6usize);
        for n in [1usize, 511, 513, 1600, 1601] {
            let x = pseudo_input(n * channels, 3);
            let coef = pseudo_input(channels * grid, 4);
            let grad_y = pseudo_input(n * channels, 5);
            let (_, cache) =
                catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid));
            let got = catmull_rom_backward(&coef, &cache, &grad_y);
            let (want_gx, want_gc) = naive_backward(&coef, &cache, &grad_y);

            assert_eq!(got.grad_x, want_gx, "n={n}: grad_x must be bit-exact");

            // Relative to the magnitude actually accumulated, not to 1.0 — the sum
            // grows with n, so an absolute tolerance would silently loosen.
            let scale = want_gc.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
            for (i, (g, w)) in got.grad_coef.iter().zip(&want_gc).enumerate() {
                let rel = (g - w).abs() / scale;
                assert!(rel < 1e-5, "n={n} coef[{i}]: {g} vs {w} (rel {rel:.2e})");
            }
        }
    }

    /// **The load-bearing test.** A reduction whose result depends on the thread
    /// count makes every downstream training run irreproducible across machines.
    /// `GRAD_COEF_CHUNKS` is a constant precisely so this holds; if someone
    /// replaces it with `current_num_threads()`, this fails.
    #[cfg(feature = "parallel")]
    #[test]
    fn backward_is_byte_identical_across_thread_counts() {
        let (n, channels, grid) = (4099usize, 8usize, 6usize);
        let x = pseudo_input(n * channels, 6);
        let coef = pseudo_input(channels * grid, 7);
        let grad_y = pseudo_input(n * channels, 8);
        let (_, cache) =
            catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid));

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| catmull_rom_backward(&coef, &cache, &grad_y))
        };

        let reference = run(1);
        for threads in [2usize, 3, 8, 16] {
            let got = run(threads);
            assert_eq!(
                got.grad_coef, reference.grad_coef,
                "grad_coef changed at {threads} threads — the reduction order is \
                 leaking the thread count"
            );
            assert_eq!(got.grad_x, reference.grad_x, "grad_x at {threads} threads");
        }
    }

    /// The forward must likewise not depend on the thread count. It is a pure map,
    /// so this is bit-exactness rather than merely determinism.
    #[cfg(feature = "parallel")]
    #[test]
    fn forward_is_byte_identical_across_thread_counts() {
        let (n, channels, grid) = (4099usize, 8usize, 6usize);
        let x = pseudo_input(n * channels, 9);
        let coef = pseudo_input(channels * grid, 10);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| {
                    catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid))
                })
        };
        let (reference, _) = run(1);
        for threads in [2usize, 8, 16] {
            let (got, _) = run(threads);
            assert_eq!(got, reference, "forward changed at {threads} threads");
        }
    }

    /// The dispatch decision itself, which is a performance contract rather than a
    /// numerical one: iris-scale work (`120 x 4 = 480` elements) must never enter a
    /// rayon region — the 2026-07-16 record has a dense layer at that scale going
    /// 6x *slower* on 18 threads — while california-scale work must.
    ///
    /// Exercised through `use_parallel` inside real pools, so this tests behaviour
    /// rather than restating the constants.
    #[cfg(feature = "parallel")]
    #[test]
    fn dispatch_sends_only_large_shapes_to_rayon() {
        let in_pool = |threads: usize, elems: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| use_parallel(elems))
        };
        const IRIS: usize = 120 * 4;
        const CALIFORNIA: usize = 1600 * 8;

        assert!(!in_pool(8, IRIS), "iris-scale work must stay sequential");
        assert!(in_pool(8, CALIFORNIA), "california-scale work must go wide");

        // With one worker there is nothing to gain and ~13 % of overhead to lose,
        // so even a large shape stays sequential. This is only sound because both
        // paths produce identical bytes — see `backward_is_byte_identical_…`.
        assert!(
            !in_pool(1, CALIFORNIA),
            "a single-threaded host must not pay for a rayon region"
        );
    }

    #[test]
    fn catmull_rom_interpolates_control_points() {
        let coef = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (y, _) = catmull_rom_forward(&coef, &x, 5, 1, &crate::knots::Knots::uniform(5));
        for (got, want) in y.iter().zip(coef.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn catmull_rom_backward_matches_finite_difference() {
        let n = 3;
        let channels = 2;
        let grid = 5;
        let coef = vec![0.1, -0.2, 0.3, 0.7, -0.1, 0.5, -0.4, 0.2, 0.1, 0.9];
        let x = vec![-0.7, -0.2, 0.15, 0.4, 0.72, -0.55];
        let (_, cache) =
            catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid));
        let grad_y = vec![1.0; x.len()];
        let grad = catmull_rom_backward(&coef, &cache, &grad_y);
        let eps = 1e-3;

        for idx in 0..coef.len() {
            let mut plus = coef.clone();
            let mut minus = coef.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let num = (loss_sum_cr(&plus, &x, n, channels, grid)
                - loss_sum_cr(&minus, &x, n, channels, grid))
                / (2.0 * eps);
            assert!(
                (grad.grad_coef[idx] - num).abs() < 1e-3,
                "coef[{idx}] analytic={} numeric={num}",
                grad.grad_coef[idx]
            );
        }

        for idx in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let num = (loss_sum_cr(&coef, &plus, n, channels, grid)
                - loss_sum_cr(&coef, &minus, n, channels, grid))
                / (2.0 * eps);
            assert!(
                (grad.grad_x[idx] - num).abs() < 1e-3,
                "x[{idx}] analytic={} numeric={num}",
                grad.grad_x[idx]
            );
        }
    }

    #[test]
    fn chebyshev_control_points_match_manual_basis() {
        let basis = chebyshev_knot_basis(3, 3);
        assert_eq!(basis, vec![1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0]);
        let coef = vec![2.0, 3.0, 5.0];
        let control = chebyshev_control_points(&coef, &basis, 1, 3, 3);
        assert_eq!(control, vec![4.0, -3.0, 10.0]);
    }

    #[test]
    fn chebyshev_deploy_backward_matches_finite_difference() {
        let n = 3;
        let channels = 2;
        let k = 4;
        let coef = vec![0.2, -0.3, 0.1, 0.05, -0.4, 0.7, -0.2, 0.3];
        let x = vec![-0.8, -0.3, 0.1, 0.35, 0.7, -0.6];
        let grad_y = vec![1.0; x.len()];
        let grad = chebyshev_deploy_backward(&coef, &x, &grad_y, n, channels, k);
        let eps = 1e-3;

        for idx in 0..coef.len() {
            let mut plus = coef.clone();
            let mut minus = coef.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let yp: f32 = chebyshev_deploy_forward(&plus, &x, n, channels, k)
                .iter()
                .sum();
            let ym: f32 = chebyshev_deploy_forward(&minus, &x, n, channels, k)
                .iter()
                .sum();
            let num = (yp - ym) / (2.0 * eps);
            assert!((grad.grad_coef[idx] - num).abs() < 1e-3);
        }

        for idx in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let yp: f32 = chebyshev_deploy_forward(&coef, &plus, n, channels, k)
                .iter()
                .sum();
            let ym: f32 = chebyshev_deploy_forward(&coef, &minus, n, channels, k)
                .iter()
                .sum();
            let num = (yp - ym) / (2.0 * eps);
            assert!((grad.grad_x[idx] - num).abs() < 1e-3);
        }
    }

    #[test]
    fn chebyshev_cr_backward_matches_chain_rule_shape() {
        let n = 2;
        let channels = 2;
        let grid = 6;
        let k = 3;
        let coef = vec![0.1, 0.2, -0.1, 0.4, -0.3, 0.2];
        let x = vec![-0.6, -0.1, 0.25, 0.8];
        let (y, cache, control, basis) = chebyshev_cr_forward(&coef, &x, n, channels, grid, k);
        assert_eq!(y.len(), x.len());
        let grad = chebyshev_cr_backward(&control, &basis, &cache, &vec![1.0; y.len()], k);
        assert_eq!(grad.grad_x.len(), x.len());
        assert_eq!(grad.grad_coef.len(), coef.len());
        assert!(grad.grad_x.iter().all(|v| v.is_finite()));
        assert!(grad.grad_coef.iter().all(|v| v.is_finite()));
    }
}
