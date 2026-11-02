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
//!
//! # The power form
//!
//! On every leaf `ℓ` the four weights are a cubic in the local coordinate, `w(t) = B_ℓ τ(t)` with
//! `τ = (1, t, t², t³)` and `B_ℓ` the leaf's basis ([`Knots`]), so a channel's spline on that leaf is
//! one cubic in `t` (Nagare plan 2026-10-23-incidence-partition, Amendment 7). For the leaf's four
//! control values `q`:
//!
//! * the value is `a₀ + t(a₁ + t(a₂ + t a₃))` with `a = B_ℓᵀ q`;
//! * the input derivative is `β₁ + t(β₂ + t β₃)` with `β = s_ℓ (a₁, 2a₂, 3a₃)` and `s_ℓ = dt/dx =
//!   span · 2^d`;
//! * the control gradient is `B_ℓ m` with the moments `m = Σ_e g_e τ(t_e)` of the leaf's elements.
//!
//! `a` and `β` are computed in `f64` and rounded to `f32` once per bank, channel and leaf per call;
//! the moments accumulate per leaf and map through `B_ℓ` once per block. Nothing per element depends
//! on the knot count or the knot placement. Plain `*` and `+` throughout, no fused multiply-add, so
//! every target gives the same bits.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::borrow::Cow;

#[cfg(doc)]
use crate::closed_form::ChebyshevClosedForm;
use crate::closed_form::ClosedForm;
use crate::knots::{Knots, Tangents};

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
    block_rows(n) * channels
}

/// Rows per block of the kernels' fixed partition of `n` rows.
#[inline]
fn block_rows(n: usize) -> usize {
    n.div_ceil(ROW_BLOCKS).max(MIN_ROWS_PER_BLOCK)
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

/// The first pass's values for a run of up to [`RUN`] elements: each element's leaf and its coordinate
/// `t` on it. The first pass locates the whole run at once and vectorises; the second reads them
/// beside the per-channel tables, which it indexes by channel and leaf and which do not vectorise.
struct ElementPass {
    segment: [u32; RUN],
    t: [f32; RUN],
}

impl ElementPass {
    fn new() -> Self {
        Self {
            segment: [0; RUN],
            t: [0.0; RUN],
        }
    }
}

/// The leaf of a clamped input and its coordinate on it: uniform knots by the uniform segment
/// arithmetic ([`segment`]), which vectorises, refined ones by [`Knots::locate`]. The one expression
/// the forward evaluates and the backward recomputes from the tape.
///
/// # Preconditions
/// As [`segment`] and [`Knots::locate`]: `span == 0.5 · (grid − 1)`, `last == (grid − 2) as f32`;
/// `UNIFORM` exactly when `knots.is_uniform()`.
#[inline(always)]
fn leaf_of<const UNIFORM: bool>(knots: &Knots, xc: f32, (span, last): (f32, f32)) -> (u32, f32) {
    if UNIFORM {
        let (i, t) = segment(xc, span, last);
        (i as u32, t)
    } else {
        let (leaf, t) = knots.locate(xc, span);
        (leaf as u32, t)
    }
}

/// Each leaf of each channel of one bank as the polynomial an element on it evaluates (Nagare plan
/// 2026-10-23-incidence-partition, Amendment 7), computed once per call: the forward's power
/// coefficients ([`Polys`]) or the pullback's slope coefficients ([`Slopes`]).
///
/// # Invariants
/// `table[ch · leaves + ℓ]` is channel `ch`'s entry on leaf `ℓ`, `leaves == knots.controls().len()`; `bases` the leaf
/// bases the entries were computed through, which the pullback's fold reads again.
#[derive(Debug, Clone)]
struct Segments<C> {
    table: Vec<C>,
    leaves: usize,
    bases: LeafBases,
}

/// The basis each leaf of each channel maps its four control values through, `a_j = Σ_k B[k][j] q_k`, laid out
/// `bases[ch · stride + ℓ]`: under [`Tangents::Chord`] the knots' own per leaf, shared by every channel (`stride` 0);
/// under [`Tangents::Bounded`] one per channel and leaf, the Hermite basis of the tangents its knots select (Nagare
/// plan 2026-10-23-c1-lattice-activations, Amendment 1). One indexing for both, no branch at a read.
#[derive(Debug, Clone)]
struct LeafBases {
    bases: Vec<[[f64; 4]; 4]>,
    stride: usize,
}

impl LeafBases {
    /// The bases of `coef`, `channels × knots.len()`, under the knots' tangent rule: one choice per call.
    fn of(coef: &[f32], channels: usize, knots: &Knots) -> Self {
        match knots.tangents() {
            Tangents::Chord => Self {
                bases: knots.basis().to_vec(),
                stride: 0,
            },
            Tangents::Bounded => Self {
                bases: coef
                    .chunks_exact(knots.len())
                    .take(channels)
                    .flat_map(|channel| bounded_bases(channel, knots))
                    .collect(),
                stride: knots.controls().len(),
            },
        }
    }

    /// Channel `ch`'s basis on leaf `leaf`.
    #[inline(always)]
    fn get(&self, (ch, leaf): (usize, usize)) -> &[[f64; 4]; 4] {
        &self.bases[ch * self.stride + leaf]
    }
}

/// `1` where `condition` holds, `0` where it does not: a comparison as a weight, not a branch.
#[inline(always)]
fn indicator(condition: bool) -> f64 {
    f64::from(u8::from(condition))
}

/// Each knot's selection under [`Tangents::Bounded`], as weights on its three candidates (chord, three times the
/// left secant, three times the right one), with no data-dependent branch:
/// - the class of the knot in the signed knot graph is the weight `agree = [s_left · s_right > 0]`, zero at a knot
///   whose secants differ in sign or where one is zero;
/// - the route is a one-hot over the candidates, the least magnitude winning and the earlier on a tie.
///
/// An end knot reads its one edge on both sides, so its chord (half that secant) always wins.
fn selections(channel: &[f32], widths: &[f64]) -> Vec<[f64; 3]> {
    let n = channel.len();
    let q = |k: usize| f64::from(channel[k]);
    let secant: Vec<f64> = (0..n - 1).map(|l| (q(l + 1) - q(l)) / widths[l]).collect();
    (0..n)
        .map(|k| {
            let (left, right) = (k.max(1) - 1, k.min(n - 2));
            let (s_left, s_right) = (secant[left], secant[right]);
            let chord =
                (q((k + 1).min(n - 1)) - q(k.saturating_sub(1))) / (widths[left] + widths[right]);
            let (c, l, r) = (chord.abs(), 3.0 * s_left.abs(), 3.0 * s_right.abs());
            let agree = indicator(s_left * s_right > 0.0);
            let take_chord = indicator(c <= l && c <= r);
            let take_left = (1.0 - take_chord) * indicator(l <= r);
            let take_right = (1.0 - take_chord) * (1.0 - indicator(l <= r));
            [agree * take_chord, agree * take_left, agree * take_right]
        })
        .collect()
}

/// One channel's basis per leaf under [`Tangents::Bounded`]: the cubic Hermite in powers,
/// `a = P (q_ℓ, q_{ℓ+1}, d_ℓ, d_{ℓ+1})`, whose tangents `d` (in the leaf's coordinate) are the selection-weighted sums
/// of the candidates' linear functionals over the leaf's window `(q_{ℓ−1}, q_ℓ, q_{ℓ+1}, q_{ℓ+2})` (ends replicated,
/// as the controls). One formula for every leaf: where both knots select the chord it is the knots' own basis,
/// entry by entry.
fn bounded_bases(channel: &[f32], knots: &Knots) -> Vec<[[f64; 4]; 4]> {
    let widths = knots.widths();
    let pick = selections(channel, widths);
    let last = widths.len() - 1;
    (0..widths.len())
        .map(|leaf| {
            let w = widths[leaf];
            let (before, after) = (widths[leaf.saturating_sub(1)], widths[(leaf + 1).min(last)]);
            let (r0, r1) = (w / (before + w), w / (w + after));
            let ([c0, l0, g0], [c1, l1, g1]) = (pick[leaf], pick[leaf + 1]);
            // The tangents over the window: chord, left and right candidates, each weighted by its selection.
            let d0 = [
                c0 * -r0 + l0 * (-3.0 * w / before),
                l0 * (3.0 * w / before) + g0 * -3.0,
                c0 * r0 + g0 * 3.0,
                0.0,
            ];
            let d1 = [
                0.0,
                c1 * -r1 + l1 * -3.0,
                l1 * 3.0 + g1 * (-3.0 * w / after),
                c1 * r1 + g1 * (3.0 * w / after),
            ];
            let (q0, q1) = ([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]);
            // a0 = q0, a1 = d0, a2 = −3 q0 + 3 q1 − 2 d0 − d1, a3 = 2 q0 − 2 q1 + d0 + d1.
            std::array::from_fn(|k| {
                [
                    q0[k],
                    d0[k],
                    -3.0 * q0[k] + 3.0 * q1[k] - 2.0 * d0[k] - d1[k],
                    2.0 * q0[k] - 2.0 * q1[k] + d0[k] + d1[k],
                ]
            })
        })
        .collect()
}

/// Per channel and leaf the power coefficients `a = B_ℓᵀ q` of the value: `y = a₀ + t(a₁ + t(a₂ + t a₃))`.
type Polys = Segments<[f32; 4]>;

/// Per channel and leaf the coefficients `β = s_ℓ (a₁, 2a₂, 3a₃)` of the input derivative, `s_ℓ = dt/dx`:
/// `dy/dx = β₁ + t(β₂ + t β₃)`.
type Slopes = Segments<[f32; 3]>;

impl<C> Segments<C> {
    /// The table of `coef`, `channels × knots.len()`: per channel and leaf, `entry` of the leaf's index
    /// and its power coefficients `a_j = Σ_k B_ℓ[k][j] q_k`, summed in `f64` in weight order.
    ///
    /// # Preconditions
    /// `coef.len() == channels · knots.len()`.
    fn of(
        coef: &[f32],
        channels: usize,
        knots: &Knots,
        entry: impl Fn(usize, [f64; 4]) -> C,
    ) -> Self {
        let bases = LeafBases::of(coef, channels, knots);
        let table = coef
            .chunks_exact(knots.len())
            .take(channels)
            .enumerate()
            .flat_map(|(ch, channel)| {
                let (bases, entry) = (&bases, &entry);
                knots
                    .controls()
                    .iter()
                    .enumerate()
                    .map(move |(leaf, controls)| {
                        let b = bases.get((ch, leaf));
                        let q = controls.map(|j| f64::from(channel[j]));
                        let a = std::array::from_fn(|j| {
                            b[0][j] * q[0] + b[1][j] * q[1] + b[2][j] * q[2] + b[3][j] * q[3]
                        });
                        entry(leaf, a)
                    })
            })
            .collect();
        Self {
            table,
            leaves: knots.controls().len(),
            bases,
        }
    }
}

impl Polys {
    /// The forward's table: each leaf's `a`, rounded to `f32`.
    fn values(coef: &[f32], channels: usize, knots: &Knots) -> Self {
        Self::of(coef, channels, knots, |_, a| a.map(|v| v as f32))
    }

    /// Channel `ch`'s power coefficients on leaf `leaf`.
    #[inline(always)]
    fn at(&self, ch: usize, leaf: usize) -> [f32; 4] {
        self.table[ch * self.leaves + leaf]
    }
}

impl Slopes {
    /// The pullback's table: each leaf's `β`, the scale `s_ℓ = span · 2^d` folded in, in `f64`, then
    /// rounded to `f32`.
    fn slopes(coef: &[f32], channels: usize, knots: &Knots) -> Self {
        let span = 0.5 * (knots.grid() - 1) as f64;
        Self::of(coef, channels, knots, |leaf, a| {
            let s = span * f64::from(knots.scales()[leaf]);
            [s * a[1], s * (2.0 * a[2]), s * (3.0 * a[3])].map(|v| v as f32)
        })
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

/// The marker a route holds for an element no bank is read at (Nagare plan 2026-10-23-incidence-partition,
/// Amendment 5): the routed forward writes it `0`, and the routed pullback adds nothing for it.
pub const SKIP: u8 = u8::MAX;

/// The first pass's values for one run of a forward, as its second pass reads them.
struct EmitRun<'p> {
    /// The run's first element in the block.
    start: usize,
    /// The channel of the run's first element.
    first: usize,
    channels: usize,
    seg: &'p [u32],
    t: &'p [f32],
}

/// How a forward's second pass writes a run: every bank into its own output ([`AllBanks`]), or each element's
/// routed bank alone into one ([`RoutedBank`]). Monomorphized into the row kernel, so neither carries a choice
/// inside its element loop.
trait ForwardEmit {
    /// Writes the run from its leaves and coordinates; returns the channel after the run.
    fn emit(&mut self, banks: &[Polys], run: &EmitRun<'_>) -> usize;
}

/// One element's value on one bank: its channel's cubic on its leaf, by Horner at its coordinate.
#[inline(always)]
fn element_value(polys: &Polys, ch: usize, run: &EmitRun<'_>, e: usize) -> f32 {
    let [a0, a1, a2, a3] = polys.at(ch, run.seg[e] as usize);
    let t = run.t[e];
    a0 + t * (a1 + t * (a2 + t * a3))
}

/// Every bank's output slice over the block.
struct AllBanks<'a, 'b> {
    outs: &'b mut [&'a mut [f32]],
}

impl ForwardEmit for AllBanks<'_, '_> {
    #[inline(always)]
    fn emit(&mut self, banks: &[Polys], run: &EmitRun<'_>) -> usize {
        let n = run.seg.len();
        let mut ch = run.first;
        for (polys, out) in banks.iter().zip(self.outs.iter_mut()) {
            let out = &mut out[run.start..run.start + n];
            ch = run.first;
            for (e, slot) in out.iter_mut().enumerate() {
                *slot = element_value(polys, ch, run, e);
                ch += 1;
                if ch == run.channels {
                    ch = 0;
                }
            }
        }
        ch
    }
}

/// Each element's routed bank over the block, into one output: an element routed [`SKIP`] is written `0`.
struct RoutedBank<'a> {
    route: &'a [u8],
    out: &'a mut [f32],
}

impl ForwardEmit for RoutedBank<'_> {
    #[inline(always)]
    fn emit(&mut self, banks: &[Polys], run: &EmitRun<'_>) -> usize {
        let n = run.seg.len();
        let route = &self.route[run.start..run.start + n];
        let out = &mut self.out[run.start..run.start + n];
        let mut ch = run.first;
        for (e, (slot, &bank)) in out.iter_mut().zip(route).enumerate() {
            *slot = banks
                .get(usize::from(bank))
                .map_or(0.0, |polys| element_value(polys, ch, run, e));
            ch += 1;
            if ch == run.channels {
                ch = 0;
            }
        }
        ch
    }
}

/// The first bank on every element into its own output, and each element's routed bank among the others into a second
/// one (Nagare plan 2026-10-23-incidence-partition, Amendment 6, H1: an HSiKAN cell's gate beside its routed inner
/// splines). A route indexes the banks after the first; an element routed [`SKIP`] is written `0` in the second output.
struct EveryAndRouted<'a> {
    every: &'a mut [f32],
    route: &'a [u8],
    out: &'a mut [f32],
}

impl ForwardEmit for EveryAndRouted<'_> {
    #[inline(always)]
    fn emit(&mut self, banks: &[Polys], run: &EmitRun<'_>) -> usize {
        let n = run.seg.len();
        let span = run.start..run.start + n;
        let (every, route, out) = (
            &mut self.every[span.clone()],
            &self.route[span.clone()],
            &mut self.out[span],
        );
        let (first, routed) = banks.split_at(1);
        let mut ch = run.first;
        for (e, ((slot, every_slot), &bank)) in
            out.iter_mut().zip(every.iter_mut()).zip(route).enumerate()
        {
            *every_slot = element_value(&first[0], ch, run, e);
            *slot = routed
                .get(usize::from(bank))
                .map_or(0.0, |polys| element_value(polys, ch, run, e));
            ch += 1;
            if ch == run.channels {
                ch = 0;
            }
        }
        ch
    }
}

/// The per-element buffers a forward fills for a block of rows: its outputs, as `emit` writes them, and the one tape.
///
/// Bundled into a struct so the shared row kernel takes four arguments instead of
/// seven (§6.5 #6) and so the sequential and parallel paths call *one*
/// implementation rather than each carrying a copy of the arithmetic (§6.1).
struct ForwardRow<'a, E> {
    emit: E,
    x_clamped: &'a mut [f32],
    interior: &'a mut [bool],
}

/// Evaluate every bank's CR spline for every element of a whole block of rows.
///
/// One pass over runs of elements in row-major order, the channel a counter that wraps (not
/// `element % channels`, which measured ~24 % slower): no per-row iterator setup, which at one
/// channel cost as much as the arithmetic. Each run is two passes: the clamp, the leaf and the
/// coordinate of every element at once, then, bank by bank, each element's cubic by Horner from the
/// bank's power coefficients by channel and leaf. The first pass is the banks' one (tensor plan
/// 2026-10-22-cr-banks): banks over one input share their leaves and coordinates, and write one
/// tape.
///
/// # Preconditions
/// Every slice in `w` has the same length as `x_block`, which is a whole multiple
/// of `channels`; one output per bank in `banks`; each bank's table over `channels × knots.len()`
/// coefficients.
#[inline(always)]
fn forward_block<const UNIFORM: bool, E: ForwardEmit>(
    banks: &[Polys],
    knots: &Knots,
    channels: usize,
    x_block: &[f32],
    mut w: ForwardRow<'_, E>,
) {
    debug_assert_eq!(x_block.len() % channels, 0, "block must be whole rows");
    let (span, last) = (0.5 * (knots.grid() - 1) as f32, (knots.grid() - 2) as f32);
    let mut pass = ElementPass::new();
    let mut ch = 0;
    for start in (0..x_block.len()).step_by(RUN) {
        let end = (start + RUN).min(x_block.len());
        let n = end - start;
        let x = &x_block[start..end];
        let ElementPass { segment: seg, t } = &mut pass;
        let (seg, t) = (&mut seg[..n], &mut t[..n]);
        let (x_clamped, interior) = (&mut w.x_clamped[start..end], &mut w.interior[start..end]);
        // First pass, every element of the run at once: the clamp, the leaf and the coordinate.
        for e in 0..n {
            let raw = x[e];
            let xc = raw.clamp(-1.0, 1.0);
            (seg[e], t[e]) = leaf_of::<UNIFORM>(knots, xc, (span, last));
            x_clamped[e] = xc;
            // Closed interval: `±1` is interior, matching the guard this replaces.
            interior[e] = (-1.0..=1.0).contains(&raw);
        }
        // Second pass: each element's cubic, by its channel and leaf, on every bank or on its routed bank alone.
        let run = EmitRun {
            start,
            first: ch,
            channels,
            seg: &*seg,
            t: &*t,
        };
        ch = w.emit.emit(banks, &run);
    }
}

/// [`forward_block`] instantiated for the knots: uniform knots take the uniform segment arithmetic,
/// which vectorises; refined ones locate their leaves. One choice per call, never per element.
#[inline(always)]
fn forward_block_over<E: ForwardEmit>(
    banks: &[Polys],
    knots: &Knots,
    channels: usize,
    x_block: &[f32],
    w: ForwardRow<'_, E>,
) {
    if knots.is_uniform() {
        forward_block::<true, _>(banks, knots, channels, x_block, w);
    } else {
        forward_block::<false, _>(banks, knots, channels, x_block, w);
    }
}

/// Per channel and leaf the power coefficients of the value in the leaf's coordinate `t ∈ [0, 1]`,
/// `y = a₀ + a₁ t + a₂ t² + a₃ t³`, under the knots' tangent rule: the forward's own table `a = B_ℓᵀ q`, summed in
/// `f64`, before its rounding to `f32`. Laid out `[ch · leaves + ℓ]`, `leaves` the leaves of `knots`. What a shape
/// certificate reads (Nagare plan 2026-10-24-elliptic-cr, E3): `t` is a positive affine map of the input on each
/// leaf, `dt/dx = span · 2^d`.
///
/// # Panics
/// When `coef.len() != channels · knots.len()`.
#[must_use]
pub fn catmull_rom_leaf_coefficients(
    coef: &[f32],
    channels: usize,
    knots: &Knots,
) -> Vec<[f64; 4]> {
    assert_eq!(
        coef.len(),
        channels * knots.len(),
        "one control point per knot and channel"
    );
    Segments::of(coef, channels, knots, |_, a| a).table
}

/// The coefficient tables of one or more banks of control points on `knots`, built once per parameter version and read
/// by every forward and pullback under it (Nagare plans 2026-10-24-elliptic-cr, O2, and
/// 2026-10-26-cell-spline-tables-per-version): per bank the forward's power coefficients, the pullback's slope
/// coefficients and the leaf bases they were formed through -- under bounded tangents a function of the control points
/// themselves. Where the control points are Chebyshev terms' image ([`Self::chebyshev`]), it keeps the knot basis and
/// the term count, through which a pullback projects the control-point gradient onto the terms. Immutable: other
/// control points are another `PreparedSpline`.
///
/// # Invariants
/// At least one bank; each bank's tables are [`Segments::of`] of its control points, channels and knots, exactly as a
/// per-call entry forms them.
#[derive(Debug, Clone)]
pub struct PreparedSpline {
    polys: Vec<Polys>,
    slopes: Vec<Slopes>,
    channels: usize,
    knots: Knots,
    terms: Option<ChebyshevTerms>,
}

/// The Chebyshev side of a [`PreparedSpline`] whose control points the terms generated: the knot basis, flat
/// `(grid, k)`, `k`, and where the knots and `k` qualify, the closed form of the same banks (Nagare plan
/// 2026-10-30-t2-algebraic-lowering; [`crate::closed_form`]).
#[derive(Debug, Clone)]
struct ChebyshevTerms {
    basis: Vec<f32>,
    k: usize,
    closed: Option<ClosedForm>,
}

impl PreparedSpline {
    /// One bank's tables: `coef`, `channels × knots.len()`, on `knots`.
    ///
    /// # Panics
    /// When `coef.len() != channels · knots.len()`.
    #[must_use]
    pub fn new(coef: &[f32], channels: usize, knots: &Knots) -> Self {
        Self::banks(&[coef], channels, knots)
    }

    /// Every bank's tables, in `coefs` order, each bank `channels × knots.len()` control points on `knots`.
    ///
    /// # Panics
    /// With no bank, or when a bank is not `channels · knots.len()` long.
    #[must_use]
    pub fn banks(coefs: &[&[f32]], channels: usize, knots: &Knots) -> Self {
        TableSource::Controls(coefs).check(channels, knots);
        Self {
            polys: coefs
                .iter()
                .map(|coef| Polys::values(coef, channels, knots))
                .collect(),
            slopes: coefs
                .iter()
                .map(|coef| Slopes::slopes(coef, channels, knots))
                .collect(),
            channels,
            knots: knots.clone(),
            terms: None,
        }
    }

    /// The tables of the control points each bank of Chebyshev `terms` (`channels × k`) generates on `grid`'s uniform
    /// knots under its tangent rule -- the same control points and knots [`chebyshev_cr_banks_forward_into`] forms --
    /// with the knot basis kept for the pullback's projection. `(grid, k)` is the chord rule.
    ///
    /// # Panics
    /// As [`Self::banks`], and when a bank is not `channels · k` terms.
    #[must_use]
    pub fn chebyshev(terms: &[&[f32]], channels: usize, grid: impl Into<ChebyshevGrid>) -> Self {
        let grid = grid.into();
        let k = grid.k;
        let (controls, basis) = chebyshev_controls(terms, channels, (grid.grid, k));
        let controls: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
        let knots = grid.knots();
        let closed = ClosedForm::of(terms, channels, (&knots, k));
        Self {
            terms: Some(ChebyshevTerms { basis, k, closed }),
            ..Self::banks(&controls, channels, &knots)
        }
    }

    /// The closed form of the banks (Nagare plan 2026-10-30-t2-algebraic-lowering): where the tables were built from
    /// Chebyshev terms ([`Self::chebyshev`]) and the knots and term count qualify ([`ChebyshevClosedForm::of`]), the
    /// same real functions as the tables, evaluated without them. `None` otherwise: the tables are the only execution.
    #[must_use]
    pub fn closed_form(&self) -> Option<&ClosedForm> {
        self.terms.as_ref()?.closed.as_ref()
    }

    /// The channels the tables cover.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// The knots the tables were built on.
    #[must_use]
    pub const fn knots(&self) -> &Knots {
        &self.knots
    }

    /// The banks the tables cover.
    #[must_use]
    pub fn bank_count(&self) -> usize {
        self.polys.len()
    }

    /// Where the control points are Chebyshev terms' image ([`Self::chebyshev`]): the knot basis, flat `(grid, k)`,
    /// and `k`, as the Chebyshev pullbacks read them.
    #[must_use]
    pub fn chebyshev_terms(&self) -> Option<(&[f32], usize)> {
        self.terms.as_ref().map(|t| (t.basis.as_slice(), t.k))
    }

    /// The `f32` values the tables hold: what keeping them for a parameter version costs.
    #[must_use]
    pub fn retained_floats(&self) -> usize {
        let polys: usize = self.polys.iter().map(|p| 4 * p.table.len()).sum();
        let slopes: usize = self.slopes.iter().map(|s| 3 * s.table.len()).sum();
        let terms = self.terms.as_ref().map_or(0, |t| {
            t.basis.len() + t.closed.as_ref().map_or(0, |c| c.planes().len())
        });
        polys + slopes + terms
    }
}

/// Where an input sits on a [`PreparedSpline`]'s knots (Nagare plan 2026-10-26-fused-cells): its leaf and its coordinate
/// on it, from the clamped input, and whether the input lies in the closed clamp interval -- the values a forward
/// evaluates at and a pullback reads again, by the kernels' own expressions.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Located {
    leaf: u32,
    t: f32,
    interior: bool,
}

/// One block's partial moments of a pullback over a [`PreparedSpline`]'s banks (Nagare plan 2026-10-26-fused-cells):
/// per bank, channel and leaf `ḡ (1, t, t², t³)` in `f32`, as the kernels' blocks accumulate them. A traversal keeps one
/// per block and hands them, in block order, to [`PreparedSpline::gradients`].
#[derive(Debug, Clone)]
pub struct Moments {
    acc: Vec<[f32; 4]>,
    per_bank: usize,
}

impl Moments {
    /// Bank `bank`'s accumulators, `[channel · leaves + leaf]`, each `ḡ (1, t, t², t³)` summed: where a traversal adds
    /// its elements' moments itself (Nagare plan 2026-10-26-fused-cells, Track C).
    pub fn bank_mut(&mut self, bank: usize) -> &mut [[f32; 4]] {
        &mut self.acc[bank * self.per_bank..(bank + 1) * self.per_bank]
    }

    /// Every bank's accumulators at once, in bank order: disjoint slices, as [`Self::bank_mut`] gives them one by one.
    pub fn banks_mut(&mut self) -> std::slice::ChunksExactMut<'_, [f32; 4]> {
        self.acc.chunks_exact_mut(self.per_bank.max(1))
    }
}

impl PreparedSpline {
    /// Where `x` sits on the knots: the clamp to `[-1, 1]`, the leaf and the coordinate of the clamped value, and the
    /// interval test of the raw one -- the forward's first pass for one element.
    #[inline(always)]
    fn locate(&self, x: f32) -> Located {
        let (span, last) = (
            0.5 * (self.knots.grid() - 1) as f32,
            (self.knots.grid() - 2) as f32,
        );
        let xc = x.clamp(-1.0, 1.0);
        let (leaf, t) = if self.knots.is_uniform() {
            leaf_of::<true>(&self.knots, xc, (span, last))
        } else {
            leaf_of::<false>(&self.knots, xc, (span, last))
        };
        Located {
            leaf,
            t,
            interior: (-1.0..=1.0).contains(&x),
        }
    }

    /// Every element of `x` located at once, into `leaf`, `t` and `interior` (each `x.len()` long): the forward's first
    /// pass over a row, which vectorises on uniform knots (Nagare plan 2026-10-26-fused-cells, Track C).
    ///
    /// # Panics
    /// When the outputs are shorter than `x`.
    #[inline(always)]
    pub fn locate_row(
        &self,
        x: &[f32],
        (leaf, t, interior): (&mut [u32], &mut [f32], &mut [bool]),
    ) {
        let (span, last) = (
            0.5 * (self.knots.grid() - 1) as f32,
            (self.knots.grid() - 2) as f32,
        );
        let (leaf, t, interior) = (
            &mut leaf[..x.len()],
            &mut t[..x.len()],
            &mut interior[..x.len()],
        );
        if self.knots.is_uniform() {
            for (((&raw, l), tt), inside) in x
                .iter()
                .zip(leaf.iter_mut())
                .zip(t.iter_mut())
                .zip(interior.iter_mut())
            {
                let (i, u) = segment(raw.clamp(-1.0, 1.0), span, last);
                *l = i as u32;
                *tt = u;
                *inside = (-1.0..=1.0).contains(&raw);
            }
        } else {
            for (((&raw, l), tt), inside) in x
                .iter()
                .zip(leaf.iter_mut())
                .zip(t.iter_mut())
                .zip(interior.iter_mut())
            {
                let at = self.locate(raw);
                (*l, *tt, *inside) = (at.leaf, at.t, at.interior);
            }
        }
    }

    /// Leaves per channel: the tables' stride.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.knots.controls().len()
    }

    /// Bank `bank`'s forward table, `[channel · leaves + leaf]`, each leaf's power coefficients `a`:
    /// `y = a₀ + t(a₁ + t(a₂ + t a₃))` (Nagare plan 2026-10-26-fused-cells, Track C).
    #[must_use]
    pub fn power_table(&self, bank: usize) -> &[[f32; 4]] {
        &self.polys[bank].table
    }

    /// Bank `bank`'s pullback table, `[channel · leaves + leaf]`, each leaf's slope coefficients `β`, the scale folded
    /// in: `dy/dx = β₁ + t(β₂ + t β₃)` inside the clamp interval.
    #[must_use]
    pub fn slope_table(&self, bank: usize) -> &[[f32; 3]] {
        &self.slopes[bank].table
    }

    /// Zeroed moments for every bank.
    #[must_use]
    pub fn moments(&self) -> Moments {
        let segs = self.knots.controls().len();
        let per_bank = self.channels * segs;
        Moments {
            acc: vec![[0.0; 4]; self.polys.len() * per_bank],
            per_bank,
        }
    }

    /// Each bank's parameter gradient from the blocks' moments, added in `f64` in the order given and folded through the
    /// leaf bases ([`fold_partials`], the kernels' reduction); for Chebyshev tables projected onto the terms.
    ///
    /// # Panics
    /// When a block's moments belong to other tables.
    #[must_use]
    pub fn gradients(&self, blocks: &[Moments]) -> Vec<Vec<f32>> {
        let len = self.polys.len() * self.channels * self.knots.controls().len();
        assert!(
            blocks.iter().all(|m| m.acc.len() == len),
            "moments of these tables"
        );
        let control = fold_partials(
            &self.slopes,
            blocks.iter().map(|m| m.acc.as_slice()),
            (self.channels, &self.knots),
        );
        match &self.terms {
            None => control,
            Some(terms) => control
                .iter()
                .map(|g| {
                    chebyshev_terms_gradient(
                        g,
                        &terms.basis,
                        (self.channels, self.knots.len(), terms.k),
                    )
                })
                .collect(),
        }
    }
}

/// Where a call's coefficient tables come from (Nagare plan 2026-10-26-cell-spline-tables-per-version): built in the
/// call from each bank's control points, or borrowed from a [`PreparedSpline`] built once for them. The two give the
/// same tables, bit for bit, and the kernels read them the same way; a prepared source builds nothing in the call.
#[derive(Debug, Clone, Copy)]
pub enum TableSource<'a> {
    /// Each bank's control points, flat `(channels, knots.len())`: the call builds their tables.
    Controls(&'a [&'a [f32]]),
    /// The tables prepared once for the control points' version.
    Prepared(&'a PreparedSpline),
}

impl<'a> From<&'a [&'a [f32]]> for TableSource<'a> {
    fn from(coefs: &'a [&'a [f32]]) -> Self {
        Self::Controls(coefs)
    }
}

impl<'a, const N: usize> From<&'a [&'a [f32]; N]> for TableSource<'a> {
    fn from(coefs: &'a [&'a [f32]; N]) -> Self {
        Self::Controls(coefs)
    }
}

impl<'a> From<&'a Vec<&'a [f32]>> for TableSource<'a> {
    fn from(coefs: &'a Vec<&'a [f32]>) -> Self {
        Self::Controls(coefs)
    }
}

impl<'a> From<&'a PreparedSpline> for TableSource<'a> {
    fn from(prepared: &'a PreparedSpline) -> Self {
        Self::Prepared(prepared)
    }
}

impl<'a> TableSource<'a> {
    /// The banks the source holds.
    fn bank_count(self) -> usize {
        match self {
            Self::Controls(coefs) => coefs.len(),
            Self::Prepared(prepared) => prepared.bank_count(),
        }
    }

    /// The source fits a call over `channels` on `knots`.
    ///
    /// # Panics
    /// With no bank; when a bank's control points are not `channels · knots.len()`; when prepared tables were built
    /// for other channels or on other knots.
    fn check(self, channels: usize, knots: &Knots) {
        match self {
            Self::Controls(coefs) => {
                assert!(!coefs.is_empty(), "at least one bank");
                for coef in coefs {
                    assert_eq!(
                        coef.len(),
                        channels * knots.len(),
                        "one control point per knot and channel"
                    );
                }
            }
            Self::Prepared(prepared) => {
                assert_eq!(prepared.channels, channels, "the tables' channels");
                assert_eq!(&prepared.knots, knots, "the tables' knots");
            }
        }
    }

    /// The forward's tables, one per bank, for a call over `channels` on `knots`: built, or borrowed.
    ///
    /// # Panics
    /// As [`Self::check`].
    fn polys(self, channels: usize, knots: &Knots) -> Cow<'a, [Polys]> {
        self.check(channels, knots);
        match self {
            Self::Controls(coefs) => Cow::Owned(
                coefs
                    .iter()
                    .map(|coef| Polys::values(coef, channels, knots))
                    .collect(),
            ),
            Self::Prepared(prepared) => Cow::Borrowed(&prepared.polys),
        }
    }

    /// The pullback's tables, one per bank, for a call over `channels` on `knots`: built, or borrowed.
    ///
    /// # Panics
    /// As [`Self::check`].
    fn slopes(self, channels: usize, knots: &Knots) -> Cow<'a, [Slopes]> {
        self.check(channels, knots);
        match self {
            Self::Controls(coefs) => Cow::Owned(
                coefs
                    .iter()
                    .map(|coef| Slopes::slopes(coef, channels, knots))
                    .collect(),
            ),
            Self::Prepared(prepared) => Cow::Borrowed(&prepared.slopes),
        }
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
/// is [`catmull_rom_forward`] of `coefs[b]`, and the clamp, the leaf and the coordinate of every element
/// are computed once for all of them, into one tape.
///
/// The tables come from `coefs` ([`TableSource`]): built in the call from each bank's control points, or borrowed from
/// a [`PreparedSpline`].
///
/// # Preconditions
/// At least one bank; every control-point bank `channels * knots.len()` long, or prepared tables for `channels` on
/// `knots`; one output per bank, each `n * channels`; `x.len() == n * channels`.
///
/// # Postconditions
/// Each `into.outs[b]` is bit for bit [`catmull_rom_forward`]'s output for bank `b`'s control points, and
/// `into.cache` its cache (the same for every bank).
pub fn catmull_rom_banks_forward_into<'a>(
    coefs: impl Into<TableSource<'a>>,
    x: &[f32],
    n: usize,
    channels: usize,
    knots: &Knots,
    into: CatmullRomBanksBuffers<'_>,
) {
    let CatmullRomBanksBuffers { outs, cache } = into;
    // Three is the smallest grid this operator **supports**: `i` ranges over `{0, 1}`, giving two
    // segments, and the outer stencil points are supplied by the endpoint replication
    // `cr_control_indices` already performs. The CogInfoCom HSiKAN configuration uses grid 3, which
    // is why the minimum moved from 4. Grid 2 is out of contract here — not asserted to be
    // mathematically impossible, simply not a configuration this operator is certified for.
    let tables = coefs.into();
    assert_eq!(x.len(), n * channels);
    assert_eq!(outs.len(), tables.bank_count(), "one output per bank");
    for out in &outs {
        assert_eq!(out.len(), x.len(), "out must be (n, channels)");
    }
    let banks = tables.polys(channels, knots);
    forward_tables(
        &banks,
        x,
        (n, channels),
        knots,
        CatmullRomBanksBuffers { outs, cache },
    );
}

/// The banks forward over tables already built or borrowed ([`TableSource`]): every element's clamp, leaf and coordinate
/// once, into the tape, and each bank's polynomial on it.
fn forward_tables(
    banks: &[Polys],
    x: &[f32],
    (n, channels): (usize, usize),
    knots: &Knots,
    into: CatmullRomBanksBuffers<'_>,
) {
    let CatmullRomBanksBuffers { mut outs, cache } = into;
    refill(cache, (n, channels), knots);
    let (x_clamped, interior) = (&mut cache.x_clamped, &mut cache.interior);

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
                    banks,
                    knots,
                    channels,
                    x_b,
                    ForwardRow {
                        emit: AllBanks { outs },
                        x_clamped: xc_b,
                        interior: in_b,
                    },
                );
            });
        return;
    }

    forward_block_over(
        banks,
        knots,
        channels,
        x,
        ForwardRow {
            emit: AllBanks { outs: &mut outs },
            x_clamped,
            interior,
        },
    );
}

/// The tape refilled for a call over `n` rows of `channels`: `resize` keeps the capacity, and the blocks overwrite
/// every element, so nothing a previous call left is read.
fn refill(cache: &mut CatmullRomCache, (n, channels): (usize, usize), knots: &Knots) {
    cache.x_clamped.resize(n * channels, 0.0);
    cache.interior.resize(n * channels, false);
    cache.n = n;
    cache.channels = channels;
    cache.grid = knots.len();
    cache.knots = Some(knots.clone());
}

/// The buffers a routed forward writes, owned by the caller: the routed output, the output of a bank read on every
/// element where there is one, and the tape (Nagare plan 2026-10-23-incidence-partition, Amendments 5 and 6).
#[derive(Debug)]
pub struct RoutedBuffers<'a> {
    /// Each element's routed bank's value, flat `(n, channels)`; every element is written.
    pub out: &'a mut [f32],
    /// Where `Some`, the first bank's value on every element, flat `(n, channels)`, and the routes index the banks
    /// after the first.
    pub every: Option<&'a mut [f32]>,
    /// Refilled in place for the call's shape; its buffers keep their capacity.
    pub cache: &'a mut CatmullRomCache,
}

/// The cotangents of a routed forward: the routed output's, and the every-element bank's where the forward had one.
#[derive(Clone, Copy, Debug)]
pub struct RoutedCotangents<'a> {
    /// The routed output's cotangent, flat `(n, channels)`.
    pub grad_y: &'a [f32],
    /// The first bank's cotangent on every element, where the forward read it there.
    pub every: Option<&'a [f32]>,
}

/// Several banks of control points over one input, each element evaluated on its routed bank alone (Nagare plan
/// 2026-10-23-incidence-partition, Amendment 5): element `e` is [`catmull_rom_forward`] of the routed bank at `e`, or
/// `0` where `route[e] == SKIP`. With `into.every`, the first bank is evaluated on every element into it as well, and
/// the routes index the banks after it (Amendment 6, H1). The clamp, the leaf and the coordinate are computed once,
/// into the one tape a banks forward writes.
///
/// The tables come from `coefs` ([`TableSource`]), as in [`catmull_rom_banks_forward_into`].
///
/// # Preconditions
/// As [`catmull_rom_banks_forward_into`]; `route.len() == x.len()`, every route a routed bank's index or [`SKIP`];
/// each output `n · channels`.
///
/// # Postconditions
/// Each written element is bit for bit its bank's single-bank output at it; `into.cache` the single-bank cache.
pub fn catmull_rom_routed_forward_into<'a>(
    coefs: impl Into<TableSource<'a>>,
    route: &[u8],
    x: &[f32],
    (n, channels): (usize, usize),
    knots: &Knots,
    into: RoutedBuffers<'_>,
) {
    let RoutedBuffers { out, every, cache } = into;
    let tables = coefs.into();
    let banks = tables.polys(channels, knots);
    let routed = banks.len() - usize::from(every.is_some());
    assert_eq!(x.len(), n * channels);
    assert_eq!(route.len(), x.len(), "one route per element");
    assert_eq!(out.len(), x.len(), "out must be (n, channels)");
    assert!(
        every.as_ref().is_none_or(|e| e.len() == x.len()),
        "every must be (n, channels)"
    );
    debug_assert!(route.iter().all(|&b| b == SKIP || usize::from(b) < routed));
    refill(cache, (n, channels), knots);
    if let Some(every) = every {
        return routed_forward_with_every(
            &banks,
            (route, out, every),
            x,
            (n, channels),
            (knots, cache),
        );
    }
    let (x_clamped, interior) = (&mut cache.x_clamped, &mut cache.interior);
    #[cfg(feature = "parallel")]
    if use_parallel(x.len()) {
        let stride = block_stride(n, channels);
        out.par_chunks_mut(stride)
            .zip(route.par_chunks(stride))
            .zip(x_clamped.par_chunks_mut(stride))
            .zip(interior.par_chunks_mut(stride))
            .zip(x.par_chunks(stride))
            .for_each(|((((out_b, route_b), xc_b), in_b), x_b)| {
                forward_block_over(
                    &banks,
                    knots,
                    channels,
                    x_b,
                    ForwardRow {
                        emit: RoutedBank {
                            route: route_b,
                            out: out_b,
                        },
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
            emit: RoutedBank { route, out },
            x_clamped,
            interior,
        },
    );
}

/// [`catmull_rom_routed_forward_into`] with the first bank on every element: the same blocks, the every-element output
/// split with the routed one.
fn routed_forward_with_every(
    banks: &[Polys],
    (route, out, every): (&[u8], &mut [f32], &mut [f32]),
    x: &[f32],
    (n, channels): (usize, usize),
    (knots, cache): (&Knots, &mut CatmullRomCache),
) {
    let (x_clamped, interior) = (&mut cache.x_clamped, &mut cache.interior);
    #[cfg(feature = "parallel")]
    if use_parallel(x.len()) {
        let stride = block_stride(n, channels);
        out.par_chunks_mut(stride)
            .zip(every.par_chunks_mut(stride))
            .zip(route.par_chunks(stride))
            .zip(x_clamped.par_chunks_mut(stride))
            .zip(interior.par_chunks_mut(stride))
            .zip(x.par_chunks(stride))
            .for_each(|(((((out_b, every_b), route_b), xc_b), in_b), x_b)| {
                forward_block_over(
                    banks,
                    knots,
                    channels,
                    x_b,
                    ForwardRow {
                        emit: EveryAndRouted {
                            every: every_b,
                            route: route_b,
                            out: out_b,
                        },
                        x_clamped: xc_b,
                        interior: in_b,
                    },
                );
            });
        return;
    }
    let _ = n;
    forward_block_over(
        banks,
        knots,
        channels,
        x,
        ForwardRow {
            emit: EveryAndRouted { every, route, out },
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
struct BackwardChunk<'a, P> {
    grad_x: &'a mut [f32],
    /// The cotangents, every bank's ([`AllCotangents`]) or the routed one's ([`RoutedCotangent`]), over the whole
    /// input; the chunk reads them from `offset`.
    pull: &'a P,
    /// The chunk's first element in the cotangents.
    offset: usize,
    /// Clamped inputs from the forward; the leaf and the coordinate are recomputed from them ([`leaf_of`]).
    x_clamped: &'a [f32],
    /// Per-element interior mask from the forward; see [`CatmullRomCache::interior`].
    interior: &'a [bool],
}

/// One element's pullback on one bank: its moments `gy · (1, t, t², t³)` into its channel's and leaf's four
/// accumulator slots, and its input-gradient term `gy · (β₁ + t(β₂ + t β₃))` returned -- the clamp's own Jacobian is
/// `1` inside the closed interval and `0` outside, read from the recorded `interior` mask (not from `x_clamped`, which
/// is inside by construction). The moments accumulate for every element, inside or not.
#[inline(always)]
fn element_pull(
    (slopes, moments): (&Slopes, &mut [f32; 4]),
    (t, at): (f32, usize),
    gy: f32,
    interior: bool,
) -> f32 {
    let g1 = gy * t;
    let g2 = g1 * t;
    moments[0] += gy;
    moments[1] += g1;
    moments[2] += g2;
    moments[3] += g2 * t;
    if !interior {
        return 0.0;
    }
    let [b1, b2, b3] = slopes.table[at];
    gy * (b1 + t * (b2 + t * b3))
}

/// One bank over one run of the backward: its table, its cotangent over the run, the run's interior mask
/// and first pass, and the bank's accumulators.
struct BankRun<'a> {
    slopes: &'a Slopes,
    grad_y: &'a [f32],
    interior: &'a [bool],
    pass: &'a ElementPass,
    accumulators: &'a mut [[f32; 4]],
}

impl BankRun<'_> {
    /// The bank's second pass over the run: each element's moments into its channel's and leaf's
    /// accumulator, and its input-gradient term written (`ASSIGN`) or added into `grad_x`. Returns the
    /// channel counter after the run.
    #[inline(always)]
    fn pull<const ASSIGN: bool>(
        self,
        grad_x: &mut [f32],
        (mut ch, channels, segs): (usize, usize, usize),
    ) -> usize {
        for (e, (gx, &gy)) in grad_x.iter_mut().zip(self.grad_y).enumerate() {
            // The element's channel and leaf: its accumulator and its slope coefficients.
            let at = ch * segs + self.pass.segment[e] as usize;
            let term = element_pull(
                (self.slopes, &mut self.accumulators[at]),
                (self.pass.t[e], at),
                gy,
                self.interior[e],
            );
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

/// The per-element context a pull reads for one run of a chunk: the run's elements in the whole input, its interior
/// mask and first pass, and the channel layout.
struct PullRun<'p> {
    at: std::ops::Range<usize>,
    interior: &'p [bool],
    pass: &'p ElementPass,
    first: usize,
    channels: usize,
    segs: usize,
}

/// How a backward's second pass reads its cotangents for a run: every bank's ([`AllCotangents`]), or each element's
/// routed bank's alone ([`RoutedCotangent`]). Monomorphized into the chunk kernel.
trait BackwardPull: Sync {
    /// Accumulates the run into the banks' accumulators (`per_bank` each, bank-major in `accumulators`) and writes its
    /// input gradient into `grad_x`; returns the channel after the run.
    fn pull(
        &self,
        banks: (&[Slopes], &mut [[f32; 4]], usize),
        grad_x: &mut [f32],
        run: &PullRun<'_>,
    ) -> usize;
}

/// Every bank's whole output cotangent: the banks backward.
struct AllCotangents<'a> {
    grad_ys: &'a [&'a [f32]],
}

impl BackwardPull for AllCotangents<'_> {
    #[inline(always)]
    fn pull(
        &self,
        (banks, accumulators, per_bank): (&[Slopes], &mut [[f32; 4]], usize),
        grad_x: &mut [f32],
        run: &PullRun<'_>,
    ) -> usize {
        let mut ch = run.first;
        for (b, (slopes, grad_y)) in banks.iter().zip(self.grad_ys).enumerate() {
            let bank = BankRun {
                slopes,
                grad_y: &grad_y[run.at.clone()],
                interior: run.interior,
                pass: run.pass,
                accumulators: &mut accumulators[b * per_bank..(b + 1) * per_bank],
            };
            // The first bank writes the input gradient, the others add to it: one bank is the
            // single-bank kernel exactly, with no choice inside the element loop.
            ch = if b == 0 {
                bank.pull::<true>(grad_x, (run.first, run.channels, run.segs))
            } else {
                bank.pull::<false>(grad_x, (run.first, run.channels, run.segs))
            };
        }
        ch
    }
}

/// The one output cotangent of a routed forward and its routes: each element's into its routed bank alone, an element
/// routed [`SKIP`] adding nothing and its input gradient `0` (Nagare plan 2026-10-23-incidence-partition,
/// Amendment 5).
struct RoutedCotangent<'a> {
    route: &'a [u8],
    grad_y: &'a [f32],
}

impl BackwardPull for RoutedCotangent<'_> {
    #[inline(always)]
    fn pull(
        &self,
        (banks, accumulators, per_bank): (&[Slopes], &mut [[f32; 4]], usize),
        grad_x: &mut [f32],
        run: &PullRun<'_>,
    ) -> usize {
        let (route, grad_y) = (&self.route[run.at.clone()], &self.grad_y[run.at.clone()]);
        let mut ch = run.first;
        for (e, ((gx, &gy), &bank)) in grad_x.iter_mut().zip(grad_y).zip(route).enumerate() {
            let b = usize::from(bank);
            *gx = match banks.get(b) {
                Some(slopes) => {
                    let at = ch * run.segs + run.pass.segment[e] as usize;
                    element_pull(
                        (slopes, &mut accumulators[b * per_bank + at]),
                        (run.pass.t[e], at),
                        gy,
                        run.interior[e],
                    )
                }
                None => 0.0,
            };
            ch += 1;
            if ch == run.channels {
                ch = 0;
            }
        }
        ch
    }
}

/// The first bank's cotangent on every element and the routed cotangent beside it (Amendment 6, H1): each element's
/// first-bank term, then its routed bank's, the input gradient their sum in that order; a route indexes the banks after
/// the first, and an element routed [`SKIP`] adds its first-bank term alone.
struct EveryAndRoutedCotangent<'a> {
    every: &'a [f32],
    route: &'a [u8],
    grad_y: &'a [f32],
}

impl BackwardPull for EveryAndRoutedCotangent<'_> {
    #[inline(always)]
    fn pull(
        &self,
        (banks, accumulators, per_bank): (&[Slopes], &mut [[f32; 4]], usize),
        grad_x: &mut [f32],
        run: &PullRun<'_>,
    ) -> usize {
        let at_run = run.at.clone();
        let (every, route, grad_y) = (
            &self.every[at_run.clone()],
            &self.route[at_run.clone()],
            &self.grad_y[at_run],
        );
        let mut ch = run.first;
        for (e, (((gx, &gy), &g_every), &bank)) in grad_x
            .iter_mut()
            .zip(grad_y)
            .zip(every)
            .zip(route)
            .enumerate()
        {
            let (at, t) = (ch * run.segs + run.pass.segment[e] as usize, run.pass.t[e]);
            let first = element_pull(
                (&banks[0], &mut accumulators[at]),
                (t, at),
                g_every,
                run.interior[e],
            );
            let b = 1 + usize::from(bank);
            *gx = match banks.get(b) {
                Some(slopes) => {
                    first
                        + element_pull(
                            (slopes, &mut accumulators[b * per_bank + at]),
                            (t, at),
                            gy,
                            run.interior[e],
                        )
                }
                None => first,
            };
            ch += 1;
            if ch == run.channels {
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
fn backward_chunk<const UNIFORM: bool, P: BackwardPull>(
    banks: &[Slopes],
    channels: usize,
    knots: &Knots,
    c: BackwardChunk<'_, P>,
    by_segment: &mut [[f32; 4]],
) {
    debug_assert_eq!(c.grad_x.len() % channels, 0, "chunk must be whole rows");
    let (span, last) = (0.5 * (knots.grid() - 1) as f32, (knots.grid() - 2) as f32);
    let segs = knots.controls().len();
    // The coefficient gradient by leaf (Nagare plan 2026-10-08-orthtree-hypergraph-convolution,
    // Amendments 3 and 6; 2026-10-23-incidence-partition, Amendment 7): one accumulator per bank,
    // channel and leaf, four moments wide, each element adding `gy · (1, t, t², t³)` into its own
    // leaf's slots. The block's accumulators are its partial; [`backward_over`] adds the blocks'
    // and maps the sums through each leaf's basis once ([`fold_moments`]), `∂L/∂q_k = Σ_e gy_e
    // w_k(t_e) = (B_ℓ m)_k`. A pass over every leaf per element made the pullback slower, not faster:
    // 7x at one channel, 1.9x at 32 (measured 2026-10-09).
    let per_bank = channels * segs;
    debug_assert_eq!(by_segment.len(), banks.len() * per_bank);
    // Runs of elements in row-major order, the channel a counter that wraps, as the forward: the
    // leaves and coordinates of a run at once, then, bank by bank, its moments and input gradients
    // by channel and leaf.
    let mut pass = ElementPass::new();
    let mut ch = 0;
    for start in (0..c.x_clamped.len()).step_by(RUN) {
        let end = (start + RUN).min(c.x_clamped.len());
        let n = end - start;
        let x_clamped = &c.x_clamped[start..end];
        let ElementPass { segment: seg, t } = &mut pass;
        let (seg, t) = (&mut seg[..n], &mut t[..n]);
        for e in 0..n {
            (seg[e], t[e]) = leaf_of::<UNIFORM>(knots, x_clamped[e], (span, last));
        }
        let run = PullRun {
            at: c.offset + start..c.offset + end,
            interior: &c.interior[start..end],
            pass: &pass,
            first: ch,
            channels,
            segs,
        };
        ch = c.pull.pull(
            (banks, &mut *by_segment, per_bank),
            &mut c.grad_x[start..end],
            &run,
        );
    }
}

/// One channel's moments, one entry per leaf, mapped through each leaf's basis into its four control
/// points and added there, in leaf then weight order: `grad[controls[ℓ][k]] += Σ_j B_ℓ[k][j] m_ℓ[j]`,
/// the dot product in `f64`, rounded to `f32` before it is added.
///
/// # Preconditions
/// `moments.len() == knots.controls().len()`, `grad.len() == knots.len()`.
fn fold_moments(
    moments: &[[f64; 4]],
    (knots, bases, ch): (&Knots, &LeafBases, usize),
    grad: &mut [f32],
) {
    for (leaf, (m, controls)) in moments.iter().zip(knots.controls()).enumerate() {
        let basis = bases.get((ch, leaf));
        for (row, &j) in basis.iter().zip(controls) {
            grad[j] += (row[0] * m[0] + row[1] * m[1] + row[2] * m[2] + row[3] * m[3]) as f32;
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
/// The tables come from `coefs` ([`TableSource`]): built in the call, or borrowed from a [`PreparedSpline`].
///
/// # Preconditions
/// The banks and the cache of one banks forward; one cotangent per bank, each `n · channels`;
/// `grad_x.len() == n · channels`.
///
/// # Postconditions
/// Bank `b`'s gradient is bit for bit [`catmull_rom_backward_into`]'s for bank `b`'s control points and
/// `grad_ys[b]`; `grad_x` is the banks' input gradients summed in bank order, every element written, `0`
/// outside the clamp interval.
pub fn catmull_rom_banks_backward_into<'a>(
    coefs: impl Into<TableSource<'a>>,
    cache: &CatmullRomCache,
    grad_ys: &[&[f32]],
    grad_x: &mut [f32],
) -> Vec<Vec<f32>> {
    let tables = coefs.into();
    let elems = cache.n * cache.channels;
    assert_eq!(grad_ys.len(), tables.bank_count(), "one cotangent per bank");
    for grad_y in grad_ys {
        assert_eq!(grad_y.len(), elems);
    }
    backward_over(tables, cache, grad_x, &AllCotangents { grad_ys })
}

/// The pullback of [`catmull_rom_routed_forward_into`] (Nagare plan 2026-10-23-incidence-partition, Amendments 5 and
/// 6): each bank's control-point gradient, in `coefs` order, from the elements routed to it (and, for the every-element
/// bank, from every element), and each element's input gradient written into `grad_x`: its every-element bank's term,
/// then its routed bank's. An element routed [`SKIP`] adds no routed term.
///
/// # Preconditions
/// The banks, routes and cache of one routed forward; `grads.every` exactly where the forward had an every-element
/// bank; every cotangent and `grad_x` `n · channels`.
///
/// # Postconditions
/// The gradients are [`catmull_rom_banks_backward_into`]'s under each bank's cotangent at the elements it was read at
/// and zero elsewhere, the input gradient's terms summed in bank order; bit for bit for finite values, the zeros it no
/// longer adds aside.
pub fn catmull_rom_routed_backward_into<'a>(
    coefs: impl Into<TableSource<'a>>,
    route: &[u8],
    cache: &CatmullRomCache,
    grads: RoutedCotangents<'_>,
    grad_x: &mut [f32],
) -> Vec<Vec<f32>> {
    let tables = coefs.into();
    let elems = cache.n * cache.channels;
    let RoutedCotangents { grad_y, every } = grads;
    assert_eq!(grad_y.len(), elems);
    assert_eq!(route.len(), elems, "one route per element");
    match every {
        Some(every) => {
            assert_eq!(every.len(), elems);
            backward_over(
                tables,
                cache,
                grad_x,
                &EveryAndRoutedCotangent {
                    every,
                    route,
                    grad_y,
                },
            )
        }
        None => backward_over(tables, cache, grad_x, &RoutedCotangent { route, grad_y }),
    }
}

/// The one pullback of a banks or routed forward: blocks of rows, each its own `grad_x` slice and accumulators, the
/// partials summed in block order.
fn backward_over<P: BackwardPull>(
    tables: TableSource<'_>,
    cache: &CatmullRomCache,
    grad_x: &mut [f32],
    pull: &P,
) -> Vec<Vec<f32>> {
    let elems = cache.n * cache.channels;
    assert_eq!(grad_x.len(), elems, "grad_x must be (n, channels)");
    let knots = cache
        .knots
        .as_ref()
        .expect("a forward wrote the cache, and recorded its knots");
    let banks = tables.slopes(cache.channels, knots);
    backward_tables(&banks, cache, grad_x, pull)
}

/// The one pullback over tables already built or borrowed ([`TableSource`]): blocks of rows, each its own `grad_x` slice and accumulators, the
/// partials summed in block order.
fn backward_tables<P: BackwardPull>(
    banks: &[Slopes],
    cache: &CatmullRomCache,
    grad_x: &mut [f32],
    pull: &P,
) -> Vec<Vec<f32>> {
    let elems = cache.n * cache.channels;
    debug_assert_eq!(grad_x.len(), elems, "grad_x must be (n, channels)");
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
    //
    // A block's partial is its raw moments, four per bank, channel and leaf. The blocks' moments are
    // added and mapped through the leaf bases once per call (Nagare plan
    // 2026-10-24-computed-parameters-once, Amendment 1): the fold ran once per block until then, and
    // a micro-batched update runs many more blocks than a batched one (866 against 304 measured).
    let stride = block_stride(cache.n, cache.channels);
    let blocks = grad_x.len().div_ceil(stride).max(1);
    let segs = knots.controls().len();
    let per_bank = cache.channels * segs;
    let block_len = banks.len() * per_bank;
    let mut partials = vec![[0.0f32; 4]; blocks * block_len];
    let uniform = knots.is_uniform();
    let kernel = |(block, (gx, local)): (usize, (&mut [f32], &mut [[f32; 4]]))| {
        let span = block * stride..(block * stride + gx.len());
        let chunk = BackwardChunk {
            pull,
            offset: span.start,
            x_clamped: &cache.x_clamped[span.clone()],
            interior: &cache.interior[span],
            grad_x: gx,
        };
        // One choice per call: uniform knots take the uniform segment arithmetic.
        if uniform {
            backward_chunk::<true, _>(banks, cache.channels, knots, chunk, local);
        } else {
            backward_chunk::<false, _>(banks, cache.channels, knots, chunk, local);
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
    // The blocks' moments are added in `f64`: the fold cancels between them (the leaf bases mix `±1/2`, `1`, `5/2`),
    // and a sum over every block grows them, so their `f32` sum would make the folded gradient about twice as
    // inexact as folding each block on its own (measured: tests/fold_accuracy.rs).
    fold_partials(banks, partials.chunks(block_len), (cache.channels, knots))
}

/// Each bank's control-point gradient from its blocks' partial moments (`[bank][channel][leaf]`, four each): the
/// partials added in `f64` in the order given, then mapped through the leaf bases. The one reduction of every pullback
/// ([`backward_tables`] and [`PreparedSpline::gradients`]).
fn fold_partials<'p>(
    banks: &[Slopes],
    partials: impl Iterator<Item = &'p [[f32; 4]]>,
    (channels, knots): (usize, &Knots),
) -> Vec<Vec<f32>> {
    let segs = knots.controls().len();
    let per_bank = channels * segs;
    let mut moments = vec![[0.0f64; 4]; banks.len() * per_bank];
    for local in partials {
        for (acc, value) in moments.iter_mut().zip(local) {
            for (a, &v) in acc.iter_mut().zip(value) {
                *a += f64::from(v);
            }
        }
    }
    let mut grad_coefs = vec![vec![0.0f32; channels * knots.len()]; banks.len()];
    for ((bank_grad, bank_moments), bank) in grad_coefs
        .iter_mut()
        .zip(moments.chunks_exact(per_bank))
        .zip(banks)
    {
        for (ch, (channel_grad, channel_moments)) in bank_grad
            .chunks_exact_mut(knots.len())
            .zip(bank_moments.chunks_exact(segs))
            .enumerate()
        {
            fold_moments(channel_moments, (knots, &bank.bases, ch), channel_grad);
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

/// A Chebyshev--CR spline's grid: `grid` uniform knots, the tangent rule its control points are interpolated under,
/// and the `k` Chebyshev terms read at those knots. `(grid, k)` is the chord rule, `(grid, k, tangents)` names one.
///
/// # Invariants
/// The terms are read at the uniform knots whatever the rule, so the rule changes the interpolation only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChebyshevGrid {
    grid: usize,
    k: usize,
    tangents: Tangents,
}

impl ChebyshevGrid {
    /// The knots' count.
    #[must_use]
    pub const fn grid(&self) -> usize {
        self.grid
    }

    /// The terms per channel.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// The tangent rule.
    #[must_use]
    pub const fn tangents(&self) -> Tangents {
        self.tangents
    }

    /// The uniform knots under the grid's tangent rule.
    #[must_use]
    pub fn knots(&self) -> Knots {
        Knots::uniform(self.grid).with_tangents(self.tangents)
    }
}

impl From<(usize, usize)> for ChebyshevGrid {
    fn from((grid, k): (usize, usize)) -> Self {
        Self {
            grid,
            k,
            tangents: Tangents::Chord,
        }
    }
}

impl From<(usize, usize, Tangents)> for ChebyshevGrid {
    fn from((grid, k, tangents): (usize, usize, Tangents)) -> Self {
        Self { grid, k, tangents }
    }
}

/// Each bank of Chebyshev `terms` (`channels × k`) as the control points it generates at `grid` uniform knots, and the
/// knot basis: the one composition every Chebyshev-CR forward and [`PreparedSpline::chebyshev`] share.
fn chebyshev_controls(
    terms: &[&[f32]],
    channels: usize,
    (grid, k): (usize, usize),
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let basis = chebyshev_knot_basis(grid, k);
    let controls = terms
        .iter()
        .map(|bank| chebyshev_control_points(bank, &basis, channels, grid, k))
        .collect();
    (controls, basis)
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
/// ([`catmull_rom_banks_forward_into`]) on the grid's knots under its tangent rule ([`ChebyshevGrid`]; `(grid, k)` is
/// the chord rule). Returns each bank's control points and the knot basis. The tape carries the knots, so the
/// pullback ([`chebyshev_cr_banks_backward_into`]) interpolates under the same rule.
///
/// # Preconditions
/// At least one bank, each `channels · k` terms; `grid >= 3`; the buffers as the banks forward's.
pub fn chebyshev_cr_banks_forward_into(
    coefs: &[&[f32]],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: impl Into<ChebyshevGrid>,
    into: CatmullRomBanksBuffers<'_>,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let grid = grid.into();
    let (controls, basis) = chebyshev_controls(coefs, channels, (grid.grid, grid.k));
    let banks: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
    catmull_rom_banks_forward_into(&banks, x, n, channels, &grid.knots(), into);
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
/// ([`catmull_rom_banks_backward_into`]). The control points' tables come from `controls` ([`TableSource`]):
/// the forward's control points, or a [`PreparedSpline::chebyshev`] of the same terms.
pub fn chebyshev_cr_banks_backward_into<'a>(
    controls: impl Into<TableSource<'a>>,
    basis: &[f32],
    cache: &CatmullRomCache,
    grad_ys: &[&[f32]],
    k: usize,
    grad_x: &mut [f32],
) -> Vec<Vec<f32>> {
    catmull_rom_banks_backward_into(controls, cache, grad_ys, grad_x)
        .iter()
        .map(|cr_grad_coef| {
            chebyshev_terms_gradient(cr_grad_coef, basis, (cache.channels, cache.grid, k))
        })
        .collect()
}

/// [`chebyshev_cr_banks_forward_into`] with each element evaluated on its routed bank alone, into one output
/// ([`catmull_rom_routed_forward_into`]), on the grid's knots under its tangent rule. Returns each bank's control
/// points and the knot basis.
///
/// # Preconditions
/// As [`chebyshev_cr_banks_forward_into`] with one output, and the routes as [`catmull_rom_routed_forward_into`]'s.
pub fn chebyshev_cr_routed_forward_into(
    coefs: &[&[f32]],
    route: &[u8],
    x: &[f32],
    (n, channels): (usize, usize),
    grid: impl Into<ChebyshevGrid>,
    into: RoutedBuffers<'_>,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let grid = grid.into();
    let (controls, basis) = chebyshev_controls(coefs, channels, (grid.grid, grid.k));
    let banks: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
    catmull_rom_routed_forward_into(&banks, route, x, (n, channels), &grid.knots(), into);
    (controls, basis)
}

/// The pullback of [`chebyshev_cr_routed_forward_into`]: each bank's Chebyshev-term gradient, projected from its
/// control-point gradient, and each element's input gradient from its routed bank, into `grad_x`
/// ([`catmull_rom_routed_backward_into`]). The control points' tables come from `controls` ([`TableSource`]), as in
/// [`chebyshev_cr_banks_backward_into`].
pub fn chebyshev_cr_routed_backward_into<'a>(
    controls: impl Into<TableSource<'a>>,
    basis: &[f32],
    (cache, route): (&CatmullRomCache, &[u8]),
    grads: RoutedCotangents<'_>,
    k: usize,
    grad_x: &mut [f32],
) -> Vec<Vec<f32>> {
    catmull_rom_routed_backward_into(controls, route, cache, grads, grad_x)
        .iter()
        .map(|cr_grad_coef| {
            chebyshev_terms_gradient(cr_grad_coef, basis, (cache.channels, cache.grid, k))
        })
        .collect()
}

/// The Chebyshev terms' gradient from their control points' (`(channels, grid)`): each term's the
/// control gradients weighted by the term at the knots.
fn chebyshev_terms_gradient(
    cr_grad_coef: &[f32],
    basis: &[f32],
    (channels, grid, k): (usize, usize, usize),
) -> Vec<f32> {
    let mut grad_coef = vec![0.0; channels * k];
    for ch in 0..channels {
        for term in 0..k {
            let mut acc = 0.0;
            for g in 0..grid {
                acc += cr_grad_coef[ch * grid + g] * basis[g * k + term];
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

    /// The uniform Catmull--Rom weights at `t`, the closed form: the minimal analytical reference the
    /// power form is checked against.
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

    /// The `t`-derivatives of [`cr_weights`].
    fn cr_weight_derivatives(t: f32) -> [f32; 4] {
        let t2 = t * t;
        [
            0.5 * (-3.0 * t2 + 4.0 * t - 1.0),
            0.5 * (9.0 * t2 - 10.0 * t),
            0.5 * (-9.0 * t2 + 8.0 * t + 1.0),
            0.5 * (3.0 * t2 - 2.0 * t),
        ]
    }

    /// The largest magnitude in `v`.
    fn largest(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    /// The largest `|got − want|` over the two, and where.
    fn worst(got: &[f32], want: &[f32]) -> (f32, usize) {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .enumerate()
            .fold(
                (0.0, 0),
                |(m, at), (e, d)| if d > m { (d, e) } else { (m, at) },
            )
    }

    fn loss_sum_cr(coef: &[f32], x: &[f32], n: usize, channels: usize, grid: usize) -> f32 {
        catmull_rom_forward(coef, x, n, channels, &crate::knots::Knots::uniform(grid))
            .0
            .iter()
            .sum()
    }

    // ── row-parallelism (2026-08-17) ────────────────────────────────────────
    //
    // These pin the two guarantees the parallel paths must keep: the pure maps agree
    // with an independent reference to its rounding (the kernels evaluate the power
    // form, the reference the closed-form weights), and every result is identical
    // across thread counts. The reference below is deliberately a separate naive
    // implementation rather than a call back into the production code, so a bug in
    // the shared row kernel cannot hide by cancelling itself.

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

    /// The forward agrees with the closed-form weights on both sides of the parallel
    /// threshold, to `2e-6` of the control points' magnitude — including a row count
    /// that does not divide the chunk count, which is where an off-by-one in the
    /// chunking would show up. The tape is bit-exact: the power form changes what the
    /// second pass computes, not the clamp, the leaf or the coordinate.
    #[test]
    fn forward_matches_the_closed_form_weights_on_both_paths() {
        let (channels, grid) = (8usize, 6usize);
        // 511 * 8 = 4088 < PAR_MIN_ELEMS <= 513 * 8 = 4104: straddles the boundary,
        // and 1601 is coprime with the 64 chunks.
        for n in [1usize, 7, 511, 513, 1600, 1601, 4099] {
            let x = pseudo_input(n * channels, 1);
            let coef = pseudo_input(channels * grid, 2);
            let (out, cache) =
                catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid));
            let (want_out, want_xc, want_idx, want_t) = naive_forward(&coef, &x, n, channels, grid);
            let (gap, at) = worst(&out, &want_out);
            assert!(
                gap <= 2e-6 * largest(&coef),
                "n={n}: output {} vs {} at {at}",
                out[at],
                want_out[at]
            );
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

    /// `grad_x` agrees with the closed-form weight derivatives to `2e-6` of
    /// `|gy| · |q| · dt/dx` at their largest. `grad_coef` is a reduction, folded from
    /// per-leaf moments, so it is required to be *close* to the reference's sequential
    /// sum of `gy · w`, `1e-5` of its largest entry — the strict requirement on it is
    /// thread-count independence, tested separately.
    #[test]
    fn backward_matches_the_closed_form_weights_closely() {
        let (channels, grid) = (8usize, 6usize);
        for n in [1usize, 511, 513, 1600, 1601] {
            let x = pseudo_input(n * channels, 3);
            let coef = pseudo_input(channels * grid, 4);
            let grad_y = pseudo_input(n * channels, 5);
            let (_, cache) =
                catmull_rom_forward(&coef, &x, n, channels, &crate::knots::Knots::uniform(grid));
            let got = catmull_rom_backward(&coef, &cache, &grad_y);
            let (want_gx, want_gc) = naive_backward(&coef, &cache, &grad_y);

            let reach = largest(&grad_y) * largest(&coef) * 0.5 * (grid - 1) as f32;
            let (gap, at) = worst(&got.grad_x, &want_gx);
            assert!(
                gap <= 2e-6 * reach,
                "n={n}: grad_x {} vs {} at {at}",
                got.grad_x[at],
                want_gx[at]
            );
            assert!(
                got.grad_x
                    .iter()
                    .zip(&cache.interior)
                    .all(|(g, &inside)| inside || g.to_bits() == 0),
                "n={n}: grad_x is exactly 0 outside the clamp interval"
            );

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

    /// The non-uniform Catmull--Rom's weights at `t` on a leaf of tangent shares `(r₀, r₁)`, the
    /// Hermite form `(−h₁₀ r₀, h₀₀ − h₁₁ r₁, h₀₁ + h₁₀ r₀, h₁₁ r₁)`, in `f64`.
    fn hermite_weights(t: f64, (r0, r1): (f64, f64)) -> [f64; 4] {
        let (t2, t3) = (t * t, t * t * t);
        let (h00, h10) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t);
        let (h01, h11) = (-2.0 * t3 + 3.0 * t2, t3 - t2);
        [-h10 * r0, h00 - h11 * r1, h01 + h10 * r0, h11 * r1]
    }

    /// Per leaf of `depths` its tangent shares `h / (h₋ + h)` and `h / (h + h₊)`, the widths in base
    /// segment units (`2^-d`), a missing neighbour mirroring the leaf's own width.
    fn tangent_shares(depths: &[u8]) -> Vec<(f64, f64)> {
        let widths: Vec<f64> = depths
            .iter()
            .flat_map(|&d| (0..1u32 << d).map(move |_| 1.0 / f64::from(1u32 << d)))
            .collect();
        (0..widths.len())
            .map(|k| {
                let h = widths[k];
                let before = if k == 0 { h } else { widths[k - 1] };
                let after = widths.get(k + 1).copied().unwrap_or(h);
                (h / (before + h), h / (h + after))
            })
            .collect()
    }

    /// **Every leaf's basis is its weights in powers of `t`** (Nagare plan
    /// 2026-10-23-incidence-partition, Amendment 7): `B_ℓ τ(t)` against the Hermite form with the
    /// leaf's tangent shares, derived here from the depths, over refined knots of several depths
    /// (the deepest 8), and each leaf's scale `2^d`. The uniform basis is
    /// `½ [[0, −1, 2, −1], [2, 0, −5, 3], [0, 1, 4, −3], [0, 0, −1, 1]]` exactly, and is [`cr_weights`].
    #[test]
    fn every_leaf_basis_is_its_hermite_weights_in_powers() {
        let uniform = [
            [0.0, -0.5, 1.0, -0.5],
            [1.0, 0.0, -2.5, 1.5],
            [0.0, 0.5, 2.0, -1.5],
            [0.0, 0.0, -0.5, 0.5],
        ];
        for grid in [3, 6, 9] {
            let knots = Knots::uniform(grid);
            assert!(knots.basis().iter().all(|b| *b == uniform), "grid {grid}");
            assert!(knots.scales().iter().all(|&s| s == 1.0), "grid {grid}");
        }
        for t in [0.0f32, 0.125, 0.3, 0.5, 0.77, 1.0] {
            let tau = [1.0, t, t * t, t * t * t].map(f64::from);
            for (k, &w) in cr_weights(t).iter().enumerate() {
                let got: f64 = (0..4).map(|j| uniform[k][j] * tau[j]).sum();
                assert!(
                    (got - f64::from(w)).abs() < 1e-6,
                    "t={t}, w{k}: {got} vs {w}"
                );
            }
        }
        let cases: [(usize, &[u8]); 4] = [
            (6, &[0, 1, 2, 1, 0]),
            (4, &[3, 0, 1]),
            (5, &[0, 8, 0, 2]),
            (3, &[2, 5]),
        ];
        for (grid, depths) in cases {
            let knots = Knots::refined(grid, depths);
            let shares = tangent_shares(depths);
            assert_eq!(
                knots.basis().len(),
                shares.len(),
                "{depths:?}: one basis a leaf"
            );
            let scales: Vec<f32> = depths
                .iter()
                .flat_map(|&d| (0..1u32 << d).map(move |_| (1u32 << d) as f32))
                .collect();
            assert_eq!(knots.scales(), scales.as_slice(), "{depths:?}: 2^d a leaf");
            for (leaf, (basis, &share)) in knots.basis().iter().zip(&shares).enumerate() {
                for t in [0.0f64, 0.2, 0.5, 0.9, 1.0] {
                    let want = hermite_weights(t, share);
                    for (k, row) in basis.iter().enumerate() {
                        let got: f64 = (0..4).map(|j| row[j] * t.powi(j as i32)).sum();
                        assert!(
                            (got - want[k]).abs() < 1e-12,
                            "{depths:?} leaf {leaf}, t={t}, w{k}: {got} vs {}",
                            want[k]
                        );
                    }
                }
            }
        }
    }

    /// **Constant control points give a constant spline, exactly** (partition of unity): every leaf's
    /// power coefficients are `(c, 0, 0, 0)` to far below `c`'s last bit, so the output is `c` bit for
    /// bit over uniform and refined knots. Over uniform knots the coefficients are exactly
    /// `(c, 0, 0, 0)` and the input gradient exactly zero.
    #[test]
    fn constant_control_points_give_a_constant_spline_exactly() {
        let (n, channels) = (4099usize, 2usize);
        let x = pseudo_input(n * channels, 11);
        let knot_sets = [
            Knots::uniform(6),
            Knots::refined(6, &[0, 1, 2, 1, 0]),
            Knots::refined(5, &[0, 8, 0, 2]),
        ];
        for knots in &knot_sets {
            for c in [0.7f32, -1.3e3, 3.1e-5] {
                let coef = vec![c; channels * knots.len()];
                let (y, cache) = catmull_rom_forward(&coef, &x, n, channels, knots);
                assert!(
                    y.iter().all(|v| v.to_bits() == c.to_bits()),
                    "{:?}, c={c}: the output is c",
                    knots.depths()
                );
                if knots.is_uniform() {
                    let polys = Polys::values(&coef, channels, knots);
                    assert!(
                        polys.table.iter().all(|a| *a == [c, 0.0, 0.0, 0.0]),
                        "c={c}"
                    );
                    let back = catmull_rom_backward(&coef, &cache, &x);
                    assert!(back.grad_x.iter().all(|&g| g == 0.0), "c={c}: no slope");
                }
            }
        }
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
