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
#[derive(Debug, Clone)]
pub struct CatmullRomCache {
    /// Clamped input values, flat `(n, channels)`.
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
    /// Segment index per input element.
    ///
    /// `u8`, not `usize`: the value is a CR segment index bounded by `grid - 2`,
    /// and every grid in this codebase is 5-6. Storing it in 8 bytes cost 8x the
    /// cache traffic of the buffer for no reachable range.
    pub indices: Vec<u8>,
    /// Local segment coordinate per input element.
    pub t: Vec<f32>,
    /// Number of rows in the flat input.
    pub n: usize,
    /// Channel count.
    pub channels: usize,
    /// Number of CR control points per channel.
    pub grid: usize,
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

/// The four per-element buffers [`catmull_rom_forward`] fills for a single row.
///
/// Bundled into a struct so the shared row kernel takes four arguments instead of
/// seven (§6.5 #6) and so the sequential and parallel paths call *one*
/// implementation rather than each carrying a copy of the arithmetic (§6.1).
struct ForwardRow<'a> {
    out: &'a mut [f32],
    x_clamped: &'a mut [f32],
    interior: &'a mut [bool],
    indices: &'a mut [u8],
    t: &'a mut [f32],
}

/// Evaluate the CR spline for every element of a whole block of rows.
///
/// Iterating rows-then-channels (rather than one flat loop with `element %
/// channels`) keeps the channel index a loop counter instead of an integer
/// division — the modulo version measured ~24 % slower.
///
/// # Preconditions
/// Every slice in `w` has the same length as `x_block`, which is a whole multiple
/// of `channels`; `coef.len() == channels * grid`; `grid >= 3`.
#[inline(always)]
fn forward_block(coef: &[f32], grid: usize, channels: usize, x_block: &[f32], w: ForwardRow<'_>) {
    debug_assert_eq!(x_block.len() % channels, 0, "block must be whole rows");
    let span = 0.5 * (grid - 1) as f32;
    for row in 0..x_block.len() / channels {
        let base_e = row * channels;
        for ch in 0..channels {
            let e = base_e + ch;
            let raw = x_block[e];
            let xc = raw.clamp(-1.0, 1.0);
            let u = (xc + 1.0) * span;
            let i = (u.floor() as usize).min(grid - 2);
            let t = u - i as f32;
            let weights = cr_weights(t);
            let control = cr_control_indices(i, grid);
            let base = ch * grid;
            w.out[e] = weights[0] * coef[base + control[0]]
                + weights[1] * coef[base + control[1]]
                + weights[2] * coef[base + control[2]]
                + weights[3] * coef[base + control[3]];
            w.x_clamped[e] = xc;
            // Closed interval: `±1` is interior, matching the guard this replaces.
            w.interior[e] = (-1.0..=1.0).contains(&raw);
            w.indices[e] = i as u8;
            w.t[e] = t;
        }
    }
}

/// Forward Catmull--Rom activation.
///
/// # Preconditions
/// - `grid >= 3`
/// - `x.len() == n * channels`
/// - `coef.len() == channels * grid`
///
/// # Postconditions
/// Returns output flat `(n, channels)` and a cache for backward. Inputs are
/// clamped to `[-1, 1]`, matching the canonical Python evaluator.
pub fn catmull_rom_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
) -> (Vec<f32>, CatmullRomCache) {
    // Three is the smallest grid this operator **supports**: `i` ranges over `{0, 1}`, giving two
    // segments, and the outer stencil points are supplied by the endpoint replication
    // `cr_control_indices` already performs. The CogInfoCom HSiKAN configuration uses grid 3, which
    // is why the minimum moved from 4. Grid 2 is out of contract here — not asserted to be
    // mathematically impossible, simply not a configuration this operator is certified for.
    assert!(
        grid >= 3,
        "Catmull--Rom needs at least 3 control points, got {grid}"
    );
    // `indices` is `u8`. The stored value is `min(floor(u), grid - 2)`, so it fits
    // exactly when `grid - 2 <= u8::MAX`. Grids in this codebase run 3-6, so the stored
    // value is at most 4; this is the guard that turns a silent wraparound into a failed
    // precondition if that ever stops being true.
    assert!(
        grid - 2 <= u8::MAX as usize,
        "grid {grid} exceeds the u8 segment-index range; widen CatmullRomCache::indices"
    );
    assert_eq!(x.len(), n * channels);
    assert_eq!(coef.len(), channels * grid);

    let mut out = vec![0.0; x.len()];
    let mut x_clamped = vec![0.0; x.len()];
    let mut interior = vec![false; x.len()];
    let mut indices = vec![0u8; x.len()];
    let mut ts = vec![0.0; x.len()];

    // Rows are independent and write disjoint slices, so this is a pure map and the
    // parallel result is bit-identical to the sequential one at any thread count.
    // Blocks are many rows wide: one row is far too little work per rayon item.
    #[cfg(feature = "parallel")]
    if use_parallel(x.len()) {
        let stride = block_stride(n, channels);
        out.par_chunks_mut(stride)
            .zip(x_clamped.par_chunks_mut(stride))
            .zip(interior.par_chunks_mut(stride))
            .zip(indices.par_chunks_mut(stride))
            .zip(ts.par_chunks_mut(stride))
            .zip(x.par_chunks(stride))
            .for_each(|(((((out_b, xc_b), in_b), idx_b), t_b), x_b)| {
                forward_block(
                    coef,
                    grid,
                    channels,
                    x_b,
                    ForwardRow {
                        out: out_b,
                        x_clamped: xc_b,
                        interior: in_b,
                        indices: idx_b,
                        t: t_b,
                    },
                );
            });
    } else {
        forward_block(
            coef,
            grid,
            channels,
            x,
            ForwardRow {
                out: &mut out,
                x_clamped: &mut x_clamped,
                interior: &mut interior,
                indices: &mut indices,
                t: &mut ts,
            },
        );
    }

    #[cfg(not(feature = "parallel"))]
    forward_block(
        coef,
        grid,
        channels,
        x,
        ForwardRow {
            out: &mut out,
            x_clamped: &mut x_clamped,
            interior: &mut interior,
            indices: &mut indices,
            t: &mut ts,
        },
    );

    (
        out,
        CatmullRomCache {
            x_clamped,
            interior,
            indices,
            t: ts,
            n,
            channels,
            grid,
        },
    )
}

/// One block's slices as the six-way zip yields them.
///
/// Named so the sequential and parallel branches can share a single closure: both
/// `chunks_mut` and `par_chunks_mut` produce this same nesting, and the shared
/// closure is what guarantees the two paths run identical arithmetic.
type BackwardZip<'a> = (
    (
        (((&'a mut [f32], &'a [f32]), &'a [u8]), &'a [f32]),
        &'a [bool],
    ),
    &'a mut [f32],
);

/// One chunk's view of the backward's per-element inputs and outputs.
///
/// Every slice covers the same contiguous element range, a whole number of rows
/// long. Bundled for the same reason as [`ForwardRow`]: one shared kernel, and a
/// parameter list that stays under the §6.5 #6 limit.
struct BackwardChunk<'a> {
    grad_x: &'a mut [f32],
    grad_y: &'a [f32],
    indices: &'a [u8],
    t: &'a [f32],
    /// Per-element interior mask from the forward; see [`CatmullRomCache::interior`].
    interior: &'a [bool],
}

/// Accumulate one contiguous chunk of rows into `grad_x` and `grad_coef`.
///
/// # Preconditions
/// Every slice in `c` has the same length, a multiple of `channels`, and the chunk
/// starts on a row boundary — otherwise the `element % channels` channel index is
/// wrong. `grad_coef.len() == channels * grid`.
///
/// # Postconditions
/// `grad_x` is *written* (each element assigned once); `grad_coef` is
/// *accumulated* into, so the caller owns its initial value.
#[inline(always)]
fn backward_chunk(
    coef: &[f32],
    channels: usize,
    grid: usize,
    c: BackwardChunk<'_>,
    grad_coef: &mut [f32],
) {
    debug_assert_eq!(c.grad_x.len() % channels, 0, "chunk must be whole rows");
    let du_dx = 0.5 * (grid - 1) as f32;
    // Rows outer, channels inner — the channel index must not become an integer
    // division per element (that cost ~24 % when first written that way).
    for row in 0..c.grad_x.len() / channels {
        let base_e = row * channels;
        for ch in 0..channels {
            let e = base_e + ch;
            let gy = c.grad_y[e];
            let t = c.t[e];
            let weights = cr_weights(t);
            let control = cr_control_indices(c.indices[e] as usize, grid);
            let base = ch * grid;
            for k in 0..4 {
                grad_coef[base + control[k]] += gy * weights[k];
            }
            // The clamp's own Jacobian: `1` inside the closed interval, `0` outside. Read from
            // the recorded `interior` mask, not from `x_clamped` — the latter is always inside by
            // construction, which is what made this guard a tautology before 2026-08-25.
            if c.interior[e] {
                let dweights = cr_weight_derivatives(t);
                let mut dy_dt = 0.0;
                for k in 0..4 {
                    dy_dt += dweights[k] * coef[base + control[k]];
                }
                c.grad_x[e] = gy * dy_dt * du_dx;
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
pub fn catmull_rom_backward(
    coef: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
) -> CatmullRomBackward {
    assert_eq!(grad_y.len(), cache.n * cache.channels);
    assert_eq!(coef.len(), cache.channels * cache.grid);

    let mut grad_x = vec![0.0; grad_y.len()];
    let mut grad_coef = vec![0.0; coef.len()];

    // Rows split into a FIXED number of blocks. Each block writes its own disjoint
    // `grad_x` slice (bit-exact) and accumulates into its own private `grad_coef`
    // slot — 48 floats for california.
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
    let mut partials = vec![0.0f32; blocks * coef.len()];

    let kernel = |(((((gx, gy), indices), t), interior), local): BackwardZip<'_>| {
        backward_chunk(
            coef,
            cache.channels,
            cache.grid,
            BackwardChunk {
                grad_x: gx,
                grad_y: gy,
                indices,
                t,
                interior,
            },
            local,
        );
    };

    #[cfg(feature = "parallel")]
    if use_parallel(grad_y.len()) {
        grad_x
            .par_chunks_mut(stride)
            .zip(grad_y.par_chunks(stride))
            .zip(cache.indices.par_chunks(stride))
            .zip(cache.t.par_chunks(stride))
            .zip(cache.interior.par_chunks(stride))
            .zip(partials.par_chunks_mut(coef.len()))
            .for_each(kernel);
    } else {
        grad_x
            .chunks_mut(stride)
            .zip(grad_y.chunks(stride))
            .zip(cache.indices.chunks(stride))
            .zip(cache.t.chunks(stride))
            .zip(cache.interior.chunks(stride))
            .zip(partials.chunks_mut(coef.len()))
            .for_each(kernel);
    }

    #[cfg(not(feature = "parallel"))]
    grad_x
        .chunks_mut(stride)
        .zip(grad_y.chunks(stride))
        .zip(cache.indices.chunks(stride))
        .zip(cache.t.chunks(stride))
        .zip(cache.interior.chunks(stride))
        .zip(partials.chunks_mut(coef.len()))
        .for_each(kernel);

    // Summed in block order, which is fixed by the buffer layout and not by the
    // scheduler — this is what makes `grad_coef` thread-count independent.
    for local in partials.chunks(coef.len()) {
        for (acc, value) in grad_coef.iter_mut().zip(local) {
            *acc += value;
        }
    }

    CatmullRomBackward { grad_x, grad_coef }
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
    let basis = chebyshev_knot_basis(grid, k);
    let control = chebyshev_control_points(coef, &basis, channels, grid, k);
    let (y, cache) = catmull_rom_forward(&control, x, n, channels, grid);
    (y, cache, control, basis)
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
    let cr = catmull_rom_backward(control_points, cache, grad_y);
    let mut grad_coef = vec![0.0; cache.channels * k];
    for ch in 0..cache.channels {
        for term in 0..k {
            let mut acc = 0.0;
            for g in 0..cache.grid {
                acc += cr.grad_coef[ch * cache.grid + g] * basis[g * k + term];
            }
            grad_coef[ch * k + term] = acc;
        }
    }
    ChebyshevCrBackward {
        grad_x: cr.grad_x,
        grad_coef,
    }
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

fn cr_control_indices(i: usize, grid: usize) -> [usize; 4] {
    [
        i.saturating_sub(1).min(grid - 1),
        i.min(grid - 1),
        (i + 1).min(grid - 1),
        (i + 2).min(grid - 1),
    ]
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
    use super::*;

    fn loss_sum_cr(coef: &[f32], x: &[f32], n: usize, channels: usize, grid: usize) -> f32 {
        catmull_rom_forward(coef, x, n, channels, grid)
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
                let w = cr_weights(cache.t[e]);
                let dw = cr_weight_derivatives(cache.t[e]);
                let control = cr_control_indices(cache.indices[e] as usize, cache.grid);
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
            let (out, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
            let (want_out, want_xc, want_idx, want_t) = naive_forward(&coef, &x, n, channels, grid);
            assert_eq!(out, want_out, "n={n}: output must be bit-exact");
            assert_eq!(cache.x_clamped, want_xc, "n={n}: clamped input");
            assert_eq!(cache.indices, want_idx, "n={n}: segment indices");
            assert_eq!(cache.t, want_t, "n={n}: segment coordinates");
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
            let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
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
        let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);

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
                .install(|| catmull_rom_forward(&coef, &x, n, channels, grid))
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
        let (y, _) = catmull_rom_forward(&coef, &x, 5, 1, 5);
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
        let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
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
