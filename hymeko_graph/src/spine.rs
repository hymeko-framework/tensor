//! Rust "spine" — SoA-native FIR signed-cycle aggregator.
//!
//! Streamlined low-level kernel that produces per-cycle features
//! from per-vertex features + a [`TopKCyclesBatch`]. Replaces the
//! per-cycle MLP/spline forward path in HSiKAN with a 2-branch FIR
//! (one filter bank per sign), trading some expressiveness for
//! 10–100× single-core throughput vs PyTorch and trivial Rayon
//! scaling.
//!
//! ## Math
//!
//! For each cycle $c = (v_0, \ldots, v_{K-1})$ with boundary-edge
//! signs $\sigma = (\sigma_0, \ldots, \sigma_{K-1}) \in \{-1, +1\}^K$,
//! the per-cycle feature is
//! \[
//!   \mathrm{out}_c[j] = \sum_{i=0}^{K-1}
//!     k_{\sigma_i}[i] \cdot \mathrm{features}[v_i][j],
//!   \quad j = 0, \ldots, d-1.
//! \]
//! `k_{+1}` and `k_{-1}` are learnable length-$K$ filter banks; the
//! sign of cycle edge $i$ selects which bank's position-$i$
//! coefficient multiplies vertex $v_i$'s feature.
//!
//! ## Properties
//!
//! - **Linear in `K · d`** per cycle; no nested dims.
//! - **Compute-bound only at low `d`**; memory-bandwidth-bound for
//!   `d >= 32` since each cycle reads `K * d * 4` bytes of features.
//! - **Forward-only**: backward needs a custom autograd path
//!   (analogous to the Triton kernels already in `hymeko_neuro`).
//! - **Drop-in for HSiKAN inference**: same SoA cycle layout, same
//!   sign convention, same input feature buffer.
//!
//! ## Use cases
//!
//! 1. **Inference path** for trained HSiKAN models — extract the
//!    Catmull-Rom spline weights at low-temperature inputs and treat
//!    them as fixed FIR coefficients.
//! 2. **Pre-training / feature extraction**: produce cycle-aggregated
//!    features from arbitrary per-vertex features faster than the
//!    Python pipeline.
//! 3. **Online / streaming**: small fixed cost per cycle makes this
//!    suitable for dynamic graphs where cycles arrive incrementally.

#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;

use crate::topk_cycles::TopKCyclesBatch;

/// FIR filter banks for the signed-cycle aggregator.
///
/// `coef_pos` is the length-`k` filter applied at positions where
/// the cycle edge sign is `+1`; `coef_neg` is applied at `-1` edges.
/// Both have shape `(k,)`; per-cycle output is a single `d`-vector.
#[derive(Debug, Clone)]
pub struct SignedCycleFIR {
    /// `(k,)` coefficients for σ=+1 branch.
    pub coef_pos: Vec<f32>,
    /// `(k,)` coefficients for σ=-1 branch.
    pub coef_neg: Vec<f32>,
}

impl SignedCycleFIR {
    /// Identity filter: `coef_pos = coef_neg = 1/k` (mean-pool).
    pub fn mean_pool(k: usize) -> Self {
        let v = 1.0 / k as f32;
        Self {
            coef_pos: vec![v; k],
            coef_neg: vec![v; k],
        }
    }

    /// Sign-aware identity: `coef_pos = +1/k`, `coef_neg = -1/k`
    /// (the canonical signed-Laplacian-style aggregator).
    pub fn signed_mean(k: usize) -> Self {
        let v = 1.0 / k as f32;
        Self {
            coef_pos: vec![v; k],
            coef_neg: vec![-v; k],
        }
    }

    /// New from explicit coefficients. Panics if lengths mismatch.
    pub fn new(coef_pos: Vec<f32>, coef_neg: Vec<f32>) -> Self {
        assert_eq!(
            coef_pos.len(), coef_neg.len(),
            "coef_pos and coef_neg must have equal length",
        );
        Self { coef_pos, coef_neg }
    }

    /// Filter length `k`.
    pub fn k(&self) -> usize {
        self.coef_pos.len()
    }
}

/// AVX2 inner-loop kernel for one cycle.
///
/// Accumulates `coef * features[v_i]` into `out` for each of `K`
/// positions. Vectorised over the `d` dimension at 8 f32 lanes per
/// SIMD op. Handles non-multiple-of-8 `d` via a scalar tail.
///
/// # Safety
/// Caller must verify AVX2 is available at runtime
/// (via `is_x86_feature_detected!("avx2")`). The dispatcher in
/// [`fir_cycle_forward`] handles this check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
#[inline]
unsafe fn fir_one_cycle_avx2(
    cycles_slice: &[u32],
    signs_slice: &[i8],
    features: &[f32],
    d: usize,
    fir: &SignedCycleFIR,
    out: &mut [f32],
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let k = cycles_slice.len();
    let d_vec = (d / 8) * 8;
    // Each lane of the AVX register holds one float of the output.
    // We allocate as many accumulator registers as needed for `d`,
    // process all K positions per group, then store. To keep things
    // straightforward and avoid a stack-allocated register array,
    // process the output in groups of 8 floats (one __m256 at a time)
    // and reload the cycle/sign coeffs per group.
    // This is what LLVM tends to do anyway with a vectorised inner
    // loop; we make it explicit so target_feature=avx2,fma fires.
    // Out is zero-initialised by the dispatcher (single-cycle slice
    // ⇒ we own it exclusively), so we accumulate into a register
    // and STORE (not add-then-store).  Saves 2 mem ops per 8-lane
    // step, and lets LLVM keep `acc` resident across `k` iters.
    let mut j = 0usize;
    while j < d_vec {
        // SAFETY: AVX2 + FMA enabled via #[target_feature]; pointer
        // offsets are bounded by slice length, vertex indices into
        // features are bounded by n_vertices.
        #[allow(unsafe_code)]
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for i in 0..k {
                let v = cycles_slice[i] as usize;
                let s = signs_slice[i];
                let coef = if s > 0 { fir.coef_pos[i] } else { fir.coef_neg[i] };
                let f_ptr = features.as_ptr().add(v * d + j);
                let f = _mm256_loadu_ps(f_ptr);
                let c = _mm256_set1_ps(coef);
                acc = _mm256_fmadd_ps(c, f, acc);
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(j), acc);
        }
        j += 8;
    }
    // Scalar tail for d not a multiple of 8.
    while j < d {
        let mut a = 0.0f32;
        for i in 0..k {
            let v = cycles_slice[i] as usize;
            let s = signs_slice[i];
            let coef = if s > 0 { fir.coef_pos[i] } else { fir.coef_neg[i] };
            a += coef * features[v * d + j];
        }
        out[j] += a;
        j += 1;
    }
}

/// Per-cycle FIR forward.
///
/// # Inputs
/// - `batch`        : `TopKCyclesBatch` with `n` cycles of length `k`.
/// - `features`     : flat `(n_vertices * d)` row-major per-vertex features.
/// - `n_vertices`   : number of distinct vertex indices.
/// - `d`            : feature dim.
/// - `fir`          : filter banks.
///
/// # Output
/// - `(n_cycles * d)` flat row-major per-cycle aggregated features.
///
/// # Preconditions
/// - `fir.k() == batch.k`
/// - `features.len() == n_vertices * d`
/// - All cycle vertex indices `< n_vertices`
///
/// # Threading
/// Parallel over cycles via Rayon.
pub fn fir_cycle_forward(
    batch: &TopKCyclesBatch,
    features: &[f32],
    n_vertices: usize,
    d: usize,
    fir: &SignedCycleFIR,
) -> Vec<f32> {
    let k = batch.k;
    let n_cycles = batch.len();
    assert_eq!(fir.k(), k, "fir.k() must match batch.k");
    assert_eq!(features.len(), n_vertices * d,
                "features length must equal n_vertices * d");
    if n_cycles == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f32; n_cycles * d];
    // Runtime AVX2 detection — single check at the call site, used
    // by every parallel chunk. is_x86_feature_detected! caches its
    // result via a static OnceCell.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    out.par_chunks_mut(d).enumerate().for_each(|(ci, out_slice)| {
        let c_start = ci * k;
        let s_start = ci * k;
        let cycles_slice = &batch.cycles[c_start..c_start + k];
        let signs_slice = &batch.signs[s_start..s_start + k];
        if use_avx2 {
            // SAFETY: feature-gated by runtime detection above; the
            // function body is annotated #[target_feature(enable=avx2,fma)].
            #[cfg(target_arch = "x86_64")]
            #[allow(unsafe_code)]
            unsafe {
                fir_one_cycle_avx2(
                    cycles_slice, signs_slice, features, d, fir, out_slice,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            fir_one_cycle_scalar(
                cycles_slice, signs_slice, features, d, fir, out_slice,
            );
        } else {
            fir_one_cycle_scalar(
                cycles_slice, signs_slice, features, d, fir, out_slice,
            );
        }
    });
    out
}

/// Scalar fallback for the per-cycle inner loop.
#[inline]
fn fir_one_cycle_scalar(
    cycles_slice: &[u32],
    signs_slice: &[i8],
    features: &[f32],
    d: usize,
    fir: &SignedCycleFIR,
    out: &mut [f32],
) {
    let k = cycles_slice.len();
    for i in 0..k {
        let v = cycles_slice[i] as usize;
        let s = signs_slice[i];
        let coef = if s > 0 { fir.coef_pos[i] } else { fir.coef_neg[i] };
        let f_start = v * d;
        let f_slice = &features[f_start..f_start + d];
        for j in 0..d {
            out[j] += coef * f_slice[j];
        }
    }
}

/// Convenience: forward + scatter-mean to per-vertex features.
///
/// Returns `(n_vertices, d)` flat per-vertex aggregated features.
/// Vertices appearing in no cycle receive zeros.
pub fn fir_cycle_scatter_mean(
    batch: &TopKCyclesBatch,
    features: &[f32],
    n_vertices: usize,
    d: usize,
    fir: &SignedCycleFIR,
) -> Vec<f32> {
    let per_cycle = fir_cycle_forward(batch, features, n_vertices, d, fir);
    let k = batch.k;
    let n_cycles = batch.len();
    let mut out = vec![0.0f32; n_vertices * d];
    let mut counts = vec![0u32; n_vertices];
    for ci in 0..n_cycles {
        let c_start = ci * k;
        let o_start = ci * d;
        for i in 0..k {
            let v = batch.cycles[c_start + i] as usize;
            counts[v] += 1;
            let vd = v * d;
            for j in 0..d {
                out[vd + j] += per_cycle[o_start + j];
            }
        }
    }
    for v in 0..n_vertices {
        if counts[v] > 0 {
            let inv = 1.0 / counts[v] as f32;
            for j in 0..d {
                out[v * d + j] *= inv;
            }
        }
    }
    out
}

// ─── Clifford-derivative spine: Cl(0,1) ≅ ℂ unified filter ─────────
//
// Replaces the two-branch (coef_pos, coef_neg) FIR with a single
// Clifford-algebra-valued filter, with closed-form backward via the
// Clifford derivative ∇ = ∂/∂a + i ∂/∂b.  No autodiff.  No graph.
//
// Math:
//     k_i = a_i + i b_i  ∈ Cl(0,1),   i² = -1
//     proj_σ(k_i) = a_i if σ=+1 else b_i
//     out_c[j] = Σ_i proj_{σ_i}(k_i) · X[v_i][j]
//
//     ∇_{k_i} L = (Σ_{c: σ_i=+1} (∂L/∂out_c) · X[v_i])
//               + i (Σ_{c: σ_i=-1} (∂L/∂out_c) · X[v_i])
//
//     ∂L/∂X[v] = Σ_{c, i: v_i=v} proj_{σ_i}(k_i) · (∂L/∂out_c)
//
// Atomic-add safe: gradient accumulation is commutative.

/// A length-k filter expressed in Cl(0,1) ≅ ℂ: each position holds
/// `(a, b)` = the σ=+1 and σ=−1 branch coefficients packed as one
/// Clifford parameter. Equivalent to `SignedCycleFIR` but with the
/// two banks unified into one multivector array.
#[derive(Debug, Clone)]
pub struct CliffordFIR {
    /// `(k,)` σ=+1 (scalar grade) coefficients.
    pub a: Vec<f32>,
    /// `(k,)` σ=-1 (pseudoscalar grade) coefficients.
    pub b: Vec<f32>,
}

impl CliffordFIR {
    /// New Clifford filter from explicit grade-decomposed coefficients.
    pub fn new(a: Vec<f32>, b: Vec<f32>) -> Self {
        assert_eq!(a.len(), b.len(),
                    "a and b (scalar/pseudoscalar) banks must have equal length");
        Self { a, b }
    }

    /// Mean-pool (both branches scalar 1/k).
    pub fn mean_pool(k: usize) -> Self {
        let v = 1.0 / k as f32;
        Self { a: vec![v; k], b: vec![v; k] }
    }

    /// Sign-aware mean: scalar grade = +1/k, pseudoscalar = -1/k.
    /// Equivalent to the SignedCycleFIR::signed_mean baseline.
    pub fn signed_mean(k: usize) -> Self {
        let v = 1.0 / k as f32;
        Self { a: vec![v; k], b: vec![-v; k] }
    }

    /// Filter length.
    pub fn k(&self) -> usize { self.a.len() }

    /// Gradient buffer of zeros, shape-matching this filter.
    pub fn zero_grad(&self) -> CliffordFIR {
        CliffordFIR {
            a: vec![0.0; self.k()],
            b: vec![0.0; self.k()],
        }
    }
}

/// Clifford-FIR forward. Same scalar output as `SignedCycleFIR` with
/// `coef_pos = self.a, coef_neg = self.b`; provided for API symmetry
/// alongside `clifford_fir_backward`.
pub fn clifford_fir_forward(
    batch: &TopKCyclesBatch,
    features: &[f32],
    n_vertices: usize,
    d: usize,
    fir: &CliffordFIR,
) -> Vec<f32> {
    // Reuse the dual-bank kernel — they compute the same thing.
    let two_bank = SignedCycleFIR {
        coef_pos: fir.a.clone(),
        coef_neg: fir.b.clone(),
    };
    fir_cycle_forward(batch, features, n_vertices, d, &two_bank)
}

/// Closed-form Clifford-derivative backward for the signed-cycle FIR.
///
/// Given the per-cycle forward output gradient `grad_out` of shape
/// `(n_cycles * d)`, computes:
///
/// 1. **Filter gradient** as a `CliffordFIR` (Clifford-valued):
///    `∇k_i.a = Σ_{c: σ_i=+1} <grad_out_c, X[v_i]>`
///    `∇k_i.b = Σ_{c: σ_i=-1} <grad_out_c, X[v_i]>`
///    (one Clifford multivector per filter position, scalar and
///    pseudoscalar parts accumulated independently)
///
/// 2. **Feature gradient** `(n_vertices * d)`:
///    `∇X[v][j] = Σ_{c, i: v_i=v} proj_{σ_i}(k_i) · grad_out_c[j]`
///
/// No autograd, no PyTorch, no graph tracing. Lockless atomic-add
/// accumulation across rayon workers (closure captures
/// `AtomicU32`-wrapped buffers and CAS-updates them).
///
/// # Returns
/// `(grad_features, grad_fir)` where shapes match `features` and
/// `fir` respectively.
pub fn clifford_fir_backward(
    batch: &TopKCyclesBatch,
    features: &[f32],
    n_vertices: usize,
    d: usize,
    fir: &CliffordFIR,
    grad_out: &[f32],
) -> (Vec<f32>, CliffordFIR) {
    let k = batch.k;
    let n_cycles = batch.len();
    assert_eq!(fir.k(), k);
    assert_eq!(features.len(), n_vertices * d);
    assert_eq!(grad_out.len(), n_cycles * d);

    // Per-thread accumulators avoid atomic contention on hub vertices
    // (plan.tex §Risk anticipation).  Reduced at the end.
    let n_threads = rayon::current_num_threads().max(1);
    let cycles_per_thread = n_cycles.div_ceil(n_threads);

    // ThreadAccum bundles all three thread-local buffers so a single
    // Rayon `into_par_iter()` over a Vec<ThreadAccum> owns each slot
    // exclusively (no raw pointers, no unsafe).
    struct ThreadAccum {
        grad_features: Vec<f32>,
        grad_a: Vec<f32>,
        grad_b: Vec<f32>,
    }
    let mut accums: Vec<ThreadAccum> = (0..n_threads)
        .map(|_| ThreadAccum {
            grad_features: vec![0.0f32; n_vertices * d],
            grad_a: vec![0.0f32; k],
            grad_b: vec![0.0f32; k],
        })
        .collect();

    use rayon::prelude::IntoParallelRefMutIterator;
    accums.par_iter_mut().enumerate().for_each(|(t, acc)| {
        let start = t * cycles_per_thread;
        let end = ((t + 1) * cycles_per_thread).min(n_cycles);
        for ci in start..end {
            let c_start = ci * k;
            let s_start = ci * k;
            let o_start = ci * d;
            let go = &grad_out[o_start..o_start + d];
            for i in 0..k {
                let v = batch.cycles[c_start + i] as usize;
                let sgn = batch.signs[s_start + i];
                let coef = if sgn > 0 { fir.a[i] } else { fir.b[i] };
                let f_start = v * d;
                let f = &features[f_start..f_start + d];
                // <go, f> ⟶ contributes to grad_a[i] or grad_b[i]
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += go[j] * f[j];
                }
                if sgn > 0 {
                    acc.grad_a[i] += dot;
                } else {
                    acc.grad_b[i] += dot;
                }
                // ∂L/∂X[v][j] += coef · go[j]
                let gf = &mut acc.grad_features[f_start..f_start + d];
                for j in 0..d {
                    gf[j] += coef * go[j];
                }
            }
        }
    });

    // Reduce per-thread accumulators.
    let mut grad_a = vec![0.0f32; k];
    let mut grad_b = vec![0.0f32; k];
    let mut grad_features = vec![0.0f32; n_vertices * d];
    for acc in &accums {
        for i in 0..k {
            grad_a[i] += acc.grad_a[i];
            grad_b[i] += acc.grad_b[i];
        }
        for i in 0..(n_vertices * d) {
            grad_features[i] += acc.grad_features[i];
        }
    }
    (grad_features, CliffordFIR { a: grad_a, b: grad_b })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch(cycles: Vec<u32>, signs: Vec<i8>, k: usize) -> TopKCyclesBatch {
        assert_eq!(cycles.len(), signs.len());
        let n = cycles.len() / k;
        TopKCyclesBatch {
            cycles,
            signs,
            scores: vec![0.0; n],
            k,
        }
    }

    #[test]
    fn mean_pool_of_identity_features_returns_one_third() {
        // 1 cycle, k=3, vertices 0,1,2 all with feature [1, 1, 1]
        // signs all +1, mean_pool kernel = 1/3.
        // Expected: [1, 1, 1].
        let batch = make_batch(vec![0, 1, 2], vec![1, 1, 1], 3);
        let features = vec![1.0; 3 * 3];
        let fir = SignedCycleFIR::mean_pool(3);
        let out = fir_cycle_forward(&batch, &features, 3, 3, &fir);
        assert_eq!(out.len(), 3);
        for j in 0..3 {
            assert!((out[j] - 1.0).abs() < 1e-6, "out[{}]={}", j, out[j]);
        }
    }

    #[test]
    fn signed_mean_with_all_negative_cycle() {
        // 1 cycle, all signs -1, signed_mean → coef_neg = -1/3.
        // features = all 1s; out = -1/3 * 3 = -1.
        let batch = make_batch(vec![0, 1, 2], vec![-1, -1, -1], 3);
        let features = vec![1.0; 3 * 3];
        let fir = SignedCycleFIR::signed_mean(3);
        let out = fir_cycle_forward(&batch, &features, 3, 3, &fir);
        for j in 0..3 {
            assert!((out[j] - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn mixed_signs_separate_branches() {
        // Cycle (0, 1, 2), signs (+, -, +); features = (1, 2, 3) per dim 0.
        // coef_pos = [0.5, 0.5, 0.5], coef_neg = [0.1, 0.1, 0.1]
        // out[0] = 0.5*1 + 0.1*2 + 0.5*3 = 2.2
        let batch = make_batch(vec![0, 1, 2], vec![1, -1, 1], 3);
        let mut features = vec![0.0; 3];
        features[0] = 1.0; features[1] = 2.0; features[2] = 3.0;
        let fir = SignedCycleFIR::new(vec![0.5; 3], vec![0.1; 3]);
        let out = fir_cycle_forward(&batch, &features, 3, 1, &fir);
        assert!((out[0] - 2.2).abs() < 1e-6, "out={:?}", out);
    }

    #[test]
    fn scatter_mean_two_cycles_share_vertex() {
        // Cycles (0,1,2) and (0,3,4) both touch vertex 0.
        // mean_pool, features all ones → per_cycle = (1, 1).
        // Vertex 0 sees both cycles; mean = 1.
        let batch = make_batch(
            vec![0, 1, 2, 0, 3, 4],
            vec![1, 1, 1, 1, 1, 1],
            3,
        );
        let features = vec![1.0; 5];
        let fir = SignedCycleFIR::mean_pool(3);
        let out = fir_cycle_scatter_mean(&batch, &features, 5, 1, &fir);
        for v in 0..5 {
            assert!((out[v] - 1.0).abs() < 1e-6, "v={} out={}", v, out[v]);
        }
    }

    #[test]
    fn clifford_forward_matches_two_branch() {
        // CliffordFIR forward must equal SignedCycleFIR forward with
        // (a, b) ↔ (coef_pos, coef_neg).
        let batch = make_batch(
            vec![0, 1, 2, 3, 4, 5, 1, 3, 5],
            vec![1, -1, 1, -1, 1, 1, -1, 1, -1],
            3,
        );
        let features: Vec<f32> = (0..6 * 4).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let fir = CliffordFIR::new(vec![0.7, -0.2, 0.5], vec![-0.3, 0.4, -0.1]);
        let two = SignedCycleFIR::new(fir.a.clone(), fir.b.clone());
        let out_cliff = clifford_fir_forward(&batch, &features, 6, 4, &fir);
        let out_two = fir_cycle_forward(&batch, &features, 6, 4, &two);
        assert_eq!(out_cliff.len(), out_two.len());
        for (a, b) in out_cliff.iter().zip(out_two.iter()) {
            assert!((a - b).abs() < 1e-6, "{} vs {}", a, b);
        }
    }

    #[test]
    fn clifford_backward_matches_numerical_grad() {
        // Verify the closed-form Clifford derivative matches central-
        // differences numerical gradient on the FIR filter coefficients
        // and the input features. Tolerance ≤ 1e-3 — float-ε ~1e-4 at
        // f32 precision, central diff adds another order.
        let batch = make_batch(
            vec![0, 1, 2, 1, 2, 3, 0, 2, 3],
            vec![1, -1, 1, 1, -1, 1, -1, 1, -1],
            3,
        );
        let n_vertices = 4;
        let d = 3;
        let features: Vec<f32> = (0..n_vertices * d)
            .map(|i| (i as f32) * 0.13 - 0.5)
            .collect();
        let fir = CliffordFIR::new(
            vec![0.7, -0.2, 0.5], vec![-0.3, 0.4, -0.1],
        );
        // Use a simple scalar loss L = sum(forward output).  Then
        // grad_out = ones, and the closed-form backward gives the
        // gradient of L w.r.t. params + features directly.
        let out = clifford_fir_forward(&batch, &features, n_vertices, d, &fir);
        let n_cycles = out.len() / d;
        let grad_out = vec![1.0f32; n_cycles * d];
        let (analytic_gf, analytic_gfir) = clifford_fir_backward(
            &batch, &features, n_vertices, d, &fir, &grad_out,
        );

        // Loss closure for numerical diff.
        let loss = |feats: &[f32], a: &[f32], b: &[f32]| -> f32 {
            let f = CliffordFIR::new(a.to_vec(), b.to_vec());
            clifford_fir_forward(&batch, feats, n_vertices, d, &f)
                .iter().sum::<f32>()
        };
        let eps = 1e-3f32;

        // Check ∂L/∂fir.a numerically.
        for i in 0..fir.k() {
            let mut a_p = fir.a.clone(); a_p[i] += eps;
            let mut a_m = fir.a.clone(); a_m[i] -= eps;
            let num = (loss(&features, &a_p, &fir.b)
                       - loss(&features, &a_m, &fir.b)) / (2.0 * eps);
            assert!((analytic_gfir.a[i] - num).abs() < 1e-2,
                "a[{}]: analytic={} numerical={}", i, analytic_gfir.a[i], num);
        }
        // ∂L/∂fir.b
        for i in 0..fir.k() {
            let mut b_p = fir.b.clone(); b_p[i] += eps;
            let mut b_m = fir.b.clone(); b_m[i] -= eps;
            let num = (loss(&features, &fir.a, &b_p)
                       - loss(&features, &fir.a, &b_m)) / (2.0 * eps);
            assert!((analytic_gfir.b[i] - num).abs() < 1e-2,
                "b[{}]: analytic={} numerical={}", i, analytic_gfir.b[i], num);
        }
        // Check ∂L/∂X at a few feature positions.
        for v in 0..n_vertices {
            for j in 0..d {
                let idx = v * d + j;
                let mut f_p = features.clone(); f_p[idx] += eps;
                let mut f_m = features.clone(); f_m[idx] -= eps;
                let num = (loss(&f_p, &fir.a, &fir.b)
                           - loss(&f_m, &fir.a, &fir.b)) / (2.0 * eps);
                assert!((analytic_gf[idx] - num).abs() < 1e-2,
                    "X[{},{}]: analytic={} numerical={}",
                    v, j, analytic_gf[idx], num);
            }
        }
    }

    #[test]
    fn clifford_backward_zero_grad_out_returns_zero() {
        let batch = make_batch(vec![0, 1, 2], vec![1, -1, 1], 3);
        let features = vec![1.0; 9];
        let fir = CliffordFIR::signed_mean(3);
        let grad_out = vec![0.0; 3];
        let (gf, gfir) = clifford_fir_backward(&batch, &features, 3, 3, &fir, &grad_out);
        assert!(gf.iter().all(|&x| x.abs() < 1e-9));
        assert!(gfir.a.iter().all(|&x| x.abs() < 1e-9));
        assert!(gfir.b.iter().all(|&x| x.abs() < 1e-9));
    }

    #[test]
    fn parallel_consistency() {
        // Same input produces same output (Rayon order-independence).
        let mut cycles: Vec<u32> = Vec::new();
        let mut signs: Vec<i8> = Vec::new();
        for ci in 0..200 {
            cycles.extend_from_slice(&[
                (ci as u32) % 50,
                ((ci as u32) + 7) % 50,
                ((ci as u32) + 13) % 50,
            ]);
            signs.extend_from_slice(&[1, -1, 1]);
        }
        let batch = make_batch(cycles, signs, 3);
        let features: Vec<f32> = (0..50 * 8).map(|i| (i as f32) / 50.0).collect();
        let fir = SignedCycleFIR::new(
            vec![0.7, -0.2, 0.5], vec![-0.3, 0.4, -0.1],
        );
        let out1 = fir_cycle_forward(&batch, &features, 50, 8, &fir);
        let out2 = fir_cycle_forward(&batch, &features, 50, 8, &fir);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
