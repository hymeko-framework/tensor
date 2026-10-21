//! Matrix-based Rényi-α=2 estimators and count-based Shannon summaries.
//!
//! Rényi quantities (Giraldo, Rao & Principe 2015) use **traces only** (no eigendecomposition, no density estimation), which is what
//! makes them affordable as a per-step signal. The order-2 case uses `tr(K²) = Σ K_ij²`. These are pure,
//! `std`-only functions on flat row-major slices; they carry no notion of a network — the caller supplies the
//! activation matrix `A` (flat `m×d`) and, where relevant, a precomputed label/other Gram `K_Y` (flat `m×m`).

/// Activation kernel for the matrix-based MI. `Linear` = `A Aᵀ` (a linear dependence measure). `Cheby(kf)` uses a
/// Chebyshev–Catmull-Rom feature map `φ(a) = [T₁(ā),…,T_{kf}(ā)]` per channel (`ā = clamp(a,−1,1)`), giving a
/// polynomial kernel `K_A = ΦΦᵀ` with NO bandwidth hyperparameter. `kf` is the feature-map order (degree).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiKernel {
    Linear,
    Cheby(usize),
}

/// Linear (degree-1 polynomial) Gram `K[i][j] = ⟨row_i, row_j⟩` for `m` rows of dim `d` (flat `m·d` → `m·m`).
/// Symmetric; the no-bandwidth kernel the matrix-based Rényi entropy uses.
pub fn linear_gram(a: &[f32], m: usize, d: usize) -> Vec<f32> {
    let mut k = vec![0.0f32; m * m];
    for i in 0..m {
        for j in i..m {
            let s: f32 = (0..d).map(|r| a[i * d + r] * a[j * d + r]).sum();
            k[i * m + j] = s;
            k[j * m + i] = s;
        }
    }
    k
}

/// Matrix-based Rényi-α=2 entropy `S₂(K) = −log₂(tr(K̂²))`, `K̂ = K/tr(K)` (traces only, no eigendecomposition).
/// For a linear Gram this is `log₂` of the participation ratio (effective rank) of the rows.
///
/// # Postconditions
/// `≥ 0` and finite for any PSD `K` with positive trace (traces floored by `1e-12`).
pub fn s2_entropy(k: &[f32], m: usize) -> f32 {
    let z: f32 = (0..m).map(|i| k[i * m + i]).sum::<f32>() + 1e-12; // tr(K)
    let t: f32 = k.iter().map(|v| v * v).sum::<f32>() + 1e-12; // tr(K²) = Σ K_ij²
    (2.0 * z.ln() - t.ln()) / std::f32::consts::LN_2
}

/// Shared core: given the activation Gram `ka` and label Gram `ky`, the Rényi-2 MI `I = S₂(K_A)+S₂(K_Y)−S₂(K_A∘K_Y)`
/// and the `m×m` coefficient `coeff = ∂S₂(K_A)/∂K_A − ∂S₂(K_A∘K_Y)/∂K_A` — to be contracted with the kernel's
/// `∂K_A/∂A`. Joint entropy via the Hadamard-product Gram `K_A∘K_Y`, so the label enters the metric.
pub fn mi_coeff(ka: &[f32], ky: &[f32], m: usize) -> (f32, Vec<f32>) {
    let mm: Vec<f32> = ka.iter().zip(ky).map(|(k, y)| k * y).collect(); // joint K_A∘K_Y
    let tr = |k: &[f32]| -> f32 { (0..m).map(|i| k[i * m + i]).sum::<f32>() + 1e-12 };
    let fro2 = |k: &[f32]| -> f32 { k.iter().map(|v| v * v).sum::<f32>() + 1e-12 }; // tr(K²)=Σ K_ij²
    let (za, ta) = (tr(ka), fro2(ka));
    let (zm, tm) = (tr(&mm), fro2(&mm));
    let (zy, ty) = (tr(ky), fro2(ky));
    let ln2 = std::f32::consts::LN_2;
    let s2 = |z: f32, t: f32| (2.0 * z.ln() - t.ln()) / ln2; // −log₂(tr(K²)/tr(K)²)
    let mi = s2(za, ta) + s2(zy, ty) - s2(zm, tm);
    // ∂S₂(K)/∂K_ij = (2/Z)·δ_ij − (2/T)·K_ij  (S₂(K_Y) ⟂ A ⇒ omitted)
    let mut coeff = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let ga = if i == j { 2.0 / za } else { 0.0 } - 2.0 / ta * ka[i * m + j];
            let gm = if i == j { 2.0 / zm } else { 0.0 } - 2.0 / tm * mm[i * m + j];
            coeff[i * m + j] = (ga - gm * ky[i * m + j]) / ln2;
        }
    }
    (mi, coeff)
}

/// Matrix-based Rényi-α=2 mutual information `I(A;Y)` between a minibatch of activations `a` (flat `m×d`) and a
/// label Gram `ky` (flat `m×m`), plus `∂I/∂A` (flat `m×d`) under the chosen `kernel`. **Ascending** `∂I/∂A` makes
/// the activations more informative about the labels — a per-sample, task-directional signal.
///
/// # Preconditions
/// `a.len() == m·d`, `ky.len() == m·m`, `m ≥ 2`, `d ≥ 1`.
/// # Postconditions
/// Returns `(mi, grad)` with `grad.len() == m·d`; all finite (traces floored by `1e-12`).
pub fn renyi2_mi_grad(
    a: &[f32],
    ky: &[f32],
    m: usize,
    d: usize,
    kernel: MiKernel,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(a.len(), m * d);
    debug_assert_eq!(ky.len(), m * m);
    match kernel {
        MiKernel::Linear => {
            let ka = linear_gram(a, m, d); // K_A = A Aᵀ
            let (mi, coeff) = mi_coeff(&ka, ky, m);
            // ∂K_ij/∂A linear ⇒ ∂I/∂A = 2·coeff·A
            let mut grad = vec![0.0f32; m * d];
            for i in 0..m {
                for r in 0..d {
                    let s: f32 = (0..m).map(|j| coeff[i * m + j] * a[j * d + r]).sum();
                    grad[i * d + r] = 2.0 * s;
                }
            }
            (mi, grad)
        }
        MiKernel::Cheby(kf) => {
            let fd = d * kf; // expanded feature dim
            let (phi, dphi) = cheby_features(a, m, d, kf);
            let ka = linear_gram(&phi, m, fd); // K_A = Φ Φᵀ
            let (mi, coeff) = mi_coeff(&ka, ky, m);
            // ∂I/∂Φ = 2·coeff·Φ ; chain through the basis: ∂I/∂a[i,dim] = Σ_t ∂I/∂Φ[i,dim·kf+t]·φ′_t
            let mut grad = vec![0.0f32; m * d];
            for i in 0..m {
                for dim in 0..d {
                    let mut acc = 0.0f32;
                    for t in 0..kf {
                        let c = dim * kf + t;
                        let gphi: f32 = 2.0
                            * (0..m)
                                .map(|j| coeff[i * m + j] * phi[j * fd + c])
                                .sum::<f32>();
                        acc += gphi * dphi[i * fd + c];
                    }
                    grad[i * d + dim] = acc;
                }
            }
            (mi, grad)
        }
    }
}

/// Chebyshev-CR basis features `φ_t(ā)=T_{t+1}(ā)` and their derivatives `φ′_t=T_{t+1}′(ā)` for `t∈0..kf`
/// (i.e. `T₁…T_{kf}`, skipping the constant `T₀`), with `ā=clamp(a,−1,1)` and the derivative masked to `0`
/// outside `[−1,1]`. Returns two flat `m×(d·kf)` matrices (features, derivatives).
pub fn cheby_features(a: &[f32], m: usize, d: usize, kf: usize) -> (Vec<f32>, Vec<f32>) {
    let fd = d * kf;
    let mut phi = vec![0.0f32; m * fd];
    let mut dphi = vec![0.0f32; m * fd];
    for i in 0..m {
        for dim in 0..d {
            let xv = a[i * d + dim];
            let xc = xv.clamp(-1.0, 1.0);
            let inside = (-1.0..=1.0).contains(&xv);
            let base = i * fd + dim * kf;
            // Chebyshev recurrence T_{k+1}=2x·T_k−T_{k-1}, derivative T′_{k+1}=2T_k+2x·T′_k−T′_{k-1}.
            let (mut tprev, mut tcur) = (1.0f32, xc); // T₀, T₁
            let (mut dprev, mut dcur) = (0.0f32, 1.0f32); // T₀′, T₁′
            phi[base] = tcur; // t=0 ⇒ T₁
            dphi[base] = if inside { dcur } else { 0.0 };
            for t in 2..=kf {
                let tnext = 2.0 * xc * tcur - tprev;
                let dnext = 2.0 * tcur + 2.0 * xc * dcur - dprev;
                phi[base + t - 1] = tnext;
                dphi[base + t - 1] = if inside { dnext } else { 0.0 };
                tprev = tcur;
                tcur = tnext;
                dprev = dcur;
                dcur = dnext;
            }
        }
    }
    (phi, dphi)
}

/// Stable rank `‖A‖_F² / σ_max(A)²` of an `n×d` matrix — a cheap effective-rank / rank-collapse proxy in `[1, d]`
/// (→1 under total collapse, →d when isotropic). `σ_max² = λ_max(AᵀA)` via power iteration on the `d×d` Gram;
/// `‖A‖_F² = trace(AᵀA)`. Deterministic (fixed unit-vector init).
pub fn stable_rank(a: &[f32], d: usize) -> f32 {
    if a.is_empty() || d == 0 {
        return 1.0;
    }
    let mut g = vec![0.0f32; d * d];
    for row in a.chunks(d) {
        for i in 0..d {
            for j in 0..d {
                g[i * d + j] += row[i] * row[j];
            }
        }
    }
    let frob2: f32 = a.iter().map(|v| v * v).sum();
    let mut v = vec![1.0f32 / (d as f32).sqrt(); d];
    for _ in 0..60 {
        let mut gv = vec![0.0f32; d];
        for i in 0..d {
            gv[i] = (0..d).map(|j| g[i * d + j] * v[j]).sum();
        }
        let nrm = gv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for i in 0..d {
            v[i] = gv[i] / nrm;
        }
    }
    // Rayleigh quotient λ_max ≈ vᵀ G v.
    let mut gv = vec![0.0f32; d];
    for i in 0..d {
        gv[i] = (0..d).map(|j| g[i * d + j] * v[j]).sum();
    }
    let lam: f32 = (0..d).map(|i| v[i] * gv[i]).sum::<f32>().max(1e-12);
    (frob2 / lam).clamp(1.0, d as f32)
}

/// Shannon entropy of a nonnegative count distribution, in bits.
///
/// This descriptor is an ordering heuristic, not a certified search bound.
/// Zero counts contribute nothing; empty distributions have entropy zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShannonEntropy {
    total: f64,
    weighted_log: f64,
}
impl ShannonEntropy {
    /// Summarize counts without retaining or normalizing an input vector.
    pub fn from_counts(counts: impl IntoIterator<Item = usize>) -> Self {
        counts
            .into_iter()
            .filter(|&n| n != 0)
            .fold(Self::default(), |mut state, n| {
                let n = n as f64;
                state.total += n;
                state.weighted_log += n * n.log2();
                state
            })
    }
    /// `log2(total) - sum(count * log2(count))/total`, clamped at zero
    /// against cancellation. This does not certify a pruning inequality.
    pub fn bits(self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            (self.total.log2() - self.weighted_log / self.total).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in `[-1,1]` (a tiny LCG) — no `rand` dependency in the test.
    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn linear_gram_is_symmetric() {
        let (m, d) = (5, 3);
        let a = det(m * d, 7);
        let k = linear_gram(&a, m, d);
        for i in 0..m {
            for j in 0..m {
                assert!((k[i * m + j] - k[j * m + i]).abs() < 1e-6, "Gram symmetric");
            }
        }
    }

    #[test]
    fn s2_entropy_rank_one_zero_orthogonal_log2m() {
        // rank-1 (parallel rows) ⇒ participation ratio 1 ⇒ S₂≈0; orthogonal rows (K=I_m) ⇒ S₂=log₂(m).
        let m = 8;
        let base = [1.0f32, 2.0, 3.0];
        let rank1: Vec<f32> = (0..m)
            .flat_map(|i| base.iter().map(move |b| (i as f32 + 1.0) * b))
            .collect();
        assert!(s2_entropy(&linear_gram(&rank1, m, 3), m).abs() < 1e-3);
        let mut ortho = vec![0.0f32; m * m];
        for i in 0..m {
            ortho[i * m + i] = 1.0;
        }
        assert!((s2_entropy(&linear_gram(&ortho, m, m), m) - (m as f32).log2()).abs() < 1e-3);
    }

    #[test]
    fn renyi2_mi_grad_matches_finite_difference() {
        // Analytic ∂I/∂A must match a central finite difference, for both kernels.
        let (m, d) = (6usize, 3usize);
        let y = [0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ky: Vec<f32> = (0..m * m)
            .map(|k| f32::from((y[k / m] > 0.5) == (y[k % m] > 0.5)))
            .collect();
        for kernel in [MiKernel::Linear, MiKernel::Cheby(4)] {
            let a = det(m * d, 21);
            let (_mi, grad) = renyi2_mi_grad(&a, &ky, m, d, kernel);
            let eps = 1e-3f32;
            let mut maxdiff = 0.0f32;
            for idx in 0..m * d {
                let (mut ap, mut am) = (a.clone(), a.clone());
                ap[idx] += eps;
                am[idx] -= eps;
                let fd = (renyi2_mi_grad(&ap, &ky, m, d, kernel).0
                    - renyi2_mi_grad(&am, &ky, m, d, kernel).0)
                    / (2.0 * eps);
                maxdiff = maxdiff.max((fd - grad[idx]).abs());
            }
            assert!(
                maxdiff < 2e-2,
                "analytic == finite-diff for {kernel:?} (max {maxdiff})"
            );
        }
    }

    #[test]
    fn stable_rank_bounds() {
        // rank-1 rows ⇒ stable rank ≈ 1; isotropic (identity-ish) rows ⇒ near d.
        let d = 4;
        let rank1: Vec<f32> = (0..6)
            .flat_map(|i| (0..d).map(move |r| (i + 1) as f32 * (r + 1) as f32))
            .collect();
        assert!((stable_rank(&rank1, d) - 1.0).abs() < 1e-2, "rank-1 ⇒ ~1");
        let mut iso = vec![0.0f32; d * d];
        for i in 0..d {
            iso[i * d + i] = 1.0;
        }
        assert!(
            (stable_rank(&iso, d) - d as f32).abs() < 1e-2,
            "isotropic ⇒ ~d"
        );
    }
}
