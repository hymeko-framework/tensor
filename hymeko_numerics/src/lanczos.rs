//! **Lanczos** for the extreme eigenpairs of a symmetric operator (plan
//! 2026-10-19-numerics-optimizers, O7, Amendment 9).
//!
//! From a start vector, `m` steps build an orthonormal Krylov basis `Q = [q₁ … q_m]` and the
//! tridiagonal `T = Qᵀ A Q` (diagonal `α`, off-diagonal `β`). Every new direction is
//! reorthogonalised against the whole basis, twice ("twice is enough"), so the basis stays
//! orthonormal to rounding and no spurious copies of converged eigenvalues appear: the price is
//! `O(n m²)` work and `n m` memory, the right trade for the tens of steps an extreme eigenvalue needs.
//! `T`'s eigenpairs `(θ_j, s_j)` -- [`tridiagonal_eigh`], implicit-shift QL -- give the Ritz pairs
//! `(θ_j, Q s_j)` and their residual bounds `‖A x_j − θ_j x_j‖ = |β_m s_{m,j}|`.
//!
//! The run stops early on an invariant subspace: when the new direction's norm falls below
//! `tolerance` times `‖A q_j‖`, the Ritz values are eigenvalues of `A` and every bound is zero.

use crate::{dot, norm, Operator};

/// Lanczos's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lanczos {
    /// Steps at most: the Krylov dimension.
    pub steps: usize,
    /// The relative size of a new direction below which the subspace is taken as invariant.
    pub tolerance: f64,
}

/// The Ritz pairs of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Ritz {
    /// The Ritz values, ascending.
    pub values: Vec<f64>,
    /// The Ritz vectors, unit, in the values' order.
    pub vectors: Vec<Vec<f64>>,
    /// Each pair's residual `‖A x − θ x‖`, as `|β_m s_{m,j}|`.
    pub residuals: Vec<f64>,
    /// Steps taken: the Krylov dimension reached.
    pub steps: usize,
}

/// **Lanczos with full reorthogonalisation** on the symmetric `op` from `start`.
///
/// # Preconditions
/// `op` is symmetric; `start.len() == op.dim()`, `start` not zero; `settings.steps ≥ 1`.
///
/// # Postconditions
/// `values` ascending, `vectors` orthonormal to rounding, each of length `op.dim()`; at most
/// `min(steps, dim)` pairs.
///
/// # Panics
/// When `start` is zero or of another length than `op.dim()`.
pub fn lanczos<O: Operator>(op: &O, start: &[f64], settings: Lanczos) -> Ritz {
    let n = op.dim();
    assert_eq!(start.len(), n, "one start entry per dimension");
    let start_norm = norm(start);
    assert!(start_norm > 0.0, "a zero start vector spans nothing");
    let limit = settings.steps.min(n).max(1);
    let mut basis: Vec<Vec<f64>> = vec![start.iter().map(|v| v / start_norm).collect()];
    let (mut alpha, mut beta) = (Vec::with_capacity(limit), Vec::with_capacity(limit));
    let mut w = vec![0.0; n];
    let mut last_beta = 0.0;
    for j in 0..limit {
        op.apply(&basis[j], &mut w);
        let scale = norm(&w);
        let a = dot(&basis[j], &w);
        alpha.push(a);
        for _ in 0..2 {
            for q in &basis {
                let projection = dot(q, &w);
                for (wi, qi) in w.iter_mut().zip(q) {
                    *wi -= projection * qi;
                }
            }
        }
        let b = norm(&w);
        if b <= settings.tolerance * scale.max(f64::MIN_POSITIVE) || j + 1 == limit {
            last_beta = if b <= settings.tolerance * scale.max(f64::MIN_POSITIVE) {
                0.0
            } else {
                b
            };
            break;
        }
        beta.push(b);
        basis.push(w.iter().map(|v| v / b).collect());
    }
    let m = alpha.len();
    let (values, s) = tridiagonal_eigh(&alpha, &beta[..m - 1]);
    let vectors = (0..m)
        .map(|k| {
            let mut x = vec![0.0; n];
            for (i, q) in basis.iter().take(m).enumerate() {
                let c = s[i * m + k];
                for (xi, qi) in x.iter_mut().zip(q) {
                    *xi += c * qi;
                }
            }
            x
        })
        .collect();
    let residuals = (0..m)
        .map(|k| (last_beta * s[(m - 1) * m + k]).abs())
        .collect();
    Ritz {
        values,
        vectors,
        residuals,
        steps: m,
    }
}

/// **The eigenpairs of a symmetric tridiagonal matrix** with diagonal `diagonal` and off-diagonal
/// `off` (`off[i]` between rows `i` and `i + 1`), by the implicit QL method with Wilkinson shifts.
///
/// # Preconditions
/// `off.len() + 1 == diagonal.len()`, or both empty.
///
/// # Postconditions
/// `(values, vectors)`: the eigenvalues ascending, and the row-major `m × m` matrix whose column `k`
/// is the unit eigenvector of `values[k]`.
///
/// # Panics
/// When the lengths disagree, or a value fails to converge in 60 QL sweeps (not seen on finite
/// input: the method converges cubically).
#[must_use]
pub fn tridiagonal_eigh(diagonal: &[f64], off: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = diagonal.len();
    assert!(
        off.len() + 1 == m || (m == 0 && off.is_empty()),
        "off-diagonal of length m - 1"
    );
    let mut d = diagonal.to_vec();
    let mut e: Vec<f64> = off.iter().copied().chain(core::iter::once(0.0)).collect();
    let mut z = vec![0.0; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }
    for l in 0..m {
        let mut sweeps = 0;
        loop {
            // The first negligible off-diagonal at or past `l` splits the matrix there.
            let split = (l..m.saturating_sub(1))
                .find(|&k| e[k].abs() <= f64::EPSILON * (d[k].abs() + d[k + 1].abs()))
                .unwrap_or(m - 1);
            if split == l {
                break;
            }
            sweeps += 1;
            assert!(sweeps <= 60, "QL did not converge at row {l}");
            ql_sweep(&mut d, &mut e, &mut z, m, l, split);
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let values = order.iter().map(|&k| d[k]).collect();
    let mut vectors = vec![0.0; m * m];
    for (new, &old) in order.iter().enumerate() {
        for i in 0..m {
            vectors[i * m + new] = z[i * m + old];
        }
    }
    (values, vectors)
}

/// One implicit QL sweep with a Wilkinson shift over rows `l..=split`, the rotations accumulated
/// into `z`'s columns.
fn ql_sweep(d: &mut [f64], e: &mut [f64], z: &mut [f64], m: usize, l: usize, split: usize) {
    let g0 = (d[l + 1] - d[l]) / (2.0 * e[l]);
    let r0 = g0.hypot(1.0);
    let mut g = d[split] - d[l] + e[l] / (g0 + r0.copysign(g0));
    let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
    for i in (l..split).rev() {
        let f = s * e[i];
        let b = c * e[i];
        let r = f.hypot(g);
        e[i + 1] = r;
        if r == 0.0 {
            // An exact underflow: deflate and let the caller restart the sweep.
            d[i + 1] -= p;
            e[split] = 0.0;
            return;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        let t = (d[i] - g) * s + 2.0 * c * b;
        p = s * t;
        d[i + 1] = g + p;
        g = c * t - b;
        for k in 0..m {
            let zk1 = z[k * m + i + 1];
            z[k * m + i + 1] = s * z[k * m + i] + c * zk1;
            z[k * m + i] = c * z[k * m + i] - s * zk1;
        }
    }
    d[l] -= p;
    e[l] = g;
    e[split] = 0.0;
}
