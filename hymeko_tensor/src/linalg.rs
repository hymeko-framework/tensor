//! Generic linear solving over `f32` — deliberately independent of any graph representation.
//!
//! Two solvers, and nothing else:
//!
//! - [`cg_matrix_free`] — conjugate gradients whose only primitive is an **operator closure**
//!   `x ↦ A x`. The matrix is never formed, never assembled and never named.
//! - [`dense_cholesky_solve`] — a dense SPD factorisation for a row-major matrix.
//!
//! They live here rather than beside a graph type because neither knows what a graph is: the first
//! takes a closure, the second takes `&[f32]`. A caller that has a hypergraph operator supplies the
//! closure; a caller that has assembled a dense matrix supplies the slice.
//!
//! # Pinned policies
//!
//! These are contracts, not incidental behaviour — a consumer reproducing a recorded number depends
//! on every one of them:
//!
//! | policy | choice |
//! |---|---|
//! | convergence | `‖r‖₂ < tol`, an **absolute** residual norm on the running CG residual |
//! | relative residual | reported separately as `‖r‖ / ‖b‖`, and as `‖r‖` when `b = 0` |
//! | zero RHS | returns the zero vector, `iters = 0`, residual `0.0` — never iterates |
//! | iteration convention | `iters` counts **completed** iterations, so it is the number of operator applications |
//! | breakdown | `pᵀAp ≤ 0` (indefinite or numerically dead) stops and returns what it has |
//! | accumulation | `f32` throughout, in the order written — no `f64` promotion, so a consumer's recorded values move if this changes |
//! | SPD / ridge | both solvers require SPD; a positive ridge on the diagonal is the caller's job |
//! | factorisation order | column-major-inner `j ≤ i` loops, the classical Cholesky–Banachiewicz order |
//! | failure | a non-positive pivot yields [`CholeskyError::NotPositiveDefinite`] with the offending index — never a panic, never a silent `NaN` |

/// Why a dense Cholesky factorisation could not proceed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CholeskyError {
    /// The leading minor at this index is not positive: the matrix is not SPD.
    NotPositiveDefinite {
        /// Diagonal index whose pivot was `<= 0`.
        index: usize,
    },
    /// `m.len() != n * n`, or `b.len() != n`.
    ShapeMismatch,
}

impl core::fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CholeskyError::NotPositiveDefinite { index } => {
                write!(f, "matrix is not positive definite at diagonal {index}")
            }
            CholeskyError::ShapeMismatch => write!(f, "matrix or right-hand side has a bad shape"),
        }
    }
}

impl std::error::Error for CholeskyError {}

/// Outcome of a matrix-free conjugate-gradient solve.
#[derive(Clone, Debug, PartialEq)]
pub struct CgOutcome {
    /// The solution.
    pub x: Vec<f32>,
    /// Completed iterations — equivalently, operator applications.
    pub iters: usize,
    /// `‖r‖ / ‖b‖`, or `‖r‖` when `b = 0`.
    pub relative_residual: f32,
    /// Whether the absolute residual reached `tol`.
    pub converged: bool,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Conjugate gradients on an SPD operator supplied as a **closure**.
///
/// The operator is never materialised: `apply` is the only thing the solver knows about `A`. That is
/// what makes this usable for a message-passing operator whose whole point is not to be assembled.
///
/// # Preconditions
/// `apply` is linear, symmetric and positive definite on `b.len()` coordinates, and returns a vector
/// of that same length. Starting iterate is zero, so `r₀ = b`.
///
/// # Postconditions
/// `x.len() == b.len()`. `iters <= max_iter`. With `b = 0` the result is the zero vector at
/// `iters = 0`. Convergence is `‖r‖₂ < tol` on the running residual.
///
/// # Panics
/// If `apply` returns a vector of the wrong length.
pub fn cg_matrix_free<F>(mut apply: F, b: &[f32], tol: f32, max_iter: usize) -> CgOutcome
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let n = b.len();
    let nb = dot(b, b).sqrt();
    if nb == 0.0 {
        return CgOutcome {
            x: vec![0.0; n],
            iters: 0,
            relative_residual: 0.0,
            converged: true,
        };
    }
    let mut x = vec![0.0f32; n];
    let mut r = b.to_vec(); // r = b - A·0
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    let mut iters = 0usize;
    let mut converged = rs.sqrt() < tol;
    while iters < max_iter && !converged {
        let ap = apply(&p);
        assert_eq!(ap.len(), n, "the operator must preserve the dimension");
        let denom = dot(&p, &ap);
        if denom <= 0.0 {
            break; // indefinite or numerically dead — stop, do not fabricate a step
        }
        let alpha = rs / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        iters += 1;
        if rs_new.sqrt() < tol {
            converged = true;
            break;
        }
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
    }
    CgOutcome {
        relative_residual: dot(&r, &r).sqrt() / nb,
        x,
        iters,
        converged,
    }
}

/// Dense Cholesky solve of `M x = b` for row-major SPD `M` of order `n`.
///
/// # Preconditions
/// `m.len() == n * n`, `b.len() == n`, `M` symmetric positive definite.
///
/// # Postconditions
/// `x.len() == n`. `n == 0` yields an empty solution.
///
/// # Errors
/// [`CholeskyError::ShapeMismatch`] on a bad shape; [`CholeskyError::NotPositiveDefinite`] with the
/// offending diagonal index if a pivot is not positive.
pub fn dense_cholesky_solve(m: &[f32], b: &[f32], n: usize) -> Result<Vec<f32>, CholeskyError> {
    if m.len() != n * n || b.len() != n {
        return Err(CholeskyError::ShapeMismatch);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i * n + j];
            for kk in 0..j {
                s -= l[i * n + kk] * l[j * n + kk];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(CholeskyError::NotPositiveDefinite { index: i });
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    // forward substitution: L y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        y[i] = s / l[i * n + i];
    }
    // back substitution: Lᵀ x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small SPD matrix and its exact solution.
    fn spd3() -> (Vec<f32>, Vec<f32>) {
        let m = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        (m, b)
    }

    fn matvec(m: &[f32], x: &[f32], n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (0..n).map(|j| m[i * n + j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn cholesky_solves_an_spd_system() {
        let (m, b) = spd3();
        let x = dense_cholesky_solve(&m, &b, 3).expect("spd");
        let r = matvec(&m, &x, 3);
        for (i, (rv, bv)) in r.iter().zip(&b).enumerate() {
            assert!((rv - bv).abs() < 1e-4, "residual {i}: {rv} vs {bv}");
        }
    }

    #[test]
    fn cg_and_dense_cholesky_agree() {
        let (m, b) = spd3();
        let chol = dense_cholesky_solve(&m, &b, 3).expect("spd");
        let cg = cg_matrix_free(|x| matvec(&m, x, 3), &b, 1e-8, 100);
        assert!(cg.converged, "CG must converge on a 3x3 SPD system");
        for (i, (cgv, cv)) in cg.x.iter().zip(&chol).enumerate() {
            assert!(
                (cgv - cv).abs() < 1e-4,
                "coordinate {i}: cg {cgv} vs cholesky {cv}"
            );
        }
        // CG's "terminates in at most n steps" is an EXACT-ARITHMETIC theorem. At f32 with an
        // absolute tolerance it can need one pass more to drive the residual under `tol` — measured
        // at 4 for n = 3. Asserting `<= n` would assert the textbook, not the code.
        assert!(
            cg.iters <= 3 + 1,
            "CG on n=3 needs at most one pass beyond the exact-arithmetic bound: {}",
            cg.iters
        );
    }

    #[test]
    fn a_zero_right_hand_side_never_iterates() {
        let (m, _) = spd3();
        let out = cg_matrix_free(|x| matvec(&m, x, 3), &[0.0, 0.0, 0.0], 1e-8, 100);
        assert_eq!(out.x, vec![0.0, 0.0, 0.0]);
        assert_eq!(out.iters, 0);
        assert_eq!(out.relative_residual, 0.0);
        assert!(out.converged);
    }

    #[test]
    fn the_iteration_count_equals_the_operator_applications() {
        let (m, b) = spd3();
        let mut applications = 0usize;
        let out = cg_matrix_free(
            |x| {
                applications += 1;
                matvec(&m, x, 3)
            },
            &b,
            1e-8,
            100,
        );
        assert_eq!(
            out.iters, applications,
            "iters counts operator applications"
        );
    }

    #[test]
    fn a_non_spd_matrix_is_reported_not_panicked() {
        // negative diagonal: not positive definite at index 0
        let m = vec![-1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            dense_cholesky_solve(&m, &[1.0, 1.0], 2),
            Err(CholeskyError::NotPositiveDefinite { index: 0 })
        );
        // a zero pivot in the trailing minor
        let m2 = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            dense_cholesky_solve(&m2, &[1.0, 1.0], 2),
            Err(CholeskyError::NotPositiveDefinite { index: 1 })
        );
    }

    #[test]
    fn bad_shapes_and_degenerate_sizes_are_handled() {
        assert_eq!(
            dense_cholesky_solve(&[1.0, 2.0], &[1.0], 1),
            Err(CholeskyError::ShapeMismatch)
        );
        assert_eq!(
            dense_cholesky_solve(&[1.0], &[1.0, 2.0], 1),
            Err(CholeskyError::ShapeMismatch)
        );
        assert_eq!(dense_cholesky_solve(&[], &[], 0), Ok(Vec::new()));
        let empty = cg_matrix_free(|_| Vec::new(), &[], 1e-8, 10);
        assert_eq!(empty.iters, 0);
        assert!(empty.x.is_empty());
    }

    #[test]
    fn a_ridge_makes_a_singular_operator_solvable() {
        // rank-1 outer product: singular on its own
        let a = [1.0f32, 2.0, -1.0];
        let n = 3;
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = a[i] * a[j];
            }
        }
        assert!(
            dense_cholesky_solve(&m, &[1.0; 3], n).is_err(),
            "singular without a ridge"
        );
        for i in 0..n {
            m[i * n + i] += 0.5;
        }
        let x = dense_cholesky_solve(&m, &[1.0; 3], n).expect("ridge makes it SPD");
        for rv in matvec(&m, &x, n) {
            assert!((rv - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn cg_stops_rather_than_stepping_on_an_indefinite_operator() {
        // an indefinite diagonal operator: p^T A p can be <= 0
        let d = [1.0f32, -1.0];
        let out = cg_matrix_free(|x| vec![d[0] * x[0], d[1] * x[1]], &[1.0, 1.0], 1e-10, 50);
        assert!(!out.converged || out.relative_residual.is_finite());
        assert!(out.x.iter().all(|v| v.is_finite()), "no NaN is fabricated");
    }
}
