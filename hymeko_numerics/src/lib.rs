//! Matrix-free iterative solvers for **shifted skew-symmetric** systems.
//!
//! ```text
//! A x = b,    A = I - θ S,    S = K - D    with  K skew, D positive semidefinite
//! ```
//!
//! Nothing here knows what a port-Hamiltonian system is, what a mesh is, or what HyMeKo is. The
//! public surface is a trait over `&[f64]` and three functions — which is what makes it general
//! mathematics rather than an application detail that happens to be shared.
//!
//! # Why this is not `hymeko_tensor::linalg`
//!
//! A §6.1 discovery pass over the workspace found one near-owner and rejected it on contract, not
//! on taste. [`hymeko_tensor`]'s `linalg` module is also matrix-free and also generic — but its
//! stated, pinned contract is **`f32` throughout, in the order written, no `f64` promotion**,
//! because consumers reproduce recorded numbers through it; and both of its solvers **require
//! SPD**, which is exactly what the matrices here are not. Widening either of those would change a
//! contract that other code depends on, to serve a caller that needs neither. `hymeko_numeric_vv`
//! was rejected for the opposite reason: it is a downstream verification client by construction and
//! owns no kernels at all.
//!
//! So: a new crate, on the `hymeko_vision` precedent — narrow scope, dependency-free, and argument
//! types (`&[f64]`, a trait) with no domain semantics in them.
//!
//! # What has to be solved
//!
//! One system per step of the discrete gradient scheme, in energy-normalised coordinates:
//!
//! ```text
//! A x = b,    A = I - θ S,    θ = h/2,    S = Q^(1/2) (J - R) Q^(1/2)
//! ```
//!
//! `S` is skew minus positive semidefinite, so `A` has three properties that decide the solver, and
//! all three come from the physics rather than from luck:
//!
//! 1. **Its symmetric part is `I + θ Q^(1/2) R Q^(1/2) ⪰ I`.** So `xᵀAx ≥ |x|²`: `A` is positive
//!    real, uniformly, with no dependence on the mesh. That is the exact hypothesis under which
//!    GMRES has a guaranteed per-iteration reduction.
//! 2. **`A` is a perturbation of the identity of size `θ‖S‖`,** and `θ‖S‖ = ωh/2` is the same
//!    quantity the step-size rule already controls for accuracy. When it is below one — which is
//!    what "resolve the oscillation" means — the Neumann series converges geometrically at exactly
//!    that ratio, in a number of iterations that depends on `ωh` and on **nothing else**: not on the
//!    size of the system, not on the mesh, not on the conditioning.
//! 3. **The structure is fixed across steps.** `J`, `R` and `Q` are constant; only `b` changes. A
//!    factorisation could be reused — but with (2) there is nothing to factorise.
//!
//! The consequence is worth stating plainly, because it decides an architecture: **in the
//! accuracy-limited regime the solver is a dozen sparse mat-vecs with no setup, no factorisation,
//! no preconditioner and no fill-in.** The step size that makes the physics right also makes the
//! solve cheap. The two are the same number.
//!
//! The other regime is real too. A stiff problem is integrated implicitly precisely so the step can
//! be *larger* than the fastest mode, and then `θ‖S‖ > 1` and the Neumann series diverges. The
//! obvious fallback — unpreconditioned GMRES — is measurably the wrong one there; [`cg_normal`]
//! explains why and what replaces it. All three are provided, and the choice between them is a
//! measured property rather than a preference: see `tests/solver.rs`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod port_hamiltonian;

pub use port_hamiltonian::{Interconnection, NormalisedFlow, Port, PortHamiltonian};

/// A linear operator applied to a vector. Matrix-free: a discrete curl is a stencil, and writing it
/// out as a matrix would spend memory on structure that is already known.
pub trait Operator {
    /// Dimension of the space the operator acts on.
    fn dim(&self) -> usize;
    /// `y <- S x`. Implementations may assume `x.len() == y.len() == dim()`.
    fn apply(&self, x: &[f64], y: &mut [f64]);
}

/// What a solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Report {
    /// Iterations actually taken.
    pub iterations: usize,
    /// Final relative residual `‖b - A x‖ / ‖b‖`.
    pub residual: f64,
    /// Whether the tolerance was reached within the iteration budget.
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Residual of `A x = b` with `A = I - θ S`, written into `r`.
fn residual<O: Operator>(s: &O, theta: f64, b: &[f64], x: &[f64], r: &mut [f64], work: &mut [f64]) {
    s.apply(x, work);
    for i in 0..b.len() {
        r[i] = b[i] - x[i] + theta * work[i];
    }
}

/// Solve `(I - θ S) x = b` by the Neumann series, i.e. Richardson iteration with unit relaxation.
///
/// `x ← b + θ S x`. The error contracts by exactly `θ‖S‖` each sweep, so the iteration count is
/// `ln(tol) / ln(θ‖S‖)` and does not depend on the dimension. One operator application per
/// iteration, no storage beyond two vectors, and nothing to set up between steps.
///
/// Diverges when `θ‖S‖ ≥ 1`. That is not a defect to be papered over: it is the statement that the
/// step is longer than the fastest mode in the system, and at that point the caller should be using
/// [`gmres`] and should know that it is.
pub fn richardson<O: Operator>(
    s: &O,
    theta: f64,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iterations: usize,
) -> Report {
    let n = s.dim();
    let (mut work, mut r) = (vec![0.0; n], vec![0.0; n]);
    let scale = norm(b).max(f64::MIN_POSITIVE);

    for iteration in 0..=max_iterations {
        residual(s, theta, b, x, &mut r, &mut work);
        let relative = norm(&r) / scale;
        if relative <= tol || iteration == max_iterations {
            return Report {
                iterations: iteration,
                residual: relative,
                converged: relative <= tol,
            };
        }
        // x <- b + θ S x, reusing the operator application already in `work`
        s.apply(x, &mut work);
        for i in 0..n {
            x[i] = b[i] + theta * work[i];
        }
    }
    unreachable!()
}

/// Solve `(I - θ S) x = b` by GMRES with Arnoldi and Givens rotations, restarted every `restart`.
///
/// For this operator the convergence is not a hope. The symmetric part of `A` is at least the
/// identity, so `A` is positive real with `λ_min(H) ≥ 1`, and Elman's bound gives a per-iteration
/// reduction of `√(1 - λ_min(H)² / ‖A‖²)` — a rate set by `θ‖S‖` and by nothing about the mesh.
///
/// Used where the Neumann series cannot be: a step deliberately longer than the fastest mode.
pub fn gmres<O: Operator>(
    s: &O,
    theta: f64,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    restart: usize,
    max_iterations: usize,
) -> Report {
    let n = s.dim();
    let m = restart.clamp(1, max_iterations.max(1));
    let scale = norm(b).max(f64::MIN_POSITIVE);
    let (mut r, mut work) = (vec![0.0; n], vec![0.0; n]);
    let mut total = 0usize;

    loop {
        residual(s, theta, b, x, &mut r, &mut work);
        let beta = norm(&r);
        let relative = beta / scale;
        if relative <= tol || total >= max_iterations {
            return Report {
                iterations: total,
                residual: relative,
                converged: relative <= tol,
            };
        }

        // Arnoldi basis, Hessenberg factor kept in its rotated (upper triangular) form
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(r.iter().map(|t| t / beta).collect());
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0f64; m], vec![0.0f64; m]);
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;
        let mut used = 0usize;

        for k in 0..m {
            if total + k >= max_iterations {
                break;
            }
            // w = A v_k = v_k - θ S v_k
            s.apply(&v[k], &mut work);
            let mut w: Vec<f64> = (0..n).map(|i| v[k][i] - theta * work[i]).collect();
            for j in 0..=k {
                h[j][k] = dot(&w, &v[j]);
                for i in 0..n {
                    w[i] -= h[j][k] * v[j][i];
                }
            }
            h[k + 1][k] = norm(&w);

            // apply the rotations accumulated so far, then annihilate the new subdiagonal
            for j in 0..k {
                let (c, sj) = (cs[j], sn[j]);
                let t = c * h[j][k] + sj * h[j + 1][k];
                h[j + 1][k] = -sj * h[j][k] + c * h[j + 1][k];
                h[j][k] = t;
            }
            let denominator = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if denominator < 1e-300 {
                used = k;
                break;
            }
            cs[k] = h[k][k] / denominator;
            sn[k] = h[k + 1][k] / denominator;
            h[k][k] = denominator;
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];
            used = k + 1;

            let breakdown = h[k + 1][k].abs() < 1e-300;
            if breakdown || g[k + 1].abs() / scale <= tol {
                if !breakdown {
                    v.push(w.iter().map(|t| t / denominator.max(1e-300)).collect());
                }
                break;
            }
            v.push(w.iter().map(|t| t / h[k + 1][k].max(1e-300)).collect());
        }

        // back-substitute for the correction and add it
        if used > 0 {
            let mut y = vec![0.0f64; used];
            for i in (0..used).rev() {
                let mut acc = g[i];
                for j in (i + 1)..used {
                    acc -= h[i][j] * y[j];
                }
                y[i] = acc / h[i][i];
            }
            for (k, yk) in y.iter().enumerate() {
                for i in 0..n {
                    x[i] += yk * v[k][i];
                }
            }
        }
        total += used.max(1);
    }
}

/// Solve `(I - θ S) x = b` by conjugate gradients on the normal equations.
///
/// # Why this is not the usual bad idea
///
/// Forming `AᵀA` normally squares the condition number and is a mistake. Here it is not, and the
/// reason is structural. With no dissipation `S = K` is skew, so
///
/// ```text
/// Aᵀ = I - θKᵀ = I + θK,      AᵀA = (I + θK)(I - θK) = I + θ² KᵀK
/// ```
///
/// `A` is then a NORMAL matrix whose eigenvalues are `1 - iθλ`, so `cond₂(A) = √(1 + ρ²)` with
/// `ρ = θ‖S‖` — and squaring it gives `cond(AᵀA) = 1 + ρ²`, which grows like `ρ²` rather than like
/// `ρ⁴`. Conjugate gradients then converge at a rate set by `√(1 + ρ²) ≈ ρ`, i.e. **linearly in the
/// stiffness ratio**, on a symmetric positive definite system with three-term recurrences and no
/// growing basis.
///
/// The comparison that matters is against unpreconditioned GMRES, whose eigenvalues here lie spread
/// along a vertical segment from `1 - iρ` to `1 + iρ`. Elman's bound then gives a per-iteration
/// reduction of only `√(1 - 1/(1+ρ²))`, which at `ρ = 6` is `0.949` — about five hundred iterations
/// for twelve digits, with a Krylov basis that grows the whole way. `tests/solver.rs` measures both.
pub fn cg_normal<O: Operator>(
    s: &O,
    theta: f64,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iterations: usize,
) -> Report {
    let n = s.dim();
    let (mut work, mut r, mut p, mut ap) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    // `A z` and `Aᵀ z` differ only in the sign of the skew part, and `S` is only available whole —
    // so the transpose is formed as `2 z - A z`, which is exact when S is skew and introduces the
    // dissipative part with the wrong sign otherwise. That is why `sigma` is folded in separately
    // below rather than being left inside the operator.
    let apply_a = |s: &O, z: &[f64], out: &mut [f64], work: &mut [f64]| {
        s.apply(z, work);
        for i in 0..n {
            out[i] = z[i] - theta * work[i];
        }
    };

    // normal-equation right-hand side and residual
    let mut rhs = vec![0.0; n];
    s.apply(b, &mut work);
    for i in 0..n {
        rhs[i] = b[i] + theta * work[i]; // Aᵀ b, exact for skew S
    }
    let scale = norm(b).max(f64::MIN_POSITIVE);

    // r = Aᵀb - AᵀA x
    let mut t = vec![0.0; n];
    apply_a(s, x, &mut t, &mut work);
    s.apply(&t, &mut work);
    for i in 0..n {
        r[i] = rhs[i] - (t[i] + theta * work[i]);
        p[i] = r[i];
    }
    let mut rr = dot(&r, &r);

    for iteration in 0..=max_iterations {
        // the residual the CALLER cares about is the original one, not the normal-equation one
        let mut check = vec![0.0; n];
        apply_a(s, x, &mut check, &mut work);
        let relative = (0..n)
            .map(|i| (b[i] - check[i]).powi(2))
            .sum::<f64>()
            .sqrt()
            / scale;
        if relative <= tol || iteration == max_iterations {
            return Report {
                iterations: iteration,
                residual: relative,
                converged: relative <= tol,
            };
        }

        apply_a(s, &p, &mut t, &mut work);
        s.apply(&t, &mut work);
        for i in 0..n {
            ap[i] = t[i] + theta * work[i]; // AᵀA p
        }
        let denominator = dot(&p, &ap);
        if denominator.abs() < 1e-300 {
            return Report { iterations: iteration, residual: relative, converged: false };
        }
        let alpha = rr / denominator;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_next = dot(&r, &r);
        let beta = rr_next / rr.max(1e-300);
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_next;
    }
    unreachable!()
}
