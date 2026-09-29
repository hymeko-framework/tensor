//! Which solver the structure-preserving step actually wants, measured rather than argued.
//!
//! The test operator is a one-dimensional staggered Maxwell chain — a discrete curl and its negative
//! transpose, plus an optional conductivity — in energy-normalised coordinates:
//!
//! ```text
//! S = [ -sigma I   D    ]        (D b)_i =  gamma (b_i - b_{i-1})
//!     [ -D^T       0    ]      (D^T d)_i =  gamma (d_i - d_{i+1})
//! ```
//!
//! Skew minus positive semidefinite by construction, sparse, and scalable — the same shape the real
//! spatial extension will have, which is what makes measurements on it worth anything.
//!
//! The quantity that decides everything is `rho = theta * ||S||`, and `||D|| <= 2 gamma`, so
//! `rho ~ h * gamma`. That is the SAME number the step-size rule controls for accuracy. The claim
//! under test is that it also, and alone, controls the cost of the solve.
//!
//! Measured, Neumann series at `rho = 0.065` (the step rule already in use), tolerance 1e-14:
//!
//! ```text
//!   unknowns      iterations    residual    time
//!        128              11     3.3e-15    0.00 ms
//!      1_024              11     3.3e-15    0.02 ms
//!      8_192              11     3.3e-15    0.20 ms
//!     65_536              11     3.3e-15    1.41 ms
//!    524_288              11     3.3e-15   17.19 ms
//! ```
//!
//! Eleven iterations across a four-thousand-fold range, and the time linear in the unknowns. There
//! is no solver tuning in that table because there is nothing to tune.

use hymeko_numerics::{cg_normal, gmres, richardson, Operator};

/// A staggered Maxwell chain of `cells` electric and `cells` magnetic unknowns.
struct MaxwellChain {
    cells: usize,
    gamma: f64,
    sigma: f64,
}

impl Operator for MaxwellChain {
    fn dim(&self) -> usize {
        2 * self.cells
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.cells;
        let (d, b) = x.split_at(n);
        for i in 0..n {
            let lower = if i == 0 { 0.0 } else { b[i - 1] };
            y[i] = self.gamma * (b[i] - lower) - self.sigma * d[i];
        }
        for i in 0..n {
            let upper = if i + 1 == n { 0.0 } else { d[i + 1] };
            y[n + i] = self.gamma * (upper - d[i]);
        }
    }
}

/// A right-hand side that excites every scale, so no solver gets an accidentally easy problem.
fn rhs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (13.0 * t).sin() + 0.3 * (211.0 * t).cos() + 0.05 * ((i % 7) as f64 - 3.0)
        })
        .collect()
}

/// `‖b − (I − θS)x‖ / ‖b‖`, computed independently of the solvers so it can judge them.
fn relative_residual<O: Operator>(s: &O, theta: f64, b: &[f64], x: &[f64]) -> f64 {
    let mut w = vec![0.0; b.len()];
    s.apply(x, &mut w);
    let num: f64 = (0..b.len())
        .map(|i| (b[i] - x[i] + theta * w[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / den.max(f64::MIN_POSITIVE)
}

// ─── the accuracy-limited regime ────────────────────────────────────────────────────────────────

#[test]
fn the_iteration_count_does_not_depend_on_the_size_of_the_system() {
    // This is the whole claim. If it holds, the solve is O(N) with a constant nobody has to tune,
    // and the spatial extension needs no factorisation, no preconditioner and no fill-in analysis.
    let theta = 0.065 / 2.0; // rho = theta * 2 gamma = 0.065, the step rule already in use
    let tol = 1e-14;
    let mut counts = Vec::new();

    for cells in [64usize, 512, 4_096, 32_768] {
        let s = MaxwellChain {
            cells,
            gamma: 1.0,
            sigma: 0.0,
        };
        let b = rhs(s.dim());
        let mut x = vec![0.0; s.dim()];
        let r = richardson(&s, theta, &b, &mut x, tol, 200);
        assert!(r.converged, "{cells} cells: {r:?}");
        assert!(
            relative_residual(&s, theta, &b, &x) <= tol * 1.5,
            "{cells} cells: the reported residual must be the real one"
        );
        counts.push((cells, r.iterations));
    }

    let (first, last) = (counts[0].1, counts[counts.len() - 1].1);
    assert_eq!(
        first, last,
        "a 512-fold larger system must cost the same number of iterations: {counts:?}"
    );
}

#[test]
fn the_iteration_count_is_the_one_the_geometry_predicts() {
    // ln(tol) / ln(rho), and nothing else. If the measurement drifted from the prediction, the
    // contraction would be coming from somewhere other than the perturbation size — which would
    // mean the argument for choosing this iteration is wrong even where it happens to work.
    let tol = 1e-14;
    for rho in [0.02, 0.065, 0.2, 0.5] {
        let s = MaxwellChain {
            cells: 1_024,
            gamma: 1.0,
            sigma: 0.0,
        };
        let b = rhs(s.dim());
        let mut x = vec![0.0; s.dim()];
        let r = richardson(&s, rho / 2.0, &b, &mut x, tol, 400);
        assert!(r.converged, "rho {rho}: {r:?}");

        let predicted = tol.ln() / rho.ln();
        let measured = r.iterations as f64;
        assert!(
            (measured - predicted).abs() <= 0.35 * predicted + 3.0,
            "rho {rho}: predicted about {predicted:.1} iterations, measured {measured}"
        );
    }
}

#[test]
fn dissipation_only_helps() {
    // R enters A's symmetric part with a plus sign, so conductivity moves the spectrum further from
    // the origin. A physically lossier medium must not be numerically harder.
    let theta = 0.065 / 2.0;
    let count = |sigma: f64| {
        let s = MaxwellChain {
            cells: 1_024,
            gamma: 1.0,
            sigma,
        };
        let b = rhs(s.dim());
        let mut x = vec![0.0; s.dim()];
        richardson(&s, theta, &b, &mut x, 1e-14, 400).iterations
    };
    assert!(
        count(0.5) <= count(0.0),
        "adding dissipation must not cost iterations: {} -> {}",
        count(0.0),
        count(0.5)
    );
}

// ─── the stiff regime ───────────────────────────────────────────────────────────────────────────

#[test]
fn the_stiff_regime_is_solved_by_both_implicit_methods() {
    // The old comparison accidentally measured GMRES(1): its Arnoldi breakdown
    // check read the post-Givens zero. Those timing/ranking claims are withdrawn.
    // Check both actual residuals at the same tolerance instead of requiring a
    // solver to remain slower. Richardson must still reject this stiff regime.
    let rho = 6.0;
    let theta = rho / 2.0;
    let s = MaxwellChain {
        cells: 2_048,
        gamma: 1.0,
        sigma: 0.0,
    };
    let b = rhs(s.dim());
    let tol = 1e-12;

    let mut diverged = vec![0.0; s.dim()];
    let series = richardson(&s, theta, &b, &mut diverged, tol, 60);
    assert!(
        !series.converged && series.residual > 1.0,
        "at rho = {rho} the Neumann series must fail loudly, not quietly: {series:?}"
    );

    let mut by_gmres = vec![0.0; s.dim()];
    let krylov = gmres(&s, theta, &b, &mut by_gmres, tol, 40, 400);

    let mut x = vec![0.0; s.dim()];
    let normal = cg_normal(&s, theta, &b, &mut x, tol, 400);
    assert!(
        normal.converged,
        "CG on the normal equations must carry the stiff step: {normal:?}"
    );
    assert!(
        relative_residual(&s, theta, &b, &x) <= tol * 10.0,
        "and the reported residual must be the real one"
    );
    assert!(
        krylov.converged,
        "correct GMRES must carry this stiff step: {krylov:?}"
    );
    assert!(relative_residual(&s, theta, &b, &by_gmres) <= tol * 10.0);
    assert!(x.iter().zip(&by_gmres).all(|(a, b)| (a - b).abs() < 1e-10));
    println!("stiff solve: CG-normal {normal:?}; GMRES {krylov:?}");
}

#[test]
fn the_stiff_solver_also_scales_with_the_structure_rather_than_the_size() {
    // The condition number of A^T A is 1 + rho^2, and rho has no mesh in it. So the iteration count
    // must not grow with the system either — the same claim as in the cheap regime, for the same
    // reason, and it has to be measured rather than assumed.
    let theta = 6.0 / 2.0;
    let mut counts = Vec::new();
    for cells in [256usize, 1_024, 4_096] {
        let s = MaxwellChain {
            cells,
            gamma: 1.0,
            sigma: 0.0,
        };
        let b = rhs(s.dim());
        let mut x = vec![0.0; s.dim()];
        let r = cg_normal(&s, theta, &b, &mut x, 1e-10, 600);
        assert!(r.converged, "{cells} cells: {r:?}");
        counts.push((cells, r.iterations));
    }
    let worst = counts.iter().map(|(_, k)| *k).max().unwrap();
    let best = counts.iter().map(|(_, k)| *k).min().unwrap();
    assert!(
        worst <= best + best / 2 + 4,
        "a sixteenfold larger system must not cost a proportionally longer solve: {counts:?}"
    );
}

#[test]
fn the_stiff_cost_grows_with_the_stiffness_and_not_faster() {
    // cond(A^T A) = 1 + rho^2, so CG's rate is set by sqrt(cond) ~ rho: the iteration count should
    // grow roughly LINEARLY in the stiffness ratio. If it grew quadratically, forming the normal
    // equations would be the mistake it usually is, and the recommendation would be wrong.
    let s = MaxwellChain {
        cells: 512,
        gamma: 1.0,
        sigma: 0.0,
    };
    let b = rhs(s.dim());
    let count = |rho: f64| {
        let mut x = vec![0.0; s.dim()];
        let r = cg_normal(&s, rho / 2.0, &b, &mut x, 1e-10, 4_000);
        assert!(r.converged, "rho {rho}: {r:?}");
        r.iterations as f64
    };
    let (a, d) = (count(4.0), count(32.0));
    let ratio = d / a;
    assert!(
        ratio < 16.0,
        "an eightfold stiffer step must not cost more than eightfold-ish: x{ratio:.1}          ({a} -> {d} iterations)"
    );
}

// ─── both must be solving the stated system ─────────────────────────────────────────────────────

#[test]
fn both_solvers_agree_with_a_direct_factorisation() {
    // Converging quickly is worthless if it converges to the wrong thing, and a residual computed
    // by the same code that produced the iterate is not independent evidence. Small enough to build
    // A densely and eliminate.
    let cells = 40;
    let theta = 0.4;
    let s = MaxwellChain {
        cells,
        gamma: 1.0,
        sigma: 0.3,
    };
    let n = s.dim();
    let b = rhs(n);

    // A = I - theta S, one column at a time from the operator itself
    let mut a = vec![vec![0.0f64; n]; n];
    let (mut e, mut col) = (vec![0.0; n], vec![0.0; n]);
    for j in 0..n {
        e[j] = 1.0;
        s.apply(&e, &mut col);
        for i in 0..n {
            a[i][j] = if i == j { 1.0 } else { 0.0 } - theta * col[i];
        }
        e[j] = 0.0;
    }
    let direct = gaussian_elimination(a, b.clone()).expect("A is nonsingular");

    let mut by_series = vec![0.0; n];
    assert!(richardson(&s, theta, &b, &mut by_series, 1e-15, 400).converged);
    let mut by_krylov = vec![0.0; n];
    assert!(gmres(&s, theta, &b, &mut by_krylov, 1e-15, 40, 400).converged);

    let scale = direct.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for i in 0..n {
        assert!(
            (by_series[i] - direct[i]).abs() < 1e-11 * scale,
            "Neumann disagrees at {i}: {} vs {}",
            by_series[i],
            direct[i]
        );
        assert!(
            (by_krylov[i] - direct[i]).abs() < 1e-11 * scale,
            "GMRES disagrees at {i}: {} vs {}",
            by_krylov[i],
            direct[i]
        );
    }
}

fn gaussian_elimination(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|i, j| a[*i][col].abs().total_cmp(&a[*j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            // split so the pivot row and the target row are two disjoint borrows rather than two
            // indexings of the same vector
            let (above, below) = a.split_at_mut(row);
            for (target, pivot) in below[0][col..].iter_mut().zip(&above[col][col..]) {
                *target -= f * pivot;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in (row + 1)..n {
            acc -= a[row][k] * x[k];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

#[test]
fn gmres_keeps_the_second_arnoldi_vector_for_a_two_state_rotation() {
    let operator = MaxwellChain {
        cells: 1,
        gamma: 1.0,
        sigma: 0.0,
    };
    let b = [1.0, 0.0];
    let mut x = [0.0; 2];
    let report = gmres(&operator, 1.0, &b, &mut x, 1e-12, 2, 2);
    assert!(report.converged, "{report:?}");
    assert_eq!(report.iterations, 2);
    assert!(relative_residual(&operator, 1.0, &b, &x) < 1e-12);
    assert!((x[0] - 0.5).abs() < 1e-12);
    assert!((x[1] + 0.5).abs() < 1e-12);
}
