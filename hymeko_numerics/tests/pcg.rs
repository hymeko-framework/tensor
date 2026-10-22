//! Preconditioned conjugate gradients on symmetric positive definite operators.
//!
//! The operator is a scaled, shifted one-dimensional Dirichlet Laplacian, `A = S (L + c I) S` with
//! `S` diagonal: symmetric positive definite for every positive `S`, and as badly scaled as `S` is
//! uneven. Jacobi returns it to `(L + c I) / (2 + c)`, whose condition number is at most
//! `(4 + c) / c`; that is the case a diagonal preconditioner exists for, so it is where its effect
//! must show. (Unshifted, Jacobi returns `L / 2`, conditioned like `n²`, and CG needs about `n`
//! steps either way.)

use hymeko_numerics::{pcg, Additive, Identity, Jacobi, Operator, PcgProgress, Preconditioner};

/// `A = S (L + c I) S`, `L = tridiag(−1, 2, −1)`.
struct ScaledLaplacian {
    scale: Vec<f64>,
    shift: f64,
}

impl ScaledLaplacian {
    fn new(n: usize, unevenness: f64, shift: f64) -> Self {
        Self {
            scale: (0..n)
                .map(|i| 1.0 + unevenness * ((i * 7919) % 211) as f64 / 211.0)
                .collect(),
            shift,
        }
    }

    fn diagonal(&self) -> Vec<f64> {
        self.scale
            .iter()
            .map(|s| (2.0 + self.shift) * s * s)
            .collect()
    }
}

impl Operator for ScaledLaplacian {
    fn dim(&self) -> usize {
        self.scale.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.dim();
        let s = &self.scale;
        for i in 0..n {
            let mut v = (2.0 + self.shift) * s[i] * x[i];
            if i > 0 {
                v -= s[i - 1] * x[i - 1];
            }
            if i + 1 < n {
                v -= s[i + 1] * x[i + 1];
            }
            y[i] = s[i] * v;
        }
    }
}

fn exact(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64) * 0.37).sin() + 0.1 * i as f64)
        .collect()
}

fn rhs_of(a: &impl Operator, x: &[f64]) -> Vec<f64> {
    let mut b = vec![0.0; x.len()];
    a.apply(x, &mut b);
    b
}

fn max_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn solves_a_laplacian_to_its_tolerance() {
    let a = ScaledLaplacian::new(64, 0.0, 0.0);
    let star = exact(64);
    let b = rhs_of(&a, &star);
    let mut x = vec![0.0; 64];
    let report = pcg(&a, &b, &mut x, &Identity, 1e-12, 500, &mut |_| {});
    assert!(report.converged, "{report:?}");
    assert!(report.residual <= 1e-12, "{report:?}");
    // CG on n distinct eigenvalues terminates in at most n steps in exact arithmetic.
    assert!(report.iterations <= 64 + 8, "{report:?}");
    assert!(max_error(&x, &star) < 1e-8, "{}", max_error(&x, &star));
}

#[test]
fn jacobi_cuts_the_iterations_of_a_badly_scaled_operator() {
    // 200 distinct scales in [1, 50): a spread spectrum, not clusters CG would resolve one by one.
    let a = ScaledLaplacian::new(200, 49.0, 1.0);
    let b = rhs_of(&a, &exact(200));
    let count = |m: &dyn Preconditioner| {
        let mut x = vec![0.0; 200];
        let report = pcg(&a, &b, &mut x, m, 1e-10, 5000, &mut |_| {});
        assert!(report.converged, "{report:?}");
        report.iterations
    };
    let (plain, jacobi) = (count(&Identity), count(&Jacobi::of(&a.diagonal())));
    // κ ≤ 5 after Jacobi: a few tens of steps; κ in the thousands without it.
    assert!(jacobi < 40, "Jacobi {jacobi} iterations");
    assert!(
        4 * jacobi < plain,
        "Jacobi {jacobi} iterations against {plain} unpreconditioned"
    );
}

#[test]
fn the_observer_sees_every_iteration_and_the_report_the_true_residual() {
    let a = ScaledLaplacian::new(40, 1.0, 0.0);
    let star = exact(40);
    let b = rhs_of(&a, &star);
    let mut x = vec![0.0; 40];
    let mut seen: Vec<PcgProgress> = Vec::new();
    let report = pcg(
        &a,
        &b,
        &mut x,
        &Jacobi::of(&a.diagonal()),
        1e-9,
        1000,
        &mut |p| seen.push(*p),
    );
    let indices: Vec<usize> = seen.iter().map(|p| p.iteration).collect();
    assert_eq!(indices, (1..=report.iterations).collect::<Vec<_>>());
    let r = rhs_of(&a, &x);
    let true_residual = r
        .iter()
        .zip(&b)
        .map(|(ax, bi)| (bi - ax) * (bi - ax))
        .sum::<f64>()
        .sqrt()
        / b.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert_eq!(report.residual, true_residual);
}

#[test]
fn a_zero_right_hand_side_is_solved_without_iterating() {
    let a = ScaledLaplacian::new(10, 1.0, 0.0);
    let mut x = vec![0.0; 10];
    let report = pcg(&a, &[0.0; 10], &mut x, &Identity, 1e-12, 100, &mut |_| {});
    assert_eq!(report.iterations, 0);
    assert!(report.converged);
    assert!(x.iter().all(|&v| v == 0.0));
}

/// The negated Laplacian: every direction has negative curvature.
struct Negated(ScaledLaplacian);

impl Operator for Negated {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.0.apply(x, y);
        y.iter_mut().for_each(|v| *v = -*v);
    }
}

#[test]
fn negative_curvature_stops_the_iteration_unconverged() {
    let a = Negated(ScaledLaplacian::new(16, 0.0, 0.0));
    let b = rhs_of(&a, &exact(16));
    let mut x = vec![0.0; 16];
    let report = pcg(&a, &b, &mut x, &Identity, 1e-12, 100, &mut |_| {});
    assert_eq!(report.iterations, 0, "{report:?}");
    assert!(!report.converged);
    assert!(x.iter().all(|&v| v == 0.0));
}

#[test]
fn preconditioners_have_their_closed_forms() {
    let r = [2.0, -4.0, 0.5];
    let mut z = [0.0; 3];
    Identity.apply(&r, &mut z);
    assert_eq!(z, r);
    // A non-positive diagonal entry leaves its coordinate still.
    Jacobi::of(&[2.0, 0.0, -1.0]).apply(&r, &mut z);
    assert_eq!(z, [1.0, 0.0, 0.0]);
    Additive(Identity, Jacobi::of(&[2.0, 4.0, 0.5])).apply(&r, &mut z);
    assert_eq!(z, [3.0, -5.0, 1.5]);
}
