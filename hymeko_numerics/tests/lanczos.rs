//! Tests of `hymeko_numerics::lanczos` (plan 2026-10-19-numerics-optimizers, O7, Amendment 9).

use core::f64::consts::PI;

use hymeko_numerics::{lanczos, tridiagonal_eigh, Lanczos, Operator};

/// A diagonal operator with the given spectrum.
struct Diagonal(Vec<f64>);

impl Operator for Diagonal {
    fn dim(&self) -> usize {
        self.0.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for ((yi, xi), di) in y.iter_mut().zip(x).zip(&self.0) {
            *yi = di * xi;
        }
    }
}

/// The 1-D Dirichlet Laplacian `tridiag(−1, 2, −1)` of order `n`: eigenvalues
/// `2 − 2 cos(kπ / (n + 1))`, `k = 1 … n`.
struct Laplacian(usize);

impl Operator for Laplacian {
    fn dim(&self) -> usize {
        self.0
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.0;
        for i in 0..n {
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i + 1 < n { x[i + 1] } else { 0.0 };
            y[i] = 2.0 * x[i] - left - right;
        }
    }
}

fn laplacian_spectrum(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|k| 2.0 - 2.0 * (k as f64 * PI / (n as f64 + 1.0)).cos())
        .collect()
}

/// A deterministic start vector with every component nonzero.
fn start(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + 0.37 * ((i * 7919) % 101) as f64 / 101.0)
        .collect()
}

/// The tridiagonal eigensolver reproduces the Toeplitz spectrum in closed form, and its vectors
/// are orthonormal eigenvectors.
#[test]
fn the_tridiagonal_eigenpairs_are_the_closed_form_ones() {
    for n in [1usize, 2, 3, 10, 50] {
        let (values, vectors) = tridiagonal_eigh(&vec![2.0; n], &vec![-1.0; n.saturating_sub(1)]);
        for (got, want) in values.iter().zip(laplacian_spectrum(n)) {
            assert!((got - want).abs() < 1e-12, "n {n}: {got} vs {want}");
        }
        let operator = Laplacian(n);
        for k in 0..n {
            let x: Vec<f64> = (0..n).map(|i| vectors[i * n + k]).collect();
            let mut ax = vec![0.0; n];
            operator.apply(&x, &mut ax);
            let residual = ax
                .iter()
                .zip(&x)
                .map(|(a, b)| (a - values[k] * b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual < 1e-12, "n {n}, pair {k}: {residual:e}");
            for j in 0..n {
                let inner: f64 = (0..n)
                    .map(|i| vectors[i * n + k] * vectors[i * n + j])
                    .sum();
                let want = if j == k { 1.0 } else { 0.0 };
                assert!((inner - want).abs() < 1e-12, "n {n}: ⟨{k}, {j}⟩ = {inner}");
            }
        }
    }
}

/// On the 1-D Laplacian of order 400, 60 steps: every Ritz value lies within its residual bound of
/// an eigenvalue (Weinstein's theorem), each bound is the true residual, and the Ritz vectors are
/// orthonormal. (The bottom of this spectrum is clustered, relative gap `5·10⁻⁵`: 60 steps do not
/// resolve it, and the bounds say so.)
#[test]
fn every_ritz_value_is_within_its_bound_of_an_eigenvalue() {
    let n = 400;
    let operator = Laplacian(n);
    let ritz = lanczos(
        &operator,
        &start(n),
        Lanczos {
            steps: 60,
            tolerance: 1e-12,
        },
    );
    let spectrum = laplacian_spectrum(n);
    assert_eq!(ritz.steps, 60);
    for (k, x) in ritz.vectors.iter().enumerate() {
        let nearest = spectrum
            .iter()
            .map(|l| (l - ritz.values[k]).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest <= ritz.residuals[k] + 1e-12,
            "pair {k}: {nearest:e} > {:e}",
            ritz.residuals[k]
        );
        let mut ax = vec![0.0; n];
        operator.apply(x, &mut ax);
        let residual = ax
            .iter()
            .zip(x)
            .map(|(a, b)| (a - ritz.values[k] * b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (residual - ritz.residuals[k]).abs() < 1e-8 * (1.0 + residual),
            "pair {k}: true {residual:e}, bound {:e}",
            ritz.residuals[k]
        );
        for (j, y) in ritz.vectors.iter().enumerate().take(k + 1) {
            let inner: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
            let want = if j == k { 1.0 } else { 0.0 };
            assert!((inner - want).abs() < 1e-10, "⟨{k}, {j}⟩ = {inner}");
        }
    }
}

/// With both ends of the spectrum separated -- `−3` and `5` beside 398 values in `[0, 1]` -- 30
/// steps resolve both to `10⁻¹⁰`, with bounds as small.
#[test]
fn lanczos_resolves_separated_ends() {
    let n = 400;
    let mut spectrum: Vec<f64> = (0..n - 2).map(|i| i as f64 / (n - 3) as f64).collect();
    spectrum.extend([-3.0, 5.0]);
    let ritz = lanczos(
        &Diagonal(spectrum),
        &start(n),
        Lanczos {
            steps: 30,
            tolerance: 1e-12,
        },
    );
    let (bottom, top) = (ritz.values[0], ritz.values[ritz.steps - 1]);
    assert!(
        (bottom + 3.0).abs() < 1e-10 && ritz.residuals[0] < 1e-8,
        "{bottom} {:e}",
        ritz.residuals[0]
    );
    assert!(
        (top - 5.0).abs() < 1e-10 && ritz.residuals[ritz.steps - 1] < 1e-8,
        "{top}"
    );
}

/// An operator with three distinct eigenvalues spans a three-dimensional Krylov space: the run stops
/// there with those eigenvalues exactly and zero residual bounds.
#[test]
fn lanczos_stops_on_an_invariant_subspace() {
    let spectrum: Vec<f64> = (0..300).map(|i| [-1.0, 0.5, 4.0][i % 3]).collect();
    let ritz = lanczos(
        &Diagonal(spectrum),
        &start(300),
        Lanczos {
            steps: 50,
            tolerance: 1e-10,
        },
    );
    assert_eq!(ritz.steps, 3);
    for (got, want) in ritz.values.iter().zip([-1.0, 0.5, 4.0]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert!(
        ritz.residuals.iter().all(|&r| r == 0.0),
        "{:?}",
        ritz.residuals
    );
}

/// A zero start vector spans nothing and is refused.
#[test]
#[should_panic(expected = "zero start vector")]
fn a_zero_start_is_refused() {
    let _ = lanczos(
        &Laplacian(5),
        &[0.0; 5],
        Lanczos {
            steps: 3,
            tolerance: 1e-12,
        },
    );
}
