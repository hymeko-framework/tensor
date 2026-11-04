//! Reusable storage for the single restarted GMRES implementation.
use crate::{dot, norm, Operator, Report};
use std::mem::size_of;

/// Stopping threshold for the Euclidean norm of the true residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GmresTolerance {
    /// Divide by `max(norm(b), f64::MIN_POSITIVE)` before comparing.
    Relative(f64),
    /// Compare the residual norm directly; the report remains relative.
    Absolute(f64),
}
impl GmresTolerance {
    fn reached(self, residual: f64, scale: f64) -> bool {
        match self {
            Self::Relative(value) => residual / scale <= value,
            Self::Absolute(value) => residual <= value,
        }
    }
}

/// Restart and stopping controls for a workspace-backed GMRES solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmresConfig {
    /// Euclidean residual threshold. Finite and nonnegative.
    pub tolerance: GmresTolerance,
    /// Arnoldi directions per cycle; zero retains the existing one-direction clamp.
    pub restart: usize,
    /// Total Arnoldi-step budget. Zero checks the initial residual only.
    pub max_iterations: usize,
}

/// Caller-owned GMRES basis and scratch. No operator or right-hand side is cached.
///
/// Capacity survives restarts and calls, including changes in dimension. A warm
/// solve whose buffers fit retained capacity performs no heap allocation here.
/// The operator may allocate independently. Storage is O(n * restart + restart²).
#[derive(Default)]
pub struct GmresWorkspace {
    residual: Vec<f64>,
    work: Vec<f64>,
    basis: Vec<f64>,
    hessenberg: Vec<f64>,
    cosine: Vec<f64>,
    sine: Vec<f64>,
    projected: Vec<f64>,
    correction: Vec<f64>,
}

impl GmresWorkspace {
    /// Empty storage; allocation is deferred until a solve needs it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Owned buffer capacity in bytes, excluding allocator metadata and outputs.
    pub fn retained_bytes(&self) -> usize {
        [
            &self.residual,
            &self.work,
            &self.basis,
            &self.hessenberg,
            &self.cosine,
            &self.sine,
            &self.projected,
            &self.correction,
        ]
        .iter()
        .map(|v| v.capacity() * size_of::<f64>())
        .sum()
    }

    /// Solve `(I - theta S)x = b`, reusing this workspace.
    ///
    /// Modified Gram--Schmidt, Givens rotations and solution updates retain the
    /// allocating facade's scalar arithmetic order. Convergence is checked using
    /// the true residual at each restart/return, not only its Hessenberg estimate.
    /// An absolute tolerance avoids recomputing `norm(b)` in a caller that already
    /// has an absolute error target. `Report::residual` is always relative.
    ///
    /// # Panics
    /// If the input/output dimensions disagree with the operator, or the requested
    /// basis dimensions overflow addressable storage. Memory exhaustion follows
    /// the standard `Vec` allocation behavior. Tolerances must be finite and >= 0.
    pub fn solve<O: Operator>(
        &mut self,
        operator: &O,
        theta: f64,
        b: &[f64],
        x: &mut [f64],
        config: GmresConfig,
    ) -> Report {
        let n = operator.dim();
        assert_eq!(b.len(), n, "GMRES right-hand side dimension");
        assert_eq!(x.len(), n, "GMRES solution dimension");
        let m = config.restart.clamp(1, config.max_iterations.max(1));
        let scale = norm(b).max(f64::MIN_POSITIVE);
        self.residual.resize(n, 0.0);
        self.work.resize(n, 0.0);
        let mut total = 0;
        loop {
            let beta = self.residual_norm(operator, theta, b, x);
            let converged = config.tolerance.reached(beta, scale);
            if converged || total >= config.max_iterations {
                return Report {
                    iterations: total,
                    residual: beta / scale,
                    converged,
                };
            }
            self.prepare_cycle(n, m, beta);
            let used = self.cycle(operator, theta, total, config, scale);
            self.update_solution(x, n, m, used);
            total += used.max(1);
        }
    }

    fn residual_norm<O: Operator>(
        &mut self,
        operator: &O,
        theta: f64,
        b: &[f64],
        x: &[f64],
    ) -> f64 {
        operator.apply(x, &mut self.work);
        // Same element arithmetic and left-to-right Sum as residual() + norm(),
        // with the norm's second read of the residual vector removed.
        self.residual
            .iter_mut()
            .zip(b)
            .zip(x)
            .zip(&self.work)
            .map(|(((r, &b), &x), &work)| {
                *r = b - x + theta * work;
                *r * *r
            })
            .sum::<f64>()
            .sqrt()
    }

    fn prepare_cycle(&mut self, n: usize, m: usize, beta: f64) {
        assert!(
            m < usize::MAX && m <= usize::MAX / (m + 1) && n <= usize::MAX / m,
            "GMRES basis dimension overflow"
        );
        self.basis.clear();
        self.basis.reserve(n * m);
        self.hessenberg.resize((m + 1) * m, 0.0);
        self.hessenberg.fill(0.0);
        self.cosine.resize(m, 0.0);
        self.sine.resize(m, 0.0);
        self.projected.resize(m + 1, 0.0);
        self.projected.fill(0.0);
        self.correction.resize(m, 0.0);
        self.projected[0] = beta;
        self.basis.extend(self.residual.iter().map(|r| r / beta));
    }

    fn cycle<O: Operator>(
        &mut self,
        operator: &O,
        theta: f64,
        total: usize,
        config: GmresConfig,
        scale: f64,
    ) -> usize {
        let n = self.work.len();
        let m = self.cosine.len();
        let mut used = 0;
        for k in 0..m.min(config.max_iterations - total) {
            operator.apply(&self.basis[k * n..(k + 1) * n], &mut self.work);
            for (w, &v) in self.work.iter_mut().zip(&self.basis[k * n..(k + 1) * n]) {
                *w = v - theta * *w;
            }
            let column = k * (m + 1);
            for j in 0..=k {
                let basis = &self.basis[j * n..(j + 1) * n];
                self.hessenberg[column + j] = dot(&self.work, basis);
                for (w, &v) in self.work.iter_mut().zip(basis) {
                    *w -= self.hessenberg[column + j] * v;
                }
            }
            let arnoldi_norm = norm(&self.work);
            self.hessenberg[column + k + 1] = arnoldi_norm;
            for j in 0..k {
                let (c, s) = (self.cosine[j], self.sine[j]);
                let h = &mut self.hessenberg[column..column + m + 1];
                let value = c * h[j] + s * h[j + 1];
                h[j + 1] = -s * h[j] + c * h[j + 1];
                h[j] = value;
            }
            let h = &mut self.hessenberg[column..column + m + 1];
            let denominator = (h[k] * h[k] + h[k + 1] * h[k + 1]).sqrt();
            if denominator < 1e-300 {
                used = k;
                break;
            }
            self.cosine[k] = h[k] / denominator;
            self.sine[k] = h[k + 1] / denominator;
            h[k] = denominator;
            h[k + 1] = 0.0;
            self.projected[k + 1] = -self.sine[k] * self.projected[k];
            self.projected[k] *= self.cosine[k];
            used = k + 1;
            if arnoldi_norm < 1e-300 || config.tolerance.reached(self.projected[k + 1].abs(), scale)
            {
                break;
            }
            if k + 1 < m && total + k + 1 < config.max_iterations {
                self.basis
                    .extend(self.work.iter().map(|w| w / arnoldi_norm));
            }
        }
        used
    }

    fn update_solution(&mut self, x: &mut [f64], n: usize, m: usize, used: usize) {
        for i in (0..used).rev() {
            let mut value = self.projected[i];
            for j in i + 1..used {
                value -= self.hessenberg[j * (m + 1) + i] * self.correction[j];
            }
            self.correction[i] = value / self.hessenberg[i * (m + 1) + i];
        }
        for k in 0..used {
            for (x, &v) in x.iter_mut().zip(&self.basis[k * n..(k + 1) * n]) {
                *x += self.correction[k] * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Zero;
    impl Operator for Zero {
        fn dim(&self) -> usize {
            3
        }
        fn apply(&self, _: &[f64], y: &mut [f64]) {
            y.fill(0.0);
        }
    }
    struct Cyclic;
    impl Operator for Cyclic {
        fn dim(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f64], y: &mut [f64]) {
            y.copy_from_slice(&[x[1], x[2], -x[0]]);
        }
    }
    #[test]
    fn unused_restart_directions_are_reserved_but_never_materialized() {
        let mut workspace = GmresWorkspace::new();
        let mut x = [0.0; 3];
        let report = workspace.solve(
            &Zero,
            1.0,
            &[1.0, 2.0, 3.0],
            &mut x,
            GmresConfig {
                tolerance: GmresTolerance::Relative(1e-12),
                restart: 20,
                max_iterations: 40,
            },
        );
        assert!(report.converged);
        assert_eq!(report.iterations, 1);
        assert_eq!(workspace.basis.len(), 3);
        assert!(workspace.basis.capacity() >= 60);
        for (restart, max_iterations) in [(20, 1), (2, 3)] {
            x.fill(0.0);
            let report = workspace.solve(
                &Cyclic,
                1.0,
                &[1.0, 0.0, 0.0],
                &mut x,
                GmresConfig {
                    tolerance: GmresTolerance::Relative(1e-15),
                    restart,
                    max_iterations,
                },
            );
            assert!(!report.converged);
            assert_eq!(report.iterations, max_iterations);
            assert_eq!(
                workspace.basis.len(),
                3,
                "no unused next direction at the budget boundary"
            );
        }
    }
}
