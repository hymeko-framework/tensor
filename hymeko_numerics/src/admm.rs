//! **Consensus ADMM** over blocks that share variables (plan 2026-10-19-numerics-optimizers, O5).
//!
//! ```text
//! min Σ_i f_i(x_i)   subject to   x_i = z_{S_i}            (general-form consensus)
//! x_i ← argmin f_i(x) + ρ/2 ‖x − z_{S_i} + u_i‖²            (each block, independently)
//! z_j ← mean over the blocks i holding j of (x_i + u_i)_j
//! u_i ← u_i + x_i − z_{S_i}
//! ```
//!
//! Boyd, Parikh, Chu, Peleato and Eckstein (2011), §7.2; stopping on the primal residual
//! `‖x − z̃‖` and the dual residual `ρ ‖z̃ − z̃_prev‖` with the tolerances of their §3.3. A block knows
//! only its own term and the global indices it holds; the solve of its proximal step is its own --
//! a factorisation, a warm-started CG, a closed form.

use crate::dot;

/// **One block of a consensus problem**: a term `f_i` over a copy of some global variables.
pub trait ProximalBlock {
    /// Why a proximal step is refused.
    type Error;

    /// The global indices this block holds a copy of, in its local order.
    fn indices(&self) -> &[usize];

    /// `x ← argmin f_i(x) + ρ/2 ‖x − v‖²`; `x` holds the previous iterate on entry.
    ///
    /// # Errors
    /// The block's refusal.
    ///
    /// # Preconditions
    /// `v.len() == x.len() == self.indices().len()`; `rho > 0`.
    fn prox(&mut self, v: &[f64], rho: f64, x: &mut [f64]) -> Result<(), Self::Error>;
}

/// ADMM's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Admm {
    /// The penalty `ρ`.
    pub rho: f64,
    /// Iterations allowed.
    pub max_iterations: usize,
    /// `ε_abs`.
    pub absolute_tolerance: f64,
    /// `ε_rel`.
    pub relative_tolerance: f64,
}

/// What an ADMM run did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmmReport {
    /// Iterations taken.
    pub iterations: usize,
    /// The last primal residual `‖x − z̃‖`.
    pub primal_residual: f64,
    /// The last dual residual `ρ ‖z̃ − z̃_prev‖`.
    pub dual_residual: f64,
    /// Whether both fell below their tolerances.
    pub converged: bool,
}

/// **Consensus ADMM** on `blocks` from the global estimate `z`, which holds the consensus on
/// return. Every block's copy starts at `z` restricted, its scaled dual at zero.
///
/// # Errors
/// What a block refuses.
///
/// # Preconditions
/// Every block's indices are below `z.len()`; every global variable is held by at least one block
/// (one held by none keeps its value).
pub fn admm<B: ProximalBlock>(
    blocks: &mut [B],
    z: &mut [f64],
    settings: Admm,
) -> Result<AdmmReport, B::Error> {
    let mut holders = vec![0usize; z.len()];
    for block in blocks.iter() {
        for &j in block.indices() {
            holders[j] += 1;
        }
    }
    let local =
        |block: &B, z: &[f64]| -> Vec<f64> { block.indices().iter().map(|&j| z[j]).collect() };
    let mut x: Vec<Vec<f64>> = blocks.iter().map(|b| local(b, z)).collect();
    let mut u: Vec<Vec<f64>> = x.iter().map(|xi| vec![0.0; xi.len()]).collect();
    let copies: usize = x.iter().map(Vec::len).sum();
    let rho = settings.rho;
    let mut report = AdmmReport {
        iterations: 0,
        primal_residual: f64::INFINITY,
        dual_residual: f64::INFINITY,
        converged: false,
    };
    let mut sum = vec![0.0; z.len()];
    for iteration in 1..=settings.max_iterations {
        for ((block, xi), ui) in blocks.iter_mut().zip(&mut x).zip(&u) {
            let v: Vec<f64> = block
                .indices()
                .iter()
                .zip(ui)
                .map(|(&j, uj)| z[j] - uj)
                .collect();
            block.prox(&v, rho, xi)?;
        }
        sum.iter_mut().for_each(|s| *s = 0.0);
        for ((block, xi), ui) in blocks.iter().zip(&x).zip(&u) {
            for ((&j, xj), uj) in block.indices().iter().zip(xi).zip(ui) {
                sum[j] += xj + uj;
            }
        }
        let previous = z.to_vec();
        for ((zj, s), &h) in z.iter_mut().zip(&sum).zip(&holders) {
            if h > 0 {
                *zj = s / h as f64;
            }
        }
        let (mut primal, mut dual, mut x_norm, mut z_norm, mut u_norm) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for ((block, xi), ui) in blocks.iter().zip(&x).zip(&mut u) {
            for ((&j, xj), uj) in block.indices().iter().zip(xi).zip(ui.iter_mut()) {
                let gap = xj - z[j];
                *uj += gap;
                primal += gap * gap;
                let moved = z[j] - previous[j];
                dual += moved * moved;
                x_norm += xj * xj;
                z_norm += z[j] * z[j];
            }
            u_norm += dot(ui, ui);
        }
        report.iterations = iteration;
        report.primal_residual = primal.sqrt();
        report.dual_residual = rho * dual.sqrt();
        let scale = (copies as f64).sqrt() * settings.absolute_tolerance;
        let primal_tolerance =
            scale + settings.relative_tolerance * x_norm.sqrt().max(z_norm.sqrt());
        let dual_tolerance = scale + settings.relative_tolerance * rho * u_norm.sqrt();
        if report.primal_residual <= primal_tolerance && report.dual_residual <= dual_tolerance {
            report.converged = true;
            break;
        }
    }
    Ok(report)
}
