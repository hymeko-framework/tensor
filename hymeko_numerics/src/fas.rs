//! **The full approximation scheme (FAS)**: nonlinear multigrid over a hierarchy of levels (plan
//! 2026-10-19-numerics-optimizers, O6).
//!
//! ```text
//! A(u) = f on the finest level;  one V-cycle at level l:
//!   smooth ν₁ times;   r = f − A(u)
//!   u_c⁰ = R̂ u,   f_c = A_c(u_c⁰) + R r,   V-cycle at level l + 1 on A_c(u_c) = f_c
//!   u ← u + P (u_c − u_c⁰);   smooth ν₂ times
//! the coarsest level is smoothed `coarse_sweeps` times.
//! ```
//!
//! The coarse level solves for the full approximation, not a correction, so the scheme needs no
//! linearisation: on a linear problem it is the correction scheme, on a nonlinear one it is
//! Brandt's FAS. A [`FasLevel`] states its operator and its smoother; a [`FasHierarchy`] its levels
//! and the transfers between neighbours -- a state restriction `R̂`, a residual restriction `R` and a
//! prolongation `P`, each the hierarchy's own.

use crate::norm;

/// **One level** of a hierarchy: its operator and its smoother.
pub trait FasLevel {
    /// The level's unknowns.
    fn dim(&self) -> usize;

    /// `A(u)` into `out`.
    fn apply(&self, u: &[f64], out: &mut [f64]);

    /// One relaxation sweep for `A(u) = f`.
    fn smooth(&self, u: &mut [f64], f: &[f64]);
}

/// **A hierarchy of levels**, 0 the finest, and the transfers between neighbours.
pub trait FasHierarchy {
    /// A level.
    type Level: FasLevel;

    /// The number of levels, at least one.
    fn depth(&self) -> usize;

    /// Level `l`.
    fn level(&self, l: usize) -> &Self::Level;

    /// `R̂`: a state at level `l` restricted to level `l + 1`, written into `coarse`.
    fn restrict_state(&self, l: usize, fine: &[f64], coarse: &mut [f64]);

    /// `R`: a residual at level `l` restricted to level `l + 1`, written into `coarse`.
    fn restrict_residual(&self, l: usize, fine: &[f64], coarse: &mut [f64]);

    /// `P`: a correction at level `l + 1` prolonged to level `l` and added into `fine`.
    fn prolong_add(&self, l: usize, coarse: &[f64], fine: &mut [f64]);
}

/// FAS's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fas {
    /// V-cycles allowed.
    pub cycles: usize,
    /// The run stops once `‖f − A(u)‖` is below this fraction of its starting value.
    pub tolerance: f64,
    /// Smoothing sweeps before the coarse correction.
    pub pre: usize,
    /// Smoothing sweeps after it.
    pub post: usize,
    /// Sweeps on the coarsest level.
    pub coarse_sweeps: usize,
}

/// What a FAS run did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FasReport {
    /// V-cycles taken.
    pub cycles: usize,
    /// `‖f − A(u)‖` relative to its start.
    pub residual: f64,
    /// Whether the tolerance was met.
    pub converged: bool,
}

/// **FAS V-cycles** on `hierarchy` for `A₀(u) = f` from `u`.
///
/// # Preconditions
/// `u.len() == f.len() == hierarchy.level(0).dim()`; `hierarchy.depth() ≥ 1`.
pub fn fas<H: FasHierarchy>(hierarchy: &H, u: &mut [f64], f: &[f64], settings: Fas) -> FasReport {
    debug_assert!(hierarchy.depth() >= 1);
    let start = residual_norm(hierarchy.level(0), u, f).max(f64::MIN_POSITIVE);
    let mut report = FasReport {
        cycles: 0,
        residual: 1.0,
        converged: false,
    };
    for cycle in 1..=settings.cycles {
        v_cycle(hierarchy, 0, u, f, settings);
        report.cycles = cycle;
        report.residual = residual_norm(hierarchy.level(0), u, f) / start;
        if report.residual <= settings.tolerance {
            report.converged = true;
            break;
        }
    }
    report
}

/// One V-cycle at level `l`.
fn v_cycle<H: FasHierarchy>(hierarchy: &H, l: usize, u: &mut [f64], f: &[f64], settings: Fas) {
    let level = hierarchy.level(l);
    if l + 1 == hierarchy.depth() {
        for _ in 0..settings.coarse_sweeps {
            level.smooth(u, f);
        }
        return;
    }
    for _ in 0..settings.pre {
        level.smooth(u, f);
    }
    let mut r = vec![0.0; u.len()];
    level.apply(u, &mut r);
    for (ri, fi) in r.iter_mut().zip(f) {
        *ri = fi - *ri;
    }
    let coarse = hierarchy.level(l + 1);
    let nc = coarse.dim();
    let mut start = vec![0.0; nc];
    hierarchy.restrict_state(l, u, &mut start);
    let mut rc = vec![0.0; nc];
    hierarchy.restrict_residual(l, &r, &mut rc);
    let mut fc = vec![0.0; nc];
    coarse.apply(&start, &mut fc);
    for (fci, rci) in fc.iter_mut().zip(&rc) {
        *fci += rci;
    }
    let mut uc = start.clone();
    v_cycle(hierarchy, l + 1, &mut uc, &fc, settings);
    for (c, s) in uc.iter_mut().zip(&start) {
        *c -= s;
    }
    hierarchy.prolong_add(l, &uc, u);
    for _ in 0..settings.post {
        level.smooth(u, f);
    }
}

/// `‖f − A(u)‖`.
fn residual_norm<L: FasLevel>(level: &L, u: &[f64], f: &[f64]) -> f64 {
    let mut r = vec![0.0; u.len()];
    level.apply(u, &mut r);
    for (ri, fi) in r.iter_mut().zip(f) {
        *ri = fi - *ri;
    }
    norm(&r)
}
