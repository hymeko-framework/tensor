//! **Robust least squares by iteratively reweighted solves (IRLS)** (plan
//! 2026-10-19-numerics-optimizers, O1 and O1b).
//!
//! A [`WeightedSolve`] alternates with a [`Reweight`] of chosen blocks: each block's weight becomes
//! a [`RobustKernel`] of its residual norm over the scale a [`ScaleSchedule`] sets for the round, and
//! one more solve follows the last reweight. The inner solve is Gauss--Newton on a
//! [`LeastSquares`] problem ([`GaussNewtonRuns`]), or a direct weighted fit, or a caller's closure.
//!
//! The schedules are where robust estimators differ. [`MedianWithFloor`] is the classical IRLS
//! scale, correct while inliers are the majority and wrong once they are not: the median then sits
//! among the outliers. [`Graduated`] is graduated non-convexity (Yang et al. 2020): the scale starts
//! wide, seeded by a high quantile of the first solve's residuals, and shrinks geometrically to a
//! floor, so the reject boundary tightens from outside and survives an outlier majority.
//!
//! Moved here from `nagare_slam` -- the SE(2) back end's loop (Cauchy, median scale floored at
//! `10⁻³`), the manifold back end's median IRLS and its GNC schedule.
//!
//! # Arithmetic
//! A kernel's weight is computed in the [`Arithmetic`] of its type parameter, `f64` by default.
//! The parameter exists because the weight feeds a nonlinear solve that need not converge within
//! its budget: measured on `nagare_slam`'s robust back end (2026-10-19), computing the Cauchy weight
//! in `f64` instead of the `f32` it had always used moved the trajectory RMSE by `5·10⁻⁴`
//! relative, and computing it in `f32` here reproduced every pose bit for bit. A caller keeps its
//! recorded numbers by naming its arithmetic, `Cauchy::<f32>::new()`; nothing selects it at run
//! time.

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Range};

use crate::least_squares::{
    gauss_newton, GaussNewton, GaussNewtonReport, IncrementSolver, LeastSquares,
};

/// The arithmetic a kernel computes its weight in. Arguments arrive and the weight leaves as
/// `f64`; [`Arithmetic::narrow`] and [`Arithmetic::widen`] are the only conversions.
pub trait Arithmetic:
    Copy + PartialOrd + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// `1`.
    const ONE: Self;
    /// `x`, rounded to this arithmetic.
    fn narrow(x: f64) -> Self;
    /// The value as `f64`, exactly.
    fn widen(self) -> f64;
}

impl Arithmetic for f64 {
    const ONE: Self = 1.0;
    fn narrow(x: f64) -> Self {
        x
    }
    fn widen(self) -> f64 {
        self
    }
}

impl Arithmetic for f32 {
    const ONE: Self = 1.0;
    fn narrow(x: f64) -> Self {
        x as f32
    }
    fn widen(self) -> f64 {
        f64::from(self)
    }
}

/// A robust kernel's IRLS weight `w(r / s)` for a block of residual norm `r` under scale `s`.
pub trait RobustKernel {
    /// The weight.
    ///
    /// # Preconditions
    /// `residual ≥ 0`, `scale > 0`.
    fn weight(&self, residual: f64, scale: f64) -> f64;
}

/// Cauchy (Lorentzian): `w = 1 / (1 + (r/s)²)`, computed in `A`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cauchy<A = f64>(PhantomData<A>);

impl<A> Cauchy<A> {
    /// The kernel.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<A: Arithmetic> RobustKernel for Cauchy<A> {
    fn weight(&self, residual: f64, scale: f64) -> f64 {
        let x = A::narrow(residual) / A::narrow(scale);
        (A::ONE / (A::ONE + x * x)).widen()
    }
}

/// Huber: `w = 1` while `r/s ≤ k`, `k s / r` beyond, computed in `A`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Huber<A = f64> {
    /// The threshold `k`, in units of the scale.
    pub k: f64,
    arithmetic: PhantomData<A>,
}

impl<A> Huber<A> {
    /// The kernel of threshold `k`.
    #[must_use]
    pub const fn new(k: f64) -> Self {
        Self {
            k,
            arithmetic: PhantomData,
        }
    }
}

impl<A: Arithmetic> RobustKernel for Huber<A> {
    fn weight(&self, residual: f64, scale: f64) -> f64 {
        let (x, k) = (A::narrow(residual) / A::narrow(scale), A::narrow(self.k));
        if x <= k {
            1.0
        } else {
            (k / x).widen()
        }
    }
}

/// **The scale rule** of a reweight: the scale `s` for round `round` (0 first) from the reweighted
/// blocks' residual norms. A schedule may keep state between rounds.
pub trait ScaleSchedule {
    /// The scale.
    ///
    /// # Preconditions
    /// `norms` is not empty.
    fn scale(&mut self, round: usize, norms: &[f64]) -> f64;
}

/// A fixed scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedScale(pub f64);

impl ScaleSchedule for FixedScale {
    fn scale(&mut self, _: usize, _: &[f64]) -> f64 {
        self.0
    }
}

/// The upper median of the norms -- element `n / 2` of the sorted norms -- floored at the value,
/// every round. Selection and floor round nothing, so norms and a floor that came from `f32` give
/// the `f32` answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MedianWithFloor(pub f64);

impl ScaleSchedule for MedianWithFloor {
    fn scale(&mut self, _: usize, norms: &[f64]) -> f64 {
        quantile(norms, (1, 2)).max(self.0)
    }
}

/// **Graduated non-convexity's schedule**: round 0 takes the `quantile` of the norms, at least
/// `start_floor × floor`; every later round multiplies the previous scale by `shrink`, down to
/// `floor`. The arithmetic of the scale is `A`'s (see the module's note); the quantile itself is a
/// selection and rounds nothing.
///
/// GNC as `nagare_slam` measured it: quantile `9/10`, `start_floor` 10, `shrink` 0.7, `floor` the
/// inlier residual scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Graduated<A = f64> {
    /// The starting quantile as `(numerator, denominator)`: element `min(n·num/den, n − 1)`.
    pub quantile: (usize, usize),
    /// The starting scale is at least this multiple of the floor.
    pub start_floor: f64,
    /// The per-round multiplier, in `(0, 1)`.
    pub shrink: f64,
    /// The floor.
    pub floor: f64,
    current: Option<A>,
}

impl<A> Graduated<A> {
    /// The schedule, not yet started.
    #[must_use]
    pub const fn new(quantile: (usize, usize), start_floor: f64, shrink: f64, floor: f64) -> Self {
        Self {
            quantile,
            start_floor,
            shrink,
            floor,
            current: None,
        }
    }
}

impl<A: Arithmetic> ScaleSchedule for Graduated<A> {
    fn scale(&mut self, round: usize, norms: &[f64]) -> f64 {
        let floor = A::narrow(self.floor);
        let scale = match self.current {
            Some(previous) if round > 0 => larger(previous * A::narrow(self.shrink), floor),
            _ => larger(
                A::narrow(quantile(norms, self.quantile)),
                A::narrow(self.start_floor) * floor,
            ),
        };
        self.current = Some(scale);
        scale.widen()
    }
}

/// `max(a, b)` for ordered values.
fn larger<A: PartialOrd>(a: A, b: A) -> A {
    if a >= b {
        a
    } else {
        b
    }
}

/// Element `min(n·num/den, n − 1)` of the sorted norms.
///
/// # Preconditions
/// `norms` is not empty; `num ≤ den`, `den > 0`.
fn quantile(norms: &[f64], (numerator, denominator): (usize, usize)) -> f64 {
    let mut sorted = norms.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * numerator / denominator).min(sorted.len() - 1)]
}

/// **Which blocks are reweighted, and how**: the blocks in `blocks` get `kernel`'s weight of their
/// residual norm over the scale `schedule` sets; the others keep theirs. An empty range reweights
/// nothing and never consults the schedule.
#[derive(Clone, Debug)]
pub struct Reweight<K, S> {
    /// The reweighted blocks.
    pub blocks: Range<usize>,
    /// The weight of a residual over the scale.
    pub kernel: K,
    /// The scale of each round.
    pub schedule: S,
}

impl<K: RobustKernel, S: ScaleSchedule> Reweight<K, S> {
    /// Reweight `weights` from the blocks' residual norms `norms` at `round`.
    ///
    /// # Preconditions
    /// `self.blocks` is not empty and lies within both slices.
    fn apply(&mut self, round: usize, norms: &[f64], weights: &mut [f64]) {
        let norms = &norms[self.blocks.clone()];
        let s = self.schedule.scale(round, norms);
        for (w, &r) in weights[self.blocks.clone()].iter_mut().zip(norms) {
            *w = self.kernel.weight(r, s);
        }
    }
}

/// **A solve under block weights**: what IRLS alternates with its reweight.
pub trait WeightedSolve {
    /// The estimate; a slice is allowed.
    type Point: ?Sized;
    /// What one solve reports.
    type Report;
    /// Why a solve is refused.
    type Error;

    /// The residual blocks: the weights' length.
    fn blocks(&self) -> usize;

    /// Solve under `weights`, from `x` where the solve is iterative, into `x`.
    ///
    /// # Errors
    /// The solve's refusal.
    ///
    /// # Preconditions
    /// `weights.len() == self.blocks()`.
    fn solve(&mut self, x: &mut Self::Point, weights: &[f64]) -> Result<Self::Report, Self::Error>;

    /// Each block's residual norm at `x`, unweighted, into `out`.
    ///
    /// # Preconditions
    /// `out.len() == self.blocks()`.
    fn residual_norms(&self, x: &Self::Point, out: &mut [f64]);
}

/// The O1 inner solve: a [`gauss_newton`] run on `problem` with `solver`, under `settings`.
pub struct GaussNewtonRuns<'a, P, S: ?Sized> {
    /// The problem.
    pub problem: &'a P,
    /// The increment solver.
    pub solver: &'a mut S,
    /// Each run's settings.
    pub settings: GaussNewton,
}

impl<P: LeastSquares, S: IncrementSolver + ?Sized> WeightedSolve for GaussNewtonRuns<'_, P, S> {
    type Point = P::Point;
    type Report = GaussNewtonReport;
    type Error = S::Error;

    fn blocks(&self) -> usize {
        self.problem.blocks()
    }

    fn solve(&mut self, x: &mut P::Point, weights: &[f64]) -> Result<GaussNewtonReport, S::Error> {
        gauss_newton(self.problem, x, weights, self.solver, self.settings)
    }

    fn residual_norms(&self, x: &P::Point, out: &mut [f64]) {
        self.problem.residual_norms(x, out);
    }
}

/// **IRLS** on `problem` from `x`: `rounds` of a solve under `weights` and a [`Reweight`], then a
/// final solve, whose report is returned. `weights` holds the final weights on return.
///
/// # Errors
/// What the solve refuses.
///
/// # Preconditions
/// `weights.len() == problem.blocks()`; `reweight.blocks` lies within it.
pub fn irls<W: WeightedSolve + ?Sized, K: RobustKernel, S: ScaleSchedule>(
    problem: &mut W,
    x: &mut W::Point,
    weights: &mut [f64],
    reweight: &mut Reweight<K, S>,
    rounds: usize,
) -> Result<W::Report, W::Error> {
    debug_assert!(weights.len() == problem.blocks() && reweight.blocks.end <= weights.len());
    let mut norms = vec![0.0; problem.blocks()];
    for round in 0..rounds {
        problem.solve(x, weights)?;
        if !reweight.blocks.is_empty() {
            problem.residual_norms(x, &mut norms);
            reweight.apply(round, &norms, weights);
        }
    }
    problem.solve(x, weights)
}
