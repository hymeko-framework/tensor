//! Gauss--Newton and IRLS over the [`LeastSquares`] trait.
//!
//! Two problems a reader can check by hand: an exponential fit `y = a e^{b t}` whose optimum is the
//! generating `(a, b)` (exact data), and a straight-line fit with one gross outlier, which plain
//! least squares bends towards and a robust kernel must not.

use hymeko_numerics::{
    gauss_newton, irls, levenberg_marquardt, Cauchy, FixedScale, GaussNewton, GaussNewtonRuns,
    Graduated, Huber, IncrementSolver, LeastSquares, LevenbergMarquardt, MedianWithFloor,
    NormalSystem, Operator, PcgIncrement, Reweight, RobustKernel, ScaleSchedule,
};

/// A model `y(t; p)` with two parameters and its gradient in them, over scalar samples.
trait Model {
    fn value(&self, t: f64, p: [f64; 2]) -> f64;
    fn gradient(&self, t: f64, p: [f64; 2]) -> [f64; 2];
}

struct Exponential;
impl Model for Exponential {
    fn value(&self, t: f64, p: [f64; 2]) -> f64 {
        p[0] * (p[1] * t).exp()
    }
    fn gradient(&self, t: f64, p: [f64; 2]) -> [f64; 2] {
        let e = (p[1] * t).exp();
        [e, p[0] * t * e]
    }
}

struct Line;
impl Model for Line {
    fn value(&self, t: f64, p: [f64; 2]) -> f64 {
        p[0] + p[1] * t
    }
    fn gradient(&self, t: f64, _: [f64; 2]) -> [f64; 2] {
        [1.0, t]
    }
}

/// One block per sample, residual `r_k = y(t_k; p) − y_k`.
struct Fit<M> {
    model: M,
    samples: Vec<(f64, f64)>,
}

impl<M: Model> LeastSquares for Fit<M> {
    type Point = [f64; 2];

    fn dim(&self) -> usize {
        2
    }

    fn blocks(&self) -> usize {
        self.samples.len()
    }

    fn normal_system(&self, p: &[f64; 2], weights: &[f64]) -> NormalSystem {
        let (mut h, mut g) = ([0.0; 3], [0.0; 2]);
        for (&(t, y), w) in self.samples.iter().zip(weights) {
            let r = self.model.value(t, *p) - y;
            let j = self.model.gradient(t, *p);
            h[0] += w * j[0] * j[0];
            h[1] += w * j[0] * j[1];
            h[2] += w * j[1] * j[1];
            g[0] += w * j[0] * r;
            g[1] += w * j[1] * r;
        }
        NormalSystem {
            diagonal: vec![h[0], h[2]],
            upper: vec![(0, 1, h[1])],
            rhs: vec![-g[0], -g[1]],
        }
    }

    fn residual_norms(&self, p: &[f64; 2], out: &mut [f64]) {
        for (o, &(t, y)) in out.iter_mut().zip(&self.samples) {
            *o = (self.model.value(t, *p) - y).abs();
        }
    }

    fn retract(&self, p: &mut [f64; 2], delta: &[f64]) -> f64 {
        p[0] += delta[0];
        p[1] += delta[1];
        delta[0].abs().max(delta[1].abs())
    }
}

fn exponential_fit() -> Fit<Exponential> {
    let truth = [1.5, -0.8];
    Fit {
        model: Exponential,
        samples: (0..20)
            .map(|k| {
                let t = 0.1 * k as f64;
                (t, Exponential.value(t, truth))
            })
            .collect(),
    }
}

/// `y = 1 + 2 t` sampled at 21 points, sample 10 raised by 50.
fn line_with_outlier() -> Fit<Line> {
    Fit {
        model: Line,
        samples: (0..21)
            .map(|k| {
                let t = 0.1 * k as f64;
                (t, 1.0 + 2.0 * t + if k == 10 { 50.0 } else { 0.0 })
            })
            .collect(),
    }
}

const SOLVER: PcgIncrement = PcgIncrement {
    tolerance: 1e-14,
    max_iterations: 10,
};

const SETTINGS: GaussNewton = GaussNewton {
    max_iterations: 50,
    step_tolerance: 1e-12,
};

#[test]
fn gauss_newton_recovers_an_exponential_from_exact_data() {
    let fit = exponential_fit();
    let mut p = [1.0, 0.0];
    let weights = vec![1.0; fit.blocks()];
    let report = gauss_newton(&fit, &mut p, &weights, &mut SOLVER.clone(), SETTINGS).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(report.iterations < 15, "{report:?}");
    assert!(
        (p[0] - 1.5).abs() < 1e-10 && (p[1] + 0.8).abs() < 1e-10,
        "{p:?}"
    );
}

#[test]
fn gauss_newton_reports_an_unconverged_run() {
    let fit = exponential_fit();
    let mut p = [1.0, 0.0];
    let weights = vec![1.0; fit.blocks()];
    let settings = GaussNewton {
        max_iterations: 2,
        ..SETTINGS
    };
    let report = gauss_newton(&fit, &mut p, &weights, &mut SOLVER.clone(), settings).unwrap();
    assert_eq!(report.iterations, 2);
    assert!(!report.converged);
    assert!(report.last_step >= settings.step_tolerance);
}

struct Refusing;
impl IncrementSolver for Refusing {
    type Error = &'static str;

    fn solve(&mut self, _: &NormalSystem) -> Result<Vec<f64>, &'static str> {
        Err("singular")
    }
}

#[test]
fn a_solver_refusal_is_returned() {
    let fit = exponential_fit();
    let mut p = [1.0, 0.0];
    let weights = vec![1.0; fit.blocks()];
    let error = gauss_newton(&fit, &mut p, &weights, &mut Refusing, SETTINGS).unwrap_err();
    assert_eq!(error, "singular");
    assert_eq!(p, [1.0, 0.0]);
}

#[test]
fn the_normal_system_applies_its_upper_triangle_symmetrically() {
    let system = NormalSystem {
        diagonal: vec![4.0, 5.0, 6.0],
        upper: vec![(0, 2, 1.0), (1, 2, -2.0)],
        rhs: vec![0.0; 3],
    };
    let mut y = [0.0; 3];
    system.apply(&[1.0, 2.0, 3.0], &mut y);
    assert_eq!(y, [4.0 + 3.0, 10.0 - 6.0, 18.0 + 1.0 - 4.0]);
}

/// Plain least squares on the outlier line: the closed-form fit, one Gauss--Newton step.
fn plain_line() -> [f64; 2] {
    let fit = line_with_outlier();
    let mut p = [0.0, 0.0];
    let weights = vec![1.0; fit.blocks()];
    gauss_newton(&fit, &mut p, &weights, &mut SOLVER.clone(), SETTINGS).unwrap();
    p
}

fn robust_line<K: RobustKernel, S: ScaleSchedule>(
    fit: &Fit<Line>,
    kernel: K,
    schedule: S,
    rounds: usize,
) -> ([f64; 2], Vec<f64>) {
    let mut p = [0.0, 0.0];
    let mut weights = vec![1.0; fit.blocks()];
    let mut reweight = Reweight {
        blocks: 0..fit.blocks(),
        kernel,
        schedule,
    };
    let mut runs = GaussNewtonRuns {
        problem: fit,
        solver: &mut SOLVER.clone(),
        settings: SETTINGS,
    };
    irls(&mut runs, &mut p, &mut weights, &mut reweight, rounds).unwrap();
    (p, weights)
}

fn error_from_truth(p: [f64; 2]) -> f64 {
    (p[0] - 1.0).abs().max((p[1] - 2.0).abs())
}

#[test]
fn irls_with_cauchy_rejects_the_outlier_plain_least_squares_follows() {
    let plain = plain_line();
    let (robust, weights) = robust_line(
        &line_with_outlier(),
        Cauchy::<f64>::new(),
        MedianWithFloor(1e-3),
        8,
    );
    assert!(error_from_truth(plain) > 1.0, "plain {plain:?}");
    assert!(error_from_truth(robust) < 1e-3, "robust {robust:?}");
    assert!(weights[10] < 1e-4, "outlier weight {}", weights[10]);
}

#[test]
fn irls_with_huber_bounds_the_outliers_pull() {
    let plain = plain_line();
    let (robust, weights) = robust_line(
        &line_with_outlier(),
        Huber::<f64>::new(1.345),
        FixedScale(0.1),
        8,
    );
    assert!(
        error_from_truth(robust) < 0.1 * error_from_truth(plain),
        "huber {robust:?} plain {plain:?}"
    );
    assert!(weights[10] < 0.01, "outlier weight {}", weights[10]);
}

#[test]
fn an_empty_reweight_range_is_plain_gauss_newton() {
    let fit = line_with_outlier();
    let mut p = [0.0, 0.0];
    let mut weights = vec![1.0; fit.blocks()];
    let mut reweight = Reweight {
        blocks: 0..0,
        kernel: Cauchy::<f64>::new(),
        schedule: MedianWithFloor(1e-3),
    };
    let mut runs = GaussNewtonRuns {
        problem: &fit,
        solver: &mut SOLVER.clone(),
        settings: SETTINGS,
    };
    irls(&mut runs, &mut p, &mut weights, &mut reweight, 3).unwrap();
    assert!(weights.iter().all(|&w| w == 1.0));
    // Four Gauss--Newton runs instead of one: the same optimum, each run's last step a rounding.
    let plain = plain_line();
    assert!(
        (p[0] - plain[0]).abs().max((p[1] - plain[1]).abs()) < 1e-12,
        "{p:?} {plain:?}"
    );
}

#[test]
fn kernels_have_their_closed_forms() {
    let cauchy = Cauchy::<f64>::new();
    assert_eq!(cauchy.weight(0.0, 2.0), 1.0);
    assert_eq!(cauchy.weight(2.0, 2.0), 0.5);
    assert_eq!(cauchy.weight(6.0, 2.0), 0.1);
    let huber = Huber::<f64>::new(2.0);
    assert_eq!(huber.weight(3.0, 2.0), 1.0);
    assert_eq!(huber.weight(4.0, 2.0), 1.0);
    assert_eq!(huber.weight(8.0, 2.0), 0.5);
}

/// The weight in `f32` is the weight `f32` arithmetic computes, operation by operation -- not the
/// `f64` weight rounded once. The two differ, which is why the arithmetic is a type parameter.
#[test]
fn a_kernel_computes_in_its_arithmetic() {
    let s = 0.7f32;
    let (single, double) = (Cauchy::<f32>::new(), Cauchy::<f64>::new());
    let mut differing = 0;
    for k in 1..200u16 {
        let r = f32::from(k) * 0.013;
        let x = r / s;
        let by_hand = f64::from(1.0f32 / (1.0 + x * x));
        let (r, s) = (f64::from(r), f64::from(s));
        assert_eq!(single.weight(r, s), by_hand);
        if by_hand != f64::from(double.weight(r, s) as f32) {
            differing += 1;
        }
    }
    assert!(
        differing > 0,
        "f32 and once-rounded f64 weights agree on the whole grid"
    );
    let huber = Huber::<f32>::new(1.5);
    assert_eq!(huber.weight(0.3, 0.7), 1.0);
    assert_eq!(huber.weight(3.0, 1.0), f64::from(1.5f32 / 3.0f32));
}

/// The graduated schedule's closed form: round 0 the quantile, at least `start_floor × floor`;
/// later rounds shrink geometrically to the floor; the arithmetic is the type parameter's.
#[test]
fn the_graduated_schedule_shrinks_from_a_quantile_to_its_floor() {
    let norms: Vec<f64> = (1..=10).map(f64::from).collect();
    let mut gnc = Graduated::<f64>::new((9, 10), 10.0, 0.5, 1.0);
    assert_eq!(gnc.scale(0, &norms), 10.0);
    assert_eq!(gnc.scale(1, &norms), 5.0);
    assert_eq!(gnc.scale(2, &norms), 2.5);
    assert_eq!(gnc.scale(3, &norms), 1.25);
    assert_eq!(gnc.scale(4, &norms), 1.0);
    assert_eq!(gnc.scale(5, &norms), 1.0);
    // The start is held at start_floor × floor when the quantile is below it.
    let mut wide = Graduated::<f64>::new((9, 10), 20.0, 0.5, 1.0);
    assert_eq!(wide.scale(0, &norms), 20.0);
    // In f32 the products are f32 products.
    let mut single = Graduated::<f32>::new((1, 2), 1.0, 0.7, 0.05);
    let s0 = single.scale(0, &[0.3, 0.9, 1.7]);
    assert_eq!(s0, f64::from(0.9f32));
    assert_eq!(single.scale(1, &[]), f64::from(0.9f32 * 0.7f32));
    assert_eq!(MedianWithFloor(0.5).scale(7, &[0.1, 0.2, 0.3]), 0.5);
    assert_eq!(MedianWithFloor(0.0).scale(0, &[3.0, 1.0, 2.0, 4.0]), 3.0);
}

const LM: LevenbergMarquardt = LevenbergMarquardt {
    max_iterations: 200,
    step_tolerance: 1e-12,
    gradient_tolerance: 1e-14,
    initial_damping: 1e-3,
};

/// From `(a, b) = (1, 3)` the model overflows the data by `e^{5.7}`. Gauss--Newton's step drives
/// `b` to about `−2.5·10⁴`, where `e^{bt}` underflows, the Jacobian's `b` column vanishes and the
/// run stops on a zero step -- converged, at the wrong point. Levenberg--Marquardt only takes steps
/// that lower the cost and reaches the generating `(1.5, −0.8)`.
#[test]
fn levenberg_marquardt_reaches_what_gauss_newton_falsely_converges_from() {
    let fit = exponential_fit();
    let weights = vec![1.0; fit.blocks()];
    let mut gn = [1.0, 3.0];
    let report = gauss_newton(&fit, &mut gn, &weights, &mut SOLVER.clone(), SETTINGS).unwrap();
    assert!(report.converged && gn[1] < -1e3, "{report:?} {gn:?}");
    for start in [[1.0, 3.0], [0.1, 4.0], [10.0, -5.0]] {
        let mut p = start;
        let report = levenberg_marquardt(&fit, &mut p, &weights, &mut SOLVER.clone(), LM).unwrap();
        assert!(report.converged, "{start:?}: {report:?}");
        assert!(
            report.accepted < report.iterations,
            "no step was ever rejected"
        );
        assert!(
            (p[0] - 1.5).abs() < 1e-10 && (p[1] + 0.8).abs() < 1e-10,
            "{start:?}: {p:?}"
        );
    }
}

/// Rosenbrock's function as least squares, `r = (10 (y − x²), 1 − x)`, from `(−1.2, 1)`.
struct Rosenbrock;

impl LeastSquares for Rosenbrock {
    type Point = [f64; 2];
    fn dim(&self) -> usize {
        2
    }
    fn blocks(&self) -> usize {
        2
    }
    fn normal_system(&self, p: &[f64; 2], weights: &[f64]) -> NormalSystem {
        let r = [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]];
        let j = [[-20.0 * p[0], 10.0], [-1.0, 0.0]];
        let mut h = [0.0; 3];
        let mut g = [0.0; 2];
        for k in 0..2 {
            let w = weights[k];
            h[0] += w * j[k][0] * j[k][0];
            h[1] += w * j[k][0] * j[k][1];
            h[2] += w * j[k][1] * j[k][1];
            g[0] += w * j[k][0] * r[k];
            g[1] += w * j[k][1] * r[k];
        }
        NormalSystem {
            diagonal: vec![h[0], h[2]],
            upper: vec![(0, 1, h[1])],
            rhs: vec![-g[0], -g[1]],
        }
    }
    fn residual_norms(&self, p: &[f64; 2], out: &mut [f64]) {
        out[0] = (10.0 * (p[1] - p[0] * p[0])).abs();
        out[1] = (1.0 - p[0]).abs();
    }
    fn retract(&self, p: &mut [f64; 2], delta: &[f64]) -> f64 {
        p[0] += delta[0];
        p[1] += delta[1];
        delta[0].abs().max(delta[1].abs())
    }
}

#[test]
fn levenberg_marquardt_solves_rosenbrock() {
    let mut p = [-1.2, 1.0];
    let report =
        levenberg_marquardt(&Rosenbrock, &mut p, &[1.0, 1.0], &mut SOLVER.clone(), LM).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(
        (p[0] - 1.0).abs() < 1e-10 && (p[1] - 1.0).abs() < 1e-10,
        "{p:?}"
    );
    assert!(report.cost < 1e-20, "{report:?}");
}

/// At the optimum the first trial is a zero step: below the step tolerance, it stops the run as
/// converged, lowers nothing and moves nothing.
#[test]
fn a_step_below_the_tolerance_stops_even_when_rejected() {
    let mut p = [1.0, 1.0];
    let settings = LevenbergMarquardt {
        gradient_tolerance: -1.0,
        ..LM
    };
    let report = levenberg_marquardt(
        &Rosenbrock,
        &mut p,
        &[1.0, 1.0],
        &mut SOLVER.clone(),
        settings,
    )
    .unwrap();
    assert_eq!(p, [1.0, 1.0]);
    assert_eq!((report.iterations, report.accepted), (1, 0));
    assert!(report.converged);
}

/// The true increment times `10⁶`: every trial overshoots.
struct Overshooting;

impl IncrementSolver for Overshooting {
    type Error = hymeko_numerics::Report;

    fn solve(&mut self, system: &NormalSystem) -> Result<Vec<f64>, Self::Error> {
        let delta = SOLVER.clone().solve(system)?;
        Ok(delta.into_iter().map(|d| 1e6 * d).collect())
    }
}

/// A rejected trial leaves the point bit for bit and multiplies the damping by `ν = 2, 4, 8, …`.
#[test]
fn every_rejected_trial_leaves_the_point_and_grows_the_damping() {
    let fit = exponential_fit();
    let weights = vec![1.0; fit.blocks()];
    let start = [1.0, 0.0];
    let mut p = start;
    let settings = LevenbergMarquardt {
        max_iterations: 3,
        ..LM
    };
    let report = levenberg_marquardt(&fit, &mut p, &weights, &mut Overshooting, settings).unwrap();
    assert_eq!(p, start);
    assert_eq!((report.iterations, report.accepted), (3, 0));
    assert_eq!(report.damping, 64.0 * LM.initial_damping);
    assert!(!report.converged);
    let mut q = start;
    let error = levenberg_marquardt(&fit, &mut q, &weights, &mut Refusing, LM).unwrap_err();
    assert_eq!((error, q), ("singular", start));
}

/// **Trust-region Gauss--Newton** (O4) through `GaussNewtonModel`: from the starts where plain
/// Gauss--Newton converges falsely, the trust region reaches the generating `(1.5, −0.8)`.
#[test]
fn trust_region_gauss_newton_reaches_the_exponentials_truth() {
    use hymeko_numerics::{trust_region, GaussNewtonModel, TrustRegion};
    let fit = exponential_fit();
    let weights = vec![1.0; fit.blocks()];
    let model = GaussNewtonModel {
        problem: &fit,
        weights: &weights,
    };
    let settings = TrustRegion {
        max_iterations: 300,
        gradient_tolerance: 1e-12,
        step_tolerance: 1e-14,
        initial_radius: 1.0,
        max_radius: 1e3,
        acceptance: 1e-4,
        cg_max_iterations: 10,
        max_forcing: 0.5,
    };
    for start in [[1.0, 3.0], [0.1, 4.0], [10.0, -5.0]] {
        let mut p = start;
        let report = trust_region(&model, &mut p, settings);
        assert!(report.converged, "{start:?}: {report:?}");
        assert!(
            (p[0] - 1.5).abs() < 1e-9 && (p[1] + 0.8).abs() < 1e-9,
            "{start:?}: {p:?} {report:?}"
        );
    }
}
