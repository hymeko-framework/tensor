//! L-BFGS on objectives with known minimisers: Rosenbrock's valley, an ill-conditioned quadratic,
//! an objective that is not finite outside a ball, and one whose gradient lies.

use std::cell::Cell;

use hymeko_numerics::{lbfgs, Lbfgs, Objective};

const SETTINGS: Lbfgs = Lbfgs {
    memory: 8,
    max_iterations: 500,
    gradient_tolerance: 1e-10,
    armijo: 1e-4,
    curvature: 0.9,
    max_line_search: 60,
};

/// `ℝⁿ` with addition: the retraction of every Euclidean objective here.
fn add(x: &mut [f64], step: &[f64]) -> f64 {
    for (xi, s) in x.iter_mut().zip(step) {
        *xi += s;
    }
    step.iter().fold(0.0, |m, s| m.max(s.abs()))
}

struct Rosenbrock;

impl Objective for Rosenbrock {
    type Point = [f64];
    fn dim(&self) -> usize {
        2
    }
    fn value_and_gradient(&self, p: &[f64], g: &mut [f64]) -> f64 {
        let (x, y) = (p[0], p[1]);
        g[0] = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x);
        g[1] = 200.0 * (y - x * x);
        100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
    }
    fn retract(&self, x: &mut [f64], step: &[f64]) -> f64 {
        add(x, step)
    }
}

#[test]
fn lbfgs_follows_rosenbrocks_valley_to_its_minimum() {
    let mut p = vec![-1.2, 1.0];
    let report = lbfgs(&Rosenbrock, p.as_mut_slice(), SETTINGS);
    assert!(report.converged, "{report:?}");
    assert!(
        (p[0] - 1.0).abs() < 1e-8 && (p[1] - 1.0).abs() < 1e-8,
        "{p:?}"
    );
    assert!(report.value < 1e-16, "{report:?}");
}

/// `f = ½ Σ λ_i x_i² − Σ b_i x_i`, `λ_i` from 1 to `10⁴`: minimiser `b_i / λ_i`.
struct Quadratic {
    curvature: Vec<f64>,
    linear: Vec<f64>,
}

impl Objective for Quadratic {
    type Point = [f64];
    fn dim(&self) -> usize {
        self.curvature.len()
    }
    fn value_and_gradient(&self, x: &[f64], g: &mut [f64]) -> f64 {
        let mut value = 0.0;
        for i in 0..x.len() {
            g[i] = self.curvature[i] * x[i] - self.linear[i];
            value += 0.5 * self.curvature[i] * x[i] * x[i] - self.linear[i] * x[i];
        }
        value
    }
    fn retract(&self, x: &mut [f64], step: &[f64]) -> f64 {
        add(x, step)
    }
}

#[test]
fn lbfgs_minimises_an_ill_conditioned_quadratic() {
    let n = 50;
    let q = Quadratic {
        curvature: (0..n)
            .map(|i| 10f64.powf(4.0 * i as f64 / (n - 1) as f64))
            .collect(),
        linear: (0..n).map(|i| (i as f64 * 0.7).sin()).collect(),
    };
    let mut x = vec![0.0; n];
    // A weak-Wolfe search accepts t = 1 nearly always, so with 8 pairs and κ = 10⁴ the run is long:
    // an independent numpy port of the same method took 1129 iterations to 10⁻¹⁰ (00:56); with an
    // exact line search and 50 pairs it terminated in 49, as CG does -- the recursion is right.
    // Below ‖g‖ ≈ 3·10⁻⁷ the Armijo decrease falls under f's resolution (|f| ≈ 1.3) and the
    // search fails by rounding (measured 00:56): the tolerance sits above that floor.
    let settings = Lbfgs {
        max_iterations: 3000,
        gradient_tolerance: 1e-6,
        ..SETTINGS
    };
    let report = lbfgs(&q, x.as_mut_slice(), settings);
    assert!(report.converged, "{report:?}");
    for (i, ((xi, b), l)) in x.iter().zip(&q.linear).zip(&q.curvature).enumerate() {
        // |x_i − x*_i| = |g_i| / λ_i ≤ 10⁻⁶.
        let star = b / l;
        assert!((xi - star).abs() <= 1e-6, "{i}: {xi} vs {star}");
    }
}

/// `f = ½ (x − 0.9)²` on the line, `NaN` beyond `x = 1`, from `x = −10`. The unit first step is too
/// short for Wolfe's curvature condition, so the search doubles `t` to 16 -- past the fence -- and
/// must bisect back to a finite point; the run then finishes at 0.9.
struct Fenced {
    calls: Cell<usize>,
}

impl Objective for Fenced {
    type Point = [f64];
    fn dim(&self) -> usize {
        1
    }
    fn value_and_gradient(&self, x: &[f64], g: &mut [f64]) -> f64 {
        self.calls.set(self.calls.get() + 1);
        g[0] = x[0] - 0.9;
        if x[0] > 1.0 {
            return f64::NAN;
        }
        0.5 * g[0] * g[0]
    }
    fn retract(&self, x: &mut [f64], step: &[f64]) -> f64 {
        add(x, step)
    }
}

#[test]
fn a_non_finite_trial_is_shrunk_and_never_taken() {
    let objective = Fenced {
        calls: Cell::new(0),
    };
    let mut x = vec![-10.0];
    let report = lbfgs(&objective, x.as_mut_slice(), SETTINGS);
    assert!(report.converged, "{report:?}");
    assert!((x[0] - 0.9).abs() < 1e-10, "{x:?}");
    assert!(report.value.is_finite());
    assert_eq!(report.evaluations, objective.calls.get());
}

/// Rosenbrock with its gradient negated: every direction the method computes climbs.
struct Lying;

impl Objective for Lying {
    type Point = [f64];
    fn dim(&self) -> usize {
        2
    }
    fn value_and_gradient(&self, p: &[f64], g: &mut [f64]) -> f64 {
        let value = Rosenbrock.value_and_gradient(p, g);
        g.iter_mut().for_each(|v| *v = -*v);
        value
    }
    fn retract(&self, x: &mut [f64], step: &[f64]) -> f64 {
        add(x, step)
    }
}

/// No step satisfies Armijo along an ascent direction: the line search gives up within its budget,
/// the run reports no convergence, and the point is where it started.
#[test]
fn a_failed_line_search_ends_the_run_where_it_started() {
    let mut p = vec![-1.2, 1.0];
    let settings = Lbfgs {
        max_line_search: 20,
        ..SETTINGS
    };
    let report = lbfgs(&Lying, p.as_mut_slice(), settings);
    assert!(!report.converged);
    assert_eq!(report.iterations, 0);
    assert_eq!(report.evaluations, 1 + 20);
    assert_eq!(p, [-1.2, 1.0]);
}
