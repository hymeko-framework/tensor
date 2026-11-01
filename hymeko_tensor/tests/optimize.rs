//! Forward-mode duals and the augmented-Lagrangian solver on problems with known answers.

use hymeko_tensor::dual::Dual;
use hymeko_tensor::optimize::{AugmentedLagrangian, Constrained, Point, SolveError};

/// A closure-backed problem, so each test states its objective and constraints inline.
struct Problem<F> {
    bounds: Vec<(f64, f64)>,
    point: F,
}

impl<F: Fn(&[Dual]) -> Result<Point, &'static str>> Constrained for Problem<F> {
    type Error = &'static str;

    fn dimension(&self) -> usize {
        self.bounds.len()
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }

    fn point(&self, x: &[f64]) -> Result<Point, Self::Error> {
        let n = x.len();
        let variables: Vec<Dual> = x.iter().enumerate().map(|(k, &v)| Dual::variable(v, k, n)).collect();
        (self.point)(&variables)
    }
}

fn square(d: &Dual) -> Dual {
    d.mul(d)
}

#[test]
fn product_and_quotient_carry_exact_gradients() {
    let x = Dual::variable(3.0, 0, 2);
    let y = Dual::variable(2.0, 1, 2);
    let p = x.mul(&y);
    assert_eq!(p.value, 6.0);
    assert_eq!(p.grad, vec![2.0, 3.0]);
    // d(x/y) = (1/y, -x/y^2)
    let q = x.div(&y);
    assert_eq!(q.value, 1.5);
    assert_eq!(q.grad, vec![0.5, -0.75]);
    let s = x.sub(&y).scale(2.0).neg();
    assert_eq!(s.value, -2.0);
    assert_eq!(s.grad, vec![-2.0, 2.0]);
}

#[test]
fn an_equality_constrained_quadratic_reaches_its_projection() {
    // min (x-2)^2 + (y-1)^2  s.t.  x + y = 1   ->   (1, 0)
    let problem = Problem {
        bounds: vec![(-10.0, 10.0); 2],
        point: |v: &[Dual]| {
            let n = v.len();
            let objective = square(&v[0].sub(&Dual::constant(2.0, n))).add(&square(&v[1].sub(&Dual::constant(1.0, n))));
            let h = v[0].add(&v[1]).sub(&Dual::constant(1.0, n));
            Ok(Point { objective, equalities: vec![h], inequalities: vec![] })
        },
    };
    let solution = AugmentedLagrangian::default().solve(&problem, &[0.0, 0.0]).expect("converges");
    assert!((solution.x[0] - 1.0).abs() < 1e-7, "{:?}", solution.x);
    assert!(solution.x[1].abs() < 1e-7, "{:?}", solution.x);
}

#[test]
fn an_active_inequality_holds_the_optimum_on_its_boundary() {
    // min (x-3)^2  s.t.  1 - x >= 0   ->   x = 1, reached on the constraint
    let problem = Problem {
        bounds: vec![(-10.0, 10.0)],
        point: |v: &[Dual]| {
            let n = v.len();
            let objective = square(&v[0].sub(&Dual::constant(3.0, n)));
            let g = Dual::constant(1.0, n).sub(&v[0]);
            Ok(Point { objective, equalities: vec![], inequalities: vec![g] })
        },
    };
    let solution = AugmentedLagrangian::default().solve(&problem, &[-4.0]).expect("converges");
    assert!((solution.x[0] - 1.0).abs() < 1e-7, "{:?}", solution.x);
}

#[test]
fn a_bound_the_gradient_pushes_through_pins_the_variable() {
    // min -x on [0, 2]   ->   x = 2, with no constraint at all
    let problem = Problem {
        bounds: vec![(0.0, 2.0)],
        point: |v: &[Dual]| Ok(Point { objective: v[0].neg(), equalities: vec![], inequalities: vec![] }),
    };
    let solution = AugmentedLagrangian::default().solve(&problem, &[0.5]).expect("converges");
    assert_eq!(solution.x, vec![2.0]);
}

#[test]
fn an_evaluation_failure_is_returned_not_hidden() {
    let problem = Problem {
        bounds: vec![(-1.0, 1.0)],
        point: |_: &[Dual]| Err("outside the model's domain"),
    };
    let error = AugmentedLagrangian::default().solve(&problem, &[0.0]).unwrap_err();
    assert_eq!(error, SolveError::Evaluation("outside the model's domain"));
}

#[test]
fn an_exhausted_budget_reports_its_residuals() {
    let problem = Problem {
        bounds: vec![(-10.0, 10.0)],
        point: |v: &[Dual]| {
            let n = v.len();
            let h = v[0].sub(&Dual::constant(5.0, n));
            Ok(Point { objective: square(&v[0]), equalities: vec![h], inequalities: vec![] })
        },
    };
    let solver = AugmentedLagrangian { outer_iterations: 0, ..AugmentedLagrangian::default() };
    match solver.solve(&problem, &[0.0]) {
        Err(SolveError::NotConverged { iterations: 0, violation, .. }) => assert_eq!(violation, 5.0),
        other => panic!("expected NotConverged, got {other:?}"),
    }
}
