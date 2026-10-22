//! Steihaug's truncated CG against closed forms, and the trust-region method on Rosenbrock's
//! function with its exact Hessian -- indefinite away from the valley, where Newton's step is not
//! defined and the trust region must follow negative curvature.

use hymeko_numerics::{
    steihaug, trust_region, Identity, Jacobi, Operator, Preconditioner, StepEnding, TrustRegion,
    TrustRegionModel,
};

/// A small dense symmetric matrix, row-major.
struct Dense {
    n: usize,
    a: Vec<f64>,
}

impl Operator for Dense {
    fn dim(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = (0..self.n).map(|j| self.a[i * self.n + j] * x[j]).sum();
        }
    }
}

fn spd() -> Dense {
    Dense {
        n: 3,
        a: vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 9.0],
    }
}

const G: [f64; 3] = [1.0, -2.0, 0.5];

#[test]
fn an_interior_step_is_the_newton_step() {
    let h = spd();
    let step = steihaug(&h, &G, 1e6, &Identity, 1e-14, 50);
    assert_eq!(step.ending, StepEnding::Interior);
    let mut hp = [0.0; 3];
    h.apply(&step.step, &mut hp);
    for (a, g) in hp.iter().zip(&G) {
        assert!((a + g).abs() < 1e-12, "H p = {hp:?}, −g = {G:?}");
    }
    let euclidean = step.step.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((step.m_norm - euclidean).abs() < 1e-12);
}

#[test]
fn a_boundary_step_has_the_radius_in_the_preconditioners_norm() {
    let h = spd();
    let plain = steihaug(&h, &G, 0.1, &Identity, 1e-14, 50);
    assert_eq!(plain.ending, StepEnding::Boundary);
    let euclidean = plain.step.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((euclidean - 0.1).abs() < 1e-12, "{euclidean}");
    // Jacobi: M = diag H, so ‖p‖²_M = Σ H_ii p_i².
    let diagonal = [4.0, 3.0, 9.0];
    let jacobi = steihaug(&h, &G, 0.1, &Jacobi::of(&diagonal), 1e-14, 50);
    assert_eq!(jacobi.ending, StepEnding::Boundary);
    let m_norm = jacobi
        .step
        .iter()
        .zip(&diagonal)
        .map(|(p, d)| d * p * p)
        .sum::<f64>()
        .sqrt();
    assert!((m_norm - 0.1).abs() < 1e-12, "{m_norm}");
    assert!((jacobi.m_norm - 0.1).abs() < 1e-12);
}

/// On `diag(1, −1)` with `g = (1, 1)` the first direction `−g` has zero curvature: the step follows
/// it to the boundary and lowers the model.
#[test]
fn negative_curvature_is_followed_to_the_boundary() {
    let h = Dense {
        n: 2,
        a: vec![1.0, 0.0, 0.0, -1.0],
    };
    let g = [1.0, 1.0];
    let step = steihaug(&h, &g, 2.0, &Identity, 1e-14, 50);
    assert_eq!(step.ending, StepEnding::NegativeCurvature);
    let length = step.step.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((length - 2.0).abs() < 1e-12);
    let mut hp = [0.0; 2];
    h.apply(&step.step, &mut hp);
    let model: f64 = (0..2)
        .map(|i| g[i] * step.step[i] + 0.5 * step.step[i] * hp[i])
        .sum();
    assert!(model < 0.0, "{model}");
}

#[test]
fn a_zero_gradient_takes_no_step() {
    let step = steihaug(&spd(), &[0.0; 3], 1.0, &Identity, 1e-12, 50);
    assert_eq!((step.iterations, step.ending), (0, StepEnding::Interior));
    assert_eq!(step.step, vec![0.0; 3]);
}

/// Rosenbrock with its exact Hessian.
struct Rosenbrock;

impl TrustRegionModel for Rosenbrock {
    type Point = [f64; 2];
    type Hessian = Dense;
    type Preconditioner = Identity;

    fn dim(&self) -> usize {
        2
    }
    fn value(&self, p: &[f64; 2]) -> f64 {
        let (x, y) = (p[0], p[1]);
        100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
    }
    fn model(&self, p: &[f64; 2], g: &mut [f64]) -> Dense {
        let (x, y) = (p[0], p[1]);
        g[0] = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x);
        g[1] = 200.0 * (y - x * x);
        Dense {
            n: 2,
            a: vec![
                1200.0 * x * x - 400.0 * y + 2.0,
                -400.0 * x,
                -400.0 * x,
                200.0,
            ],
        }
    }
    fn preconditioner(&self, _: &Dense) -> Identity {
        Identity
    }
    fn retract(&self, p: &mut [f64; 2], step: &[f64]) -> f64 {
        p[0] += step[0];
        p[1] += step[1];
        step[0].abs().max(step[1].abs())
    }
}

const SETTINGS: TrustRegion = TrustRegion {
    max_iterations: 500,
    gradient_tolerance: 1e-10,
    step_tolerance: 1e-14,
    initial_radius: 1.0,
    max_radius: 100.0,
    acceptance: 1e-4,
    cg_max_iterations: 50,
    max_forcing: 0.5,
};

/// From `(−1.2, 1)`, and from `(0, 1)` where `∂²f/∂x² = −398`: both reach `(1, 1)`.
#[test]
fn the_trust_region_solves_rosenbrock_through_indefinite_curvature() {
    let at = Rosenbrock.model(&[0.0, 1.0], &mut [0.0; 2]);
    assert!(at.a[0] < 0.0, "the start is not indefinite");
    for start in [[-1.2, 1.0], [0.0, 1.0]] {
        let mut p = start;
        let report = trust_region(&Rosenbrock, &mut p, SETTINGS);
        assert!(report.converged, "{start:?}: {report:?}");
        assert!(
            (p[0] - 1.0).abs() < 1e-8 && (p[1] - 1.0).abs() < 1e-8,
            "{start:?}: {p:?}"
        );
    }
}

/// `Preconditioner` is object-safe through `steihaug`'s `&dyn` argument; `Identity` passes as one.
#[test]
fn the_preconditioner_argument_is_dynamic() {
    let m: &dyn Preconditioner = &Identity;
    let step = steihaug(&spd(), &G, 1e6, m, 1e-14, 50);
    assert_eq!(step.ending, StepEnding::Interior);
}
