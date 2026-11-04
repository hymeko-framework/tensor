//! Consensus ADMM on blocks with closed-form proximal steps.

use hymeko_numerics::{admm, Admm, ProximalBlock};

/// `f(x) = ½ ‖x − a‖²` over the global variables `indices`: prox `x = (a + ρ v) / (1 + ρ)`.
struct Pull {
    indices: Vec<usize>,
    a: Vec<f64>,
}

impl ProximalBlock for Pull {
    type Error = &'static str;
    fn indices(&self) -> &[usize] {
        &self.indices
    }
    fn prox(&mut self, v: &[f64], rho: f64, x: &mut [f64]) -> Result<(), &'static str> {
        for ((xi, ai), vi) in x.iter_mut().zip(&self.a).zip(v) {
            *xi = (ai + rho * vi) / (1.0 + rho);
        }
        Ok(())
    }
}

fn blocks() -> Vec<Pull> {
    vec![
        Pull {
            indices: vec![0, 1],
            a: vec![1.0, 2.0],
        },
        Pull {
            indices: vec![1, 2, 3],
            a: vec![4.0, -1.0, 0.5],
        },
        Pull {
            indices: vec![3, 0],
            a: vec![2.5, 3.0],
        },
    ]
}

const SETTINGS: Admm = Admm {
    rho: 1.0,
    max_iterations: 500,
    absolute_tolerance: 1e-12,
    relative_tolerance: 1e-12,
};

/// `Σ_i ½ ‖x_{S_i} − a_i‖²` is minimised by each variable's mean over the blocks holding it.
#[test]
fn the_consensus_of_pulls_is_each_variables_mean() {
    let mut z = vec![0.0; 4];
    let report = admm(&mut blocks(), &mut z, SETTINGS).unwrap();
    assert!(report.converged, "{report:?}");
    let means = [
        (1.0 + 3.0) / 2.0,
        (2.0 + 4.0) / 2.0,
        -1.0,
        (0.5 + 2.5) / 2.0,
    ];
    for (zi, m) in z.iter().zip(&means) {
        assert!((zi - m).abs() < 1e-10, "{z:?} vs {means:?}");
    }
}

#[test]
fn the_iteration_budget_ends_an_unconverged_run() {
    let mut z = vec![0.0; 4];
    let settings = Admm {
        max_iterations: 2,
        ..SETTINGS
    };
    let report = admm(&mut blocks(), &mut z, settings).unwrap();
    assert_eq!(report.iterations, 2);
    assert!(!report.converged);
    assert!(report.primal_residual > 0.0 && report.dual_residual > 0.0);
}

struct Refusing(Vec<usize>);

impl ProximalBlock for Refusing {
    type Error = &'static str;
    fn indices(&self) -> &[usize] {
        &self.0
    }
    fn prox(&mut self, _: &[f64], _: f64, _: &mut [f64]) -> Result<(), &'static str> {
        Err("singular")
    }
}

#[test]
fn a_blocks_refusal_is_returned_and_z_is_untouched() {
    let mut z = vec![1.0, 2.0];
    let error = admm(&mut [Refusing(vec![0, 1])], &mut z, SETTINGS).unwrap_err();
    assert_eq!((error, z), ("singular", vec![1.0, 2.0]));
}
