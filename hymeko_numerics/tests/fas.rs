//! FAS on a nonlinear two-point problem: `−u'' + λ u³ = f` on `(0, 1)`, `u = 0` at both ends,
//! manufactured from `u* = sin(πx)`. Second-order differences, nonlinear Gauss--Seidel (one Newton
//! step per point), injection for the state, full weighting for the residual, linear interpolation.

use std::f64::consts::PI;

use hymeko_numerics::{fas, Fas, FasHierarchy, FasLevel};

const LAMBDA: f64 = 10.0;

/// The interior points of `N` intervals.
struct Level {
    n: usize,
    h: f64,
}

impl Level {
    fn new(intervals: usize) -> Self {
        Self {
            n: intervals - 1,
            h: 1.0 / intervals as f64,
        }
    }

    fn at(&self, u: &[f64], i: isize) -> f64 {
        if i < 0 || i as usize >= self.n {
            0.0
        } else {
            u[i as usize]
        }
    }
}

impl FasLevel for Level {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, u: &[f64], out: &mut [f64]) {
        let h2 = self.h * self.h;
        for (i, o) in out.iter_mut().enumerate() {
            let k = i as isize;
            *o = (2.0 * u[i] - self.at(u, k - 1) - self.at(u, k + 1)) / h2
                + LAMBDA * u[i] * u[i] * u[i];
        }
    }

    fn smooth(&self, u: &mut [f64], f: &[f64]) {
        let h2 = self.h * self.h;
        for i in 0..self.n {
            let k = i as isize;
            let residual = (2.0 * u[i] - self.at(u, k - 1) - self.at(u, k + 1)) / h2
                + LAMBDA * u[i] * u[i] * u[i]
                - f[i];
            u[i] -= residual / (2.0 / h2 + 3.0 * LAMBDA * u[i] * u[i]);
        }
    }
}

struct Hierarchy {
    levels: Vec<Level>,
}

impl Hierarchy {
    fn new(intervals: usize, coarsest: usize) -> Self {
        let mut levels = Vec::new();
        let mut n = intervals;
        while n >= coarsest {
            levels.push(Level::new(n));
            n /= 2;
        }
        Self { levels }
    }
}

impl FasHierarchy for Hierarchy {
    type Level = Level;
    fn depth(&self) -> usize {
        self.levels.len()
    }
    fn level(&self, l: usize) -> &Level {
        &self.levels[l]
    }
    fn restrict_state(&self, _: usize, fine: &[f64], coarse: &mut [f64]) {
        for (j, c) in coarse.iter_mut().enumerate() {
            *c = fine[2 * j + 1];
        }
    }
    fn restrict_residual(&self, _: usize, fine: &[f64], coarse: &mut [f64]) {
        for (j, c) in coarse.iter_mut().enumerate() {
            *c = 0.25 * fine[2 * j] + 0.5 * fine[2 * j + 1] + 0.25 * fine[2 * j + 2];
        }
    }
    fn prolong_add(&self, _: usize, coarse: &[f64], fine: &mut [f64]) {
        let c = |j: isize| {
            if j < 0 || j as usize >= coarse.len() {
                0.0
            } else {
                coarse[j as usize]
            }
        };
        for (i, f) in fine.iter_mut().enumerate() {
            let k = i as isize;
            *f += if i % 2 == 1 {
                c((k - 1) / 2)
            } else {
                0.5 * (c(k / 2 - 1) + c(k / 2))
            };
        }
    }
}

fn rhs(level: &Level) -> Vec<f64> {
    (1..=level.n)
        .map(|i| {
            let s = (PI * i as f64 * level.h).sin();
            PI * PI * s + LAMBDA * s * s * s
        })
        .collect()
}

const SETTINGS: Fas = Fas {
    cycles: 50,
    tolerance: 1e-10,
    pre: 2,
    post: 2,
    coarse_sweeps: 50,
};

/// The residual falls by `10¹⁰` in a number of V-cycles that does not grow with the grid, and the
/// discrete solution is second-order close to `sin(πx)`.
#[test]
fn fas_solves_a_nonlinear_problem_in_a_grid_independent_number_of_cycles() {
    let mut counts = Vec::new();
    for intervals in [64usize, 128, 256, 512] {
        let hierarchy = Hierarchy::new(intervals, 4);
        let finest = hierarchy.level(0);
        let f = rhs(finest);
        let mut u = vec![0.0; finest.n];
        let report = fas(&hierarchy, &mut u, &f, SETTINGS);
        assert!(report.converged, "{intervals}: {report:?}");
        let error = u
            .iter()
            .enumerate()
            .map(|(i, v)| (v - (PI * (i + 1) as f64 * finest.h).sin()).abs())
            .fold(0.0f64, f64::max);
        assert!(
            error < 2.0 * finest.h * finest.h,
            "{intervals}: error {error}"
        );
        counts.push(report.cycles);
    }
    println!("FAS cycles at N = 64, 128, 256, 512: {counts:?}");
    let (lo, hi) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
    assert!(hi - lo <= 1 && *hi <= 12, "cycles {counts:?}");
}

/// On one level FAS is the smoother: `coarse_sweeps` sweeps a cycle.
#[test]
fn a_single_level_hierarchy_is_its_smoother() {
    let hierarchy = Hierarchy::new(8, 8);
    assert_eq!(hierarchy.depth(), 1);
    let f = rhs(hierarchy.level(0));
    let mut by_fas = vec![0.0; 7];
    let settings = Fas {
        cycles: 1,
        tolerance: 0.0,
        coarse_sweeps: 3,
        ..SETTINGS
    };
    let report = fas(&hierarchy, &mut by_fas, &f, settings);
    let mut by_hand = vec![0.0; 7];
    for _ in 0..3 {
        hierarchy.level(0).smooth(&mut by_hand, &f);
    }
    assert_eq!(by_fas, by_hand);
    assert!(!report.converged && report.residual < 1.0);
}
