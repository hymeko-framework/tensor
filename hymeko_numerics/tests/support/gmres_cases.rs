//! Deterministic inputs shared by the pre-change capture and workspace checks.
#![allow(dead_code)] // Different test binaries exercise different parts of this input catalog.
use hymeko_numerics::Operator;
use std::cell::Cell;

pub struct Case {
    pub n: usize,
    pub variant: usize,
    pub restart: usize,
    pub budget: usize,
    pub calls: Cell<usize>,
}
impl Case {
    pub fn catalog() -> Vec<Self> {
        let mut cases = Vec::new();
        for n in [0, 1, 2, 5, 17, 65] {
            for variant in [0, 1] {
                for restart in [0, 2, 8] {
                    for budget in [0, 1, 3, 17] {
                        cases.push(Self {
                            n,
                            variant,
                            restart,
                            budget,
                            calls: Cell::new(0),
                        });
                    }
                }
            }
        }
        cases
    }
    pub fn rhs(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| ((i * 7 + self.variant) % 11) as f64 / 8.0 - 0.5)
            .collect()
    }
    pub fn initial(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                if self.variant == 0 {
                    0.0
                } else {
                    (i % 5) as f64 / 16.0
                }
            })
            .collect()
    }
    pub fn coefficient(&self, row: usize, col: usize) -> f64 {
        if row == col {
            -((row % 3 + 1) as f64) / 8.0
        } else if row + 1 == col {
            0.5
        } else if col + 1 == row {
            if self.variant == 0 {
                -0.5
            } else {
                -0.25
            }
        } else {
            0.0
        }
    }
}
impl Operator for Case {
    fn dim(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.calls.set(self.calls.get() + 1);
        for (i, output) in y.iter_mut().enumerate() {
            *output = self.coefficient(i, i) * x[i];
            if i + 1 < self.n {
                *output += self.coefficient(i, i + 1) * x[i + 1];
            }
            if i > 0 {
                *output += self.coefficient(i, i - 1) * x[i - 1];
            }
        }
    }
}
