//! Conservative weighted diffusion over positive capacities and pair conductances.
//! No graph, mesh, Core or application dependency. Uses the existing GMRES solver.
use crate::{gmres, Operator, Report};

/// Symmetric negative semidefinite operator in capacity-normalised coordinates.
#[derive(Clone, Debug)]
pub struct Diffusion {
    capacity: Vec<f64>,
    sqrt_capacity: Vec<f64>,
    conductances: Vec<(usize, usize, f64)>,
}
impl Diffusion {
    /// Build a closed transport system. Each conductance exchanges equal quantities.
    pub fn new(
        capacity: Vec<f64>,
        conductances: Vec<(usize, usize, f64)>,
    ) -> Result<Self, &'static str> {
        if capacity.is_empty() || capacity.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("capacities must be positive and finite");
        }
        let n = capacity.len();
        if conductances
            .iter()
            .any(|&(a, b, g)| a >= n || b >= n || a == b || !g.is_finite() || g < 0.0)
        {
            return Err("invalid conductance");
        }
        let sqrt_capacity = capacity.iter().map(|v| v.sqrt()).collect();
        Ok(Self {
            capacity,
            sqrt_capacity,
            conductances,
        })
    }
    /// Independent state count.
    pub fn len(&self) -> usize {
        self.capacity.len()
    }
    /// A successfully constructed system always has state.
    pub fn is_empty(&self) -> bool {
        self.capacity.is_empty()
    }
    /// Capacity-weighted conserved quantity (e.g. mass for concentration).
    pub fn quantity(&self, state: &[f64]) -> f64 {
        self.capacity.iter().zip(state).map(|(v, c)| v * c).sum()
    }
    /// Quadratic Lyapunov functional, not necessarily thermodynamic energy.
    pub fn energy(&self, state: &[f64]) -> f64 {
        0.5 * self
            .capacity
            .iter()
            .zip(state)
            .map(|(v, c)| v * c * c)
            .sum::<f64>()
    }
    /// Transfer concentrations using a complete coupling of source/target capacities.
    /// Each tuple is (source index, target index, shared capacity). Both margins
    /// must match their capacities; no quantity is silently discarded or created.
    pub fn remap(
        &self,
        state: &[f64],
        target: &Self,
        overlaps: &[(usize, usize, f64)],
    ) -> Result<Vec<f64>, &'static str> {
        if state.len() != self.len() || state.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err("invalid source concentration");
        }
        let mut source_coverage = vec![0.0; self.len()];
        let mut target_coverage = vec![0.0; target.len()];
        let mut next = vec![0.0; target.len()];
        for &(a, b, volume) in overlaps {
            if a >= self.len() || b >= target.len() || !volume.is_finite() || volume <= 0.0 {
                return Err("invalid transfer weight");
            }
            source_coverage[a] += volume;
            target_coverage[b] += volume;
            next[b] += volume * state[a];
        }
        for (coverage, capacity) in [
            (&source_coverage, &self.capacity),
            (&target_coverage, &target.capacity),
        ] {
            if coverage
                .iter()
                .zip(capacity)
                .any(|(v, c)| !v.is_finite() || (v - c).abs() > c * 1e-12)
            {
                return Err("transfer does not cover both capacity partitions");
            }
        }
        for (value, capacity) in next.iter_mut().zip(&target.capacity) {
            *value /= capacity;
        }
        if next.iter().any(|v| !v.is_finite()) {
            return Err("transfer concentration overflow");
        }
        Ok(next)
    }
    /// Backward Euler. Reject invalid input or failed solve without changing state.
    pub fn step(&self, state: &mut [f64], dt: f64, tolerance: f64) -> Result<Report, &'static str> {
        if state.len() != self.len()
            || state.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !dt.is_finite()
            || dt <= 0.0
            || !tolerance.is_finite()
            || !(0.0..1.0).contains(&tolerance)
            || tolerance == 0.0
        {
            return Err("invalid diffusion step");
        }
        let rhs: Vec<_> = state
            .iter()
            .zip(&self.sqrt_capacity)
            .map(|(v, s)| v * s)
            .collect();
        let mut next = rhs.clone();
        let report = gmres(self, dt, &rhs, &mut next, tolerance, 32, 512);
        if !report.converged {
            return Err("diffusion solve did not converge");
        }
        for (v, s) in next.iter_mut().zip(&self.sqrt_capacity) {
            *v /= s;
        }
        if next.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("diffusion solve produced invalid concentration");
        }
        state.copy_from_slice(&next);
        Ok(report)
    }
}
impl Operator for Diffusion {
    fn dim(&self) -> usize {
        self.len()
    }
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for &(a, b, g) in &self.conductances {
            let flux = g * (x[a] / self.sqrt_capacity[a] - x[b] / self.sqrt_capacity[b]);
            out[a] -= flux / self.sqrt_capacity[a];
            out[b] += flux / self.sqrt_capacity[b];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conservative_remap_preserves_quantity_and_constants_but_averages_detail() {
        let fine = Diffusion::new(vec![1.0, 3.0], vec![]).unwrap();
        let coarse = Diffusion::new(vec![4.0], vec![]).unwrap();
        let down = [(0, 0, 1.0), (1, 0, 3.0)];
        let up = [(0, 0, 1.0), (0, 1, 3.0)];
        let mean = fine.remap(&[4.0, 0.0], &coarse, &down).unwrap();
        assert_eq!(mean, vec![1.0]);
        assert_eq!(coarse.quantity(&mean), 4.0);
        assert!(coarse.energy(&mean) < fine.energy(&[4.0, 0.0]));
        assert_eq!(coarse.remap(&mean, &fine, &up).unwrap(), vec![1.0, 1.0]);
        let constant = fine.remap(&[2.0, 2.0], &coarse, &down).unwrap();
        assert_eq!(coarse.remap(&constant, &fine, &up).unwrap(), vec![2.0, 2.0]);
        for bad in [
            vec![],
            vec![(0, 0, 4.0)],
            vec![(2, 0, 1.0)],
            vec![(0, 0, f64::NAN)],
            vec![(0, 0, -1.0)],
        ] {
            assert!(fine.remap(&[4.0, 0.0], &coarse, &bad).is_err());
        }
        assert!(fine.remap(&[4.0], &coarse, &down).is_err());
        assert!(fine.remap(&[-1.0, 0.0], &coarse, &down).is_err());
    }
    #[test]
    fn unequal_capacities_match_closed_form_backward_euler() {
        let op = Diffusion::new(vec![1.0, 2.0], vec![(0, 1, 3.0)]).unwrap();
        let mut c = vec![1.0, 0.0];
        let dt = 0.2;
        op.step(&mut c, dt, 1e-13).unwrap();
        let difference = 1.0 / (1.0 + dt * 3.0 * 1.5);
        assert!((c[0] - (1.0 + 2.0 * difference) / 3.0).abs() < 1e-12);
        assert!((c[1] - (1.0 - difference) / 3.0).abs() < 1e-12);
        assert!((op.quantity(&c) - 1.0).abs() < 1e-12);
        assert!(op.energy(&c) < 0.5);
    }
    #[test]
    fn constants_zero_and_invalid_updates_preserve_state() {
        let op = Diffusion::new(vec![1.0, 0.125, 2.0], vec![(0, 1, 2.0), (1, 2, 0.7)]).unwrap();
        for initial in [0.0, 2.0] {
            let mut c = vec![initial; 3];
            op.step(&mut c, 0.1, 1e-12).unwrap();
            assert!(c.iter().all(|v| (*v - initial).abs() < 1e-12));
            let before = c.clone();
            assert!(op.step(&mut c, f64::NAN, 1e-12).is_err());
            assert_eq!(c, before);
        }
        assert!(Diffusion::new(vec![0.0], vec![]).is_err());
        assert!(Diffusion::new(vec![1.0], vec![(0, 1, 1.0)]).is_err());
    }
}
