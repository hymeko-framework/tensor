//! Port-Hamiltonian systems over a signed hypergraph.
//!
//! ```text
//! x' = (J - R) grad H(x) + G u,      J skew,  R positive semidefinite
//! ```
//!
//! The content of this module is that **`J` is not a matrix somebody wrote down.** It is the signed
//! incidence of a hypergraph, and the skew-symmetry that makes the energy balance hold is a
//! consequence of the sign convention rather than a property to be checked after the fact.
//!
//! # The structure is the hypergraph
//!
//! A G-SPHF is `(V, E, B, sigma, P)`: vertices, hyperedges, an incidence relation, a sign on each
//! incidence (`-1` source, `+1` target), and a partition into patches. Read as a physical network:
//!
//! | G-SPHF   | port-Hamiltonian                                                     |
//! | -------- | -------------------------------------------------------------------- |
//! | `V`      | energy storage — one state coordinate each                           |
//! | `E`      | power ports — where energy moves between coordinates                 |
//! | `sigma`  | the DIRECTION of that flow, which is the sign in `J`                 |
//! | `P`      | subsystems, which is what lets two of these be composed              |
//!
//! Each port contributes `w (e_u e_v^T - e_v e_u^T)` summed over its sources `u` and targets `v`.
//! That is antisymmetric term by term, so `J` is skew whatever the hypergraph looks like — a chain,
//! a matching, a ternary junction — and `e^T J e = 0` identically. **Energy cannot be created by the
//! interconnection, because the interconnection is a sign convention.**
//!
//! # Three systems, one structure
//!
//! * A driven RLC circuit coupled to a field mode is a **chain**: charge to flux to displacement to
//!   magnetic flux. Four vertices, three edges.
//! * Canonical mechanics is a **perfect matching**: each `q_k` paired with its own `p_k`, weight
//!   one. `J = [[0, I], [-I, 0]]` is what a matching looks like written out.
//! * A constraint — a closed loop, a contact — is an **extra edge**, and extra edges are exactly
//!   what a spanning tree does not have. Their number is the cyclomatic number of the hypergraph.
//!
//! All three step through the same integrator below and obey the same balance.
//!
//! # The step, and why it is exact
//!
//! For a quadratic Hamiltonian `H = x^T Q x / 2` the discrete gradient IS the implicit midpoint
//! rule, and the balance holds to machine precision at every step rather than approximately. Work
//! in energy-normalised coordinates `z = Q^(1/2) x`, where `H = |z|^2 / 2` and the system is
//!
//! ```text
//! z' = S z + Q^(1/2) G u,     S = Q^(1/2) (J - R) Q^(1/2)
//! ```
//!
//! so a step is `(I - h/2 S) z_next = (I + h/2 S) z + h Q^(1/2) G u` — precisely the shifted
//! skew-symmetric system the solvers in this crate exist for. The normalisation is not cosmetic: a
//! badly scaled `Q` puts a floor under the residual that has nothing to do with the scheme, and
//! moving to `z` removes it.
//!
//! Subtracting the two sides against `z_bar = (z + z_next)/2` gives
//!
//! ```text
//! H(z') - H(z) = -h z_bar^T Q^(1/2) R Q^(1/2) z_bar  +  h z_bar^T Q^(1/2) G u
//! ```
//!
//! with the skew part cancelling **exactly**, for the same reason it cancels in the continuous
//! equation. [`PortHamiltonian::balance_residual`] measures the difference between the two sides,
//! and on a correct implementation it is rounding and stays rounding however long the run.

use crate::{gmres, Operator, Report};

/// One signed hyperedge: a power port.
///
/// Sources carry `sigma = -1` and targets `sigma = +1`, which is the convention G-SPHF uses and the
/// reason the resulting `J` comes out skew rather than merely antisymmetric-by-inspection.
///
/// A port with one source and one target is an ordinary edge and gives `J[u][v] = +w`,
/// `J[v][u] = -w`. More than one of either is a junction: every source is coupled to every target,
/// which is what a Kirchhoff node, a gear train or a lever does.
#[derive(Debug, Clone, PartialEq)]
pub struct Port {
    /// State coordinates the port draws from.
    pub sources: Vec<usize>,
    /// State coordinates the port delivers to.
    pub targets: Vec<usize>,
    /// Coupling strength. Its units are whatever makes `w * effort` a flow for this pair.
    pub weight: f64,
}

impl Port {
    /// An ordinary directed edge `from -> to`.
    pub fn edge(from: usize, to: usize, weight: f64) -> Self {
        Self {
            sources: vec![from],
            targets: vec![to],
            weight,
        }
    }
}

/// The interconnection `J`, as a signed hypergraph rather than as a stored matrix.
///
/// [`Self::apply`] costs one pass over the incidences, not `dim^2`. On a sparse network — and a
/// physical network is sparse, because a component is wired to its neighbours and not to everything
/// — that is the whole difference between a structure and a matrix that happens to have that
/// structure.
#[derive(Debug, Clone, PartialEq)]
pub struct Interconnection {
    dim: usize,
    ports: Vec<Port>,
}

impl Interconnection {
    /// Build from a dimension and a list of ports.
    ///
    /// # Panics
    /// If any port names a coordinate outside `0..dim`.
    pub fn new(dim: usize, ports: Vec<Port>) -> Self {
        for (k, p) in ports.iter().enumerate() {
            for &v in p.sources.iter().chain(&p.targets) {
                assert!(v < dim, "port {k} names coordinate {v} of {dim}");
            }
        }
        Self { dim, ports }
    }

    /// Number of state coordinates.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The ports, in the order given.
    pub fn ports(&self) -> &[Port] {
        &self.ports
    }

    /// `out <- J e`, accumulating nothing from the previous contents of `out`.
    ///
    /// # Panics
    /// If either slice is shorter than [`Self::dim`].
    pub fn apply(&self, e: &[f64], out: &mut [f64]) {
        assert!(e.len() >= self.dim && out.len() >= self.dim, "dimension");
        out[..self.dim].fill(0.0);
        for p in &self.ports {
            // Each side is summed ONCE and then distributed, so a port with `s` sources and `t`
            // targets costs `s + t` rather than `s * t`. On a plain edge the two are the same; on a
            // junction of any size they are not.
            let from: f64 = p.sources.iter().map(|&u| e[u]).sum();
            let to: f64 = p.targets.iter().map(|&v| e[v]).sum();
            for &u in &p.sources {
                out[u] += p.weight * to;
            }
            for &v in &p.targets {
                out[v] -= p.weight * from;
            }
        }
    }

    /// `J` written out, row-major. For tests and for systems small enough that it is the clearer
    /// thing to look at; nothing in the stepping path calls it.
    pub fn dense(&self) -> Vec<f64> {
        let n = self.dim;
        let mut m = vec![0.0; n * n];
        for p in &self.ports {
            for &u in &p.sources {
                for &v in &p.targets {
                    m[u * n + v] += p.weight;
                    m[v * n + u] -= p.weight;
                }
            }
        }
        m
    }
}

/// A port-Hamiltonian system with a quadratic Hamiltonian and a diagonal energy metric.
///
/// `H(x) = sum_i q_i x_i^2 / 2`, so `grad H = Q x`. Diagonal because that is what a network of
/// storage elements gives — a capacitor's energy depends on its own charge — and because it makes
/// `Q^(1/2)` a square root of numbers rather than of a matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct PortHamiltonian {
    /// Where energy can flow, and in which direction.
    pub interconnection: Interconnection,
    /// The diagonal of `R`. Non-negative; a positive entry is a resistance, a conductance, a
    /// friction — whatever the coordinate's units make it.
    pub dissipation: Vec<f64>,
    /// The diagonal of `Q`. Strictly positive: a coordinate with no energy in it is not a state.
    pub energy: Vec<f64>,
}

impl PortHamiltonian {
    /// Check the pieces agree in dimension and sign, and prepare the square roots.
    ///
    /// # Panics
    /// If the lengths disagree, if any dissipation is negative, or if any energy weight is not
    /// strictly positive.
    pub fn new(interconnection: Interconnection, dissipation: Vec<f64>, energy: Vec<f64>) -> Self {
        let n = interconnection.dim();
        assert_eq!(dissipation.len(), n, "dissipation must cover every coordinate");
        assert_eq!(energy.len(), n, "energy must cover every coordinate");
        assert!(
            dissipation.iter().all(|&r| r >= 0.0),
            "a negative dissipation would create energy"
        );
        assert!(
            energy.iter().all(|&q| q > 0.0),
            "every coordinate must store energy"
        );
        Self {
            interconnection,
            dissipation,
            energy,
        }
    }

    /// Number of state coordinates.
    pub fn dim(&self) -> usize {
        self.interconnection.dim()
    }

    /// `H(x)`.
    pub fn hamiltonian(&self, x: &[f64]) -> f64 {
        self.energy
            .iter()
            .zip(x)
            .map(|(&q, &v)| 0.5 * q * v * v)
            .sum()
    }

    /// `x -> z = Q^(1/2) x`, the coordinates the step is taken in.
    pub fn normalise(&self, x: &[f64], z: &mut [f64]) {
        for (k, (&q, &v)) in self.energy.iter().zip(x).enumerate() {
            z[k] = q.sqrt() * v;
        }
    }

    /// The inverse of [`Self::normalise`].
    pub fn denormalise(&self, z: &[f64], x: &mut [f64]) {
        for (k, (&q, &v)) in self.energy.iter().zip(z).enumerate() {
            x[k] = v / q.sqrt();
        }
    }

    /// `xdot = (J - R) Q x + u`, the continuous flow. For checking the discrete step against, and
    /// for callers that want an explicit integrator.
    pub fn flow(&self, x: &[f64], input: &[f64], out: &mut [f64]) {
        let n = self.dim();
        let mut effort = vec![0.0; n];
        for k in 0..n {
            effort[k] = self.energy[k] * x[k];
        }
        self.interconnection.apply(&effort, out);
        for k in 0..n {
            out[k] += input[k] - self.dissipation[k] * effort[k];
        }
    }

    /// The operator `S = Q^(1/2) (J - R) Q^(1/2)` the solvers act on.
    pub fn operator(&self) -> NormalisedFlow<'_> {
        NormalisedFlow { system: self }
    }

    /// One step of the discrete gradient scheme, in place.
    ///
    /// `input` is held at its midpoint value across the step, which is what makes the balance below
    /// exact rather than first-order in the input's variation.
    ///
    /// Returns what the linear solve did, so a caller can tell a converged step from one that ran
    /// out of iterations and quietly returned something plausible.
    pub fn step(&self, x: &mut [f64], input: &[f64], h: f64, tolerance: f64) -> Report {
        let n = self.dim();
        let mut z = vec![0.0; n];
        self.normalise(x, &mut z);

        // b = (I + h/2 S) z + h Q^(1/2) u
        let flow = self.operator();
        let mut sz = vec![0.0; n];
        flow.apply(&z, &mut sz);
        let mut b = vec![0.0; n];
        for k in 0..n {
            b[k] = z[k] + 0.5 * h * sz[k] + h * self.energy[k].sqrt() * input[k];
        }

        let mut next = z.clone();
        let report = gmres(&flow, 0.5 * h, &b, &mut next, tolerance, n.clamp(1, 30), 200);
        self.denormalise(&next, x);
        report
    }

    /// How far one step missed the power balance, relative to the energy in the system.
    ///
    /// ```text
    /// H(x') - H(x)  ==  h * ( -e_bar^T R e_bar  +  e_bar^T u ),   e_bar = Q (x + x')/2
    /// ```
    ///
    /// The skew part cancels exactly on both sides, so this is not a discretisation error that
    /// shrinks with `h`: on a correct implementation it is rounding, and it stays rounding however
    /// long the run. A scheme that merely conserves energy *approximately* shows up here as a
    /// residual that grows.
    ///
    /// Normalised by the run's characteristic power `H_max / h` rather than by the step's own rate,
    /// which crosses zero and would make the ratio meaningless exactly where the balance is
    /// tightest.
    pub fn balance_residual(
        &self,
        before: &[f64],
        after: &[f64],
        input: &[f64],
        h: f64,
        energy_scale: f64,
    ) -> f64 {
        let n = self.dim();
        let mut supplied = 0.0;
        let mut lost = 0.0;
        for k in 0..n {
            let effort = self.energy[k] * 0.5 * (before[k] + after[k]);
            supplied += effort * input[k];
            lost += self.dissipation[k] * effort * effort;
        }
        let change = self.hamiltonian(after) - self.hamiltonian(before);
        let predicted = h * (supplied - lost);
        (change - predicted).abs() / (energy_scale / h).max(f64::MIN_POSITIVE)
    }
}

/// `S = Q^(1/2) (J - R) Q^(1/2)`, matrix-free.
///
/// Skew minus positive semidefinite, which is exactly the class this crate's solvers are written
/// for — and not by coincidence: `J` is skew because the hypergraph's signs make it so, and `R` is
/// positive semidefinite because a dissipation that was not would create energy.
#[derive(Debug, Clone, Copy)]
pub struct NormalisedFlow<'a> {
    system: &'a PortHamiltonian,
}

impl Operator for NormalisedFlow<'_> {
    fn dim(&self) -> usize {
        self.system.dim()
    }

    fn apply(&self, z: &[f64], y: &mut [f64]) {
        let n = self.dim();
        let s = self.system;
        let mut scaled = vec![0.0; n];
        for k in 0..n {
            scaled[k] = s.energy[k].sqrt() * z[k];
        }
        s.interconnection.apply(&scaled, y);
        for k in 0..n {
            y[k] = s.energy[k].sqrt() * (y[k] - s.dissipation[k] * scaled[k]);
        }
    }
}
