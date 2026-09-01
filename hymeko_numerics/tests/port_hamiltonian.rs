//! Port-Hamiltonian systems built from a signed hypergraph.
//!
//! The claim under test is that `J` need not be written down. If it is the signed incidence of a
//! hypergraph, then skew-symmetry — and with it the energy balance — is a consequence of the sign
//! convention and not a property to be checked afterwards.
//!
//! So the tests are of two kinds: **identities** that must hold for every hypergraph whatever its
//! shape, and one **exact reproduction** of a matrix that was written by hand for a specific
//! circuit, which is what makes the general construction a generalisation rather than a parallel
//! implementation.

use hymeko_numerics::{Interconnection, Port, PortHamiltonian};

struct Rng(u64);
impl Rng {
    fn f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.f64() - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
    /// A hypergraph with ordinary edges AND junctions, so nothing that only works for binary edges
    /// can pass.
    fn hypergraph(&mut self, dim: usize) -> Interconnection {
        let count = 1 + self.below(2 * dim);
        let mut ports = Vec::with_capacity(count);
        for _ in 0..count {
            let (ns, nt) = (1 + self.below(3), 1 + self.below(3));
            let mut sources = Vec::new();
            let mut targets = Vec::new();
            for _ in 0..ns {
                let v = self.below(dim);
                if !sources.contains(&v) {
                    sources.push(v);
                }
            }
            for _ in 0..nt {
                let v = self.below(dim);
                if !targets.contains(&v) && !sources.contains(&v) {
                    targets.push(v);
                }
            }
            if sources.is_empty() || targets.is_empty() {
                continue;
            }
            ports.push(Port {
                sources,
                targets,
                weight: self.signed() * 3.0,
            });
        }
        if ports.is_empty() {
            ports.push(Port::edge(0, dim - 1, 1.0));
        }
        Interconnection::new(dim, ports)
    }
}

// --- the exact reproduction ----------------------------------------------------------------------

#[test]
fn a_chain_reproduces_the_hand_written_structure_matrix_of_a_driven_rlc_circuit() {
    // `hymeko_simulator::electro::port_hamiltonian::structure_matrix` writes this out by hand for a
    // driven RLC circuit coupled to one electromagnetic field mode, with the state
    // `x = [charge, flux, displacement, magnetic flux]`:
    //
    //     [  0   1   0   0 ]
    //     [ -1   0   k   0 ]
    //     [  0  -k   0   g ]
    //     [  0   0  -g   0 ]
    //
    // Read as a hypergraph it is a CHAIN of three edges: charge to flux at unit weight, flux to
    // displacement at the coupling `k`, displacement to magnetic flux at the discrete curl `g`. If
    // the general construction is a generalisation and not a second implementation, it must produce
    // that matrix and not merely something with the same symmetry.
    for (k, g) in [(0.0, 0.0), (0.3, 1.7), (-2.5, 0.04), (1.0, 1.0)] {
        let chain = Interconnection::new(
            4,
            vec![
                Port::edge(0, 1, 1.0),
                Port::edge(1, 2, k),
                Port::edge(2, 3, g),
            ],
        );
        let want = [
            0.0, 1.0, 0.0, 0.0, //
            -1.0, 0.0, k, 0.0, //
            0.0, -k, 0.0, g, //
            0.0, 0.0, -g, 0.0,
        ];
        let got = chain.dense();
        assert_eq!(
            got, want,
            "k={k} g={g}: the chain does not reproduce the hand-written matrix"
        );
    }
}

#[test]
fn canonical_mechanics_is_a_perfect_matching() {
    // `J = [[0, I], [-I, 0]]` is what a matching looks like written out: each position paired with
    // its own momentum, weight one, and nothing else coupled to anything. Stating it this way is
    // what puts mechanics and circuits under one integrator.
    for n in [1usize, 2, 5] {
        let matching = Interconnection::new(
            2 * n,
            (0..n).map(|k| Port::edge(k, n + k, 1.0)).collect(),
        );
        let j = matching.dense();
        let d = 2 * n;
        for r in 0..d {
            for c in 0..d {
                let want = if c == r + n {
                    1.0
                } else if r == c + n {
                    -1.0
                } else {
                    0.0
                };
                assert_eq!(j[r * d + c], want, "n={n}: J[{r}][{c}]");
            }
        }
    }
}

// --- the identities ------------------------------------------------------------------------------

#[test]
fn the_interconnection_is_skew_whatever_the_hypergraph_looks_like() {
    let mut rng = Rng(0x5164);
    for case in 0..400 {
        let dim = 2 + rng.below(10);
        let j = rng.hypergraph(dim);
        let m = j.dense();
        for r in 0..dim {
            for c in 0..dim {
                assert_eq!(
                    m[r * dim + c],
                    -m[c * dim + r],
                    "case {case}: J[{r}][{c}] is not the negative of J[{c}][{r}]"
                );
            }
            assert_eq!(m[r * dim + r], 0.0, "case {case}: J[{r}][{r}] is not zero");
        }
    }
}

#[test]
fn no_effort_does_work_against_the_interconnection() {
    // `e^T J e == 0` for every effort, which is the statement that the interconnection moves energy
    // and never makes any. Checked numerically because it is the identity the energy balance rests
    // on, and it must survive junctions and negative weights, not just tidy chains.
    let mut rng = Rng(31);
    for case in 0..500 {
        let dim = 2 + rng.below(12);
        let j = rng.hypergraph(dim);
        let e: Vec<f64> = (0..dim).map(|_| rng.signed() * 4.0).collect();
        let mut out = vec![0.0; dim];
        j.apply(&e, &mut out);
        let work: f64 = e.iter().zip(&out).map(|(a, b)| a * b).sum();
        let scale = e.iter().zip(&out).map(|(a, b)| (a * b).abs()).sum::<f64>();
        assert!(
            work.abs() <= 1e-12 * scale.max(1.0),
            "case {case}: e^T J e = {work:e} against a scale of {scale:e}"
        );
    }
}

#[test]
fn the_matrix_free_application_agrees_with_the_matrix() {
    // `apply` walks the incidences and `dense` writes the matrix out; they are separate code, and
    // the fast one is the one every step uses.
    let mut rng = Rng(777);
    for case in 0..300 {
        let dim = 2 + rng.below(10);
        let j = rng.hypergraph(dim);
        let m = j.dense();
        let e: Vec<f64> = (0..dim).map(|_| rng.signed() * 2.0).collect();
        let mut got = vec![0.0; dim];
        j.apply(&e, &mut got);
        for r in 0..dim {
            let want: f64 = (0..dim).map(|c| m[r * dim + c] * e[c]).sum();
            assert!(
                (got[r] - want).abs() < 1e-12 * (1.0 + want.abs()),
                "case {case}: row {r}: {} against {want}",
                got[r]
            );
        }
    }
}

// --- along a trajectory --------------------------------------------------------------------------

fn system(rng: &mut Rng, dim: usize, lossless: bool) -> PortHamiltonian {
    let j = rng.hypergraph(dim);
    let dissipation = (0..dim)
        .map(|_| if lossless { 0.0 } else { 0.5 * rng.f64() })
        .collect();
    let energy = (0..dim).map(|_| 0.2 + 3.0 * rng.f64()).collect();
    PortHamiltonian::new(j, dissipation, energy)
}

#[test]
fn a_lossless_undriven_system_keeps_its_energy_to_machine_precision() {
    // Not "does not drift much" — does not drift. For a quadratic Hamiltonian the discrete gradient
    // is the implicit midpoint rule and the skew part cancels in the discrete equation for the same
    // reason it cancels in the continuous one, so the energy is conserved at every step and the
    // error does not accumulate with the length of the run.
    let mut rng = Rng(2024);
    for case in 0..25 {
        let dim = 2 + rng.below(8);
        let ph = system(&mut rng, dim, true);
        let mut x: Vec<f64> = (0..dim).map(|_| rng.signed() * 2.0).collect();
        let input = vec![0.0; dim];
        let h0 = ph.hamiltonian(&x);
        if h0 < 1e-6 {
            continue;
        }

        let mut worst: f64 = 0.0;
        for step in 0..2000 {
            let report = ph.step(&mut x, &input, 1e-3, 1e-13);
            assert!(
                report.converged,
                "case {case} step {step}: the solve did not converge ({} iterations, residual {:e})",
                report.iterations, report.residual
            );
            worst = worst.max(((ph.hamiltonian(&x) - h0) / h0).abs());
        }
        assert!(
            worst < 1e-9,
            "case {case}: energy drifted by {worst:e} relative over two thousand steps"
        );
    }
}

#[test]
fn the_power_balance_holds_at_every_step_with_dissipation_and_a_source() {
    // The full statement: what the Hamiltonian does over a step equals what was supplied minus what
    // was dissipated, exactly. This is where a scheme that conserves energy only approximately
    // separates from one that satisfies the balance identically.
    let mut rng = Rng(99);
    for case in 0..25 {
        let dim = 3 + rng.below(8);
        let ph = system(&mut rng, dim, false);
        let mut x: Vec<f64> = (0..dim).map(|_| rng.signed() * 2.0).collect();
        let input: Vec<f64> = (0..dim)
            .map(|k| if k == 0 { 1.5 } else { 0.0 })
            .collect();

        let h = 1e-3;
        let mut scale = ph.hamiltonian(&x).max(1.0);
        let mut worst: f64 = 0.0;
        for step in 0..500 {
            let before = x.clone();
            let report = ph.step(&mut x, &input, h, 1e-14);
            assert!(
                report.converged,
                "case {case} step {step}: solve did not converge"
            );
            scale = scale.max(ph.hamiltonian(&x));
            worst = worst.max(ph.balance_residual(&before, &x, &input, h, scale));
        }
        assert!(
            worst < 1e-9,
            "case {case}: the power balance missed by {worst:e} of the characteristic power"
        );
    }
}

#[test]
fn dissipation_removes_energy_and_a_source_puts_it_back() {
    // The signs, which no identity above pins: `R` must take energy out and the input must be able
    // to put it in. A sign error in either leaves every balance test above passing.
    let mut rng = Rng(5150);
    let dim = 6;
    let j = rng.hypergraph(dim);
    let energy: Vec<f64> = (0..dim).map(|_| 1.0 + rng.f64()).collect();

    let lossy = PortHamiltonian::new(j.clone(), vec![0.4; dim], energy.clone());
    let mut x: Vec<f64> = (0..dim).map(|_| rng.signed() * 2.0).collect();
    let start = x.clone();
    let quiet = vec![0.0; dim];
    let h0 = lossy.hamiltonian(&x);
    for _ in 0..3000 {
        lossy.step(&mut x, &quiet, 1e-3, 1e-13);
    }
    let after = lossy.hamiltonian(&x);
    assert!(
        after < 0.5 * h0,
        "damping left {after} of an initial {h0}; it should be draining"
    );

    // and driving the same system from rest raises its energy
    let mut y = vec![0.0; dim];
    let drive: Vec<f64> = (0..dim).map(|k| if k == 1 { 2.0 } else { 0.0 }).collect();
    for _ in 0..500 {
        lossy.step(&mut y, &drive, 1e-3, 1e-13);
    }
    assert!(
        lossy.hamiltonian(&y) > 1e-6,
        "a driven system stayed at rest"
    );
    let _ = start;
}

#[test]
fn a_harmonic_oscillator_runs_at_the_frequency_it_should() {
    // The smallest matching — one position, one momentum — against a result from outside this
    // file: `H = k x^2/2 + p^2/(2m)` oscillates at `sqrt(k/m)`.
    //
    // The crossing times are INTERPOLATED rather than counted. Counting quantises the period by the
    // whole run divided by the number of crossings, and over a run short enough to be a test that is
    // a fifteen per cent granularity — which is a property of the measurement and says nothing about
    // the integrator. Linear interpolation between the two samples straddling zero is accurate to
    // order `h^2` and needs only two crossings.
    for (k, m) in [(4.0, 1.0), (1.0, 0.25), (9.0, 3.0)] {
        let matching = Interconnection::new(2, vec![Port::edge(0, 1, 1.0)]);
        // grad H = (k x, p/m), so the energy diagonal is (k, 1/m)
        let ph = PortHamiltonian::new(matching, vec![0.0; 2], vec![k, 1.0 / m]);
        let mut x = vec![1.0, 0.0];
        let quiet = vec![0.0; 2];

        let want = std::f64::consts::TAU / (k / m).sqrt();
        let dt = 1e-3;
        let steps = (20.0 * want / dt) as usize; // twenty periods
        let (mut first, mut last, mut crossings) = (0.0f64, 0.0f64, 0usize);
        let mut previous = x[0];
        for i in 0..steps {
            ph.step(&mut x, &quiet, dt, 1e-13);
            if previous > 0.0 && x[0] <= 0.0 {
                // where the straight line between the two samples reaches zero
                let t = (i as f64 + previous / (previous - x[0])) * dt;
                if crossings == 0 {
                    first = t;
                }
                last = t;
                crossings += 1;
            }
            previous = x[0];
        }
        assert!(crossings >= 10, "k={k} m={m}: only {crossings} crossings");
        let period = (last - first) / (crossings - 1) as f64;
        assert!(
            (period - want).abs() < 1e-4 * want,
            "k={k} m={m}: period {period} against the expected {want}              ({crossings} crossings, relative error {:e})",
            (period - want).abs() / want
        );
    }
}
