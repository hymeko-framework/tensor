//! The discrete connection: gauge fixing, cotree residuals, curvature.
//!
//! The topology used here is a wheel — a hub joined to every node of a rim — because it is the
//! smallest graph with more than one independent cycle whose fundamental cycles are also its
//! faces, so the closed-form estimate and the plaquette holonomy can be compared term by term
//! rather than only in the mean.
//!
//! The wheel, the sampler and the random source live in this file and not in the library. The
//! library owns rotor algebra over an ordered structure; a graph generator is a task, and a task
//! that leaks into a core crate is how a core crate stops being one.

use hymeko_clifford::{
    TreeStep, closed_form_curvature, cotree_residuals, curvature_energy, quat_conjugate, quat_mul,
    rotor_angle, rotor_holonomy_forward, tree_gauge_frames,
};

/// xorshift64, so every number below is reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    /// A unit quaternion with rotation angle `theta`, about a direction drawn from the sphere.
    fn rotor(&mut self, theta: f32) -> [f32; 4] {
        let z = 2.0 * self.next_f32() - 1.0;
        let phi = std::f32::consts::TAU * self.next_f32();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let axis = [r * phi.cos(), r * phi.sin(), z];
        let (s, c) = (theta * 0.5).sin_cos();
        [c, s * axis[0], s * axis[1], s * axis[2]]
    }
}

/// A wheel: hub `0`, rim `1..=rim`. Spokes are the spanning tree, rim edges the cotree.
struct Wheel {
    nodes: usize,
    edges: Vec<(u32, u32)>,
    steps: Vec<TreeStep>,
    cotree: Vec<usize>,
    /// Each fundamental cycle as `(edge, traversed_forward)`, hub → a → b → hub.
    cycles: Vec<[(usize, bool); 3]>,
}

fn wheel(rim: usize) -> Wheel {
    assert!(rim >= 3);
    let mut edges = Vec::with_capacity(2 * rim);
    // spokes first, so spoke i-1 joins the hub to node i
    for i in 1..=rim {
        edges.push((0u32, i as u32));
    }
    // then the rim, edge `rim + p` joining node 1+p to its successor
    for p in 0..rim {
        let a = 1 + p;
        let b = if p + 1 == rim { 1 } else { p + 2 };
        edges.push((a as u32, b as u32));
    }
    let steps: Vec<TreeStep> = (1..=rim).map(|i| (i, i - 1, true)).collect();
    let cotree: Vec<usize> = (rim..2 * rim).collect();
    let cycles = (0..rim)
        .map(|p| {
            let back = if p + 1 == rim { 0 } else { p + 1 };
            [(p, true), (rim + p, true), (back, false)]
        })
        .collect();
    Wheel {
        nodes: rim + 1,
        edges,
        steps,
        cotree,
        cycles,
    }
}

/// The oracle: the mean plaquette holonomy angle, read from the true cycles through the crate's
/// own ordered rotor product. This is what the closed form has to reproduce.
fn oracle_curvature(w: &Wheel, edge_q: &[f32]) -> f32 {
    let k = 3usize;
    let mut ordered = vec![0.0f32; w.cycles.len() * k * 4];
    for (c, cycle) in w.cycles.iter().enumerate() {
        for (i, &(edge, forward)) in cycle.iter().enumerate() {
            let q = [
                edge_q[edge * 4],
                edge_q[edge * 4 + 1],
                edge_q[edge * 4 + 2],
                edge_q[edge * 4 + 3],
            ];
            let q = if forward { q } else { quat_conjugate(q) };
            ordered[(c * k + i) * 4..(c * k + i) * 4 + 4].copy_from_slice(&q);
        }
    }
    let (holo, _) = rotor_holonomy_forward(&ordered, w.cycles.len(), k);
    let total: f32 = (0..w.cycles.len())
        .map(|c| {
            rotor_angle([
                holo[c * 4],
                holo[c * 4 + 1],
                holo[c * 4 + 2],
                holo[c * 4 + 3],
            ])
        })
        .sum();
    total / w.cycles.len() as f32
}

fn sample(w: &Wheel, rng: &mut Rng, theta: f32) -> Vec<f32> {
    let mut q = vec![0.0f32; w.edges.len() * 4];
    for e in 0..w.edges.len() {
        let t = theta * rng.next_f32();
        let r = rng.rotor(t);
        q[e * 4..e * 4 + 4].copy_from_slice(&r);
    }
    q
}

// ─── the gauge ──────────────────────────────────────────────────────────────────────────────────

#[test]
fn the_root_frame_is_the_identity_and_every_frame_is_a_unit_rotor() {
    let w = wheel(12);
    let q = sample(&w, &mut Rng(3), 1.2);
    let frames = tree_gauge_frames(w.nodes, &w.edges, &w.steps, &q);
    assert_eq!(frames.len(), w.nodes * 4);
    assert!(rotor_angle([frames[0], frames[1], frames[2], frames[3]]) < 1e-6);
    for i in 0..w.nodes {
        let f = [
            frames[i * 4],
            frames[i * 4 + 1],
            frames[i * 4 + 2],
            frames[i * 4 + 3],
        ];
        let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt();
        assert!((n - 1.0).abs() < 1e-4, "frame {i} has norm {n}");
    }
}

/// The sharpest test in the file.
///
/// A **pure gauge** assigns `g_e = F_b · F_a⁻¹` for arbitrary node frames `F`. Every edge rotor is
/// then far from the identity — the connection looks thoroughly non-trivial — and yet every loop
/// closes exactly, because the transport telescopes. An estimator that did not gauge-fix correctly
/// would report the size of the `F`s instead of the curvature, and would look perfectly reasonable
/// doing it.
#[test]
fn a_pure_gauge_has_no_curvature_however_large_its_rotors() {
    let w = wheel(16);
    let mut rng = Rng(0x5EED);
    let node_frames: Vec<[f32; 4]> = (0..w.nodes)
        .map(|_| {
            let t = 2.4 * rng.next_f32();
            rng.rotor(t)
        })
        .collect();

    let mut q = vec![0.0f32; w.edges.len() * 4];
    let mut largest = 0.0f32;
    for (e, &(a, b)) in w.edges.iter().enumerate() {
        let g = quat_mul(
            node_frames[b as usize],
            quat_conjugate(node_frames[a as usize]),
        );
        largest = largest.max(rotor_angle(g));
        q[e * 4..e * 4 + 4].copy_from_slice(&g);
    }
    assert!(
        largest > 1.0,
        "the fixture must actually be non-trivial: largest edge rotation {largest} rad"
    );

    let curvature = closed_form_curvature(w.nodes, &w.edges, &w.steps, &w.cotree, &q);
    assert!(
        curvature < 1e-4,
        "a pure gauge is flat: got {curvature} rad with edge rotations up to {largest} rad"
    );
}

#[test]
fn a_gauge_transformation_leaves_the_curvature_alone() {
    // Curvature is the observable; the frames are a choice of description. Transforming the
    // connection by arbitrary node frames must not move the answer at all.
    let w = wheel(14);
    let mut rng = Rng(11);
    let q = sample(&w, &mut rng, 1.0);
    let before = closed_form_curvature(w.nodes, &w.edges, &w.steps, &w.cotree, &q);

    let gauge: Vec<[f32; 4]> = (0..w.nodes)
        .map(|_| {
            let t = 2.0 * rng.next_f32();
            rng.rotor(t)
        })
        .collect();
    let mut transformed = vec![0.0f32; q.len()];
    for (e, &(a, b)) in w.edges.iter().enumerate() {
        let g = [q[e * 4], q[e * 4 + 1], q[e * 4 + 2], q[e * 4 + 3]];
        let t = quat_mul(
            gauge[b as usize],
            quat_mul(g, quat_conjugate(gauge[a as usize])),
        );
        transformed[e * 4..e * 4 + 4].copy_from_slice(&t);
    }
    let after = closed_form_curvature(w.nodes, &w.edges, &w.steps, &w.cotree, &transformed);

    assert!(
        before > 0.05,
        "the fixture must be curved to begin with: {before}"
    );
    assert!(
        (before - after).abs() < 1e-4,
        "curvature is gauge invariant: {before} became {after}"
    );
}

// ─── the identity the closed form exists for ────────────────────────────────────────────────────

#[test]
fn the_closed_form_reproduces_the_plaquette_holonomy_cycle_by_cycle() {
    // The correctness identity behind the whole construction: the cotree residual in the tree gauge
    // IS the fundamental cycle's holonomy. Compared per cycle rather than in the mean, because two
    // different distributions can share a mean and the claim is stronger than that.
    let w = wheel(20);
    let mut rng = Rng(2026);
    for theta in [0.2f32, 0.9, 2.0] {
        let q = sample(&w, &mut rng, theta);
        let frames = tree_gauge_frames(w.nodes, &w.edges, &w.steps, &q);
        let residuals = cotree_residuals(&w.edges, &w.cotree, &q, &frames);

        // per-cycle: the residual angle against the plaquette holonomy angle
        let k = 3usize;
        let mut ordered = vec![0.0f32; w.cycles.len() * k * 4];
        for (c, cycle) in w.cycles.iter().enumerate() {
            for (i, &(edge, forward)) in cycle.iter().enumerate() {
                let g = [
                    q[edge * 4],
                    q[edge * 4 + 1],
                    q[edge * 4 + 2],
                    q[edge * 4 + 3],
                ];
                let g = if forward { g } else { quat_conjugate(g) };
                ordered[(c * k + i) * 4..(c * k + i) * 4 + 4].copy_from_slice(&g);
            }
        }
        let (holo, _) = rotor_holonomy_forward(&ordered, w.cycles.len(), k);

        for c in 0..w.cycles.len() {
            let a = rotor_angle([
                residuals[c * 4],
                residuals[c * 4 + 1],
                residuals[c * 4 + 2],
                residuals[c * 4 + 3],
            ]);
            let b = rotor_angle([
                holo[c * 4],
                holo[c * 4 + 1],
                holo[c * 4 + 2],
                holo[c * 4 + 3],
            ]);
            assert!(
                (a - b).abs() < 1e-4,
                "theta {theta}, cycle {c}: estimate {a} against oracle {b}"
            );
        }
        assert!(
            (curvature_energy(&residuals) - oracle_curvature(&w, &q)).abs() < 1e-5,
            "and so the means agree too"
        );
    }
}

#[test]
fn the_reading_grows_with_the_curvature_and_vanishes_without_it() {
    let w = wheel(24);

    // a flat connection: every edge the identity
    let flat = {
        let mut q = vec![0.0f32; w.edges.len() * 4];
        for e in 0..w.edges.len() {
            q[e * 4] = 1.0;
        }
        q
    };
    assert_eq!(
        closed_form_curvature(w.nodes, &w.edges, &w.steps, &w.cotree, &flat),
        0.0,
        "an identity connection must read exactly zero, not nearly zero"
    );

    // and the reading must be monotone in the injected rotation
    let mut previous = 0.0f32;
    for theta in [0.05f32, 0.2, 0.6, 1.4, 2.4] {
        let q = sample(&w, &mut Rng(7), theta);
        let curvature = closed_form_curvature(w.nodes, &w.edges, &w.steps, &w.cotree, &q);
        assert!(
            curvature > previous,
            "theta {theta}: curvature {curvature} did not exceed {previous}"
        );
        previous = curvature;
    }
}

#[test]
fn the_cotree_carries_one_residual_per_independent_cycle() {
    // |E| - |V| + 1, the cyclomatic number. If this ever disagreed, the estimate would be summing
    // over the wrong set and no other assertion here would notice.
    for rim in [3usize, 7, 32] {
        let w = wheel(rim);
        assert_eq!(w.cotree.len(), w.edges.len() - w.nodes + 1);
        let q = sample(&w, &mut Rng(rim as u64 + 1), 0.8);
        let frames = tree_gauge_frames(w.nodes, &w.edges, &w.steps, &q);
        assert_eq!(
            cotree_residuals(&w.edges, &w.cotree, &q, &frames).len(),
            w.cotree.len() * 4
        );
    }
}
