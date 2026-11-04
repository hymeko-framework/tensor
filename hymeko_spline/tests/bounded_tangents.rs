//! The bounded tangent rule (Nagare plan 2026-10-23-c1-lattice-activations, Amendment 1).
//!
//! - Every leaf of a bounded Catmull–Rom spline lies between its two knot values, so the spline stays within its
//!   control points' range without a clamp.
//! - It equals the chord spline, bit for bit, wherever every knot takes the chord.
//! - It is flat at an interior extremum of its control points.
//! - Its pullback is its derivative.
//!
//! The chord rule fails the first and third checks on the same data, so neither is vacuous.

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, Knots, Tangents};

/// Values in `[-1.5, 1.5]`, deterministic.
fn field(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 3001) as f32 / 1000.0 - 1.5
        })
        .collect()
}

/// Uniform and refined knot sets, with the given rule.
fn knot_sets(tangents: Tangents) -> Vec<Knots> {
    [
        Knots::uniform(6),
        Knots::uniform(9),
        Knots::refined(6, &[0, 2, 1, 0, 1]),
        Knots::refined(5, &[3, 0, 1, 2]),
    ]
    .into_iter()
    .map(|knots| knots.with_tangents(tangents))
    .collect()
}

/// `points` inputs spread over `[-1, 1]` for each of `channels` channels, row-major.
fn grid(points: usize, channels: usize) -> Vec<f32> {
    (0..points)
        .flat_map(|i| {
            let x = -1.0 + 2.0 * i as f32 / (points - 1) as f32;
            std::iter::repeat_n(x, channels)
        })
        .collect()
}

/// The largest excursion of the spline outside the range of the two knot values of the leaf each input lies on.
fn worst_excursion(knots: &Knots, coef: &[f32], channels: usize) -> f32 {
    let points = 2001;
    let x = grid(points, channels);
    let (y, _) = catmull_rom_forward(coef, &x, points, channels, knots);
    let positions = knots.positions();
    let mut worst = 0.0f32;
    for (e, (&xe, &ye)) in x.iter().zip(&y).enumerate() {
        let q = &coef[(e % channels) * knots.len()..][..knots.len()];
        let leaf = positions
            .partition_point(|&p| p <= xe)
            .saturating_sub(1)
            .min(knots.len() - 2);
        let (lo, hi) = (q[leaf].min(q[leaf + 1]), q[leaf].max(q[leaf + 1]));
        worst = worst.max(lo - ye).max(ye - hi);
    }
    worst
}

#[test]
fn every_leaf_of_a_bounded_spline_lies_between_its_two_knot_values() {
    let channels = 4;
    let mut chord_escapes = false;
    for (bounded, chord) in knot_sets(Tangents::Bounded)
        .into_iter()
        .zip(knot_sets(Tangents::Chord))
    {
        for seed in 1..6 {
            let coef = field(channels * bounded.len(), seed);
            let worst = worst_excursion(&bounded, &coef, channels);
            assert!(
                worst <= 1e-6,
                "bounded {:?} seed {seed}: {worst}",
                bounded.depths()
            );
            chord_escapes |= worst_excursion(&chord, &coef, channels) > 1e-3;
        }
    }
    assert!(chord_escapes, "the chord rule overshoots on the same data");
}

#[test]
fn the_bounded_rule_is_the_chord_rule_where_every_knot_takes_the_chord() {
    let channels = 3;
    for (bounded, chord) in knot_sets(Tangents::Bounded)
        .into_iter()
        .zip(knot_sets(Tangents::Chord))
    {
        // A gentle monotone curve per channel: neighbouring secants within a factor 5, so every knot takes the chord.
        let coef: Vec<f32> = (0..channels)
            .flat_map(|c| {
                bounded
                    .positions()
                    .iter()
                    .map(move |&p| 0.2 * c as f32 + 0.5 * p + 0.1 * p * p)
                    .collect::<Vec<_>>()
            })
            .collect();
        let x = field(257 * channels, 9);
        let (yb, cache_b) = catmull_rom_forward(&coef, &x, 257, channels, &bounded);
        let (yc, cache_c) = catmull_rom_forward(&coef, &x, 257, channels, &chord);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&yb), bits(&yc), "{:?}: forward", bounded.depths());
        let g = field(x.len(), 10);
        let (b, c) = (
            catmull_rom_backward(&coef, &cache_b, &g),
            catmull_rom_backward(&coef, &cache_c, &g),
        );
        assert_eq!(
            bits(&b.grad_x),
            bits(&c.grad_x),
            "{:?}: grad_x",
            bounded.depths()
        );
        assert_eq!(
            bits(&b.grad_coef),
            bits(&c.grad_coef),
            "{:?}: grad_coef",
            bounded.depths()
        );
    }
}

/// `dy/dx` of one channel's spline at `x`, from the pullback of a unit cotangent.
fn slope(coef: &[f32], knots: &Knots, x: f32) -> f32 {
    let (_, cache) = catmull_rom_forward(coef, &[x], 1, 1, knots);
    catmull_rom_backward(coef, &cache, &[1.0]).grad_x[0]
}

#[test]
fn a_bounded_spline_is_flat_at_an_interior_extremum() {
    // Uniform knots -1, -0.6, -0.2, 0.2, 0.6, 1; the control points peak at the third.
    let coef = [0.0f32, 0.4, 1.0, 0.3, 0.2, 0.1];
    let bounded = Knots::uniform(6).with_tangents(Tangents::Bounded);
    let chord = Knots::uniform(6);
    // `-0.2` is not exact in `f32`: the input lands a few ulp into the next leaf, where the slope starts at zero.
    let peak = -0.2f32;
    assert!(
        slope(&coef, &bounded, peak).abs() < 1e-5,
        "zero slope at the peak"
    );
    let (left, right) = (peak - 1e-3, peak + 1e-3);
    assert!(slope(&coef, &bounded, left) > 0.0 && slope(&coef, &bounded, right) < 0.0);
    // Continuous through zero: ten times closer to the peak, the slope is about ten times smaller on either side.
    for side in [-1.0f32, 1.0] {
        let (far, near) = (
            slope(&coef, &bounded, peak + side * 1e-3).abs(),
            slope(&coef, &bounded, peak + side * 1e-4).abs(),
        );
        assert!(near < 0.2 * far, "side {side}: {near} against {far}");
    }
    assert!(
        slope(&coef, &chord, peak).abs() > 0.1,
        "the chord rule is not flat there"
    );
}

#[test]
fn the_bounded_pullback_is_the_derivative() {
    // Uniform knots over seven control points whose routes are, knot by knot: chord (end), left, right, critical,
    // chord, right, chord (end); every margin between candidates exceeds the difference step's effect.
    let coef = [0.0f32, 0.1, 0.9, 1.0, 0.2, -0.5, -0.6];
    let knots = Knots::uniform(7).with_tangents(Tangents::Bounded);
    let x: Vec<f32> = (0..41)
        .map(|i| -0.975 + 0.0475 * i as f32)
        .filter(|x| ((x + 1.0) * 3.0).fract().abs() > 0.05)
        .collect();
    let g = field(x.len(), 11);
    let (_, cache) = catmull_rom_forward(&coef, &x, x.len(), 1, &knots);
    let back = catmull_rom_backward(&coef, &cache, &g);
    let loss = |coef: &[f32], x: &[f32]| -> f64 {
        catmull_rom_forward(coef, x, x.len(), 1, &knots)
            .0
            .iter()
            .zip(&g)
            .map(|(y, g)| f64::from(*y) * f64::from(*g))
            .sum()
    };
    let eps = 1e-3f32;
    for j in 0..coef.len() {
        let (mut up, mut down) = (coef, coef);
        up[j] += eps;
        down[j] -= eps;
        let numeric = (loss(&up, &x) - loss(&down, &x)) / (2.0 * f64::from(eps));
        let got = f64::from(back.grad_coef[j]);
        assert!(
            (numeric - got).abs() <= 2e-3 + 2e-3 * numeric.abs(),
            "coef {j}: {numeric} vs {got}"
        );
    }
    for e in 0..x.len() {
        let (mut up, mut down) = (x.clone(), x.clone());
        up[e] += eps;
        down[e] -= eps;
        let numeric = (loss(&coef, &up) - loss(&coef, &down)) / (2.0 * f64::from(eps));
        let got = f64::from(back.grad_x[e]);
        assert!(
            (numeric - got).abs() <= 5e-3 + 5e-3 * numeric.abs(),
            "x {e}: {numeric} vs {got}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn bounded_results_are_the_same_on_any_thread_count() {
    let (rows, channels) = (2003usize, 8usize);
    let knots = Knots::refined(6, &[0, 2, 1, 0, 1]).with_tangents(Tangents::Bounded);
    let coef = field(channels * knots.len(), 21);
    let x = field(rows * channels, 22);
    let g = field(rows * channels, 23);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool")
            .install(|| {
                let (y, cache) = catmull_rom_forward(&coef, &x, rows, channels, &knots);
                let back = catmull_rom_backward(&coef, &cache, &g);
                [y, back.grad_x, back.grad_coef]
                    .map(|v| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>())
            })
    };
    assert_eq!(run(1), run(4));
}
