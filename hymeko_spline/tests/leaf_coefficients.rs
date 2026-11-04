//! **The leaf coefficients a shape certificate reads** (Nagare plan 2026-10-24-elliptic-cr, E3):
//! `catmull_rom_leaf_coefficients` is the forward's own table, so its cubic on each leaf is the curve the forward
//! evaluates -- under both tangent rules, on uniform and refined knots.

use hymeko_spline::{catmull_rom_forward, catmull_rom_leaf_coefficients, Knots, Tangents};

/// `n` values in `[-scale, scale]` from `seed`.
fn field(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            scale * (((state >> 33) % 2001) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

/// The uniform Catmull--Rom on the addendum's increasing, discretely convex values `(-0.1, 0, 0.2, 0.7)`: the middle
/// leaf is `0 + 0.15 t − 0.05 t² + 0.1 t³`, its quadratic coefficient negative -- convex samples, a cubic that is not.
#[test]
fn the_middle_leaf_of_convex_samples_is_the_closed_form_cubic() {
    let points = [-0.1f32, 0.0, 0.2, 0.7];
    let a = catmull_rom_leaf_coefficients(&points, 1, &Knots::uniform(4))[1];
    let q = points.map(f64::from);
    let exact = [
        q[1],
        0.5 * (q[2] - q[0]),
        0.5 * (2.0 * q[0] - 5.0 * q[1] + 4.0 * q[2] - q[3]),
        0.5 * (-q[0] + 3.0 * q[1] - 3.0 * q[2] + q[3]),
    ];
    for (got, want) in a.iter().zip(exact) {
        assert!((got - want).abs() <= 1e-15, "{a:?} against {exact:?}");
    }
    // −0.05 up to the f32 rounding of the decimal controls.
    assert!(a[2] < 0.0 && (a[2] + 0.05).abs() <= 1e-7, "a₂ = {}", a[2]);
}

/// On every leaf the cubic in the leaf's coordinate is the forward's value there, to the forward's `f32` rounding:
/// chord and bounded tangents, uniform and refined knots, several channels.
#[test]
fn each_leaf_cubic_is_the_forwards_curve() {
    let refined = Knots::refined(6, &[0, 2, 0, 1, 0]);
    let cases = [
        Knots::uniform(6),
        Knots::uniform(6).with_tangents(Tangents::Bounded),
        refined.clone(),
        refined.with_tangents(Tangents::Bounded),
    ];
    let channels = 3usize;
    for (case, knots) in cases.iter().enumerate() {
        let coef = field(channels * knots.len(), 11 + case as u64, 1.0);
        let table = catmull_rom_leaf_coefficients(&coef, channels, knots);
        let positions = knots.positions();
        let leaves = positions.len() - 1;
        assert_eq!(table.len(), channels * leaves);
        // Three points inside each leaf, every channel reading the same input.
        let xs: Vec<(usize, f64, f32)> = (0..leaves)
            .flat_map(|leaf| {
                let (lo, hi) = (positions[leaf], positions[leaf + 1]);
                [0.2f32, 0.5, 0.85].map(move |t| (leaf, f64::from(t), lo + t * (hi - lo)))
            })
            .collect();
        let x: Vec<f32> = xs
            .iter()
            .flat_map(|&(_, _, x)| std::iter::repeat_n(x, channels))
            .collect();
        let (y, _) = catmull_rom_forward(&coef, &x, xs.len(), channels, knots);
        for (row, &(leaf, _, xv)) in xs.iter().enumerate() {
            let (lo, hi) = (positions[leaf], positions[leaf + 1]);
            let t = f64::from((xv - lo) / (hi - lo));
            for ch in 0..channels {
                let a = table[ch * leaves + leaf];
                let value = a[0] + t * (a[1] + t * (a[2] + t * a[3]));
                let got = f64::from(y[row * channels + ch]);
                assert!(
                    (value - got).abs() <= 2e-6,
                    "case {case}, leaf {leaf}, channel {ch}: cubic {value}, forward {got}"
                );
            }
        }
    }
}

/// The table refuses control points that are not one per knot and channel.
#[test]
#[should_panic(expected = "one control point per knot and channel")]
fn a_short_table_is_refused() {
    let _ = catmull_rom_leaf_coefficients(&[0.0; 5], 1, &Knots::uniform(6));
}
