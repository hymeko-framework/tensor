//! Catmull–Rom knots on an orthtree (plan `docs/plans/2026-10-16-cr-orthtree-knots/`): each base
//! segment refined to its own dyadic depth, evaluated by the one kernel the uniform grid uses.

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, Knots};

/// Grid 6 with its middle refined: depths `(0, 1, 2, 1, 0)`, eleven leaves of three widths.
fn graded() -> Knots {
    Knots::refined(6, &[0, 1, 2, 1, 0])
}

/// Control points over `knots`, `channels` of them, with distinct curvature per channel.
fn control(knots: &Knots, channels: usize) -> Vec<f32> {
    (0..channels)
        .flat_map(|ch| {
            knots
                .positions()
                .iter()
                .map(move |&x| (2.3 * x + 0.4 * ch as f32).sin() * 0.8 + 0.1 * x * x)
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// **The knots are the base grid and the leaves' ends** (D1): `n = g + Σ (2^d − 1)`, sorted, the
/// base knots among them; depth zero everywhere is the uniform grid.
#[test]
fn the_knots_are_the_base_grid_and_the_leaves_ends() {
    let knots = graded();
    assert_eq!(knots.len(), 6 + 1 + 3 + 1);
    assert!(knots.positions().windows(2).all(|w| w[0] < w[1]));
    for b in [-1.0f32, -0.6, -0.2, 0.2, 0.6, 1.0] {
        assert!(
            knots.positions().iter().any(|&x| (x - b).abs() < 1e-6),
            "base knot {b}"
        );
    }
    assert_eq!(Knots::refined(6, &[0; 5]), Knots::uniform(6));
    assert!(Knots::uniform(6).is_uniform() && !knots.is_uniform());
}

/// **A refined spline interpolates its control points** (T2): at every knot the output is that
/// knot's control point.
#[test]
fn a_refined_spline_interpolates_its_control_points() {
    let (knots, channels) = (graded(), 2);
    let coef = control(&knots, channels);
    let x: Vec<f32> = knots.positions().iter().flat_map(|&x| [x; 2]).collect();
    let (y, _) = catmull_rom_forward(&coef, &x, knots.len(), channels, &knots);
    for (k, row) in y.chunks(channels).enumerate() {
        for (ch, &v) in row.iter().enumerate() {
            let want = coef[ch * knots.len() + k];
            assert!(
                (v - want).abs() < 1e-5,
                "knot {k}, channel {ch}: {v} vs {want}"
            );
        }
    }
}

/// **Refinement keeps the function at every knot** (T3, D4): `transfer` gives the refined spline's
/// control points as the coarse spline at the new knots -- the old control points at the old knots,
/// and at a midpoint of four equally spaced knots the 4-point rule `(−1, 9, 9, −1) / 16`.
#[test]
fn a_transfer_keeps_the_function_at_every_knot() {
    let coarse = Knots::uniform(6);
    let channels = 2;
    let coef = control(&coarse, channels);
    let fine = graded();
    let moved = coarse.transfer(&coef, channels, &fine);
    assert_eq!(moved.len(), channels * fine.len());
    for ch in 0..channels {
        let old = &coef[ch * 6..(ch + 1) * 6];
        let new = &moved[ch * fine.len()..(ch + 1) * fine.len()];
        for (k, &x) in fine.positions().iter().enumerate() {
            if let Some(j) = coarse
                .positions()
                .iter()
                .position(|&b| (b - x).abs() < 1e-6)
            {
                assert!((new[k] - old[j]).abs() < 1e-6, "old knot {x}");
            }
        }
        // The midpoint of base segment 2 (0.0) has equally spaced neighbours -0.6, -0.2, 0.2, 0.6.
        let mid = fine
            .positions()
            .iter()
            .position(|&x| x.abs() < 1e-6)
            .expect("0 is a knot");
        let rule = (-old[1] + 9.0 * old[2] + 9.0 * old[3] - old[4]) / 16.0;
        assert!(
            (new[mid] - rule).abs() < 1e-5,
            "4-point rule: {} vs {rule}",
            new[mid]
        );
    }
}

/// **Every input lands in the leaf that holds it** (T5): the spline's slope sign and value come from
/// the right segment -- checked by evaluating at points just inside each leaf's ends, where a wrong
/// leaf would extrapolate a neighbour's cubic.
#[test]
fn every_input_lands_in_the_leaf_that_holds_it() {
    let knots = graded();
    let coef = control(&knots, 1);
    let positions = knots.positions();
    for k in 0..positions.len() - 1 {
        let (a, b) = (positions[k], positions[k + 1]);
        let probe = [a + 1e-4 * (b - a), 0.5 * (a + b), b - 1e-4 * (b - a)];
        let (y, _) = catmull_rom_forward(&coef, &probe, 3, 1, &knots);
        // Near the leaf's ends the spline is near its control points there.
        assert!((y[0] - coef[k]).abs() < 1e-2, "leaf {k} start");
        assert!((y[2] - coef[k + 1]).abs() < 1e-2, "leaf {k} end");
    }
}

/// `Σ wᵢ yᵢ` over refined knots.
fn weighted(coef: &[f32], x: &[f32], channels: usize, knots: &Knots, w: &[f32]) -> f64 {
    let (y, _) = catmull_rom_forward(coef, x, x.len() / channels, channels, knots);
    y.iter().zip(w).map(|(y, w)| f64::from(y * w)).sum()
}

/// **The pullback is the derivative over refined knots** (T4): central differences in every input
/// and every control point, the inputs kept a step away from the knots, where the cubic's second
/// derivative jumps.
#[test]
fn the_pullback_matches_central_differences_over_refined_knots() {
    let (knots, channels) = (graded(), 2);
    let coef = control(&knots, channels);
    let positions = knots.positions().to_vec();
    let x: Vec<f32> = (0..24)
        .map(|i| ((i as f32 * 0.618_034).fract() * 2.0 - 1.0) * 0.97)
        .map(|v| {
            let near = positions.iter().any(|&p| (p - v).abs() < 0.01);
            if near {
                v + 0.015
            } else {
                v
            }
        })
        .collect();
    let w: Vec<f32> = (0..x.len()).map(|i| 0.5 - 0.07 * i as f32).collect();
    let (_, cache) = catmull_rom_forward(&coef, &x, x.len() / channels, channels, &knots);
    let grads = catmull_rom_backward(&coef, &cache, &w);
    let h = 1e-3f32;
    for i in 0..x.len() {
        let mut up = x.clone();
        let mut down = x.clone();
        up[i] += h;
        down[i] -= h;
        let fd = (weighted(&coef, &up, channels, &knots, &w)
            - weighted(&coef, &down, channels, &knots, &w))
            / f64::from(2.0 * h);
        let a = f64::from(grads.grad_x[i]);
        assert!(
            (fd - a).abs() <= 3e-3 * fd.abs().max(1.0),
            "x[{i}] = {}: {fd} vs {a}",
            x[i]
        );
    }
    for j in 0..coef.len() {
        let mut up = coef.clone();
        let mut down = coef.clone();
        up[j] += h;
        down[j] -= h;
        let fd = (weighted(&up, &x, channels, &knots, &w)
            - weighted(&down, &x, channels, &knots, &w))
            / f64::from(2.0 * h);
        let a = f64::from(grads.grad_coef[j]);
        assert!(
            (fd - a).abs() <= 3e-3 * fd.abs().max(1.0),
            "coef[{j}]: {fd} vs {a}"
        );
    }
}

/// **The leaves follow the inputs' mass** (D5): inputs gathered around zero refine the middle base
/// segment deepest and leave the empty edges at depth zero.
#[test]
fn a_density_refinement_follows_the_inputs() {
    let samples: Vec<f32> = (0..10_000)
        .map(|i| 0.15 * ((i as f32 * 0.618_034).fract() * 2.0 - 1.0))
        .collect();
    let knots = Knots::refined_by_density(6, &samples, 8.0, 4);
    assert_eq!(knots.depths(), &[0, 0, 3, 0, 0]);
    let spread: Vec<f32> = (0..10_000)
        .map(|i| (i as f32 * 0.618_034).fract() * 2.0 - 1.0)
        .collect();
    let even = Knots::refined_by_density(6, &spread, 8.0, 4);
    assert!(even.depths().iter().all(|&d| d == 1), "{:?}", even.depths());
}

/// **Thread count changes no bit over refined knots** (T6).
#[cfg(feature = "parallel")]
#[test]
fn refined_knots_are_thread_independent() {
    let (knots, channels) = (graded(), 8);
    let coef = control(&knots, channels);
    let n = 2048;
    let x: Vec<f32> = (0..n * channels)
        .map(|i| (i as f32 * 0.618_034).fract() * 2.4 - 1.2)
        .collect();
    let g: Vec<f32> = (0..x.len()).map(|i| (i as f32 * 0.013).sin()).collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| {
                let (y, cache) = catmull_rom_forward(&coef, &x, n, channels, &knots);
                let b = catmull_rom_backward(&coef, &cache, &g);
                (bits(&y), bits(&b.grad_x), bits(&b.grad_coef))
            })
    };
    assert_eq!(run(1), run(8));
}
