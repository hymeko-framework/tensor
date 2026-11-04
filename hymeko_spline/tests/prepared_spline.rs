//! **Prepared coefficient tables** (Nagare plans 2026-10-24-elliptic-cr, O2, and
//! 2026-10-26-cell-spline-tables-per-version): a `PreparedSpline` built once for one or more banks of control points --
//! or for the control points Chebyshev terms generate -- is a [`TableSource`] the banks and routed entries read in place
//! of control points. Every output, tape field, input gradient and parameter gradient equals the per-call path's bit for
//! bit: one to four banks, routed with and without the every-element bank, `SKIP` routes, under both tangent rules, on
//! uniform knots of grids 3 to 8 and on refined knots, on both sides of the parallel threshold and at any thread count,
//! at boundary knots, signed zeros, ties and non-finite inputs. The finite values match an f64 evaluation of the
//! uniform Catmull–Rom definition. Tables of other control points give other values; tables of other channels or knots
//! are refused.

use hymeko_spline::{
    catmull_rom_backward_into, catmull_rom_banks_backward_into, catmull_rom_banks_forward_into,
    catmull_rom_forward_into, catmull_rom_routed_backward_into, catmull_rom_routed_forward_into,
    chebyshev_cr_banks_backward_into, chebyshev_cr_banks_forward_into,
    chebyshev_cr_routed_backward_into, chebyshev_cr_routed_forward_into, chebyshev_knot_basis,
    CatmullRomBanksBuffers, CatmullRomBuffers, CatmullRomCache, Knots, PreparedSpline,
    RoutedBuffers, RoutedCotangents, TableSource, Tangents, SKIP,
};

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Uniform knots of every grid from 3 to 8 under the chord, and grid 6 and refined knots under both rules.
fn knot_cases() -> Vec<Knots> {
    let refined = Knots::refined(6, &[0, 2, 0, 1, 0]);
    (3..=8)
        .map(Knots::uniform)
        .chain([
            Knots::uniform(6).with_tangents(Tangents::Bounded),
            refined.clone(),
            refined.with_tangents(Tangents::Bounded),
        ])
        .collect()
}

/// Rows from one run to past the parallel threshold, `(n, channels)`.
const SHAPES: [(usize, usize); 3] = [(7, 3), (256, 32), (3136, 32)];

/// What a forward and its pullback leave: the outputs, the tape, the input gradient and each bank's parameter
/// gradient, all as bits.
#[derive(Debug, PartialEq, Eq)]
struct Observed {
    outs: Vec<Vec<u32>>,
    x_clamped: Vec<u32>,
    interior: Vec<bool>,
    grad_x: Vec<u32>,
    grads: Vec<Vec<u32>>,
}

/// A banks forward and pullback over `tables`, `banks` of them.
fn banks_run(
    tables: TableSource<'_>,
    banks: usize,
    (x, gys): (&[f32], &[Vec<f32>]),
    (n, channels): (usize, usize),
    knots: &Knots,
) -> Observed {
    let mut outs = vec![vec![0.0f32; x.len()]; banks];
    let mut cache = CatmullRomCache::default();
    catmull_rom_banks_forward_into(
        tables,
        x,
        n,
        channels,
        knots,
        CatmullRomBanksBuffers {
            outs: outs.iter_mut().map(Vec::as_mut_slice).collect(),
            cache: &mut cache,
        },
    );
    let gys: Vec<&[f32]> = gys.iter().map(Vec::as_slice).collect();
    let mut grad_x = vec![f32::NAN; x.len()];
    let grads = catmull_rom_banks_backward_into(tables, &cache, &gys, &mut grad_x);
    Observed {
        outs: outs.iter().map(|o| bits(o)).collect(),
        x_clamped: bits(&cache.x_clamped),
        interior: cache.interior.clone(),
        grad_x: bits(&grad_x),
        grads: grads.iter().map(|g| bits(g)).collect(),
    }
}

/// A routed forward and pullback over `tables`: the first bank on every element where `every`, the routes over the
/// rest.
fn routed_run(
    tables: TableSource<'_>,
    (route, every): (&[u8], bool),
    (x, gy, g_every): (&[f32], &[f32], &[f32]),
    (n, channels): (usize, usize),
    knots: &Knots,
) -> Observed {
    let mut out = vec![f32::NAN; x.len()];
    let mut every_out = vec![f32::NAN; x.len()];
    let mut cache = CatmullRomCache::default();
    catmull_rom_routed_forward_into(
        tables,
        route,
        x,
        (n, channels),
        knots,
        RoutedBuffers {
            out: &mut out,
            every: every.then_some(every_out.as_mut_slice()),
            cache: &mut cache,
        },
    );
    let mut grad_x = vec![f32::NAN; x.len()];
    let grads = catmull_rom_routed_backward_into(
        tables,
        route,
        &cache,
        RoutedCotangents {
            grad_y: gy,
            every: every.then_some(g_every),
        },
        &mut grad_x,
    );
    let mut outs = vec![bits(&out)];
    if every {
        outs.push(bits(&every_out));
    }
    Observed {
        outs,
        x_clamped: bits(&cache.x_clamped),
        interior: cache.interior.clone(),
        grad_x: bits(&grad_x),
        grads: grads.iter().map(|g| bits(g)).collect(),
    }
}

/// Each element's route: one of `routed` banks or `SKIP`, drawn from `seed`.
fn routes(len: usize, routed: usize, seed: u64) -> Vec<u8> {
    field(len, seed, 1.0)
        .iter()
        .map(|v| match ((v + 1.0) * 1000.0) as usize % (routed + 1) {
            b if b < routed => b as u8,
            _ => SKIP,
        })
        .collect()
}

/// One to four banks, per call and prepared, on every knot case and shape: the same bits.
#[test]
fn prepared_banks_give_the_per_call_bits() {
    for (case, knots) in knot_cases().iter().enumerate() {
        for &(n, channels) in &SHAPES {
            for banks in 1..=4 {
                let seed = 1000 * case as u64 + 10 * n as u64 + banks as u64;
                let coefs: Vec<Vec<f32>> = (0..banks)
                    .map(|b| field(channels * knots.len(), seed + b as u64, 1.0))
                    .collect();
                let coefs: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
                let x = field(n * channels, seed + 11, 1.2);
                let gys: Vec<Vec<f32>> = (0..banks)
                    .map(|b| field(n * channels, seed + 20 + b as u64, 1.0))
                    .collect();
                let prepared = PreparedSpline::banks(&coefs, channels, knots);
                assert_eq!(prepared.bank_count(), banks);
                let what = format!("case {case}, n {n}, banks {banks}");
                assert_eq!(
                    banks_run(
                        TableSource::Controls(&coefs),
                        banks,
                        (&x, &gys),
                        (n, channels),
                        knots
                    ),
                    banks_run(
                        TableSource::Prepared(&prepared),
                        banks,
                        (&x, &gys),
                        (n, channels),
                        knots
                    ),
                    "{what}"
                );
            }
        }
    }
}

/// One bank prepared is the single-bank entries' forward and pullback, bit for bit.
#[test]
fn one_prepared_bank_is_the_single_bank_call() {
    for (case, knots) in knot_cases().iter().enumerate() {
        let (n, channels) = (256, 32);
        let coef = field(channels * knots.len(), 7 + case as u64, 1.0);
        let x = field(n * channels, 8 + case as u64, 1.2);
        let gy = field(n * channels, 9 + case as u64, 1.0);
        let mut out = vec![0.0f32; x.len()];
        let mut cache = CatmullRomCache::default();
        catmull_rom_forward_into(
            &coef,
            &x,
            n,
            channels,
            knots,
            CatmullRomBuffers {
                out: &mut out,
                cache: &mut cache,
            },
        );
        let mut grad_x = vec![0.0f32; x.len()];
        let grad = catmull_rom_backward_into(&coef, &cache, &gy, &mut grad_x);
        let single = Observed {
            outs: vec![bits(&out)],
            x_clamped: bits(&cache.x_clamped),
            interior: cache.interior.clone(),
            grad_x: bits(&grad_x),
            grads: vec![bits(&grad)],
        };
        let prepared = PreparedSpline::new(&coef, channels, knots);
        let prepared = banks_run(
            TableSource::Prepared(&prepared),
            1,
            (&x, std::slice::from_ref(&gy)),
            (n, channels),
            knots,
        );
        assert_eq!(single, prepared, "case {case}");
    }
}

/// Routed calls, with and without the every-element bank and with `SKIP` routes, per call and prepared: the same bits.
#[test]
fn prepared_routes_give_the_per_call_bits() {
    for (case, knots) in knot_cases().iter().enumerate() {
        for &(n, channels) in &SHAPES {
            for every in [false, true] {
                let routed = 2;
                let banks = routed + usize::from(every);
                let seed = 5000 + 1000 * case as u64 + n as u64 + u64::from(every);
                let coefs: Vec<Vec<f32>> = (0..banks)
                    .map(|b| field(channels * knots.len(), seed + b as u64, 1.0))
                    .collect();
                let coefs: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
                let x = field(n * channels, seed + 11, 1.2);
                let (gy, g_every) = (
                    field(n * channels, seed + 12, 1.0),
                    field(n * channels, seed + 13, 1.0),
                );
                let route = routes(n * channels, routed, seed + 14);
                let prepared = PreparedSpline::banks(&coefs, channels, knots);
                let run = |tables| {
                    routed_run(
                        tables,
                        (&route, every),
                        (&x, &gy, &g_every),
                        (n, channels),
                        knots,
                    )
                };
                assert_eq!(
                    run(TableSource::Controls(&coefs)),
                    run(TableSource::Prepared(&prepared)),
                    "case {case}, n {n}, every {every}"
                );
            }
        }
    }
}

/// Chebyshev terms: `PreparedSpline::chebyshev` read by the Catmull–Rom entries gives the Chebyshev-CR wrappers'
/// outputs, and its tables and knot basis give their pullbacks' term gradients, bit for bit -- banks and routed, with
/// the every-element bank, terms 1 to 4, grids 3 to 8.
#[test]
fn prepared_chebyshev_gives_the_per_call_bits() {
    let (n, channels) = (3136, 32);
    for grid in 3..=8 {
        for k in 1..=4 {
            let seed = 100 * grid as u64 + k as u64;
            let terms: Vec<Vec<f32>> = (0..3).map(|b| field(channels * k, seed + b, 0.8)).collect();
            let terms: Vec<&[f32]> = terms.iter().map(Vec::as_slice).collect();
            let x = field(n * channels, seed + 11, 1.2);
            let (gy, g_every) = (
                field(n * channels, seed + 12, 1.0),
                field(n * channels, seed + 13, 1.0),
            );
            let route = routes(n * channels, 2, seed + 14);
            let prepared = PreparedSpline::chebyshev(&terms, channels, (grid, k));
            let (basis, kept_k) = prepared
                .chebyshev_terms()
                .expect("Chebyshev tables keep their basis");
            assert_eq!(kept_k, k);
            assert_eq!(bits(basis), bits(&chebyshev_knot_basis(grid, k)));
            let what = format!("grid {grid}, k {k}");

            // Routed, the first bank on every element.
            let (mut out_a, mut every_a) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
            let mut cache_a = CatmullRomCache::default();
            let (controls, basis_a) = chebyshev_cr_routed_forward_into(
                &terms,
                &route,
                &x,
                (n, channels),
                (grid, k),
                RoutedBuffers {
                    out: &mut out_a,
                    every: Some(&mut every_a),
                    cache: &mut cache_a,
                },
            );
            let (mut out_b, mut every_b) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
            let mut cache_b = CatmullRomCache::default();
            catmull_rom_routed_forward_into(
                &prepared,
                &route,
                &x,
                (n, channels),
                prepared.knots(),
                RoutedBuffers {
                    out: &mut out_b,
                    every: Some(&mut every_b),
                    cache: &mut cache_b,
                },
            );
            assert_eq!(bits(&out_a), bits(&out_b), "{what}: routed output");
            assert_eq!(bits(&every_a), bits(&every_b), "{what}: every output");
            assert_eq!(bits(&cache_a.x_clamped), bits(&cache_b.x_clamped));
            let grads = RoutedCotangents {
                grad_y: &gy,
                every: Some(&g_every),
            };
            let controls: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
            let (mut gx_a, mut gx_b) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
            let ga = chebyshev_cr_routed_backward_into(
                &controls,
                &basis_a,
                (&cache_a, &route),
                grads,
                k,
                &mut gx_a,
            );
            let gb = chebyshev_cr_routed_backward_into(
                &prepared,
                basis,
                (&cache_b, &route),
                grads,
                k,
                &mut gx_b,
            );
            assert_eq!(bits(&gx_a), bits(&gx_b), "{what}: routed grad_x");
            for (a, b) in ga.iter().zip(&gb) {
                assert_eq!(bits(a), bits(b), "{what}: routed term gradient");
            }

            // Banks.
            let mut outs_a = vec![vec![0.0f32; x.len()]; 3];
            let mut cache_c = CatmullRomCache::default();
            let (controls, basis_c) = chebyshev_cr_banks_forward_into(
                &terms,
                &x,
                n,
                channels,
                (grid, k),
                CatmullRomBanksBuffers {
                    outs: outs_a.iter_mut().map(Vec::as_mut_slice).collect(),
                    cache: &mut cache_c,
                },
            );
            let mut outs_b = vec![vec![0.0f32; x.len()]; 3];
            let mut cache_d = CatmullRomCache::default();
            catmull_rom_banks_forward_into(
                &prepared,
                &x,
                n,
                channels,
                prepared.knots(),
                CatmullRomBanksBuffers {
                    outs: outs_b.iter_mut().map(Vec::as_mut_slice).collect(),
                    cache: &mut cache_d,
                },
            );
            for (a, b) in outs_a.iter().zip(&outs_b) {
                assert_eq!(bits(a), bits(b), "{what}: banks output");
            }
            let gys = [gy.as_slice(), g_every.as_slice(), gy.as_slice()];
            let controls: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
            let ga =
                chebyshev_cr_banks_backward_into(&controls, &basis_c, &cache_c, &gys, k, &mut gx_a);
            let gb =
                chebyshev_cr_banks_backward_into(&prepared, basis, &cache_d, &gys, k, &mut gx_b);
            assert_eq!(bits(&gx_a), bits(&gx_b), "{what}: banks grad_x");
            for (a, b) in ga.iter().zip(&gb) {
                assert_eq!(bits(a), bits(b), "{what}: banks term gradient");
            }
        }
    }
}

/// The prepared routed call is the same on one and on eight threads, and equal to the per-call path there.
#[cfg(feature = "parallel")]
#[test]
fn prepared_tables_are_the_same_on_any_thread_count() {
    let knots = Knots::uniform(6);
    let (n, channels) = (3136, 32);
    let coefs: Vec<Vec<f32>> = (0..3)
        .map(|b| field(channels * knots.len(), 70 + b, 1.0))
        .collect();
    let coefs: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
    let x = field(n * channels, 80, 1.2);
    let (gy, g_every) = (field(n * channels, 81, 1.0), field(n * channels, 82, 1.0));
    let route = routes(n * channels, 2, 83);
    let prepared = PreparedSpline::banks(&coefs, channels, &knots);
    let run = |threads: usize, tables: TableSource<'_>| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool")
            .install(|| {
                routed_run(
                    tables,
                    (&route, true),
                    (&x, &gy, &g_every),
                    (n, channels),
                    &knots,
                )
            })
    };
    let one = run(1, TableSource::Prepared(&prepared));
    assert_eq!(
        one,
        run(8, TableSource::Prepared(&prepared)),
        "1 against 8 threads"
    );
    assert_eq!(
        one,
        run(8, TableSource::Controls(&coefs)),
        "prepared against per call"
    );
}

/// The inputs at the edges of the contract: the boundary knots and every interior knot, signed zeros, the smallest
/// normal and a subnormal, ties, values outside the clamp interval, both infinities and NaN. Prepared and per call agree
/// by bits, NaN included.
fn edge_inputs(grid: usize) -> Vec<f32> {
    let knots = (0..grid).map(|g| -1.0 + 2.0 * g as f32 / (grid - 1) as f32);
    knots
        .chain([
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0e-45,
            -1.0e-45,
            1.0e-7,
            -1.0e-7,
            0.25,
            0.25,
            0.25,
            1.0 - f32::EPSILON,
            -1.0 + f32::EPSILON,
            1.5,
            -1.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ])
        .collect()
}

/// The uniform Catmull–Rom weights at `t` and their derivatives, from the definition: `w(t) = ½ (−t + 2t² − t³,
/// 2 − 5t² + 3t³, t + 4t² − 3t³, −t² + t³)`.
fn weights(t: f64) -> ([f64; 4], [f64; 4]) {
    let (t2, t3) = (t * t, t * t * t);
    (
        [
            0.5 * (-t + 2.0 * t2 - t3),
            0.5 * (2.0 - 5.0 * t2 + 3.0 * t3),
            0.5 * (t + 4.0 * t2 - 3.0 * t3),
            0.5 * (-t2 + t3),
        ],
        [
            0.5 * (-1.0 + 4.0 * t - 3.0 * t2),
            0.5 * (-10.0 * t + 9.0 * t2),
            0.5 * (1.0 + 8.0 * t - 9.0 * t2),
            0.5 * (-2.0 * t + 3.0 * t2),
        ],
    )
}

/// One channel's value, `dy/dx` and per-control weights at `x`, in f64 from the definition: the clamp to `[-1, 1]`,
/// the segment `i = min(⌊u⌋, grid − 2)` of `u = (x + 1)(grid − 1)/2`, the four controls `i − 1 … i + 2` with the ends
/// replicated, `dy/dx = 0` outside the closed interval.
fn reference(q: &[f32], x: f32) -> (f64, f64, [(usize, f64); 4]) {
    let grid = q.len();
    let xc = f64::from(x).clamp(-1.0, 1.0);
    let span = 0.5 * (grid - 1) as f64;
    let u = (xc + 1.0) * span;
    let i = (u.floor() as usize).min(grid - 2);
    let t = u - i as f64;
    let at = [i.saturating_sub(1), i, i + 1, (i + 2).min(grid - 1)];
    let (w, dw) = weights(t);
    let value = (0..4).map(|k| w[k] * f64::from(q[at[k]])).sum();
    let interior = (-1.0..=1.0).contains(&f64::from(x));
    let slope: f64 = (0..4).map(|k| dw[k] * f64::from(q[at[k]])).sum::<f64>() * span;
    let per_control = std::array::from_fn(|k| (at[k], w[k]));
    (value, if interior { slope } else { 0.0 }, per_control)
}

/// Edge inputs, prepared against per call by bits on every knot case, and on uniform chord knots against the f64
/// definition: the value and the input gradient within `2e-6` of the scale, the control-point gradient within `1e-5`
/// of its terms' magnitude.
#[test]
fn edge_inputs_give_the_per_call_bits_and_the_f64_definition() {
    for (case, knots) in knot_cases().iter().enumerate() {
        let grid = knots.grid();
        let column = edge_inputs(grid);
        let channels = 2;
        let n = column.len();
        let x: Vec<f32> = column.iter().flat_map(|&v| [v, -v]).collect();
        let coef = field(channels * knots.len(), 40 + case as u64, 1.0);
        let gy = field(n * channels, 50 + case as u64, 1.0);
        let prepared = PreparedSpline::new(&coef, channels, knots);
        let per_call = banks_run(
            TableSource::Controls(&[&coef]),
            1,
            (&x, std::slice::from_ref(&gy)),
            (n, channels),
            knots,
        );
        let tables = banks_run(
            TableSource::Prepared(&prepared),
            1,
            (&x, std::slice::from_ref(&gy)),
            (n, channels),
            knots,
        );
        assert_eq!(per_call, tables, "case {case}");
        if knots.tangents() != Tangents::Chord || knots.len() != grid {
            continue;
        }
        let y: Vec<f32> = tables.outs[0].iter().map(|&b| f32::from_bits(b)).collect();
        let gx: Vec<f32> = tables.grad_x.iter().map(|&b| f32::from_bits(b)).collect();
        let gc: Vec<f32> = tables.grads[0].iter().map(|&b| f32::from_bits(b)).collect();
        let mut want_gc = vec![0.0f64; coef.len()];
        let mut scale_gc = vec![0.0f64; coef.len()];
        let mut checked = 0usize;
        for (e, &xe) in x.iter().enumerate() {
            let ch = e % channels;
            let q = &coef[ch * grid..(ch + 1) * grid];
            if xe.is_nan() {
                assert!(y[e].is_nan(), "case {case}: NaN in, NaN out");
                continue;
            }
            let (value, slope, per_control) = reference(q, xe);
            let what = format!("case {case}, x {xe:e}");
            assert!((f64::from(y[e]) - value).abs() <= 2e-6, "{what}: value");
            assert!(
                (f64::from(gx[e]) - f64::from(gy[e]) * slope).abs() <= 2e-6 * (1.0 + slope.abs()),
                "{what}: input gradient"
            );
            for (j, w) in per_control {
                want_gc[ch * grid + j] += f64::from(gy[e]) * w;
                scale_gc[ch * grid + j] += (f64::from(gy[e]) * w).abs();
            }
            checked += 1;
        }
        assert_eq!(
            checked,
            x.len() - 2,
            "case {case}: every finite input checked"
        );
        for (j, ((&got, want), scale)) in gc.iter().zip(&want_gc).zip(&scale_gc).enumerate() {
            if got.is_nan() {
                continue;
            }
            assert!(
                (f64::from(got) - want).abs() <= 1e-5 * (1.0 + scale),
                "case {case}: control {j} gradient {got} against {want}"
            );
        }
    }
}

/// Tables are their control points': repeated calls over one `PreparedSpline` give the same bits, and tables of
/// mutated control points of the same shape give the per-call values of the mutated points, not the old ones.
#[test]
fn mutated_control_points_give_other_tables() {
    let knots = Knots::uniform(6);
    let (n, channels) = (256, 32);
    let coef = field(channels * knots.len(), 90, 1.0);
    let mut mutated = coef.clone();
    mutated[channels * knots.len() / 2] += 0.25;
    let x = field(n * channels, 91, 1.2);
    let gys = [field(n * channels, 92, 1.0)];
    let old = PreparedSpline::new(&coef, channels, &knots);
    let new = PreparedSpline::new(&mutated, channels, &knots);
    let run = |tables| banks_run(tables, 1, (&x, &gys), (n, channels), &knots);
    let first = run(TableSource::Prepared(&old));
    assert_eq!(first, run(TableSource::Prepared(&old)), "repeated calls");
    let changed = run(TableSource::Prepared(&new));
    assert_eq!(
        changed,
        run(TableSource::Controls(&[&mutated])),
        "the mutated points' values"
    );
    assert_ne!(first.outs, changed.outs, "other points, other values");
    assert_ne!(first.grad_x, changed.grad_x, "other points, other slopes");
}

/// The tables refuse a cache written over other knots: they belong to their control points and knots alone.
#[test]
#[should_panic(expected = "the tables' knots")]
fn prepared_tables_refuse_a_cache_of_other_knots() {
    let (n, channels) = (5usize, 2usize);
    let chord = Knots::uniform(6);
    let bounded = Knots::uniform(6).with_tangents(Tangents::Bounded);
    let coef = field(channels * 6, 3, 1.0);
    let x = field(n * channels, 4, 1.0);
    let mut out = vec![0.0f32; x.len()];
    let mut cache = CatmullRomCache::default();
    catmull_rom_forward_into(
        &coef,
        &x,
        n,
        channels,
        &chord,
        CatmullRomBuffers {
            out: &mut out,
            cache: &mut cache,
        },
    );
    let prepared = PreparedSpline::new(&coef, channels, &bounded);
    let mut gx = vec![0.0f32; x.len()];
    let _ = catmull_rom_banks_backward_into(&prepared, &cache, &[&x], &mut gx);
}

/// The tables refuse a call over other channels.
#[test]
#[should_panic(expected = "the tables' channels")]
fn prepared_tables_refuse_other_channels() {
    let knots = Knots::uniform(6);
    let prepared = PreparedSpline::new(&field(3 * 6, 5, 1.0), 3, &knots);
    let x = field(8, 6, 1.0);
    let mut out = vec![0.0f32; x.len()];
    let mut cache = CatmullRomCache::default();
    catmull_rom_banks_forward_into(
        &prepared,
        &x,
        4,
        2,
        &knots,
        CatmullRomBanksBuffers {
            outs: vec![&mut out],
            cache: &mut cache,
        },
    );
}

/// A banks call over prepared tables takes one output per prepared bank.
#[test]
#[should_panic(expected = "one output per bank")]
fn prepared_tables_refuse_another_bank_count() {
    let knots = Knots::uniform(6);
    let coefs = [field(2 * 6, 7, 1.0), field(2 * 6, 8, 1.0)];
    let coefs: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
    let prepared = PreparedSpline::banks(&coefs, 2, &knots);
    let x = field(8, 9, 1.0);
    let mut out = vec![0.0f32; x.len()];
    let mut cache = CatmullRomCache::default();
    catmull_rom_banks_forward_into(
        &prepared,
        &x,
        4,
        2,
        &knots,
        CatmullRomBanksBuffers {
            outs: vec![&mut out],
            cache: &mut cache,
        },
    );
}

/// **Row primitives give the kernels' values and gradients** (Nagare plan 2026-10-26-fused-cells, Track C): a traversal
/// that locates each row at once (`locate_row`), evaluates the routed cubics from `power_table` and the slopes from
/// `slope_table`, and adds the moments into `Moments::bank_mut` in its own order (one block here), then reduces with
/// `gradients`, reproduces the routed kernels' outputs, input gradient and every bank's gradient within `1e-6` of their
/// scale -- control points and Chebyshev terms, on both sides of the parallel threshold. Every compared value is finite.
#[test]
fn row_primitives_give_the_kernels_values_and_gradients() {
    for chebyshev in [false, true] {
        for &(n, channels) in &[(7usize, 3usize), (256, 32), (3136, 32)] {
            let knots = Knots::uniform(6);
            let seed = 900 + n as u64 + u64::from(chebyshev);
            let k = 3;
            let width = if chebyshev { k } else { knots.len() };
            let coefs: Vec<Vec<f32>> = (0..3)
                .map(|b| field(channels * width, seed + b, 0.8))
                .collect();
            let coefs: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
            let prepared = if chebyshev {
                PreparedSpline::chebyshev(&coefs, channels, (6, k))
            } else {
                PreparedSpline::banks(&coefs, channels, &knots)
            };
            let x = field(n * channels, seed + 11, 1.2);
            let (gy, g_every) = (
                field(n * channels, seed + 12, 1.0),
                field(n * channels, seed + 13, 1.0),
            );
            let route: Vec<u8> = routes(n * channels, 2, seed + 14)
                .iter()
                .map(|&r| if r == SKIP { 0 } else { r })
                .collect();
            // The kernels.
            let (mut out, mut every) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
            let mut cache = CatmullRomCache::default();
            catmull_rom_routed_forward_into(
                &prepared,
                &route,
                &x,
                (n, channels),
                prepared.knots(),
                RoutedBuffers {
                    out: &mut out,
                    every: Some(&mut every),
                    cache: &mut cache,
                },
            );
            let mut grad_x = vec![0.0f32; x.len()];
            let grads = RoutedCotangents {
                grad_y: &gy,
                every: Some(&g_every),
            };
            let want = match prepared.chebyshev_terms() {
                Some((basis, k)) => chebyshev_cr_routed_backward_into(
                    &prepared,
                    basis,
                    (&cache, &route),
                    grads,
                    k,
                    &mut grad_x,
                ),
                None => {
                    catmull_rom_routed_backward_into(&prepared, &route, &cache, grads, &mut grad_x)
                }
            };
            // The row primitives.
            let leaves = prepared.leaves();
            let (mut leaf, mut t, mut inside) = (
                vec![0u32; channels],
                vec![0.0f32; channels],
                vec![false; channels],
            );
            let (mut out_r, mut every_r, mut grad_r) = (
                vec![0.0f32; x.len()],
                vec![0.0f32; x.len()],
                vec![0.0f32; x.len()],
            );
            let mut moments = prepared.moments();
            for r in 0..n {
                let span = r * channels..(r + 1) * channels;
                prepared.locate_row(&x[span.clone()], (&mut leaf, &mut t, &mut inside));
                for c in 0..channels {
                    let i = r * channels + c;
                    let (at, tc) = (c * leaves + leaf[c] as usize, t[c]);
                    let bank = 1 + usize::from(route[i]);
                    let cubic = |a: [f32; 4]| a[0] + tc * (a[1] + tc * (a[2] + tc * a[3]));
                    let slope = |b: [f32; 3]| {
                        if inside[c] {
                            b[0] + tc * (b[1] + tc * b[2])
                        } else {
                            0.0
                        }
                    };
                    every_r[i] = cubic(prepared.power_table(0)[at]);
                    out_r[i] = cubic(prepared.power_table(bank)[at]);
                    grad_r[i] = g_every[i] * slope(prepared.slope_table(0)[at])
                        + gy[i] * slope(prepared.slope_table(bank)[at]);
                    for (b, g) in [(0usize, g_every[i]), (bank, gy[i])] {
                        let m = &mut moments.bank_mut(b)[at];
                        let g1 = g * tc;
                        let g2 = g1 * tc;
                        m[0] += g;
                        m[1] += g1;
                        m[2] += g2;
                        m[3] += g2 * tc;
                    }
                }
            }
            let got = prepared.gradients(&[moments]);
            let what = format!("chebyshev {chebyshev}, n {n}");
            let close = |name: &str, a: &[f32], b: &[f32], tol: f32| {
                assert!(
                    a.iter().chain(b).all(|v| v.is_finite()),
                    "{what}: {name} finite"
                );
                let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
                let dev = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    dev <= tol * scale.max(1.0),
                    "{what}: {name} deviates {dev} (scale {scale})"
                );
            };
            // Element-wise values: 1e-6. The gradients accumulate n rows per (channel, leaf) in f32, here in one block
            // against the kernels' 64 blocks added in f64: 1e-5 (measured 1.1e-6 at n = 3136).
            close("routed output", &out_r, &out, 1e-6);
            close("every output", &every_r, &every, 1e-6);
            close("input gradient", &grad_r, &grad_x, 1e-6);
            for (b, (g, w)) in got.iter().zip(&want).enumerate() {
                close(&format!("bank {b}'s gradient"), g, w, 1e-5);
            }
        }
    }
}
