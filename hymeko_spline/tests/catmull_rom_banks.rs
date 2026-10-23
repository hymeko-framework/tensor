//! Catmull--Rom banks (tensor plan 2026-10-22-cr-banks): several banks of control points over one
//! input share the clamp, the segment and the weights of every element, and one tape. Each bank's
//! output and control-point gradient are its own single-bank call's, bit for bit; the input gradient
//! is the single-bank input gradients summed in bank order. On either side of the parallel threshold,
//! over uniform and refined knots, under Chebyshev-CR, and at any thread count.

use hymeko_spline::{
    catmull_rom_backward_into, catmull_rom_banks_backward_into, catmull_rom_banks_forward_into,
    catmull_rom_forward, catmull_rom_routed_backward_into, catmull_rom_routed_forward_into,
    chebyshev_cr_backward_into, chebyshev_cr_banks_backward_into, chebyshev_cr_banks_forward_into,
    chebyshev_cr_forward, chebyshev_cr_routed_backward_into, chebyshev_cr_routed_forward_into,
    CatmullRomBanksBuffers, CatmullRomBuffers, CatmullRomCache, Knots, SKIP,
};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % 3001) as f32 / 1000.0 - 1.5
    }
}

/// Values in `[-1.5, 1.5]`: a third of the inputs lie outside the clamp interval.
fn field(len: usize, seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..len).map(|_| next()).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Shapes on both sides of the parallel threshold: `(n, channels)`.
const SHAPES: [(usize, usize); 3] = [(37, 3), (4096, 8), (9001, 5)];

const BANKS: usize = 3;

/// The single-bank reference: each bank's output, the cache, each bank's control-point gradient, and the
/// input gradients summed in bank order.
fn separately(
    coefs: &[Vec<f32>],
    x: &[f32],
    grad_ys: &[Vec<f32>],
    (n, channels): (usize, usize),
    knots: &Knots,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let mut outs = Vec::new();
    let mut grads = Vec::new();
    let mut grad_x: Vec<f32> = Vec::new();
    for (coef, grad_y) in coefs.iter().zip(grad_ys) {
        let (out, cache) = catmull_rom_forward(coef, x, n, channels, knots);
        let mut gx = vec![0.0f32; x.len()];
        grads.push(catmull_rom_backward_into(coef, &cache, grad_y, &mut gx));
        grad_x = if grad_x.is_empty() {
            gx
        } else {
            grad_x.iter().zip(&gx).map(|(a, b)| a + b).collect()
        };
        outs.push(out);
    }
    (outs, grads, grad_x)
}

/// The banks call: each bank's output, each bank's control-point gradient, the summed input gradient.
fn together(
    coefs: &[Vec<f32>],
    x: &[f32],
    grad_ys: &[Vec<f32>],
    (n, channels): (usize, usize),
    knots: &Knots,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let mut outs = vec![vec![0.0f32; x.len()]; coefs.len()];
    let mut cache = CatmullRomCache::default();
    let banks: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
    catmull_rom_banks_forward_into(
        &banks,
        x,
        n,
        channels,
        knots,
        CatmullRomBanksBuffers {
            outs: outs.iter_mut().map(Vec::as_mut_slice).collect(),
            cache: &mut cache,
        },
    );
    let cotangents: Vec<&[f32]> = grad_ys.iter().map(Vec::as_slice).collect();
    let mut grad_x = vec![0.0f32; x.len()];
    let grads = catmull_rom_banks_backward_into(&banks, &cache, &cotangents, &mut grad_x);
    (outs, grads, grad_x)
}

fn fixture(
    (n, channels): (usize, usize),
    per: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
    let coefs = (0..BANKS)
        .map(|b| field(channels * per, seed + b as u64))
        .collect();
    let grad_ys = (0..BANKS)
        .map(|b| field(n * channels, seed + 10 + b as u64))
        .collect();
    (coefs, field(n * channels, seed + 20), grad_ys)
}

fn assert_same(
    what: &str,
    got: &(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>),
    want: &(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>),
) {
    for b in 0..BANKS {
        assert_eq!(
            bits(&got.0[b]),
            bits(&want.0[b]),
            "{what}: bank {b}'s output"
        );
        assert_eq!(
            bits(&got.1[b]),
            bits(&want.1[b]),
            "{what}: bank {b}'s gradient"
        );
    }
    assert_eq!(bits(&got.2), bits(&want.2), "{what}: the input gradient");
}

/// **Banks are their single-bank calls**, over uniform knots and knots refined in the middle segment.
#[test]
fn banks_are_their_single_bank_calls() {
    let knot_sets = [Knots::uniform(6), Knots::refined(6, &[0, 0, 3, 0, 0])];
    for knots in &knot_sets {
        for (i, &shape) in SHAPES.iter().enumerate() {
            let (coefs, x, grad_ys) = fixture(shape, knots.len(), 100 + i as u64);
            let want = separately(&coefs, &x, &grad_ys, shape, knots);
            let got = together(&coefs, &x, &grad_ys, shape, knots);
            assert_same(
                &format!("{shape:?} over {} knots", knots.len()),
                &got,
                &want,
            );
        }
    }
}

/// **One bank is the single-bank call**, which runs through it: the input gradient is assigned, not
/// added, so even a negative zero survives.
#[test]
fn one_bank_is_the_single_bank_call() {
    let knots = Knots::uniform(5);
    let (n, channels) = (4096, 8);
    let (coefs, x, grad_ys) = fixture((n, channels), knots.len(), 7);
    let want = separately(&coefs[..1], &x, &grad_ys[..1], (n, channels), &knots);
    let got = together(&coefs[..1], &x, &grad_ys[..1], (n, channels), &knots);
    assert_eq!(bits(&got.0[0]), bits(&want.0[0]));
    assert_eq!(bits(&got.1[0]), bits(&want.1[0]));
    assert_eq!(bits(&got.2), bits(&want.2));
}

/// **Chebyshev-CR banks are their single-bank calls**: each bank's terms generate its control points, and
/// each bank's term gradient is its own call's.
#[test]
fn chebyshev_banks_are_their_single_bank_calls() {
    let (grid, k) = (6, 3);
    for (i, &(n, channels)) in SHAPES.iter().enumerate() {
        let (terms, x, grad_ys) = fixture((n, channels), k, 300 + i as u64);
        let mut want_x: Vec<f32> = vec![0.0; x.len()];
        let mut want = Vec::new();
        for (b, (coef, grad_y)) in terms.iter().zip(&grad_ys).enumerate() {
            let (out, cache, control, basis) = chebyshev_cr_forward(coef, &x, n, channels, grid, k);
            let mut gx = vec![0.0f32; x.len()];
            let grad = chebyshev_cr_backward_into(&control, &basis, &cache, grad_y, k, &mut gx);
            want_x = if b == 0 {
                gx
            } else {
                want_x.iter().zip(&gx).map(|(a, g)| a + g).collect()
            };
            want.push((out, grad));
        }
        let mut outs = vec![vec![0.0f32; x.len()]; BANKS];
        let mut cache = CatmullRomCache::default();
        let banks: Vec<&[f32]> = terms.iter().map(Vec::as_slice).collect();
        let (controls, basis) = chebyshev_cr_banks_forward_into(
            &banks,
            &x,
            n,
            channels,
            (grid, k),
            CatmullRomBanksBuffers {
                outs: outs.iter_mut().map(Vec::as_mut_slice).collect(),
                cache: &mut cache,
            },
        );
        let controls: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
        let cotangents: Vec<&[f32]> = grad_ys.iter().map(Vec::as_slice).collect();
        let mut grad_x = vec![0.0f32; x.len()];
        let grads = chebyshev_cr_banks_backward_into(
            &controls,
            &basis,
            &cache,
            &cotangents,
            k,
            &mut grad_x,
        );
        for b in 0..BANKS {
            assert_eq!(bits(&outs[b]), bits(&want[b].0), "bank {b}'s output");
            assert_eq!(
                bits(&grads[b]),
                bits(&want[b].1),
                "bank {b}'s term gradient"
            );
        }
        assert_eq!(bits(&grad_x), bits(&want_x), "the input gradient");
    }
}

/// **Banks are the same on any thread count**: the parallel path's blocks are the sequential path's.
#[cfg(feature = "parallel")]
#[test]
fn banks_are_the_same_on_any_thread_count() {
    let knots = Knots::uniform(6);
    let shape = (9001, 5);
    let (coefs, x, grad_ys) = fixture(shape, knots.len(), 500);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool")
            .install(|| together(&coefs, &x, &grad_ys, shape, &knots))
    };
    assert_same("1 against 4 threads", &run(4), &run(1));
}

/// Each element's route: one of the banks or [`SKIP`], drawn from `seed`.
fn routes(len: usize, seed: u64) -> Vec<u8> {
    let mut next = lcg(seed);
    (0..len)
        .map(|_| match ((next() + 1.5) * 1000.0) as usize % (BANKS + 1) {
            b if b < BANKS => b as u8,
            _ => SKIP,
        })
        .collect()
}

/// Each bank's cotangent: `grad_y` at the elements routed to it, zero elsewhere.
fn masked(grad_y: &[f32], route: &[u8]) -> Vec<Vec<f32>> {
    (0..BANKS)
        .map(|b| {
            grad_y
                .iter()
                .zip(route)
                .map(|(&g, &r)| if usize::from(r) == b { g } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Equal as numbers: the routed pullback adds no zero terms, so a sum that is exactly zero may differ in its sign.
fn assert_equal(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (j, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(a == b, "{what} {j}: {a} vs {b}");
    }
}

/// The routed calls (Nagare plan 2026-10-23-incidence-partition, Amendment 5): the output and the cache, and each
/// bank's control-point gradient with the input gradient.
fn routed(
    coefs: &[Vec<f32>],
    (x, route, grad_y): (&[f32], &[u8], &[f32]),
    shape: (usize, usize),
    knots: &Knots,
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let banks: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
    let (mut out, mut cache) = (vec![0.0f32; x.len()], CatmullRomCache::default());
    let into = CatmullRomBuffers {
        out: &mut out,
        cache: &mut cache,
    };
    catmull_rom_routed_forward_into(&banks, route, x, shape, knots, into);
    let mut grad_x = vec![0.0f32; x.len()];
    let grads = catmull_rom_routed_backward_into(&banks, route, &cache, grad_y, &mut grad_x);
    (out, grads, grad_x)
}

/// **A routed element is its routed bank's** (Nagare plan 2026-10-23-incidence-partition, Amendment 5), over uniform
/// and refined knots on both sides of the parallel threshold:
/// - the output is, element by element and bit for bit, its routed bank's single-bank output, and `0` where it is
///   skipped;
/// - the gradients are the banks backward's under each bank's cotangent at the elements routed to it and zero
///   elsewhere, equal as numbers.
#[test]
fn a_routed_element_is_its_routed_banks() {
    let knot_sets = [Knots::uniform(6), Knots::refined(6, &[0, 0, 3, 0, 0])];
    for knots in &knot_sets {
        for (i, &shape) in SHAPES.iter().enumerate() {
            let (coefs, x, grad_ys) = fixture(shape, knots.len(), 700 + 10 * i as u64);
            let route = routes(x.len(), 800 + i as u64);
            let grad_y = &grad_ys[0];
            let (out, grads, grad_x) = routed(&coefs, (&x, &route, grad_y), shape, knots);
            let alone = separately(&coefs, &x, &grad_ys, shape, knots).0;
            let want: Vec<f32> = route
                .iter()
                .enumerate()
                .map(|(e, &r)| alone.get(usize::from(r)).map_or(0.0, |o| o[e]))
                .collect();
            assert_eq!(bits(&out), bits(&want), "{shape:?}: the routed output");
            let want = together(&coefs, &x, &masked(grad_y, &route), shape, knots);
            for (b, (got, want)) in grads.iter().zip(&want.1).enumerate() {
                assert_equal(&format!("{shape:?}: bank {b}'s gradient"), got, want);
            }
            assert_equal(&format!("{shape:?}: the input gradient"), &grad_x, &want.2);
        }
    }
}

/// **Routed Chebyshev-CR is its routed bank's**: the generated control points route as the Catmull--Rom banks do,
/// and the Chebyshev-term gradients are the banks' under the masked cotangents.
#[test]
fn routed_chebyshev_is_its_routed_banks() {
    let (grid, k) = (6usize, 4usize);
    let shape = (4096, 8);
    let coefs: Vec<Vec<f32>> = (0..BANKS)
        .map(|b| field(shape.1 * k, 900 + b as u64))
        .collect();
    let x = field(shape.0 * shape.1, 910);
    let grad_y = field(x.len(), 911);
    let route = routes(x.len(), 912);
    let banks: Vec<&[f32]> = coefs.iter().map(Vec::as_slice).collect();
    let (mut out, mut cache) = (vec![0.0f32; x.len()], CatmullRomCache::default());
    let into = CatmullRomBuffers {
        out: &mut out,
        cache: &mut cache,
    };
    let (controls, basis) =
        chebyshev_cr_routed_forward_into(&banks, &route, &x, shape, (grid, k), into);
    let control_banks: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
    let mut grad_x = vec![0.0f32; x.len()];
    let grads = chebyshev_cr_routed_backward_into(
        &control_banks,
        &basis,
        (&cache, &route),
        &grad_y,
        k,
        &mut grad_x,
    );
    let mut outs = vec![vec![0.0f32; x.len()]; BANKS];
    let mut banks_cache = CatmullRomCache::default();
    chebyshev_cr_banks_forward_into(
        &banks,
        &x,
        shape.0,
        shape.1,
        (grid, k),
        CatmullRomBanksBuffers {
            outs: outs.iter_mut().map(Vec::as_mut_slice).collect(),
            cache: &mut banks_cache,
        },
    );
    let want: Vec<f32> = route
        .iter()
        .enumerate()
        .map(|(e, &r)| outs.get(usize::from(r)).map_or(0.0, |o| o[e]))
        .collect();
    assert_eq!(bits(&out), bits(&want), "the routed output");
    let cotangents = masked(&grad_y, &route);
    let cotangents: Vec<&[f32]> = cotangents.iter().map(Vec::as_slice).collect();
    let mut want_x = vec![0.0f32; x.len()];
    let want = chebyshev_cr_banks_backward_into(
        &control_banks,
        &basis,
        &banks_cache,
        &cotangents,
        k,
        &mut want_x,
    );
    for (b, (got, want)) in grads.iter().zip(&want).enumerate() {
        assert_equal(&format!("bank {b}'s terms"), got, want);
    }
    assert_equal("the input gradient", &grad_x, &want_x);
}

/// **Routed calls are the same on any thread count.**
#[cfg(feature = "parallel")]
#[test]
fn routed_calls_are_the_same_on_any_thread_count() {
    let knots = Knots::uniform(6);
    let shape = (9001, 5);
    let (coefs, x, grad_ys) = fixture(shape, knots.len(), 950);
    let route = routes(x.len(), 951);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool")
            .install(|| routed(&coefs, (&x, &route, &grad_ys[0]), shape, &knots))
    };
    let (one, four) = (run(1), run(4));
    assert_eq!(bits(&one.0), bits(&four.0), "the output");
    for (b, (one, four)) in one.1.iter().zip(&four.1).enumerate() {
        assert_eq!(bits(one), bits(four), "bank {b}'s gradient");
    }
    assert_eq!(bits(&one.2), bits(&four.2), "the input gradient");
}
