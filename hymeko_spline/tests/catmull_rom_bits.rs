//! The Catmull--Rom kernels' bits, pinned before the segment-table rewrite (Nagare plan
//! 2026-10-08-orthtree-hypergraph-convolution, Amendment 1).
//!
//! The fingerprints were recorded on 2026-10-09 from `main` at `87116b6`, before either kernel was
//! touched. They cover the output, the cache, the input gradient and the coefficient gradient, over
//! shapes on both sides of the parallel threshold, three grids, and one and four threads. A rewrite
//! that keeps every expression and every accumulation order keeps them all.
//!
//! **Re-pinned on 2026-10-09, coefficient gradient only** (user decision: "AVX2 és CR
//! újrapinnelés"). The pullback's coefficient gradient now sums each block's elements per segment
//! and control slot, then folds those sums into the coefficients, so the loops vectorise; a
//! coefficient's terms are added in a different order and the last bits move. The output, the cache
//! and the input gradient keep the `87116b6` fingerprints. The one-element shape keeps all five,
//! since a single term has no order.
//!
//! **Re-pinned on 2026-10-23, output, input gradient and coefficient gradient** (user decision:
//! "Do it"; Nagare plan 2026-10-23-incidence-partition, Amendment 7): the power form evaluates each
//! leaf's cubic by Horner from coefficients computed once per call, and accumulates moments folded
//! through the leaf basis; the cache keeps its fingerprints. The one-element shape moves too: its
//! single term is now a different expression, not a different order.
//!
//! No retired body is copied into the test tree: the oracle is the recorded bytes.

use hymeko_spline::{catmull_rom_backward, catmull_rom_forward, PAR_MIN_ELEMS};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % 3001) as f32 / 1000.0 - 1.5
    }
}

/// Values in `[-1.5, 1.5]`: a third outside the clamp interval, and `±1` and the knots reachable.
fn field(len: usize, seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..len).map(|_| next()).collect()
}

/// FNV-1a over the bit patterns.
fn fingerprint(values: impl IntoIterator<Item = u32>) -> u64 {
    values.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        v.to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

/// `(n, channels, grid)`, the last two past the parallel threshold.
const SHAPES: [(usize, usize, usize); 5] = [
    (1, 1, 3),
    (37, 7, 6),
    (200, 32, 8),
    (4096, 8, 6),
    (2003, 32, 6),
];

/// The fingerprints of `(out, x_clamped, interior, grad_x, grad_coef)` per shape: the cache's two
/// recorded at `87116b6`, the other three on 2026-10-23 (the power form).
const RECORDED: [[u64; 5]; 5] = [
    [
        0xe828_5c3a_1b23_0d5a,
        0xd9cc_4926_3773_0386,
        0xad2a_ca77_4798_5764,
        0xb659_0c7f_7bbe_1975,
        0x4f26_ee56_5f86_74cc,
    ],
    [
        0xff32_3701_21a8_37e0,
        0x0928_766a_74cc_3a07,
        0xdd19_2b39_5c64_5995,
        0x11dc_3103_0722_2fe1,
        0x7fb9_dda2_544a_d83b,
    ],
    [
        0xfd37_93ed_6fc2_7b62,
        0xd7b7_fbf8_935a_f5ae,
        0xcee0_b8a0_9342_0275,
        0xd756_fc96_0df9_4b00,
        0xa322_b9d5_7820_aeac,
    ],
    [
        0x2706_8754_e128_4475,
        0x807b_f26a_d1c6_3fcb,
        0x51bd_a1e5_a365_4e25,
        0xc1dc_65f2_d990_8004,
        0x20a6_8a10_c6b5_e2f1,
    ],
    [
        0x9489_d4fb_a588_9488,
        0x360c_bac4_abe7_0a57,
        0xd74c_ba22_0e56_8c55,
        0xb4a7_de24_29f0_2d43,
        0x1013_3575_bb6b_f7b6,
    ],
];

fn run(n: usize, channels: usize, grid: usize, seed: u64) -> [u64; 5] {
    let (x, coef) = (field(n * channels, seed), field(channels * grid, 10 + seed));
    let grad_y = field(n * channels, 20 + seed);
    let (out, cache) =
        catmull_rom_forward(&coef, &x, n, channels, &hymeko_spline::Knots::uniform(grid));
    let back = catmull_rom_backward(&coef, &cache, &grad_y);
    let bits = |v: &[f32]| fingerprint(v.iter().map(|x| x.to_bits()));
    [
        bits(&out),
        bits(&cache.x_clamped),
        fingerprint(cache.interior.iter().map(|&b| u32::from(b))),
        bits(&back.grad_x),
        bits(&back.grad_coef),
    ]
}

fn measured() -> Vec<[u64; 5]> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(seed, &(n, channels, grid))| run(n, channels, grid, seed as u64))
        .collect()
}

#[test]
fn the_catmull_rom_kernels_keep_their_recorded_bits() {
    const {
        assert!(2003 * 32 >= PAR_MIN_ELEMS && 4096 * 8 >= PAR_MIN_ELEMS);
    }
    let sequential = measured();
    assert_eq!(sequential, RECORDED, "{sequential:#x?}");
}

/// With the `parallel` feature the large shapes take the rayon path, whose bits must be the same at
/// one and at four threads.
#[cfg(feature = "parallel")]
#[test]
fn the_parallel_path_keeps_the_recorded_bits() {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool");
        let parallel = pool.install(measured);
        assert_eq!(parallel, RECORDED, "{threads} threads: {parallel:#x?}");
    }
}
