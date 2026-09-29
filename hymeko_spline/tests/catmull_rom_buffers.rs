//! The Catmull--Rom kernels into caller-owned buffers: the same bits as the allocating forms,
//! whatever the buffers held before, and one set of buffers reusable across shapes.

use hymeko_spline::{
    catmull_rom_backward, catmull_rom_backward_into, catmull_rom_forward, catmull_rom_forward_into,
    chebyshev_cr_backward, chebyshev_cr_backward_into, chebyshev_cr_forward,
    chebyshev_cr_forward_into, CatmullRomBuffers, CatmullRomCache, PAR_MIN_ELEMS,
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

/// Values in `[-1.5, 1.5]`: a third of the inputs lie outside the clamp interval, where the input
/// gradient is the clamp's zero and a stale buffer would show.
fn field(len: usize, seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..len).map(|_| next()).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A cache left by a forward of a different shape: every buffer it holds is wrong for the next
/// call.
fn dirty_cache() -> CatmullRomCache {
    let (x, coef) = (field(333 * 5, 90), field(5 * 4, 91));
    catmull_rom_forward(&coef, &x, 333, 5, 4).1
}

/// Shapes on both sides of the parallel threshold.
const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 3), (37, 3, 5), (4096, 8, 6), (9001, 8, 3)];

#[test]
fn the_into_forms_are_the_allocating_forms_bit_for_bit() {
    const {
        assert!(
            9001 * 8 >= PAR_MIN_ELEMS,
            "the largest shape takes the parallel path"
        );
    }
    for (seed, &(n, channels, grid)) in SHAPES.iter().enumerate() {
        let seed = seed as u64;
        let (x, coef) = (field(n * channels, seed), field(channels * grid, 10 + seed));
        let grad_y = field(n * channels, 20 + seed);
        let (want_y, want_cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
        let want_back = catmull_rom_backward(&coef, &want_cache, &grad_y);

        let mut out = vec![f32::NAN; n * channels];
        let mut cache = dirty_cache();
        let into = CatmullRomBuffers {
            out: &mut out,
            cache: &mut cache,
        };
        catmull_rom_forward_into(&coef, &x, n, channels, grid, into);
        assert_eq!(
            bits(&out),
            bits(&want_y),
            "forward, shape {n}x{channels} grid {grid}"
        );
        assert_eq!(bits(&cache.x_clamped), bits(&want_cache.x_clamped));
        assert_eq!(cache.interior, want_cache.interior);
        assert_eq!(
            (cache.n, cache.channels, cache.grid),
            (want_cache.n, want_cache.channels, want_cache.grid)
        );

        let mut grad_x = vec![f32::NAN; n * channels];
        let grad_coef = catmull_rom_backward_into(&coef, &cache, &grad_y, &mut grad_x);
        assert_eq!(
            bits(&grad_x),
            bits(&want_back.grad_x),
            "backward, shape {n}x{channels}"
        );
        assert_eq!(bits(&grad_coef), bits(&want_back.grad_coef));
    }
}

#[test]
fn the_chebyshev_into_forms_are_the_allocating_forms_bit_for_bit() {
    for (seed, &(n, channels, grid)) in SHAPES.iter().enumerate() {
        let (seed, k) = (seed as u64, 4);
        let (x, coef) = (
            field(n * channels, 30 + seed),
            field(channels * k, 40 + seed),
        );
        let grad_y = field(n * channels, 50 + seed);
        let (want_y, want_cache, want_control, want_basis) =
            chebyshev_cr_forward(&coef, &x, n, channels, grid, k);
        let want_back = chebyshev_cr_backward(&want_control, &want_basis, &want_cache, &grad_y, k);

        let mut y = vec![f32::NAN; n * channels];
        let mut cache = dirty_cache();
        let into = CatmullRomBuffers {
            out: &mut y,
            cache: &mut cache,
        };
        let (control, basis) = chebyshev_cr_forward_into(&coef, &x, n, channels, grid, k, into);
        assert_eq!(bits(&y), bits(&want_y));
        assert_eq!(bits(&control), bits(&want_control));
        assert_eq!(bits(&basis), bits(&want_basis));

        let mut grad_x = vec![f32::NAN; n * channels];
        let grad_coef =
            chebyshev_cr_backward_into(&control, &basis, &cache, &grad_y, k, &mut grad_x);
        assert_eq!(bits(&grad_x), bits(&want_back.grad_x));
        assert_eq!(bits(&grad_coef), bits(&want_back.grad_coef));
    }
}

/// One output, one gradient buffer and one cache serve a large, a small and a large shape again,
/// as a training loop's would; each call gives the allocating form's bits.
#[test]
fn one_set_of_buffers_serves_three_shapes() {
    let mut cache = CatmullRomCache::default();
    let (mut out, mut grad_x) = (Vec::new(), Vec::new());
    for (seed, &(n, channels, grid)) in [(9001, 8, 3), (37, 3, 5), (9001, 8, 3)].iter().enumerate()
    {
        let seed = 60 + seed as u64;
        let (x, coef) = (field(n * channels, seed), field(channels * grid, seed + 1));
        let grad_y = field(n * channels, seed + 2);
        out.resize(n * channels, f32::NAN);
        grad_x.resize(n * channels, f32::NAN);
        let into = CatmullRomBuffers {
            out: &mut out,
            cache: &mut cache,
        };
        catmull_rom_forward_into(&coef, &x, n, channels, grid, into);
        let grad_coef = catmull_rom_backward_into(&coef, &cache, &grad_y, &mut grad_x);

        let (want_y, want_cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
        let want_back = catmull_rom_backward(&coef, &want_cache, &grad_y);
        assert_eq!(bits(&out), bits(&want_y));
        assert_eq!(bits(&grad_x), bits(&want_back.grad_x));
        assert_eq!(bits(&grad_coef), bits(&want_back.grad_coef));
    }
}

#[test]
#[should_panic(expected = "out must be (n, channels)")]
fn an_output_of_the_wrong_shape_is_refused() {
    let (x, coef) = (field(10 * 2, 1), field(2 * 4, 2));
    let mut out = vec![0.0f32; 10 * 2 - 1];
    let mut cache = CatmullRomCache::default();
    let into = CatmullRomBuffers {
        out: &mut out,
        cache: &mut cache,
    };
    catmull_rom_forward_into(&coef, &x, 10, 2, 4, into);
}
