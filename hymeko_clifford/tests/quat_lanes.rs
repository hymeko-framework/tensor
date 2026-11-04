//! **Quaternion lanes** (plan nagare 2026-11-03-monogenic-cells, M1): the lane-wise product is the by-value Hamilton
//! product per lane, under each conjugation, and the squared norm is multiplicative (the algebra's identity).

use hymeko_clifford::{quat_conjugate, quat_mul, quat_mul_lanes, quat_norm_sq_lanes};

/// Lanes past one vector width and not a multiple of it, so both the vector body and the remainder are read.
const LANES: usize = 37;

/// A deterministic finite quaternion: ordinary magnitudes with signed zeros and ±1 mixed in.
fn sample(i: usize) -> [f32; 4] {
    const SPECIAL: [f32; 4] = [0.0, -0.0, 1.0, -1.0];
    let mut s = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    std::array::from_fn(|_| {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (s >> 40) as u32;
        if u.is_multiple_of(7) {
            SPECIAL[(u as usize / 7) % 4]
        } else {
            (f64::from(u % 20_001) / 1000.0 - 10.0) as f32
        }
    })
}

/// The four planes borrowed.
fn refs(p: &[Vec<f32>; 4]) -> [&[f32]; 4] {
    [&p[0], &p[1], &p[2], &p[3]]
}

/// `quats` as four planes.
fn planes(quats: &[[f32; 4]]) -> [Vec<f32>; 4] {
    std::array::from_fn(|c| quats.iter().map(|q| q[c]).collect())
}

fn product_lanes<const CA: bool, const CB: bool>(
    a: &[Vec<f32>; 4],
    b: &[Vec<f32>; 4],
) -> [Vec<f32>; 4] {
    let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| vec![f32::NAN; LANES]);
    let [w, x, y, z] = &mut out;
    quat_mul_lanes::<CA, CB>(refs(a), refs(b), w, x, y, z);
    out
}

fn check<const CA: bool, const CB: bool>(a: &[[f32; 4]], b: &[[f32; 4]]) {
    let out = product_lanes::<CA, CB>(&planes(a), &planes(b));
    for i in 0..LANES {
        let p = if CA { quat_conjugate(a[i]) } else { a[i] };
        let q = if CB { quat_conjugate(b[i]) } else { b[i] };
        let want = quat_mul(p, q);
        for c in 0..4 {
            assert_eq!(
                out[c][i].to_bits(),
                want[c].to_bits(),
                "lane {i} component {c}, conj ({CA}, {CB})"
            );
        }
    }
}

#[test]
fn each_lane_is_the_by_value_product_under_each_conjugation() {
    let a: Vec<[f32; 4]> = (0..LANES).map(sample).collect();
    let b: Vec<[f32; 4]> = (0..LANES).map(|i| sample(i + 1000)).collect();
    check::<false, false>(&a, &b);
    check::<true, false>(&a, &b);
    check::<false, true>(&a, &b);
    check::<true, true>(&a, &b);
}

#[test]
fn the_squared_norm_is_multiplicative() {
    let (a, b): (Vec<[f32; 4]>, Vec<[f32; 4]>) = (0..LANES)
        .map(|i| (sample(i + 7), sample(i + 2000)))
        .unzip();
    let (pa, pb) = (planes(&a), planes(&b));
    let ab = product_lanes::<false, false>(&pa, &pb);
    let norms = |p: &[Vec<f32>; 4]| {
        let mut out = vec![0.0f32; LANES];
        quat_norm_sq_lanes([&p[0], &p[1], &p[2], &p[3]], &mut out);
        out
    };
    let (na, nb, nab) = (norms(&pa), norms(&pb), norms(&ab));
    for i in 0..LANES {
        let want = f64::from(na[i]) * f64::from(nb[i]);
        let err = (f64::from(nab[i]) - want).abs();
        assert!(
            err <= 1e-5 * want.max(f64::MIN_POSITIVE),
            "lane {i}: |ab|^2 {} vs |a|^2|b|^2 {want}",
            nab[i]
        );
    }
}
