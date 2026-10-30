//! The closed form of the uniform, chord-tangent Chebyshev--CR spline (Nagare plan 2026-10-30-t2-algebraic-lowering)
//! against an `f64` Catmull--Rom reference written from the definition: control points `q_j = Σ θ_i T_i(ξ_j)`, the
//! clamp, leaf and coordinate, the uniform CR weights over the replicated ends, slopes `(g − 1)/2 · Σ w'_m q` on the
//! closed interval and zero outside, `∂φ/∂θ_i` the reference at `θ = e_i`. Every grid 3..12, every term count 1..3,
//! every leaf (interior and both boundary leaves), random, degenerate, signed and large terms; then the qualification,
//! the term planes and the prepared tables.

use hymeko_spline::catmull_rom::PreparedSpline;
use hymeko_spline::closed_form::{ChebyshevClosedForm, ClosedForm, TermPlanes, TERMS};
use hymeko_spline::knots::{Knots, Tangents};

/// A small deterministic generator (splitmix64), uniform in `[-1, 1)`.
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn chebyshev(i: usize, x: f64) -> f64 {
    [1.0, x, 2.0 * x * x - 1.0][i]
}

/// `(φ, φ')` of terms `theta` on `g` uniform knots at `x`, in `f64`, from the definition.
fn reference(theta: &[f64], g: usize, x: f64) -> (f64, f64) {
    let q: Vec<f64> = (0..g)
        .map(|j| {
            let xi = -1.0 + 2.0 * j as f64 / (g - 1) as f64;
            theta
                .iter()
                .enumerate()
                .map(|(i, &t)| t * chebyshev(i, xi))
                .sum()
        })
        .collect();
    let xc = x.clamp(-1.0, 1.0);
    let u = (xc + 1.0) * 0.5 * (g - 1) as f64;
    let leaf = (u.floor() as usize).min(g - 2);
    let t = u - leaf as f64;
    let w = [
        0.5 * (-t * t * t + 2.0 * t * t - t),
        0.5 * (3.0 * t * t * t - 5.0 * t * t + 2.0),
        0.5 * (-3.0 * t * t * t + 4.0 * t * t + t),
        0.5 * (t * t * t - t * t),
    ];
    let dw = [
        0.5 * (-3.0 * t * t + 4.0 * t - 1.0),
        0.5 * (9.0 * t * t - 10.0 * t),
        0.5 * (-9.0 * t * t + 8.0 * t + 1.0),
        0.5 * (3.0 * t * t - 2.0 * t),
    ];
    let at = |m: usize| q[(leaf + m).saturating_sub(1).min(g - 1)];
    let value = (0..4).map(|m| w[m] * at(m)).sum();
    let slope = if (-1.0..=1.0).contains(&x) {
        0.5 * (g - 1) as f64 * (0..4).map(|m| dw[m] * at(m)).sum::<f64>()
    } else {
        0.0
    };
    (value, slope)
}

/// Inputs on every leaf of `g` knots (its ends and interior points), the clamp's edges, outside, signed zeros and
/// subnormals.
fn inputs(g: usize) -> Vec<f32> {
    let h = 2.0 / (g - 1) as f32;
    let mut xs: Vec<f32> = (0..g - 1)
        .flat_map(|leaf| {
            [0.0f32, 1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6].map(|t| -1.0 + h * (leaf as f32 + t))
        })
        .collect();
    let tiny = f32::from_bits(1);
    xs.extend([
        1.0,
        -1.0,
        1.0 - f32::EPSILON,
        -1.0 + f32::EPSILON,
        1.5,
        -1.5,
        0.0,
        -0.0,
        tiny,
        -tiny,
    ]);
    xs
}

/// The spline of `theta` by the closed form, `(φ, φ')`, in `f32`.
fn closed(form: &ChebyshevClosedForm, theta: &[f32; TERMS], x: f32) -> (f32, f32) {
    let [p1, p2, dp1, dp2] = form.basis_slopes(x);
    let value = theta[2].mul_add(p2, theta[1].mul_add(p1, theta[0]));
    let slope = if (-1.0..=1.0).contains(&x) {
        theta[2].mul_add(dp2, theta[1] * dp1)
    } else {
        0.0
    };
    (value, slope)
}

/// The `f32` evaluation budget of `Σ θ_i P_i` (|P_i| ≤ 1.3): 16 ε Σ|θ_i| for values, eight times that for slopes.
fn budget(theta: &[f32; TERMS]) -> f64 {
    16.0 * f64::from(f32::EPSILON)
        * theta
            .iter()
            .map(|t| f64::from(t.abs()))
            .sum::<f64>()
            .max(1e-30)
}

fn check(theta: [f32; TERMS], k: usize, g: usize) {
    let form = ChebyshevClosedForm::of(&Knots::uniform(g), k).expect("uniform chord knots");
    let th: Vec<f64> = theta.iter().take(k).map(|&v| f64::from(v)).collect();
    for x in inputs(g) {
        let (value, slope) = closed(&form, &theta, x);
        let (want, want_slope) = reference(&th, g, f64::from(x));
        assert!(
            (f64::from(value) - want).abs() <= budget(&theta),
            "value g {g} k {k} theta {theta:?} at {x}: {value} vs {want}"
        );
        assert!(
            (f64::from(slope) - want_slope).abs() <= 8.0 * budget(&theta) * (g as f64),
            "slope g {g} k {k} theta {theta:?} at {x}: {slope} vs {want_slope}"
        );
        let (p1, p2) = form.basis(x);
        let [q1, q2, _, _] = form.basis_slopes(x);
        assert_eq!(
            (p1.to_bits(), p2.to_bits()),
            (q1.to_bits(), q2.to_bits()),
            "one basis"
        );
        for (i, p) in [1.0, p1, p2].into_iter().enumerate().take(k) {
            let mut e = vec![0.0; k];
            e[i] = 1.0;
            let (want_p, _) = reference(&e, g, f64::from(x));
            assert!(
                (f64::from(p) - want_p).abs() <= 8.0 * f64::from(f32::EPSILON),
                "P_{i} g {g} at {x}"
            );
        }
    }
}

#[test]
fn the_closed_form_is_the_spline_on_every_leaf_of_every_grid() {
    let mut draw = Draw(301030);
    for g in 3..=12 {
        for k in 1..=TERMS {
            for _ in 0..24 {
                let mut theta = [draw.next(), draw.next(), draw.next()];
                theta[k..].fill(0.0);
                check(theta, k, g);
            }
        }
    }
}

#[test]
fn degenerate_signed_and_large_terms_keep_the_identity() {
    let cases: [[f32; 3]; 9] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.5, 0.54, 0.0],
        [-3.0, -2.0, -1.0],
        [1e3, -1e3, 5e2],
        [1e-30, -1e-30, 1e-30],
        [0.5, -0.25, 2.0],
    ];
    for g in [3, 4, 6, 9, 12] {
        for theta in cases {
            check(theta, 3, g);
        }
    }
}

#[test]
fn a_nan_input_stays_nan_and_its_slope_is_zero() {
    let form = ChebyshevClosedForm::of(&Knots::uniform(6), 3).expect("uniform chord knots");
    let (p1, p2) = form.basis(f32::NAN);
    assert!(p1.is_nan() && p2.is_nan());
    let (value, slope) = closed(&form, &[0.1, 0.2, 0.3], f32::NAN);
    assert!(value.is_nan());
    assert_eq!(slope, 0.0, "outside the closed interval, as the tables");
}

#[test]
fn the_closed_form_is_refused_where_it_does_not_hold() {
    assert!(
        ChebyshevClosedForm::of(&Knots::uniform(6), 4).is_none(),
        "four terms: a cubic is not reproduced"
    );
    assert!(
        ChebyshevClosedForm::of(&Knots::uniform(6), 0).is_none(),
        "no term"
    );
    let bounded = Knots::uniform(6).with_tangents(Tangents::Bounded);
    assert!(
        ChebyshevClosedForm::of(&bounded, 3).is_none(),
        "bounded tangents select per knot"
    );
    let refined = Knots::refined(6, &[0, 1, 0, 0, 0]);
    assert!(
        ChebyshevClosedForm::of(&refined, 3).is_none(),
        "refined knots are not uniform"
    );
    for g in 3..=12 {
        assert!(
            ChebyshevClosedForm::of(&Knots::uniform(g), 3).is_some(),
            "uniform grid {g}"
        );
    }
}

#[test]
fn term_planes_read_the_parameter_layout_and_map_gradients_back() {
    let (channels, k) = (5, 2);
    let a: Vec<f32> = (0..channels * k).map(|v| v as f32).collect();
    let b: Vec<f32> = (0..channels * k).map(|v| -(v as f32)).collect();
    let planes = TermPlanes::new(&[&a, &b], channels, k);
    assert_eq!(planes.len(), 2 * TERMS * channels);
    let [t0, t1, t2] = planes.bank(1);
    for c in 0..channels {
        assert_eq!(
            (t0[c], t1[c], t2[c]),
            (b[c * k], b[c * k + 1], 0.0),
            "bank 1 channel {c}"
        );
    }
    // Two blocks of accumulators equal to the planes: the gradient is twice the parameters, in their layout.
    let acc = planes.bank(0).concat();
    let mut block = acc.clone();
    block.extend(planes.bank(1).concat());
    let grad = planes.terms_gradient(&[&block, &block], 1);
    assert_eq!(grad, b.iter().map(|v| 2.0 * v).collect::<Vec<_>>());
    assert_eq!(planes.terms_gradient(&[&block], 0), a);
}

#[test]
fn prepared_chebyshev_tables_carry_their_closed_form_and_agree_with_it() {
    let (channels, grid) = (7, 6);
    let mut draw = Draw(7);
    let banks: Vec<Vec<f32>> = (0..3)
        .map(|_| (0..channels * 3).map(|_| draw.next()).collect())
        .collect();
    let refs: Vec<&[f32]> = banks.iter().map(Vec::as_slice).collect();
    let prepared = PreparedSpline::chebyshev(&refs, channels, (grid, 3));
    let closed: &ClosedForm = prepared
        .closed_form()
        .expect("uniform chord knots, three terms");
    assert_eq!(closed.planes().banks(), 3);
    let leaves = prepared.leaves();
    let (mut leaf, mut t, mut inside) = (
        vec![0u32; channels],
        vec![0.0f32; channels],
        vec![false; channels],
    );
    for x in inputs(grid) {
        let row = vec![x; channels];
        prepared.locate_row(&row, (&mut leaf, &mut t, &mut inside));
        let (p1, p2) = closed.form().basis(x);
        for bank in 0..3 {
            let [t0, t1, t2] = closed.planes().bank(bank);
            for c in 0..channels {
                let [a0, a1, a2, a3] = prepared.power_table(bank)[c * leaves + leaf[c] as usize];
                let table = a0 + t[c] * (a1 + t[c] * (a2 + t[c] * a3));
                let form = t2[c].mul_add(p2, t1[c].mul_add(p1, t0[c]));
                let theta = [t0[c], t1[c], t2[c]];
                // The tables round the control points to f32 before forming their coefficients: one more rounding.
                assert!(
                    (f64::from(table) - f64::from(form)).abs() <= 4.0 * budget(&theta),
                    "bank {bank} channel {c} at {x}: tables {table} vs closed form {form}"
                );
            }
        }
    }
    let four = PreparedSpline::chebyshev(&[&vec![0.1; channels * 4]], channels, (grid, 4));
    assert!(
        four.closed_form().is_none(),
        "four terms keep the tables only"
    );
    let plain = PreparedSpline::new(&vec![0.1; channels * grid], channels, &Knots::uniform(grid));
    assert!(
        plain.closed_form().is_none(),
        "control points without terms have no closed form here"
    );
}
