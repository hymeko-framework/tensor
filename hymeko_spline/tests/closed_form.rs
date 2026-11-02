//! The closed form of the uniform, chord-tangent Chebyshev--CR spline (Nagare plan 2026-10-30-t2-algebraic-lowering)
//! against an `f64` Catmull--Rom reference written from the definition: control points `q_j = Σ θ_i T_i(ξ_j)`, the
//! clamp, leaf and coordinate, the uniform CR weights over the replicated ends, slopes `(g − 1)/2 · Σ w'_m q` on the
//! closed interval and zero outside, `∂φ/∂θ_i` the reference at `θ = e_i`. Every grid 3..12, every term count 1..3,
//! every leaf (interior and both boundary leaves), random, degenerate, signed and large terms; then the qualification,
//! the term planes and the prepared tables. Under bounded tangents (Nagare plan 2026-11-02-bounded-closed-form) the same
//! against an `f64` reference written from the bounded rule's definition, with the parabola's vertex placed on and
//! around every interior knot; and the Chebyshev--CR tables under bounded tangents (banks, prepared and routed, forward
//! and pullback) against the bounded closed form.

use hymeko_spline::catmull_rom::{
    chebyshev_cr_banks_backward_into, chebyshev_cr_banks_forward_into,
    chebyshev_cr_routed_backward_into, chebyshev_cr_routed_forward_into, CatmullRomBanksBuffers,
    CatmullRomCache, PreparedSpline, RoutedBuffers, RoutedCotangents,
};
use hymeko_spline::closed_form::{
    BoundedClosedForm, ChebyshevClosedForm, ClosedForm, KnotDefect, TermPlanes, TERMS,
};
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

/// A knot's choice under the bounded rule.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Choice {
    Zero,
    Chord,
    Left,
    Right,
}

/// The bounded rule from its definition, per knot of control points `q` on spacing `h`: zero where the two neighbouring
/// secants do not agree in sign; otherwise the chord, unless three times a secant is smaller in magnitude, then that
/// (the left one on a tie). An end knot reads its one secant on both sides.
fn choices(q: &[f64], h: f64) -> Vec<Choice> {
    let g = q.len();
    (0..g)
        .map(|k| {
            let (l, r) = (k.max(1) - 1, k.min(g - 2));
            let (sl, sr) = ((q[l + 1] - q[l]) / h, (q[r + 1] - q[r]) / h);
            let chord = (q[(k + 1).min(g - 1)] - q[k.saturating_sub(1)]) / (2.0 * h);
            let (c, a, b) = (chord.abs(), 3.0 * sl.abs(), 3.0 * sr.abs());
            if sl * sr <= 0.0 {
                Choice::Zero
            } else if c <= a && c <= b {
                Choice::Chord
            } else if a <= b {
                Choice::Left
            } else {
                Choice::Right
            }
        })
        .collect()
}

/// Knot `k`'s tangent, in `x` units, of control points `q` under `choice`: a linear functional of `q`.
fn tangent(choice: Choice, q: &[f64], h: f64, k: usize) -> f64 {
    let g = q.len();
    let (l, r) = (k.max(1) - 1, k.min(g - 2));
    match choice {
        Choice::Zero => 0.0,
        Choice::Chord => (q[(k + 1).min(g - 1)] - q[k.saturating_sub(1)]) / (2.0 * h),
        Choice::Left => 3.0 * (q[l + 1] - q[l]) / h,
        Choice::Right => 3.0 * (q[r + 1] - q[r]) / h,
    }
}

/// `(φ, φ')` of control points `q` on uniform knots under fixed `choices`: the cubic Hermite leaf, its slope on the
/// closed interval and zero outside.
fn hermite(q: &[f64], picks: &[Choice], x: f64) -> (f64, f64) {
    let g = q.len();
    let h = 2.0 / (g - 1) as f64;
    let xc = x.clamp(-1.0, 1.0);
    let u = (xc + 1.0) / h;
    let leaf = (u.floor() as usize).min(g - 2);
    let t = u - leaf as f64;
    let (m0, m1) = (
        tangent(picks[leaf], q, h, leaf),
        tangent(picks[leaf + 1], q, h, leaf + 1),
    );
    let (q0, q1) = (q[leaf], q[leaf + 1]);
    let value = q0 * (2.0 * t * t * t - 3.0 * t * t + 1.0)
        + h * m0 * t * (1.0 - t) * (1.0 - t)
        + q1 * (-2.0 * t * t * t + 3.0 * t * t)
        - h * m1 * t * t * (1.0 - t);
    let slope = (q0 * (6.0 * t * t - 6.0 * t)
        + h * m0 * (3.0 * t * t - 4.0 * t + 1.0)
        + q1 * (-6.0 * t * t + 6.0 * t)
        + h * m1 * (3.0 * t * t - 2.0 * t))
        / h;
    (
        value,
        if (-1.0..=1.0).contains(&x) {
            slope
        } else {
            0.0
        },
    )
}

/// Whether control points `q` lie on a border of the rule's cells, within `1e-9` relative (the `f64` secants'
/// rounding): a zero secant product, or where the secants agree the chord equal to a candidate, or the two candidates
/// equal where one is taken. The spline has a kink in its control points there, so it has no gradient to compare.
fn on_border(q: &[f64], h: f64) -> bool {
    let g = q.len();
    let equal = |p: f64, r: f64| (p - r).abs() <= 1e-9 * (p.abs() + r.abs());
    (0..g).any(|k| {
        let (l, r) = (k.max(1) - 1, k.min(g - 2));
        let (sl, sr) = ((q[l + 1] - q[l]) / h, (q[r + 1] - q[r]) / h);
        let chord = (q[(k + 1).min(g - 1)] - q[k.saturating_sub(1)]) / (2.0 * h);
        let (c, a, b) = (chord.abs(), 3.0 * sl.abs(), 3.0 * sr.abs());
        let agree = sl * sr > 0.0;
        (sl * sr).abs() <= 1e-9 * (sl * sl + sr * sr)
            || (agree && (equal(c, a) || equal(c, b)))
            || (agree && !(c <= a && c <= b) && equal(a, b))
    })
}

/// Whether the vertex of terms `theta` lies within `1e-4 h` of a cell border: `|ξ_k − x*|` at `h/2` or `3h/4` for an
/// interior knot (a zero secant or the chord against three times a secant; the end secants' zero is the first's), or
/// a constant spline.
fn near_border(theta: &[f32; TERMS], g: usize) -> bool {
    let (t1, t2) = (f64::from(theta[1]), f64::from(theta[2]));
    if t2 == 0.0 {
        return t1 == 0.0;
    }
    let h = 2.0 / (g - 1) as f64;
    let vertex = -t1 / (4.0 * t2);
    (1..g - 1).any(|k| {
        let d = ((-1.0 + h * k as f64) - vertex).abs() / h;
        (d - 0.5).abs() < 1e-4 || (d - 0.75).abs() < 1e-4
    })
}

/// `(φ, φ', ∂φ/∂θ, on a border)` of terms `theta` on `g` uniform knots under bounded tangents, in `f64`: the choices
/// of the terms' control points; the gradient the same choices' leaf over each `T_i`'s control points (the rule is
/// linear in the control points once its choices are fixed).
fn bounded_reference(theta: &[f64; TERMS], g: usize, x: f64) -> (f64, f64, [f64; TERMS], bool) {
    let h = 2.0 / (g - 1) as f64;
    let image = |w: &[f64; TERMS]| -> Vec<f64> {
        (0..g)
            .map(|j| {
                let xi = -1.0 + h * j as f64;
                (0..TERMS).map(|i| w[i] * chebyshev(i, xi)).sum()
            })
            .collect()
    };
    let q = image(theta);
    let picks = choices(&q, h);
    let (value, slope) = hermite(&q, &picks, x);
    let grad = std::array::from_fn(|i| {
        let mut e = [0.0; TERMS];
        e[i] = 1.0;
        hermite(&image(&e), &picks, x).0
    });
    (value, slope, grad, on_border(&q, h))
}

/// `(φ, φ', ∂φ/∂θ)` by the bounded closed form, in `f32`.
fn bounded_closed(
    form: &BoundedClosedForm,
    defects: &[KnotDefect; 2],
    theta: &[f32; TERMS],
    x: f32,
) -> (f32, f32, [f32; TERMS]) {
    let [p1, p2, dp1, dp2] = form.form().basis_slopes(x);
    let [c, dc, g1, g2] = form.correction(defects, x);
    let value = theta[2].mul_add(p2, theta[1].mul_add(p1, theta[0])) + c;
    let slope = if (-1.0..=1.0).contains(&x) {
        theta[2].mul_add(dp2, theta[1] * dp1) + dc
    } else {
        0.0
    };
    (value, slope, [1.0, p1 + g1, p2 + g2])
}

/// The bounded form's `f32` budget: the chord form's (`|P_i| ≤ 1.3`) with the defects' scale, `|δ| ≲ |θ₁| + 4|θ₂|`,
/// in place of `|θ₂|`, and twice the roundings.
fn bounded_budget(theta: &[f32; TERMS]) -> f64 {
    32.0 * f64::from(f32::EPSILON)
        * (f64::from(theta[0].abs()) + f64::from(theta[1].abs()) + 4.0 * f64::from(theta[2].abs()))
            .max(1e-30)
}

/// The parameter gradient's budget: `P_i` within `8ε`, the corrections' `|∂δ| ≤ 8 + 6h`, `hK ≤ 4h/27`, two defects
/// and a handful of roundings each.
const GRAD_BUDGET: f64 = 64.0 * f32::EPSILON as f64;

/// Terms placing the parabola's vertex at `ξ_k + o h` for every interior knot `k` and offset `o` (zero region, the
/// window's inside and borders, the chord region), with both signs of `θ₂`.
fn vertex_terms(g: usize) -> Vec<[f32; TERMS]> {
    let h = 2.0 / (g - 1) as f64;
    let offsets = [
        0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 0.499, 0.501, 0.749, 0.751, -0.25, -0.5, -0.6, -0.75, -1.0,
        -0.499, -0.501, -0.749, -0.751,
    ];
    (1..g - 1)
        .flat_map(|k| {
            offsets.into_iter().enumerate().map(move |(n, o)| {
                let t2 = if (k + n) % 2 == 0 { 0.43 } else { -0.45 };
                let vertex = -1.0 + h * k as f64 + o * h;
                [0.22, (-4.0 * t2 * vertex) as f32, t2 as f32]
            })
        })
        .collect()
}

/// The bounded closed form against the reference at every input; whether the chord form alone missed it anywhere.
fn check_bounded(theta: [f32; TERMS], k: usize, g: usize) -> bool {
    let knots = Knots::uniform(g).with_tangents(Tangents::Bounded);
    let form = BoundedClosedForm::of(&knots, k).expect("uniform bounded knots");
    let defects = form.defects(theta);
    let th = theta.map(f64::from);
    let mut chord_misses = false;
    for x in inputs(g) {
        let (value, slope, grad) = bounded_closed(&form, &defects, &theta, x);
        let (want, want_slope, want_grad, border) = bounded_reference(&th, g, f64::from(x));
        let tol = bounded_budget(&theta);
        assert!(
            (f64::from(value) - want).abs() <= tol,
            "value g {g} k {k} theta {theta:?} at {x}: {value} vs {want} ({defects:?})"
        );
        assert!(
            (f64::from(slope) - want_slope).abs() <= 8.0 * tol * g as f64,
            "slope g {g} k {k} theta {theta:?} at {x}: {slope} vs {want_slope} ({defects:?})"
        );
        for i in (0..k).filter(|_| !border) {
            assert!(
                (f64::from(grad[i]) - want_grad[i]).abs() <= GRAD_BUDGET,
                "d/dθ_{i} g {g} k {k} theta {theta:?} at {x}: {} vs {} ({defects:?})",
                grad[i],
                want_grad[i]
            );
        }
        let (chord, _) = closed(form.form(), &theta, x);
        chord_misses |= (f64::from(chord) - want).abs() > 1e3 * tol;
    }
    chord_misses
}

#[test]
fn the_bounded_closed_form_is_the_bounded_spline_on_every_leaf_of_every_grid() {
    let mut draw = Draw(110211);
    let mut corrected = 0;
    for g in 3..=12 {
        for k in 1..=TERMS {
            for _ in 0..24 {
                let mut theta = [draw.next(), draw.next(), draw.next()];
                theta[k..].fill(0.0);
                corrected += usize::from(check_bounded(theta, k, g));
            }
        }
        for theta in vertex_terms(g) {
            corrected += usize::from(check_bounded(theta, TERMS, g));
        }
    }
    assert!(
        corrected > 0,
        "the correction is not vacuous: the chord form alone misses the bounded spline"
    );
}

#[test]
fn degenerate_signed_and_large_terms_keep_the_bounded_identity() {
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
            check_bounded(theta, 3, g);
        }
    }
}

/// Two banks of a grid's off-border vertex terms (the tables' rounded control points may take the other side's
/// derivative on a border): the terms, and half their shape plus an offset (the same vertices).
fn bounded_banks(g: usize) -> [Vec<[f32; TERMS]>; 2] {
    let first: Vec<[f32; TERMS]> = vertex_terms(g)
        .into_iter()
        .filter(|theta| !near_border(theta, g))
        .collect();
    let second = first
        .iter()
        .map(|t| [t[0] + 0.1, 0.5 * t[1], 0.5 * t[2]])
        .collect();
    [first, second]
}

/// One bank of terms against a table path over the inputs `xs` (`bank.len()` channels per row): its values where the
/// element's cotangent `weight(e)` is not zero, and its term gradient (`channels × k`) against the closed form's
/// `Σ weight · ∂φ/∂θ`; adds `weight · φ'` to `slopes`, the input gradient the path should have formed.
fn agree(
    form: &BoundedClosedForm,
    bank: &[[f32; TERMS]],
    (xs, g): (&[f32], usize),
    (values, grad): (&[f32], &[f32]),
    weight: impl Fn(usize) -> f64,
    slopes: &mut [f64],
) {
    let channels = bank.len();
    for (c, theta) in bank.iter().enumerate() {
        let defects = form.defects(*theta);
        let tol = 4.0 * bounded_budget(theta);
        let mut want = [0.0f64; TERMS];
        for (row, &xv) in xs.iter().enumerate() {
            let e = row * channels + c;
            let w = weight(e);
            if w == 0.0 {
                continue;
            }
            let (value, slope, d) = bounded_closed(form, &defects, theta, xv);
            assert!(
                (f64::from(values[e]) - f64::from(value)).abs() <= tol,
                "g {g} theta {theta:?} at {xv}: tables {} vs closed form {value}",
                values[e]
            );
            slopes[e] += w * f64::from(slope);
            for (sum, di) in want.iter_mut().zip(d) {
                *sum += w * f64::from(di);
            }
        }
        for (i, &w) in want.iter().enumerate() {
            let got = f64::from(grad[c * TERMS + i]);
            assert!(
                (got - w).abs() <= xs.len() as f64 * 2.0 * GRAD_BUDGET,
                "g {g} theta {theta:?}: d/dθ_{i} tables {got} vs closed form {w}"
            );
        }
    }
}

/// The input gradient a path formed against the closed form's, per element within both banks' slope budgets.
fn agree_slopes(banks: &[Vec<[f32; TERMS]>; 2], g: usize, grad_x: &[f32], slopes: &[f64]) {
    let channels = banks[0].len();
    for (e, (&got, &want)) in grad_x.iter().zip(slopes).enumerate() {
        let c = e % channels;
        let tol = 32.0 * g as f64 * (bounded_budget(&banks[0][c]) + bounded_budget(&banks[1][c]));
        assert!(
            (f64::from(got) - want).abs() <= tol,
            "g {g} element {e}: tables' input gradient {got} vs closed form {want}"
        );
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn the_bounded_chebyshev_tables_agree_with_the_bounded_closed_form() {
    for g in 3..=12 {
        let grid = (g, TERMS, Tangents::Bounded);
        let knots = Knots::uniform(g).with_tangents(Tangents::Bounded);
        let form = BoundedClosedForm::of(&knots, TERMS).expect("uniform bounded knots");
        let banks = bounded_banks(g);
        let channels = banks[0].len();
        let flat: Vec<Vec<f32>> = banks.iter().map(|bank| bank.concat()).collect();
        let terms: Vec<&[f32]> = flat.iter().map(Vec::as_slice).collect();
        let xs = inputs(g);
        let (n, len) = (xs.len(), xs.len() * channels);
        let x: Vec<f32> = xs
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, channels))
            .collect();
        let (ones, halves) = (vec![1.0f32; len], vec![0.5f32; len]);

        // Banks: every element on both banks, cotangents 1 and 1/2.
        let (mut y0, mut y1) = (vec![0.0; len], vec![0.0; len]);
        let mut cache = CatmullRomCache::default();
        let buffers = CatmullRomBanksBuffers {
            outs: vec![&mut y0, &mut y1],
            cache: &mut cache,
        };
        let (controls, basis) =
            chebyshev_cr_banks_forward_into(&terms, &x, n, channels, grid, buffers);
        let controls: Vec<&[f32]> = controls.iter().map(Vec::as_slice).collect();
        let mut grad_x = vec![0.0; len];
        let cotangents = [ones.as_slice(), halves.as_slice()];
        let grads = chebyshev_cr_banks_backward_into(
            &controls,
            &basis,
            &cache,
            &cotangents,
            TERMS,
            &mut grad_x,
        );
        let mut slopes = vec![0.0; len];
        for (b, values) in [&y0, &y1].into_iter().enumerate() {
            let w = [1.0, 0.5][b];
            agree(
                &form,
                &banks[b],
                (&xs, g),
                (values, &grads[b]),
                |_| w,
                &mut slopes,
            );
        }
        agree_slopes(&banks, g, &grad_x, &slopes);
        // The rule reaches the tables: the chord grid's spline of the same terms differs near the vertices.
        let (mut chord, mut cache_c) = (vec![0.0; len], CatmullRomCache::default());
        let buffers = CatmullRomBanksBuffers {
            outs: vec![&mut chord],
            cache: &mut cache_c,
        };
        chebyshev_cr_banks_forward_into(&terms[..1], &x, n, channels, (g, TERMS), buffers);
        assert!(
            chord.iter().zip(&y0).any(|(c, b)| (c - b).abs() > 1e-3),
            "g {g}: the bounded grid's tables differ from the chord grid's"
        );

        // Prepared tables of the same terms: the same pullback, bit for bit (built as a per-call entry builds them);
        // the chord closed form does not hold.
        let prepared = PreparedSpline::chebyshev(&terms, channels, grid);
        assert!(
            prepared.closed_form().is_none(),
            "bounded knots have no chord closed form"
        );
        let mut grad_xp = vec![0.0; len];
        let grads_p = chebyshev_cr_banks_backward_into(
            &prepared,
            &basis,
            &cache,
            &cotangents,
            TERMS,
            &mut grad_xp,
        );
        assert_eq!(
            bits(&grad_xp),
            bits(&grad_x),
            "g {g}: prepared input gradient"
        );
        for (p, c) in grads_p.iter().zip(&grads) {
            assert_eq!(bits(p), bits(c), "g {g}: prepared term gradient");
        }

        // Routed: element e on bank e mod 2.
        let route: Vec<u8> = (0..len).map(|e| (e % 2) as u8).collect();
        let (mut yr, mut cache_r) = (vec![0.0; len], CatmullRomCache::default());
        let buffers = RoutedBuffers {
            out: &mut yr,
            every: None,
            cache: &mut cache_r,
        };
        let (controls_r, basis_r) =
            chebyshev_cr_routed_forward_into(&terms, &route, &x, (n, channels), grid, buffers);
        let controls_r: Vec<&[f32]> = controls_r.iter().map(Vec::as_slice).collect();
        let mut grad_xr = vec![0.0; len];
        let cotangents = RoutedCotangents {
            grad_y: &ones,
            every: None,
        };
        let grads_r = chebyshev_cr_routed_backward_into(
            &controls_r,
            &basis_r,
            (&cache_r, &route),
            cotangents,
            TERMS,
            &mut grad_xr,
        );
        let mut slopes_r = vec![0.0; len];
        for b in 0..2 {
            let mine = |e: usize| f64::from(u8::from(usize::from(route[e]) == b));
            agree(
                &form,
                &banks[b],
                (&xs, g),
                (&yr, &grads_r[b]),
                mine,
                &mut slopes_r,
            );
        }
        agree_slopes(&banks, g, &grad_xr, &slopes_r);
    }
}

#[test]
fn the_bounded_closed_form_holds_only_under_bounded_tangents() {
    for g in 3..=12 {
        for k in 1..=TERMS {
            let bounded = Knots::uniform(g).with_tangents(Tangents::Bounded);
            assert!(
                BoundedClosedForm::of(&bounded, k).is_some(),
                "grid {g}, {k} terms"
            );
            assert!(
                BoundedClosedForm::of(&Knots::uniform(g), k).is_none(),
                "chord knots have the chord form"
            );
        }
    }
    let bounded = Knots::uniform(6).with_tangents(Tangents::Bounded);
    assert!(BoundedClosedForm::of(&bounded, 4).is_none(), "four terms");
    assert!(BoundedClosedForm::of(&bounded, 0).is_none(), "no term");
    let refined = Knots::refined(6, &[0, 1, 0, 0, 0]).with_tangents(Tangents::Bounded);
    assert!(
        BoundedClosedForm::of(&refined, 3).is_none(),
        "refined knots"
    );
    let form = BoundedClosedForm::of(&bounded, 3).expect("uniform bounded knots");
    assert_eq!(
        form.defects([0.3, -0.7, 0.0]),
        [KnotDefect::default(); 2],
        "no vertex, no defect"
    );
    let defects = form.defects([0.1, 0.0, 0.5]);
    assert!(
        defects.iter().any(|d| d.delta() != 0.0),
        "a vertex at 0 inside"
    );
    assert!(
        form.correction(&defects, f32::NAN)[0].is_nan(),
        "NaN stays NaN"
    );
}
