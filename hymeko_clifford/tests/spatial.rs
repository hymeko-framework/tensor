//! Spatial vector algebra.
//!
//! Every test here is an **identity**, not a recorded number. A sign error in a spatial transform
//! does not produce an error or an obviously wrong figure — it produces a plausible six-vector, and
//! a robot that falls over slightly differently. So the checks are the structural facts that a
//! wrong sign cannot satisfy: power does not depend on the frame you measure it in, the two cross
//! products are dual, and transforming an inertia is the same as transforming what it acts on.

use hymeko_clifford::{Se3, SpatialInertia, Twist, Wrench};

/// xorshift64, so every case is reproducible without a dependency.
struct Rng(u64);
impl Rng {
    fn f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.f64() - 1.0
    }
    fn vec3(&mut self) -> [f64; 3] {
        [self.signed(), self.signed(), self.signed()]
    }
    fn pose(&mut self) -> Se3 {
        Se3::rpy_xyz(
            [
                self.signed() * 3.0,
                self.signed() * 3.0,
                self.signed() * 3.0,
            ],
            self.vec3(),
        )
    }
    fn twist(&mut self) -> Twist {
        Twist {
            w: self.vec3(),
            v: self.vec3(),
        }
    }
    fn wrench(&mut self) -> Wrench {
        Wrench {
            n: self.vec3(),
            f: self.vec3(),
        }
    }
    /// A body with a genuine, non-diagonal rotational inertia — a diagonal one would let a
    /// transposition slip through unnoticed.
    fn body(&mut self) -> SpatialInertia {
        let m = 0.5 + 2.0 * self.f64();
        let c = self.vec3();
        let a = [
            self.signed(),
            self.signed(),
            self.signed(),
            self.signed(),
            self.signed(),
            self.signed(),
            self.signed(),
            self.signed(),
            self.signed(),
        ];
        // A A^T, so it is symmetric and positive semi-definite like a real inertia tensor.
        let mut i = [0.0f64; 9];
        for r in 0..3 {
            for cc in 0..3 {
                for k in 0..3 {
                    i[r * 3 + cc] += a[r * 3 + k] * a[cc * 3 + k];
                }
            }
        }
        SpatialInertia::rigid(m, c, i)
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
        "{what}: {a} against {b}"
    );
}

fn twists_close(a: &Twist, b: &Twist, tol: f64, what: &str) {
    for k in 0..3 {
        close(a.w[k], b.w[k], tol, &format!("{what}: w[{k}]"));
        close(a.v[k], b.v[k], tol, &format!("{what}: v[{k}]"));
    }
}

fn wrenches_close(a: &Wrench, b: &Wrench, tol: f64, what: &str) {
    for k in 0..3 {
        close(a.n[k], b.n[k], tol, &format!("{what}: n[{k}]"));
        close(a.f[k], b.f[k], tol, &format!("{what}: f[{k}]"));
    }
}

// --- the transforms ----------------------------------------------------------------------------

#[test]
fn power_does_not_depend_on_the_frame_it_is_measured_in() {
    // THE test for the motion/force duality. A velocity and a force pair to a scalar in watts, and
    // watts are watts in any frame. If either transform has a wrong sign or a missing `t x` term,
    // this number moves — and nothing else in a simulation would have said so.
    let mut rng = Rng(0xF0CE);
    for case in 0..300 {
        let x = rng.pose();
        let v = rng.twist();
        let f = rng.wrench();
        close(
            v.to_parent(&x).dot(&f.to_parent(&x)),
            v.dot(&f),
            1e-12,
            &format!("power under to_parent, case {case}"),
        );
        close(
            v.to_child(&x).dot(&f.to_child(&x)),
            v.dot(&f),
            1e-12,
            &format!("power under to_child, case {case}"),
        );
    }
}

#[test]
fn changing_frame_and_changing_back_is_the_identity() {
    let mut rng = Rng(11);
    for case in 0..200 {
        let x = rng.pose();
        let v = rng.twist();
        let f = rng.wrench();
        twists_close(
            &v.to_parent(&x).to_child(&x),
            &v,
            1e-12,
            &format!("twist round trip {case}"),
        );
        wrenches_close(
            &f.to_parent(&x).to_child(&x),
            &f,
            1e-12,
            &format!("wrench round trip {case}"),
        );
    }
}

#[test]
fn a_transform_along_a_chain_is_the_composition_of_its_steps() {
    // Forward kinematics folds transforms; the adjoint has to fold the same way or a velocity
    // propagated down a chain drifts from one propagated in one jump. A two-link chain is the
    // smallest case that can tell the two composition orders apart.
    let mut rng = Rng(2718);
    for case in 0..200 {
        let a = rng.pose(); // child in parent
        let b = rng.pose(); // grandchild in child
        let ab = a.compose(&b); // grandchild in parent
        let v = rng.twist();
        let f = rng.wrench();
        twists_close(
            &v.to_parent(&b).to_parent(&a),
            &v.to_parent(&ab),
            1e-11,
            &format!("twist chain {case}"),
        );
        wrenches_close(
            &f.to_parent(&b).to_parent(&a),
            &f.to_parent(&ab),
            1e-11,
            &format!("wrench chain {case}"),
        );
    }
}

// --- the brackets ------------------------------------------------------------------------------

#[test]
fn the_force_cross_product_is_minus_the_transpose_of_the_motion_one() {
    // `(a x b) . f == -b . (a x* f)`. This duality is what makes the bias force `v x* (I v)` the
    // true rate of change of momentum rather than an approximation of it, and it is the identity
    // that fails first if the two cross products are transcribed independently.
    let mut rng = Rng(13);
    for case in 0..300 {
        let a = rng.twist();
        let b = rng.twist();
        let f = rng.wrench();
        close(
            a.cross_motion(&b).dot(&f),
            -b.dot(&a.cross_force(&f)),
            1e-12,
            &format!("duality {case}"),
        );
    }
}

#[test]
fn the_motion_bracket_is_antisymmetric_and_obeys_jacobi() {
    let mut rng = Rng(4);
    for case in 0..150 {
        let a = rng.twist();
        let b = rng.twist();
        let c = rng.twist();
        twists_close(
            &a.cross_motion(&b),
            &b.cross_motion(&a).scale(-1.0),
            1e-12,
            &format!("antisymmetry {case}"),
        );
        // [a,[b,c]] + [b,[c,a]] + [c,[a,b]] = 0 — this is a Lie algebra or it is not.
        let jacobi = a
            .cross_motion(&b.cross_motion(&c))
            .add(&b.cross_motion(&c.cross_motion(&a)))
            .add(&c.cross_motion(&a.cross_motion(&b)));
        twists_close(&jacobi, &Twist::ZERO, 1e-11, &format!("Jacobi {case}"));
    }
}

// --- inertia -----------------------------------------------------------------------------------

#[test]
fn a_point_masss_momentum_is_the_one_newton_would_write() {
    // Derived independently of the implementation: a point mass `m` at `c` on a body moving with
    // `(w, v)` has velocity `v + w x c`, so its linear momentum is `m` times that and its angular
    // momentum about the origin is `c x p`. Nothing in the six-vector formulation is assumed.
    let mut rng = Rng(1729);
    for case in 0..200 {
        let m = 0.3 + 3.0 * rng.f64();
        let c = rng.vec3();
        let t = rng.twist();
        let got = SpatialInertia::point_mass(m, c).apply(&t);

        let vel = [
            t.v[0] + t.w[1] * c[2] - t.w[2] * c[1],
            t.v[1] + t.w[2] * c[0] - t.w[0] * c[2],
            t.v[2] + t.w[0] * c[1] - t.w[1] * c[0],
        ];
        let p = [m * vel[0], m * vel[1], m * vel[2]];
        let want = Wrench {
            n: [
                c[1] * p[2] - c[2] * p[1],
                c[2] * p[0] - c[0] * p[2],
                c[0] * p[1] - c[1] * p[0],
            ],
            f: p,
        };
        wrenches_close(&got, &want, 1e-12, &format!("point mass {case}"));

        // and the kinetic energy is the elementary one
        close(
            0.5 * t.dot(&got),
            0.5 * m * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]),
            1e-11,
            &format!("kinetic energy {case}"),
        );
    }
}

#[test]
fn transforming_an_inertia_is_the_same_as_transforming_what_it_acts_on() {
    // The congruence, checked without ever writing the congruence out again: if `I` in the child's
    // frame maps `v` to `f`, then `I` in the parent's frame must map the parent's `v` to the
    // parent's `f`. This is the single fact the whole articulated-inertia recursion rests on.
    let mut rng = Rng(31337);
    for case in 0..250 {
        let x = rng.pose();
        let body = rng.body();
        let v = rng.twist();

        let here = body.articulated();
        let there = here.to_parent(&x);
        wrenches_close(
            &there.apply(&v.to_parent(&x)),
            &here.apply(&v).to_parent(&x),
            1e-9,
            &format!("congruence {case}"),
        );

        // and it stays symmetric — an asymmetric articulated inertia is a physically impossible
        // body, and the recursion would carry the defect all the way to the root
        for i in 0..3 {
            for j in 0..3 {
                close(
                    there.aa[i * 3 + j],
                    there.aa[j * 3 + i],
                    1e-10,
                    &format!("aa symmetry {case}"),
                );
                close(
                    there.ll[i * 3 + j],
                    there.ll[j * 3 + i],
                    1e-10,
                    &format!("ll symmetry {case}"),
                );
            }
        }
    }
}

#[test]
fn widening_a_rigid_inertia_does_not_change_what_it_does() {
    let mut rng = Rng(555);
    for case in 0..200 {
        let body = rng.body();
        let v = rng.twist();
        wrenches_close(
            &body.articulated().apply(&v),
            &body.apply(&v),
            1e-12,
            &format!("widen {case}"),
        );
    }
}

#[test]
fn the_rank_one_downdate_removes_exactly_one_direction() {
    // `(I - u u^T/d) v == I v - u (u . v)/d`, where `u . v` is the pairing. The recursion applies
    // this once per joint; if it removed anything more than the one direction the joint is free in,
    // a chain would go soft.
    let mut rng = Rng(909);
    for case in 0..200 {
        let ia = rng.body().articulated();
        let u = rng.wrench();
        let d = 0.5 + rng.f64();
        let v = rng.twist();

        let scale = v.dot(&u) / d;
        wrenches_close(
            &ia.downdate(&u, 1.0 / d).apply(&v),
            &ia.apply(&v).sub(&u.scale(scale)),
            1e-10,
            &format!("downdate {case}"),
        );
    }
}

// --- the exponential map --------------------------------------------------------------------

/// `exp` of the 4x4 `[[w]x, v; 0, 0]`, by scaling, a Taylor sum, and repeated squaring.
///
/// A second implementation that shares nothing with the first: no quaternion, no Rodrigues formula,
/// no series coefficient, no case split on the angle. It is slow and it is obvious, which is exactly
/// what a reference should be.
fn matrix_exp_reference(t: &Twist) -> ([f64; 9], [f64; 3]) {
    let mut m = [[0.0f64; 4]; 4];
    m[0][1] = -t.w[2];
    m[0][2] = t.w[1];
    m[1][0] = t.w[2];
    m[1][2] = -t.w[0];
    m[2][0] = -t.w[1];
    m[2][1] = t.w[0];
    for (row, &v) in m.iter_mut().zip(&t.v) {
        row[3] = v;
    }

    // scale until the norm is small enough that twenty terms are past double precision
    let norm = m
        .iter()
        .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let squarings = if norm > 0.25 {
        (norm / 0.25).log2().ceil() as u32
    } else {
        0
    };
    let scale = 1.0 / (1u64 << squarings) as f64;
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v *= scale;
        }
    }

    let mul = |a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]| {
        let mut o = [[0.0f64; 4]; 4];
        for (i, orow) in o.iter_mut().enumerate() {
            for (k, &aik) in a[i].iter().enumerate() {
                for (j, ov) in orow.iter_mut().enumerate() {
                    *ov += aik * b[k][j];
                }
            }
        }
        o
    };

    let mut acc = [[0.0f64; 4]; 4];
    for (i, row) in acc.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let mut term = acc;
    for k in 1..24 {
        term = mul(&term, &m);
        for row in term.iter_mut() {
            for v in row.iter_mut() {
                *v /= k as f64;
            }
        }
        for (a, t) in acc.iter_mut().zip(&term) {
            for (av, tv) in a.iter_mut().zip(t) {
                *av += tv;
            }
        }
    }
    for _ in 0..squarings {
        acc = mul(&acc, &acc);
    }

    let mut r = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = acc[i][j];
        }
    }
    (r, [acc[0][3], acc[1][3], acc[2][3]])
}

/// A twist whose rotation has exactly the magnitude asked for.
fn twist_of_magnitude(rng: &mut Rng, magnitude: f64) -> Twist {
    let mut w = rng.vec3();
    let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().max(1e-300);
    for c in w.iter_mut() {
        *c *= magnitude / n;
    }
    Twist { w, v: rng.vec3() }
}

#[test]
fn the_exponential_agrees_with_a_matrix_exponential_that_shares_no_code_with_it() {
    // Across eleven orders of magnitude in the rotation angle, so the series branch, the closed-form
    // branch and the crossover between them are all covered by the same reference.
    let mut rng = Rng(0xE4F);
    for decade in 0..11 {
        let magnitude = 10f64.powi(-decade) * 3.0;
        for case in 0..40 {
            let t = twist_of_magnitude(&mut rng, magnitude);

            let got = t.exp();
            let (want_r, want_t) = matrix_exp_reference(&t);
            let mut got_r = [0.0f64; 9];
            got.rotation_matrix_into(&mut got_r);

            for k in 0..9 {
                close(
                    got_r[k],
                    want_r[k],
                    1e-11,
                    &format!("|w|={magnitude:e} case {case}: rotation[{k}]"),
                );
            }
            for (k, (&got_v, &want_v)) in got.translation.iter().zip(&want_t).enumerate() {
                close(
                    got_v,
                    want_v,
                    1e-11,
                    &format!("|w|={magnitude:e} case {case}: translation[{k}]"),
                );
            }
        }
    }
}

#[test]
fn the_logarithm_undoes_the_exponential_and_the_other_way_round() {
    let mut rng = Rng(1234);
    for decade in 0..12 {
        // below pi, where the logarithm is single-valued
        let magnitude = 10f64.powi(-decade) * 3.0;
        for case in 0..30 {
            let t = twist_of_magnitude(&mut rng, magnitude);
            twists_close(
                &t.exp().log(),
                &t,
                1e-9,
                &format!("|w|={magnitude:e} log(exp) case {case}"),
            );
        }
    }

    // and from the group's side
    for case in 0..200 {
        let x = rng.pose();
        let back = x.log().exp();
        for k in 0..3 {
            close(
                back.translation[k],
                x.translation[k],
                1e-10,
                &format!("exp(log) case {case}: translation[{k}]"),
            );
        }
        let (mut a, mut b) = ([0.0f64; 9], [0.0f64; 9]);
        back.rotation_matrix_into(&mut a);
        x.rotation_matrix_into(&mut b);
        for k in 0..9 {
            close(
                a[k],
                b[k],
                1e-10,
                &format!("exp(log) case {case}: rotation[{k}]"),
            );
        }
    }
}

#[test]
fn a_constant_twist_traces_a_one_parameter_subgroup() {
    // `exp(s v) . exp(t v) == exp((s+t) v)`. This is what makes it a screw motion rather than a
    // sequence of small corrections, and it is the property an integrator relies on: a step of `dt`
    // taken twice must equal one step of `2 dt`, exactly, at any size.
    let mut rng = Rng(88);
    for case in 0..200 {
        let t = Twist {
            w: rng.vec3(),
            v: rng.vec3(),
        };
        for (a, b) in [(0.3, 0.7), (1.0, 1.0), (2.5, -1.5), (1e-9, 1.0)] {
            let split = t.scale(a).exp().compose(&t.scale(b).exp());
            let whole = t.scale(a + b).exp();
            for k in 0..3 {
                close(
                    split.translation[k],
                    whole.translation[k],
                    1e-10,
                    &format!("case {case} split ({a},{b}): translation[{k}]"),
                );
            }
        }
    }
}

#[test]
fn integrating_a_twist_in_many_small_steps_lands_where_one_big_step_does() {
    // The claim a rigid-body integrator actually makes: composing `exp(v/n)` n times equals
    // `exp(v)`, for any n, with no drift and no renormalisation. Ten thousand steps, and the rotor
    // must still be unit to machine precision — a linear update would have left the group at the
    // first one.
    let mut rng = Rng(31);
    for case in 0..20 {
        let t = Twist {
            w: rng.vec3(),
            v: rng.vec3(),
        };
        let n = 10_000;
        let small = t.scale(1.0 / n as f64).exp();
        let mut acc = Se3::identity();
        for _ in 0..n {
            acc = acc.compose(&small);
        }
        let whole = t.exp();

        for k in 0..3 {
            close(
                acc.translation[k],
                whole.translation[k],
                1e-9,
                &format!("case {case}: translation[{k}] after {n} steps"),
            );
        }
        // still exactly on the group
        let r = &acc.rotor;
        let norm = (r[0] * r[0] + r[3] * r[3] + r[5] * r[5] + r[6] * r[6]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "case {case}: the rotor drifted off the group, |R| = {norm}"
        );
    }
}

#[test]
fn the_degenerate_cases_are_the_ones_they_should_be() {
    // A twist with no rotation is a straight translation, and one with no translation is a rotation
    // about the origin. Both are where the series branch is reached with nothing left to cancel.
    let mut rng = Rng(7);
    for case in 0..100 {
        let v = rng.vec3();
        let straight = Twist { w: [0.0; 3], v }.exp();
        for (&got, &want) in straight.translation.iter().zip(&v) {
            assert_eq!(
                got, want,
                "case {case}: a twist with no rotation must translate by exactly v"
            );
        }
        assert_eq!(
            straight.rotor,
            Se3::identity().rotor,
            "case {case}: and not turn"
        );

        let axis = rng.vec3();
        let angle = rng.signed() * 3.0;
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let w = [
            axis[0] / n * angle,
            axis[1] / n * angle,
            axis[2] / n * angle,
        ];
        let spun = Twist { w, v: [0.0; 3] }.exp();
        let reference = Se3::axis_angle(axis, angle);
        // Close, NOT bit-identical, and deliberately so: `exp` recovers the angle as `|w|` and
        // normalises `w = axis * angle / |axis|`, while `axis_angle` normalises `axis` and is handed
        // the angle. Those are different sequences of roundings for the same function, and demanding
        // the last bit of one from the other would be asserting that floating-point multiplication
        // is associative.
        for k in 0..8 {
            close(
                spun.rotor[k],
                reference.rotor[k],
                1e-14,
                &format!("case {case}: pure rotation rotor[{k}]"),
            );
        }
        assert_eq!(
            spun.translation,
            [0.0; 3],
            "case {case}: a rotation about the origin moves the origin nowhere"
        );
    }
    // and the identity logs to zero
    twists_close(&Se3::identity().log(), &Twist::ZERO, 1e-15, "identity");
}
