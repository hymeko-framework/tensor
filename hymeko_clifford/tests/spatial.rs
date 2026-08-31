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
