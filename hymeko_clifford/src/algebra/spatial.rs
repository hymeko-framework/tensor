//! Spatial vectors — the Lie algebra of the group `se_fixed` already carries.
//!
//! A rigid body's velocity is not a vector in space; it is an element of `se(3)`, and its momentum
//! is an element of the dual. Writing them as separate angular and linear three-vectors and then
//! remembering by hand which one picks up a `t x` term under a change of frame is the classical
//! source of sign errors in multibody dynamics. Here the pair is one object and the frame change is
//! one method.
//!
//! # Why this is here and not in the kinematics crate
//!
//! Nothing below knows what a link, a joint or a robot is. [`Twist`] is the derivative of an
//! [`Se3`] along a curve, [`Wrench`] is its dual under the pairing that gives power, and
//! [`SpatialInertia`] is the metric that connects them. That is the algebra of the group, in the
//! crate that owns the group.
//!
//! # Ordering and convention
//!
//! **Angular first**, throughout: a motion vector is `(w, v)` and a force vector is `(n, f)`. This
//! is Featherstone's convention, and it is stated rather than inferred because the opposite order
//! is equally common and the two produce plausible-looking wrong answers rather than errors.
//!
//! A [`Se3`] argument named `x` always means **the child's pose expressed in the parent's frame** —
//! the same thing a kinematic plan composes when it walks a chain. So [`Twist::to_parent`] takes a
//! velocity written in the child's frame and rewrites it in the parent's.
//!
//! # The pairing
//!
//! `motion.dot(&force)` is power, in watts, and it is the only way the two kinds combine. They are
//! deliberately distinct types: adding a velocity to a momentum is meaningless, and in a codebase
//! where both are six numbers a compiler that will not let you do it is worth more than a comment.

use super::se_fixed::Se3;

// --- small dense helpers, row-major 3x3 --------------------------------------------------------

/// `a x b`.
#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The matrix `t x`, so that `skew(t) . v == cross(&t, &v)`.
#[inline]
fn skew(t: &[f64; 3]) -> [f64; 9] {
    [0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0]
}

/// `a . b`, both row-major.
#[inline]
fn mat_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut o = [0.0; 9];
    for i in 0..3 {
        for k in 0..3 {
            let aik = a[i * 3 + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..3 {
                o[i * 3 + j] += aik * b[k * 3 + j];
            }
        }
    }
    o
}

/// `a . v`.
#[inline]
fn mat_vec(a: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

/// `a^T . v`.
#[inline]
fn mat_t_vec(a: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [
        a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
        a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
        a[2] * v[0] + a[5] * v[1] + a[8] * v[2],
    ]
}

#[inline]
fn transpose(a: &[f64; 9]) -> [f64; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

#[inline]
fn add_scaled(a: &mut [f64; 9], b: &[f64; 9], s: f64) {
    for i in 0..9 {
        a[i] += s * b[i];
    }
}

/// The rotation of `x` as a row-major 3x3, extracted once.
///
/// The blockwise congruences below need `R . A . R^T`, which cannot be spelled through
/// [`Se3::rotate`] without a rotor sandwich per column. Pulling the matrix out once and using dense
/// arithmetic is both clearer and cheaper.
#[inline]
fn rotation_of(x: &Se3) -> [f64; 9] {
    let mut r = [0.0f64; 9];
    x.rotation_matrix_into(&mut r);
    r
}

/// `R . a . R^T`, the congruence that moves a 3x3 block between frames.
#[inline]
fn conjugate(r: &[f64; 9], a: &[f64; 9]) -> [f64; 9] {
    mat_mul(&mat_mul(r, a), &transpose(r))
}

// --- motion ------------------------------------------------------------------------------------

/// A spatial motion vector: `(w, v)`, angular first.
///
/// `v` is the velocity of the body-fixed point **currently at the frame's origin**, not the velocity
/// of the body's centre of mass. That distinction is what makes the algebra work: it is what lets a
/// single six-vector describe the motion of the whole rigid body, and what lets two of them add.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Twist {
    /// Angular velocity.
    pub w: [f64; 3],
    /// Linear velocity of the body-fixed point at the origin.
    pub v: [f64; 3],
}

impl Twist {
    /// The zero motion.
    pub const ZERO: Self = Self {
        w: [0.0; 3],
        v: [0.0; 3],
    };

    /// The motion subspace of a revolute joint turning about `axis`, which must be a unit vector in
    /// the child's frame passing through that frame's origin.
    #[inline]
    pub fn revolute(axis: [f64; 3]) -> Self {
        Self {
            w: axis,
            v: [0.0; 3],
        }
    }

    /// The motion subspace of a prismatic joint sliding along `axis`, a unit vector in the child's
    /// frame.
    #[inline]
    pub fn prismatic(axis: [f64; 3]) -> Self {
        Self {
            w: [0.0; 3],
            v: axis,
        }
    }

    /// `self * s`.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            w: [self.w[0] * s, self.w[1] * s, self.w[2] * s],
            v: [self.v[0] * s, self.v[1] * s, self.v[2] * s],
        }
    }

    /// `self + rhs`.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            w: [
                self.w[0] + rhs.w[0],
                self.w[1] + rhs.w[1],
                self.w[2] + rhs.w[2],
            ],
            v: [
                self.v[0] + rhs.v[0],
                self.v[1] + rhs.v[1],
                self.v[2] + rhs.v[2],
            ],
        }
    }

    /// Power: the pairing of a motion with a force, in watts.
    ///
    /// The only product defined between the two kinds, and the reason they are separate types.
    #[inline]
    pub fn dot(&self, f: &Wrench) -> f64 {
        self.w[0] * f.n[0]
            + self.w[1] * f.n[1]
            + self.w[2] * f.n[2]
            + self.v[0] * f.f[0]
            + self.v[1] * f.f[1]
            + self.v[2] * f.f[2]
    }

    /// The spatial cross product on motion vectors, `self x rhs`.
    ///
    /// This is the Lie bracket of `se(3)`. It appears in the equations of motion as the velocity
    /// product term — the Coriolis and centrifugal effects — which is precisely what a
    /// finite-difference Jacobian formulation has no way to express and therefore silently omits.
    #[inline]
    pub fn cross_motion(&self, rhs: &Self) -> Self {
        let a = cross(&self.w, &rhs.v);
        let b = cross(&self.v, &rhs.w);
        Self {
            w: cross(&self.w, &rhs.w),
            v: [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
        }
    }

    /// The dual cross product, `self x* f`, which acts on force vectors.
    ///
    /// The minus-transpose of [`Self::cross_motion`] under the pairing, which is what makes the
    /// bias force `v x* (I v)` the rate of change of momentum seen in a moving frame.
    #[inline]
    pub fn cross_force(&self, f: &Wrench) -> Wrench {
        let a = cross(&self.w, &f.n);
        let b = cross(&self.v, &f.f);
        Wrench {
            n: [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            f: cross(&self.w, &f.f),
        }
    }

    /// Rewrite this velocity, given in the child's frame, in the parent's.
    ///
    /// `x` is the child's pose in the parent's frame.
    #[inline]
    pub fn to_parent(&self, x: &Se3) -> Self {
        let (w, v) = to_parent_pair(x, &self.w, &self.v);
        Self { w, v }
    }

    /// Rewrite this velocity, given in the parent's frame, in the child's. The inverse of
    /// [`Self::to_parent`].
    #[inline]
    pub fn to_child(&self, x: &Se3) -> Self {
        let (w, v) = to_child_pair(x, &self.w, &self.v);
        Self { w, v }
    }
}

// --- force ---------------------------------------------------------------------------------

/// A spatial force vector: `(n, f)`, moment first.
///
/// `n` is the moment **about the frame's origin**, so a pure force applied away from the origin has
/// a non-zero `n` in that frame and none in a frame on its line of action. Momentum is the same kind
/// of object: [`SpatialInertia::apply`] returns one.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Wrench {
    /// Moment about the frame's origin.
    pub n: [f64; 3],
    /// Resultant force.
    pub f: [f64; 3],
}

impl Wrench {
    /// The zero force.
    pub const ZERO: Self = Self {
        n: [0.0; 3],
        f: [0.0; 3],
    };

    /// `self * s`.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            n: [self.n[0] * s, self.n[1] * s, self.n[2] * s],
            f: [self.f[0] * s, self.f[1] * s, self.f[2] * s],
        }
    }

    /// `self + rhs`.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            n: [
                self.n[0] + rhs.n[0],
                self.n[1] + rhs.n[1],
                self.n[2] + rhs.n[2],
            ],
            f: [
                self.f[0] + rhs.f[0],
                self.f[1] + rhs.f[1],
                self.f[2] + rhs.f[2],
            ],
        }
    }

    /// `self - rhs`.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-1.0))
    }

    /// Rewrite this force, given in the child's frame, in the parent's.
    ///
    /// Identical in form to [`Twist::to_parent`] with the roles of the two halves swapped: the
    /// force rotates and the moment picks up the `t x` term, exactly as the angular velocity
    /// rotates and the linear velocity picks it up. That is not a coincidence — it is the statement
    /// that the force transform is the inverse-transpose of the motion transform, and it is why
    /// both are one shared function below.
    #[inline]
    pub fn to_parent(&self, x: &Se3) -> Self {
        let (f, n) = to_parent_pair(x, &self.f, &self.n);
        Self { n, f }
    }

    /// Rewrite this force, given in the parent's frame, in the child's.
    #[inline]
    pub fn to_child(&self, x: &Se3) -> Self {
        let (f, n) = to_child_pair(x, &self.f, &self.n);
        Self { n, f }
    }
}

/// The shared body of both `to_parent`s: `a` rotates, `b` rotates and picks up `t x Ra`.
#[inline]
fn to_parent_pair(x: &Se3, a: &[f64; 3], b: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let ra = x.rotate(a);
    let rb = x.rotate(b);
    let tc = cross(&x.translation, &ra);
    (ra, [rb[0] + tc[0], rb[1] + tc[1], rb[2] + tc[2]])
}

/// The inverse of [`to_parent_pair`].
#[inline]
fn to_child_pair(x: &Se3, a: &[f64; 3], b: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let xi = x.inverse();
    let tc = cross(&x.translation, a);
    let shifted = [b[0] - tc[0], b[1] - tc[1], b[2] - tc[2]];
    (xi.rotate(a), xi.rotate(&shifted))
}

// --- inertia -----------------------------------------------------------------------------------

/// The spatial inertia of one rigid body, about its frame's origin.
///
/// Stored as the mass, the first moment `h = m.c`, and the rotational inertia about the **origin**
/// rather than about the centre of mass. The parallel-axis shift is done once at construction
/// because that is the form every use below wants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialInertia {
    /// Total mass.
    pub mass: f64,
    /// First moment of mass, `m . c`.
    pub h: [f64; 3],
    /// Rotational inertia about the frame's origin, row-major and symmetric.
    pub rot: [f64; 9],
}

impl SpatialInertia {
    /// A massless body.
    pub const ZERO: Self = Self {
        mass: 0.0,
        h: [0.0; 3],
        rot: [0.0; 9],
    };

    /// A point mass `mass` at `c`.
    ///
    /// Its rotational inertia about the origin is `m(|c|^2 1 - c c^T)`, which is **singular along
    /// `c`**: a point has no inertia about an axis through itself. That is a property of the model,
    /// not a defect of the algorithm, and it is the same degeneracy an explicit mass matrix hides
    /// behind a ridge term. A caller that can be handed such a configuration must regularise the
    /// joint's articulated inertia, not this.
    pub fn point_mass(mass: f64, c: [f64; 3]) -> Self {
        let cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        let mut rot = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                rot[i * 3 + j] = mass * (if i == j { cc } else { 0.0 } - c[i] * c[j]);
            }
        }
        Self {
            mass,
            h: [mass * c[0], mass * c[1], mass * c[2]],
            rot,
        }
    }

    /// A rigid body of mass `mass` whose centre of mass is at `c` and whose rotational inertia
    /// **about that centre of mass** is `inertia_com`, row-major and symmetric.
    pub fn rigid(mass: f64, c: [f64; 3], inertia_com: [f64; 9]) -> Self {
        let mut s = Self::point_mass(mass, c);
        for (r, i) in s.rot.iter_mut().zip(inertia_com) {
            *r += i;
        }
        s
    }

    /// Momentum: `I . v`.
    #[inline]
    pub fn apply(&self, v: &Twist) -> Wrench {
        let n = mat_vec(&self.rot, &v.w);
        let hv = cross(&self.h, &v.v);
        let wh = cross(&v.w, &self.h);
        Wrench {
            n: [n[0] + hv[0], n[1] + hv[1], n[2] + hv[2]],
            f: [
                self.mass * v.v[0] + wh[0],
                self.mass * v.v[1] + wh[1],
                self.mass * v.v[2] + wh[2],
            ],
        }
    }

    /// Widen to the general symmetric form.
    ///
    /// A rigid body's inertia has structure — the linear block is `m.1` and the coupling block is
    /// `h x`. An **articulated** inertia does not: a subtree resists a push differently in
    /// different directions, because its joints let it give way in some and not others. So the
    /// recursion has to widen, and this is where.
    #[inline]
    pub fn articulated(&self) -> ArticulatedInertia {
        let mut ll = [0.0f64; 9];
        ll[0] = self.mass;
        ll[4] = self.mass;
        ll[8] = self.mass;
        ArticulatedInertia {
            aa: self.rot,
            al: skew(&self.h),
            ll,
        }
    }
}

/// A general symmetric spatial inertia, in three 3x3 blocks: `[[aa, al], [al^T, ll]]`.
///
/// The articulated-body inertia of a subtree — what the subtree's base feels when pushed, with
/// every joint below it free to move. It is symmetric and positive semi-definite but has none of a
/// rigid body's structure.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ArticulatedInertia {
    /// Angular-angular block, symmetric.
    pub aa: [f64; 9],
    /// Angular-linear coupling. **Not** symmetric in general.
    pub al: [f64; 9],
    /// Linear-linear block, symmetric.
    pub ll: [f64; 9],
}

impl ArticulatedInertia {
    /// The zero inertia.
    pub const ZERO: Self = Self {
        aa: [0.0; 9],
        al: [0.0; 9],
        ll: [0.0; 9],
    };

    /// `I^A . v`.
    #[inline]
    pub fn apply(&self, v: &Twist) -> Wrench {
        let n1 = mat_vec(&self.aa, &v.w);
        let n2 = mat_vec(&self.al, &v.v);
        let f1 = mat_t_vec(&self.al, &v.w);
        let f2 = mat_vec(&self.ll, &v.v);
        Wrench {
            n: [n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]],
            f: [f1[0] + f2[0], f1[1] + f2[1], f1[2] + f2[2]],
        }
    }

    /// `self += rhs`.
    #[inline]
    pub fn add_assign(&mut self, rhs: &Self) {
        for i in 0..9 {
            self.aa[i] += rhs.aa[i];
            self.al[i] += rhs.al[i];
            self.ll[i] += rhs.ll[i];
        }
    }

    /// `self - (u u^T)/d`, the rank-one downdate that removes what one free joint lets slip.
    ///
    /// This is the whole reason the articulated-body recursion is linear in the degrees of freedom:
    /// a joint's effect on its parent is a rank-**one** correction, so it can be folded in as the
    /// sweep passes rather than assembled into a matrix and factorised afterwards.
    #[inline]
    pub fn downdate(&self, u: &Wrench, d_inv: f64) -> Self {
        let mut o = *self;
        for i in 0..3 {
            for j in 0..3 {
                o.aa[i * 3 + j] -= d_inv * u.n[i] * u.n[j];
                o.al[i * 3 + j] -= d_inv * u.n[i] * u.f[j];
                o.ll[i * 3 + j] -= d_inv * u.f[i] * u.f[j];
            }
        }
        o
    }

    /// Rewrite this inertia, given in the child's frame, in the parent's.
    ///
    /// The congruence `Ad* . I^A . (Ad*)^T`, blockwise. Written out rather than assembled as a 6x6
    /// because the 6x6 is mostly the same three rotations applied three times, and because this
    /// form preserves the symmetry of `aa` and `ll` exactly — a general 6x6 product would lose it
    /// in the last bit and the recursion would drift asymmetric over a long chain.
    pub fn to_parent(&self, x: &Se3) -> Self {
        let r = rotation_of(x);
        let t = skew(&x.translation);

        let ll = conjugate(&r, &self.ll);
        let hp = conjugate(&r, &self.al);
        let ip = conjugate(&r, &self.aa);

        // al_p = H' + t~ M'
        let mut al = hp;
        add_scaled(&mut al, &mat_mul(&t, &ll), 1.0);

        // aa_p = I' + t~ H'^T - H' t~ - t~ M' t~
        let mut aa = ip;
        add_scaled(&mut aa, &mat_mul(&t, &transpose(&hp)), 1.0);
        add_scaled(&mut aa, &mat_mul(&hp, &t), -1.0);
        add_scaled(&mut aa, &mat_mul(&mat_mul(&t, &ll), &t), -1.0);

        Self { aa, al, ll }
    }
}
