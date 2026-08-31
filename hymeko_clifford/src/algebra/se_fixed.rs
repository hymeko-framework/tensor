//! $SE(n)$ with the dimension supplied at compile time.
//!
//! The same group and the same arithmetic as [`super::Se`], laid out on the stack. [`super::Se`]
//! stores its rotor as a `Multivector` with a `Vec<f64>` of $2^n$ coefficients and its translation
//! as a second `Vec<f64>`, so **every rigid motion is two heap allocations** — and a forward
//! kinematics pass builds four motions per link. Measured on a 33-link quadruped, that is about
//! 1200 allocations for one pose and roughly seventy per cent of the time.
//!
//! Nothing about the algebra requires it. A rigid motion in three dimensions is eight coefficients
//! and three coordinates, both known when the code is compiled.
//!
//! # A specialisation, not a second implementation
//!
//! This is one generic body parameterised by the dimension. [`Se3`] is the alias
//! `SeFixed<Dim3>`, not a hand-written three-dimensional type sitting beside a general one — a
//! second implementation of a group law is a second thing to keep correct, and the arithmetic here
//! is dimension-agnostic apart from the constructors that name axes.
//!
//! Rust cannot yet spell `[f64; 1 << N]` on stable, so the dimension arrives as a type carrying its
//! own storage rather than as a `const N: usize`. [`EuclideanDim`] is that carrier: it names $n$,
//! the $2^n$ blade array and the $n$-coordinate array, and the group law is written once against it.
//!
//! # Bit-identical, deliberately
//!
//! Every loop below reproduces its counterpart in `Multivector` and [`super::Se`] exactly: the same
//! bounds, the same skip-if-zero guards, the same accumulation order. Floating-point addition is not
//! associative, so a faster summation order would be a *different function* whose answers differ in
//! the last bit — and a caller that had recorded a pose would see it move. `tests/se_fixed.rs`
//! checks the equality rather than assuming it.

use super::blade::blade_product;
use super::multivector::Multivector;
use super::se::Se;
use super::signature::Signature;

/// A Euclidean dimension, and the fixed-size storage its rigid motions need.
///
/// Implementors are markers: they carry no data, only the two array types and $n$.
pub trait EuclideanDim: Copy + 'static {
    /// The Euclidean dimension $n$.
    const N: usize;
    /// $2^n$ blade coefficients, indexed by blade bitmask exactly as `Multivector` indexes them.
    type Blades: Copy + AsRef<[f64]> + AsMut<[f64]>;
    /// $n$ coordinates.
    type Coords: Copy + AsRef<[f64]> + AsMut<[f64]>;
    /// An all-zero blade array.
    fn zero_blades() -> Self::Blades;
    /// An all-zero coordinate array.
    fn zero_coords() -> Self::Coords;
}

/// The plane: $SE(2)$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim2;
/// Space: $SE(3)$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3;

impl EuclideanDim for Dim2 {
    const N: usize = 2;
    type Blades = [f64; 4];
    type Coords = [f64; 2];
    fn zero_blades() -> Self::Blades {
        [0.0; 4]
    }
    fn zero_coords() -> Self::Coords {
        [0.0; 2]
    }
}

impl EuclideanDim for Dim3 {
    const N: usize = 3;
    type Blades = [f64; 8];
    type Coords = [f64; 3];
    fn zero_blades() -> Self::Blades {
        [0.0; 8]
    }
    fn zero_coords() -> Self::Coords {
        [0.0; 3]
    }
}

/// A rigid motion $(R, t)$ of $\mathbb{R}^n$, stored inline.
#[derive(Clone, Copy, Debug)]
pub struct SeFixed<D: EuclideanDim> {
    /// The rotor, by blade bitmask.
    pub rotor: D::Blades,
    /// The translation, applied after the rotation.
    pub translation: D::Coords,
}

/// Rigid motions of space, the case a robot's kinematics is written in.
pub type Se3 = SeFixed<Dim3>;
/// Rigid motions of the plane.
pub type Se2 = SeFixed<Dim2>;

impl<D: EuclideanDim> SeFixed<D> {
    fn signature() -> Signature {
        Signature::euclidean(D::N)
    }

    /// The identity motion.
    pub fn identity() -> Self {
        let mut rotor = D::zero_blades();
        rotor.as_mut()[0] = 1.0;
        Self {
            rotor,
            translation: D::zero_coords(),
        }
    }

    /// A pure translation.
    pub fn translation(t: &[f64]) -> Self {
        assert_eq!(t.len(), D::N, "translation must have n components");
        let mut out = Self::identity();
        out.translation.as_mut().copy_from_slice(t);
        out
    }

    /// The geometric product of two blade arrays — the mirror of `Multivector::geo`.
    fn geo(a: &D::Blades, b: &D::Blades) -> D::Blades {
        let sig = Self::signature();
        let (a, b) = (a.as_ref(), b.as_ref());
        let mut out = D::zero_blades();
        let o = out.as_mut();
        // The index IS the blade bitmask, which is why it is enumerated rather than iterated over:
        // `blade_product` needs the mask, not the coefficient.
        for (a_idx, &av) in a.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            for (b_idx, &bv) in b.iter().enumerate() {
                if bv == 0.0 {
                    continue;
                }
                let (r_idx, sign) = blade_product(a_idx, b_idx, &sig);
                o[r_idx] += sign * av * bv;
            }
        }
        out
    }

    /// The reverse $\tilde{R}$ — the mirror of `Multivector::reverse`.
    fn reverse(a: &D::Blades) -> D::Blades {
        let src = a.as_ref();
        let mut out = D::zero_blades();
        let o = out.as_mut();
        for (i, o_i) in o.iter_mut().enumerate() {
            let k = i.count_ones();
            let sign = if (k * (k.wrapping_sub(1)) / 2).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            *o_i = sign * src[i];
        }
        out
    }

    /// Rotate a vector, ignoring the translation: $R\,v\,\tilde{R}$.
    ///
    /// # Panics
    /// If `v.len() != n`.
    pub fn rotate(&self, v: &[f64]) -> D::Coords {
        assert_eq!(v.len(), D::N, "vector must have n components");
        let mut mv = D::zero_blades();
        {
            let m = mv.as_mut();
            for (i, &x) in v.iter().enumerate() {
                m[1 << i] = x;
            }
        }
        let product = Self::geo(&Self::geo(&self.rotor, &mv), &Self::reverse(&self.rotor));
        let mut out = D::zero_coords();
        {
            let (o, p) = (out.as_mut(), product.as_ref());
            for i in 0..D::N {
                o[i] = p[1 << i];
            }
        }
        out
    }

    /// Apply the full motion to a point: $R\,p\,\tilde{R} + t$.
    pub fn apply(&self, p: &[f64]) -> D::Coords {
        let mut out = self.rotate(p);
        let (o, t) = (out.as_mut(), self.translation.as_ref());
        for i in 0..D::N {
            o[i] += t[i];
        }
        out
    }

    /// Compose: the motion that applies `rhs` first, then `self`.
    ///
    /// $(R_1, t_1) \circ (R_2, t_2) = (R_1 R_2,\; R_1 t_2 \tilde{R_1} + t_1)$.
    pub fn compose(&self, rhs: &Self) -> Self {
        let moved = self.rotate(rhs.translation.as_ref());
        let mut translation = D::zero_coords();
        {
            let (out, m, t) = (translation.as_mut(), moved.as_ref(), self.translation.as_ref());
            for i in 0..D::N {
                out[i] = m[i] + t[i];
            }
        }
        Self {
            rotor: Self::geo(&self.rotor, &rhs.rotor),
            translation,
        }
    }

    /// The inverse motion, $(\tilde{R},\; -\tilde{R}\,t\,R)$.
    pub fn inverse(&self) -> Self {
        let inv = Self {
            rotor: Self::reverse(&self.rotor),
            translation: D::zero_coords(),
        };
        let t = inv.rotate(self.translation.as_ref());
        let mut translation = D::zero_coords();
        {
            let (o, s) = (translation.as_mut(), t.as_ref());
            for i in 0..D::N {
                o[i] = -s[i];
            }
        }
        Self {
            rotor: inv.rotor,
            translation,
        }
    }

    /// Write the rotation as an $n \times n$ row-major matrix — column $j$ is the image of
    /// $e_{j+1}$. Into a caller's buffer, so a pose loop allocates nothing.
    ///
    /// # Panics
    /// If `out.len() < n * n`.
    pub fn rotation_matrix_into(&self, out: &mut [f64]) {
        assert!(out.len() >= D::N * D::N, "out must hold n*n");
        let mut e = D::zero_coords();
        for j in 0..D::N {
            {
                let ev = e.as_mut();
                ev.iter_mut().for_each(|x| *x = 0.0);
                ev[j] = 1.0;
            }
            let col = self.rotate(e.as_ref());
            let c = col.as_ref();
            for (i, &v) in c.iter().enumerate().take(D::N) {
                out[i * D::N + j] = v;
            }
        }
    }

    /// The same motion in the heap-backed general representation.
    pub fn to_se(&self) -> Se {
        let mut rotor = Multivector::zero(D::N);
        rotor.components.copy_from_slice(self.rotor.as_ref());
        Se::from_parts(rotor, self.translation.as_ref().to_vec())
    }

    /// The same motion, taken from the general representation.
    ///
    /// `None` if `se` is not of this dimension — a caller that mixes dimensions has a bug, and
    /// silently reinterpreting the coefficients would hide it.
    pub fn from_se(se: &Se) -> Option<Self> {
        if se.n != D::N || se.rotor.components.len() != (1usize << D::N) {
            return None;
        }
        let mut rotor = D::zero_blades();
        rotor.as_mut().copy_from_slice(&se.rotor.components);
        let mut translation = D::zero_coords();
        translation.as_mut().copy_from_slice(&se.translation);
        Some(Self { rotor, translation })
    }
}

// ─── the three-dimensional specialisation ───────────────────────────────────────────────────────
//
// Only the constructors that NAME axes are dimension-specific; everything above is the group law,
// written once.

impl Se3 {
    /// A rotation of `angle` about `axis` (need not be unit), $n = 3$.
    pub fn axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-12 {
            return Self::identity();
        }
        let a = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        let (s, c) = (angle * 0.5).sin_cos();
        // blade masks in Cl(3): e1=1, e2=2, e3=4, so e1e2=3, e1e3=5, e2e3=6.
        let mut rotor = [0.0f64; 8];
        rotor[0] = c;
        rotor[6] = -s * a[0]; //  e2e3
        rotor[5] = s * a[1]; // -e3e1 == +e1e3
        rotor[3] = -s * a[2]; //  e1e2
        Self {
            rotor,
            translation: [0.0; 3],
        }
    }

    /// A joint origin in the URDF/SDF convention: translate by `xyz`, rotate by fixed-axis
    /// roll-pitch-yaw in **radians**, composed $R = R_z R_y R_x$.
    pub fn rpy_xyz(rpy: [f64; 3], xyz: [f64; 3]) -> Self {
        let r = Self::axis_angle([1.0, 0.0, 0.0], rpy[0]);
        let p = Self::axis_angle([0.0, 1.0, 0.0], rpy[1]);
        let y = Self::axis_angle([0.0, 0.0, 1.0], rpy[2]);
        let mut m = y.compose(&p).compose(&r);
        m.translation = xyz;
        m
    }

    /// A translation of `distance` along `axis` — a prismatic joint's motion.
    pub fn axis_translation(axis: [f64; 3], distance: f64) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-12 {
            return Self::identity();
        }
        let mut out = Self::identity();
        for (t, a) in out.translation.iter_mut().zip(axis) {
            *t = a / norm * distance;
        }
        out
    }

    /// The rotor as a unit quaternion `[w, x, y, z]`.
    pub fn quaternion(&self) -> [f64; 4] {
        // e2e3 -> x, e1e3 -> y (with the sign the rotor convention above puts there), e1e2 -> z
        [
            self.rotor[0],
            -self.rotor[6],
            self.rotor[5],
            -self.rotor[3],
        ]
    }
}
