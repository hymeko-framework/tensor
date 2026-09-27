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

use super::multivector::Multivector;
use super::se::Se;
#[cfg(test)]
use super::signature::Signature;

/// A Euclidean dimension, and the fixed-size storage its rigid motions need.
///
/// Implementors are markers: they carry no data, only the two array types and $n$.
pub trait EuclideanDim: Copy + 'static {
    /// The Euclidean dimension $n$.
    const N: usize;
    /// Blade masks of the **even** subalgebra, ascending.
    ///
    /// A rotor lives here and nowhere else: it is a product of an even number of vectors, so every
    /// odd-grade coefficient of one is structurally zero. Naming the set lets the products below
    /// iterate the coefficients a rotor can actually have rather than all `2^n` of them and
    /// discovering the zeros at runtime.
    const EVEN: &'static [usize];
    /// Blade masks of the **odd** subalgebra, ascending.
    ///
    /// Where a vector lives, and where a rotor sandwich passes through: `even * odd` is odd, and
    /// `odd * even` is odd again, so the intermediate of a rotation is confined to this set.
    const ODD: &'static [usize];
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
    // Cl(2): the scalar and e1e2 are even; e1 and e2 are odd. The even part is the complex numbers.
    const EVEN: &'static [usize] = &[0, 3];
    const ODD: &'static [usize] = &[1, 2];
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
    // Cl(3): the scalar and the three bivectors are even, and that four-dimensional even part IS the
    // quaternions. The three vectors and the pseudoscalar are odd.
    const EVEN: &'static [usize] = &[0, 3, 5, 6];
    const ODD: &'static [usize] = &[1, 2, 4, 7];
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

/// `canonical_reorder_sign(a, b)` for every pair of blades, precomputed at compile time.
///
/// # Why this is exact and not an approximation of the general function
///
/// [`blade_product`] answers two questions: which blade the product lands on, and with what sign.
/// The first is `a ^ b` and always was. The second is the parity of the reordering, multiplied by
/// the square of each shared basis vector — and this type's signature is
/// [`Signature::euclidean`], where **every** basis vector squares to `+1`. So that loop multiplies
/// the sign by one, as many times as there are shared bits, and removing it changes nothing:
/// multiplication by `1.0` is exact.
///
/// What is left is a function of two bit patterns and nothing else, so it is a table. The
/// arithmetic in [`SeFixed::geo`] is untouched — same loop, same order, same accumulation — and only
/// the way `(index, sign)` is obtained has changed. That is what makes the bit-for-bit contract with
/// `Se` survive it, and `tests/se_fixed.rs` is what checks that claim rather than trusting it.
///
/// # Why it was worth doing
///
/// The sign was being recomputed for every term of every product: two bit loops with a
/// `leading_zeros`, a mask and a `count_ones` each. A rotor product is sixteen terms and a sandwich
/// is over forty, and forward kinematics over a thirty-three link mechanism is sixty-six products.
/// Measured, `compose` was 131 ns and a whole pose walk 5.3 us — the largest single cost in the
/// physics step, larger than the dynamics it was feeding.
///
/// Eight by eight covers `Cl(3)`; `Dim2` uses the leading four by four. The sign does not depend on
/// the dimension at all — only on the bit patterns — so one table serves both.
const REORDER: [[f64; 8]; 8] = build_reorder_table();

/// The parity of the reordering, as a `const fn` so the table above is built by the compiler.
const fn reorder_sign(a: usize, b: usize) -> f64 {
    let mut remaining = a;
    let mut inversions: u32 = 0;
    while remaining != 0 {
        let bit = usize::BITS as usize - 1 - remaining.leading_zeros() as usize;
        inversions += (b & ((1usize << bit) - 1)).count_ones();
        remaining &= !(1usize << bit);
    }
    if inversions.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

const fn build_reorder_table() -> [[f64; 8]; 8] {
    let mut table = [[0.0f64; 8]; 8];
    let mut a = 0usize;
    while a < 8 {
        let mut b = 0usize;
        while b < 8 {
            table[a][b] = reorder_sign(a, b);
            b += 1;
        }
        a += 1;
    }
    table
}

impl<D: EuclideanDim> SeFixed<D> {
    #[cfg(test)]
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
    /// The general geometric product over every blade.
    ///
    /// Kept as the REFERENCE the restricted form is checked against, not because anything on the hot
    /// path calls it: `geo_over_agrees_with_the_general_product` in this module's tests runs both on
    /// random rotors and vectors and demands the same bits. Without it, "the same multiply-adds in
    /// the same order" would be a claim in a comment.
    #[cfg(test)]
    fn geo(a: &D::Blades, b: &D::Blades) -> D::Blades {
        let sig = Self::signature();
        let (a, b) = (a.as_ref(), b.as_ref());
        let mut out = D::zero_blades();
        let o = out.as_mut();
        // The index IS the blade bitmask, which is why it is enumerated rather than iterated over:
        // `blade_product` needs the mask, not the coefficient.
        let _ = sig;
        for (a_idx, &av) in a.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            for (b_idx, &bv) in b.iter().enumerate() {
                if bv == 0.0 {
                    continue;
                }
                // The blade index is the exclusive-or of the masks and the sign is the parity of the
                // reordering — both looked up rather than recomputed. See [`REORDER`].
                o[a_idx ^ b_idx] += REORDER[a_idx][b_idx] * av * bv;
            }
        }
        out
    }

    /// The same product, over stated blade sets instead of over everything.
    ///
    /// # Why this is not an optimisation of the arithmetic
    ///
    /// It performs **exactly** the multiply-accumulates [`Self::geo`] performs, in exactly the same
    /// order. What changes is which coefficients are looked at: a rotor's odd coefficients are
    /// structurally zero, so the general loop loads them, compares them to zero and moves on, and it
    /// does that forty-eight times out of sixty-four to reach sixteen products.
    ///
    /// The zero test stays. A coefficient inside the named set can still be zero — an identity rotor
    /// has three — and skipping it there is what the general loop already did.
    ///
    /// # Preconditions
    /// Every coefficient of `a` outside `a_blades`, and of `b` outside `b_blades`, must be zero.
    /// Nothing checks it; the sets come from [`EuclideanDim`] and the callers are the group law.
    fn geo_over(a: &D::Blades, a_blades: &[usize], b: &D::Blades, b_blades: &[usize]) -> D::Blades {
        let (a, b) = (a.as_ref(), b.as_ref());
        let mut out = D::zero_blades();
        let o = out.as_mut();
        for &a_idx in a_blades {
            let av = a[a_idx];
            if av == 0.0 {
                continue;
            }
            for &b_idx in b_blades {
                let bv = b[b_idx];
                if bv == 0.0 {
                    continue;
                }
                o[a_idx ^ b_idx] += REORDER[a_idx][b_idx] * av * bv;
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
        // rotor (even) * vector (odd) is odd; that times the reversed rotor (even) is odd again, and
        // grade one of it is the answer. Three grade sets, none of them "all of them".
        let half = Self::geo_over(&self.rotor, D::EVEN, &mv, D::ODD);
        let product = Self::geo_over(&half, D::ODD, &Self::reverse(&self.rotor), D::EVEN);
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
            let (out, m, t) = (
                translation.as_mut(),
                moved.as_ref(),
                self.translation.as_ref(),
            );
            for i in 0..D::N {
                out[i] = m[i] + t[i];
            }
        }
        Self {
            rotor: Self::geo_over(&self.rotor, D::EVEN, &rhs.rotor, D::EVEN),
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
        [self.rotor[0], -self.rotor[6], self.rotor[5], -self.rotor[3]]
    }
}

#[cfg(test)]
mod grade_tests {
    use super::*;

    fn rng(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        2.0 * ((*state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }

    /// The restricted product against the general one, bit for bit.
    ///
    /// `geo_over` claims to perform exactly the multiply-accumulates `geo` performs, in exactly the
    /// same order, differing only in which coefficients it bothers to look at. That is a claim about
    /// the last bit, so it is checked as one.
    #[test]
    fn geo_over_agrees_with_the_general_product() {
        let mut state = 0x6EED_u64;
        for case in 0..500 {
            let x = Se3::rpy_xyz(
                [
                    rng(&mut state) * 3.2,
                    rng(&mut state) * 3.2,
                    rng(&mut state) * 3.2,
                ],
                [rng(&mut state), rng(&mut state), rng(&mut state)],
            );
            let y = Se3::rpy_xyz(
                [
                    rng(&mut state) * 3.2,
                    rng(&mut state) * 3.2,
                    rng(&mut state) * 3.2,
                ],
                [rng(&mut state), rng(&mut state), rng(&mut state)],
            );

            // even * even, which is what `compose` does
            assert_eq!(
                SeFixed::<Dim3>::geo_over(&x.rotor, Dim3::EVEN, &y.rotor, Dim3::EVEN),
                SeFixed::<Dim3>::geo(&x.rotor, &y.rotor),
                "case {case}: even times even"
            );

            // even * odd and odd * even, which is what `rotate` does
            let mut v = [0.0f64; 8];
            v[1] = rng(&mut state);
            v[2] = rng(&mut state);
            v[4] = rng(&mut state);
            let half_fast = SeFixed::<Dim3>::geo_over(&x.rotor, Dim3::EVEN, &v, Dim3::ODD);
            let half_slow = SeFixed::<Dim3>::geo(&x.rotor, &v);
            assert_eq!(half_fast, half_slow, "case {case}: even times odd");

            let back = SeFixed::<Dim3>::reverse(&x.rotor);
            assert_eq!(
                SeFixed::<Dim3>::geo_over(&half_fast, Dim3::ODD, &back, Dim3::EVEN),
                SeFixed::<Dim3>::geo(&half_slow, &back),
                "case {case}: odd times even"
            );
        }
    }

    /// A rotor really does live in the even subalgebra, and a vector in the odd one.
    ///
    /// The precondition `geo_over` relies on, checked rather than assumed — if a constructor ever
    /// put a coefficient outside its grade set, the restricted product would silently drop it.
    #[test]
    fn the_grade_sets_hold_everything_the_constructors_produce() {
        let mut state = 0x9AB_u64;
        for case in 0..500 {
            let x = Se3::rpy_xyz(
                [
                    rng(&mut state) * 3.2,
                    rng(&mut state) * 3.2,
                    rng(&mut state) * 3.2,
                ],
                [rng(&mut state), rng(&mut state), rng(&mut state)],
            );
            for (i, &c) in x.rotor.iter().enumerate() {
                if !Dim3::EVEN.contains(&i) {
                    assert_eq!(
                        c, 0.0,
                        "case {case}: rotor has an odd component at blade {i}"
                    );
                }
            }
            let axis = [rng(&mut state), rng(&mut state), rng(&mut state)];
            let y = Se3::axis_angle(axis, rng(&mut state) * 6.0);
            for (i, &c) in y.rotor.iter().enumerate() {
                if !Dim3::EVEN.contains(&i) {
                    assert_eq!(c, 0.0, "case {case}: axis_angle produced blade {i}");
                }
            }
        }
        assert_eq!(Dim3::EVEN.len() + Dim3::ODD.len(), 8);
        assert_eq!(Dim2::EVEN.len() + Dim2::ODD.len(), 4);
    }
}
