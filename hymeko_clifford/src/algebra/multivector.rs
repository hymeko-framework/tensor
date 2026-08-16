//! Dense multivector representation.
//!
//! A [`Multivector`] is a $2^N$-component vector indexed by basis
//! blades. Component $i$ is the coefficient of the basis blade whose
//! bitmask is $i$. Storage is heap-allocated (`Vec<f64>` of length
//! `1 << n`); a stack-allocated stable-Rust generic-const variant can
//! be added once `generic_const_exprs` stabilises. This is fine for
//! the typical G-SPHF regime $N \leq 12$ but a sparse variant is
//! planned in Phase 5.

use super::{Signature, blade_product};

/// Dense multivector in $Cl(p, q)$ with $n = p + q$.
///
/// `components[i]` is the coefficient of the basis blade whose bitmask
/// is `i`. The trivial scalar lives at `components[0]`; the
/// pseudoscalar at `components[(1 << n) - 1]`. The dimension `n` is
/// stored runtime-side; consistency between operands is checked.
#[derive(Debug, Clone, PartialEq)]
pub struct Multivector {
    /// Number of basis vectors. Storage is `1 << n` `f64`s.
    pub n: usize,
    /// Coefficients indexed by blade bitmask.
    pub components: Vec<f64>,
}

impl Multivector {
    /// The zero multivector in $Cl(p, q)$ where $n = p + q$.
    pub fn zero(n: usize) -> Self {
        assert!(
            n <= 16,
            "Multivector::zero: n = {} exceeds the safe upper bound (16)",
            n
        );
        Self {
            n,
            components: vec![0.0; 1 << n],
        }
    }

    /// The scalar `1`.
    pub fn one(n: usize) -> Self {
        let mut mv = Self::zero(n);
        mv.components[0] = 1.0;
        mv
    }

    /// Scalar value (grade-0 component).
    pub fn scalar(value: f64, n: usize) -> Self {
        let mut mv = Self::zero(n);
        mv.components[0] = value;
        mv
    }

    /// The basis vector $e_i$ (1-indexed) in $Cl(\cdot)$ with $n$
    /// basis vectors total.
    ///
    /// # Panics
    /// If `i == 0` or `i > n`.
    pub fn basis_vector(i: usize, n: usize) -> Self {
        assert!(i >= 1 && i <= n, "basis vector index out of range");
        let mut mv = Self::zero(n);
        mv.components[1 << (i - 1)] = 1.0;
        mv
    }

    /// Geometric product `self * rhs` under the given signature.
    ///
    /// Each pair of nonzero components contributes a single
    /// [`blade_product`] call. Total cost is $O(2^{2n})$ — fine for
    /// $n \leq 12$.
    pub fn geo(&self, rhs: &Self, sig: &Signature) -> Self {
        debug_assert_eq!(self.n, rhs.n, "Multivector::geo: dimension mismatch");
        let n = self.n;
        let mut out = Self::zero(n);
        for a_idx in 0..(1usize << n) {
            let a = self.components[a_idx];
            if a == 0.0 {
                continue;
            }
            for b_idx in 0..(1usize << n) {
                let b = rhs.components[b_idx];
                if b == 0.0 {
                    continue;
                }
                let (r_idx, sign) = blade_product(a_idx, b_idx, sig);
                out.components[r_idx] += sign * a * b;
            }
        }
        out
    }

    /// Sum of two multivectors.
    pub fn add(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.n, rhs.n, "Multivector::add: dimension mismatch");
        let n = self.n;
        let mut out = Self::zero(n);
        for i in 0..(1usize << n) {
            out.components[i] = self.components[i] + rhs.components[i];
        }
        out
    }

    /// Componentwise scalar multiplication.
    pub fn scale(&self, s: f64) -> Self {
        let n = self.n;
        let mut out = Self::zero(n);
        for i in 0..(1usize << n) {
            out.components[i] = s * self.components[i];
        }
        out
    }

    /// Negation.
    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Grade projection $\langle A \rangle_k$: keep only components
    /// whose blade has popcount equal to `k`, zero the rest.
    pub fn grade_proj(&self, k: u32) -> Self {
        let n = self.n;
        let mut out = Self::zero(n);
        for i in 0..(1usize << n) {
            if i.count_ones() == k {
                out.components[i] = self.components[i];
            }
        }
        out
    }

    /// Reverse $\tilde{A}$: reverses the factor order in every blade.
    /// For grade-$k$ components this multiplies by $(-1)^{k(k-1)/2}$.
    pub fn reverse(&self) -> Self {
        let n = self.n;
        let mut out = Self::zero(n);
        for i in 0..(1usize << n) {
            let k = i.count_ones();
            let sign = if (k * (k.wrapping_sub(1)) / 2).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            out.components[i] = sign * self.components[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_one_geo_anything_is_anything() {
        let sig = Signature::euclidean(3);
        let one = Multivector::one(3);
        let v = Multivector::basis_vector(2, 3);
        let p = one.geo(&v, &sig);
        assert_eq!(p.components, v.components);
        let q = v.geo(&one, &sig);
        assert_eq!(q.components, v.components);
    }

    #[test]
    fn anticommutator_zero() {
        let sig = Signature::euclidean(3);
        let e1 = Multivector::basis_vector(1, 3);
        let e2 = Multivector::basis_vector(2, 3);
        let ab = e1.geo(&e2, &sig);
        let ba = e2.geo(&e1, &sig);
        let sum = ab.add(&ba);
        assert!(sum.components.iter().all(|c| c.abs() < 1e-12));
    }

    #[test]
    fn reverse_grade3_negates() {
        let mut g3 = Multivector::zero(3);
        g3.components[0b111] = 1.0;
        let r = g3.reverse();
        assert_eq!(r.components[0b111], -1.0);
    }

    #[test]
    fn grade_proj_partition() {
        let mut mv = Multivector::zero(3);
        mv.components[0b000] = 1.0;
        mv.components[0b001] = 2.0;
        mv.components[0b011] = 3.0;
        mv.components[0b111] = 4.0;
        let p0 = mv.grade_proj(0);
        let p1 = mv.grade_proj(1);
        let p2 = mv.grade_proj(2);
        let p3 = mv.grade_proj(3);
        let recovered = p0.add(&p1).add(&p2).add(&p3);
        assert_eq!(recovered.components, mv.components);
    }
}
