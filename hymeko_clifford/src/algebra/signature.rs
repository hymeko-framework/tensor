//! Metric signature $(p, q, r)$ for the Clifford algebra $Cl(p, q, r)$.

/// $(p, q, r)$ metric signature.
///
/// In index order, basis vectors $e_1, \dots, e_p$ square to $+1$; the next $q$ square to $-1$; the final
/// $r$ are **degenerate** (null), squaring to $0$. The degenerate sector ($r > 0$) enables **projective /
/// plane-based geometric algebra** (PGA, e.g. $Cl(3,0,1)$) where motors unify rotation and translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signature {
    /// Number of positive-square basis vectors.
    pub p: usize,
    /// Number of negative-square basis vectors.
    pub q: usize,
    /// Number of degenerate (null, $e_i^2 = 0$) basis vectors.
    pub r: usize,
}

impl Signature {
    /// Construct a Euclidean signature $Cl(n, 0)$ — every basis vector
    /// squares to $+1$.
    pub const fn euclidean(n: usize) -> Self {
        Self { p: n, q: 0, r: 0 }
    }

    /// Construct a Lorentzian signature $Cl(p, q)$.
    pub const fn lorentzian(p: usize, q: usize) -> Self {
        Self { p, q, r: 0 }
    }

    /// Construct a degenerate signature $Cl(p, q, r)$ with `r` null basis vectors last.
    pub const fn degenerate(p: usize, q: usize, r: usize) -> Self {
        Self { p, q, r }
    }

    /// 3D Projective Geometric Algebra $Cl(3, 0, 1)$: $e_1, e_2, e_3$ square to $+1$, the null $e_0$ (the
    /// 4th basis vector here) squares to $0$. Even-grade elements are **motors** (screw motions).
    pub const fn pga3() -> Self {
        Self { p: 3, q: 0, r: 1 }
    }

    /// Total dimension $n = p + q + r$.
    pub const fn n(&self) -> usize {
        self.p + self.q + self.r
    }

    /// Square of the basis vector $e_i$ (1-indexed): $+1$ for the first $p$, $-1$ for the next $q$, and
    /// $0$ for the final $r$ (degenerate) vectors.
    ///
    /// # Panics
    /// If `i` is zero or larger than `p + q + r`.
    pub fn basis_square(&self, i: usize) -> f64 {
        debug_assert!(i >= 1, "basis vectors are 1-indexed; got 0");
        debug_assert!(
            i <= self.n(),
            "basis index {} out of range for signature ({}, {}, {})",
            i,
            self.p,
            self.q,
            self.r
        );
        if i <= self.p {
            1.0
        } else if i <= self.p + self.q {
            -1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_squares_to_plus_one() {
        let s = Signature::euclidean(4);
        for i in 1..=4 {
            assert_eq!(s.basis_square(i), 1.0);
        }
    }

    #[test]
    fn lorentzian_3_1() {
        let s = Signature::lorentzian(3, 1);
        assert_eq!(s.basis_square(1), 1.0);
        assert_eq!(s.basis_square(2), 1.0);
        assert_eq!(s.basis_square(3), 1.0);
        assert_eq!(s.basis_square(4), -1.0);
    }

    #[test]
    fn pga3_has_null_fourth_basis() {
        let s = Signature::pga3();
        assert_eq!(s.n(), 4);
        assert_eq!(s.basis_square(1), 1.0);
        assert_eq!(s.basis_square(2), 1.0);
        assert_eq!(s.basis_square(3), 1.0);
        assert_eq!(s.basis_square(4), 0.0); // degenerate e0
    }

    #[test]
    fn degenerate_mixes_all_three_sectors() {
        let s = Signature::degenerate(1, 1, 1); // Cl(1,1,1)
        assert_eq!(s.n(), 3);
        assert_eq!(s.basis_square(1), 1.0);
        assert_eq!(s.basis_square(2), -1.0);
        assert_eq!(s.basis_square(3), 0.0);
    }
}
