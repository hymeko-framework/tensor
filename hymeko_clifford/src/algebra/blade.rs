//! Blade-bitmask arithmetic.
//!
//! A basis blade in $Cl(p, q)$ is identified with a bitmask
//! `idx ∈ [0, 2^N)`, where bit $i$ is set iff the basis vector
//! $e_{i+1}$ is a factor in the wedge product. The grade is
//! `popcount(idx)`. Multiplication of two basis blades is governed by
//! the *canonical reorder sign* — the parity of adjacent transpositions
//! needed to sort the combined factor sequence in increasing order —
//! plus a metric contribution from the [`Signature`].
//!
//! [`canonical_reorder_sign`] is the single most error-prone primitive
//! in the crate. It is exercised by an exhaustive test on all
//! $2^N \times 2^N$ pairs for $N = 4$ and against a brute-force
//! reference for $N = 5$.

use super::Signature;

/// Sign of the canonical reordering needed to merge two ascending
/// factor sequences encoded as bitmasks `a` and `b`.
///
/// Algorithm: walk `a` from highest to lowest bit; for each bit set in
/// `a` at position `i`, count how many bits of `b` lie strictly below
/// `i` — each such bit contributes one transposition. Total parity
/// determines the sign.
///
/// Returns `+1.0` for an even number of transpositions, `-1.0` for
/// odd. Bits in both `a` and `b` are NOT skipped — they are kept to
/// pass to the metric in [`blade_product`]. This function only
/// computes the parity, not the metric, so contiguous duplicates are
/// counted as transpositions; the metric contributes its own sign per
/// duplicated factor.
pub fn canonical_reorder_sign(a: usize, b: usize) -> f64 {
    // For each bit set in `a` (from highest to lowest), count bits in
    // `b` that are strictly below it. Each such pair is one inversion.
    let mut a_remaining = a;
    let mut inversions: u32 = 0;
    while a_remaining != 0 {
        // Highest set bit of `a_remaining`.
        let bit_pos = 63 - a_remaining.leading_zeros() as usize;
        // Count bits of `b` strictly below `bit_pos`.
        let mask_below = (1usize << bit_pos) - 1;
        inversions += (b & mask_below).count_ones();
        // Clear that bit.
        a_remaining &= !(1usize << bit_pos);
    }
    if inversions.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Brute-force reference implementation: build the factor sequences
/// explicitly and count adjacent swaps to bubble-sort. Used only by
/// tests to validate [`canonical_reorder_sign`].
#[cfg(test)]
fn brute_canonical_sign(a: usize, b: usize) -> f64 {
    let mut factors: Vec<usize> = Vec::new();
    for i in 0..(usize::BITS as usize) {
        if (a >> i) & 1 == 1 {
            factors.push(i);
        }
    }
    for i in 0..(usize::BITS as usize) {
        if (b >> i) & 1 == 1 {
            factors.push(i);
        }
    }
    // Bubble-sort, counting swaps.
    let mut swaps: u32 = 0;
    let n = factors.len();
    for i in 0..n {
        for j in 0..n - i - 1 {
            if factors[j] > factors[j + 1] {
                factors.swap(j, j + 1);
                swaps += 1;
            }
        }
    }
    if swaps.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// Product of two basis blades represented as bitmasks.
///
/// Returns `(result_idx, sign)` where:
/// - `result_idx = a ^ b` (factors that appear in both `a` and `b`
///   cancel via the basis-vector-square rule);
/// - `sign` is the product of the canonical reorder sign and the
///   metric contribution from each shared factor (each $e_i^2$
///   contributes $\pm 1$ per [`Signature::basis_square`]).
pub fn blade_product(a: usize, b: usize, sig: &Signature) -> (usize, f64) {
    let result_idx = a ^ b;
    let mut sign = canonical_reorder_sign(a, b);
    let shared = a & b;
    let mut s = shared;
    while s != 0 {
        let bit_pos = s.trailing_zeros() as usize;
        // Basis vectors are 1-indexed: bit 0 → e_1, bit 1 → e_2, …
        let i = bit_pos + 1;
        sign *= sig.basis_square(i);
        s &= s - 1; // clear the lowest set bit
    }
    (result_idx, sign)
}

/// Convenience: grade of a blade = number of factors in the wedge.
#[allow(dead_code)]
// Public API helper; not all internal call sites use it yet.
pub fn grade(idx: usize) -> u32 {
    idx.count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical-reorder-sign agrees with the brute-force reference
    /// across every (a, b) pair for N = 4 (256 × 256 = 65,536 pairs).
    #[test]
    fn canonical_reorder_exhaustive_n4() {
        let n = 4;
        let bound = 1usize << n;
        for a in 0..bound {
            for b in 0..bound {
                let s_fast = canonical_reorder_sign(a, b);
                let s_brute = brute_canonical_sign(a, b);
                assert_eq!(
                    s_fast, s_brute,
                    "mismatch at a=0b{:04b} b=0b{:04b}: fast={} brute={}",
                    a, b, s_fast, s_brute
                );
            }
        }
    }

    /// Same exhaustive sweep at N = 5 (1024 × 1024 = ~1M pairs); still
    /// well below the 60-second test budget for a release-mode test.
    #[test]
    fn canonical_reorder_exhaustive_n5() {
        let bound = 1usize << 5;
        for a in 0..bound {
            for b in 0..bound {
                let s_fast = canonical_reorder_sign(a, b);
                let s_brute = brute_canonical_sign(a, b);
                assert_eq!(s_fast, s_brute, "mismatch at a={} b={}", a, b);
            }
        }
    }

    /// $e_i e_i = +1$ in Euclidean signature.
    #[test]
    fn basis_squares_euclidean() {
        let sig = Signature::euclidean(4);
        for i in 0..4 {
            let blade_i = 1usize << i;
            let (idx, sign) = blade_product(blade_i, blade_i, &sig);
            assert_eq!(idx, 0, "e_{}^2 should reduce to scalar", i + 1);
            assert_eq!(sign, 1.0, "e_{}^2 sign", i + 1);
        }
    }

    /// $e_i e_i = -1$ for negative-square basis vectors.
    #[test]
    fn basis_squares_negative() {
        let sig = Signature::lorentzian(3, 1); // e_4^2 = -1
        let blade_4 = 1usize << 3;
        let (idx, sign) = blade_product(blade_4, blade_4, &sig);
        assert_eq!(idx, 0);
        assert_eq!(sign, -1.0);
    }

    /// Anticommutativity: $e_i e_j = -e_j e_i$ for $i \neq j$.
    #[test]
    fn anticommutativity() {
        let sig = Signature::euclidean(4);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                let bi = 1usize << i;
                let bj = 1usize << j;
                let (idx_ij, sign_ij) = blade_product(bi, bj, &sig);
                let (idx_ji, sign_ji) = blade_product(bj, bi, &sig);
                assert_eq!(
                    idx_ij,
                    idx_ji,
                    "e_{} e_{} and e_{} e_{} differ in result idx",
                    i + 1,
                    j + 1,
                    j + 1,
                    i + 1
                );
                assert_eq!(
                    sign_ij,
                    -sign_ji,
                    "anticommutativity violated for e_{} e_{}",
                    i + 1,
                    j + 1
                );
            }
        }
    }

    /// $(e_1 \wedge e_2)(e_2 \wedge e_3) = e_1 \wedge e_3$ in
    /// $Cl(4, 0)$. With factor sequence `[1, 2, 2, 3]` already in
    /// non-decreasing order, `canonical_reorder_sign` returns $+1$;
    /// the contracted $e_2^2 = +1$ in Euclidean signature contributes
    /// nothing further, so the overall sign is $+1$.
    #[test]
    fn grade2_grade2_product() {
        let sig = Signature::euclidean(4);
        let e12 = 0b0011usize;
        let e23 = 0b0110usize;
        let (idx, sign) = blade_product(e12, e23, &sig);
        assert_eq!(idx, 0b0101);
        assert_eq!(sign, 1.0);
    }

    /// $(e_1 \wedge e_2)(e_1 \wedge e_3) = -e_2 \wedge e_3$ in
    /// $Cl(4, 0)$. Useful as a *failure-detecting* sign test: the e_1
    /// factors cross over before contracting, so the canonical-reorder
    /// path must produce a $-1$ sign.
    #[test]
    fn grade2_grade2_negative_sign() {
        let sig = Signature::euclidean(4);
        let e12 = 0b0011usize;
        let e13 = 0b0101usize;
        let (idx, sign) = blade_product(e12, e13, &sig);
        assert_eq!(idx, 0b0110); // e_2 ∧ e_3
        assert_eq!(sign, -1.0);
    }
}
