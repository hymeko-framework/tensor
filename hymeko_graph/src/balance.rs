//! Pre-built [`CyclePruner`](crate::pruner::CyclePruner) instances
//! for signed-balance theory: Cartwright–Harary cycle balance
//! (1956), Davis weak balance (1967), and bipartite-only emission
//! (used to filter the star-expansion path).

use crate::pruner::{CyclePruner, PrunerDecision};

/// Emit only Cartwright–Harary balanced ($\prod s = +1$) cycles
/// when `mode = OnlyBalanced`, or only unbalanced when
/// `mode = OnlyUnbalanced`.
///
/// Balance is the classical sign-product test.  In an unsigned
/// graph all cycles are trivially balanced; in a signed graph
/// the fraction balanced is a known correlate of structural
/// stability (Cartwright–Harary 1956).
#[derive(Debug, Copy, Clone)]
pub struct CartwrightHararyPruner {
    /// What to keep.
    pub mode: BalanceMode,
}

/// Selection mode for [`CartwrightHararyPruner`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BalanceMode {
    /// Keep only balanced cycles (sign product $= +1$).
    OnlyBalanced,
    /// Keep only unbalanced cycles (sign product $= -1$).
    OnlyUnbalanced,
}

impl CyclePruner for CartwrightHararyPruner {
    #[inline]
    fn emit_ok(&self, _cycle: &[u32], edge_signs: &[i8]) -> PrunerDecision {
        let mut prod: i32 = 1;
        for &s in edge_signs {
            prod *= s as i32;
        }
        let balanced = prod > 0;
        let want_balanced = matches!(self.mode, BalanceMode::OnlyBalanced);
        PrunerDecision::from_bool(balanced == want_balanced)
    }
}

/// Davis 1967 weak balance: a signed graph is *weakly balanced*
/// iff it contains no triangle whose three edges are all negative.
/// This pruner emits all cycles **except** all-negative triangles
/// — a clean filter for the "no fully hostile triad" empirical
/// regularity in social-balance studies.
#[derive(Debug, Copy, Clone, Default)]
pub struct DavisWeakBalancePruner;

impl CyclePruner for DavisWeakBalancePruner {
    #[inline]
    fn emit_ok(&self, cycle: &[u32], edge_signs: &[i8]) -> PrunerDecision {
        // Only triangles can violate Davis weak balance.
        if cycle.len() != 3 {
            return PrunerDecision::Accept;
        }
        let all_neg = edge_signs.iter().all(|&s| s < 0);
        PrunerDecision::from_bool(!all_neg)
    }
}

/// Bipartite-only emission: keep cycles whose length is even.
///
/// Used after a *star expansion* of a hypergraph: the resulting
/// graph alternates original-vertex / centroid-vertex, so every
/// cycle has even length.  Odd-length cycles are structural
/// impossibilities, but a generic enumerator will still try to
/// build them; this pruner short-circuits at emit time.
///
/// For a stronger pre-check that prunes during the DFS, use
/// [`crate::friedler::FriedlerAxiomPruner`] which knows the
/// node-kind partition and refuses to extend across same-kind
/// vertices.
#[derive(Debug, Copy, Clone, Default)]
pub struct BipartiteOnlyPruner;

impl CyclePruner for BipartiteOnlyPruner {
    #[inline]
    fn emit_ok(&self, cycle: &[u32], _edge_signs: &[i8]) -> PrunerDecision {
        PrunerDecision::from_bool(cycle.len().is_multiple_of(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartwright_harary_balanced_keeps_two_negs() {
        // Triangle with edge signs (-, -, +): product = +1 → balanced.
        let p = CartwrightHararyPruner {
            mode: BalanceMode::OnlyBalanced,
        };
        assert!(p.emit_ok(&[0, 1, 2], &[-1, -1, 1]).is_accept());
        // (+, -, -): product = +1 → balanced too.
        assert!(p.emit_ok(&[0, 1, 2], &[1, -1, -1]).is_accept());
        // (-, +, +): product = -1 → unbalanced.
        assert!(!p.emit_ok(&[0, 1, 2], &[-1, 1, 1]).is_accept());
    }

    #[test]
    fn cartwright_harary_unbalanced_inverse() {
        let p = CartwrightHararyPruner {
            mode: BalanceMode::OnlyUnbalanced,
        };
        assert!(!p.emit_ok(&[0, 1, 2], &[-1, -1, 1]).is_accept());
        assert!(p.emit_ok(&[0, 1, 2], &[-1, 1, 1]).is_accept());
    }

    #[test]
    fn davis_weak_balance_excludes_all_negative_triangle() {
        let p = DavisWeakBalancePruner;
        assert!(!p.emit_ok(&[0, 1, 2], &[-1, -1, -1]).is_accept());
        // Two-neg triangle is fine (not all-negative).
        assert!(p.emit_ok(&[0, 1, 2], &[-1, -1, 1]).is_accept());
        // Quadrilateral is fine regardless of signs.
        assert!(p.emit_ok(&[0, 1, 2, 3], &[-1, -1, -1, -1]).is_accept());
    }

    #[test]
    fn bipartite_only_emits_even_cycles() {
        let p = BipartiteOnlyPruner;
        assert!(!p.emit_ok(&[0, 1, 2], &[1; 3]).is_accept());
        assert!(p.emit_ok(&[0, 1, 2, 3], &[1; 4]).is_accept());
        assert!(!p.emit_ok(&[0, 1, 2, 3, 4], &[1; 5]).is_accept());
        assert!(p.emit_ok(&[0, 1, 2, 3, 4, 5], &[1; 6]).is_accept());
    }
}
