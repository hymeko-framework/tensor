//! Knuth MMIX linear-congruential RNG.
//!
//! Replaces three duplicated `lcg_next` / `lcg_next_in_range` free functions
//! from the legacy `hymeko_py/src/cycles.rs` (CLAUDE.md §6.5 #11: shared
//! random-number state passed explicitly, not module-level mutable state).
//!
//! `Lcg` is `Send` (state is owned) — one instance per rayon worker.

/// Knuth MMIX LCG state (`Send`); one instance per rayon worker, no globals.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed with Knuth's MMIX kickstart (matches the legacy ad-hoc init).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }

    /// Raw next state. Returns the post-step value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform `[0, n)` (modulo bias-free up to `n` close to `2^31`).
    #[inline]
    pub fn next_in_range(&mut self, n: u32) -> u32 {
        let r = self.next_u64() >> 33;
        (r as u32) % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_from_seed() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn range_is_bounded() {
        let mut r = Lcg::new(7);
        for _ in 0..1000 {
            assert!(r.next_in_range(13) < 13);
        }
    }
}
