//! The streamed HyMeKo IR format — a network's parameters laid out in a **single flat arena addressed by precomputed
//! offsets** (the "single flat arena" data-oriented pattern). This is the format only: it is deliberately decoupled
//! from any network type. A learner encodes its per-layer parameter blocks into the arena and decodes them back
//! bit-exactly; the learner ↔ stream bridge (the forward pass, the concrete net struct) stays with the learner.
//!
//! Layered-DAG specialization: each layer is one multi-input/multi-output hyperedge, and the incidence is the
//! implicit full cascade. Per layer `l` the arena block is `[w_l (n_i·n_o) | b_l (n_o) | c_l (n_o·cheby_order)]`,
//! where `dims = [d_in, W_1, …, W_L, d_out]` and `cheby_order` is the per-unit basis-coefficient count.

/// Per-layer parameter blocks `(w, b, c)`: `w[l]` (flat `n_i·n_o` weights), `b[l]` (`n_o` biases), `c[l]`
/// (`n_o·cheby_order` basis coefficients). The decoded / to-be-encoded form of a [`HyMeKoStream`].
pub type LayerParams = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// A streamed HyMeKo IR: the cold topology `dims`, the hot single flat parameter `arena`, and the `cheby_order`
/// that fixes the per-layer block layout.
#[derive(Clone, Debug, PartialEq)]
pub struct HyMeKoStream {
    pub dims: Vec<usize>,
    pub arena: Vec<f32>,
    pub cheby_order: usize,
}

impl HyMeKoStream {
    /// Per-layer arena block length `n_i·n_o + n_o + n_o·cheby_order` for layer `l` of `dims`.
    pub fn block_len(dims: &[usize], l: usize, cheby_order: usize) -> usize {
        let (ni, no) = (dims[l], dims[l + 1]);
        ni * no + no + no * cheby_order
    }

    /// Total arena length `Σ_l block_len(dims, l, cheby_order)`.
    pub fn arena_len(dims: &[usize], cheby_order: usize) -> usize {
        (0..dims.len().saturating_sub(1))
            .map(|l| Self::block_len(dims, l, cheby_order))
            .sum()
    }

    /// Encode per-layer parameter blocks `(w, b, c)` into a single flat arena.
    ///
    /// # Preconditions
    /// `dims.len() ≥ 2`; for each layer `l`: `w[l].len() == dims[l]·dims[l+1]`, `b[l].len() == dims[l+1]`,
    /// `c[l].len() == dims[l+1]·cheby_order`.
    /// # Postconditions
    /// `arena.len() == arena_len(dims, cheby_order)`; `decode()` reproduces `(w, b, c)` exactly.
    pub fn encode(
        dims: &[usize],
        w: &[Vec<f32>],
        b: &[Vec<f32>],
        c: &[Vec<f32>],
        cheby_order: usize,
    ) -> Self {
        let nl = dims.len() - 1;
        debug_assert_eq!(w.len(), nl);
        let mut arena = Vec::with_capacity(Self::arena_len(dims, cheby_order));
        for l in 0..nl {
            debug_assert_eq!(w[l].len(), dims[l] * dims[l + 1]);
            debug_assert_eq!(b[l].len(), dims[l + 1]);
            debug_assert_eq!(c[l].len(), dims[l + 1] * cheby_order);
            arena.extend_from_slice(&w[l]);
            arena.extend_from_slice(&b[l]);
            arena.extend_from_slice(&c[l]);
        }
        HyMeKoStream {
            dims: dims.to_vec(),
            arena,
            cheby_order,
        }
    }

    /// Decode the arena back into per-layer `(w, b, c)` blocks. Exact inverse of [`encode`](Self::encode).
    ///
    /// # Preconditions
    /// `arena.len() == arena_len(dims, cheby_order)`.
    pub fn decode(&self) -> LayerParams {
        let nl = self.dims.len() - 1;
        let (mut w, mut b, mut c) = (
            Vec::with_capacity(nl),
            Vec::with_capacity(nl),
            Vec::with_capacity(nl),
        );
        let mut off = 0usize;
        for l in 0..nl {
            let (ni, no) = (self.dims[l], self.dims[l + 1]);
            let (wl, bl, cl) = (ni * no, no, no * self.cheby_order);
            w.push(self.arena[off..off + wl].to_vec());
            b.push(self.arena[off + wl..off + wl + bl].to_vec());
            c.push(self.arena[off + wl + bl..off + wl + bl + cl].to_vec());
            off += wl + bl + cl;
        }
        debug_assert_eq!(
            off,
            self.arena.len(),
            "arena fully consumed by the dims-derived offsets"
        );
        (w, b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks(dims: &[usize], k: usize) -> LayerParams {
        let mut val = 0.0f32;
        let mut next = || {
            val += 1.0;
            val
        };
        let nl = dims.len() - 1;
        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for l in 0..nl {
            let (ni, no) = (dims[l], dims[l + 1]);
            w.push((0..ni * no).map(|_| next()).collect());
            b.push((0..no).map(|_| next()).collect());
            c.push((0..no * k).map(|_| next()).collect());
        }
        (w, b, c)
    }

    #[test]
    fn encode_decode_round_trips_exactly() {
        let dims = [3usize, 8, 5, 1];
        let k = 6;
        let (w, b, c) = blocks(&dims, k);
        let s = HyMeKoStream::encode(&dims, &w, &b, &c, k);
        assert_eq!(
            s.arena.len(),
            HyMeKoStream::arena_len(&dims, k),
            "arena length == Σ block lengths"
        );
        assert_eq!(s.dims, dims.to_vec());
        let (w2, b2, c2) = s.decode();
        assert_eq!(
            (w, b, c),
            (w2, b2, c2),
            "decode ∘ encode is the identity on (w,b,c)"
        );
    }

    #[test]
    fn block_len_matches_layout() {
        // block = w (ni·no) + b (no) + c (no·k)
        assert_eq!(HyMeKoStream::block_len(&[3, 8, 1], 0, 6), 3 * 8 + 8 + 8 * 6); // 24+8+48
        assert_eq!(HyMeKoStream::block_len(&[3, 8, 1], 1, 6), 8 + 1 + 6); // ni=8,no=1,k=6
    }
}
