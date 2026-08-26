//! Signed hypergraph incidence as a **linear operator**, and its star→clique contraction.
//!
//! # What this adds that the crate did not have
//!
//! [`crate::incidence`] builds an edge→cycle (line-graph) incidence matrix — a different object.
//! The *star expansion* elsewhere in HyMeKo means the bipartite vertex↔hyperedge representation:
//! members stay attached to their hyperedge and messages pass between the two sides. What was
//! missing is the **contraction of the hyperedge index**, which turns that bipartite object into a
//! dense node–node operator in which every hyperedge becomes a clique over its members.
//!
//! (`junction_tree::star_clique_tree` in the consuming application is unrelated: it builds a
//! star-shaped *clique tree topology*, not this contraction.)
//!
//! # The contract, stated exactly
//!
//! For a signed incidence with per-member sign `σ`, per-node structural scale `s_v` and uniform
//! arity `k`, write the node→hyperedge map `A` and its adjoint. Then, with `ρ` the ridge:
//!
//! ```text
//! (A x)_c   = Σ_{i∈c} σ_{c,i} · s_{v(c,i)} · (1/k) · x_{v(c,i)}          … node_to_edge
//! (E h)_v   = s_v · Σ_{c ∋ v} σ_{c,i} · h_c            where  E = k · Aᵀ … edge_to_node
//! star matvec = E(A x) + ρ x  =  k · Aᵀ(A x) + ρ x     … the operator is NEVER formed
//! clique      = k · AᵀA + ρ I                          … the hyperedge index contracted out
//! rhs         = E y  =  k · Aᵀ y
//! ```
//!
//! **`edge_to_node` is `k · Aᵀ`, not `Aᵀ`.** `A` carries the `1/k` and the edge→node direction does
//! not, so the composition already supplies the `k ·` of `k · AᵀA` without a separate multiply. The
//! exact adjoint identity is therefore the *scaled* one,
//! `k · ⟨A x, h⟩ == ⟨x, E h⟩`, which [`SignedHyperIncidence::edge_to_node`] documents and the tests
//! assert. This mirrors the historical `hg_message` pair exactly; writing `(Aᵀ h)` for the edge→node
//! op would be off by `k`.
//!
//! # Accumulation order is part of the contract
//!
//! Both directions accumulate **edge-major, then member-in-declaration-order**, and the adjoint
//! applies `s_v` **after** summing, not per term. A consumer reproducing a recorded number depends
//! on this; changing it is a numerical change, not a refactor.

/// Why a signed hypergraph is not well formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperIncidenceError {
    /// `arity` was zero — a hyperedge must have at least one member.
    ZeroArity,
    /// `members.len()` is not `n_edges * arity`, or `signs.len()` disagrees with `members.len()`.
    ShapeMismatch,
    /// `scale.len() != n_nodes`.
    ScaleLengthMismatch,
    /// A member index is `>= n_nodes`.
    MemberOutOfRange {
        /// Position in the flat `members` table.
        at: usize,
        /// The offending node index.
        node: u32,
    },
}

impl core::fmt::Display for HyperIncidenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HyperIncidenceError::ZeroArity => write!(f, "arity must be at least 1"),
            HyperIncidenceError::ShapeMismatch => {
                write!(f, "members/signs length must be n_edges * arity")
            }
            HyperIncidenceError::ScaleLengthMismatch => {
                write!(f, "scale must have one entry per node")
            }
            HyperIncidenceError::MemberOutOfRange { at, node } => {
                write!(
                    f,
                    "member {at} references node {node}, which is out of range"
                )
            }
        }
    }
}

impl std::error::Error for HyperIncidenceError {}

/// A signed hypergraph of **uniform arity**, viewed as a linear operator on node values.
///
/// Uniform arity is an explicit contract, not an oversight: the flat `members`/`signs` layout is
/// `n_edges * arity`, and every accessor assumes it. A ragged hypergraph needs a different
/// representation and is out of scope here.
///
/// Duplicate members within one hyperedge are **permitted and meaningful**: each occurrence
/// contributes its own term, so a node appearing twice in a hyperedge is weighted twice. A self-pair
/// (the same node twice) therefore lands on the diagonal of the clique matrix. Nothing is
/// deduplicated.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedHyperIncidence {
    n_nodes: usize,
    n_edges: usize,
    arity: usize,
    members: Vec<u32>,
    signs: Vec<f32>,
    scale: Vec<f32>,
}

impl SignedHyperIncidence {
    /// Validate and build.
    ///
    /// # Errors
    /// See [`HyperIncidenceError`]. Every member index is range-checked.
    pub fn new(
        n_nodes: usize,
        arity: usize,
        members: Vec<u32>,
        signs: Vec<f32>,
        scale: Vec<f32>,
    ) -> Result<Self, HyperIncidenceError> {
        if arity == 0 {
            return Err(HyperIncidenceError::ZeroArity);
        }
        if members.len() % arity != 0 || signs.len() != members.len() {
            return Err(HyperIncidenceError::ShapeMismatch);
        }
        if scale.len() != n_nodes {
            return Err(HyperIncidenceError::ScaleLengthMismatch);
        }
        for (at, &node) in members.iter().enumerate() {
            if node as usize >= n_nodes {
                return Err(HyperIncidenceError::MemberOutOfRange { at, node });
            }
        }
        let n_edges = members.len() / arity;
        Ok(SignedHyperIncidence {
            n_nodes,
            n_edges,
            arity,
            members,
            signs,
            scale,
        })
    }

    /// Node count.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }
    /// Hyperedge count.
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }
    /// Members per hyperedge.
    pub fn arity(&self) -> usize {
        self.arity
    }
    /// Flat `n_edges * arity` member table.
    pub fn members(&self) -> &[u32] {
        &self.members
    }
    /// Flat `n_edges * arity` sign table.
    pub fn signs(&self) -> &[f32] {
        &self.signs
    }
    /// Per-node structural scale.
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// `A x` — the signed node→hyperedge contraction, one scalar per hyperedge.
    ///
    /// # Preconditions
    /// `x.len() == n_nodes`.
    /// # Postconditions
    /// Length `n_edges`. Accumulated edge-major, members in declaration order.
    pub fn node_to_edge(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n_nodes, "x must have one value per node");
        let inv_k = 1.0 / self.arity as f32;
        let mut h = vec![0.0f32; self.n_edges];
        for c in 0..self.n_edges {
            let mut acc = 0.0f32;
            for i in 0..self.arity {
                let v = self.members[c * self.arity + i] as usize;
                acc += self.signs[c * self.arity + i] * self.scale[v] * inv_k * x[v];
            }
            h[c] = acc;
        }
        h
    }

    /// `E h = k · Aᵀ h` — the hyperedge→node direction.
    ///
    /// **Scaled adjoint, by design.** `A` (see [`Self::node_to_edge`]) carries the `1/k`; this
    /// direction does not, so `k · ⟨A x, h⟩ == ⟨x, E h⟩`. The composition `E ∘ A` is therefore
    /// already `k · AᵀA`.
    ///
    /// # Preconditions
    /// `h.len() == n_edges`.
    /// # Postconditions
    /// Length `n_nodes`. Accumulated edge-major with the per-node scale applied **after** summing —
    /// part of the numerical contract, not an implementation detail.
    pub fn edge_to_node(&self, h: &[f32]) -> Vec<f32> {
        assert_eq!(h.len(), self.n_edges, "h must have one value per hyperedge");
        let mut out = vec![0.0f32; self.n_nodes];
        for c in 0..self.n_edges {
            let hc = h[c];
            for i in 0..self.arity {
                let v = self.members[c * self.arity + i] as usize;
                out[v] += self.signs[c * self.arity + i] * hc;
            }
        }
        for v in 0..self.n_nodes {
            out[v] *= self.scale[v];
        }
        out
    }

    /// The **star** normal operator applied matrix-free: `E(A x) + ρ x = k · Aᵀ(A x) + ρ x`.
    ///
    /// Nothing is assembled — this is two message-passing half-steps.
    ///
    /// # Preconditions
    /// `x.len() == n_nodes`; `ridge >= 0` for the result to stay positive semi-definite.
    pub fn star_matvec(&self, x: &[f32], ridge: f32) -> Vec<f32> {
        let mut mx = self.edge_to_node(&self.node_to_edge(x));
        for (m, &xv) in mx.iter_mut().zip(x) {
            *m += ridge * xv;
        }
        mx
    }

    /// The right-hand side `E y = k · Aᵀ y` from per-hyperedge measurements.
    ///
    /// # Preconditions
    /// `y.len() == n_edges`.
    pub fn rhs(&self, y: &[f32]) -> Vec<f32> {
        self.edge_to_node(y)
    }

    /// The **clique** expansion: contract the hyperedge index into a dense row-major
    /// `n_nodes × n_nodes` operator, `k · AᵀA + ρ I`.
    ///
    /// Each hyperedge contributes `∝ a_c a_cᵀ` over its members — a clique. This is the star→clique
    /// (star-mesh / Schur) contraction done edge by edge.
    ///
    /// # Postconditions
    /// Length `n_nodes²`, symmetric up to floating-point, and equal to [`Self::star_matvec`] as an
    /// operator. Accumulated edge-major, then `(a, b)` member pairs in declaration order.
    pub fn assemble_clique(&self, ridge: f32) -> Vec<f32> {
        let (n, k) = (self.n_nodes, self.arity);
        let inv_k = 1.0 / k as f32;
        let mut m = vec![0.0f32; n * n];
        for c in 0..self.n_edges {
            for a in 0..k {
                let va = self.members[c * k + a] as usize;
                let wa = self.signs[c * k + a] * self.scale[va] * inv_k;
                for b in 0..k {
                    let vb = self.members[c * k + b] as usize;
                    let wb = self.signs[c * k + b] * self.scale[vb] * inv_k;
                    m[va * n + vb] += (k as f32) * wa * wb;
                }
            }
        }
        for v in 0..n {
            m[v * n + v] += ridge;
        }
        m
    }

    /// Multiply an assembled dense operator by `x` — the reference path the matrix-free form must
    /// match.
    ///
    /// # Preconditions
    /// `m.len() == n_nodes²`, `x.len() == n_nodes`.
    pub fn dense_matvec(&self, m: &[f32], x: &[f32]) -> Vec<f32> {
        let n = self.n_nodes;
        assert_eq!(m.len(), n * n);
        assert_eq!(x.len(), n);
        (0..n)
            .map(|i| (0..n).map(|j| m[i * n + j] * x[j]).sum())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The historical Nagare fixture generator, reproduced here so core owns the shape it must
    /// support: a spanning chain for connectivity plus random members, **mixed** signs, and the
    /// degree-derived scale `1/sqrt(1+deg)`.
    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        }
    }

    fn fixture(n_nodes: usize, n_edges: usize, k: usize, seed: u64) -> SignedHyperIncidence {
        let mut rng = lcg(seed);
        let mut members = Vec::with_capacity(n_edges * k);
        let mut signs = Vec::with_capacity(n_edges * k);
        let mut deg = vec![0u32; n_nodes];
        for c in 0..n_edges {
            let mut mem = Vec::with_capacity(k);
            if c + 1 < n_nodes {
                mem.push(c as u32);
                mem.push((c + 1) as u32);
            }
            while mem.len() < k {
                let v = (rng().abs() * n_nodes as f32) as u32 % n_nodes as u32;
                if !mem.contains(&v) {
                    mem.push(v);
                }
            }
            for &v in &mem {
                members.push(v);
                signs.push(if rng() < 0.0 { -1.0 } else { 1.0 });
                deg[v as usize] += 1;
            }
        }
        let scale = (0..n_nodes)
            .map(|v| 1.0 / (1.0 + deg[v] as f32).sqrt())
            .collect();
        SignedHyperIncidence::new(n_nodes, k, members, signs, scale).expect("a valid fixture")
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// **The central identity**: the matrix-free star operator equals multiplication by the
    /// assembled clique matrix.
    #[test]
    fn the_star_matvec_equals_the_assembled_clique_matrix() {
        for (n, e, k, seed) in [
            (30usize, 40usize, 3usize, 7u64),
            (60, 90, 4, 7),
            (40, 60, 3, 9),
        ] {
            let hg = fixture(n, e, k, seed);
            let ridge = 1e-2;
            let m = hg.assemble_clique(ridge);
            let mut rng = lcg(55);
            let x: Vec<f32> = (0..n).map(|_| rng()).collect();
            let star = hg.star_matvec(&x, ridge);
            let clique = hg.dense_matvec(&m, &x);
            assert!(
                max_abs_diff(&star, &clique) < 1e-4,
                "n={n} k={k}: star vs clique matvec diverged by {}",
                max_abs_diff(&star, &clique)
            );
        }
    }

    /// The **scaled** adjoint identity `k · ⟨A x, h⟩ == ⟨x, E h⟩`, because `edge_to_node` is
    /// `k · Aᵀ`. Asserting the unscaled identity would be asserting a contract this operator pair
    /// does not have — it fails by exactly a factor of `k`, which is how this was caught.
    #[test]
    fn the_scaled_adjoint_identity_holds() {
        for (n, e, k, seed) in [(25usize, 35usize, 3usize, 3u64), (18, 22, 4, 8)] {
            let hg = fixture(n, e, k, seed);
            let mut rng = lcg(11);
            let x: Vec<f32> = (0..hg.n_nodes()).map(|_| rng()).collect();
            let h: Vec<f32> = (0..hg.n_edges()).map(|_| rng()).collect();
            let lhs: f32 = k as f32
                * hg.node_to_edge(&x)
                    .iter()
                    .zip(&h)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            let rhs: f32 = hg.edge_to_node(&h).iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (lhs - rhs).abs() < 1e-4 * lhs.abs().max(1.0),
                "k={k} scaled adjoint: {lhs} vs {rhs}"
            );
        }
    }

    /// Duplicate members contribute twice, member order does not change the sum, and a self-pair
    /// lands on the diagonal. Nothing is deduplicated.
    #[test]
    fn duplicate_members_and_ordering_behave_as_documented() {
        // one hyperedge, node 0 listed twice
        let dup =
            SignedHyperIncidence::new(2, 2, vec![0, 0], vec![1.0, 1.0], vec![1.0, 1.0]).unwrap();
        let h = dup.node_to_edge(&[1.0, 0.0]);
        assert_eq!(
            h[0], 1.0,
            "two half-weight terms on the same node sum to one"
        );
        let m = dup.assemble_clique(0.0);
        assert!(m[0] > 0.0, "the self-pair lands on the diagonal: {}", m[0]);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[2], 0.0);
        assert_eq!(m[3], 0.0);

        // permuting the members of one hyperedge leaves A x unchanged (a sum), by construction
        let a = SignedHyperIncidence::new(3, 3, vec![0, 1, 2], vec![1.0, -1.0, 1.0], vec![1.0; 3])
            .unwrap();
        let b = SignedHyperIncidence::new(3, 3, vec![2, 1, 0], vec![1.0, -1.0, 1.0], vec![1.0; 3])
            .unwrap();
        let x = [0.5f32, -0.25, 2.0];
        assert!((a.node_to_edge(&x)[0] - b.node_to_edge(&x)[0]).abs() < 1e-7);
    }

    /// Mixed signs are load-bearing: flipping one member's sign changes the operator.
    #[test]
    fn mixed_signs_change_the_operator() {
        let pos =
            SignedHyperIncidence::new(2, 2, vec![0, 1], vec![1.0, 1.0], vec![1.0, 1.0]).unwrap();
        let neg =
            SignedHyperIncidence::new(2, 2, vec![0, 1], vec![1.0, -1.0], vec![1.0, 1.0]).unwrap();
        let x = [1.0f32, 1.0];
        assert!((pos.node_to_edge(&x)[0] - 1.0).abs() < 1e-7);
        assert!(
            neg.node_to_edge(&x)[0].abs() < 1e-7,
            "opposite signs cancel"
        );
        assert_ne!(
            pos.assemble_clique(0.0)[1].to_bits(),
            neg.assemble_clique(0.0)[1].to_bits(),
            "the off-diagonal coupling flips with the sign product"
        );
        // the fixture generator really produces both signs
        let hg = fixture(20, 30, 3, 7);
        assert!(hg.signs().iter().any(|&s| s > 0.0) && hg.signs().iter().any(|&s| s < 0.0));
    }

    /// Uniform arity is an explicit contract, and every degenerate shape is rejected.
    #[test]
    fn degenerate_and_invalid_shapes_are_rejected() {
        assert_eq!(
            SignedHyperIncidence::new(2, 0, vec![], vec![], vec![1.0, 1.0]),
            Err(HyperIncidenceError::ZeroArity)
        );
        assert_eq!(
            SignedHyperIncidence::new(2, 2, vec![0], vec![1.0], vec![1.0, 1.0]),
            Err(HyperIncidenceError::ShapeMismatch)
        );
        assert_eq!(
            SignedHyperIncidence::new(2, 2, vec![0, 1], vec![1.0], vec![1.0, 1.0]),
            Err(HyperIncidenceError::ShapeMismatch)
        );
        assert_eq!(
            SignedHyperIncidence::new(2, 2, vec![0, 1], vec![1.0, 1.0], vec![1.0]),
            Err(HyperIncidenceError::ScaleLengthMismatch)
        );
        assert_eq!(
            SignedHyperIncidence::new(2, 2, vec![0, 5], vec![1.0, 1.0], vec![1.0, 1.0]),
            Err(HyperIncidenceError::MemberOutOfRange { at: 1, node: 5 })
        );
        // an empty hypergraph over real nodes is valid and yields the ridge alone
        let empty = SignedHyperIncidence::new(3, 2, vec![], vec![], vec![1.0; 3]).unwrap();
        assert_eq!(empty.n_edges(), 0);
        assert_eq!(
            empty.star_matvec(&[1.0, 2.0, 3.0], 0.5),
            vec![0.5, 1.0, 1.5]
        );
        assert_eq!(empty.assemble_clique(0.5)[0], 0.5);
    }

    /// A positive ridge makes the operator strictly positive definite: `xᵀMx > 0` for `x != 0`,
    /// including for a hypergraph whose unridged operator is singular.
    #[test]
    fn a_ridge_makes_the_operator_positive_definite() {
        let hg = fixture(20, 25, 3, 5);
        let n = hg.n_nodes();
        let ridge = 1e-2;
        let mut rng = lcg(77);
        for _ in 0..8 {
            let x: Vec<f32> = (0..n).map(|_| rng()).collect();
            let q: f32 = hg
                .star_matvec(&x, ridge)
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum();
            let xx: f32 = x.iter().map(|v| v * v).sum();
            assert!(
                q >= ridge * xx - 1e-4,
                "quadratic form {q} below the ridge floor"
            );
            assert!(q > 0.0, "positive definite");
        }
    }

    /// The historical mixed-sign fixture is representable and validates.
    #[test]
    fn the_historical_fixture_shape_is_supported() {
        for (n, e, k, seed) in [
            (30usize, 40usize, 3usize, 7u64),
            (50, 70, 4, 1),
            (120, 200, 5, 7),
        ] {
            let hg = fixture(n, e, k, seed);
            assert_eq!(hg.n_nodes(), n);
            assert_eq!(hg.n_edges(), e);
            assert_eq!(hg.arity(), k);
            assert_eq!(hg.members().len(), e * k);
            assert_eq!(hg.signs().len(), e * k);
            assert_eq!(hg.scale().len(), n);
            assert!(hg.scale().iter().all(|&s| s > 0.0 && s <= 1.0));
        }
    }
}
