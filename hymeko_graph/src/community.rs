//! Community detection on signed graphs.
//!
//! Label propagation (Raghavan-Albert-Kumara 2007) is the cheap MVP:
//! O(|V| + |E|) per iteration, deterministic with a fixed seed, ~50
//! lines.  Each vertex's label is set to the most common label among
//! its neighbours; iterate until convergence.  We do **not** use the
//! signed structure for the propagation itself — communities are
//! detected on the underlying unsigned topology, and the *signed*
//! information enters at the next stage (per-community balance
//! ratio, used to pick the right axiom pruner per community).
//!
//! This is the Phase-A plumbing for the user's information-theoretic
//! clustering idea: `community_id` per vertex + `balance_ratio` per
//! community lets us apply the Cartwright-Harary axiom locally
//! instead of globally.

use std::collections::HashMap;

use crate::signed_graph::SignedGraph;
use crate::traversal::Csr;

/// Label propagation with deterministic tie-break.  Returns a
/// `Vec<u32>` of length `n_nodes`; entry $i$ is the community ID of
/// vertex $i$ (initially $i$ itself; communities are renumbered
/// densely after convergence).
///
/// Parameters:
/// - `csr`: undirected adjacency.
/// - `max_iters`: hard cap on iterations.
/// - `seed`: shuffle seed for vertex visit order (LP is sensitive
///   to order; a random shuffle per iteration is standard).
///
/// Returns `(labels, n_communities)`.
pub fn label_propagation(csr: &Csr, max_iters: u32, seed: u64) -> (Vec<u32>, u32) {
    let n = csr.row_ptr.len() - 1;
    let mut labels: Vec<u32> = (0..n as u32).collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut state = seed | 1;

    for _ in 0..max_iters {
        // Per-iteration xorshift shuffle (Fisher-Yates).
        for i in (1..order.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut changed = 0u32;
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &v in &order {
            let s = csr.row_ptr[v as usize] as usize;
            let e = csr.row_ptr[v as usize + 1] as usize;
            counts.clear();
            for &u in &csr.col_idx[s..e] {
                *counts.entry(labels[u as usize]).or_insert(0) += 1;
            }
            // Find label with max count, ties broken by smallest id.
            let mut best_count = 0;
            let mut best_label = labels[v as usize];
            for (&lab, &cnt) in &counts {
                if cnt > best_count || (cnt == best_count && lab < best_label) {
                    best_count = cnt;
                    best_label = lab;
                }
            }
            if best_label != labels[v as usize] {
                labels[v as usize] = best_label;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }

    // Renumber labels densely (0..K-1).
    let mut id_map: HashMap<u32, u32> = HashMap::new();
    for &l in &labels {
        let next = id_map.len() as u32;
        id_map.entry(l).or_insert(next);
    }
    let n_comm = id_map.len() as u32;
    for l in labels.iter_mut() {
        *l = id_map[l];
    }
    (labels, n_comm)
}

/// Per-community balance ratio, computed on triangles (k=3 cycles).
/// Returns a Vec of length `n_communities`; entry $c$ is the
/// fraction of balanced triangles internal to community $c$
/// (NaN if the community has zero internal triangles).
///
/// "Internal" means all three vertices share the same community.
/// This is the simplest definition; a fancier version would also
/// count triangles partially in a community with vertex-weighted
/// fractional contribution.
pub fn balance_ratio_per_community(
    g: &SignedGraph,
    csr: &Csr,
    labels: &[u32],
    n_comm: u32,
) -> Vec<f64> {
    let sign_lookup = g.build_sign_lookup();
    let mut bal: Vec<u64> = vec![0; n_comm as usize];
    let mut tot: Vec<u64> = vec![0; n_comm as usize];
    let n = csr.row_ptr.len() - 1;
    for u in 0..n as u32 {
        let cu = labels[u as usize];
        let s = csr.row_ptr[u as usize] as usize;
        let e = csr.row_ptr[u as usize + 1] as usize;
        let nbrs_u: Vec<u32> = csr.col_idx[s..e].to_vec();
        for &v in &nbrs_u {
            if v <= u || labels[v as usize] != cu {
                continue;
            }
            // Common neighbours of u and v that share community cu.
            let sv = csr.row_ptr[v as usize] as usize;
            let ev = csr.row_ptr[v as usize + 1] as usize;
            let nbrs_v: &[u32] = &csr.col_idx[sv..ev];
            for &w in nbrs_v {
                if w <= v || labels[w as usize] != cu {
                    continue;
                }
                if !nbrs_u.contains(&w) {
                    continue;
                }
                // Triangle (u, v, w) all in community cu.
                let s_uv = sign_lookup.get(&(u.min(v), u.max(v))).copied().unwrap_or(1);
                let s_vw = sign_lookup.get(&(v.min(w), v.max(w))).copied().unwrap_or(1);
                let s_uw = sign_lookup.get(&(u.min(w), u.max(w))).copied().unwrap_or(1);
                let prod = (s_uv as i32) * (s_vw as i32) * (s_uw as i32);
                tot[cu as usize] += 1;
                if prod > 0 {
                    bal[cu as usize] += 1;
                }
            }
        }
    }
    bal.iter()
        .zip(tot.iter())
        .map(|(b, t)| {
            if *t == 0 {
                f64::NAN
            } else {
                *b as f64 / *t as f64
            }
        })
        .collect()
}

// ─── Community-axiom pruner ────────────────────────────────────────

use crate::pruner::{CyclePruner, PrunerDecision};

/// Per-community axiom choice. Compatible with the existing
/// `CyclePruner` trait: this is an emit-time pruner that looks up
/// the cycle's dominant community and applies the corresponding
/// axiom rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxiomChoice {
    /// Accept any closed cycle (no axiom filter).
    None,
    /// Accept only balanced cycles ($\prod s_i = +1$). Use in
    /// communities with high balance ratio (cooperative regime).
    Balance,
    /// Accept only unbalanced cycles ($\prod s_i = -1$). Use in
    /// communities with low balance ratio (adversarial regime).
    Unbalanced,
    /// Davis weak balance — reject only all-negative cycles.
    Davis,
}

/// Pruner that consults a per-community axiom map. The cycle's
/// "community" is its **dominant community** — the one shared by
/// the plurality of its vertices.
#[derive(Debug)]
pub struct CommunityAxiomPruner {
    /// Per-vertex community label.
    pub labels: Vec<u32>,
    /// Per-community axiom choice. Length must equal `n_communities`.
    pub axiom_per_community: Vec<AxiomChoice>,
}

impl CommunityAxiomPruner {
    /// Auto-derive the per-community axiom from each community's
    /// balance ratio:
    ///
    /// - $\beta \geq \tau_{\mathrm{coop}}$ (default 0.85) → `Balance`
    /// - $\beta \leq \tau_{\mathrm{adv}}$ (default 0.75)  → `Unbalanced`
    /// - else (intermediate)                              → `None`
    pub fn auto(labels: Vec<u32>, balance_ratios: &[f64], tau_coop: f64, tau_adv: f64) -> Self {
        let axioms: Vec<AxiomChoice> = balance_ratios
            .iter()
            .map(|&b| {
                if b.is_nan() {
                    AxiomChoice::None
                } else if b >= tau_coop {
                    AxiomChoice::Balance
                } else if b <= tau_adv {
                    AxiomChoice::Unbalanced
                } else {
                    AxiomChoice::None
                }
            })
            .collect();
        Self {
            labels,
            axiom_per_community: axioms,
        }
    }

    fn dominant_community(&self, cycle: &[u32]) -> u32 {
        // Cycle length is small (3–6), so a tiny linear sweep beats a
        // HashMap.
        let mut max_label = self.labels[cycle[0] as usize];
        let mut max_count = 0u32;
        for &v in cycle {
            let lab = self.labels[v as usize];
            let mut count = 0u32;
            for &w in cycle {
                if self.labels[w as usize] == lab {
                    count += 1;
                }
            }
            if count > max_count || (count == max_count && lab < max_label) {
                max_count = count;
                max_label = lab;
            }
        }
        max_label
    }
}

impl CyclePruner for CommunityAxiomPruner {
    fn emit_ok(&self, cycle: &[u32], edge_signs: &[i8]) -> PrunerDecision {
        let dom = self.dominant_community(cycle) as usize;
        let axiom = self
            .axiom_per_community
            .get(dom)
            .copied()
            .unwrap_or(AxiomChoice::None);
        match axiom {
            AxiomChoice::None => PrunerDecision::Accept,
            AxiomChoice::Balance => {
                let prod: i32 = edge_signs.iter().map(|&s| s as i32).product();
                if prod > 0 {
                    PrunerDecision::Accept
                } else {
                    PrunerDecision::Reject
                }
            }
            AxiomChoice::Unbalanced => {
                let prod: i32 = edge_signs.iter().map(|&s| s as i32).product();
                if prod < 0 {
                    PrunerDecision::Accept
                } else {
                    PrunerDecision::Reject
                }
            }
            AxiomChoice::Davis => {
                let n_neg = edge_signs.iter().filter(|&&s| s < 0).count();
                if n_neg == edge_signs.len() {
                    PrunerDecision::Reject
                } else {
                    PrunerDecision::Accept
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two disjoint triangles → label propagation must put them in
    /// separate communities, and the per-community balance ratio
    /// computes correctly.
    #[test]
    fn two_disjoint_triangles_separate_communities() {
        let g = SignedGraph::from_parts(
            6,
            &[0, 1, 2, 3, 4, 5],
            &[1, 2, 0, 4, 5, 3],
            &[1, 1, 1, 1, 1, -1], // first balanced, second unbalanced
        );
        let csr = Csr::from_graph(&g);
        let (labels, n_comm) = label_propagation(&csr, 50, 42);
        assert_eq!(n_comm, 2);
        // {0, 1, 2} share a community; {3, 4, 5} share another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);

        let bal = balance_ratio_per_community(&g, &csr, &labels, n_comm);
        // Sort by community label to make the assertion deterministic.
        // One community has all-positive triangle (balance=1.0), the
        // other has 2 positive + 1 negative (balance=0.0).
        let c0 = labels[0] as usize;
        let c3 = labels[3] as usize;
        assert!((bal[c0] - 1.0).abs() < 1e-9);
        assert!(bal[c3].abs() < 1e-9);
    }

    /// Single connected triangle → one community, balance = 1.0.
    #[test]
    fn single_triangle_single_community() {
        let g = SignedGraph::from_parts(3, &[0, 1, 2], &[1, 2, 0], &[1, 1, 1]);
        let csr = Csr::from_graph(&g);
        let (labels, n_comm) = label_propagation(&csr, 50, 0);
        assert_eq!(n_comm, 1);
        assert_eq!(labels, vec![0, 0, 0]);
        let bal = balance_ratio_per_community(&g, &csr, &labels, n_comm);
        assert_eq!(bal.len(), 1);
        assert!((bal[0] - 1.0).abs() < 1e-9);
    }

    /// `CommunityAxiomPruner::auto` correctly derives Balance for
    /// high-β community and Unbalanced for low-β community.
    #[test]
    fn community_axiom_pruner_auto_picks_per_community() {
        let labels = vec![0, 0, 0, 1, 1, 1];
        let bal = vec![1.0, 0.2]; // comm 0 cooperative, comm 1 adversarial
        let p = CommunityAxiomPruner::auto(labels.clone(), &bal, 0.85, 0.75);
        assert_eq!(
            p.axiom_per_community,
            vec![AxiomChoice::Balance, AxiomChoice::Unbalanced]
        );

        // Cycle internal to comm 0 with balanced sign-product → accept
        let dec = p.emit_ok(&[0, 1, 2], &[1, 1, 1]);
        assert_eq!(dec, PrunerDecision::Accept);
        // Cycle internal to comm 0 with unbalanced sign-product → reject
        let dec = p.emit_ok(&[0, 1, 2], &[1, 1, -1]);
        assert_eq!(dec, PrunerDecision::Reject);
        // Cycle internal to comm 1 with unbalanced sign-product → accept
        let dec = p.emit_ok(&[3, 4, 5], &[1, 1, -1]);
        assert_eq!(dec, PrunerDecision::Accept);
        // Cycle internal to comm 1 with balanced sign-product → reject
        let dec = p.emit_ok(&[3, 4, 5], &[1, 1, 1]);
        assert_eq!(dec, PrunerDecision::Reject);
    }

    /// 4-vertex bipartite path with one negative edge → label
    /// propagation result is deterministic with a fixed seed.
    #[test]
    fn label_propagation_is_deterministic_with_fixed_seed() {
        let g = SignedGraph::from_parts(5, &[0, 1, 2, 3, 0], &[1, 2, 3, 4, 4], &[1, 1, 1, 1, -1]);
        let csr = Csr::from_graph(&g);
        let (labels1, n1) = label_propagation(&csr, 50, 42);
        let (labels2, n2) = label_propagation(&csr, 50, 42);
        assert_eq!(labels1, labels2);
        assert_eq!(n1, n2);
    }
}
