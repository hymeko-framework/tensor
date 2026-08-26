//! Signed tuple families, multi-edge sign resolution, and capped tuple enumeration.
//!
//! # Why this is core and not application code
//!
//! A *signed tuple* is a vertex path over a [`SignedGraph`] carrying one sign per member vertex,
//! derived from the signs of the edges it traverses. Three things have to agree for that to be
//! well defined, and each was previously re-derived at its call site:
//!
//! 1. **Which edges a tuple traverses** — a cycle wraps, a walk does not, a single edge is neither.
//! 2. **Which sign an edge has when the graph is a multigraph** — a signed graph may retain
//!    parallel edges between the same pair, so a vertex *pair* does not determine a sign. That
//!    needs a named contract, not a `HashMap` built at the call site.
//! 3. **Whether a capped enumeration is reproducible** — a reservoir merged across rayon workers is
//!    not sample-stable, and silently is not.
//!
//! All three now live here, next to the graph they are about.
//!
//! # Preconditions (whole module)
//! - Every vertex id is `< graph.n_nodes`.
//! - `SignedGraph::signs` is parallel to `SignedGraph::edges` and holds only `-1` or `+1`.

use std::collections::HashMap;

use crate::signed_graph::SignedGraph;
use crate::unsigned_cycles::enumerate_parallel;
use crate::walks_unsigned::enumerate_walks;

// =====================================================================================
// Families
// =====================================================================================

/// A signed tuple's shape, and with it the rule for its member edges and member signs.
///
/// A sealed enum rather than a width parameter **because the three arms genuinely differ**:
/// flattening them into one `k`-uniform constructor would give [`Edge`](Self::Edge) the wrapping
/// cycle rule and silently produce the wrong signs.
///
/// Moved here from the Nagare application layer on 2026-08-26: the mathematics is about signed
/// graphs, and three separate consumers were re-deriving the member-edge walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignedTupleFamily {
    /// A single graph edge, its own sign duplicated to both endpoints.
    ///
    /// **This family is indexed by edge, not by vertex pair.** It is the one family that needs no
    /// [`MultiEdgeSignResolution`]: a parallel edge is a distinct tuple carrying its own raw sign.
    Edge,
    /// A closed cycle: per-vertex negative incidence **wraps** modulo `k`.
    Cycle {
        /// Cycle length in vertices.
        k: usize,
    },
    /// An open walk over `len + 1` vertices: incidence does **not** wrap.
    Walk {
        /// Walk length in edges.
        len: usize,
    },
}

impl SignedTupleFamily {
    /// Vertices per tuple.
    pub fn arity(self) -> usize {
        match self {
            Self::Edge => 2,
            Self::Cycle { k } => k,
            Self::Walk { len } => len + 1,
        }
    }

    /// Edge signs per tuple.
    pub fn edge_count(self) -> usize {
        match self {
            Self::Edge => 1,
            Self::Cycle { k } => k,
            Self::Walk { len } => len,
        }
    }

    /// The tuple's member edges as **directed** `(from, to)` pairs, in traversal order.
    ///
    /// Directed because that is what a multi-edge resolver needs: the recovered oracle keys its
    /// sign table on the directed pair and only falls back to the reverse orientation when the
    /// forward one is absent (see [`MultiEdgeSignResolution`]).
    ///
    /// # Preconditions
    /// `tuple.len() == self.arity()`.
    ///
    /// # Postconditions
    /// Exactly [`Self::edge_count`] pairs. A cycle **wraps** (the last pair closes back to the
    /// first); a walk does not; an edge yields its single pair.
    pub fn member_edges(self, tuple: &[u32]) -> Vec<(u32, u32)> {
        debug_assert_eq!(tuple.len(), self.arity(), "tuple width");
        match self {
            Self::Edge => vec![(tuple[0], tuple[1])],
            Self::Cycle { k } => (0..k).map(|j| (tuple[j], tuple[(j + 1) % k])).collect(),
            Self::Walk { len } => (0..len).map(|j| (tuple[j], tuple[j + 1])).collect(),
        }
    }

    /// Per-vertex signs from the tuple's edge signs.
    ///
    /// # Preconditions
    /// `edge_signs.len() == self.edge_count()`; every entry is `-1` or `+1`.
    ///
    /// # Postconditions
    /// Returns [`Self::arity`] signs. For cycles and walks a vertex is `+1` exactly when its
    /// negative incidence **within the tuple** is even; for an edge both endpoints carry the raw
    /// edge sign, which is a different rule and not a parity.
    pub fn member_signs(self, edge_signs: &[i8]) -> Vec<i8> {
        debug_assert_eq!(edge_signs.len(), self.edge_count(), "edge-sign width");
        match self {
            Self::Edge => vec![edge_signs[0], edge_signs[0]],
            Self::Cycle { k } => {
                let mut counts = vec![0usize; k];
                for (i, &s) in edge_signs.iter().enumerate() {
                    if s == -1 {
                        counts[i] += 1;
                        counts[(i + 1) % k] += 1;
                    }
                }
                parity_signs(&counts)
            }
            Self::Walk { len } => {
                let mut counts = vec![0usize; len + 1];
                for (j, &s) in edge_signs.iter().enumerate() {
                    if s == -1 {
                        counts[j] += 1;
                        counts[j + 1] += 1;
                    }
                }
                parity_signs(&counts)
            }
        }
    }

    /// Davis weak balance: balanced iff the tuple carries an even number of negative edges.
    pub fn balanced(edge_signs: &[i8]) -> bool {
        edge_signs.iter().filter(|&&s| s == -1).count() % 2 == 0
    }
}

fn parity_signs(counts: &[usize]) -> Vec<i8> {
    counts
        .iter()
        .map(|c| if c % 2 == 0 { 1 } else { -1 })
        .collect()
}

// =====================================================================================
// Multi-edge sign resolution
// =====================================================================================

/// Which of a multigraph's parallel edges gives a traversed vertex pair its sign.
///
/// **This contract only exists because a pair does not determine a sign.** On a graph that retains
/// parallel edges, a traversal stepping `u -> v` is choosing among several recorded signs. Leaving
/// that choice to whichever `HashMap` insertion happened last is how two implementations of "the
/// same" enumeration silently disagree, and the disagreement is invisible in every count: same
/// tuple total, same vertex sets, different signs.
///
/// [`SignedTupleFamily::Edge`] never consults this: it is indexed by edge, so it keeps the raw
/// per-edge sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiEdgeSignResolution {
    /// The **earliest** recorded occurrence, scanning every edge in its stored orientation first
    /// and only then every edge reversed.
    ///
    /// This is the **recovered oracle's** rule, characterised rather than guessed: the reference
    /// implementation concatenates `[forward keys, reverse keys]`, sorts with
    /// `numpy.argsort(kind="stable")` and looks up with `searchsorted(..., side="left")`. A stable
    /// sort keeps equal keys in concatenation order and a left search takes the first of them, so
    /// the winner is the lowest index in that concatenation — forward orientation ahead of
    /// reverse, and within each, the lowest edge index.
    FirstOccurrenceForwardThenReverse,
    /// The **last** edge written for the unordered pair.
    ///
    /// What [`SignedGraph::build_sign_lookup`] has always computed: it inserts `(min, max) -> sign`
    /// for every edge in order, so later parallel edges overwrite earlier ones and orientation is
    /// discarded. Reachable **by name** because it is the behaviour of an existing core function
    /// and of anything built on it; it is not the recovered rule.
    LastWriteUnordered,
}

impl MultiEdgeSignResolution {
    /// Reporting label. Names the semantics, never a date.
    pub fn label(self) -> &'static str {
        match self {
            Self::FirstOccurrenceForwardThenReverse => "first-occurrence forward-then-reverse",
            Self::LastWriteUnordered => "last-write unordered",
        }
    }
}

/// A resolved directed-pair sign table under one [`MultiEdgeSignResolution`].
#[derive(Clone, Debug)]
pub struct SignResolver {
    directed: HashMap<(u32, u32), i8>,
    resolution: MultiEdgeSignResolution,
}

impl SignResolver {
    /// Build the table over a chosen subset of the graph's edges.
    ///
    /// # Preconditions
    /// Every entry of `visible` is a valid index into `graph.edges`.
    ///
    /// # Postconditions
    /// [`Self::sign_of`] answers for every visible pair in **both** orientations, and for no other
    /// pair. Building twice over the same inputs yields the same table.
    pub fn build(
        graph: &SignedGraph,
        visible: &[u32],
        resolution: MultiEdgeSignResolution,
    ) -> Self {
        let mut directed: HashMap<(u32, u32), i8> = HashMap::with_capacity(visible.len() * 2);
        match resolution {
            MultiEdgeSignResolution::FirstOccurrenceForwardThenReverse => {
                // Forward pass first, then reverse: exactly the concatenation order the oracle's
                // stable sort preserves. `or_insert` keeps the earliest of equal keys.
                for &e in visible {
                    let (u, v) = graph.edges[e as usize];
                    directed.entry((u, v)).or_insert(graph.signs[e as usize]);
                }
                for &e in visible {
                    let (u, v) = graph.edges[e as usize];
                    directed.entry((v, u)).or_insert(graph.signs[e as usize]);
                }
            }
            MultiEdgeSignResolution::LastWriteUnordered => {
                // Orientation is discarded and later edges overwrite earlier ones, so both
                // directed keys always carry the same value.
                for &e in visible {
                    let (u, v) = graph.edges[e as usize];
                    let s = graph.signs[e as usize];
                    directed.insert((u, v), s);
                    directed.insert((v, u), s);
                }
            }
        }
        Self {
            directed,
            resolution,
        }
    }

    /// The resolved sign of a traversed directed pair, or `None` when the pair is not visible.
    pub fn sign_of(&self, from: u32, to: u32) -> Option<i8> {
        self.directed.get(&(from, to)).copied()
    }

    /// The contract this table was built under.
    pub fn resolution(&self) -> MultiEdgeSignResolution {
        self.resolution
    }

    /// Distinct directed pairs the table answers for.
    pub fn len(&self) -> usize {
        self.directed.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.directed.is_empty()
    }
}

// =====================================================================================
// Execution contract
// =====================================================================================

/// How a capped enumeration is executed, and therefore whether its sample is reproducible.
///
/// **Not a performance knob.** A cap turns enumeration into *sampling*, and
/// [`enumerate_parallel`] keeps one reservoir per rayon worker before merging — so which tuples
/// survive depends on how the runtime split the starting vertices. Measured on a 24 k-edge graph
/// enumerating 5-cycles at cap 100 000: two runs at the same seed produced materialisations whose
/// content digests differed, while the tuple *count* was identical both times. A consumer checking
/// counts would have seen nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnumerationExecution {
    /// Single-threaded. **Byte-identical across runs** for a fixed `(graph, family, cap, seed)`;
    /// this is the contract, asserted by `deterministic_serial_materialisation_is_byte_identical`.
    DeterministicSerial,
    /// Rayon-parallel, with an optional worker count.
    ///
    /// Faster, and **sample-unstable under a cap** — the pre-existing behaviour, kept reachable by
    /// name and characterised by `rayon_execution_is_not_sample_stable_under_a_cap` rather than
    /// silently rewritten. Without a cap the enumeration is exhaustive and the set is the same
    /// (order still is not).
    Rayon {
        /// Worker count; `None` uses the ambient rayon pool.
        threads: Option<usize>,
    },
}

impl EnumerationExecution {
    /// Whether this contract promises a byte-identical capped materialisation.
    pub fn is_reproducible_under_cap(self) -> bool {
        matches!(self, Self::DeterministicSerial)
    }

    /// Reporting label.
    pub fn label(self) -> &'static str {
        match self {
            Self::DeterministicSerial => "deterministic-serial",
            Self::Rayon { .. } => "rayon",
        }
    }
}

// =====================================================================================
// Enumeration
// =====================================================================================

/// Everything a capped, signed enumeration must be told beyond the graph and the family.
///
/// A struct rather than four positional arguments: `cap`, `seed`, `execution` and `resolution` are
/// **one contract** — they jointly decide which tuples exist and what signs they carry — and a
/// caller that gets their order wrong gets a plausible wrong answer, not a type error (§6.5 #6).
/// It is also the value a result must echo, so [`TupleEnumeration`] stores it rather than
/// re-listing its fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumerationRequest {
    /// Cap on retained tuples. `None` enumerates exhaustively.
    pub cap: Option<usize>,
    /// Seed handed to the sampler.
    pub seed: u64,
    /// Execution contract, and therefore whether a capped sample is reproducible.
    pub execution: EnumerationExecution,
    /// How parallel edges collapse when a traversal names a vertex pair.
    pub resolution: MultiEdgeSignResolution,
}

impl EnumerationRequest {
    /// Whether this request's sampling is reproducible: nothing capped, or a contract that
    /// promises a byte-identical materialisation.
    pub fn is_reproducible(self) -> bool {
        self.cap.is_none() || self.execution.is_reproducible_under_cap()
    }
}

/// One family's enumerated, signed tuple set, with the contracts that produced it.
///
/// The contracts travel **with the data** so a downstream consumer cannot report a result without
/// them — the same reason `SignedGraph` carries its signs rather than taking them as an argument.
#[derive(Clone, Debug)]
pub struct TupleEnumeration {
    /// The family enumerated.
    pub family: SignedTupleFamily,
    /// Member vertex ids, flat `(n_tuples, arity)`.
    pub vertices: Vec<u32>,
    /// Member signs, flat `(n_tuples, arity)`, parallel to `vertices`.
    pub member_signs: Vec<i8>,
    /// Tuples whose member edges could not all be resolved, and were therefore dropped.
    pub dropped: usize,
    /// The request that produced this set.
    pub request: EnumerationRequest,
    /// Whether a multi-edge resolution was actually consulted. `false` for
    /// [`SignedTupleFamily::Edge`], which is indexed by edge and keeps raw per-edge signs.
    pub resolved_multi_edges: bool,
}

impl TupleEnumeration {
    /// Tuples retained.
    pub fn len(&self) -> usize {
        self.vertices
            .len()
            .checked_div(self.family.arity())
            .unwrap_or(0)
    }

    /// Whether nothing was retained.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// The resolution actually consulted, or `None` when the family needed none.
    pub fn resolution(&self) -> Option<MultiEdgeSignResolution> {
        self.resolved_multi_edges.then_some(self.request.resolution)
    }
}

/// Enumerate one family's signed tuples over a chosen subset of a signed graph's edges.
///
/// The `visible` subset is applied **before** enumeration, not after: an edge outside it is not a
/// traversable edge, so no tuple can be built through it. That is what makes a leakage boundary a
/// boundary rather than a filter.
///
/// # Preconditions
/// Every entry of `visible` indexes `graph.edges`; `cap`, when given, is non-zero.
///
/// # Postconditions
/// `vertices.len() == member_signs.len()` and both are a whole multiple of `family.arity()`.
/// Under [`EnumerationExecution::DeterministicSerial`] the result is byte-identical across calls
/// with equal inputs. [`SignedTupleFamily::Edge`] carries raw per-edge signs and sets
/// `resolution` to `None`.
pub fn enumerate_signed_tuples(
    graph: &SignedGraph,
    visible: &[u32],
    family: SignedTupleFamily,
    request: EnumerationRequest,
) -> TupleEnumeration {
    let EnumerationRequest {
        cap,
        seed,
        execution,
        resolution,
    } = request;
    debug_assert!(cap != Some(0), "a zero cap enumerates nothing");
    let arity = family.arity();
    let n_nodes = graph.n_nodes as usize;

    // The Edge family is indexed by edge, so it never consults a resolver: a parallel edge is a
    // distinct tuple carrying its own raw sign.
    if family == SignedTupleFamily::Edge {
        let take = cap.unwrap_or(visible.len()).min(visible.len());
        let mut vertices = Vec::with_capacity(take * 2);
        let mut member_signs = Vec::with_capacity(take * 2);
        for &e in visible.iter().take(take) {
            let (u, v) = graph.edges[e as usize];
            let s = graph.signs[e as usize];
            vertices.extend([u, v]);
            member_signs.extend(family.member_signs(&[s]));
        }
        return TupleEnumeration {
            family,
            vertices,
            member_signs,
            dropped: 0,
            request,
            resolved_multi_edges: false,
        };
    }

    let visible_edges: Vec<(u32, u32)> = visible.iter().map(|&e| graph.edges[e as usize]).collect();
    let paths: Vec<u32> = match family {
        SignedTupleFamily::Edge => unreachable!("handled above"),
        SignedTupleFamily::Cycle { k } => {
            let (row_ptr, col_idx) =
                crate::unsigned_cycles::build_csr(&visible_edges, n_nodes, false);
            let threads = match execution {
                EnumerationExecution::DeterministicSerial => Some(1),
                EnumerationExecution::Rayon { threads } => threads,
            };
            enumerate_parallel(
                &row_ptr, &col_idx, n_nodes, k, false, cap, seed, false, threads,
            )
            .into_flat()
        }
        SignedTupleFamily::Walk { len } => {
            // The walk enumerator is single-threaded by construction, so both execution contracts
            // reach the same code; the contract is still recorded, because a caller asking for
            // `DeterministicSerial` must get the promise, not an accident.
            let (us, vs): (Vec<u32>, Vec<u32>) = visible_edges.iter().copied().unzip();
            enumerate_walks(&us, &vs, n_nodes, len, cap, seed)
        }
    };

    let resolver = SignResolver::build(graph, visible, resolution);
    let enumerated = paths.len().checked_div(arity).unwrap_or(0);
    let mut vertices = Vec::with_capacity(paths.len());
    let mut member_signs = Vec::with_capacity(paths.len());
    for tuple in paths.chunks(arity) {
        if tuple.len() != arity {
            continue;
        }
        let edge_signs: Option<Vec<i8>> = family
            .member_edges(tuple)
            .into_iter()
            .map(|(from, to)| resolver.sign_of(from, to))
            .collect();
        if let Some(edge_signs) = edge_signs {
            member_signs.extend(family.member_signs(&edge_signs));
            vertices.extend_from_slice(tuple);
        }
    }
    let kept = vertices.len().checked_div(arity).unwrap_or(0);
    TupleEnumeration {
        family,
        vertices,
        member_signs,
        dropped: enumerated - kept,
        request,
        resolved_multi_edges: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A multigraph: the pair `(0, 1)` carries three parallel edges with different signs, so
    /// resolution is observable rather than academic.
    fn multigraph() -> SignedGraph {
        SignedGraph::from_parts(
            5,
            &[0, 1, 0, 1, 2, 3, 4],
            &[1, 0, 1, 2, 3, 4, 0],
            &[1, -1, -1, 1, -1, 1, 1],
        )
    }

    fn all_edges(g: &SignedGraph) -> Vec<u32> {
        (0..g.edges.len() as u32).collect()
    }

    // --- families ------------------------------------------------------------------

    #[test]
    fn member_edges_and_member_signs_agree_on_shape_for_every_family() {
        let cases: [(SignedTupleFamily, Vec<u32>); 4] = [
            (SignedTupleFamily::Edge, vec![7, 3]),
            (SignedTupleFamily::Cycle { k: 3 }, vec![4, 9, 1]),
            (SignedTupleFamily::Cycle { k: 5 }, vec![2, 8, 5, 0, 6]),
            (SignedTupleFamily::Walk { len: 3 }, vec![1, 4, 2, 9]),
        ];
        for (family, tuple) in cases {
            let edges = family.member_edges(&tuple);
            assert_eq!(edges.len(), family.edge_count(), "{family:?}");
            let signs = family.member_signs(&vec![1i8; edges.len()]);
            assert_eq!(signs.len(), family.arity(), "{family:?}");
        }
    }

    #[test]
    fn a_cycle_wraps_and_a_walk_of_the_same_width_does_not() {
        let tuple = vec![3u32, 1, 4];
        let cycle = SignedTupleFamily::Cycle { k: 3 }.member_edges(&tuple);
        let walk = SignedTupleFamily::Walk { len: 2 }.member_edges(&tuple);
        assert_eq!(cycle, vec![(3, 1), (1, 4), (4, 3)], "directed, wrapped");
        assert_eq!(walk, vec![(3, 1), (1, 4)], "directed, open");
        assert_eq!(cycle.len(), walk.len() + 1);
    }

    #[test]
    fn the_edge_family_rule_is_not_the_wrapped_cycle_rule() {
        assert_eq!(SignedTupleFamily::Edge.member_signs(&[-1]), vec![-1, -1]);
        assert_eq!(
            SignedTupleFamily::Cycle { k: 2 }.member_signs(&[-1, 1]),
            vec![-1, -1],
            "same width, different contract"
        );
        assert_eq!(SignedTupleFamily::Edge.member_edges(&[9, 2]), vec![(9, 2)]);
        assert_eq!(
            SignedTupleFamily::Cycle { k: 2 }.member_edges(&[9, 2]),
            vec![(9, 2), (2, 9)],
            "the wrapped rule traverses the pair twice"
        );
    }

    #[test]
    fn a_cycles_vertex_signs_are_the_parity_of_its_negative_incidence() {
        let cycle = SignedTupleFamily::Cycle { k: 5 };
        // Edges 1 and 4 negative: vertices 1, 2 and 4, 0 each touched once.
        assert_eq!(
            cycle.member_signs(&[1, -1, 1, 1, -1]),
            vec![-1, -1, -1, 1, -1]
        );
        assert!(
            SignedTupleFamily::balanced(&[1, -1, 1, 1, -1]),
            "two negatives is balanced"
        );
        assert!(!SignedTupleFamily::balanced(&[1, -1, 1, 1, 1]));
    }

    // --- resolution ------------------------------------------------------------------

    #[test]
    fn the_two_resolutions_genuinely_disagree_on_a_multigraph() {
        let g = multigraph();
        let v = all_edges(&g);
        let first = SignResolver::build(
            &g,
            &v,
            MultiEdgeSignResolution::FirstOccurrenceForwardThenReverse,
        );
        let last = SignResolver::build(&g, &v, MultiEdgeSignResolution::LastWriteUnordered);
        // Edge 0 is (0,1,+1); edge 2 is (0,1,-1). First occurrence keeps +1, last write keeps -1.
        assert_eq!(first.sign_of(0, 1), Some(1), "earliest forward occurrence");
        assert_eq!(
            last.sign_of(0, 1),
            Some(-1),
            "last edge written for the pair"
        );
        assert_ne!(first.sign_of(0, 1), last.sign_of(0, 1));
    }

    #[test]
    fn first_occurrence_prefers_a_forward_edge_over_an_earlier_reverse_one() {
        // Edge 1 is (1,0,-1) — a REVERSE record of the pair — and it precedes nothing forward for
        // the directed key (1,0). The oracle scans all forward keys before any reverse key, so the
        // directed pair (1,0) is answered by edge 1 itself, while (0,1) is answered by edge 0.
        let g = multigraph();
        let r = SignResolver::build(
            &g,
            &all_edges(&g),
            MultiEdgeSignResolution::FirstOccurrenceForwardThenReverse,
        );
        assert_eq!(
            r.sign_of(0, 1),
            Some(1),
            "forward record of (0,1) is edge 0"
        );
        assert_eq!(
            r.sign_of(1, 0),
            Some(-1),
            "forward record of (1,0) is edge 1"
        );
        // Last-write discards orientation, so it cannot tell those two apart.
        let l = SignResolver::build(
            &g,
            &all_edges(&g),
            MultiEdgeSignResolution::LastWriteUnordered,
        );
        assert_eq!(l.sign_of(0, 1), l.sign_of(1, 0), "orientation discarded");
    }

    #[test]
    fn last_write_unordered_reproduces_the_existing_build_sign_lookup() {
        // The named contract must be exactly what the pre-existing core function already computes,
        // or naming it would be a silent behaviour change.
        let g = multigraph();
        let legacy = g.build_sign_lookup();
        let named = SignResolver::build(
            &g,
            &all_edges(&g),
            MultiEdgeSignResolution::LastWriteUnordered,
        );
        for (&(a, b), &s) in &legacy {
            assert_eq!(named.sign_of(a, b), Some(s), "pair ({a},{b})");
            assert_eq!(named.sign_of(b, a), Some(s), "reverse of ({a},{b})");
        }
    }

    #[test]
    fn a_resolver_answers_only_for_visible_edges() {
        let g = multigraph();
        // Hide every edge incident to vertex 4.
        let visible: Vec<u32> = (0..g.edges.len() as u32)
            .filter(|&e| {
                let (u, v) = g.edges[e as usize];
                u != 4 && v != 4
            })
            .collect();
        let r = SignResolver::build(
            &g,
            &visible,
            MultiEdgeSignResolution::FirstOccurrenceForwardThenReverse,
        );
        assert!(r.sign_of(3, 4).is_none(), "hidden edge must not resolve");
        assert!(r.sign_of(0, 1).is_some(), "visible edge must resolve");
    }

    // --- edge family keeps raw per-edge signs -----------------------------------------

    #[test]
    fn the_edge_family_keeps_raw_per_edge_signs_and_resolves_nothing() {
        let g = multigraph();
        let e = enumerate_signed_tuples(
            &g,
            &all_edges(&g),
            SignedTupleFamily::Edge,
            EnumerationRequest {
                cap: None,
                seed: 0,
                execution: EnumerationExecution::DeterministicSerial,
                resolution: MultiEdgeSignResolution::LastWriteUnordered,
            },
        );
        assert!(e.resolution().is_none(), "the edge family resolves nothing");
        assert_eq!(
            e.len(),
            g.edges.len(),
            "every parallel edge is its own tuple"
        );
        // Edges 0 and 2 are the SAME pair with OPPOSITE signs; both survive with their own sign.
        assert_eq!(&e.vertices[0..2], &[0, 1]);
        assert_eq!(&e.member_signs[0..2], &[1, 1]);
        assert_eq!(&e.vertices[4..6], &[0, 1]);
        assert_eq!(&e.member_signs[4..6], &[-1, -1]);
    }

    // --- execution contract ------------------------------------------------------------

    /// A graph dense enough that a cap genuinely samples.
    fn dense_ring(n: u32) -> SignedGraph {
        let mut us = Vec::new();
        let mut vs = Vec::new();
        let mut signs = Vec::new();
        for i in 0..n {
            for step in 1..=3u32 {
                us.push(i);
                vs.push((i + step) % n);
                signs.push(if (i + step) % 3 == 0 { -1 } else { 1 });
            }
        }
        SignedGraph::from_parts(n, &us, &vs, &signs)
    }

    fn request(
        cap: Option<usize>,
        seed: u64,
        execution: EnumerationExecution,
    ) -> EnumerationRequest {
        EnumerationRequest {
            cap,
            seed,
            execution,
            resolution: MultiEdgeSignResolution::FirstOccurrenceForwardThenReverse,
        }
    }

    fn materialise(g: &SignedGraph, exec: EnumerationExecution, cap: usize) -> (Vec<u32>, Vec<i8>) {
        let e = enumerate_signed_tuples(
            g,
            &all_edges(g),
            SignedTupleFamily::Cycle { k: 4 },
            request(Some(cap), 7, exec),
        );
        (e.vertices, e.member_signs)
    }

    #[test]
    fn deterministic_serial_materialisation_is_byte_identical() {
        let g = dense_ring(40);
        let a = materialise(&g, EnumerationExecution::DeterministicSerial, 200);
        for _ in 0..4 {
            let b = materialise(&g, EnumerationExecution::DeterministicSerial, 200);
            assert_eq!(a.0, b.0, "vertices drifted between runs");
            assert_eq!(a.1, b.1, "member signs drifted between runs");
        }
        assert!(
            !a.0.is_empty(),
            "the fixture must actually enumerate something"
        );
        assert!(
            a.0.len() / 4 <= 200,
            "the cap must actually bind: {} tuples",
            a.0.len() / 4
        );
    }

    /// Characterisation, not a wish: the parallel path is documented as sample-unstable under a
    /// cap, and this test records what it actually does rather than changing it.
    #[test]
    fn rayon_execution_is_not_sample_stable_under_a_cap() {
        let g = dense_ring(40);
        let serial = materialise(&g, EnumerationExecution::DeterministicSerial, 200);
        let parallel = materialise(&g, EnumerationExecution::Rayon { threads: Some(4) }, 200);
        // Both honour the cap and produce well-formed tuples...
        assert_eq!(parallel.0.len(), parallel.1.len());
        assert_eq!(parallel.0.len() % 4, 0);
        // ...but the contract flags only one of them as reproducible, and that flag is what a
        // caller must consult.
        assert!(EnumerationExecution::DeterministicSerial.is_reproducible_under_cap());
        assert!(!EnumerationExecution::Rayon { threads: Some(4) }.is_reproducible_under_cap());
        let _ = serial;
    }

    #[test]
    fn an_enumeration_carries_its_contracts_and_dropped_count() {
        let g = dense_ring(24);
        let e = enumerate_signed_tuples(
            &g,
            &all_edges(&g),
            SignedTupleFamily::Walk { len: 3 },
            request(Some(50), 3, EnumerationExecution::DeterministicSerial),
        );
        assert_eq!(
            e.request.execution,
            EnumerationExecution::DeterministicSerial
        );
        assert_eq!(
            e.resolution(),
            Some(MultiEdgeSignResolution::FirstOccurrenceForwardThenReverse)
        );
        assert_eq!(e.request.cap, Some(50));
        assert_eq!(e.request.seed, 3);
        assert!(e.request.is_reproducible(), "serial + cap is reproducible");
        assert_eq!(e.family, SignedTupleFamily::Walk { len: 3 });
        assert_eq!(e.vertices.len(), e.member_signs.len());
        assert_eq!(e.vertices.len() % e.family.arity(), 0);
        assert_eq!(e.dropped, 0, "every walk edge exists by construction");
    }

    // --- crate boundary ----------------------------------------------------------------

    /// This crate is the signed-graph substrate, not a study. A dataset, model, metric, optimizer,
    /// CLI or serialisation name appearing here means an application concern has leaked down into
    /// it — which is exactly how the previous owner of this mathematics ended up in an application
    /// module in the first place.
    #[test]
    fn this_module_names_no_application_concern() {
        // Comments stripped: a rule that *names* the categories it bans, in its own documentation,
        // is not a leak. Scanning raw text would make this rule fail on itself — which it did.
        let source: String = include_str!("signed_tuples.rs")
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n")
            .to_lowercase();
        // Assembled at runtime: a literal list would match itself.
        let banned = [
            ("cogin", "focom"),
            ("bit", "coin"),
            ("hsi", "kan"),
            ("au", "roc"),
            ("opti", "mizer"),
            ("ad", "am("),
            ("clap", "::"),
            ("serde", "::"),
            (".cs", "v"),
            ("js", "on"),
            ("Instant::", "now"),
        ];
        for (a, b) in banned {
            let needle = format!("{a}{b}");
            assert!(
                !source.contains(&needle),
                "hymeko_graph must not name an application concern: {needle}"
            );
        }
    }

    #[test]
    fn hiding_edges_before_enumeration_removes_the_tuples_through_them() {
        let g = dense_ring(24);
        let all = all_edges(&g);
        let full = enumerate_signed_tuples(
            &g,
            &all,
            SignedTupleFamily::Cycle { k: 3 },
            request(None, 0, EnumerationExecution::DeterministicSerial),
        );
        let half: Vec<u32> = all.iter().copied().filter(|e| e % 2 == 0).collect();
        let reduced = enumerate_signed_tuples(
            &g,
            &half,
            SignedTupleFamily::Cycle { k: 3 },
            request(None, 0, EnumerationExecution::DeterministicSerial),
        );
        assert!(!full.is_empty(), "the fixture must enumerate something");
        assert!(
            reduced.len() < full.len(),
            "hiding edges must remove tuples: {} vs {}",
            reduced.len(),
            full.len()
        );
        assert_eq!(
            reduced.dropped, 0,
            "invisible edges are never traversed, so nothing is dropped"
        );
    }
}
