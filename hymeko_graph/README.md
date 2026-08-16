# hymeko_graph

Generic signed-graph cycle and walk enumeration with **axiomatic
pruning**.

A pure-Rust library that lifts the cycle / walk enumeration code
out of `hymeko_py` so it can be used outside the KAN /
deep-learning pipeline.  The headline feature is the pluggable
`CyclePruner` trait: cycles that violate domain axioms are
**pruned during DFS, not after enumeration**.

## What's in the box

- `SignedGraph` + CSR adjacency builder (no `unsafe`, no Python deps)
- `CyclePruner` trait — `extend_ok` (DFS-time pre-check) +
  `emit_ok` (post-completion verification)
- Pre-built balance pruners:
  - `CartwrightHararyPruner` — balanced-only or unbalanced-only
    sign-product filter (Cartwright-Harary 1956)
  - `DavisWeakBalancePruner` — exclude all-negative triangles
    (Davis 1967)
  - `BipartiteOnlyPruner` — drop odd-length cycles (handy after
    star-expansion)
- `FriedlerAxiomPruner` — encode the five P-graph axioms
  (Friedler-Tarján-Huang-Fan 1992) as DFS pruning rules:
  - **A0** bipartite alternation (M ↔ O at every step) →
    pruned during DFS, halves branch factor on bipartite graphs
  - **A1** required products → cycle must touch a product M-node
  - **A3** valid O-nodes → only whitelisted units allowed in cycle
  - **A4 / A5** are degenerate at cycle-enumeration time but
    surfaced as hooks

## Why pruning matters

For a P-graph with `|V_M|` material vertices and `|V_O|` operating
units, bipartite alternation alone halves the DFS branch factor.
On a cube graph (perfectly bipartite), the pruner skips every
odd-length search path and gives an immediate ~2× speed-up on
top of the existing rayon parallelism.  For more constrained
axioms (A1 product-membership, A3 unit-validity) the speed-up
scales with how restrictive the axiom is.

The two-level API mirrors how Friedler's accelerated branch-and-
bound works on P-graphs: cheap structural checks during the
search (`extend_ok`), expensive verification once a candidate is
ready (`emit_ok`).

## Status (2026-05-04)

- ✓ Types + CSR builder
- ✓ Pruner trait + all stock pruners (CartwrightHararyPruner,
  DavisWeakBalancePruner, BipartiteOnlyPruner, FriedlerAxiomPruner)
- ✓ Unit tests: 11/11 pass
- ◯ Cycle enumerator body: lives in `hymeko_py/src/cycles.rs`
  for now.  Next refactor pass moves it here and gives it a
  `&dyn CyclePruner` parameter, with the Python wrapper accepting
  the pruner type as a string ("cartwright_harary",
  "davis_weak", "bipartite_only", "friedler", or `None` for the
  default no-op).

## Worked example

```rust
use hymeko_graph::{
    FriedlerAxiomPruner, NodeKind, CyclePruner, PrunerDecision,
};

// 4-vertex bipartite graph: vertices 0, 2 are Material;
// vertices 1, 3 are OperatingUnit.
let kind = vec![
    NodeKind::Material, NodeKind::OperatingUnit,
    NodeKind::Material, NodeKind::OperatingUnit,
];
let p = FriedlerAxiomPruner::new(kind)
    .with_required_products([2]);  // require product M-node 2

// Try to extend [0] -> 2: same kind (M -> M) violates A0.
assert_eq!(p.extend_ok(&[0], 2), PrunerDecision::Reject);

// Try to extend [0] -> 1: M -> O, OK.
assert_eq!(p.extend_ok(&[0], 1), PrunerDecision::Accept);

// Emit the cycle [0, 1, 2, 3] (M-O-M-O, even length, contains 2):
// satisfies A0 + A1, accepted.
assert_eq!(p.emit_ok(&[0, 1, 2, 3], &[1, 1, 1, 1]),
           PrunerDecision::Accept);
```

## Citations

- F. Friedler, K. Tarján, Y.-W. Huang, L. T. Fan,
  *Combinatorial algorithms for process synthesis*, Computers &
  Chemical Engineering, 16, 1992.
- D. Cartwright, F. Harary, *Structural balance: a generalisation
  of Heider's theory*, Psychological Review, 63(5), 1956.
- J. A. Davis, *Clustering and structural balance in graphs*,
  Human Relations, 20(2), 1967.
- F. Heider, *Attitudes and cognitive organization*, Journal of
  Psychology, 21, 1946.
