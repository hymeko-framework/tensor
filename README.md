# tensor — HyMeKo Tensor pillar

The mathematical foundation and streamed representation of the HyMeKo **Tensor** pillar, as a standalone Cargo
workspace.

## Crates

- **`hymeko_tensor`** — matrix-based Rényi-α=2 information estimators (`renyi`: Gram, `S₂` entropy, mutual
  information + gradient, Chebyshev features, stable rank) and the streamed HyMeKo IR format (`stream`:
  `HyMeKoStream` = `{dims, single flat arena, cheby_order}` with a bit-exact encode/decode round-trip). Pure,
  `std`-only, no dependencies.
- **`hymeko_clifford`** — Clifford / geometric-algebra foundation: dense `Multivector<N>`, signature, blade product,
  `Rotor`. `std`-only.
- **`hymeko_graph`** — signed-incidence hypergraph algorithms (the native hyperconv substrate).

## Provenance

Extracted 2026-08-16 from `nagare` (`feat/auto-holonomy-arc`, examples/nagare_deep_signed.rs): the matrix-Rényi
estimators and the streamed IR (`HyMeKoStream`) developed in the information-theoretic arc; `hymeko_clifford` /
`hymeko_graph` mirrored from the framework. The Nagare learner still carries its own copies pending a rewire to
depend on this workspace.

## Build

```
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```
