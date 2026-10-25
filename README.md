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

`hymeko_numerics::GmresWorkspace` provides reusable, dependency-free restarted
GMRES storage for `(I - theta S)x = b`. Its `solve` method accepts a `GmresConfig`
with explicit `GmresTolerance::Relative` or `Absolute`, restart and iteration
budget. The existing `gmres(...)` facade calls this same engine with a temporary
workspace. Operator implementations still supply only `dim` and `apply`.

The workspace retains a contiguous basis and the small Hessenberg/rotation
buffers between restarts and calls. A warmed solve allocates nothing when its
buffers fit retained capacity and the operator itself does not allocate. It
retains O(n * restart) storage; this is not a claim of lower resident memory.
The true residual is checked at return, and `Report::residual` remains relative
even when the stopping threshold is absolute. The 144 pre-change regression
cases preserve solution/residual bits, iteration counts and operator-call counts.

The Hungarian mathematical summary is
[`docs/gmres/mathematical-summary.pdf`](docs/gmres/mathematical-summary.pdf), with
its LaTeX source beside it; the numerical owner,
scope and verification plan are in `docs/plans/2026-10-25-gmres-workspace/`.
This allocation refactor has no new latency measurement yet.

The existing compute arena supports
`Pool::reclaim_completed_budgeted(watermark, max_checks)`. Each call inspects at
most `max_checks` retired resources; incomplete epochs rotate behind the others.
Pending sessions remain protected and cache byte/buffer budgets still apply.
The return value counts completed resources removed from retirement, including
ones evicted instead of cached. The original `reclaim_completed` checks all
currently retired resources through the same implementation. The budget bounds
epoch checks; free-list eviction, allocation and backend destruction retain
their existing costs. Reclamation visits resources in FIFO rotation order.

```
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```
