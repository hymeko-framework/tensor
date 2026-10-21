# tensor - HyMeKo numerical foundations

Tensor owns numerical storage, kernels and arithmetic contracts shared by the
HyMeKo framework. Core owns the canonical semantic IR; Graph owns graph algorithms.
This workspace contains no second graph implementation or semantic IR.

## Crates

| Crate | Responsibility |
|---|---|
| `hymeko_tensor` | Tensor/CSR storage, numerical kernels, host compute arenas, information estimators, virtual row operators and contraction planning |
| `hymeko_clifford` | Clifford algebra, rotors and rigid-motion primitives |
| `hymeko_spline` | Spline kernels, including Catmull-Rom and NURBS |
| `hymeko_numerics` | Matrix-free iterative solvers and port-Hamiltonian numerical contracts |
| `hymeko_vision` | Vision primitives |
| `hymeko_numeric_vv` | Downstream numerical verification with independent/frozen oracles |
| `hymeko_tensor_py` | Optional Python bindings to Tensor-owned functionality |

## Virtual operators and bounded planning

`operator` provides ordered sparse-row gather/scatter, independent accumulator
policies and range contractions without materializing a matrix product. Caller-owned
workspaces retain storage across calls. `contraction` provides bounded branch-and-bound
planning and sparse matrix-chain execution; reassociation is explicit because it
can change floating-point results. Shannon entropy is an ordering descriptor, not
a certified pruning bound.

`Pool::reclaim_completed_budgeted(watermark, max_checks)` inspects at most the stated
number of retired resources. Incomplete epochs rotate behind the others and open
sessions stay protected. The limit bounds epoch checks; allocation, eviction and
backend destruction retain their own costs.

## Validation and results

The October 2026 release preparation passed **52 focused correctness tests**:
33 compute-arena, 7 contraction-planning, 4 sparse matrix-chain and 8 virtual-operator
tests. They cover invalid inputs, independent numerical references, ordering,
buffer reuse and bounded reclamation. Run them with:

```sh
cargo test -p hymeko_tensor --features sparse --test compute_arena --test contraction_planning --test matrix_chain --test virtual_operators
```

The virtual operators are used by Core incidence projections and Graph PageRank.
[Graph's retained performance evidence](https://github.com/hymeko-framework/graph/tree/main/docs/evidence/graph-algorithms)
reports complete graph operations on explicit workloads. Those results are not
standalone Tensor-kernel timings. No speedup is inferred from refactoring alone.

## Build

```sh
cargo build -p hymeko_tensor --features sparse
```

Default Tensor features are empty. Sparse parallel kernels, interop and Arrow
support are explicit features; Python bindings are a separate crate. Criterion
benchmarks require explicit `cargo bench` invocation and are not the correctness
commands above. The workspace does not track Cargo.lock; retain the resolved lock
and toolchain when reproducing an experiment.

## Provenance and license

The repositories were split from the earlier HyMeKo workspace. Tensor received
Core's generic numerical storage and compute-arena contracts in September 2026.
Use the current owning repositories rather than historical copied implementations.
See [LICENSE](LICENSE) and individual crate metadata for licensing declarations.
