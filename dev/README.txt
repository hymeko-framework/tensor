Native Tensor development linkage -- 2026-10-28

Run from the Tensor workspace:
  cargo dev-test --test virtual_operators --test contraction_planning
  cargo dev-test --test gmres_workspace --test port_hamiltonian
  cargo dev-test --test se_fixed --test spatial
  cargo dev-test --test build_contract --test scenario_feature_parity
  cargo dev-test --features hymeko-tensor-dev/sparse,hymeko-spline-dev/parallel --test matrix_chain --test sparse_authority

Four downstream clients register the original Tensor, Numerics, Clifford and Spline
integration-test source paths. Four tiny publish=false Rust DLL crates only
re-export their respective production libraries. No arithmetic, data structure,
test body or fixture is copied. Original type identity is retained.

Sparse features and Spline parallel execution are explicitly forwarded; they
remain off by default. The SSE4.1 rustflag is retained. The existing bit-pattern
and build-contract tests verify these obligations. Interop and arrow-schema are
also forwarded but are not enabled by ordinary DLL correctness commands.

Private library unit tests, custom-global-allocator tests, benchmark targets,
Python bindings and the source-inspecting V&V package retain their original
targets. Vision currently has no integration-test client, so no unused Vision
DLL is introduced. No production crate depends on the dev client or DLL layer.
The previous workspace members and ordinary static/release profiles remain the
defaults. This linkage change is not new numerical performance evidence.

The dev-dll profile is incremental with line-table debug information. It keeps
backtrace locations but omits full variable/type debug data. The alias puts its
intermediate files at
  {cargo-cache-home}/hymeko-dev-build/{workspace-path-hash}
so clients in one workspace share artifacts and distinct source roots cannot
reuse each other's local packages. A first build per workspace is required.
Do not override this with an unqualified shared build directory.

Cargo supplies DLL search paths. A copied EXE alone is not a standalone bundle:
it also needs the actually imported DLLs and the matching Rust standard library.
These libraries require the same compiler/dependency build; they are not a
stable C ABI or plugin-update boundary. Count shared DLLs once when comparing
the size of a group of clients. Generics can still instantiate code in clients;
a smaller EXE does not guarantee lower total disk usage, RAM or execution time.

Use Cargo compiler-artifact JSON to obtain actual executable paths. Repeating
the same inputs should report Fresh; source, feature, profile or toolchain
changes can legitimately rebuild them. Do not cargo clean the inherited shared
cache: it can contain other workspaces' intermediate files.

This library workspace intentionally does not track Cargo.lock. The alias
therefore permits initial resolution on a fresh checkout; once a local lockfile
exists, append --locked to require it. Verification records that lockfile hash.

Each client depends on only its own DLL. Feature or source changes in one
mathematical library do not add a dependency edge to the other test clients.
The common alias selects all four clients; --test narrows the requested targets.
