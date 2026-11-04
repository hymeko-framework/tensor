# tensor

Numerical storage, kernels and solvers for the HyMeKo framework, as one Cargo
workspace. The canonical semantic IR lives in
[core](https://github.com/hymeko-framework/core) and the hypergraph algorithms
in [graph](https://github.com/hymeko-framework/graph); this workspace depends
on neither.

## Crates

| Crate | Contents |
|---|---|
| `hymeko_tensor` | Scalar algebra, dense solvers (`linalg`), forward-mode duals (`dual`), box-bounded minimisation (`optimize`), matrix-based Rényi-2 estimators (`renyi`), parameter layout (`stream`), virtual sparse-row operators (`operator`), bounded contraction planning (`contraction`) and host compute arenas (`compute`) |
| `hymeko_clifford` | Clifford algebra: metric signature, dense multivectors, blade product, rotors and reverse-mode gradients for rotor chains |
| `hymeko_spline` | Catmull-Rom, Chebyshev-CR, Kochanek-Bartels and NURBS kernels on flat `f32` slices, their closed forms and orthtree knots |
| `hymeko_numerics` | Matrix-free solvers: restarted GMRES with a reusable workspace, preconditioned CG, Lanczos, L-BFGS, trust region, ADMM, FAS multigrid, Gauss-Newton and IRLS least squares, weighted diffusion and port-Hamiltonian operators |
| `hymeko_vision` | Skeleton rasterisation and masking primitives |
| `hymeko_numeric_vv` | Verification of the crates above: frozen oracles, finite-difference checks and manifest gates; not published, and no production crate depends on it |
| `hymeko_tensor_py` | Python (pyo3) bindings to `hymeko_tensor`; not published |

`hymeko_tensor`, `hymeko_clifford`, `hymeko_numerics` and `hymeko_vision` have
no dependencies in their default builds. `hymeko_tensor` adds COO/CSR storage
and Rayon kernels with `sparse`, Nalgebra conversion with `interop` and Arrow
schemas with `arrow-schema`. `hymeko_spline` uses Rayon only with `parallel`;
its results are byte-identical with the feature on or off.

## Use

```toml
[dependencies]
hymeko_tensor = { git = "https://github.com/hymeko-framework/tensor.git" }
hymeko_numerics = { git = "https://github.com/hymeko-framework/tensor.git" }
```

## Build and test

```sh
cargo test
cargo test -p hymeko_tensor --features sparse
cargo clippy --all-targets -- -D warnings
```

`cargo test` covers the default members, every crate in the table.
`.cargo/config.toml` enables SSE4.1 on x86-64, so the Catmull-Rom `floor`
compiles to one instruction instead of a libm call; the results are
bit-identical either way. The workspace does not track `Cargo.lock`.

The Python bindings build with [maturin](https://www.maturin.rs/) into one
abi3 wheel for CPython 3.9 and later:
`maturin build --release -m hymeko_tensor_py/pyproject.toml`.

`cargo dev-test` runs the integration tests against dynamic-library builds of
the four production libraries, which shortens incremental rebuilds; it is
described in [dev/README.txt](dev/README.txt). The default static build is
unchanged by it.

## Documentation

[docs/gmres/mathematical-summary.pdf](docs/gmres/mathematical-summary.pdf) is a
mathematical summary (in Hungarian) of the GMRES workspace, with its LaTeX
source beside it.

## License

MIT; see [LICENSE](LICENSE).
