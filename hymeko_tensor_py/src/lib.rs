//! # hymeko_tensor_py — the Tensor pillar, callable from Python
//!
//! A thin, total binding over [`hymeko_tensor`]. It adds no mathematics: every number returned here is produced by
//! the compiled crate, so a Python consumer and a Rust consumer cannot drift. The binding's own job is the
//! boundary — turning documented preconditions into `ValueError` instead of an out-of-bounds abort, and mapping
//! [`hymeko_tensor::linalg::CholeskyError`] onto Python exceptions.
//!
//! ## Why this exists
//!
//! `hymeko_tensor`'s `renyi` functions were once copied into a `nagare` example and lived there for months as a
//! second implementation (retired 2026-08-19). The same pressure exists on the Python side: without a binding, a
//! consumer that needs a Gram matrix or a stable rank writes one in NumPy, and the copy becomes permanent. A
//! binding is cheaper than a duplicate and cannot disagree with the owner.
//!
//! ## `unsafe_code` is not forbidden here, and that is deliberate
//!
//! The framework bar (CLAUDE.md §2) is `forbid(unsafe_code)`. That is impossible in a pyo3 extension: the
//! `#[pymodule]` and `#[pyfunction]` macros expand to FFI glue containing `unsafe` blocks, so `forbid` would fail
//! to compile the generated code rather than reveal anything about the code written here. No `unsafe` is written
//! by hand in this crate. The core crates it binds keep `forbid(unsafe_code)` untouched — which is the point of
//! putting the FFI in a separate downstream package.
//!
//! ## Data
//!
//! Matrices are flat, row-major `f32` sequences with explicit dimensions, exactly as the Rust API takes them. No
//! NumPy dependency: the crate stays on `pyo3` alone, and `numpy.asarray(x, dtype=numpy.float32).ravel()` on the
//! caller's side is a copy the caller can see rather than one hidden in a conversion layer.

#![deny(missing_docs)]
#![deny(clippy::all)]

mod shape;

use pyo3::exceptions::{PyArithmeticError, PyValueError};
use pyo3::prelude::*;

use hymeko_tensor::linalg::{self, CholeskyError};
use hymeko_tensor::renyi::{self, MiKernel};

fn shape_err(e: shape::ShapeError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// Linear (degree-1 polynomial) Gram matrix `K[i][j] = <row_i, row_j>`.
///
/// Args: `a` flat `m*d` row-major, `m` rows, `d` width. Returns the flat `m*m` symmetric Gram.
///
/// Raises `ValueError` if `len(a) != m*d`, `m < 2` or `d < 1`.
#[pyfunction]
#[pyo3(signature = (a, m, d))]
fn linear_gram(a: Vec<f32>, m: usize, d: usize) -> PyResult<Vec<f32>> {
    shape::batch(m)
        .and_then(|()| shape::width(d))
        .map_err(shape_err)?;
    shape::matrix("a", a.len(), m, d).map_err(shape_err)?;
    Ok(renyi::linear_gram(&a, m, d))
}

/// Matrix-based Renyi-alpha=2 entropy `S2` of a Gram matrix.
///
/// Args: `k` flat `m*m` Gram, `m` rows. Raises `ValueError` on a bad shape.
#[pyfunction]
#[pyo3(signature = (k, m))]
fn s2_entropy(k: Vec<f32>, m: usize) -> PyResult<f32> {
    shape::batch(m).map_err(shape_err)?;
    shape::matrix("k", k.len(), m, m).map_err(shape_err)?;
    Ok(renyi::s2_entropy(&k, m))
}

/// Mutual information `I(A;Y)` from two Gram matrices, with the per-entry coefficient matrix.
///
/// Args: `ka`, `ky` flat `m*m`, `m` rows. Returns `(mi, coeff)` with `len(coeff) == m*m`.
#[pyfunction]
#[pyo3(signature = (ka, ky, m))]
fn mi_coeff(ka: Vec<f32>, ky: Vec<f32>, m: usize) -> PyResult<(f32, Vec<f32>)> {
    shape::batch(m).map_err(shape_err)?;
    shape::matrix("ka", ka.len(), m, m).map_err(shape_err)?;
    shape::matrix("ky", ky.len(), m, m).map_err(shape_err)?;
    Ok(renyi::mi_coeff(&ka, &ky, m))
}

/// Renyi-2 mutual information and `dI/dA` under a kernel.
///
/// Args: `a` flat `m*d`, `ky` flat `m*m`, `m`, `d`, `kernel` either `"linear"` or `"cheby"`, and `cheby_order`
/// (required, and only meaningful, when `kernel == "cheby"`). Returns `(mi, grad)` with `len(grad) == m*d`.
///
/// Raises `ValueError` on a bad shape, an unknown kernel name, or a `"cheby"` request with no order.
#[pyfunction]
#[pyo3(signature = (a, ky, m, d, kernel="linear", cheby_order=None))]
fn renyi2_mi_grad(
    a: Vec<f32>,
    ky: Vec<f32>,
    m: usize,
    d: usize,
    kernel: &str,
    cheby_order: Option<usize>,
) -> PyResult<(f32, Vec<f32>)> {
    shape::batch(m)
        .and_then(|()| shape::width(d))
        .map_err(shape_err)?;
    shape::matrix("a", a.len(), m, d).map_err(shape_err)?;
    shape::matrix("ky", ky.len(), m, m).map_err(shape_err)?;
    let k = match (kernel, cheby_order) {
        ("linear", _) => MiKernel::Linear,
        ("cheby", Some(o)) if o >= 1 => MiKernel::Cheby(o),
        ("cheby", Some(_)) => return Err(PyValueError::new_err("cheby_order must be >= 1")),
        ("cheby", None) => {
            return Err(PyValueError::new_err("kernel='cheby' requires cheby_order"))
        }
        (other, _) => {
            return Err(PyValueError::new_err(format!(
                "unknown kernel {other:?}: expected 'linear' or 'cheby'"
            )))
        }
    };
    Ok(renyi::renyi2_mi_grad(&a, &ky, m, d, k))
}

/// Chebyshev feature map of order `kf`. Returns `(features, gram)` as produced by the owner.
#[pyfunction]
#[pyo3(signature = (a, m, d, kf))]
fn cheby_features(a: Vec<f32>, m: usize, d: usize, kf: usize) -> PyResult<(Vec<f32>, Vec<f32>)> {
    shape::batch(m)
        .and_then(|()| shape::width(d))
        .map_err(shape_err)?;
    shape::matrix("a", a.len(), m, d).map_err(shape_err)?;
    if kf < 1 {
        return Err(PyValueError::new_err("kf: feature-map order must be >= 1"));
    }
    Ok(renyi::cheby_features(&a, m, d, kf))
}

/// Stable rank of a flat `m*d` row-major matrix: `||A||_F^2 / ||A||_2^2`.
///
/// A continuous, outcome-blind measure of how many directions a set of rows actually occupies -- the numeric
/// question an applicability study asks when it wants "is this set degenerate?" without thresholding a rank.
#[pyfunction]
#[pyo3(signature = (a, d))]
fn stable_rank(a: Vec<f32>, d: usize) -> PyResult<f32> {
    shape::width(d).map_err(shape_err)?;
    if !a.len().is_multiple_of(d) {
        return Err(PyValueError::new_err(format!(
            "a: length {} is not a multiple of d={d}",
            a.len()
        )));
    }
    Ok(renyi::stable_rank(&a, d))
}

/// Dense Cholesky solve of `M x = b` for row-major SPD `M` of order `n`.
///
/// Raises `ValueError` on a shape mismatch and `ArithmeticError` naming the offending diagonal index when `M` is
/// not positive definite -- the owner returns that as a value, and it stays one here rather than becoming a `NaN`.
#[pyfunction]
#[pyo3(signature = (m, b, n))]
fn dense_cholesky_solve(m: Vec<f32>, b: Vec<f32>, n: usize) -> PyResult<Vec<f32>> {
    linalg::dense_cholesky_solve(&m, &b, n).map_err(|e| match e {
        CholeskyError::ShapeMismatch => PyValueError::new_err(format!(
            "expected m of {n}x{n} = {} elements and b of {n}, got {} and {}",
            n * n,
            m.len(),
            b.len()
        )),
        CholeskyError::NotPositiveDefinite { index } => PyArithmeticError::new_err(format!(
            "matrix is not positive definite: non-positive pivot at diagonal index {index}"
        )),
    })
}

/// The Tensor pillar's numeric surface.
#[pymodule]
fn hymeko_tensor_py(mod_: &Bound<'_, PyModule>) -> PyResult<()> {
    mod_.add(
        "__doc__",
        "Compiled bindings to the hymeko_tensor crate (Renyi-2 estimators, dense linear algebra).",
    )?;
    mod_.add_function(wrap_pyfunction!(linear_gram, mod_)?)?;
    mod_.add_function(wrap_pyfunction!(s2_entropy, mod_)?)?;
    mod_.add_function(wrap_pyfunction!(mi_coeff, mod_)?)?;
    mod_.add_function(wrap_pyfunction!(renyi2_mi_grad, mod_)?)?;
    mod_.add_function(wrap_pyfunction!(cheby_features, mod_)?)?;
    mod_.add_function(wrap_pyfunction!(stable_rank, mod_)?)?;
    mod_.add_function(wrap_pyfunction!(dense_cholesky_solve, mod_)?)?;
    Ok(())
}
