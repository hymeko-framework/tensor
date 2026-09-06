//! Shape preconditions, as values rather than panics.
//!
//! Every function in [`hymeko_tensor::renyi`] documents preconditions (`a.len() == m·d`, `m >= 2`, …) and is
//! written to be fast, not defensive: violating one indexes out of bounds and ABORTS the process. An abort inside
//! a Python extension takes the interpreter with it, so the caller loses the traceback, the REPL and any unsaved
//! work -- for what is merely a wrong argument.
//!
//! This module turns each documented precondition into a checked value the binding converts to `ValueError`. It is
//! plain Rust with no `pyo3` in scope, so `cargo test` exercises the whole contract without an interpreter.

/// A precondition that did not hold, with the numbers that broke it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

impl core::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// `len == rows * cols`, naming the argument and both shapes when it does not.
///
/// # Errors
/// [`ShapeError`] describing the expected and actual length.
pub fn matrix(name: &str, len: usize, rows: usize, cols: usize) -> Result<(), ShapeError> {
    let want = rows
        .checked_mul(cols)
        .ok_or_else(|| ShapeError(format!("{name}: {rows}x{cols} overflows usize")))?;
    if len != want {
        return Err(ShapeError(format!(
            "{name}: expected {rows}x{cols} = {want} elements, got {len}"
        )));
    }
    Ok(())
}

/// `m >= 2`: the Renyi-2 estimators are defined on a minibatch, and a single row has no pairwise structure.
///
/// # Errors
/// [`ShapeError`] when `m < 2`.
pub fn batch(m: usize) -> Result<(), ShapeError> {
    if m < 2 {
        return Err(ShapeError(format!(
            "m: the estimators need at least 2 rows, got {m}"
        )));
    }
    Ok(())
}

/// `d >= 1`: a zero-width feature block carries no information and would divide by a zero trace.
///
/// # Errors
/// [`ShapeError`] when `d == 0`.
pub fn width(d: usize) -> Result<(), ShapeError> {
    if d == 0 {
        return Err(ShapeError(
            "d: feature width must be at least 1, got 0".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_correct_matrix_shape_is_accepted() {
        assert!(matrix("a", 12, 4, 3).is_ok());
    }

    #[test]
    fn a_wrong_matrix_shape_names_both_the_expectation_and_the_reality() {
        let e = matrix("a", 11, 4, 3).unwrap_err();
        assert!(e.0.contains("expected 4x3 = 12"), "{}", e.0);
        assert!(e.0.contains("got 11"), "{}", e.0);
    }

    #[test]
    fn an_overflowing_shape_is_an_error_not_a_wrapped_product() {
        // rows*cols wraps on a 64-bit usize; a wrapped product could accidentally EQUAL len and pass.
        let huge = usize::MAX / 2 + 1;
        assert!(matrix("a", 0, huge, 4).is_err());
    }

    #[test]
    fn the_batch_and_width_minima_are_enforced() {
        assert!(batch(1).is_err() && batch(2).is_ok());
        assert!(width(0).is_err() && width(1).is_ok());
    }
}
