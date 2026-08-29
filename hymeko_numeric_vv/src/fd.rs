//! Central finite differences, for certifying an analytic gradient against the function it claims
//! to differentiate.
//!
//! # Why this is a V&V mechanism and not a production one
//!
//! Nothing in production wants a numerical gradient: every kernel in `hymeko_spline` ships a
//! closed-form backward, which is the point of the crate. What production cannot do is *check
//! itself* — an adjoint compared against the same adjoint proves nothing. Central differences are
//! the independent oracle for that comparison, and they belong on this side of the boundary.
//!
//! # The step, and why it is not a constant
//!
//! The truncation error of the central difference is `O(h^2)` and its cancellation error is
//! `O(eps/h)`, so a single global `h` is wrong for every magnitude but one. [`central`] takes the
//! step explicitly and [`relative_step`] derives one from the point being probed, so the choice is
//! always visible at the call site.

/// A finite-difference step scaled to the magnitude of the coordinate being probed.
///
/// `h = base * max(1, |x|)`. With `base = 1e-3` this is the step the spline suites use: `f32`
/// kernels evaluated in `f64` accumulation give ~4-5 correct digits there, which is why the
/// tolerances in those suites are `1e-3`-ish and not `1e-9`.
#[must_use]
pub fn relative_step(x: f64, base: f64) -> f64 {
    base * x.abs().max(1.0)
}

/// Central difference of `f` at coordinate `i` of `at`, with step `h`.
///
/// # Preconditions
/// `i < at.len()`, `h > 0`, and `f` must be evaluable at `at` with coordinate `i` moved by `±h`.
///
/// # Postconditions
/// Returns `(f(x + h e_i) - f(x - h e_i)) / (2h)`. `at` is restored before returning, so the
/// caller's buffer is unchanged — the invariant that lets this be called in a loop over every
/// coordinate.
///
/// # Panics
/// If `i` is out of range or `h` is not strictly positive.
pub fn central<F>(f: &mut F, at: &mut [f32], i: usize, h: f64) -> f64
where
    F: FnMut(&[f32]) -> f64,
{
    assert!(i < at.len(), "coordinate {i} out of range {}", at.len());
    assert!(h > 0.0, "step must be positive, got {h}");
    let saved = at[i];
    at[i] = (saved as f64 + h) as f32;
    let up = f(at);
    at[i] = (saved as f64 - h) as f32;
    let down = f(at);
    at[i] = saved;
    // The realised step, not the requested one: `f32` rounding of `saved ± h` is what the function
    // actually saw, and dividing by the requested `h` would charge that rounding to the gradient.
    let realised = ((saved as f64 + h) as f32 as f64) - ((saved as f64 - h) as f32 as f64);
    (up - down) / realised
}

/// Central differences at every coordinate of `at`.
///
/// # Postconditions
/// `out.len() == at.len()`; `at` is unchanged.
#[must_use]
pub fn gradient<F>(mut f: F, at: &[f32], base_step: f64) -> Vec<f64>
where
    F: FnMut(&[f32]) -> f64,
{
    let mut probe = at.to_vec();
    (0..at.len())
        .map(|i| {
            let h = relative_step(probe[i] as f64, base_step);
            central(&mut f, &mut probe, i, h)
        })
        .collect()
}

/// Largest `|a - b| / max(1, |b|)` over the pair of slices.
///
/// Relative to the *analytic* value `b`, floored at 1 so a near-zero reference does not inflate
/// the ratio into meaninglessness.
///
/// # Preconditions
/// `a.len() == b.len()`.
///
/// # Panics
/// If the lengths differ.
#[must_use]
pub fn max_rel_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "finite-difference length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_difference_of_a_quadratic_is_exact_in_the_limit() {
        // f(x) = 3x^2 + 2x  =>  f'(1) = 8
        let mut f = |x: &[f32]| 3.0 * (x[0] as f64) * (x[0] as f64) + 2.0 * (x[0] as f64);
        let mut at = [1.0f32];
        let g = central(&mut f, &mut at, 0, 1e-3);
        assert!((g - 8.0).abs() < 1e-2, "got {g}");
        assert_eq!(at[0], 1.0, "the probe buffer must be restored");
    }

    #[test]
    fn gradient_covers_every_coordinate() {
        // f(x) = x0^2 + 5 x1  =>  grad = (2 x0, 5)
        let f = |x: &[f32]| (x[0] as f64) * (x[0] as f64) + 5.0 * (x[1] as f64);
        let g = gradient(f, &[2.0, -1.0], 1e-3);
        assert_eq!(g.len(), 2);
        assert!((g[0] - 4.0).abs() < 1e-2, "{g:?}");
        assert!((g[1] - 5.0).abs() < 1e-2, "{g:?}");
    }

    #[test]
    fn relative_step_never_shrinks_below_the_base() {
        assert!((relative_step(0.0, 1e-3) - 1e-3).abs() < 1e-18);
        assert!((relative_step(100.0, 1e-3) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn max_rel_error_is_zero_on_identical_input() {
        assert_eq!(max_rel_error(&[1.0, -2.0], &[1.0, -2.0]), 0.0);
    }

    #[test]
    fn max_rel_error_takes_the_worst_coordinate() {
        let e = max_rel_error(&[1.0, 0.0], &[1.0, 4.0]);
        assert!((e - 1.0).abs() < 1e-15, "got {e}");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn max_rel_error_rejects_a_length_mismatch() {
        let _ = max_rel_error(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn a_zero_step_is_rejected() {
        let mut f = |_: &[f32]| 0.0;
        let mut at = [0.0f32];
        let _ = central(&mut f, &mut at, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn an_out_of_range_coordinate_is_rejected() {
        let mut f = |_: &[f32]| 0.0;
        let mut at = [0.0f32];
        let _ = central(&mut f, &mut at, 5, 1e-3);
    }
}
