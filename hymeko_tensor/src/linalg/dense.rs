//! Dense `f64` kernels for small systems: pivoted elimination and damped least squares.
//!
//! One implementation for every small dense solve of the kinematics and contact code: the
//! contact-replay experiments of `hymeko_geometry`, its chain inverse kinematics, and the
//! controllers built on that. Moved here from `hymeko_geometry::contact::replay::dense`
//! (2026-10-21) without changing an operation.
//!
//! # Pinned policies
//!
//! Unlike the `f32` solvers of the parent module these work in `f64`. A consumer reproducing a
//! recorded number depends on every one of these:
//!
//! | policy | choice |
//! |---|---|
//! | pivoting | partial: the row of largest magnitude in the column, the last of equal ones |
//! | singularity | a pivot of at most `64 ε max|a_ij|` (over the input matrix) fails the solve |
//! | failure | `None` for a non-finite input or result, a singular pivot or a length mismatch — never a panic |
//! | accumulation | `f64` in the order written: elimination row by row, back substitution as one sum per row |
//! | damped least squares | `(J Jᵀ + λ² I) y = e` by [`solve_in_place`], then `Δ = Jᵀ y`; sums run over the Jacobian's columns, then its rows, in index order |

/// Solve `matrix x = rhs` in place for a row-major `n`×`n` matrix; `rhs` then holds `x` and
/// `matrix` the eliminated upper triangle.
///
/// # Errors
/// `None` when `matrix.len() != n * n` or `rhs.len() != n`, for a non-finite input or result,
/// or a pivot of at most `64 ε max|a_ij|`.
pub fn solve_in_place(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> Option<()> {
    if matrix.len() != n * n || rhs.len() != n {
        return None;
    }
    if matrix.iter().chain(rhs.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    let scale = matrix.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    for k in 0..n {
        let pivot = (k..n).max_by(|a, b| matrix[a * n + k].abs().total_cmp(&matrix[b * n + k].abs()))?;
        if matrix[pivot * n + k].abs() <= 64.0 * f64::EPSILON * scale {
            return None;
        }
        if pivot != k {
            for column in 0..n {
                matrix.swap(k * n + column, pivot * n + column);
            }
            rhs.swap(k, pivot);
        }
        for i in k + 1..n {
            let ratio = matrix[i * n + k] / matrix[k * n + k];
            for column in k..n {
                matrix[i * n + column] -= ratio * matrix[k * n + column];
            }
            rhs[i] -= ratio * rhs[k];
        }
    }
    for i in (0..n).rev() {
        rhs[i] = (rhs[i] - (i + 1..n).map(|column| matrix[i * n + column] * rhs[column]).sum::<f64>()) / matrix[i * n + i];
    }
    rhs.iter().all(|x| x.is_finite()).then_some(())
}

/// One damped-least-squares (Levenberg) step `Δ = Jᵀ (J Jᵀ + λ² I)⁻¹ e` for a row-major
/// `rows`×`columns` Jacobian and an error of `rows` entries, written into `step`
/// (`columns` entries). The damping `λ` keeps the step bounded where `J` loses rank; a caller
/// that weighs rows (orientation against position, say) scales those rows of `J` and `e`.
///
/// # Errors
/// `None` when a length does not match `rows`/`columns`, or when [`solve_in_place`] fails on
/// the damped normal matrix; `step` is then unspecified.
pub fn damped_least_squares(
    jacobian: &[f64],
    rows: usize,
    columns: usize,
    error: &[f64],
    damping: f64,
    step: &mut [f64],
) -> Option<()> {
    if jacobian.len() != rows * columns || error.len() != rows || step.len() != columns {
        return None;
    }
    let mut normal = vec![0.0; rows * rows];
    for r in 0..rows {
        for c in 0..rows {
            normal[r * rows + c] = (0..columns).map(|j| jacobian[r * columns + j] * jacobian[c * columns + j]).sum::<f64>();
        }
        normal[r * rows + r] += damping * damping;
    }
    let mut y = error.to_vec();
    solve_in_place(&mut normal, &mut y, rows)?;
    for (j, s) in step.iter_mut().enumerate() {
        *s = (0..rows).map(|r| jacobian[r * columns + j] * y[r]).sum();
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_system_that_needs_a_row_swap() {
        // The first column's largest entry is in the last row.
        let mut matrix = [1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 1.0];
        let mut rhs = [4.0, 6.0, 6.0];
        solve_in_place(&mut matrix, &mut rhs, 3).unwrap();
        let x = rhs;
        let original = [1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 1.0];
        for (row, b) in [4.0, 6.0, 6.0].iter().enumerate() {
            let ax: f64 = (0..3).map(|c| original[row * 3 + c] * x[c]).sum();
            assert!((ax - b).abs() < 1e-12, "row {row}: {ax} != {b}");
        }
    }

    #[test]
    fn refuses_singular_nonfinite_and_mismatched_input() {
        assert!(solve_in_place(&mut [1.0, 2.0, 2.0, 4.0], &mut [1.0, 2.0], 2).is_none());
        assert!(solve_in_place(&mut [1.0, 0.0, 0.0, f64::NAN], &mut [1.0, 1.0], 2).is_none());
        assert!(solve_in_place(&mut [1.0, 0.0, 0.0, 1.0], &mut [1.0], 2).is_none());
        assert!(damped_least_squares(&[1.0, 0.0], 1, 2, &[1.0], 0.1, &mut [0.0]).is_none());
    }

    #[test]
    fn undamped_square_step_inverts_the_jacobian() {
        let jacobian = [2.0, 1.0, 1.0, 3.0];
        let mut step = [0.0; 2];
        damped_least_squares(&jacobian, 2, 2, &[3.0, 5.0], 0.0, &mut step).unwrap();
        // 2a + b = 3, a + 3b = 5  ->  a = 0.8, b = 1.4
        assert!((step[0] - 0.8).abs() < 1e-12 && (step[1] - 1.4).abs() < 1e-12, "{step:?}");
    }

    #[test]
    fn step_is_bitwise_the_former_inline_geometry_formula() {
        // hymeko_geometry's IK (until 2026-10-21) kept the Jacobian as columns and solved inline:
        // a[r][c] = sum_j col_j[r] col_j[c], a[r][r] += l^2, y = solve(a, e), dq_j = sum_r col_j[r] y_r.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for (rows, columns) in [(3, 6), (6, 6), (6, 7), (3, 2)] {
            let cols: Vec<[f64; 6]> = (0..columns).map(|_| std::array::from_fn(|_| next())).collect();
            let error: Vec<f64> = (0..rows).map(|_| next()).collect();
            let damping = 0.05;
            let mut a = vec![0.0; rows * rows];
            for r in 0..rows {
                for c in 0..rows {
                    a[r * rows + c] = cols.iter().map(|col| col[r] * col[c]).sum::<f64>();
                }
                a[r * rows + r] += damping * damping;
            }
            let mut y = error.clone();
            solve_in_place(&mut a, &mut y, rows).unwrap();
            let former: Vec<f64> = cols.iter().map(|col| (0..rows).map(|r| col[r] * y[r]).sum()).collect();
            let jacobian: Vec<f64> = (0..rows).flat_map(|r| cols.iter().map(move |col| col[r])).collect();
            let mut step = vec![0.0; columns];
            damped_least_squares(&jacobian, rows, columns, &error, damping, &mut step).unwrap();
            assert!(step.iter().zip(&former).all(|(s, f)| s.to_bits() == f.to_bits()), "{rows}x{columns}");
        }
    }

    #[test]
    fn damped_step_matches_the_closed_form_and_stays_bounded_at_a_singularity() {
        // One row, two columns: J J^T + l^2 = 1 + 1 + l^2, so step = J^T e / (2 + l^2).
        let mut step = [0.0; 2];
        damped_least_squares(&[1.0, 1.0], 1, 2, &[2.0], 0.5, &mut step).unwrap();
        assert!((step[0] - 2.0 / 2.25).abs() < 1e-15 && (step[1] - 2.0 / 2.25).abs() < 1e-15);
        // A rank-deficient 2x2 Jacobian fails undamped and gives a finite step when damped.
        let singular = [1.0, 2.0, 2.0, 4.0];
        assert!(damped_least_squares(&singular, 2, 2, &[1.0, 1.0], 0.0, &mut step).is_none());
        damped_least_squares(&singular, 2, 2, &[1.0, 1.0], 0.1, &mut step).unwrap();
        assert!(step.iter().all(|s| s.is_finite() && s.abs() < 10.0), "{step:?}");
    }
}
