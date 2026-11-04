//! Numerical contracts for the shared GMRES workspace, without timing.
#[path = "support/gmres_cases.rs"]
mod cases;
use hymeko_numerics::{gmres, norm, GmresConfig, GmresTolerance, GmresWorkspace, Operator};

#[test]
fn relative_workspace_and_facade_match_144_frozen_pre_change_cases_bitwise() {
    let lines: Vec<_> = include_str!("fixtures/gmres-before.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    let catalog = cases::Case::catalog();
    assert_eq!(lines.len(), catalog.len());
    let mut workspace = GmresWorkspace::new();
    assert_eq!(workspace.retained_bytes(), 0);
    for (id, (case, line)) in catalog.iter().zip(lines).enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields[0].parse::<usize>().unwrap(), id);
        let expected: Vec<_> = fields[5]
            .split(',')
            .filter(|v| !v.is_empty())
            .map(|v| u64::from_str_radix(v, 16).unwrap())
            .collect();
        for reusable in [true, false] {
            let mut x = case.initial();
            case.calls.set(0);
            let rhs = case.rhs();
            let report = if reusable {
                workspace.solve(
                    case,
                    0.75,
                    &rhs,
                    &mut x,
                    GmresConfig {
                        tolerance: GmresTolerance::Relative(1e-12),
                        restart: case.restart,
                        max_iterations: case.budget,
                    },
                )
            } else {
                gmres(case, 0.75, &rhs, &mut x, 1e-12, case.restart, case.budget)
            };
            assert_eq!(
                report.iterations,
                fields[1].parse::<usize>().unwrap(),
                "case {id}"
            );
            assert_eq!(
                report.residual.to_bits(),
                u64::from_str_radix(fields[2], 16).unwrap(),
                "case {id}"
            );
            assert_eq!(
                report.converged,
                fields[3].parse::<bool>().unwrap(),
                "case {id}"
            );
            assert_eq!(
                case.calls.get(),
                fields[4].parse::<usize>().unwrap(),
                "case {id}"
            );
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected,
                "case {id}"
            );
        }
    }
}

struct Rotation;
impl Operator for Rotation {
    fn dim(&self) -> usize {
        2
    }
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        out[0] = x[1];
        out[1] = -x[0];
    }
}

#[test]
fn absolute_tolerance_preserves_the_true_relative_report_and_closed_form_solution() {
    let mut workspace = GmresWorkspace::default();
    let config = GmresConfig {
        tolerance: GmresTolerance::Absolute(1e-12),
        restart: 2,
        max_iterations: 2,
    };
    for scale in [0.125, 1.0, 128.0] {
        let rhs = [scale, 0.0];
        let mut x = [0.0; 2];
        let report = workspace.solve(&Rotation, 1.0, &rhs, &mut x, config);
        assert!(report.converged);
        assert_eq!(report.iterations, 2);
        assert!((x[0] - scale / 2.0).abs() <= 1e-12);
        assert!((x[1] + scale / 2.0).abs() <= 1e-12);
        let actual = norm(&[rhs[0] - x[0] + x[1], rhs[1] - x[1] - x[0]]);
        assert!(actual <= 1e-12);
        assert_eq!(report.residual, actual / norm(&rhs));
        let exact = workspace.solve(&Rotation, 1.0, &rhs, &mut x, config);
        assert_eq!(exact.iterations, 0);
    }
    let mut zero = [0.0; 2];
    assert!(
        workspace
            .solve(&Rotation, 1.0, &[0.0; 2], &mut zero, config)
            .converged
    );
    let limited = workspace.solve(
        &Rotation,
        1.0,
        &[1.0, 0.0],
        &mut zero,
        GmresConfig {
            max_iterations: 0,
            ..config
        },
    );
    assert!(!limited.converged);
    assert_eq!(limited.iterations, 0);
    assert_eq!(zero, [0.0; 2]);
}

#[test]
fn invalid_dimensions_fail_before_mutating_the_solution() {
    let mut workspace = GmresWorkspace::new();
    let mut x = [7.0, 8.0];
    let config = GmresConfig {
        tolerance: GmresTolerance::Relative(1e-12),
        restart: 2,
        max_iterations: 4,
    };
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        workspace.solve(&Rotation, 1.0, &[1.0], &mut x, config);
    }));
    assert!(error.is_err());
    assert_eq!(x, [7.0, 8.0]);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        workspace.solve(&Rotation, 1.0, &[1.0, 0.0], &mut x[..1], config);
    }));
    assert!(error.is_err());
    assert_eq!(x, [7.0, 8.0]);
}

#[test]
fn basis_dimension_overflow_is_rejected_before_reserving_the_basis() {
    let mut workspace = GmresWorkspace::new();
    let mut x = [0.0; 2];
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        workspace.solve(
            &Rotation,
            1.0,
            &[1.0, 0.0],
            &mut x,
            GmresConfig {
                tolerance: GmresTolerance::Relative(1e-12),
                restart: usize::MAX,
                max_iterations: usize::MAX,
            },
        );
    }));
    assert!(error.is_err());
    assert_eq!(x, [0.0; 2]);
    assert!(workspace.retained_bytes() < 1024);
}
