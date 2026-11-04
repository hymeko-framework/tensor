use hymeko_spline::hermite::{basis, interpolate_into, HermiteError};

/// Two channels of cubics with their exact derivatives.
fn cubic(t: f64) -> ([f64; 2], [f64; 2]) {
    (
        [
            0.3 - 1.2 * t + 0.7 * t * t + 2.5 * t * t * t,
            -2. + 0.4 * t * t * t,
        ],
        [-1.2 + 1.4 * t + 7.5 * t * t, 1.2 * t * t],
    )
}

#[test]
fn a_cubic_is_reproduced_inside_and_beyond_its_knots() {
    let (t0, t1) = (0.25, 0.37);
    let ((p0, v0), (p1, v1)) = (cubic(t0), cubic(t1));
    for t in [0.25, 0.27, 0.31, 0.369, 0.37, 0.45] {
        let mut out = [0.; 2];
        interpolate_into((t0, &p0, &v0), (t1, &p1, &v1), t, &mut out).unwrap();
        for (got, want) in out.iter().zip(cubic(t).0) {
            assert!((got - want).abs() < 1e-12, "t {t}: {got} != {want}");
        }
    }
}

#[test]
fn the_basis_meets_both_knots_with_their_values_and_slopes() {
    assert_eq!(basis(0.), [1., 0., 0., 0.]);
    assert_eq!(basis(1.), [0., 0., 1., 0.]);
    for s in [0.1, 0.5, 0.9] {
        let [h00, _, h01, _] = basis(s);
        assert!(
            (h00 + h01 - 1.).abs() < 1e-15,
            "values interpolate: weights sum to one"
        );
    }
    // d/ds of h10 is 1 at s = 0 and of h11 is 1 at s = 1; finite differences.
    let e = 1e-7;
    assert!(((basis(e)[1] - basis(0.)[1]) / e - 1.).abs() < 1e-6);
    assert!(((basis(1.)[3] - basis(1. - e)[3]) / e - 1.).abs() < 1e-6);
}

#[test]
fn malformed_segments_are_refused_and_leave_the_output() {
    let (p, v) = ([0.; 2], [0.; 2]);
    let mut out = [7.; 2];
    assert_eq!(
        interpolate_into((0., &p, &v), (1., &p, &v[..1]), 0.5, &mut out),
        Err(HermiteError::Shape)
    );
    assert_eq!(
        interpolate_into((1., &p, &v), (1., &p, &v), 1., &mut out),
        Err(HermiteError::Data)
    );
    assert_eq!(
        interpolate_into((0., &p, &v), (1., &p, &v), f64::NAN, &mut out),
        Err(HermiteError::Data)
    );
    assert_eq!(out, [7.; 2]);
}
