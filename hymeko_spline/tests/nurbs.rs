use hymeko_spline::nurbs::{NurbsCurve, NurbsError};

fn circle(scale: f64, shift: [f64; 2]) -> NurbsCurve<2> {
    // Exact rational quadratic quarter circle (Michigan Tech NURBS definition).
    let points = [[1., 0.], [1., 1.], [0., 1.]].map(|p| [p[0] + shift[0], p[1] + shift[1]]);
    NurbsCurve::new(
        2,
        &points,
        &[scale, scale * 0.5_f64.sqrt(), scale],
        &[0., 0., 0., 1., 1., 1.],
    )
    .unwrap()
}

#[test]
fn rational_circle_and_derivatives_match_independent_geometry() {
    let curve = circle(1., [0.; 2]);
    assert_eq!(curve.domain(), [0., 1.]);
    for i in 0..=200 {
        let u = i as f64 / 200.;
        let q = curve.sample(u).unwrap();
        let [x, y] = q.position;
        assert!((x * x + y * y - 1.).abs() < 8e-16);
        assert!((x * q.first[0] + y * q.first[1]).abs() < 2e-15);
        let speed2 = q.first.iter().map(|v| v * v).sum::<f64>();
        assert!((q.position[0] * q.second[0] + q.position[1] * q.second[1] + speed2).abs() < 8e-15);
        if (0.01..0.99).contains(&u) {
            let h = 1e-5;
            let a = curve.sample(u - h).unwrap();
            let b = curve.sample(u + h).unwrap();
            for d in 0..2 {
                assert!(((b.position[d] - a.position[d]) / (2. * h) - q.first[d]).abs() < 3e-10);
                assert!(((b.first[d] - a.first[d]) / (2. * h) - q.second[d]).abs() < 8e-10);
            }
        }
    }
    let q = curve.sample(0.5).unwrap();
    assert!((q.position[0] - 0.5_f64.sqrt()).abs() < 2e-16);
    assert_eq!(curve.sample(0.).unwrap().position, [1., 0.]);
    assert_eq!(curve.sample(1.).unwrap().position, [0., 1.]);
}

#[test]
fn weight_scaling_translation_and_linear_limits() {
    let a = circle(1., [0.; 2]);
    for factor in [1e-150, 0.25, 16., 1e150] {
        let b = circle(factor, [1000., -700.]);
        for i in 0..=100 {
            let u = i as f64 / 100.;
            let x = a.sample(u).unwrap();
            let y = b.sample(u).unwrap();
            for d in 0..2 {
                let shift = if d == 0 { 1000. } else { -700. };
                assert!((y.position[d] - shift - x.position[d]).abs() < 4e-13);
                assert!((y.first[d] - x.first[d]).abs() < 2e-12);
            }
        }
    }
    let line = NurbsCurve::new(
        1,
        &[[2., 3., 4.], [8., 6., -2.]],
        &[1., 1.],
        &[0., 0., 2., 2.],
    )
    .unwrap();
    for i in 0..=100 {
        let u = i as f64 / 50.;
        let q = line.sample(u).unwrap();
        assert_eq!(q.first, [3., 1.5, -3.]);
        assert_eq!(q.second, [0.; 3]);
        assert!((q.position[0] - (2. + 3. * u)).abs() < 2e-15);
    }
}

#[test]
fn repeated_knots_choose_right_derivative_and_last_endpoint() {
    let curve = NurbsCurve::new(1, &[[0.], [1.], [1.]], &[1.; 3], &[0., 0., 0.5, 1., 1.]).unwrap();
    assert_eq!(curve.sample(0.25).unwrap().first, [2.]);
    assert_eq!(curve.sample(0.5).unwrap().first, [0.]);
    assert_eq!(curve.sample(1.).unwrap().position, [1.]);
    // Constant weights reduce the cubic to Bernstein's polynomial on a clamped span.
    let cubic = NurbsCurve::new(
        3,
        &[[0.], [0.], [0.], [1.]],
        &[1.; 4],
        &[0., 0., 0., 0., 1., 1., 1., 1.],
    )
    .unwrap();
    for i in 0..=100 {
        let t = i as f64 / 100.;
        let q = cubic.sample(t).unwrap();
        assert!((q.position[0] - t * t * t).abs() < 3e-16);
        assert!((q.first[0] - 3. * t * t).abs() < 2e-15);
        assert!((q.second[0] - 6. * t).abs() < 3e-15);
    }
}

#[test]
fn invalid_data_and_parameters_are_explicit_errors() {
    let points = [[0.], [1.]];
    for (degree, weights, knots) in [
        (0, vec![1., 1.], vec![0., 0., 1., 1.]),
        (1, vec![1.], vec![0., 0., 1., 1.]),
        (1, vec![0., 1.], vec![0., 0., 1., 1.]),
        (1, vec![f64::NAN, 1.], vec![0., 0., 1., 1.]),
        (1, vec![1., 1.], vec![0., 1., 0., 1.]),
        (1, vec![1., 1.], vec![0., 0., 0., 0.]),
        (1, vec![1., 1.], vec![0., 0., f64::INFINITY, 1.]),
    ] {
        assert!(NurbsCurve::new(degree, &points, &weights, &knots).is_err());
    }
    assert!(NurbsCurve::new(1, &[[f64::NAN], [1.]], &[1., 1.], &[0., 0., 1., 1.]).is_err());
    assert!(NurbsCurve::new(
        1,
        &[[0.], [1.], [2.], [3.]],
        &[1.; 4],
        &[0., 0., 0.5, 0.5, 1., 1.]
    )
    .is_err());
    assert!(NurbsCurve::new(
        1,
        &points,
        &[f64::MIN_POSITIVE, f64::MAX],
        &[0., 0., 1., 1.]
    )
    .is_err());
    let curve = circle(1., [0.; 2]);
    for u in [-1., 1.01, f64::NAN, f64::INFINITY] {
        assert_eq!(curve.sample(u), Err(NurbsError::Parameter));
    }
    assert_eq!(curve.sample(0.).unwrap().position, [1., 0.]);
    let steep = NurbsCurve::new(1, &[[0.], [1.]], &[1e-200, 1.], &[0., 0., 1., 1.]).unwrap();
    assert_eq!(steep.sample(0.), Err(NurbsError::Conditioning));
}

#[test]
fn every_supported_degree_reproduces_its_bernstein_monomial() {
    for degree in 1..=5 {
        let mut points = vec![[0.]; degree + 1];
        points[degree] = [1.];
        let weights = vec![1.; degree + 1];
        let mut knots = vec![0.; degree + 1];
        knots.extend(vec![1.; degree + 1]);
        let curve = NurbsCurve::new(degree, &points, &weights, &knots).unwrap();
        for i in 0..=100 {
            let t = i as f64 / 100.;
            let sample = curve.sample(t).unwrap();
            let p = degree as i32;
            assert!((sample.position[0] - t.powi(p)).abs() < 2e-15);
            assert!((sample.first[0] - degree as f64 * t.powi(p - 1)).abs() < 8e-15);
            let second = if degree == 1 {
                0.
            } else {
                (degree * (degree - 1)) as f64 * t.powi(p - 2)
            };
            assert!((sample.second[0] - second).abs() < 3e-14);
        }
    }
}
