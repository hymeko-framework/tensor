//! The fixed-size specialisation against the general one.
//!
//! The whole justification for `SeFixed` is that it is the *same function*, laid out differently.
//! So the tests do not check that it is approximately right — they check that it is **bit-identical**
//! to `Se`, because floating-point addition is not associative and a faster summation order would be
//! a different function whose answers differ in the last bit. A caller that had recorded a pose
//! would see it move, and no tolerance-based test would have said why.

use hymeko_clifford::{Dim2, Dim3, EuclideanDim, Se, Se2, Se3, SeFixed};

/// xorshift64, so every case below is reproducible without a dependency.
struct Rng(u64);
impl Rng {
    fn f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.f64() - 1.0
    }
}

fn same<D: EuclideanDim>(fixed: &SeFixed<D>, general: &Se, what: &str) {
    assert_eq!(general.n, D::N, "{what}: dimension");
    assert_eq!(
        fixed.rotor.as_ref(),
        general.rotor.components.as_slice(),
        "{what}: rotor differs"
    );
    assert_eq!(
        fixed.translation.as_ref(),
        general.translation.as_slice(),
        "{what}: translation differs"
    );
}

// ─── the constructors ───────────────────────────────────────────────────────────────────────────

#[test]
fn the_specialised_constructors_match_the_general_ones_exactly() {
    let mut rng = Rng(0xA11CE);
    for case in 0..200 {
        let axis = [rng.signed(), rng.signed(), rng.signed()];
        let angle = 6.0 * rng.signed();
        same(
            &Se3::axis_angle(axis, angle),
            &Se::se3_axis_angle(axis, angle),
            &format!("axis_angle case {case}"),
        );

        let rpy = [rng.signed() * 3.0, rng.signed() * 3.0, rng.signed() * 3.0];
        let xyz = [rng.signed(), rng.signed(), rng.signed()];
        same(
            &Se3::rpy_xyz(rpy, xyz),
            &Se::se3_rpy_xyz(rpy, xyz),
            &format!("rpy_xyz case {case}"),
        );

        let d = rng.signed() * 4.0;
        same(
            &Se3::axis_translation(axis, d),
            &Se::se3_axis_translation(axis, d),
            &format!("axis_translation case {case}"),
        );
    }

    // the degenerate axis both must refuse the same way
    same(
        &Se3::axis_angle([0.0, 0.0, 0.0], 1.0),
        &Se::se3_axis_angle([0.0, 0.0, 0.0], 1.0),
        "degenerate axis",
    );
    same(&Se3::identity(), &Se::identity(3), "identity");
}

// ─── the group law ──────────────────────────────────────────────────────────────────────────────

#[test]
fn composition_is_bit_identical_along_a_chain() {
    // A chain, not a single product: forward kinematics folds these, so an error in the last bit
    // that only appears after twenty compositions is exactly the error that matters.
    let mut rng = Rng(7);
    let mut fixed = Se3::identity();
    let mut general = Se::identity(3);
    for step in 0..64 {
        let axis = [rng.signed(), rng.signed(), rng.signed()];
        let angle = 2.0 * rng.signed();
        let xyz = [rng.signed(), rng.signed(), rng.signed()];
        fixed = fixed.compose(&Se3::rpy_xyz([angle, 0.3 * angle, -angle], xyz));
        general = general.compose(&Se::se3_rpy_xyz([angle, 0.3 * angle, -angle], xyz));
        same(&fixed, &general, &format!("chain step {step}"));

        let f2 = fixed.compose(&Se3::axis_angle(axis, angle));
        let g2 = general.compose(&Se::se3_axis_angle(axis, angle));
        same(&f2, &g2, &format!("axis compose {step}"));
    }
}

#[test]
fn rotate_apply_inverse_and_the_matrix_all_agree() {
    let mut rng = Rng(31337);
    for case in 0..100 {
        let m = Se3::rpy_xyz(
            [rng.signed() * 3.0, rng.signed() * 3.0, rng.signed() * 3.0],
            [rng.signed(), rng.signed(), rng.signed()],
        );
        let g = m.to_se();
        let v = [rng.signed(), rng.signed(), rng.signed()];

        assert_eq!(m.rotate(&v).as_slice(), g.rotate(&v).as_slice(), "rotate {case}");
        assert_eq!(m.apply(&v).as_slice(), g.apply(&v).as_slice(), "apply {case}");
        same(&m.inverse(), &g.inverse(), &format!("inverse {case}"));

        let mut mat = [0.0f64; 9];
        m.rotation_matrix_into(&mut mat);
        assert_eq!(mat.as_slice(), g.rotation_matrix().as_slice(), "matrix {case}");

        // and the quaternion the SE(3) case exposes
        assert_eq!(m.quaternion(), g.se3_quaternion(), "quaternion {case}");
    }
}

// ─── it really is generic in the dimension ──────────────────────────────────────────────────────

#[test]
fn the_plane_is_the_same_body_with_a_different_dimension() {
    // If `SeFixed` had been written for three dimensions and merely dressed as generic, this would
    // not compile — or would compute nonsense. SE(2) exercises the same group law with 4 blades and
    // 2 coordinates, against the general `Se` built by the same plane rotation.
    let mut rng = Rng(2);
    let mut fixed = Se2::identity();
    let mut general = Se::identity(2);
    for step in 0..40 {
        let angle = 3.0 * rng.signed();
        let t = [rng.signed(), rng.signed()];

        let mut f = Se2::identity();
        f.rotor = SeFixed::<Dim2>::from_se(&Se::plane_rotation(2, 1, 2, angle))
            .expect("a plane rotation is an SE(2) element")
            .rotor;
        f.translation = t;
        let mut g = Se::plane_rotation(2, 1, 2, angle);
        g.translation = t.to_vec();

        fixed = fixed.compose(&f);
        general = general.compose(&g);
        same(&fixed, &general, &format!("plane step {step}"));
    }
    assert_eq!(<Dim2 as EuclideanDim>::N, 2);
    assert_eq!(<Dim3 as EuclideanDim>::N, 3);
}

// ─── and the conversion is lossless both ways ───────────────────────────────────────────────────

#[test]
fn the_two_representations_convert_without_loss() {
    let mut rng = Rng(99);
    for case in 0..80 {
        let m = Se3::rpy_xyz(
            [rng.signed() * 2.0, rng.signed() * 2.0, rng.signed() * 2.0],
            [rng.signed(), rng.signed(), rng.signed()],
        );
        let round_trip = Se3::from_se(&m.to_se()).expect("three dimensions");
        same(&round_trip, &m.to_se(), &format!("round trip {case}"));
        assert_eq!(round_trip.rotor, m.rotor);
        assert_eq!(round_trip.translation, m.translation);
    }
    // a dimension mismatch is refused rather than reinterpreted
    assert!(Se3::from_se(&Se::identity(2)).is_none());
    assert!(SeFixed::<Dim2>::from_se(&Se::identity(3)).is_none());
}
