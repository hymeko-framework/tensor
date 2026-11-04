// APPROVED-CORE-EDIT: clifford-additive (2026-10-20, user chat: "Yes, additive edits").
//
// The 3-vector cross product, once. Before this the crate held it twice -- `f32` beside the rotors,
// `f64` beside the spatial algebra -- and Nagare a third time; one body generic over the scalar
// keeps every caller's arithmetic, since the expression is the same at either precision.

use core::ops::{Mul, Sub};

/// `a × b` for 3-vectors over any scalar with `*` and `-`, each component in the textbook order
/// `a_y b_z − a_z b_y`, `a_z b_x − a_x b_z`, `a_x b_y − a_y b_x`.
#[inline]
#[must_use]
pub fn cross<T: Copy + Mul<Output = T> + Sub<Output = T>>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
