//! Cubic Hermite interpolation of sampled curves in f64: values and derivatives known at two
//! knots, read in between, channel by channel. Exact for cubics. This is what a stream of stamped
//! values with their rates (a robot's joint positions and velocities) needs between its samples.
//! No robot, scene or rendering vocabulary enters this numerical kernel.
//!
//! The activation kernels keep their own f32 basis ([`crate::kochanek_bartels`]); their outputs are
//! byte-identical by contract and this module does not touch them.

use std::{error::Error, fmt};

/// Invalid segment data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HermiteError {
    /// The value, derivative and output slices differ in length.
    Shape,
    /// Knots not finite and increasing, or a non-finite reading time.
    Data,
}

impl fmt::Display for HermiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape => "Hermite values, derivatives and output must have one length",
            Self::Data => "Hermite knots must be finite and increasing, and the time finite",
        })
    }
}

impl Error for HermiteError {}

/// Cubic Hermite basis `[h00, h10, h01, h11]` at segment coordinate `s` (0 at the first knot, 1 at
/// the second): `p(s) = h00 p0 + h10 (t1 - t0) v0 + h01 p1 + h11 (t1 - t0) v1`.
pub fn basis(s: f64) -> [f64; 4] {
    let (s2, s3) = (s * s, s * s * s);
    [
        2. * s3 - 3. * s2 + 1.,
        s3 - 2. * s2 + s,
        -2. * s3 + 3. * s2,
        s3 - s2,
    ]
}

/// The curve through `p0` (derivative `v0`, per unit of `t`) at `t0` and `p1` (`v1`) at `t1`, read
/// at `t` into `out`, one value per channel. Outside `[t0, t1]` the same cubic continues; a caller
/// that must not extrapolate clamps `t`. On an error `out` is left as it was.
pub fn interpolate_into(
    (t0, p0, v0): (f64, &[f64], &[f64]),
    (t1, p1, v1): (f64, &[f64], &[f64]),
    t: f64,
    out: &mut [f64],
) -> Result<(), HermiteError> {
    let n = out.len();
    if p0.len() != n || v0.len() != n || p1.len() != n || v1.len() != n {
        return Err(HermiteError::Shape);
    }
    if !(t0.is_finite() && t1.is_finite() && t.is_finite()) || t1 <= t0 {
        return Err(HermiteError::Data);
    }
    let h = t1 - t0;
    let [h00, h10, h01, h11] = basis((t - t0) / h);
    for (j, value) in out.iter_mut().enumerate() {
        *value = h00 * p0[j] + h10 * h * v0[j] + h01 * p1[j] + h11 * h * v1[j];
    }
    Ok(())
}
