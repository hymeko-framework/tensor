//! Bounded rational B-spline curves in homogeneous coordinates, using f64 arithmetic.
//! No geometry, scene, control or rendering vocabulary enters this numerical kernel.

use std::{error::Error, fmt};

/// Invalid curve data or an evaluation outside the numerical contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NurbsError {
    /// Unsupported dimension/degree, mismatched arrays or too many controls.
    Shape,
    /// Nonfinite data, nonpositive weights or malformed knot multiplicities/domain.
    Data,
    /// Evaluation parameter is nonfinite or outside the closed curve domain.
    Parameter,
    /// Homogeneous arithmetic overflowed or lost a positive denominator.
    Conditioning,
}
impl fmt::Display for NurbsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NURBS {:?}", self)
    }
}
impl Error for NurbsError {}

/// Position and parameter derivatives (not arc-length derivatives).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveSample<const D: usize> {
    /// Curve position.
    pub position: [f64; D],
    /// First derivative with respect to the supplied knot parameter.
    pub first: [f64; D],
    /// Second derivative with respect to the supplied knot parameter.
    pub second: [f64; D],
}

#[derive(Clone, Copy, Debug)]
struct Homogeneous<const D: usize> {
    point: [f64; D],
    weight: f64,
}
impl<const D: usize> Homogeneous<D> {
    const ZERO: Self = Self {
        point: [0.0; D],
        weight: 0.0,
    };

    fn combine(self, other: Self, a: f64, b: f64) -> Self {
        Self {
            point: std::array::from_fn(|i| a * self.point[i] + b * other.point[i]),
            weight: a * self.weight + b * other.weight,
        }
    }
    fn finite(&self) -> bool {
        self.weight.is_finite() && self.point.iter().all(|v| v.is_finite())
    }
}

#[derive(Clone, Debug)]
struct Polynomial<const D: usize> {
    degree: usize,
    controls: Vec<Homogeneous<D>>,
    knots: Vec<f64>,
}
impl<const D: usize> Polynomial<D> {
    fn derivative(&self) -> Option<Self> {
        if self.degree == 0 {
            return None;
        }
        let controls = self
            .controls
            .windows(2)
            .enumerate()
            .map(|(i, pair)| {
                let span = self.knots[i + self.degree + 1] - self.knots[i + 1];
                // Repeated knots define a zero contribution on a zero-width span.
                let factor = if span > 0.0 {
                    self.degree as f64 / span
                } else {
                    0.0
                };
                pair[1].combine(pair[0], factor, -factor)
            })
            .collect();
        Some(Self {
            degree: self.degree - 1,
            controls,
            knots: self.knots[1..self.knots.len() - 1].to_vec(),
        })
    }
    fn at(&self, u: f64) -> Homogeneous<D> {
        let last = self.controls.len() - 1;
        let span = self
            .knots
            .partition_point(|&k| k <= u)
            .saturating_sub(1)
            .clamp(self.degree, last);
        let mut work = [Homogeneous::ZERO; 6];
        work[..=self.degree].copy_from_slice(&self.controls[span - self.degree..=span]);
        for r in 1..=self.degree {
            for j in (r..=self.degree).rev() {
                let i = span - self.degree + j;
                let width = self.knots[i + self.degree + 1 - r] - self.knots[i];
                let alpha = if width > 0.0 {
                    (u - self.knots[i]) / width
                } else {
                    0.0
                };
                work[j] = work[j - 1].combine(work[j], 1.0 - alpha, alpha);
            }
        }
        work[self.degree]
    }
}

/// Immutable rational B-spline with precomputed homogeneous derivative curves.
/// Evaluation uses bounded stack scratch and allocates no memory.
#[derive(Clone, Debug)]
pub struct NurbsCurve<const D: usize> {
    curve: Polynomial<D>,
    first: Option<Polynomial<D>>,
    second: Option<Polynomial<D>>,
}
impl<const D: usize> NurbsCurve<D> {
    /// Validate and own a curve. Existing spline activation APIs are independent.
    ///
    /// # Preconditions
    /// D is 1..=16, degree 1..=5; there are degree+1..=256 controls and one
    /// positive weight per control. Knots have length controls+degree+1, are
    /// nondecreasing, and define a nonempty domain. All inputs are finite.
    /// Internal knot multiplicity is at most degree; endpoint multiplicity may
    /// be degree+1. Thus positions are continuous; derivatives can be one-sided.
    /// # Postconditions
    /// Successful construction owns validated data with maximum weight normalized
    /// to one. Failed construction changes no caller data.
    /// # Errors
    /// Reports invalid shape/data or unrepresentable normalized homogeneous values.
    /// # Invariants
    /// Input order and reduction order are deterministic on native and WASM.
    pub fn new(
        degree: usize,
        points: &[[f64; D]],
        weights: &[f64],
        knots: &[f64],
    ) -> Result<Self, NurbsError> {
        Self::validate(degree, points, weights, knots)?;
        let scale = weights.iter().copied().fold(0.0, f64::max);
        let controls: Vec<_> = points
            .iter()
            .zip(weights)
            .map(|(point, weight)| {
                let w = weight / scale;
                Homogeneous {
                    point: point.map(|v| v * w),
                    weight: w,
                }
            })
            .collect();
        if controls.iter().any(|p| !p.finite() || p.weight <= 0.0) {
            return Err(NurbsError::Conditioning);
        }
        let curve = Polynomial {
            degree,
            controls,
            knots: knots.to_vec(),
        };
        let first = curve.derivative();
        let second = first.as_ref().and_then(Polynomial::derivative);
        if first
            .iter()
            .chain(second.iter())
            .any(|p| p.controls.iter().any(|v| !v.finite()))
        {
            return Err(NurbsError::Conditioning);
        }
        Ok(Self {
            curve,
            first,
            second,
        })
    }

    fn validate(
        degree: usize,
        points: &[[f64; D]],
        weights: &[f64],
        knots: &[f64],
    ) -> Result<(), NurbsError> {
        let n = points.len();
        if !(1..=16).contains(&D)
            || !(1..=5).contains(&degree)
            || n <= degree
            || n > 256
            || weights.len() != n
            || knots.len() != n + degree + 1
        {
            return Err(NurbsError::Shape);
        }
        if points.iter().flatten().chain(knots).any(|v| !v.is_finite())
            || weights.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || knots.windows(2).any(|w| w[0] > w[1])
            || knots[degree] >= knots[n]
        {
            return Err(NurbsError::Data);
        }
        Self::validate_multiplicity(degree, knots, [knots[degree], knots[n]])
    }

    fn validate_multiplicity(
        degree: usize,
        knots: &[f64],
        domain: [f64; 2],
    ) -> Result<(), NurbsError> {
        let mut run = 1;
        for pair in knots.windows(2) {
            run = if pair[0] == pair[1] { run + 1 } else { 1 };
            let internal = pair[1] > domain[0] && pair[1] < domain[1];
            if run > degree + usize::from(!internal) {
                return Err(NurbsError::Data);
            }
        }
        Ok(())
    }

    /// Closed valid parameter interval; inspection never changes the curve.
    pub fn domain(&self) -> [f64; 2] {
        [
            self.curve.knots[self.curve.degree],
            self.curve.knots[self.curve.controls.len()],
        ]
    }

    /// Evaluate position and first/second derivatives without allocation.
    ///
    /// # Preconditions
    /// Parameter lies in `domain()`. At internal knots use right-sided derivatives;
    /// at the final endpoint use left-sided derivatives. Knot parameter is not metres.
    /// # Postconditions
    /// Returns finite values or an explicit error; the curve is never mutated.
    /// # Errors
    /// `Parameter` for an invalid parameter; `Conditioning` for a lost positive
    /// denominator or overflow in homogeneous/rational arithmetic.
    /// # Invariants
    /// Translation and uniform weight scaling preserve rational curve geometry.
    pub fn sample(&self, u: f64) -> Result<CurveSample<D>, NurbsError> {
        let [lo, hi] = self.domain();
        if !u.is_finite() || u < lo || u > hi {
            return Err(NurbsError::Parameter);
        }
        let h = self.curve.at(u);
        let d = self.first.as_ref().map_or(Homogeneous::ZERO, |p| p.at(u));
        let dd = self.second.as_ref().map_or(Homogeneous::ZERO, |p| p.at(u));
        if h.weight <= 0.0 || !h.finite() || !d.finite() || !dd.finite() {
            return Err(NurbsError::Conditioning);
        }
        let position = h.point.map(|v| v / h.weight);
        let first = std::array::from_fn(|i| (d.point[i] - position[i] * d.weight) / h.weight);
        let second = std::array::from_fn(|i| {
            (dd.point[i] - 2.0 * first[i] * d.weight - position[i] * dd.weight) / h.weight
        });
        if position
            .iter()
            .chain(&first)
            .chain(&second)
            .any(|v| !v.is_finite())
        {
            return Err(NurbsError::Conditioning);
        }
        Ok(CurveSample {
            position,
            first,
            second,
        })
    }
}
