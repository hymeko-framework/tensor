//! One-sided scientific acceptance.
//!
//! # Why this type exists
//!
//! A recorded result is evidence that a score *was reached*, not that it is the score. Asserting
//! `observed == recorded` on a scientific quantity makes an **improvement fail**, which turns the
//! test suite into a ratchet against the research. CLAUDE.md §3 already forbids resting a claim on
//! a single run; this module is the assertion-side consequence.
//!
//! A [`ScientificGate`] therefore carries a *direction* and an *error band*, and the band's
//! [`ErrorBasis`] is part of the value — not a comment. That is the whole point: a round number
//! invented after watching a test fail cannot be constructed without naming itself
//! [`ErrorBasis::SeedVariance`] or one of the other three and supplying the evidence string.
//!
//! # What this is NOT for
//!
//! Bit contracts, invariants, optimizer lockstep, artifact integrity and frozen-oracle parity are
//! **exact**. Do not route them through here; `assert_eq!` states the stronger claim and should
//! keep stating it.

use std::fmt;

/// Which way "better" runs for the quantity under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Accuracy, AUC, F1, success rate: a larger value passes.
    HigherIsBetter,
    /// Loss, error, wall time, peak RSS: a smaller value passes.
    LowerIsBetter,
}

/// Where the error band came from.
///
/// Every variant carries its evidence as text, so the justification travels with the number into
/// the failure message instead of living in a comment three files away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorBasis {
    /// Spread measured across seeds. The text records the seeds and the observed values.
    SeedVariance(String),
    /// A recorded process/order band — e.g. a float-summation order effect measured across thread
    /// counts or block layouts.
    ProcessBand(String),
    /// A finite-difference or floating-point resolution floor below which a difference is not a
    /// difference.
    NumericalFloor(String),
    /// A tolerance that already existed and was justified elsewhere. The text names where.
    PreExistingTolerance(String),
}

impl ErrorBasis {
    /// The evidence text.
    #[must_use]
    pub fn evidence(&self) -> &str {
        match self {
            Self::SeedVariance(s)
            | Self::ProcessBand(s)
            | Self::NumericalFloor(s)
            | Self::PreExistingTolerance(s) => s,
        }
    }

    /// A short tag for failure messages.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SeedVariance(_) => "seed-variance",
            Self::ProcessBand(_) => "process-band",
            Self::NumericalFloor(_) => "numerical-floor",
            Self::PreExistingTolerance(_) => "pre-existing-tolerance",
        }
    }
}

/// A one-sided acceptance bound on a scientific quantity.
///
/// # Invariants
/// `band >= 0.0` and `certified` is finite; both are checked at construction.
#[derive(Debug, Clone)]
pub struct ScientificGate {
    /// What is being judged, for the failure message.
    pub quantity: &'static str,
    /// The certified floor (higher-is-better) or ceiling (lower-is-better).
    pub certified: f64,
    /// Which way better runs.
    pub direction: Direction,
    /// The justified error band, always non-negative.
    pub band: f64,
    /// Where `band` came from.
    pub basis: ErrorBasis,
    /// Path to the raw per-seed values recorded beside this bound.
    pub raw: &'static str,
}

impl ScientificGate {
    /// Construct a gate.
    ///
    /// # Preconditions
    /// `certified` is finite and `band >= 0.0`.
    ///
    /// # Panics
    /// If either precondition fails — a gate that cannot be stated is not a gate.
    #[must_use]
    pub fn new(
        quantity: &'static str,
        certified: f64,
        direction: Direction,
        band: f64,
        basis: ErrorBasis,
        raw: &'static str,
    ) -> Self {
        assert!(
            certified.is_finite(),
            "{quantity}: certified must be finite"
        );
        assert!(
            band >= 0.0 && band.is_finite(),
            "{quantity}: band must be finite and non-negative, got {band}"
        );
        Self {
            quantity,
            certified,
            direction,
            band,
            basis,
            raw,
        }
    }

    /// The value `observed` must not cross.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        match self.direction {
            Direction::HigherIsBetter => self.certified - self.band,
            Direction::LowerIsBetter => self.certified + self.band,
        }
    }

    /// Whether `observed` is acceptable.
    ///
    /// A **better-than-certified** value always passes: that is the one-sidedness.
    #[must_use]
    pub fn accepts(&self, observed: f64) -> bool {
        if !observed.is_finite() {
            return false;
        }
        match self.direction {
            Direction::HigherIsBetter => observed >= self.threshold(),
            Direction::LowerIsBetter => observed <= self.threshold(),
        }
    }

    /// Assert [`Self::accepts`], with the whole justification in the panic message.
    ///
    /// # Panics
    /// If `observed` crosses the threshold, or is not finite.
    pub fn assert_accepts(&self, observed: f64) {
        assert!(
            self.accepts(observed),
            "{self}\n  observed  = {observed}\n  threshold = {}\n  VERDICT: rejected",
            self.threshold()
        );
    }
}

impl fmt::Display for ScientificGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (arrow, word) = match self.direction {
            Direction::HigherIsBetter => (">=", "floor"),
            Direction::LowerIsBetter => ("<=", "ceiling"),
        };
        write!(
            f,
            "{q}: one-sided gate, {word} {c} {arrow} band {b} [{k}: {e}] raw={raw}",
            q = self.quantity,
            c = self.certified,
            b = self.band,
            k = self.basis.kind(),
            e = self.basis.evidence(),
            raw = self.raw
        )
    }
}

/// Population standard deviation of `xs`, the spread [`ErrorBasis::SeedVariance`] should quote.
///
/// # Preconditions
/// `xs` is non-empty.
///
/// # Panics
/// If `xs` is empty.
#[must_use]
pub fn population_sd(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "population_sd of an empty sample");
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    (xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor_gate(band: f64) -> ScientificGate {
        ScientificGate::new(
            "auc",
            0.90,
            Direction::HigherIsBetter,
            band,
            ErrorBasis::SeedVariance("seeds 0..5 -> 0.898 0.901 0.900 0.902 0.899".into()),
            "run_outputs/example.jsonl",
        )
    }

    #[test]
    fn a_better_score_passes() {
        assert!(floor_gate(0.01).accepts(0.99));
        assert!(floor_gate(0.0).accepts(0.90));
    }

    #[test]
    fn a_worse_score_inside_the_band_passes() {
        assert!(floor_gate(0.01).accepts(0.895));
    }

    #[test]
    fn a_worse_score_outside_the_band_fails() {
        assert!(!floor_gate(0.01).accepts(0.88));
    }

    #[test]
    fn ceiling_direction_is_mirrored() {
        let g = ScientificGate::new(
            "loss",
            0.25,
            Direction::LowerIsBetter,
            0.01,
            ErrorBasis::NumericalFloor("f32 epsilon at this magnitude".into()),
            "raw",
        );
        assert!(g.accepts(0.0), "a much better loss must pass");
        assert!(g.accepts(0.26));
        assert!(!g.accepts(0.27));
    }

    #[test]
    fn nan_is_never_accepted() {
        assert!(!floor_gate(1.0).accepts(f64::NAN));
        assert!(!floor_gate(1.0).accepts(f64::NEG_INFINITY));
    }

    #[test]
    #[should_panic(expected = "band must be finite and non-negative")]
    fn a_negative_band_cannot_be_constructed() {
        let _ = floor_gate(-0.01);
    }

    #[test]
    fn population_sd_matches_the_closed_form() {
        // xs = 1,2,3,4 -> mean 2.5, variance (2.25+0.25+0.25+2.25)/4 = 1.25
        let sd = population_sd(&[1.0, 2.0, 3.0, 4.0]);
        assert!((sd - 1.25f64.sqrt()).abs() < 1e-15, "got {sd}");
    }

    #[test]
    #[should_panic(expected = "empty sample")]
    fn population_sd_rejects_an_empty_sample() {
        let _ = population_sd(&[]);
    }

    #[test]
    fn the_basis_travels_into_the_message() {
        let msg = format!("{}", floor_gate(0.01));
        assert!(msg.contains("seed-variance"), "{msg}");
        assert!(msg.contains("0.898"), "{msg}");
        assert!(msg.contains("run_outputs/example.jsonl"), "{msg}");
    }
}
