//! The closed form of the uniform, chord-tangent Chebyshev--Catmull--Rom spline of at most three terms (Nagare plan
//! 2026-10-30-t2-algebraic-lowering; derivation, symbolic proof and measurements in
//! `nagare/reports/2026-10-30-hsikan-performance-challenge.md`, §3).
//!
//! # The identity
//! On `g` uniform knots `ξ_j = −1 + jh`, `h = 2/(g − 1)`, with chord tangents and replicated ends, the spline of the
//! control points `q_j = Σ_{i<k} θ_i T_i(ξ_j)` (`k ≤ 3`) is, for every input `x`,
//!
//! `φ(x) = θ₀ + θ₁ P₁(x̂) + θ₂ P₂(x̂)`, `P₁ = x̂ − (h/2)(A − B)`, `P₂ = 2x̂² − 1 + (2h + h²)(A + B)`,
//!
//! with `x̂ = clamp(x, −1, 1)`, `u = (x̂ + 1)/h`, `α = max(0, 1 − u)`, `β = max(0, u − (g − 2))`, `A = α²(1 − α)`,
//! `B = β²(1 − β)`. The uniform spline reproduces every polynomial of degree two on its interior leaves (central
//! differences are exact for quadratics); each replicated end control point adds one cubic bump on its end leaf,
//! `w₀(t)Δ_L·θ` and `w₃(t)Δ_R·θ` with `Δ_{L,R} = (0, ±h, −(4h + 2h²))`. The bumps vanish with zero slope at the knots
//! `ξ_1` and `ξ_{g−2}`, so the closed form is `C¹` wherever the spline is.
//!
//! Derivatives on the closed interval `[−1, 1]` (one-sided at `±1`; zero outside, the tables' convention, which a
//! caller applies): `φ' = θ₁P₁' + θ₂P₂'`, `P₁' = 1 + ½(2α − 3α²) + ½(2β − 3β²)`,
//! `P₂' = 4x̂ + (2 + h)((2β − 3β²) − (2α − 3α²))`; the parameter gradient is `∂φ/∂θ_i = P_i(x̂)`, `P₀ = 1`.
//!
//! # What it is for
//! Several banks on the same knots read at the same inputs share one basis per input: each bank is a three-term dot
//! product with its own terms. No leaf location, no table read, no per-(channel, leaf) moment scatter and no fold --
//! the terms' gradient is `Σ ȳ P_i` per channel, accumulated in [`TermPlanes`]' layout and mapped back once.
//!
//! # Qualification
//! [`ChebyshevClosedForm::of`] answers only for uniform knots, chord tangents, `1 ≤ k ≤ 3` and `g ≥ 3`. Refined knots,
//! bounded tangents and four or more terms have no closed form here: their callers keep the tables. Nothing is
//! approximated: on the qualified family the closed form and the tables are the same real function.

use crate::knots::{Knots, Tangents};

/// Terms a closed form carries per bank: `k ≤ 3`, absent terms zero.
pub const TERMS: usize = 3;

/// The basis `P₁`, `P₂` of a qualified knot set, and its slopes (module documentation). Plain constants of the grid:
/// no allocation, no table, a fixed number of operations per input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChebyshevClosedForm {
    /// `1/h = (g − 1)/2`.
    span: f32,
    /// `g − 2`, the last leaf's index.
    last: f32,
    /// `h/2`.
    half_h: f32,
    /// `2h + h²`.
    bump: f32,
    /// `2 + h`, the bump's factor in `P₂'`.
    bump_slope: f32,
}

impl ChebyshevClosedForm {
    /// The closed form of `k` Chebyshev terms on `knots`, where it holds.
    ///
    /// # Postconditions
    /// `Some` exactly when `knots` are uniform with chord tangents, `1 ≤ k ≤ 3` and the grid has at least three knots.
    #[must_use]
    pub fn of(knots: &Knots, k: usize) -> Option<Self> {
        let g = knots.grid();
        let holds = knots.is_uniform()
            && knots.tangents() == Tangents::Chord
            && (1..=TERMS).contains(&k)
            && g >= 3;
        holds.then(|| {
            let h = 2.0 / (g - 1) as f64;
            Self {
                span: 0.5 * (g - 1) as f32,
                last: (g - 2) as f32,
                half_h: (0.5 * h) as f32,
                bump: (2.0 * h + h * h) as f32,
                bump_slope: (2.0 + h) as f32,
            }
        })
    }

    /// `x̂`, `α`, `β`. A NaN input keeps `x̂` NaN and both bumps zero, so `P₁` and `P₂` are NaN.
    #[inline(always)]
    fn bumps(&self, x: f32) -> (f32, f32, f32) {
        let xc = x.clamp(-1.0, 1.0);
        let u = (xc + 1.0) * self.span;
        let (a, b) = (1.0 - u, u - self.last);
        (
            xc,
            if a > 0.0 { a } else { 0.0 },
            if b > 0.0 { b } else { 0.0 },
        )
    }

    /// `(P₁, P₂)` at `x`.
    ///
    /// # Postconditions
    /// The spline of terms `θ` at `x` is `θ₀ + θ₁P₁ + θ₂P₂`; `|P₁|, |P₂| ≤ 1.3` on finite inputs.
    #[inline(always)]
    #[must_use]
    pub fn basis(&self, x: f32) -> (f32, f32) {
        let (xc, alpha, beta) = self.bumps(x);
        let a = alpha * alpha * (1.0 - alpha);
        let b = beta * beta * (1.0 - beta);
        let p1 = (-self.half_h).mul_add(a - b, xc);
        let p2 = self.bump.mul_add(a + b, (2.0 * xc).mul_add(xc, -1.0));
        (p1, p2)
    }

    /// `[P₁, P₂, P₁', P₂']` at `x`: the basis and the slopes of its polynomial pieces in `x` (one-sided at `±1`).
    ///
    /// # Postconditions
    /// The spline's slope at `x` is `θ₁P₁' + θ₂P₂'` inside `[−1, 1]`; outside it the caller takes zero, as the tables
    /// do. `basis_slopes(x)[..2]` equals [`Self::basis`] bit for bit.
    #[inline(always)]
    #[must_use]
    pub fn basis_slopes(&self, x: f32) -> [f32; 4] {
        let (xc, alpha, beta) = self.bumps(x);
        let (a2, b2) = (alpha * alpha, beta * beta);
        let (a, b) = (a2 * (1.0 - alpha), b2 * (1.0 - beta));
        let (da, db) = (
            (-3.0f32).mul_add(a2, 2.0 * alpha),
            (-3.0f32).mul_add(b2, 2.0 * beta),
        );
        [
            (-self.half_h).mul_add(a - b, xc),
            self.bump.mul_add(a + b, (2.0 * xc).mul_add(xc, -1.0)),
            0.5f32.mul_add(da + db, 1.0),
            self.bump_slope.mul_add(db - da, 4.0 * xc),
        ]
    }
}

/// One or more banks' Chebyshev terms as planes `[bank][term][channel]`, [`TERMS`] planes per bank (terms beyond `k`
/// zero): what a kernel over channel lanes reads contiguously. The same layout holds a pullback's accumulators
/// `Σ ȳ P_i`, which [`Self::terms_gradient`] maps back to the parameters' layout.
///
/// # Invariants
/// `planes.len() == banks · TERMS · channels`; `plane(b, i)[c]` is bank `b`'s term `i` of channel `c`, read from the
/// parameters' channel-major layout `terms[b][c · k + i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TermPlanes {
    planes: Vec<f32>,
    channels: usize,
    banks: usize,
    k: usize,
}

impl TermPlanes {
    /// The planes of `terms`, each bank `channels × k` channel-major.
    ///
    /// # Panics
    /// When `k` is not in `1..=3` or a bank is not `channels · k` long.
    #[must_use]
    pub fn new(terms: &[&[f32]], channels: usize, k: usize) -> Self {
        assert!((1..=TERMS).contains(&k), "one to three terms");
        assert!(
            terms.iter().all(|bank| bank.len() == channels * k),
            "each bank channels x k terms"
        );
        let mut planes = vec![0.0f32; terms.len() * TERMS * channels];
        for (b, bank) in terms.iter().enumerate() {
            for (c, row) in bank.chunks_exact(k).enumerate() {
                for (i, &theta) in row.iter().enumerate() {
                    planes[(b * TERMS + i) * channels + c] = theta;
                }
            }
        }
        Self {
            planes,
            channels,
            banks: terms.len(),
            k,
        }
    }

    /// Bank `bank`'s three term planes, each `channels` long.
    ///
    /// # Panics
    /// When `bank` is out of range.
    #[inline]
    #[must_use]
    pub fn bank(&self, bank: usize) -> [&[f32]; TERMS] {
        assert!(bank < self.banks, "a bank of these planes");
        let at = |i: usize| (bank * TERMS + i) * self.channels;
        std::array::from_fn(|i| &self.planes[at(i)..at(i) + self.channels])
    }

    /// Banks the planes hold.
    #[inline]
    #[must_use]
    pub const fn banks(&self) -> usize {
        self.banks
    }

    /// Channels per plane.
    #[inline]
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Terms per channel of the source parameters.
    #[inline]
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Floats the planes hold: also the length of one block's accumulators in this layout.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.planes.len()
    }

    /// Whether the planes hold no bank.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    /// Bank `bank`'s parameter gradient, `channels × k` channel-major, from the blocks' accumulators in this layout
    /// (each block `Σ ȳ P_i` per bank, term and channel), summed in `f64` in block order and rounded once.
    ///
    /// # Panics
    /// When `bank` is out of range or a block is not [`Self::len`] long.
    #[must_use]
    pub fn terms_gradient(&self, blocks: &[&[f32]], bank: usize) -> Vec<f32> {
        assert!(bank < self.banks, "a bank of these planes");
        assert!(
            blocks.iter().all(|acc| acc.len() == self.planes.len()),
            "accumulators in this layout"
        );
        let (channels, k) = (self.channels, self.k);
        (0..channels * k)
            .map(|at| {
                let (c, i) = (at / k, at % k);
                let plane = (bank * TERMS + i) * channels + c;
                blocks.iter().map(|acc| f64::from(acc[plane])).sum::<f64>() as f32
            })
            .collect()
    }
}

/// A prepared Chebyshev--CR spline's closed form: the basis of its knots and its banks' term planes, kept per parameter
/// version beside the tables ([`crate::catmull_rom::PreparedSpline::closed_form`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedForm {
    form: ChebyshevClosedForm,
    planes: TermPlanes,
}

impl ClosedForm {
    /// The closed form of `terms` (each bank `channels × k`) on `knots`, where it holds ([`ChebyshevClosedForm::of`]).
    ///
    /// # Panics
    /// As [`TermPlanes::new`], when the form holds.
    #[must_use]
    pub fn of(terms: &[&[f32]], channels: usize, (knots, k): (&Knots, usize)) -> Option<Self> {
        let form = ChebyshevClosedForm::of(knots, k)?;
        Some(Self {
            form,
            planes: TermPlanes::new(terms, channels, k),
        })
    }

    /// The basis of the knots.
    #[inline]
    #[must_use]
    pub const fn form(&self) -> &ChebyshevClosedForm {
        &self.form
    }

    /// The banks' term planes.
    #[inline]
    #[must_use]
    pub const fn planes(&self) -> &TermPlanes {
        &self.planes
    }
}
