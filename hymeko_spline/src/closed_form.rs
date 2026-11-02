//! The closed form of the uniform Chebyshev--Catmull--Rom spline of at most three terms, under chord tangents (Nagare
//! plan 2026-10-30-t2-algebraic-lowering; derivation, symbolic proof and measurements in
//! `nagare/reports/2026-10-30-hsikan-performance-challenge.md`, §3) and under bounded tangents ("Bounded tangents").
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
//! # Bounded tangents
//! Under [`Tangents::Bounded`] (each knot takes the chord, or three times its smaller-magnitude secant where that is
//! smaller, or zero where its secants do not agree in sign) the same spline is the chord form plus at most two bumps
//! (Nagare plan 2026-11-02-bounded-closed-form; symbolic proof, exact rationals, grids 3..12 and k = 1..3,
//! `nagare/run_outputs/2026-11-02-bounded-closed-form/m_bounded_closed_form.py`):
//!
//! `φ_B(x) = θ₀ + θ₁P₁ + θ₂P₂ + h Σ_{k∈K*} δ_k K(u − k)`, `K(s) = s(1 − |s|)²` on `|s| ≤ 1`, zero outside.
//!
//! - A Hermite leaf is linear in its knot tangents and `hK(u − k)` is the cardinal function of knot `k`'s tangent, so
//!   a tangent that departs from the quadratic's slope by `δ_k` adds `hδ_kK(u − k)`. T2's `A` and `−B` are this bump
//!   at the two end knots, whose tangent the bounded rule leaves as the chord rule has it.
//! - At an interior knot the sampled quadratic's secants are `a_k ∓ β` and its chord `a_k`, with `a_k = θ₁ + 4θ₂ξ_k`,
//!   `β = 2θ₂h`, `b = |β|`. The rule keeps the chord iff `|a_k| ≥ 3b/2`; it takes zero for `|a_k| ≤ b`
//!   (`δ_k = −a_k`) and three times the smaller secant in between (`δ_k = sgn(a_k)(2|a_k| − 3b)`).
//! - `a_k = (2β/h)(ξ_k − x*)` with the parabola's vertex `x* = −θ₁/(4θ₂)`, so the corrected knots `K*` lie within
//!   `3h/4` of it: at most two. `δ` is continuous in `θ` and vanishes on the border of `K*`.
//!
//! Slope `φ_B' = θ₁P₁' + θ₂P₂' + Σ δ_k K'(u − k)`, `K'(s) = (1 − |s|)(1 − 3|s|)`; parameter gradient
//! `∂φ_B/∂θ_i = P_i + h Σ (∂δ_k/∂θ_i) K(u − k)`, piecewise in `θ`. On a cell border (a zero secant, the chord equal to a
//! candidate) the spline has a kink in `θ`; there this is one side's derivative and need not be the tables'.
//! [`BoundedClosedForm`] gives the knot defects per channel's terms and their sum at an input.
//!
//! # Qualification
//! [`ChebyshevClosedForm::of`] answers only for uniform knots, chord tangents, `1 ≤ k ≤ 3` and `g ≥ 3`;
//! [`BoundedClosedForm::of`] for the same knots and terms under bounded tangents. Refined knots and four or more terms
//! have no closed form here: their callers keep the tables. Nothing is approximated: on the qualified family the closed
//! form and the tables are the same real function.

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
        qualifies(knots, k, Tangents::Chord).then(|| Self::uniform(knots.grid()))
    }

    /// The basis of `g ≥ 3` uniform knots.
    fn uniform(g: usize) -> Self {
        let h = 2.0 / (g - 1) as f64;
        Self {
            span: 0.5 * (g - 1) as f32,
            last: (g - 2) as f32,
            half_h: (0.5 * h) as f32,
            bump: (2.0 * h + h * h) as f32,
            bump_slope: (2.0 + h) as f32,
        }
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

/// Whether `k` terms on `knots` have a closed form under `tangents`: uniform knots of that rule, `1 ≤ k ≤ 3`, `g ≥ 3`.
fn qualifies(knots: &Knots, k: usize, tangents: Tangents) -> bool {
    knots.is_uniform()
        && knots.tangents() == tangents
        && (1..=TERMS).contains(&k)
        && knots.grid() >= 3
}

/// One interior knot's tangent defect under bounded tangents (module documentation, "Bounded tangents"): the knot,
/// `δ` and `(∂δ/∂θ₁, ∂δ/∂θ₂)`. The default is the absent defect, zero everywhere.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KnotDefect {
    knot: f32,
    delta: f32,
    d_delta: [f32; 2],
}

impl KnotDefect {
    /// The knot's index `k`, its bump centred at `u = k`.
    #[inline]
    #[must_use]
    pub const fn knot(&self) -> f32 {
        self.knot
    }

    /// `δ_k`, the bounded tangent less the quadratic's slope at the knot.
    #[inline]
    #[must_use]
    pub const fn delta(&self) -> f32 {
        self.delta
    }

    /// `(∂δ_k/∂θ₁, ∂δ_k/∂θ₂)`; `δ` never depends on `θ₀`.
    #[inline]
    #[must_use]
    pub const fn d_delta(&self) -> [f32; 2] {
        self.d_delta
    }
}

/// The closed form of the uniform Chebyshev--CR spline under bounded tangents: the chord basis of the same knots
/// ([`Self::form`]) plus, per channel's terms, at most two knot defects ([`Self::defects`], once per parameter
/// version) summed at an input ([`Self::correction`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedClosedForm {
    form: ChebyshevClosedForm,
    grid: usize,
}

impl BoundedClosedForm {
    /// The closed form of `k` Chebyshev terms on `knots` under bounded tangents, where it holds.
    ///
    /// # Postconditions
    /// `Some` exactly when `knots` are uniform with bounded tangents, `1 ≤ k ≤ 3` and the grid has at least three knots.
    #[must_use]
    pub fn of(knots: &Knots, k: usize) -> Option<Self> {
        qualifies(knots, k, Tangents::Bounded).then(|| Self {
            form: ChebyshevClosedForm::uniform(knots.grid()),
            grid: knots.grid(),
        })
    }

    /// The chord basis `P₁`, `P₂` the bounded spline departs from.
    #[inline]
    #[must_use]
    pub const fn form(&self) -> &ChebyshevClosedForm {
        &self.form
    }

    /// The defects of one channel's terms `θ` (terms beyond `k` zero), formed in `f64` and rounded once.
    ///
    /// # Postconditions
    /// The corrected knots are the interior `k` with `|a_k| < 3b/2`, at most two (the vertex window), in knot order;
    /// unused entries are [`KnotDefect::default`]. With `θ₂ = 0` (no vertex, every knot keeps the chord) both are.
    #[must_use]
    pub fn defects(&self, theta: [f32; TERMS]) -> [KnotDefect; 2] {
        let (t1, t2) = (f64::from(theta[1]), f64::from(theta[2]));
        let mut out = [KnotDefect::default(); 2];
        if t2 == 0.0 {
            return out;
        }
        let h = 2.0 / (self.grid - 1) as f64;
        let (b, sign_t2) = (2.0 * t2.abs() * h, t2.signum());
        let corrected = (1..self.grid - 1).filter_map(|knot| {
            let xi = (knot as f64).mul_add(h, -1.0);
            let a = (4.0 * t2).mul_add(xi, t1);
            let (delta, d_delta) = if a.abs() <= b {
                (-a, [-1.0, -4.0 * xi])
            } else if a.abs() < 1.5 * b {
                let s = a.signum();
                (
                    s * 3.0f64.mul_add(-b, 2.0 * a.abs()),
                    [2.0, (6.0 * h * s * sign_t2).mul_add(-1.0, 8.0 * xi)],
                )
            } else {
                return None;
            };
            Some(KnotDefect {
                knot: knot as f32,
                delta: delta as f32,
                d_delta: d_delta.map(|v| v as f32),
            })
        });
        for (slot, defect) in out.iter_mut().zip(corrected) {
            *slot = defect;
        }
        out
    }

    /// The defects' sum at `x`: `[h Σ δK, Σ δK', h Σ (∂δ/∂θ₁)K, h Σ (∂δ/∂θ₂)K]`, with `K` at `u − k`.
    ///
    /// # Postconditions
    /// The bounded spline at `x` is `θ₀ + θ₁P₁ + θ₂P₂ + c[0]`; its slope inside `[−1, 1]` is `θ₁P₁' + θ₂P₂' + c[1]`
    /// (zero outside, as [`ChebyshevClosedForm::basis_slopes`]); `∂φ/∂θ = (1, P₁ + c[2], P₂ + c[3])`. A NaN input gives
    /// NaN.
    #[inline]
    #[must_use]
    pub fn correction(&self, defects: &[KnotDefect; 2], x: f32) -> [f32; 4] {
        let u = (x.clamp(-1.0, 1.0) + 1.0) * self.form.span;
        let h = 2.0 * self.form.half_h;
        defects.iter().fold([0.0; 4], |[value, slope, g1, g2], d| {
            let s = u - d.knot;
            let m = (1.0 - s.abs()).max(0.0);
            let bump = h * (s * m * m);
            let bump_slope = m * (-3.0f32).mul_add(s.abs(), 1.0);
            [
                d.delta.mul_add(bump, value),
                d.delta.mul_add(bump_slope, slope),
                d.d_delta[0].mul_add(bump, g1),
                d.d_delta[1].mul_add(bump, g2),
            ]
        })
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
