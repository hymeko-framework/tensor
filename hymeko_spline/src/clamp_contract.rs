//! Which clamp Jacobian a spline backward applies to its **input** gradient.
//!
//! # Why this type exists
//!
//! Every spline forward in this crate clamps its input to `[-1, 1]` before evaluating, so beyond
//! that interval the output is **constant in `x`** and the true input derivative is exactly `0`.
//! Until 2026-08-25 the two *cached* backwards ([`crate::catmull_rom_backward`] and
//! [`crate::kb_backward`]) guarded that case with `(-1.0..=1.0).contains(&x_clamped)` — a
//! predicate on the already-clamped value, which is **always true**. They therefore returned the
//! interior derivative everywhere, omitting the derivative of the clamp itself.
//!
//! The intent was never in doubt: `catmull_rom_backward`'s own documented postcondition says the
//! input gradient is zero outside the interval, and the third kernel in the same file,
//! [`crate::chebyshev_deploy_backward`], receives the raw `x` and has always tested *it*
//! correctly. The two cached paths were the ones that discarded the raw value.
//!
//! Diagnosed in `nagare/reports/2026-08-25-hsikan-grad-x-root-cause-audit.md` and
//! `nagare/reports/2026-08-25-coginfocom-clamp-contract-and-core-blast-radius.md`.
//!
//! # What each variant means
//!
//! [`ClampGradientContract::Corrected`] is the mathematics and is the **default**: the plain
//! backward entry points apply it. [`ClampGradientContract::LegacyRecorded`] reproduces the
//! pre-2026-08-25 arithmetic **bit-for-bit**, and exists for exactly one purpose — so numbers
//! recorded under the old behaviour stay re-derivable instead of becoming history no one can
//! reproduce. It is never a fallback, never a compatibility mode, and is selectable only by
//! naming it at an explicitly-named call site.
//!
//! # What this type does NOT affect
//!
//! **Parameter gradients are identical under both contracts.** The clamp sits between the input
//! and the spline evaluation, so it is on the path to `grad_x` alone; `grad_coef` and `grad_tcb`
//! accumulate outside the guard and are bit-identical either way. Tests in both kernel modules
//! assert this rather than asserting it in prose.

/// The clamp Jacobian a spline backward applies to its input gradient.
///
/// See the [module documentation](self) for why both variants exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ClampGradientContract {
    /// The mathematically correct Jacobian: `d clamp(x)/dx` is `1` on `[-1, 1]` and `0` outside,
    /// so the input gradient is **exactly zero** where the forward saturated.
    ///
    /// This is the default and what every plain backward entry point applies.
    #[default]
    Corrected,
    /// The pre-2026-08-25 arithmetic, reproduced exactly: the interior derivative is returned for
    /// every element, saturated or not.
    ///
    /// Provided so recorded results remain re-derivable. Selecting it is a statement that you
    /// want the historical number, not the correct one.
    LegacyRecorded,
}

impl ClampGradientContract {
    /// Whether an element whose forward input was `interior` receives an input gradient.
    ///
    /// Under [`Self::Corrected`] this is exactly `interior`; under [`Self::LegacyRecorded`] it is
    /// unconditionally `true`, which is what the tautological guard computed.
    #[inline(always)]
    #[must_use]
    pub const fn propagates_input_gradient(self, interior: bool) -> bool {
        match self {
            Self::Corrected => interior,
            Self::LegacyRecorded => true,
        }
    }
}
