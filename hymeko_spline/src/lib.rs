//! # hymeko_spline — spline activation kernels
//!
//! Catmull-Rom, Chebyshev-CR (train-CR / deploy-Chebyshev) and Kochanek-Bartels,
//! on flat row-major `f32` slices. Nothing here knows what a network layer is.
//!
//! Moved out of `nagare/src/ops/` on 2026-08-17: `nagare`'s subject is holonomy,
//! and generic mathematics belongs on the library side of the boundary. See
//! `nagare/reports/2026-08-17-library-boundary-inventory.md`.
//!
//! # Why a separate crate
//!
//! These kernels want `rayon`; `hymeko_tensor` and `hymeko_clifford` are
//! deliberately dependency-free and should stay so. A fourth crate keeps both
//! properties instead of trading one for the other, and `rayon` is *still*
//! optional here — with `parallel` off this crate has no dependencies either.
//!
//! **The feature changes who does the work, never what is computed.** Both paths
//! block the rows identically; the block layout is a function of `(n, channels)`
//! only, never the thread count or the feature. Results are byte-identical with
//! `parallel` on or off, and across 1, 8 or 16 threads.
//!
//! # Build contract
//!
//! `catmull_rom`'s forward calls `f32::floor`, which on baseline `x86-64` becomes
//! a **libm call** — 3.71 % / 4.36 % of samples in the KAN / HSiKAN profiles.
//! `tensor/.cargo/config.toml` enables `+sse4.1` so it is one `roundss`. Removing
//! that config costs ~10 % on KAN and ~5.6 % on HSiKAN.
//!
//! **This is now enforced.** Until 2026-08-19 no test detected the loss and the config file was
//! untracked, so a `git clean` could delete the contract and leave every test green while
//! everything ran ~10 % slower. `tests/build_contract.rs` fails loudly if the target feature is
//! absent, and both config files are now under version control (see `CORE.YAML`).

// Framework-grade lints, matching `hymeko_clifford` (CORE.YAML, 2026-08-19). `forbid` rather
// than `deny`: this crate is on the measured hot path of every KAN and HSiKAN run, and a local
// override is not a decision that should be available without a core-level discussion.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Catmull-Rom and Chebyshev-CR kernels on flat row-major `f32` slices.
pub mod catmull_rom;
/// The knots a Catmull–Rom spline interpolates: a uniform base grid, each segment refined on a dyadic orthtree.
pub mod knots;
/// Kochanek-Bartels (TCB) spline kernels.
pub mod kochanek_bartels;
/// Rational B-spline curves and parameter derivatives in f64 homogeneous coordinates.
pub mod nurbs;

pub use catmull_rom::{
    catmull_rom_backward, catmull_rom_backward_into, catmull_rom_banks_backward_into,
    catmull_rom_banks_forward_into, catmull_rom_forward, catmull_rom_forward_into,
    chebyshev_control_points, chebyshev_cr_backward, chebyshev_cr_backward_into,
    chebyshev_cr_banks_backward_into, chebyshev_cr_banks_forward_into, chebyshev_cr_forward,
    chebyshev_cr_forward_into, chebyshev_deploy_backward, chebyshev_deploy_forward,
    chebyshev_knot_basis, CatmullRomBackward, CatmullRomBanksBuffers, CatmullRomBuffers,
    CatmullRomCache, ChebyshevCrBackward, PAR_MIN_ELEMS,
};
pub use knots::Knots;
pub use kochanek_bartels::{kb_backward, kb_forward, KbBackward, KbCache};
