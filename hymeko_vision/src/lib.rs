//! **HyMeKo vision primitives.**
//!
//! Created 2026-08-24 by explicit authorisation, carrying **only** the skeleton-rasterisation and
//! masking operators, moved bit-identically out of `nagare/src/vision/skeleton_render.rs`. See
//! `../nagare/reports/2026-08-24-hymeko-vision-core-crate.md` and
//! `../nagare/docs/plans/2026-08-24-hymeko-vision-core-crate/`.
//!
//! # Dependency-free, by contract
//! This crate has **no dependencies and will not acquire one without a separate CORE.YAML
//! approval**. It is written to `hymeko_clifford`'s stated standard rather than merely near it:
//! `forbid(unsafe_code)`, no `unwrap`/`expect` in production code, and Design-by-Contract docs
//! (`# Preconditions` / `# Postconditions` / `# Panics`) on every public item.
//!
//! # Scope, and why it is narrow on purpose
//! The authorisation is for these operators **initially and exclusively**. The pose generators, the
//! skeleton incidence, the bone-length penalty and the coordinate refiner that sit beside them in
//! the application are deliberately **not** here. Anything that later looks core-worthy is a new
//! proposal, not an extension of this one.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod skeleton_render;

pub use skeleton_render::{draw_segment, occlude_box, rasterise_skeleton, PixelBox, RasterGrid};
