//! # `hymeko_numeric_vv` — downstream numerical verification & validation
//!
//! This crate is a **client** of the `tensor` core crates, never a peer of them. It exists
//! because CLAUDE.md §3 asks for a `publish = false` contract/equivalence package that owns the
//! verification surface, so that verification never has to be smuggled back into `examples/` or
//! into a second production module.
//!
//! ## The boundary, stated once
//!
//! * **Production never depends on this crate.** Not in `[dependencies]`, not in
//!   `[dev-dependencies]`. [`manifest_gate`] proves it by parsing every manifest in the workspace,
//!   so the claim is checked rather than intended.
//! * **This crate owns no model, trainer, optimizer, kernel or data loader.** It owns
//!   *mechanisms for judging* them: finite differences ([`fd`]), one-sided scientific acceptance
//!   ([`gate`]), and source/manifest gates ([`manifest_gate`], [`source_gate`]). No digest lives
//!   here: `nagare_vv` owns the repository's single minimal SHA-256, and duplicating it across
//!   the two repositories is exactly the §6.1 copy this package exists to avoid.
//! * **Frozen oracles do not live here.** They live in `tests/support/`, which is not part of any
//!   library target — so nothing outside this package's own test binaries can even name them.
//!   That is a stronger statement than `pub(crate)`, and it is the one the mission asks for.
//!
//! ## What "one-sided" means here
//!
//! A scientific result recorded at some score is a *floor* (higher-is-better) or a *ceiling*
//! (lower-is-better), never an equality. A model that improves must pass. See [`gate`] for the
//! error-band rules — in particular that the band must come from measured variance, a recorded
//! process band, a numerical floor, or a pre-existing justified tolerance, and never from a round
//! number chosen after seeing a failure.
//!
//! Exact equality remains correct for bit contracts, invariants, optimizer lockstep, artifact
//! integrity and frozen-oracle parity. Those live in the exact suites, not behind [`gate`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod fd;
pub mod gate;
pub mod manifest_gate;
pub mod source_gate;
