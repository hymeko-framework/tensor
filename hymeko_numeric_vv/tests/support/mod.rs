//! Shared support for the V&V suites in this package.
//!
//! Lives under `tests/`, so it is **not part of any library target**: nothing outside this
//! package's own test binaries can name it. That is the mechanical form of "frozen oracles are
//! test-only", and it is stronger than `pub(crate)` would be.

pub mod frozen_clamp_oracle;
