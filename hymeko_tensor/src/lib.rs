//! Tensor storage, numerical contracts and kernels, independent of canonical IR.
//!
//! The default build is dependency-free: scalar algebra, degree computation,
//! weight initialization, dense solvers ([`linalg`]), information estimators
//! ([`renyi`]), and parameter layout ([`stream`]).
//!
//! Optional features:
//! - `sparse`: COO/CSR storage and serial/Rayon kernels.
//! - `interop`: sparse storage plus Nalgebra and JAX conversion.
//! - `arrow-schema`: Arrow schemas for exchanging tensor data.
//!
//! Core owns canonical-IR adapters; Nagare owns graph-aware transformations
//! and neural-network layers. Neither is a dependency of this crate.

pub mod contraction;
pub mod linalg;
pub mod operator;
pub mod renyi;
pub mod stream;

pub mod aggregation;
#[cfg(feature = "arrow-schema")]
pub mod arrow_schema;
pub mod common;
pub mod compute;
pub mod conv;
#[cfg(feature = "sparse")]
pub mod representations;
#[cfg(feature = "sparse")]
#[doc(hidden)]
pub mod sparse_workload;
#[cfg(feature = "interop")]
pub mod tensor_convert;
pub mod tensor_val;
#[cfg(feature = "sparse")]
pub mod util;
