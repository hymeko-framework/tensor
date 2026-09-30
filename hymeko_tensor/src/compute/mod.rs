// APPROVED-CORE-EDIT: compute-backend-vocabulary (2026-08-24, user chat).
//
// Minimal backend-independent vocabulary shared by the CPU backend
// (`hymeko_hnn`) and the WGPU backend (`hymeko_wgpu`). It exists so that
// `hymeko_wgpu` does not have to depend on `hymeko_hnn` merely to name a shape
// or an error.
//
// SCOPE — deliberately four types, each with a concrete consumer in Phase 1/2:
//
//   TensorShape     dense 2D extent      CPU: dimension asserts
//                                        GPU: buffer sizing, workgroup counts
//   SparseCsrShape  CSR extent + nnz     CPU: TensorCsr validation
//                                        GPU: GpuCsrMatrix upload, dispatch size
//   BackendKind     which backend ran    both: error attribution, bench labels
//   ComputeError    shared error type    both: fallible prepare/dispatch
//
// EXPLICITLY NOT HERE, per the approving instruction: any `wgpu` type, async
// runtime, GPU buffer handle, pipeline, bind group, shader, command encoder,
// adapter/device object, backend-specific usage flag, backend implementation,
// compute graph, resource repository, factory hierarchy, dynamic dispatch, or
// any abstraction lacking a concrete CPU *and* WGPU consumer today.
//
// This numeric vocabulary has no external dependency. Errors are implemented
// by hand rather than with `thiserror` to preserve the dependency-free default.
//
// DType / ScalarType is deliberately ABSENT. The first GPU path is explicitly
// f32 (`GpuTensor<f32>`), so the element type is carried by the type parameter
// and a runtime dtype tag would have no consumer yet. Introduce it when a second
// element type actually lands.

use core::fmt;

/// Which execution backend a result came from, or an error refers to.
///
/// Ordering is not meaningful. This is a label, not a preference ranking — the
/// selection policy is deliberately not modelled here (there is no consumer for
/// one yet).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Single-threaded CPU kernels (`*_into_serial`).
    CpuSerial,
    /// Rayon CPU kernels (`*_into_parallel`).
    CpuRayon,
    /// WGPU compute pipeline.
    Wgpu,
}

impl BackendKind {
    /// Stable short name, used in benchmark ids and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuSerial => "cpu-serial",
            Self::CpuRayon => "cpu-rayon",
            Self::Wgpu => "wgpu",
        }
    }

    /// Whether this backend executes on the host CPU.
    ///
    /// The distinction that matters to callers is not the vendor but whether
    /// results need an explicit readback before they are host-visible.
    #[must_use]
    pub const fn is_host(self) -> bool {
        matches!(self, Self::CpuSerial | Self::CpuRayon)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Extent of a dense, row-major 2D tensor.
///
/// # Invariants
/// `rows` and `cols` are counts, not byte sizes. A zero extent is legal and
/// means an empty tensor — the CPU kernels already accept that, and the GPU
/// backend must not diverge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorShape {
    pub rows: usize,
    pub cols: usize,
}

impl TensorShape {
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Element count, or `None` on overflow.
    ///
    /// Returns an `Option` rather than panicking because the GPU path must
    /// reject an oversized descriptor with a `ComputeError`, not abort.
    #[must_use]
    pub const fn checked_elements(&self) -> Option<usize> {
        self.rows.checked_mul(self.cols)
    }

    /// Element count.
    ///
    /// # Panics
    /// If `rows * cols` overflows `usize`. Use [`Self::checked_elements`] where
    /// the extent comes from untrusted input.
    #[must_use]
    pub const fn elements(&self) -> usize {
        match self.checked_elements() {
            Some(n) => n,
            None => panic!("TensorShape element count overflowed usize"),
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Verify a flat buffer matches this shape.
    ///
    /// # Errors
    /// [`ComputeError::ShapeMismatch`] when `len != rows * cols`.
    pub fn validate_len(&self, operand: &'static str, len: usize) -> Result<(), ComputeError> {
        let expected = self.checked_elements().ok_or(ComputeError::IndexOverflow {
            what: operand,
            value: usize::MAX,
        })?;
        if len == expected {
            Ok(())
        } else {
            Err(ComputeError::ShapeMismatch {
                operand,
                expected,
                actual: len,
            })
        }
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}

/// Extent of a CSR sparse matrix: logical shape plus stored non-zero count.
///
/// `nnz` is carried alongside the shape because every consumer needs it and
/// none can derive it: the CPU kernels size their inner loops by it, and the
/// GPU backend sizes `col_ind` / `values` buffers by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SparseCsrShape {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

impl SparseCsrShape {
    #[must_use]
    pub const fn new(rows: usize, cols: usize, nnz: usize) -> Self {
        Self { rows, cols, nnz }
    }

    /// Dense shape of `Y` in `Y = A * X` for a given feature dimension.
    #[must_use]
    pub const fn output_shape(&self, feature_dim: usize) -> TensorShape {
        TensorShape::new(self.rows, feature_dim)
    }

    /// Dense shape of `X` in `Y = A * X` for a given feature dimension.
    #[must_use]
    pub const fn input_shape(&self, feature_dim: usize) -> TensorShape {
        TensorShape::new(self.cols, feature_dim)
    }

    /// Length the `row_ptr` array must have.
    #[must_use]
    pub const fn row_ptr_len(&self) -> usize {
        self.rows + 1
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0 || self.nnz == 0
    }
}

impl fmt::Display for SparseCsrShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} nnz={}", self.rows, self.cols, self.nnz)
    }
}

/// Errors shared by every compute backend.
///
/// Concrete rather than boxed, and free of backend types, so `hymeko_core` can
/// name it without depending on any backend crate. A backend that needs to
/// report something device-specific wraps the message in
/// [`ComputeError::Backend`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeError {
    /// A buffer length did not match the shape it was declared against.
    ShapeMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A `usize` extent did not fit the index width a backend requires.
    ///
    /// WGSL indices are `u32`; this is the checked conversion's failure mode,
    /// and the reason [`try_u32`] exists rather than a bare `as` cast.
    IndexOverflow { what: &'static str, value: usize },
    /// The requested backend could not be initialised in this environment.
    ///
    /// Distinct from [`Self::Backend`] on purpose: a missing GPU adapter is an
    /// environment fact that a test suite may legitimately skip on, whereas a
    /// shader compilation failure is a defect and must not be skipped.
    BackendUnavailable {
        backend: BackendKind,
        reason: String,
    },
    /// A backend-specific failure after initialisation succeeded.
    Backend {
        backend: BackendKind,
        message: String,
    },
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                operand,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch for `{operand}`: expected {expected} elements, got {actual}"
            ),
            Self::IndexOverflow { what, value } => write!(
                f,
                "`{what}` = {value} does not fit the backend's 32-bit index width"
            ),
            Self::BackendUnavailable { backend, reason } => {
                write!(f, "backend `{backend}` unavailable: {reason}")
            }
            Self::Backend { backend, message } => {
                write!(f, "backend `{backend}` failed: {message}")
            }
        }
    }
}

impl std::error::Error for ComputeError {}

/// Checked `usize` → `u32` conversion for backends with 32-bit indices.
///
/// WGSL has no 64-bit integer indexing, so every extent crossing into a shader
/// must narrow. Doing it with `as` would silently wrap and produce a kernel that
/// reads the wrong memory; this returns [`ComputeError::IndexOverflow`] instead.
///
/// # Errors
/// When `value > u32::MAX`.
#[inline]
pub fn try_u32(what: &'static str, value: usize) -> Result<u32, ComputeError> {
    u32::try_from(value).map_err(|_| ComputeError::IndexOverflow { what, value })
}
