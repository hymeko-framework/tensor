//! Tests for the backend-independent compute vocabulary.
//!
//! APPROVED-CORE-EDIT: compute-backend-vocabulary (2026-08-24) added exactly
//! four types to `hymeko_tensor::compute`. These tests pin the parts a
//! backend actually relies on:
//!
//! * `try_u32` must **reject** rather than wrap — a silent `as` cast would
//!   produce a WGSL kernel that indexes the wrong memory, which is the failure
//!   mode this function exists to prevent;
//! * empty extents are legal, because the CPU kernels already accept them and
//!   the GPU backend must not diverge;
//! * `checked_elements` must not panic on overflow, because the GPU path has to
//!   return a `ComputeError` for an oversized descriptor rather than abort.

#[cfg(test)]
mod test_compute_vocabulary {
    use hymeko_tensor::compute::{try_u32, BackendKind, ComputeError, SparseCsrShape, TensorShape};

    #[test]
    fn test_backend_kind_names_and_host_flag() {
        assert_eq!(BackendKind::CpuSerial.as_str(), "cpu-serial");
        assert_eq!(BackendKind::CpuRayon.as_str(), "cpu-rayon");
        assert_eq!(BackendKind::Wgpu.as_str(), "wgpu");
        assert_eq!(format!("{}", BackendKind::Wgpu), "wgpu");

        // The distinction that matters is whether a readback is needed.
        assert!(BackendKind::CpuSerial.is_host());
        assert!(BackendKind::CpuRayon.is_host());
        assert!(!BackendKind::Wgpu.is_host());
    }

    #[test]
    fn test_tensor_shape_elements_and_empty() {
        let s = TensorShape::new(4, 8);
        assert_eq!(s.elements(), 32);
        assert_eq!(s.checked_elements(), Some(32));
        assert!(!s.is_empty());
        assert_eq!(format!("{s}"), "4x8");

        // Empty is legal in either axis: the CPU kernels accept a zero-row
        // matrix and a zero feature_dim, so the vocabulary must too.
        assert!(TensorShape::new(0, 8).is_empty());
        assert!(TensorShape::new(4, 0).is_empty());
        assert_eq!(TensorShape::new(0, 8).elements(), 0);
    }

    #[test]
    fn test_tensor_shape_overflow_is_reported_not_panicked() {
        let huge = TensorShape::new(usize::MAX, 2);
        assert_eq!(
            huge.checked_elements(),
            None,
            "an overflowing extent must be reportable, not fatal"
        );
        assert!(TensorShape::new(usize::MAX, 1).checked_elements().is_some());
    }

    #[test]
    fn test_tensor_shape_validate_len() {
        let s = TensorShape::new(3, 2);
        assert!(s.validate_len("x", 6).is_ok());

        match s.validate_len("x", 5) {
            Err(ComputeError::ShapeMismatch {
                operand,
                expected,
                actual,
            }) => {
                assert_eq!(operand, "x");
                assert_eq!(expected, 6);
                assert_eq!(actual, 5);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }

        // The message must name the operand - a backend reports several buffers
        // and "shape mismatch" alone would not say which.
        let msg = format!("{}", s.validate_len("row_ptr", 1).unwrap_err());
        assert!(
            msg.contains("row_ptr"),
            "message should name the operand: {msg}"
        );
    }

    #[test]
    fn test_sparse_csr_shape_derived_extents() {
        let a = SparseCsrShape::new(10, 7, 25);
        assert_eq!(a.row_ptr_len(), 11);
        assert_eq!(a.input_shape(4), TensorShape::new(7, 4), "X is cols x fd");
        assert_eq!(a.output_shape(4), TensorShape::new(10, 4), "Y is rows x fd");
        assert!(!a.is_empty());
        assert_eq!(format!("{a}"), "10x7 nnz=25");

        // Structurally empty: rows exist but nothing is stored.
        assert!(SparseCsrShape::new(5, 5, 0).is_empty());
        assert_eq!(SparseCsrShape::new(0, 0, 0).row_ptr_len(), 1);
    }

    #[test]
    fn test_try_u32_rejects_instead_of_wrapping() {
        assert_eq!(try_u32("i", 0).unwrap(), 0);
        assert_eq!(try_u32("i", u32::MAX as usize).unwrap(), u32::MAX);

        // The case that matters: one past u32::MAX must NOT wrap to 0.
        let over = (u32::MAX as usize) + 1;
        match try_u32("col_ind entry", over) {
            Err(ComputeError::IndexOverflow { what, value }) => {
                assert_eq!(what, "col_ind entry");
                assert_eq!(value, over);
            }
            Ok(v) => panic!("silently narrowed {over} to {v} - this is the bug the fn prevents"),
            Err(other) => panic!("expected IndexOverflow, got {other:?}"),
        }

        let msg = format!("{}", try_u32("nnz", over).unwrap_err());
        assert!(
            msg.contains("32-bit"),
            "message should explain the width: {msg}"
        );
    }

    #[test]
    fn test_compute_error_is_a_std_error_and_names_the_backend() {
        let unavailable = ComputeError::BackendUnavailable {
            backend: BackendKind::Wgpu,
            reason: "no adapter".to_owned(),
        };
        let failed = ComputeError::Backend {
            backend: BackendKind::Wgpu,
            message: "shader did not compile".to_owned(),
        };

        // Both name the backend, and they are distinguishable - a test suite may
        // skip on the first but never on the second.
        assert!(format!("{unavailable}").contains("wgpu"));
        assert!(format!("{failed}").contains("wgpu"));
        assert_ne!(unavailable, failed);

        // Usable as a boxed std error, so callers can mix it with `?`.
        let boxed: Box<dyn std::error::Error> = Box::new(failed);
        assert!(boxed.to_string().contains("shader did not compile"));
    }
}
