// APPROVED-CORE-EDIT: sparse-tensor-authority

//! The sparse **semantic** contract: what a sparse tensor *is*, independent of how it is stored.
//!
//! # Why this module exists
//!
//! Before it, CSR was the semantic authority: consumers read `TensorCsr`'s public fields and
//! inferred meaning from them, and the device backend re-derived structural validity for itself.
//! That put three different things in one type — what the tensor means, how it is laid out, and
//! whether anyone had checked it.
//!
//! The split this module introduces:
//!
//! | Layer | Owner | Owns |
//! |---|---|---|
//! | semantics | this module | scalar, rank, extents, nnz, layout identity, order, multiplicity, validation |
//! | layout | [`TensorCsr`], [`TensorCoo`] | the physical arrangement of stored entries |
//! | device | `hymeko_wgpu` | narrowing, allocation, limits — **no** structural re-check |
//!
//! # What it deliberately does not do
//!
//! It does not introduce a third sparse container, and it does not unify the two layouts by hiding
//! what makes them different. `TensorCsr` is rank 2; `TensorCoo` is rank 3 over `(k, i, j)`.
//! Flattening the slice axis to make one sum type fit would erase the very content of the COO
//! format, so the rank difference is carried **typed**, in the length of [`SparseShape::extents`] —
//! a real value, never a string tag and never a sentinel extent. Each layout declares its own arity
//! and is validated against it.
//!
//! # The receipt
//!
//! Both layout types keep entirely public fields — 31 consumers across two repositories read them
//! directly, so making them private would be a breaking change far outside this boundary. An
//! invalid `TensorCsr` is therefore still *constructible* by hand. [`ValidatedSparse`] is the
//! answer: it holds an immutable borrow of the very value it validated, its fields are private, and
//! its constructor is scoped to this module tree. A consumer that accepts nothing but a
//! `ValidatedSparse` — as the WGPU adapter now does — cannot be handed unvalidated structure, and
//! needs no validator of its own.
//!
//! The binding is what makes it sound. A proof carrying only *metadata* would be interchangeable
//! between two tensors that share a shape and differ in structure; a proof that borrows the tensor
//! cannot be moved to another one at all.

use crate::common::Real;

/// Which physical layout stores the entries.
///
/// Layout identity is **not** rank and **not** shape: none of the three can be inferred from
/// another. A rank-2 tensor may be stored as either layout, and two layouts of the same logical
/// tensor are two storage identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseLayout {
    /// Compressed sparse row: `row_ptr` delimits per-row runs of `(col_ind, val)`.
    Csr,
    /// Coordinate list: an ordered sequence of `(k, i, j, v)` entries.
    Coo,
}

/// The deterministic order in which a layout's stored entries are visited.
///
/// Declared rather than assumed, because numerical results depend on it: a float accumulation over
/// a different visiting order is a different number. It cannot be inferred from the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoredOrder {
    /// Rows ascending; within a row, entries in the order they are stored — **not** re-sorted by
    /// column. A CSR whose columns are unsorted within a row is valid and keeps that order.
    RowMajorRowsThenStored,
    /// The order entries were pushed. Never re-sorted.
    Insertion,
}

/// What validation does about repeated coordinates.
///
/// One variant, on purpose. This contract never merges duplicates: `CycleIncidence`'s mean counts
/// *incidences*, so a vertex listed twice in one hyperedge must contribute twice. A future
/// `Summed` variant would be a different, explicitly requested operation — not a default that
/// silently changes recorded numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Coalescing {
    /// Duplicates are preserved exactly as stored.
    Preserved,
}

/// The logical shape of a sparse tensor: its extents and how many entries are stored.
///
/// `rank()` is the extent count, so a rank-2 CSR and a rank-3 COO are distinguished by a real
/// length rather than by a tag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseShape {
    extents: Vec<usize>,
    nnz: usize,
}

impl SparseShape {
    /// # Preconditions
    /// `extents` is the logical extent of each axis, outermost first.
    #[must_use]
    pub fn new(extents: Vec<usize>, nnz: usize) -> Self {
        Self { extents, nnz }
    }

    /// Number of axes. Carried by the extent vector's own length.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    #[must_use]
    pub fn extents(&self) -> &[usize] {
        &self.extents
    }

    /// Stored entry count, including duplicates.
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.nnz
    }
}

/// A structural rule a sparse value broke. Each names the offending position and value, so a
/// caller can act on it rather than only report it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseViolation {
    /// `row_ptr` is not `rows + 1` long.
    RowPointerLength { expected: usize, actual: usize },
    /// The first offset is not zero.
    FirstOffsetNotZero { actual: usize },
    /// Offsets decrease at this index.
    OffsetNotMonotone {
        index: usize,
        previous: usize,
        actual: usize,
    },
    /// An offset exceeds the stored entry count.
    OffsetAboveNnz {
        index: usize,
        nnz: usize,
        actual: usize,
    },
    /// The terminal offset is not the stored entry count.
    TerminalOffsetNotNnz { nnz: usize, actual: usize },
    /// The index and value arrays disagree, or disagree with `nnz`.
    EntryArrayLength {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A coordinate lies outside its axis.
    CoordinateOutOfRange {
        axis: usize,
        extent: usize,
        index: usize,
        actual: usize,
    },
    /// The declared metadata cannot describe the stored entries.
    MetadataInconsistent { detail: &'static str },
    /// A metadata value is so large that deriving the structure from it would overflow.
    ///
    /// Reported rather than panicked: `usize::MAX` rows is a malformed declaration, and a caller
    /// deserves a named failure instead of an abort.
    ExtentOverflow { operand: &'static str, value: usize },
}

impl core::fmt::Display for SparseViolation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RowPointerLength { expected, actual } => {
                write!(
                    f,
                    "row_ptr must have rows+1 = {expected} entries, has {actual}"
                )
            }
            Self::FirstOffsetNotZero { actual } => {
                write!(f, "the first row offset must be 0, is {actual}")
            }
            Self::OffsetNotMonotone {
                index,
                previous,
                actual,
            } => write!(
                f,
                "row offset {index} decreases from {previous} to {actual}"
            ),
            Self::OffsetAboveNnz { index, nnz, actual } => {
                write!(f, "row offset {index} is {actual}, above nnz {nnz}")
            }
            Self::TerminalOffsetNotNnz { nnz, actual } => {
                write!(f, "the terminal row offset must be nnz {nnz}, is {actual}")
            }
            Self::EntryArrayLength {
                operand,
                expected,
                actual,
            } => {
                write!(f, "{operand} must have {expected} entries, has {actual}")
            }
            Self::CoordinateOutOfRange {
                axis,
                extent,
                index,
                actual,
            } => write!(
                f,
                "entry {index} has coordinate {actual} on axis {axis} of extent {extent}"
            ),
            Self::MetadataInconsistent { detail } => write!(f, "inconsistent metadata: {detail}"),
            Self::ExtentOverflow { operand, value } => {
                write!(f, "{operand} overflows usize at {value}")
            }
        }
    }
}

impl std::error::Error for SparseViolation {}

/// A read-only, **lifetime-bound** view of one sparse value that passed validation.
///
/// # Why this is a view and not a detachable receipt
///
/// An earlier draft returned a `SparseReceipt` carrying only layout, shape and nnz. That was
/// unsound: two tensors with the *same* metadata and *different* structure — one valid, one
/// corrupt — produced interchangeable proofs, so a consumer taking `(tensor, receipt)` could be
/// handed `B` together with `A`'s receipt and would accept it. The metadata gate cannot see the
/// difference, because the difference is not in the metadata.
///
/// Binding the proof to the value closes it in the type system rather than by discipline:
/// `ValidatedSparse` holds an immutable borrow of **the tensor it validated**, its fields are
/// private, and the only constructor is `bind`, scoped to this module tree. A consumer that takes
/// a `ValidatedSparse` therefore cannot be given a mismatched pair — there is no pair to
/// mismatch, and the borrow keeps the value alive and unmodified for as long as the proof exists.
///
/// The metadata stays observable through the accessors below; it is simply no longer separable
/// from the thing it describes.
#[derive(Debug)]
pub struct ValidatedSparse<'a, T> {
    tensor: &'a T,
    layout: SparseLayout,
    shape: SparseShape,
    stored_order: StoredOrder,
    coalescing: Coalescing,
}

impl<'a, T> ValidatedSparse<'a, T> {
    /// Visible only inside `tensor::representations` — the module tree that owns the layouts.
    ///
    /// Not `pub`: a validated view constructible from outside would prove nothing. The two
    /// `validate` implementations live in sibling modules beside their layout types, so this is the
    /// smallest scope that admits exactly the layout owners and nothing else.
    pub(in crate::representations) fn bind(
        tensor: &'a T,
        layout: SparseLayout,
        shape: SparseShape,
        stored_order: StoredOrder,
        coalescing: Coalescing,
    ) -> Self {
        Self {
            tensor,
            layout,
            shape,
            stored_order,
            coalescing,
        }
    }

    /// The validated value itself. The only way to reach it through this proof.
    ///
    /// # The replay this closes
    ///
    /// ```compile_fail
    /// use hymeko_tensor::representations::sparse::{SparseTensorContract, ValidatedSparse};
    /// use hymeko_tensor::representations::tensor_csr::TensorCsr;
    ///
    /// let a = TensorCsr::<f32> {
    ///     num_rows: 1, num_cols: 1,
    ///     row_ptr: vec![0, 1], col_ind: vec![0], val: vec![1.0],
    /// };
    /// let proof = a.validate().unwrap();
    /// // A proof cannot be built by hand: `bind` is not visible here.
    /// let forged = ValidatedSparse::bind(&a, proof.layout(), proof.shape().clone(),
    ///                                    proof.stored_order(), proof.coalescing());
    /// ```
    ///
    /// And a proof cannot outlive the value it borrows, so it cannot be carried to another one:
    ///
    /// ```compile_fail
    /// use hymeko_tensor::representations::sparse::SparseTensorContract;
    /// use hymeko_tensor::representations::tensor_csr::TensorCsr;
    ///
    /// let proof = {
    ///     let a = TensorCsr::<f32> {
    ///         num_rows: 1, num_cols: 1,
    ///         row_ptr: vec![0, 1], col_ind: vec![0], val: vec![1.0],
    ///     };
    ///     a.validate().unwrap()
    /// };
    /// let _ = proof.tensor();
    /// ```
    #[must_use]
    pub const fn tensor(&self) -> &'a T {
        self.tensor
    }

    #[must_use]
    pub const fn layout(&self) -> SparseLayout {
        self.layout
    }

    #[must_use]
    pub const fn shape(&self) -> &SparseShape {
        &self.shape
    }

    #[must_use]
    pub const fn stored_order(&self) -> StoredOrder {
        self.stored_order
    }

    #[must_use]
    pub const fn coalescing(&self) -> Coalescing {
        self.coalescing
    }
}

/// The structural facts a CSR yields once checked. Returned by the canonical validator so a caller
/// that has only borrowed parts — no owned `TensorCsr` — still gets the same answer from the same
/// implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsrStructure {
    /// Stored entry count, duplicates included.
    pub nnz: usize,
}

/// **The** CSR structural validator. One implementation, for owned and borrowed callers alike.
///
/// [`crate::representations::tensor_csr::TensorCsr`]'s `validate` delegates here, and so
/// does the WGPU backend's CSR upload — which is what keeps a second, drifting copy of these rules
/// from existing on the device side. It borrows; it copies nothing and allocates nothing.
///
/// # Postconditions
/// On `Ok`: `row_ptr` is `num_rows + 1` long, starts at 0, is monotone non-decreasing, never
/// exceeds nnz, ends exactly at nnz; `col_ind` and the values are nnz long; every column is
/// `< num_cols`. Duplicate entries are **preserved** — nothing here coalesces, sorts or mutates.
///
/// # Errors
/// The first [`SparseViolation`] found, naming the offending position.
pub fn validate_csr_parts(
    num_rows: usize,
    num_cols: usize,
    row_ptr: &[usize],
    col_ind: &[usize],
    values_len: usize,
) -> Result<CsrStructure, SparseViolation> {
    // Total, not panicking: a metadata value near `usize::MAX` is a validation failure with a name,
    // not an arithmetic overflow that aborts the process.
    let expected_len = num_rows
        .checked_add(1)
        .ok_or(SparseViolation::ExtentOverflow {
            operand: "csr row count + 1",
            value: num_rows,
        })?;
    if row_ptr.len() != expected_len {
        return Err(SparseViolation::RowPointerLength {
            expected: expected_len,
            actual: row_ptr.len(),
        });
    }
    let nnz = col_ind.len();
    // `row_ptr` is non-empty here: `expected_len >= 1`.
    if row_ptr[0] != 0 {
        return Err(SparseViolation::FirstOffsetNotZero { actual: row_ptr[0] });
    }
    for (index, pair) in row_ptr.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(SparseViolation::OffsetNotMonotone {
                index: index + 1,
                previous: pair[0],
                actual: pair[1],
            });
        }
        if pair[1] > nnz {
            return Err(SparseViolation::OffsetAboveNnz {
                index: index + 1,
                nnz,
                actual: pair[1],
            });
        }
    }
    let terminal = row_ptr[num_rows];
    if terminal != nnz {
        return Err(SparseViolation::TerminalOffsetNotNnz {
            nnz,
            actual: terminal,
        });
    }
    if values_len != nnz {
        return Err(SparseViolation::EntryArrayLength {
            operand: "csr values",
            expected: nnz,
            actual: values_len,
        });
    }
    if let Some((index, &column)) = col_ind.iter().enumerate().find(|(_, c)| **c >= num_cols) {
        return Err(SparseViolation::CoordinateOutOfRange {
            axis: 1,
            extent: num_cols,
            index,
            actual: column,
        });
    }
    Ok(CsrStructure { nnz })
}

/// What every sparse layout must be able to say about itself.
///
/// # Invariants every implementation upholds
///
/// * `validate` takes `&self` and cannot reorder, coalesce or otherwise modify anything.
/// * A repeated coordinate or incidence survives validation.
/// * `layout()` and `shape().rank()` are independent facts; neither implies the other.
pub trait SparseTensorContract {
    /// The stored scalar.
    type Scalar: Real;

    /// Which physical layout this is.
    fn layout(&self) -> SparseLayout;

    /// Logical extents and stored entry count.
    fn shape(&self) -> SparseShape;

    /// The deterministic order stored entries are visited in.
    fn stored_order(&self) -> StoredOrder;

    /// What this layout does about duplicates. Always [`Coalescing::Preserved`] here.
    fn coalescing(&self) -> Coalescing {
        Coalescing::Preserved
    }

    /// Check the structure and return a proof **bound to this value**.
    ///
    /// # Postconditions
    /// On `Ok`, every structural rule for this layout holds, and **nothing about `self` changed** —
    /// not one bit, not one position. The returned view borrows `self`, so the proof cannot outlive
    /// the value or be transferred to another.
    ///
    /// # Errors
    /// The first [`SparseViolation`] found, naming the offending position.
    fn validate(&self) -> Result<ValidatedSparse<'_, Self>, SparseViolation>
    where
        Self: Sized;
}
