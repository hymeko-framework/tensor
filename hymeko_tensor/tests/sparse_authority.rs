//! Gates for the sparse **semantic** contract.
//!
//! Two things every control here proves, in this order: that the mutation actually applied, and
//! then that the gate refused it. A control that cannot show its fault landed proves nothing.
//!
//! The positive controls matter as much as the refusals. Validation must **preserve** duplicates
//! and order; a validator that quietly coalesced or sorted would pass every refusal test above and
//! still be wrong.

use hymeko_tensor::representations::sparse::{
    validate_csr_parts, Coalescing, SparseLayout, SparseTensorContract, SparseViolation,
    StoredOrder,
};
use hymeko_tensor::representations::tensor_coo::TensorCoo;
use hymeko_tensor::representations::tensor_csr::TensorCsr;

/// A well-formed 3x3 CSR with four entries, one row empty.
fn csr() -> TensorCsr<f32> {
    TensorCsr {
        num_rows: 3,
        num_cols: 3,
        row_ptr: vec![0, 2, 2, 4],
        col_ind: vec![0, 2, 1, 2],
        val: vec![1.0, 2.0, 3.0, 4.0],
    }
}

/// A well-formed 2x3x3 COO with four entries.
fn coo() -> TensorCoo<f32> {
    let mut t = TensorCoo::new(2, 3, 3);
    t.push(0, 0, 1, 1.0);
    t.push(1, 2, 0, 2.0);
    t.push(0, 1, 1, 3.0);
    t.push(1, 0, 2, 4.0);
    t
}

// ── the contract reports what it should, and the two layouts stay distinguishable ───────────────

#[test]
fn each_layout_reports_its_own_identity_rank_and_order() {
    let csr = csr();
    assert_eq!(csr.layout(), SparseLayout::Csr);
    assert_eq!(csr.shape().rank(), 2, "CSR is rank 2");
    assert_eq!(csr.shape().extents(), &[3, 3]);
    assert_eq!(csr.shape().nnz(), 4);
    assert_eq!(csr.stored_order(), StoredOrder::RowMajorRowsThenStored);

    let coo = coo();
    assert_eq!(coo.layout(), SparseLayout::Coo);
    // The slice axis is carried, not flattened into a matrix.
    assert_eq!(coo.shape().rank(), 3, "COO is rank 3 over (k, i, j)");
    assert_eq!(coo.shape().extents(), &[2, 3, 3]);
    assert_eq!(coo.shape().nnz(), 4);
    assert_eq!(coo.stored_order(), StoredOrder::Insertion);

    // Rank and layout are independent facts: neither implies the other, and the two ranks differ,
    // so a single flattened representation could not have carried both.
    assert_ne!(csr.shape().rank(), coo.shape().rank());
    assert_ne!(csr.layout(), coo.layout());
}

#[test]
fn neither_layout_nor_order_is_inferable_from_shape_alone() {
    // Two DIFFERENT layouts can be built over extents that share a prefix; the shape alone does not
    // say which, nor in what order entries are visited.
    let a = csr();
    let mut b = TensorCoo::<f32>::new(3, 3, 1);
    b.push(0, 0, 0, 1.0);
    assert_eq!(a.shape().extents()[0], b.shape().extents()[0]);
    assert_ne!(a.layout(), b.layout());
    assert_ne!(a.stored_order(), b.stored_order());
}

#[test]
fn the_same_logical_tensor_in_two_layouts_is_two_storage_identities() {
    // The same four values, stored two ways. The proofs differ in layout and order, which is
    // exactly the fact that lets a consumer refuse to treat them as interchangeable. Each `let`
    // binding is required, not stylistic: a proof borrows its tensor and cannot outlive it.
    let csr = csr();
    let coo = coo();
    let from_csr = csr.validate().expect("csr validates");
    let from_coo = coo.validate().expect("coo validates");
    assert_eq!(from_csr.shape().nnz(), from_coo.shape().nnz());
    assert_ne!(from_csr.layout(), from_coo.layout());
    assert_ne!(from_csr.stored_order(), from_coo.stored_order());
    // And they point at different values: the proofs are not interchangeable even in principle.
    assert!(!std::ptr::eq(
        std::ptr::from_ref(from_csr.tensor()).cast::<u8>(),
        std::ptr::from_ref(from_coo.tensor()).cast::<u8>()
    ));
}

// ── CSR gates ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn csr_structural_faults_are_each_refused_after_being_shown_to_apply() {
    let base = csr();
    assert!(base.validate().is_ok(), "the base fixture must validate");

    // (mutation, what it should be refused as)
    let mut wrong_len = csr();
    wrong_len.row_ptr = vec![0, 2, 4];
    assert_ne!(
        wrong_len.row_ptr, base.row_ptr,
        "length mutation did not apply"
    );
    assert!(matches!(
        wrong_len.validate(),
        Err(SparseViolation::RowPointerLength {
            expected: 4,
            actual: 3
        })
    ));

    let mut first = csr();
    first.row_ptr[0] = 1;
    assert_ne!(
        first.row_ptr, base.row_ptr,
        "first-offset mutation did not apply"
    );
    assert!(matches!(
        first.validate(),
        Err(SparseViolation::FirstOffsetNotZero { actual: 1 })
    ));

    let mut decreasing = csr();
    decreasing.row_ptr = vec![0, 3, 2, 4];
    assert_ne!(
        decreasing.row_ptr, base.row_ptr,
        "monotonicity mutation did not apply"
    );
    assert!(matches!(
        decreasing.validate(),
        Err(SparseViolation::OffsetNotMonotone {
            index: 2,
            previous: 3,
            actual: 2
        })
    ));

    let mut above = csr();
    above.row_ptr = vec![0, 2, 9, 4];
    assert_ne!(
        above.row_ptr, base.row_ptr,
        "above-nnz mutation did not apply"
    );
    assert!(matches!(
        above.validate(),
        Err(SparseViolation::OffsetAboveNnz {
            index: 2,
            nnz: 4,
            actual: 9
        })
    ));

    let mut terminal = csr();
    terminal.row_ptr = vec![0, 2, 2, 3];
    assert_ne!(
        terminal.row_ptr, base.row_ptr,
        "terminal mutation did not apply"
    );
    assert!(matches!(
        terminal.validate(),
        Err(SparseViolation::TerminalOffsetNotNnz { nnz: 4, actual: 3 })
    ));

    let mut values = csr();
    values.val = vec![1.0, 2.0];
    assert_ne!(
        values.val.len(),
        base.val.len(),
        "value-length mutation did not apply"
    );
    assert!(matches!(
        values.validate(),
        Err(SparseViolation::EntryArrayLength {
            operand: "csr values",
            expected: 4,
            actual: 2
        })
    ));

    let mut column = csr();
    column.col_ind[3] = 7;
    assert_ne!(
        column.col_ind, base.col_ind,
        "column mutation did not apply"
    );
    assert!(matches!(
        column.validate(),
        Err(SparseViolation::CoordinateOutOfRange {
            axis: 1,
            extent: 3,
            index: 3,
            actual: 7
        })
    ));
}

#[test]
fn csr_validation_preserves_duplicates_and_changes_no_bit() {
    // A repeated (row, column) is legitimate: the mean's denominator counts incidences. Validation
    // must not merge them, and must not touch anything at all.
    let duplicated = TensorCsr::<f32> {
        num_rows: 2,
        num_cols: 2,
        row_ptr: vec![0, 3, 3],
        col_ind: vec![1, 1, 1],
        val: vec![0.25, 0.5, 0.75],
    };
    let before_cols = duplicated.col_ind.clone();
    let before_bits: Vec<u32> = duplicated.val.iter().map(|v| v.to_bits()).collect();

    let validated = duplicated.validate().expect("a duplicated CSR is valid");
    assert_eq!(validated.coalescing(), Coalescing::Preserved);
    assert_eq!(
        validated.shape().nnz(),
        3,
        "nnz counts stored entries, duplicates included"
    );

    assert_eq!(
        duplicated.col_ind, before_cols,
        "validation reordered the columns"
    );
    let after_bits: Vec<u32> = duplicated.val.iter().map(|v| v.to_bits()).collect();
    assert_eq!(after_bits, before_bits, "validation changed a value bit");
}

#[test]
fn csr_validation_does_not_sort_unsorted_columns_within_a_row() {
    // Descending columns inside a row are structurally valid. Sorting them would change a float
    // accumulation order, so the contract reports the stored order rather than imposing one.
    let unsorted = TensorCsr::<f32> {
        num_rows: 1,
        num_cols: 4,
        row_ptr: vec![0, 3],
        col_ind: vec![3, 1, 0],
        val: vec![1.0, 2.0, 3.0],
    };
    let before = unsorted.col_ind.clone();
    let validated = unsorted.validate().expect("unsorted columns are valid");
    assert_eq!(
        validated.stored_order(),
        StoredOrder::RowMajorRowsThenStored
    );
    assert_eq!(unsorted.col_ind, before, "validation sorted the columns");
}

// ── COO gates ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn coo_out_of_range_coordinates_are_refused_on_the_right_axis() {
    let base = coo();
    assert!(base.validate().is_ok(), "the base fixture must validate");

    for (axis, k, i, j) in [(0usize, 5usize, 0usize, 0usize), (1, 0, 9, 0), (2, 0, 0, 9)] {
        let mut broken = coo();
        broken.push(k, i, j, 1.0);
        assert_eq!(
            broken.len(),
            base.len() + 1,
            "axis {axis}: mutation did not apply"
        );
        match broken.validate() {
            Err(SparseViolation::CoordinateOutOfRange {
                axis: got, index, ..
            }) => {
                assert_eq!(got, axis, "refused on the wrong axis");
                assert_eq!(index, base.len(), "refused the wrong entry");
            }
            other => panic!("axis {axis}: expected a coordinate refusal, got {other:?}"),
        }
    }
}

#[test]
fn coo_metadata_that_cannot_describe_its_entries_is_refused() {
    let mut degenerate = TensorCoo::<f32>::new(0, 3, 3);
    degenerate
        .entries
        .push(hymeko_tensor::representations::tensor_coo::CooEntry {
            k: 0,
            i: 0,
            j: 0,
            v: 1.0,
        });
    assert_eq!(degenerate.len(), 1, "the metadata mutation did not apply");
    assert!(matches!(
        degenerate.validate(),
        Err(SparseViolation::MetadataInconsistent { .. })
    ));
}

#[test]
fn coo_validation_preserves_duplicate_coordinates_and_insertion_order() {
    let mut duplicated = TensorCoo::<f32>::new(1, 2, 2);
    duplicated.push(0, 1, 1, 0.25);
    duplicated.push(0, 1, 1, 0.5); // the same coordinate, deliberately
    duplicated.push(0, 0, 0, 0.75);
    let before: Vec<(usize, usize, usize, u32)> = duplicated
        .iter()
        .map(|e| (e.k, e.i, e.j, e.v.to_bits()))
        .collect();

    let validated = duplicated
        .validate()
        .expect("duplicate coordinates are valid");
    assert_eq!(validated.coalescing(), Coalescing::Preserved);
    assert_eq!(validated.shape().nnz(), 3, "the duplicate was merged away");
    assert_eq!(validated.stored_order(), StoredOrder::Insertion);

    let after: Vec<(usize, usize, usize, u32)> = duplicated
        .iter()
        .map(|e| (e.k, e.i, e.j, e.v.to_bits()))
        .collect();
    assert_eq!(
        after, before,
        "validation coalesced, sorted or rewrote entries"
    );
    // And the order really is insertion order, not sorted order — the third entry is the smallest.
    assert_eq!((after[2].1, after[2].2), (0, 0));
}

// ── the receipt is the only route, and it cannot be forged ──────────────────────────────────────

#[test]
fn a_receipt_carries_the_facts_a_device_adapter_needs_and_nothing_more() {
    let tensor = csr();
    let validated = tensor.validate().expect("validates");
    assert_eq!(validated.layout(), SparseLayout::Csr);
    assert_eq!(validated.shape().extents(), &[3, 3]);
    assert_eq!(validated.shape().nnz(), 4);
    assert_eq!(
        validated.stored_order(),
        StoredOrder::RowMajorRowsThenStored
    );
    assert_eq!(validated.coalescing(), Coalescing::Preserved);
}

#[test]
fn the_proof_constructor_is_not_reachable_from_outside_the_representations_tree() {
    // `bind` is scoped to `crate::representations`, so no other crate and no other part of
    // core can mint a proof. If that ever widened to `pub`, a consumer could fabricate one and the
    // whole no-second-validator argument would collapse silently. This fails first instead.
    let source = include_str!("../src/representations/sparse.rs");
    assert!(
        source.contains("pub(in crate::representations) fn bind("),
        "ValidatedSparse::bind is no longer scoped to the representations tree"
    );
    assert!(
        !source.contains("pub fn bind("),
        "ValidatedSparse::bind became public; proofs could then be forged"
    );
}

// ── the replay the metadata-only receipt allowed, and no longer does ─────────────────────────────

#[test]
fn a_proof_cannot_be_replayed_onto_a_different_tensor_with_the_same_metadata() {
    // A is valid. B shares its extents and its nnz exactly, and is structurally corrupt: B's second
    // row offset goes backwards. This is the pair that the earlier metadata-only receipt could not
    // tell apart.
    let a = csr();
    let mut b = csr();
    b.row_ptr = vec![0, 4, 2, 4];
    assert_ne!(b.row_ptr, a.row_ptr, "the corruption did not apply");

    // The metadata a detachable receipt would have carried is IDENTICAL for both.
    assert_eq!(a.layout(), b.layout());
    assert_eq!(
        a.shape(),
        b.shape(),
        "extents and nnz must match for this control to bite"
    );
    assert_eq!(a.stored_order(), b.stored_order());

    // So a metadata-only gate — the exact check the old `upload_validated_csr` performed against a
    // separately-passed receipt — would have ACCEPTED (B, receipt_of_A). Demonstrated, not asserted:
    let a_validated = a.validate().expect("A is valid");
    let metadata_gate_would_accept = a_validated.layout() == b.layout()
        && a_validated.shape() == &b.shape()
        && a_validated.shape().nnz() == b.shape().nnz();
    assert!(
        metadata_gate_would_accept,
        "the control is not discriminating: a metadata gate would already have rejected this pair"
    );

    // B itself is genuinely invalid, so accepting it would have been a real fault, not a harmless one.
    assert!(matches!(
        b.validate(),
        Err(SparseViolation::OffsetNotMonotone { .. })
    ));

    // And the replay is now UNEXPRESSIBLE: a proof borrows the tensor it validated, so
    // `a_validated.tensor()` is A and can never be B. There is no second argument to substitute.
    assert!(
        std::ptr::eq(a_validated.tensor(), &a),
        "the proof does not point at the tensor it validated"
    );
    assert!(
        !std::ptr::eq(a_validated.tensor(), &b),
        "the proof could be read as pointing at B"
    );
}

// ── overflow ────────────────────────────────────────────────────────────────────────────────────

#[test]
fn a_row_count_at_the_usize_ceiling_is_a_typed_violation_not_a_panic() {
    // `num_rows + 1` is the row-pointer length. At `usize::MAX` that addition overflows; in a
    // release build it would wrap silently and in debug it would abort. Neither is an acceptable
    // answer to malformed metadata, so it is total and named.
    let violation = validate_csr_parts(usize::MAX, 1, &[0], &[0], 1)
        .expect_err("usize::MAX rows must not be accepted");
    assert!(
        matches!(
            violation,
            SparseViolation::ExtentOverflow { operand: "csr row count + 1", value } if value == usize::MAX
        ),
        "expected a typed overflow violation, got {violation:?}"
    );
    // And it renders, so a caller can report it.
    assert!(format!("{violation}").contains("overflows usize"));
}
