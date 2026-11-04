//! The CSR constructions: from rows in stored order, symmetric from a diagonal and an upper
//! triangle, and the transpose -- each against a dense reference and the sparse contract.

use hymeko_tensor::representations::sparse::SparseTensorContract;
use hymeko_tensor::representations::tensor_csr::{CsrRowWriter, TensorCsr};

/// The dense matrix of a CSR, row-major.
fn dense(m: &TensorCsr<f64>) -> Vec<f64> {
    let mut out = vec![0.0; m.num_rows * m.num_cols];
    for r in 0..m.num_rows {
        for k in m.row_ptr[r]..m.row_ptr[r + 1] {
            out[r * m.num_cols + m.col_ind[k]] += m.val[k];
        }
    }
    out
}

fn rows() -> Vec<Vec<(usize, f64)>> {
    vec![
        vec![(2, 1.0), (0, 2.0)],
        vec![],
        vec![(1, 3.0), (3, 4.0), (1, 5.0)],
    ]
}

/// Rows keep their stored order, an empty row stays empty, duplicates stay separate, and the
/// result passes the sparse contract.
#[test]
fn from_rows_keeps_the_stored_order() {
    let m = TensorCsr::from_rows(rows(), 4);
    assert_eq!((m.num_rows, m.num_cols), (3, 4));
    assert_eq!(m.row_ptr, vec![0, 2, 2, 5]);
    assert_eq!(m.col_ind, vec![2, 0, 1, 3, 1]);
    assert_eq!(m.val, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(m.validate().is_ok());
    let none = TensorCsr::<f64>::from_rows(Vec::<Vec<(usize, f64)>>::new(), 5);
    assert_eq!((none.num_rows, none.row_ptr.clone()), (0, vec![0]));
}

#[test]
#[should_panic(expected = "column 4 of a matrix with 4 columns")]
fn from_rows_refuses_a_column_out_of_range() {
    let _ = TensorCsr::from_rows([[(4usize, 1.0f64)]], 4);
}

/// `(Aᵀ)_{cr} = A_{rc}`; each transposed row by ascending source row, a row's duplicates in their
/// stored order; transposing twice restores a matrix whose rows were sorted.
#[test]
fn transpose_is_the_dense_transpose_and_stable() {
    let m = TensorCsr::from_rows(rows(), 4);
    let t = m.transpose();
    assert_eq!((t.num_rows, t.num_cols), (4, 3));
    let (a, at) = (dense(&m), dense(&t));
    for r in 0..3 {
        for c in 0..4 {
            assert_eq!(a[r * 4 + c], at[c * 3 + r]);
        }
    }
    assert_eq!(t.row_ptr, vec![0, 1, 3, 4, 5]);
    assert_eq!(t.col_ind, vec![0, 2, 2, 0, 2]);
    assert_eq!(t.val, vec![2.0, 3.0, 5.0, 1.0, 4.0]);
    assert!(t.validate().is_ok());
    let sorted = TensorCsr::from_rows(
        vec![
            vec![(0, 1.0), (2, 2.0)],
            vec![(1, 3.0)],
            vec![(0, 4.0), (1, 5.0)],
        ],
        3,
    );
    let back = sorted.transpose().transpose();
    assert_eq!(
        (back.row_ptr, back.col_ind, back.val),
        (sorted.row_ptr, sorted.col_ind, sorted.val)
    );
}

/// The symmetric matrix of a diagonal and an upper triangle: equal to its transpose, the dense
/// expansion, columns ascending in every row.
#[test]
fn from_symmetric_expands_both_triangles() {
    let m = TensorCsr::from_symmetric(&[4.0, 5.0, 6.0], &[(0, 2, 1.0), (1, 2, -2.0)]);
    assert_eq!(
        dense(&m),
        vec![4.0, 0.0, 1.0, 0.0, 5.0, -2.0, 1.0, -2.0, 6.0]
    );
    for r in 0..m.num_rows {
        let row = &m.col_ind[m.row_ptr[r]..m.row_ptr[r + 1]];
        assert!(row.windows(2).all(|w| w[0] < w[1]), "row {r}: {row:?}");
    }
    let t = m.transpose();
    assert_eq!((t.row_ptr, t.col_ind, t.val), (m.row_ptr, m.col_ind, m.val));
    let x = [1.0, -1.0, 2.0];
    assert_eq!(
        TensorCsr::from_symmetric(&[4.0, 5.0, 6.0], &[(0, 2, 1.0), (1, 2, -2.0)]).spmv(&x),
        vec![6.0, -9.0, 15.0]
    );
}

/// The writer is `from_rows` without a buffer per row: the same matrix entry for entry; an
/// unclosed row is dropped; slices copy in.
#[test]
fn the_row_writer_writes_what_from_rows_builds() {
    let mut writer = CsrRowWriter::with_capacity(4, 3, 5);
    for row in rows() {
        for (c, v) in row {
            writer.push(c, v);
        }
        writer.end_row();
    }
    writer.push(0, 9.0);
    let written = writer.finish();
    let built = TensorCsr::from_rows(rows(), 4);
    assert_eq!(
        (
            written.num_rows,
            written.row_ptr,
            written.col_ind,
            written.val
        ),
        (built.num_rows, built.row_ptr, built.col_ind, built.val)
    );
    let mut slices = CsrRowWriter::with_capacity(3, 1, 2);
    slices.extend_from_slices(&[2, 0], &[1.5f64, -1.0]);
    slices.end_row();
    let m = slices.finish();
    assert_eq!(
        (m.row_ptr, m.col_ind, m.val),
        (vec![0, 2], vec![2, 0], vec![1.5, -1.0])
    );
}

/// `A B` against the dense product, columns ascending in every row, a cancellation kept as an
/// explicit zero; `Pᵀ A P` of a symmetric `A` is symmetric.
#[test]
fn matmul_is_the_dense_product() {
    let a = TensorCsr::from_rows(rows(), 4);
    let b = TensorCsr::from_rows(
        vec![
            vec![(1, 1.0), (0, 2.0)],
            vec![(0, -1.0)],
            vec![(2, 3.0)],
            vec![(1, 0.5), (2, -1.0)],
        ],
        3,
    );
    let p = a.matmul(&b);
    assert_eq!((p.num_rows, p.num_cols), (3, 3));
    let (da, db, dp) = (dense(&a), dense(&b), dense(&p));
    for r in 0..3 {
        for c in 0..3 {
            let want: f64 = (0..4).map(|k| da[r * 4 + k] * db[k * 3 + c]).sum();
            assert_eq!(dp[r * 3 + c], want, "({r}, {c})");
        }
        let row = &p.col_ind[p.row_ptr[r]..p.row_ptr[r + 1]];
        assert!(row.windows(2).all(|w| w[0] < w[1]));
    }
    let cancel = TensorCsr::from_rows(vec![vec![(0, 1.0), (1, 1.0)]], 2).matmul(
        &TensorCsr::from_rows(vec![vec![(0, 1.0)], vec![(0, -1.0)]], 1),
    );
    assert_eq!((cancel.col_ind, cancel.val), (vec![0], vec![0.0]));
    let s = TensorCsr::from_symmetric(&[4.0, 5.0, 6.0], &[(0, 2, 1.0), (1, 2, -2.0)]);
    let prolong = TensorCsr::from_rows(
        vec![vec![(0, 1.0)], vec![(0, 0.5), (1, 0.5)], vec![(1, 1.0)]],
        2,
    );
    let galerkin = prolong.transpose().matmul(&s).matmul(&prolong);
    assert_eq!(dense(&galerkin), dense(&galerkin.transpose()));
}

#[test]
fn diagonal_sums_the_diagonal_entries() {
    let m = TensorCsr::from_rows(
        vec![vec![(1, 2.0), (0, 1.0), (0, 0.5)], vec![], vec![(2, 3.0)]],
        3,
    );
    assert_eq!(m.diagonal(), vec![1.5, 0.0, 3.0]);
    assert_eq!(
        TensorCsr::<f64>::from_rows(vec![vec![(0, 7.0)]], 3).diagonal(),
        vec![7.0]
    );
}
