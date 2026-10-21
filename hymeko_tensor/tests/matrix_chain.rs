#![cfg(feature = "sparse")]
mod chain_common;
use chain_common::ChainCase;
use hymeko_tensor::{
    contraction::{MatrixChain, MatrixChainError},
    representations::tensor_csr::TensorCsr,
};

#[test]
fn real_sparse_chains_reassociate_with_checked_error_and_reuse_buffers() {
    for (scale, degree_divisor) in [(1, 1), (2, 1), (2, 8)] {
        let case = ChainCase {
            scale,
            degree_divisor,
        };
        let matrices = case.build();
        assert_eq!(
            ChainCase::fingerprint(&matrices),
            ChainCase::fingerprint(&case.build())
        );
        let chain = MatrixChain::new(&matrices).unwrap();
        let greedy = chain.plan(ChainCase::budget(0)).unwrap().plan.unwrap();
        let result = chain.plan(ChainCase::budget(200_000)).unwrap();
        assert!(result.complete);
        let optimal = result.plan.unwrap();
        assert_eq!(optimal.work(), 15125 * (scale as u128).pow(3));
        let mut a = chain.prepare(&greedy).unwrap();
        let mut b = chain.prepare(&optimal).unwrap();
        let reference = a.execute().to_vec();
        let output = b.execute();
        let norm = reference.iter().copied().fold(0.0, f64::max).max(1.0);
        assert!(reference
            .iter()
            .zip(output)
            .all(|(&x, &y)| (x - y).abs() <= norm * 1e-12));
        let pointer = b.execute().as_ptr();
        assert_eq!(pointer, b.execute().as_ptr());
        assert!(b.retained_bytes() > 0);
        let repeated = b.execute();
        assert!(repeated
            .iter()
            .zip(&reference)
            .all(|(a, b)| (*a - *b).abs() <= norm * 1e-12));
    }
}

struct DenseFixture;
impl DenseFixture {
    fn matrix(rows: usize, cols: usize, value: f64) -> TensorCsr<f64> {
        TensorCsr {
            num_rows: rows,
            num_cols: cols,
            row_ptr: (0..=rows).map(|r| r * cols).collect(),
            col_ind: (0..rows).flat_map(|_| 0..cols).collect(),
            val: vec![value; rows * cols],
        }
    }
}
#[test]
fn closed_form_signed_constant_chain_and_singleton() {
    let matrices = vec![
        DenseFixture::matrix(3, 5, -1.0),
        DenseFixture::matrix(5, 2, 2.0),
        DenseFixture::matrix(2, 4, 3.0),
    ];
    let chain = MatrixChain::new(&matrices).unwrap();
    for states in [0, 1000] {
        let plan = chain.plan(ChainCase::budget(states)).unwrap().plan.unwrap();
        assert_eq!(chain.prepare(&plan).unwrap().execute(), &[-60.0; 12]);
    }
    let matrices = vec![DenseFixture::matrix(2, 3, -2.0)];
    let chain = MatrixChain::new(&matrices).unwrap();
    let plan = chain.plan(ChainCase::budget(0)).unwrap().plan.unwrap();
    assert_eq!(chain.prepare(&plan).unwrap().execute(), &[-2.0; 6]);
}
#[test]
fn shape_invalid_sparse_foreign_plan_and_zero_extents() {
    assert!(matches!(
        MatrixChain::<f64>::new(&[]),
        Err(MatrixChainError::Shape)
    ));
    let mut invalid = DenseFixture::matrix(2, 2, 1.0);
    invalid.col_ind[0] = 2;
    assert!(matches!(
        MatrixChain::new(&[invalid]),
        Err(MatrixChainError::Sparse(_))
    ));
    assert!(matches!(
        MatrixChain::new(&[
            DenseFixture::matrix(2, 3, 1.0),
            DenseFixture::matrix(2, 2, 1.0)
        ]),
        Err(MatrixChainError::Shape)
    ));
    let a = vec![
        DenseFixture::matrix(2, 0, 1.0),
        DenseFixture::matrix(0, 3, 1.0),
    ];
    let chain = MatrixChain::new(&a).unwrap();
    let plan = chain.plan(ChainCase::budget(100)).unwrap().plan.unwrap();
    assert_eq!(chain.prepare(&plan).unwrap().execute(), &[0.0; 6]);
    let b = vec![DenseFixture::matrix(1, 1, 2.0)];
    let other = MatrixChain::new(&b).unwrap();
    assert!(matches!(other.prepare(&plan), Err(MatrixChainError::Shape)));
}

#[test]
fn noncommuting_signed_matrices_and_duplicate_right_entries_match_closed_form() {
    let matrices: Vec<_> = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, -1.0, 0.0],
        vec![2.0, 0.0, 1.0, 3.0],
    ]
    .into_iter()
    .map(|val| {
        let mut matrix = DenseFixture::matrix(2, 2, 0.0);
        matrix.val = val;
        matrix
    })
    .collect();
    let chain = MatrixChain::new(&matrices).unwrap();
    for states in [0, 1000] {
        let plan = chain.plan(ChainCase::budget(states)).unwrap().plan.unwrap();
        assert_eq!(
            chain.prepare(&plan).unwrap().execute(),
            &[-3.0, 3.0, -5.0, 9.0]
        );
    }
    let matrices = vec![
        TensorCsr {
            num_rows: 1,
            num_cols: 2,
            row_ptr: vec![0, 2],
            col_ind: vec![0, 1],
            val: vec![1.0, 2.0],
        },
        TensorCsr {
            num_rows: 2,
            num_cols: 1,
            row_ptr: vec![0, 2, 3],
            col_ind: vec![0, 0, 0],
            val: vec![1.0, 2.0, 4.0],
        },
    ];
    let chain = MatrixChain::new(&matrices).unwrap();
    let plan = chain.plan(ChainCase::budget(100)).unwrap().plan.unwrap();
    assert_eq!(chain.prepare(&plan).unwrap().execute(), &[11.0]);
}
