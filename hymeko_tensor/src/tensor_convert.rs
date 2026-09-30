use crate::common::Real;
use crate::representations::tensor_coo::TensorCoo;
use nalgebra::{DMatrix, Scalar};
use nalgebra_sparse::{coo::CooMatrix, CsrMatrix};
use num_traits::Zero;

pub struct JaxBcoo<F: Real> {
    pub shape: [usize; 3],      // [K, I, J]
    pub indices: Vec<[i32; 3]>, // (nnz, 3)
    pub data: Vec<F>,           // (nnz,)
}

pub fn to_nalgebra_dense_slice<F>(coo: &TensorCoo<F>, k_sel: usize) -> DMatrix<F>
where
    F: Real + Scalar + Zero,
{
    assert!(k_sel < coo.num_slices);
    let mut m = DMatrix::<F>::zeros(coo.dim_i, coo.dim_j);

    for e in coo.iter() {
        if e.k != k_sel {
            continue;
        }
        m[(e.i, e.j)] += e.v;
    }
    m
}

pub fn to_nalgebra_csr_slice<F>(coo: &TensorCoo<F>, k_sel: usize) -> CsrMatrix<F>
where
    F: Real + Scalar + Zero,
{
    assert!(k_sel < coo.num_slices);

    let mut m = CooMatrix::new(coo.dim_i, coo.dim_j);

    for e in coo.iter() {
        if e.k != k_sel {
            continue;
        }
        m.push(e.i, e.j, e.v);
    }
    CsrMatrix::from(&m) // will coalesce duplicates
}

pub fn to_jax_bcoo<F>(coo: &TensorCoo<F>) -> JaxBcoo<F>
where
    F: Real + Scalar + Zero,
{
    let mut indices = Vec::with_capacity(coo.len());
    let mut data = Vec::with_capacity(coo.len());

    for e in coo.iter() {
        indices.push([e.k as i32, e.i as i32, e.j as i32]);
        data.push(e.v);
    }

    JaxBcoo {
        shape: [coo.num_slices, coo.dim_i, coo.dim_j],
        indices,
        data,
    }
}
