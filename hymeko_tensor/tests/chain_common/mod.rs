//! Shared workload client; numeric CSR generation remains in sparse_workload.
use hymeko_tensor::{
    contraction::PlanningBudget,
    representations::tensor_csr::TensorCsr,
    sparse_workload::{DegreeShape, SparseSpec},
};
pub struct ChainCase {
    pub scale: usize,
    pub degree_divisor: usize,
}
impl ChainCase {
    pub fn build(&self) -> Vec<TensorCsr<f64>> {
        let dims = [30, 35, 15, 5, 10, 20, 25].map(|n| n * self.scale);
        dims.windows(2)
            .enumerate()
            .map(|(i, d)| {
                let m = SparseSpec {
                    rows: d[0],
                    cols: d[1],
                    avg_nnz_per_row: (d[1] / self.degree_divisor).max(1),
                    shape: DegreeShape::Uniform,
                    seed: 0x19c0_ffee + i as u64,
                }
                .build();
                TensorCsr {
                    num_rows: m.num_rows,
                    num_cols: m.num_cols,
                    row_ptr: m.row_ptr,
                    col_ind: m.col_ind,
                    val: m.val.into_iter().map(f64::from).collect(),
                }
            })
            .collect()
    }
    pub fn budget(states: usize) -> PlanningBudget {
        PlanningBudget {
            max_states: states,
            max_intermediate_elements: 10_000_000,
            max_intermediate_rank: 2,
        }
    }
    pub fn fingerprint(matrices: &[TensorCsr<f64>]) -> String {
        let mut hash = blake3::Hasher::new();
        for m in matrices {
            for n in [m.num_rows, m.num_cols]
                .into_iter()
                .chain(m.row_ptr.iter().copied())
                .chain(m.col_ind.iter().copied())
            {
                hash.update(&(n as u64).to_le_bytes());
            }
            for value in &m.val {
                hash.update(&value.to_bits().to_le_bytes());
            }
        }
        hash.finalize().to_hex().to_string()
    }
}
