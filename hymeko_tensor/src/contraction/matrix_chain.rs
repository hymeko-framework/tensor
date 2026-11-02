//! Rank-two plan compilation onto the existing CSR SpMM kernel.
use super::*;
use crate::{
    common::Real,
    representations::{
        sparse::{SparseTensorContract, SparseViolation},
        tensor_csr::TensorCsr,
    },
};

/// Matrix binding/compilation failure; no partially prepared executor escapes.
#[derive(Debug)]
pub enum MatrixChainError {
    /// Invalid CSR structure.
    Sparse(SparseViolation),
    /// Invalid contraction metadata.
    Planning(PlanningError),
    /// Empty chain, incompatible matrices, or plan from a different shape.
    Shape,
    /// Plan requires a non-matrix contraction; use a rank-two planning budget.
    UnsupportedStep,
    /// Buffer extent multiplication overflowed.
    Overflow,
}
impl std::fmt::Display for MatrixChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "matrix chain: {self:?}")
    }
}
impl std::error::Error for MatrixChainError {}

/// Borrowed sparse matrix chain with a canonical index-planning problem.
pub struct MatrixChain<'a, F: Real> {
    inputs: &'a [TensorCsr<F>],
    problem: ContractionProblem,
}
impl<'a, F: Real> MatrixChain<'a, F> {
    /// Validate all CSR inputs and adjacent dimensions once.
    pub fn new(inputs: &'a [TensorCsr<F>]) -> Result<Self, MatrixChainError> {
        let Some(first) = inputs.first() else {
            return Err(MatrixChainError::Shape);
        };
        let mut extents = vec![first.num_rows];
        for matrix in inputs {
            matrix.validate().map_err(MatrixChainError::Sparse)?;
            if extents.last() != Some(&matrix.num_rows) {
                return Err(MatrixChainError::Shape);
            }
            extents.push(matrix.num_cols);
        }
        let factors = (0..inputs.len()).map(|i| vec![i, i + 1]).collect();
        let problem = ContractionProblem::new(extents, factors, vec![0, inputs.len()])
            .map_err(MatrixChainError::Planning)?;
        Ok(Self { inputs, problem })
    }
    /// Plan with this backend's rank-two constraint; uses the shared planner.
    pub fn plan(&self, mut budget: PlanningBudget) -> Result<PlanningOutcome<'_>, PlanningError> {
        budget.max_intermediate_rank = budget.max_intermediate_rank.min(2);
        ContractionPlanner::plan(&self.problem, budget)
    }
    /// Allocate reusable numeric buffers and compile the generic plan visitor.
    /// Intermediates are dense-valued CSR matrices; input sparsity is preserved
    /// when an input is a left operand. This is explicit stored execution.
    pub fn prepare<'b>(
        &'b self,
        plan: &ContractionPlan<'_>,
    ) -> Result<PreparedMatrixChain<'b, 'a, F>, MatrixChainError> {
        if plan.problem != &self.problem {
            return Err(MatrixChainError::Shape);
        }
        let mut compiler = Compiler {
            chain: self,
            steps: Vec::new(),
            buffers: Vec::new(),
        };
        plan.execute(&mut compiler)?;
        let dense_inputs = self
            .inputs
            .iter()
            .map(|matrix| {
                let size = matrix
                    .num_rows
                    .checked_mul(matrix.num_cols)
                    .ok_or(MatrixChainError::Overflow)?;
                let mut dense = vec![F::zero(); size];
                for row in 0..matrix.num_rows {
                    for at in matrix.row_ptr[row]..matrix.row_ptr[row + 1] {
                        dense[row * matrix.num_cols + matrix.col_ind[at]] += matrix.val[at];
                    }
                }
                Ok(dense)
            })
            .collect::<Result<_, MatrixChainError>>()?;
        Ok(PreparedMatrixChain {
            chain: self,
            steps: compiler.steps,
            buffers: compiler.buffers,
            dense_inputs,
        })
    }
}
struct Slot {
    id: usize,
    axes: [usize; 2],
}
struct Step {
    left: usize,
    right: usize,
}
struct Compiler<'a, 'input, F: Real> {
    chain: &'a MatrixChain<'input, F>,
    steps: Vec<Step>,
    buffers: Vec<TensorCsr<F>>,
}
impl<F: Real> ContractionVisitor for Compiler<'_, '_, F> {
    type Value = Slot;
    type Error = MatrixChainError;
    fn input(&mut self, factor: usize, axes: &[usize]) -> Result<Slot, Self::Error> {
        let axes = axes
            .try_into()
            .map_err(|_| MatrixChainError::UnsupportedStep)?;
        Ok(Slot { id: factor, axes })
    }
    fn contract(
        &mut self,
        mut left: Slot,
        mut right: Slot,
        step: &ContractionStep,
    ) -> Result<Slot, Self::Error> {
        if right.axes[1] == left.axes[0] {
            std::mem::swap(&mut left, &mut right)
        }
        if left.axes[1] != right.axes[0] || step.axes != [left.axes[0], right.axes[1]] {
            return Err(MatrixChainError::UnsupportedStep);
        }
        let (rows, cols) = (
            self.chain.problem.extents[left.axes[0]],
            self.chain.problem.extents[right.axes[1]],
        );
        let size = rows.checked_mul(cols).ok_or(MatrixChainError::Overflow)?;
        rows.checked_add(1).ok_or(MatrixChainError::Overflow)?;
        let matrix = TensorCsr {
            num_rows: rows,
            num_cols: cols,
            row_ptr: (0..=rows).map(|r| r * cols).collect(),
            col_ind: (0..rows).flat_map(|_| 0..cols).collect(),
            val: vec![F::zero(); size],
        };
        let id = self.chain.inputs.len() + self.steps.len();
        self.steps.push(Step {
            left: left.id,
            right: right.id,
        });
        self.buffers.push(matrix);
        Ok(Slot {
            id,
            axes: [left.axes[0], right.axes[1]],
        })
    }
}
/// Prepared numerical execution; all numeric buffers reused across calls.
///
/// Uses exactly `TensorCsr::spmm_into_serial`. Floating association is the
/// caller-selected plan; numerical equality across trees is not bitwise promised.
pub struct PreparedMatrixChain<'a, 'input, F: Real> {
    chain: &'a MatrixChain<'input, F>,
    steps: Vec<Step>,
    buffers: Vec<TensorCsr<F>>,
    dense_inputs: Vec<Vec<F>>,
}
impl<F: Real> PreparedMatrixChain<'_, '_, F> {
    /// Apply the prepared chain. No allocation or plan search occurs here.
    pub fn execute(&mut self) -> &[F] {
        let n = self.chain.inputs.len();
        for (i, step) in self.steps.iter().enumerate() {
            let (previous, current) = self.buffers.split_at_mut(i);
            let left = if step.left < n {
                &self.chain.inputs[step.left]
            } else {
                &previous[step.left - n]
            };
            let right = if step.right < n {
                &self.dense_inputs[step.right]
            } else {
                &previous[step.right - n].val
            };
            left.spmm_into_serial(right, current[0].num_cols, &mut current[0].val);
        }
        self.buffers
            .last()
            .map_or(&self.dense_inputs[0], |matrix| matrix.val.as_slice())
    }
    /// Retained numeric/index buffer payload, excluding borrowed inputs and allocator overhead.
    pub fn retained_bytes(&self) -> usize {
        self.dense_inputs
            .iter()
            .map(|x| x.len() * std::mem::size_of::<F>())
            .sum::<usize>()
            + self
                .buffers
                .iter()
                .map(|b| {
                    b.val.len() * std::mem::size_of::<F>()
                        + (b.row_ptr.len() + b.col_ind.len()) * std::mem::size_of::<usize>()
                })
                .sum::<usize>()
    }
}
