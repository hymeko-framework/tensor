//! Bounded contraction-order planning over index metadata, independent of storage.
//!
//! Work is the sum of products of extents in each pair's union (one nominal
//! multiply/accumulate per index tuple). This is a shape-based cost model, not a
//! claim about sparse runtime. A plan may be replayed by a virtual or stored
//! executor. Reassociation is explicit: a new tree can change floating-point bits.

mod bounded;
#[cfg(feature = "sparse")]
mod matrix_chain;
mod search;
pub use bounded::{BoundedOutcome, BoundedProblem, BranchAndBound};
#[cfg(feature = "sparse")]
pub use matrix_chain::{MatrixChain, MatrixChainError, PreparedMatrixChain};
pub use search::ContractionPlanner;

/// Invalid index metadata or an unrepresentable planning problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanningError {
    /// At least one input factor is required.
    NoFactors,
    /// An axis was outside the extent table or repeated within a factor/output.
    InvalidAxes,
    /// An output index is absent, or a contracted index occurs in only one factor.
    /// Unary reductions must be represented explicitly before pair planning.
    UnpairedAxis,
    /// Shape/cost arithmetic exceeds its checked integer representation.
    Overflow,
}
impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "contraction planning: {self:?}")
    }
}
impl std::error::Error for PlanningError {}

/// Immutable hyperindex description. Axis order is canonicalized, not tensor data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractionProblem {
    extents: Vec<usize>,
    factors: Vec<Vec<usize>>,
    output: Vec<usize>,
}
impl ContractionProblem {
    /// Validate shape metadata. Shared axes may occur in more than two factors.
    ///
    /// Factor axes denote sets; executors receive their canonical ascending order.
    /// Repeated axes (diagonals/traces) require an explicit input view first.
    pub fn new(
        extents: Vec<usize>,
        mut factors: Vec<Vec<usize>>,
        mut output: Vec<usize>,
    ) -> Result<Self, PlanningError> {
        if factors.is_empty() {
            return Err(PlanningError::NoFactors);
        }
        for axes in factors.iter_mut().chain(std::iter::once(&mut output)) {
            axes.sort_unstable();
            if axes.iter().any(|&a| a >= extents.len()) || axes.windows(2).any(|p| p[0] == p[1]) {
                return Err(PlanningError::InvalidAxes);
            }
        }
        for axis in 0..extents.len() {
            let occurrences = factors
                .iter()
                .filter(|f| f.binary_search(&axis).is_ok())
                .count();
            if (output.binary_search(&axis).is_ok() && occurrences == 0)
                || (output.binary_search(&axis).is_err() && occurrences == 1)
            {
                return Err(PlanningError::UnpairedAxis);
            }
        }
        let problem = Self {
            extents,
            factors,
            output,
        };
        // Every subproduct is bounded by the product over nonzero extents.
        problem
            .extents
            .iter()
            .filter(|&&n| n != 0)
            .try_fold(1u128, |p, &n| p.checked_mul(n as u128))
            .ok_or(PlanningError::Overflow)?;
        Ok(problem)
    }
    /// Extent of each axis, indexed by the axis ID.
    pub fn extents(&self) -> &[usize] {
        &self.extents
    }
    /// Canonical input-axis sets, in input-factor order.
    pub fn factors(&self) -> &[Vec<usize>] {
        &self.factors
    }
    /// Canonical requested output-axis set.
    pub fn output_axes(&self) -> &[usize] {
        &self.output
    }
    fn volume(&self, axes: &[usize]) -> u128 {
        if axes.iter().any(|&a| self.extents[a] == 0) {
            return 0;
        }
        // Constructor checked the product over every nonzero extent.
        axes.iter().map(|&a| self.extents[a] as u128).product()
    }
}

/// Hard search-state and per-intermediate logical element limits.
#[derive(Clone, Copy, Debug)]
pub struct PlanningBudget {
    /// Largest allowed result rank. Use usize::MAX for an unrestricted executor.
    pub max_intermediate_rank: usize,
    /// Number of branch successors evaluated, excluding greedy initialization.
    /// Zero returns the greedy feasible plan without claiming optimality.
    pub max_states: usize,
    /// Maximum elements in any pair result, including the final output.
    /// This is not a total live-memory or RSS limit; virtual results may use less.
    pub max_intermediate_elements: u128,
}

/// One elementary pair contraction; IDs refer to inputs then preceding results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractionStep {
    /// First operand ID.
    pub left: usize,
    /// Second operand ID.
    pub right: usize,
    /// Surviving axes, including hyperindices still consumed elsewhere.
    pub axes: Vec<usize>,
    /// Exact work for this pair under the declared index-product model.
    pub work: u128,
    /// Logical result element count.
    pub elements: u128,
}

/// A validated, complete contraction tree tied to the problem it plans.
#[derive(Clone, Debug)]
pub struct ContractionPlan<'a> {
    problem: &'a ContractionProblem,
    steps: Vec<ContractionStep>,
    work: u128,
    peak: u128,
}
impl ContractionPlan<'_> {
    /// Ordered elementary contractions. No numerical tensor is owned here.
    pub fn steps(&self) -> &[ContractionStep] {
        &self.steps
    }
    /// Total nominal work, not elapsed time.
    pub fn work(&self) -> u128 {
        self.work
    }
    /// Largest pair result (or the sole input for a one-factor expression).
    pub fn largest_intermediate(&self) -> u128 {
        self.peak
    }
    /// Execute through the caller's existing operator visitor. Virtual handles
    /// need not store tensor entries. An executor chooses its own workspace.
    pub fn execute<V: ContractionVisitor>(&self, visitor: &mut V) -> Result<V::Value, V::Error> {
        let mut values: Vec<Option<V::Value>> =
            Vec::with_capacity(self.problem.factors.len() + self.steps.len());
        for (id, axes) in self.problem.factors.iter().enumerate() {
            values.push(Some(visitor.input(id, axes)?));
        }
        for step in &self.steps {
            // Private plan construction guarantees a tree and live operand IDs.
            let Some(left) = values[step.left].take() else {
                unreachable!("validated contraction tree")
            };
            let Some(right) = values[step.right].take() else {
                unreachable!("validated contraction tree")
            };
            values.push(Some(visitor.contract(left, right, step)?));
        }
        match values.pop().flatten() {
            Some(value) => Ok(value),
            None => unreachable!("validated nonempty contraction problem"),
        }
    }
}

/// Execution seam: an existing backend may return virtual operator handles.
pub trait ContractionVisitor {
    /// Stored tensor or virtual expression handle.
    type Value;
    /// Backend-specific failure, propagated without suppression.
    type Error;
    /// Obtain an input, with canonical axis metadata.
    fn input(&mut self, factor: usize, axes: &[usize]) -> Result<Self::Value, Self::Error>;
    /// Compose two handles according to this elementary contraction.
    fn contract(
        &mut self,
        left: Self::Value,
        right: Self::Value,
        step: &ContractionStep,
    ) -> Result<Self::Value, Self::Error>;
}

/// Result distinguishes a feasible incumbent from proof of optimality/infeasibility.
#[derive(Debug)]
pub struct PlanningOutcome<'a> {
    /// Best complete feasible tree found; absent is not infeasibility if truncated.
    pub plan: Option<ContractionPlan<'a>>,
    /// All feasible branches were exhausted or validly bounded under the cost model.
    pub complete: bool,
    /// Evaluated branch successors (never exceeds the requested limit).
    pub states: usize,
    /// Branches excluded by incumbent cost or intermediate-size limits.
    pub pruned: usize,
}
