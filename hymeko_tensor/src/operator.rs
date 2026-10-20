//! Matrix-free contraction over borrowed, ordered sparse rows.
//!
//! Coordinates and coefficients remain with their owners. A contraction visits
//! two rows, retaining one intermediate scalar; no matrix product is formed.
//! Duplicate coordinates and reduction order are part of the contract.

mod accumulator;
mod ranges;
pub use accumulator::{IndexedRow, OrderedAccumulator, RowVisitor, SparseRow};
pub use ranges::{RangeContraction, RangePartition, RangeWorkspace};

/// Row-local operation between gather and scatter.
pub trait RowTransform {
    /// Whether to evaluate this row at all. A disabled row performs no reads.
    fn enabled(&self, _row: usize) -> bool {
        true
    }
    /// Transform a completed gather. Called exactly once for an enabled row.
    fn apply(&self, row: usize, value: f64) -> f64;
    /// Shared guarded reduction for gathered and fused applications. Disabled
    /// rows perform no input/coefficient reads. Overrides must preserve that
    /// contract and the chosen accumulator's reduction order.
    #[inline(always)]
    fn gather<A: OrderedAccumulator>(
        &self,
        row: usize,
        coordinates: &impl SparseRow,
        input: &[f64],
        coefficient: impl Fn(usize) -> f64,
    ) -> Option<f64>
    where
        Self: Sized,
    {
        self.enabled(row)
            .then(|| self.apply(row, coordinates.gather::<A>(input, coefficient)))
    }
}

impl<F: Fn(usize, f64) -> f64> RowTransform for F {
    fn apply(&self, row: usize, value: f64) -> f64 {
        self(row, value)
    }
}

/// Fused `destination^T * transform(source * input)`, with caller-owned output.
///
/// Output is accumulated, never cleared. Source and destination coefficients
/// are indexed in their respective row spaces. Policies must be pure if a caller
/// intends to reuse or reorder applications. This visitor itself never reorders.
pub struct RowContraction<'a, A, S, D, T, G = A> {
    input: &'a [f64],
    output: &'a mut [A],
    source: S,
    destination: D,
    transform: T,
    gather: std::marker::PhantomData<G>,
}

impl<'a, A: OrderedAccumulator, S: Fn(usize) -> f64, D: Fn(usize) -> f64, T: RowTransform>
    RowContraction<'a, A, S, D, T>
{
    /// Borrow vectors and scalar policies; performs no allocation or evaluation.
    pub fn new(
        input: &'a [f64],
        output: &'a mut [A],
        source: S,
        destination: D,
        transform: T,
    ) -> Self {
        Self {
            input,
            output,
            source,
            destination,
            transform,
            gather: std::marker::PhantomData,
        }
    }
    /// Choose the gather policy independently of output accumulation. A caller
    /// can preserve long compensated reductions while proving a cheaper bound
    /// for the number of terms arriving at each destination.
    pub fn with_gather<G: OrderedAccumulator>(self) -> RowContraction<'a, A, S, D, T, G> {
        RowContraction {
            input: self.input,
            output: self.output,
            source: self.source,
            destination: self.destination,
            transform: self.transform,
            gather: std::marker::PhantomData,
        }
    }
}

impl<
        A: OrderedAccumulator,
        S: Fn(usize) -> f64,
        D: Fn(usize) -> f64,
        T: RowTransform,
        G: OrderedAccumulator,
    > RowContraction<'_, A, S, D, T, G>
{
    /// Vector extents, for an adapter to check before any output is changed.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.input.len(), self.output.len())
    }

    /// Contract explicitly aligned rows. `row` identifies the destination row.
    ///
    /// # Panics
    /// As the rows' gather/scatter contracts; prior writes remain visible.
    #[inline(always)]
    pub fn pair(&mut self, row: usize, source: &impl SparseRow, destination: &impl SparseRow) {
        if let Some(value) = self
            .transform
            .gather::<G>(row, source, self.input, &self.source)
        {
            destination.scatter(value, self.output, &self.destination);
        }
    }
}

/// Write one transformed scalar per visited row, without an intermediate matrix.
/// Row reduction is shared with `RowContraction`; output cells are overwritten.
pub struct RowGather<'a, S, T, G = f64> {
    input: &'a [f64],
    output: &'a mut [f64],
    source: S,
    transform: T,
    first_row: usize,
    gather: std::marker::PhantomData<G>,
}
impl<'a, S, T> RowGather<'a, S, T> {
    /// Borrow input/output and policies; performs no allocation or evaluation.
    pub fn new(input: &'a [f64], output: &'a mut [f64], source: S, transform: T) -> Self {
        Self {
            input,
            output,
            source,
            transform,
            first_row: 0,
            gather: std::marker::PhantomData,
        }
    }
}
impl<'a, S, T, G> RowGather<'a, S, T, G> {
    /// Let output cell zero represent this global row ordinal.
    pub fn starting_at(mut self, first_row: usize) -> Self {
        self.first_row = first_row;
        self
    }
    /// Select the same ordered gather policy available to fused contraction.
    pub fn with_gather<A: OrderedAccumulator>(self) -> RowGather<'a, S, T, A> {
        RowGather {
            input: self.input,
            output: self.output,
            source: self.source,
            transform: self.transform,
            first_row: self.first_row,
            gather: std::marker::PhantomData,
        }
    }
}
impl<S: Fn(usize) -> f64, T: RowTransform, G: OrderedAccumulator> RowVisitor
    for RowGather<'_, S, T, G>
{
    /// # Panics
    /// If the row is outside the configured output span, or gather accesses an
    /// invalid input/coefficient. Disabled rows write zero without reading input.
    #[inline(always)]
    fn visit<R: SparseRow>(&mut self, row: usize, coordinates: R) {
        assert!(row >= self.first_row, "row precedes output span");
        self.output[row - self.first_row] = self
            .transform
            .gather::<G>(row, &coordinates, self.input, &self.source)
            .unwrap_or(0.0);
    }
}

impl<
        A: OrderedAccumulator,
        S: Fn(usize) -> f64,
        D: Fn(usize) -> f64,
        T: RowTransform,
        G: OrderedAccumulator,
    > RowVisitor for RowContraction<'_, A, S, D, T, G>
{
    #[inline(always)]
    fn visit<R: SparseRow>(&mut self, row: usize, coordinates: R) {
        self.pair(row, &coordinates, &coordinates);
    }
}
