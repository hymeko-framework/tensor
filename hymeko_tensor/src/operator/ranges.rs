//! Range gather and transpose scatter, with explicit reassociation semantics.
use super::{OrderedAccumulator, RowTransform, RowVisitor, SparseRow};
use std::ops::Range;

/// Disjoint active intervals. A row interval must stay inside one active block.
/// Components may occupy multiple blocks; isolated coordinates can be omitted.
pub struct RangePartition {
    blocks: Vec<Range<usize>>,
    starts: Vec<bool>,
    domain: usize,
}
impl RangePartition {
    pub fn new(domain: usize, blocks: Vec<Range<usize>>) -> Option<Self> {
        let mut end = 0;
        for block in &blocks {
            if block.start < end || block.start >= block.end || block.end > domain {
                return None;
            }
            end = block.end;
        }
        let mut starts = vec![false; domain];
        for block in &blocks {
            starts[block.start] = true;
        }
        Some(Self {
            blocks,
            starts,
            domain,
        })
    }
    pub fn blocks(&self) -> &[Range<usize>] {
        &self.blocks
    }
    pub fn retained_bytes(&self) -> usize {
        self.blocks.capacity() * size_of::<Range<usize>>() + self.starts.capacity()
    }
}

/// Caller-owned buffers reused across applications. No output graph is stored.
pub struct RangeWorkspace<A> {
    prefix: Vec<f64>,
    difference: Vec<A>,
}
impl<A> Default for RangeWorkspace<A> {
    fn default() -> Self {
        Self {
            prefix: Vec::new(),
            difference: Vec::new(),
        }
    }
}
impl<A: OrderedAccumulator> RangeWorkspace<A> {
    pub fn retained_bytes(&self) -> usize {
        self.prefix.capacity() * 8 + self.difference.capacity() * size_of::<A>()
    }
    /// Prepare prefix sums and clear endpoint accumulators. The caller proves
    /// every visited row interval lies in an active block; input extent is checked.
    pub fn prepare(&mut self, input: &[f64], partition: &RangePartition) {
        self.prepare_mapped(input, partition, |coordinate| coordinate);
    }
    /// Prepare the same prefixes using an execution-to-input coordinate map.
    /// Input and execution domains have the same extent. The map is evaluated
    /// only for active coordinates; invalid mapped indices panic on access.
    pub fn prepare_mapped(
        &mut self,
        input: &[f64],
        partition: &RangePartition,
        coordinate: impl Fn(usize) -> usize,
    ) {
        assert_eq!(input.len(), partition.domain);
        self.prefix.resize(input.len(), 0.0);
        self.difference.resize_with(input.len(), A::default);
        for block in &partition.blocks {
            let mut sum = A::default();
            for ((prefix, difference), execution) in self.prefix[block.clone()]
                .iter_mut()
                .zip(&mut self.difference[block.clone()])
                .zip(block.clone())
            {
                sum.add(input[coordinate(execution)]);
                *prefix = sum.total();
                *difference = A::default();
            }
        }
    }
    pub fn contraction<'a, T: RowTransform>(
        &'a mut self,
        partition: &'a RangePartition,
        transform: T,
    ) -> RangeContraction<'a, A, T> {
        assert_eq!(self.prefix.len(), partition.domain);
        RangeContraction {
            workspace: self,
            partition,
            transform,
        }
    }
    /// Add the transpose result into an existing output, preserving other work.
    pub fn accumulate_into(&mut self, partition: &RangePartition, output: &mut [A]) {
        self.accumulate_mapped(partition, output, |coordinate| coordinate);
    }
    /// Accumulate the transpose result through an execution-to-output map.
    /// Repeated mapped indices receive ordered additions; output is never cleared.
    /// Domain extents must agree and mapped indices must be in bounds.
    pub fn accumulate_mapped(
        &mut self,
        partition: &RangePartition,
        output: &mut [A],
        coordinate: impl Fn(usize) -> usize,
    ) {
        assert_eq!(output.len(), partition.domain);
        for block in &partition.blocks {
            let mut sum = A::default();
            for (difference, execution) in
                self.difference[block.clone()].iter_mut().zip(block.clone())
            {
                sum.add(difference.take_total());
                output[coordinate(execution)].add(sum.total());
            }
        }
    }
}

/// The same row-visitor seam as ordered contraction. Unit coefficients only;
/// weights/routing must be represented by the caller or use RowContraction.
pub struct RangeContraction<'a, A, T> {
    workspace: &'a mut RangeWorkspace<A>,
    partition: &'a RangePartition,
    transform: T,
}
impl<A: OrderedAccumulator, T: RowTransform> RowVisitor for RangeContraction<'_, A, T> {
    #[inline]
    fn visit<R: SparseRow>(&mut self, row: usize, coordinates: R) {
        if !self.transform.enabled(row) {
            return;
        }
        let mut sum = A::default();
        coordinates.for_each_range(|range| {
            let before = if self.partition.starts[range.start] {
                0.0
            } else {
                self.workspace.prefix[range.start - 1]
            };
            sum.add(self.workspace.prefix[range.end - 1] - before);
        });
        let value = self.transform.apply(row, sum.total());
        coordinates.for_each_range(|range| {
            self.workspace.difference[range.start].add(value);
            if range.end < self.partition.domain && !self.partition.starts[range.end] {
                self.workspace.difference[range.end].add(-value);
            }
        });
    }
}
