//! Ordered scalar accumulation and borrowed sparse row contracts.

/// Accumulation policy for an ordered incidence fold. The operator owns traversal;
/// the consumer may preserve a compensated arithmetic contract without copying it.
pub trait OrderedAccumulator: Default {
    /// Add one term in the exact order supplied by the operator.
    fn add(&mut self, value: f64);
    /// Read the current sum, without consuming or resetting it.
    fn total(&self) -> f64;
    /// Read the completed total, then replace the accumulator with its default
    /// state. Consumers can reset a buffer while reading it, avoiding a separate
    /// clearing pass. The returned value is exactly the value of `total()`.
    #[inline]
    fn take_total(&mut self) -> f64 {
        let total = self.total();
        *self = Self::default();
        total
    }
    /// Initialize an ordered fold from its first term. An override must preserve
    /// the result of `Default::default()` followed by `add(value)`, including
    /// subsequent additions. This lets a policy avoid redundant zero-state work.
    #[inline]
    fn from_first(value: f64) -> Self {
        let mut accumulator = Self::default();
        accumulator.add(value);
        accumulator
    }
    /// Reduce two lazy terms with the same result as `sum`. The default honors
    /// overridden `sum` policies. Evaluate each term exactly once, in order;
    /// laziness preserves the order of policy calls and coefficient evaluation.
    #[inline]
    fn sum_pair(first: impl FnOnce() -> f64, second: impl FnOnce() -> f64) -> f64 {
        Self::sum(std::iter::once_with(first).chain(std::iter::once_with(second)))
    }
    /// Fold terms without reassociation. Both live and prepared incidence
    /// operators use this ordered reduction, with their chosen arithmetic policy.
    #[inline]
    fn sum(terms: impl IntoIterator<Item = f64>) -> f64 {
        let mut terms = terms.into_iter();
        let Some(first) = terms.next() else {
            return Self::default().total();
        };
        let mut accumulator = Self::from_first(first);
        for term in terms {
            accumulator.add(term);
        }
        accumulator.total()
    }
    /// Reduce ordered chunks with the exact semantics of their concatenation.
    /// The default honors `sum` overrides. Specializations may remove chunk
    /// iterator overhead but must preserve term order and lazy evaluation.
    #[inline]
    fn sum_chunks<I, C>(chunks: I) -> f64
    where
        I: IntoIterator<Item = C>,
        C: IntoIterator<Item = f64>,
    {
        Self::sum(chunks.into_iter().flatten())
    }
}

/// Ordered, borrowed incidence-row operations, specialized by coordinate storage.
/// Every implementation preserves canonical slot order and duplicate targets.
pub trait SparseRow {
    /// First global coefficient slot of this row.
    fn offset(&self) -> usize;
    /// Number of ordered slots, including duplicates.
    fn len(&self) -> usize;
    /// Whether this row contains no slots.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Canonical target indices without allocation or expansion.
    fn iter_targets(&self) -> impl Iterator<Item = usize>;
    /// Cover the same target multiset by nonempty ascending intervals. Repeated
    /// slots must remain repeated. This explicit algebraic view permits a new
    /// summation order; it does not promise bitwise ordered-fold equivalence.
    /// The fallback emits singleton intervals without extra storage.
    fn for_each_range(&self, mut visit: impl FnMut(std::ops::Range<usize>)) {
        for target in self.iter_targets() {
            visit(target..target + 1);
        }
    }
    /// Ordered gather; panics lazily if an input/coefficient slot is missing.
    fn gather<A: OrderedAccumulator>(
        &self,
        input: &[f64],
        coefficient: impl Fn(usize) -> f64,
    ) -> f64;
    /// Ordered transpose additions; never clears output or collapses duplicates.
    /// Panics at the first missing output/coefficient slot, retaining prior writes.
    fn scatter<A: OrderedAccumulator>(
        &self,
        value: f64,
        output: &mut [A],
        coefficient: impl Fn(usize) -> f64,
    );
}

/// Allocation-free sparse row over borrowed target indices.
/// Uses the shared ordered-accumulation policy, retaining the two-slot fast path.
/// The range view emits one singleton per slot, including duplicate targets.
#[derive(Clone, Copy, Debug)]
pub struct IndexedRow<'a> {
    offset: usize,
    targets: &'a [usize],
}
impl<'a> IndexedRow<'a> {
    /// Borrow ordered targets with their first global coefficient slot.
    ///
    /// Construction neither allocates nor checks vector bounds. Targets and
    /// coefficient positions are checked lazily when gather/scatter accesses
    /// them; duplicates and target order are preserved. Coefficient positions
    /// `offset + slot` must be representable as `usize` when used.
    #[inline]
    pub fn new(offset: usize, targets: &'a [usize]) -> Self {
        Self { offset, targets }
    }
}
impl SparseRow for IndexedRow<'_> {
    fn offset(&self) -> usize {
        self.offset
    }
    fn len(&self) -> usize {
        self.targets.len()
    }
    fn iter_targets(&self) -> impl Iterator<Item = usize> {
        self.targets.iter().copied()
    }
    #[inline(always)]
    fn gather<A: OrderedAccumulator>(
        &self,
        input: &[f64],
        coefficient: impl Fn(usize) -> f64,
    ) -> f64 {
        if let &[first, second] = self.targets {
            return A::sum_pair(
                || input[first] * coefficient(self.offset),
                || input[second] * coefficient(self.offset + 1),
            );
        }
        A::sum(
            self.targets
                .iter()
                .enumerate()
                .map(|(slot, &target)| input[target] * coefficient(self.offset + slot)),
        )
    }
    #[inline(always)]
    fn scatter<A: OrderedAccumulator>(
        &self,
        value: f64,
        output: &mut [A],
        coefficient: impl Fn(usize) -> f64,
    ) {
        let mut add_slot = |slot: usize, target: usize| {
            output[target].add(value * coefficient(self.offset + slot));
        };
        if let &[first, second] = self.targets {
            add_slot(0, first);
            add_slot(1, second);
        } else {
            for (slot, &target) in self.targets.iter().enumerate() {
                add_slot(slot, target);
            }
        }
    }
}

/// Statically dispatched consumer of canonical ordered rows. No graph is copied.
pub trait RowVisitor {
    /// Called once for each row, in increasing canonical arc order.
    fn visit<R: SparseRow>(&mut self, arc: usize, row: R);
}
impl OrderedAccumulator for f64 {
    #[inline]
    fn sum_chunks<I, C>(chunks: I) -> f64
    where
        I: IntoIterator<Item = C>,
        C: IntoIterator<Item = f64>,
    {
        chunks.into_iter().fold(0.0, |sum, chunk| {
            chunk.into_iter().fold(sum, |sum, term| sum + term)
        })
    }
    #[inline]
    fn add(&mut self, value: f64) {
        *self += value;
    }
    #[inline]
    fn total(&self) -> f64 {
        *self
    }
}
