// APPROVED-CORE-EDIT: csr-spmv-spmm-into (2026-08-24, user chat).
//
// Scope of the approval: add allocation-free `*_into` kernels with explicit
// serial and Rayon variants, and make the existing `spmv` / `spmm` convenience
// methods delegate to them. Constraints carried from CORE.YAML:
//   * no change to existing public signatures;
//   * no change to floating-point accumulation order;
//   * regression tests required (serial vs parallel, and vs a dense reference).
//
// The accumulation order constraint is what makes the parallel path safe to
// enable by default: SpMV and SpMM partition by *output row*, and rows are
// disjoint. Each output element is still summed over exactly the same `k` in
// exactly the same sequence as the serial loop, so the parallel result is
// bitwise identical rather than merely within tolerance. No atomics, no locks,
// no reduction tree, no reassociation.

use crate::common::Real;
use rayon::prelude::*;

pub struct TensorCsrBuilder<F> {
    pub dim_i: usize,
    pub dim_j: usize,
    pub uncoalesced_row_ptr: Vec<usize>, // Length: dim_i
    pub cols: Vec<usize>,                // Length: nnz (number of non-zeros)
    pub vals: Vec<F>,                    // Length: nnz
}

#[derive(Debug, Clone)]
pub struct TensorCsr<F> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_ptr: Vec<usize>, // Length: num_rows + 1
    pub col_ind: Vec<usize>, // Length: nnz (number of non-zeros)
    pub val: Vec<F>,         // Length: nnz
}

impl<F: Real> TensorCsr<F> {
    /// Non-zero count below which [`Self::spmv_into`] stays on the serial path.
    ///
    /// Measured on the reference host (24 logical cores) with
    /// `cargo bench --bench csr_bench -- csr_spmv_paths`:
    ///
    /// | nnz | serial | parallel |
    /// |----------:|---------:|---------:|
    /// | 1 024 | 429 ns | 18.43 us |
    /// | 16 384 | 5.20 us | 22.11 us |
    /// | 65 536 | 21.30 us | 24.41 us |
    /// | 262 144 | 108.3 us | 33.63 us |
    /// | 4 194 304 | 4.675 ms | 487.4 us |
    ///
    /// Rayon's fork/join floor is ~18.4 us here. Fitting the serial curve
    /// (~3.1 Gelem/s) against the parallel one (~18.4 us + w / 10.8 Gelem/s)
    /// puts the crossover at ~80 000 non-zeros — so the first draft of this
    /// constant, 65 536, was measurably *too low*: at exactly that size the
    /// parallel path is 15 % slower. The value below is the measured crossover
    /// rounded to 80 x 1024.
    pub const PAR_SPMV_NNZ_THRESHOLD: usize = 81_920;

    /// Multiply-accumulate count (`nnz * feature_dim`) below which
    /// [`Self::spmm_into`] stays on the serial path.
    ///
    /// Deliberately much higher than [`Self::PAR_SPMV_NNZ_THRESHOLD`]. SpMM's
    /// inner loop runs over two contiguous, equal-length slices and vectorises,
    /// reaching ~11 Gelem/s serially against SpMV's ~3 Gelem/s; more work is
    /// therefore needed before the same fork/join cost pays for itself. Using
    /// one shared threshold for both would misdispatch one of them.
    ///
    /// Measured with `cargo bench --bench csr_bench -- csr_spmm_paths`
    /// (nnz fixed at 16 384, sweeping `feature_dim`):
    ///
    /// | feature_dim | work | serial | parallel |
    /// |---:|---------:|---------:|---------:|
    /// | 1 | 16 k | 17.60 us | 26.27 us |
    /// | 8 | 131 k | 20.47 us | 24.57 us |
    /// | 32 | 524 k | 47.92 us | 27.36 us |
    /// | 128 | 2.1 M | 170.5 us | 37.52 us |
    ///
    /// The crossover falls between 131 k and 524 k; 2^18 sits just past it.
    pub const PAR_SPMM_WORK_THRESHOLD: usize = 262_144;

    /// Pre-allocates the exact memory required for the CSR structure.
    pub fn with_capacity(num_rows: usize, num_cols: usize, nnz: usize) -> Self {
        let row_ptr = vec![0; num_rows + 1];
        Self {
            num_rows,
            num_cols,
            row_ptr,
            col_ind: vec![0; nnz],
            val: vec![F::zero(); nnz],
        }
    }

    /// Number of stored non-zeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// Rows per Rayon task.
    ///
    /// Sized to hand each worker several tasks rather than exactly one, so
    /// work-stealing can rebalance an irregular row distribution — the
    /// `DegreeShape::Irregular` workloads exist precisely to keep this honest.
    /// A single task per thread would leave every worker idling behind the one
    /// that drew the hub rows.
    #[inline]
    fn row_chunk(num_rows: usize) -> usize {
        let tasks = rayon::current_num_threads().saturating_mul(8).max(1);
        num_rows.div_ceil(tasks).max(1)
    }

    // =======================================================================
    // SpMV — y = A * x
    // =======================================================================

    /// Executes y = A * x (Sparse Matrix-Vector Multiplication)
    /// Complexity: O(NNZ)
    ///
    /// Convenience wrapper: allocates the output and delegates to
    /// [`Self::spmv_into`]. Callers in a loop should hoist the buffer and call
    /// [`Self::spmv_into`] directly.
    ///
    /// # Panics
    /// If `x.len() != self.num_cols`.
    pub fn spmv(&self, x: &[F]) -> Vec<F> {
        let mut y = vec![F::zero(); self.num_rows];
        self.spmv_into(x, &mut y);
        y
    }

    /// Allocation-free `y = A * x`, dispatching on measured workload size.
    ///
    /// # Preconditions
    /// `x.len() == self.num_cols` and `y.len() == self.num_rows`.
    ///
    /// # Postconditions
    /// Every element of `y` is **overwritten**, never accumulated into — a
    /// dirty input buffer yields the same result as a freshly zeroed one.
    /// The result is bitwise identical whichever path is taken.
    ///
    /// # Panics
    /// If either precondition is violated.
    #[inline]
    pub fn spmv_into(&self, x: &[F], y: &mut [F]) {
        if self.nnz() >= Self::PAR_SPMV_NNZ_THRESHOLD {
            self.spmv_into_parallel(x, y);
        } else {
            self.spmv_into_serial(x, y);
        }
    }

    /// Single-threaded `y = A * x`.
    ///
    /// Contract as [`Self::spmv_into`]. Exposed rather than private so the
    /// regression suite can assert serial/parallel equivalence, and so a
    /// caller already inside a Rayon task can avoid nested parallelism.
    pub fn spmv_into_serial(&self, x: &[F], y: &mut [F]) {
        self.assert_spmv_dims(x, y);

        // Row-wise: row_ptr partitions col_ind/val, so this walks both arrays
        // strictly forward. Cache-friendly by construction.
        for (i, slot) in y.iter_mut().enumerate() {
            *slot = self.row_dot(i, x);
        }
    }

    /// Rayon `y = A * x`, partitioned by output row.
    ///
    /// Contract as [`Self::spmv_into`]. Rows are disjoint, so no
    /// synchronisation of any kind is required and no floating-point
    /// reassociation occurs.
    pub fn spmv_into_parallel(&self, x: &[F], y: &mut [F]) {
        self.assert_spmv_dims(x, y);
        if self.num_rows == 0 {
            return;
        }

        let chunk = Self::row_chunk(self.num_rows);
        y.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_idx, block)| {
                let base = chunk_idx * chunk;
                for (offset, slot) in block.iter_mut().enumerate() {
                    *slot = self.row_dot(base + offset, x);
                }
            });
    }

    /// Dot product of row `i` with `x`.
    ///
    /// The single definition of SpMV's inner loop; both the serial and the
    /// parallel path call it, which is what guarantees they cannot drift apart
    /// numerically.
    #[inline(always)]
    fn row_dot(&self, i: usize, x: &[F]) -> F {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];

        // Local accumulator avoids repeated heap writes.
        let mut dot_product = F::zero();
        for k in start..end {
            let col = self.col_ind[k];
            let val = self.val[k];
            dot_product += val * x[col];
        }
        dot_product
    }

    #[inline]
    fn assert_spmv_dims(&self, x: &[F], y: &[F]) {
        assert_eq!(
            self.num_cols,
            x.len(),
            "Dimension mismatch: CSR has {} columns, but vector x has length {}",
            self.num_cols,
            x.len()
        );
        assert_eq!(
            self.num_rows,
            y.len(),
            "Dimension mismatch: CSR has {} rows, but output y has length {}",
            self.num_rows,
            y.len()
        );
    }

    // =======================================================================
    // SpMM — Y = A * X
    // =======================================================================

    /// Executes Y = A * X (Sparse Matrix-Dense Matrix Multiplication)
    /// X is a flattened 2D array of shape (num_cols, feature_dim)
    /// Returns a flattened 2D array of shape (num_rows, feature_dim)
    ///
    /// Convenience wrapper: allocates the output and delegates to
    /// [`Self::spmm_into`].
    ///
    /// # Panics
    /// If `x_flat.len() != self.num_cols * feature_dim`.
    pub fn spmm(&self, x_flat: &[F], feature_dim: usize) -> Vec<F> {
        let mut y_flat = vec![F::zero(); self.num_rows * feature_dim];
        self.spmm_into(x_flat, feature_dim, &mut y_flat);
        y_flat
    }

    /// Allocation-free `Y = A * X`, dispatching on measured workload size.
    ///
    /// # Preconditions
    /// `x_flat.len() == self.num_cols * feature_dim` and
    /// `y_flat.len() == self.num_rows * feature_dim`.
    ///
    /// # Postconditions
    /// Every element of `y_flat` is **overwritten**, never accumulated into.
    /// Bitwise identical across both paths.
    ///
    /// # Panics
    /// If either precondition is violated.
    #[inline]
    pub fn spmm_into(&self, x_flat: &[F], feature_dim: usize, y_flat: &mut [F]) {
        if self.nnz().saturating_mul(feature_dim) >= Self::PAR_SPMM_WORK_THRESHOLD {
            self.spmm_into_parallel(x_flat, feature_dim, y_flat);
        } else {
            self.spmm_into_serial(x_flat, feature_dim, y_flat);
        }
    }

    /// Single-threaded `Y = A * X`. Contract as [`Self::spmm_into`].
    pub fn spmm_into_serial(&self, x_flat: &[F], feature_dim: usize, y_flat: &mut [F]) {
        self.assert_spmm_dims(x_flat, feature_dim, y_flat);
        if feature_dim == 0 {
            return;
        }

        for (i, y_row) in y_flat.chunks_mut(feature_dim).enumerate() {
            self.row_axpy(i, x_flat, feature_dim, y_row);
        }
    }

    /// Rayon `Y = A * X`, partitioned by output row block.
    /// Contract as [`Self::spmm_into`].
    pub fn spmm_into_parallel(&self, x_flat: &[F], feature_dim: usize, y_flat: &mut [F]) {
        self.assert_spmm_dims(x_flat, feature_dim, y_flat);
        if feature_dim == 0 || self.num_rows == 0 {
            return;
        }

        let chunk_rows = Self::row_chunk(self.num_rows);
        y_flat
            .par_chunks_mut(chunk_rows * feature_dim)
            .enumerate()
            .for_each(|(chunk_idx, block)| {
                let base = chunk_idx * chunk_rows;
                for (offset, y_row) in block.chunks_mut(feature_dim).enumerate() {
                    self.row_axpy(base + offset, x_flat, feature_dim, y_row);
                }
            });
    }

    /// Accumulate row `i` of `A * X` into `y_row`.
    ///
    /// Zeroes `y_row` first, so the "fully overwritten" postcondition holds
    /// per row without a separate clearing pass over the whole output — the
    /// row is already in cache at that point.
    ///
    /// The single definition of SpMM's inner loop, shared by both paths for
    /// the same reason as [`Self::row_dot`].
    #[inline(always)]
    fn row_axpy(&self, i: usize, x_flat: &[F], feature_dim: usize, y_row: &mut [F]) {
        for slot in y_row.iter_mut() {
            *slot = F::zero();
        }

        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];

        for k in start..end {
            let col = self.col_ind[k];
            let val = self.val[k];
            let x_row = &x_flat[col * feature_dim..(col + 1) * feature_dim];

            // Both operands are now `feature_dim`-length slices with a
            // compiler-visible equal length, which is what lets LLVM drop the
            // bounds checks and vectorise. The previous flat-index form
            // (`y_flat[y_off + f] += val * x_flat[x_off + f]`) could not be
            // proven in-bounds and kept a check per element.
            for (y, &xv) in y_row.iter_mut().zip(x_row.iter()) {
                *y += val * xv;
            }
        }
    }

    #[inline]
    fn assert_spmm_dims(&self, x_flat: &[F], feature_dim: usize, y_flat: &[F]) {
        assert_eq!(
            self.num_cols * feature_dim,
            x_flat.len(),
            "Dimension mismatch: X must be exactly num_cols * feature_dim"
        );
        assert_eq!(
            self.num_rows * feature_dim,
            y_flat.len(),
            "Dimension mismatch: Y must be exactly num_rows * feature_dim"
        );
    }
}

// ==========================================
// UTILITY: Universal Prefix Sum
// ==========================================
#[inline]
pub fn build_row_ptr(counts: &[usize]) -> (Vec<usize>, usize) {
    let dim = counts.len();
    let mut row_ptr = vec![0; dim + 1];
    let mut total = 0;
    for i in 0..dim {
        row_ptr[i] = total;
        total += counts[i];
    }
    row_ptr[dim] = total;
    (row_ptr, total)
}

// ---------------------------------------------------------------------------
// Sparse contract — additive. No field, signature or number above changes.
// ---------------------------------------------------------------------------

use crate::representations::sparse::{
    validate_csr_parts, Coalescing, SparseLayout, SparseShape, SparseTensorContract,
    SparseViolation, StoredOrder, ValidatedSparse,
};

impl<F: Real> SparseTensorContract for TensorCsr<F> {
    type Scalar = F;

    fn layout(&self) -> SparseLayout {
        SparseLayout::Csr
    }

    fn shape(&self) -> SparseShape {
        SparseShape::new(vec![self.num_rows, self.num_cols], self.col_ind.len())
    }

    /// Rows ascending, and **within a row the stored order is kept**. A CSR whose columns are
    /// unsorted inside a row is structurally valid; re-sorting it would change a float
    /// accumulation order, so this contract reports the order rather than imposing one.
    fn stored_order(&self) -> StoredOrder {
        StoredOrder::RowMajorRowsThenStored
    }

    /// Validate the CSR structure without touching it.
    ///
    /// Delegates to [`validate_csr_parts`], the single owner of these rules. Nothing is
    /// re-implemented here and nothing is re-implemented on the device side either: the WGPU
    /// backend's CSR upload calls the same function, so the two cannot drift.
    ///
    /// # Postconditions
    /// As [`validate_csr_parts`], plus: the returned proof borrows `self`, so it cannot be paired
    /// with a different tensor. Duplicate entries are **preserved**.
    ///
    /// # Errors
    /// The first [`SparseViolation`] found.
    fn validate(&self) -> Result<ValidatedSparse<'_, Self>, SparseViolation> {
        validate_csr_parts(
            self.num_rows,
            self.num_cols,
            &self.row_ptr,
            &self.col_ind,
            self.val.len(),
        )?;
        Ok(ValidatedSparse::bind(
            self,
            SparseLayout::Csr,
            self.shape(),
            self.stored_order(),
            Coalescing::Preserved,
        ))
    }
}

// ---------------------------------------------------------------------------
// APPROVED-CORE-EDIT: csr-construction (2026-10-20, user chat: "de van jogosultságod" -- "if we do
// not take this step we rewrite the CSR countless times").
//
// Additive: the constructions callers kept writing by hand -- a matrix written row by row (the
// stateful case: an accumulator, a merge), a matrix from its rows, a symmetric matrix from its
// diagonal and strict upper triangle, the transpose, the product, the diagonal. Nothing above changes. Every construction
// is deterministic and keeps the sparse contract's stored order: a row is not re-sorted unless the
// construction says so, and nothing is coalesced.
// ---------------------------------------------------------------------------

/// **A CSR matrix written row by row**: entries are pushed in stored order, [`Self::end_row`]
/// closes a row, [`Self::finish`] returns the matrix. For a producer that computes each row from
/// mutable state -- a sparse product's accumulator, a row merge -- without a buffer per row.
#[derive(Debug, Clone)]
pub struct CsrRowWriter<F> {
    matrix: TensorCsr<F>,
}

impl<F: Real> CsrRowWriter<F> {
    /// A writer for a matrix of `num_cols` columns, room reserved for `rows` rows and `nnz`
    /// entries.
    #[must_use]
    pub fn with_capacity(num_cols: usize, rows: usize, nnz: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        Self {
            matrix: TensorCsr {
                num_rows: 0,
                num_cols,
                row_ptr,
                col_ind: Vec::with_capacity(nnz),
                val: Vec::with_capacity(nnz),
            },
        }
    }

    /// Append an entry to the open row.
    ///
    /// # Panics
    /// If `col` is not below the column count.
    #[inline]
    pub fn push(&mut self, col: usize, value: F) {
        assert!(
            col < self.matrix.num_cols,
            "column {col} of a matrix with {} columns",
            self.matrix.num_cols
        );
        self.matrix.col_ind.push(col);
        self.matrix.val.push(value);
    }

    /// Append the entries `cols[k], vals[k]` to the open row, as two slice copies.
    ///
    /// # Panics
    /// If the slices differ in length, or a column is not below the column count.
    pub fn extend_from_slices(&mut self, cols: &[usize], vals: &[F]) {
        assert_eq!(cols.len(), vals.len(), "one value per column");
        assert!(
            cols.iter().all(|&c| c < self.matrix.num_cols),
            "a column of a matrix with {} columns is out of range",
            self.matrix.num_cols
        );
        self.matrix.col_ind.extend_from_slice(cols);
        self.matrix.val.extend_from_slice(vals);
    }

    /// Close the open row (an empty row when nothing was pushed).
    #[inline]
    pub fn end_row(&mut self) {
        self.matrix.row_ptr.push(self.matrix.col_ind.len());
        self.matrix.num_rows += 1;
    }

    /// The matrix of the closed rows; entries pushed after the last [`Self::end_row`] are dropped.
    #[must_use]
    pub fn finish(mut self) -> TensorCsr<F> {
        let closed = self.matrix.row_ptr[self.matrix.num_rows];
        self.matrix.col_ind.truncate(closed);
        self.matrix.val.truncate(closed);
        self.matrix
    }
}

impl<F: Real> TensorCsr<F> {
    /// **The matrix of `rows`** over `num_cols` columns, each row's entries in the order given --
    /// [`StoredOrder::RowMajorRowsThenStored`]: not sorted, not coalesced. A caller that needs
    /// ascending columns gives its rows sorted.
    ///
    /// # Panics
    /// If an entry's column is not below `num_cols`.
    pub fn from_rows<I, R>(rows: I, num_cols: usize) -> Self
    where
        I: IntoIterator<Item = R>,
        R: IntoIterator<Item = (usize, F)>,
    {
        let mut writer = CsrRowWriter::with_capacity(num_cols, 0, 0);
        for row in rows {
            for (c, v) in row {
                writer.push(c, v);
            }
            writer.end_row();
        }
        writer.finish()
    }

    /// **The symmetric matrix** with diagonal `diagonal` and strict upper triangle `upper` as
    /// `(row, column, value)`, `row < column`: every row holds its diagonal entry and its
    /// off-diagonal entries from both triangles, columns ascending (a stable sort, so repeated
    /// pairs keep their order in `upper`). The layout normal equations and the SPD solvers use.
    ///
    /// # Panics
    /// If an index of `upper` is not below `diagonal.len()`.
    pub fn from_symmetric(diagonal: &[F], upper: &[(u32, u32, F)]) -> Self {
        let n = diagonal.len();
        let mut rows: Vec<Vec<(usize, F)>> = diagonal
            .iter()
            .enumerate()
            .map(|(i, &d)| vec![(i, d)])
            .collect();
        for &(i, j, v) in upper {
            rows[i as usize].push((j as usize, v));
            rows[j as usize].push((i as usize, v));
        }
        for row in &mut rows {
            row.sort_by_key(|&(c, _)| c);
        }
        Self::from_rows(rows, n)
    }

    /// **The product `self · rhs`** by Gustavson's row-by-row accumulation: row `r` of the result
    /// sums `self[r, k] · rhs[k, ·]` over row `r`'s entries in stored order, and holds every column
    /// it touched -- in ascending order, exact zeros from cancellation included (a caller that wants
    /// them gone filters). Each entry's sum runs in a fixed order, so the product is deterministic.
    ///
    /// # Panics
    /// If `self.num_cols != rhs.num_rows`.
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.num_cols, rhs.num_rows,
            "a product needs the left operand's columns to be the right one's rows"
        );
        let mut values = vec![F::zero(); rhs.num_cols];
        let mut touched_flag = vec![false; rhs.num_cols];
        let mut touched: Vec<usize> = Vec::new();
        let mut out = CsrRowWriter::with_capacity(rhs.num_cols, self.num_rows, 0);
        for r in 0..self.num_rows {
            for at in self.row_ptr[r]..self.row_ptr[r + 1] {
                let (k, a) = (self.col_ind[at], self.val[at]);
                for bt in rhs.row_ptr[k]..rhs.row_ptr[k + 1] {
                    let c = rhs.col_ind[bt];
                    if !touched_flag[c] {
                        touched_flag[c] = true;
                        touched.push(c);
                    }
                    values[c] += a * rhs.val[bt];
                }
            }
            touched.sort_unstable();
            for &c in &touched {
                out.push(c, values[c]);
                values[c] = F::zero();
                touched_flag[c] = false;
            }
            touched.clear();
            out.end_row();
        }
        out.finish()
    }

    /// The diagonal: entry `i` sums row `i`'s stored entries in column `i` (zero when there is
    /// none), over `min(rows, columns)`.
    #[must_use]
    pub fn diagonal(&self) -> Vec<F> {
        (0..self.num_rows.min(self.num_cols))
            .map(|i| {
                (self.row_ptr[i]..self.row_ptr[i + 1])
                    .filter(|&at| self.col_ind[at] == i)
                    .fold(F::zero(), |sum, at| sum + self.val[at])
            })
            .collect()
    }

    /// **The transpose**: row `c` of the result lists column `c`'s entries in ascending row order
    /// of `self`, and within one row of `self` in its stored order (a stable counting sort).
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.num_cols];
        for &c in &self.col_ind {
            counts[c] += 1;
        }
        let (row_ptr, nnz) = build_row_ptr(&counts);
        let mut next = row_ptr.clone();
        let (mut col_ind, mut val) = (vec![0usize; nnz], vec![F::zero(); nnz]);
        for r in 0..self.num_rows {
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                let c = self.col_ind[k];
                col_ind[next[c]] = r;
                val[next[c]] = self.val[k];
                next[c] += 1;
            }
        }
        Self {
            num_rows: self.num_cols,
            num_cols: self.num_rows,
            row_ptr,
            col_ind,
            val,
        }
    }
}
