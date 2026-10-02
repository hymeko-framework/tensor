use crate::common::Real;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CooEntry<F: Real> {
    pub k: usize,
    pub i: usize,
    pub j: usize,
    pub v: F,
}

pub struct CooSoa<F: Real> {
    pub num_slices: usize,
    pub dim_i: usize,
    pub dim_j: usize,
    pub k: Vec<usize>,
    pub i: Vec<usize>,
    pub j: Vec<usize>,
    pub v: Vec<F>,
}

/// Sparse 3D tensor in COO form (k, i, j, value).
/// Intended as an intermediate export format for JAX/PyTorch.
///
/// - `k` selects the slice (e.g., edge index / hyperedge id).
/// - `i`, `j` are row/col indices inside the slice.
/// - `v` is the weight/value (usually f32).
#[derive(Clone, Debug, Default)]
pub struct TensorCoo<F: Real> {
    pub num_slices: usize,
    pub dim_i: usize,
    pub dim_j: usize,
    pub entries: Vec<CooEntry<F>>,
}

impl<F: Real> TensorCoo<F> {
    pub fn new(num_slices: usize, dim_i: usize, dim_j: usize) -> Self {
        Self {
            num_slices,
            dim_i,
            dim_j,
            entries: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn with_meta(num_slices: usize, dim_i: usize, dim_j: usize) -> Self {
        Self {
            num_slices,
            dim_i,
            dim_j,
            entries: Vec::new(),
        }
    }

    /// Push one non-zero entry (k, i, j, value).
    #[inline(always)]
    pub fn push(&mut self, k: usize, i: usize, j: usize, v: F) {
        self.entries.push(CooEntry { k, i, j, v });
    }

    #[inline(always)]
    pub fn reserve(&mut self, n: usize) {
        self.entries.reserve(n);
    }

    pub fn entry(&self, t: usize) -> &CooEntry<F> {
        &self.entries[t]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CooEntry<F>> {
        self.entries.iter()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn into_soa(self) -> CooSoa<F> {
        let n = self.entries.len();
        let mut k = Vec::with_capacity(n);
        let mut i = Vec::with_capacity(n);
        let mut j = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);

        for e in self.entries {
            k.push(e.k);
            i.push(e.i);
            j.push(e.j);
            v.push(e.v);
        }

        CooSoa {
            num_slices: self.num_slices,
            dim_i: self.dim_i,
            dim_j: self.dim_j,
            k,
            i,
            j,
            v,
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse contract — additive. No field, signature or number above changes.
// ---------------------------------------------------------------------------

use crate::representations::sparse::{
    Coalescing, SparseLayout, SparseShape, SparseTensorContract, SparseViolation, StoredOrder,
    ValidatedSparse,
};

impl<F: Real> SparseTensorContract for TensorCoo<F> {
    type Scalar = F;

    fn layout(&self) -> SparseLayout {
        SparseLayout::Coo
    }

    /// **Rank 3.** The slice axis `k` is the content of this format, so it is carried as a real
    /// extent rather than folded into a matrix to match the CSR's arity.
    fn shape(&self) -> SparseShape {
        SparseShape::new(
            vec![self.num_slices, self.dim_i, self.dim_j],
            self.entries.len(),
        )
    }

    /// The order entries were pushed. Never re-sorted, by this contract or by validation.
    fn stored_order(&self) -> StoredOrder {
        StoredOrder::Insertion
    }

    /// Validate the coordinate structure without touching it.
    ///
    /// # Postconditions
    /// On `Ok`, every entry's `(k, i, j)` lies inside the declared extents. Duplicate coordinates
    /// are **preserved** and insertion order is **preserved** — this neither coalesces nor sorts.
    ///
    /// # Errors
    /// The first [`SparseViolation`] found, naming the entry index and the axis.
    fn validate(&self) -> Result<ValidatedSparse<'_, Self>, SparseViolation> {
        // A zero extent cannot hold an entry; declaring one and then storing entries is the
        // metadata/entry disagreement rather than a per-coordinate fault.
        if !self.entries.is_empty() && (self.num_slices == 0 || self.dim_i == 0 || self.dim_j == 0)
        {
            return Err(SparseViolation::MetadataInconsistent {
                detail: "entries stored under a zero extent",
            });
        }
        for (index, entry) in self.entries.iter().enumerate() {
            for (axis, (coordinate, extent)) in [
                (entry.k, self.num_slices),
                (entry.i, self.dim_i),
                (entry.j, self.dim_j),
            ]
            .into_iter()
            .enumerate()
            {
                if coordinate >= extent {
                    return Err(SparseViolation::CoordinateOutOfRange {
                        axis,
                        extent,
                        index,
                        actual: coordinate,
                    });
                }
            }
        }
        Ok(ValidatedSparse::bind(
            self,
            SparseLayout::Coo,
            self.shape(),
            self.stored_order(),
            Coalescing::Preserved,
        ))
    }
}
