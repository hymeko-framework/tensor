use crate::common::Real;
use crate::representations::tensor_csr::{TensorCsr, TensorCsrBuilder};

impl<F: Real> TensorCsrBuilder<F> {
    /// O(N * d log d) time complexity where d is the average degree of a row.
    pub fn finalize_coalesced(self) -> TensorCsr<F> {
        if self.vals.is_empty() {
            return TensorCsr {
                num_rows: self.dim_i,
                num_cols: self.dim_j,
                row_ptr: vec![0; self.dim_i + 1],
                col_ind: Vec::new(),
                val: Vec::new(),
            };
        }

        let mut new_row_ptr = vec![0; self.dim_i + 1];
        let mut new_col_ind = Vec::with_capacity(self.cols.len());
        let mut new_val = Vec::with_capacity(self.vals.len());
        // Temporary buffer to hold column-value pairs for the current row
        let mut row_entries: Vec<(usize, F)> = Vec::new();

        // Process row by row using the uncoalesced row_ptr boundaries
        for i in 0..self.dim_i {
            let start = self.uncoalesced_row_ptr[i];
            let end = self.uncoalesced_row_ptr[i + 1];

            // If the row is completely empty, carry forward the previous pointer
            if start == end {
                new_row_ptr[i + 1] = new_row_ptr[i];
                continue;
            }

            // Extract just the columns and values for this specific row
            row_entries.clear();
            row_entries.extend(
                self.cols[start..end]
                    .iter()
                    .copied()
                    .zip(self.vals[start..end].iter().copied()),
            );
            // Sort locally by column index
            row_entries.sort_unstable_by_key(|a| a.0);

            // Coalesce duplicates strictly within this row
            let mut current_col = row_entries[0].0;
            let mut current_val = row_entries[0].1;
            let mut nnz_in_row = 0;

            for &(c, v) in &row_entries[1..] {
                if c == current_col {
                    current_val += v; // Combine duplicates
                } else {
                    new_col_ind.push(current_col);
                    new_val.push(current_val);
                    nnz_in_row += 1;

                    current_col = c;
                    current_val = v;
                }
            }

            // Push the final element of the row
            new_col_ind.push(current_col);
            new_val.push(current_val);
            nnz_in_row += 1;

            // Compute the correct continuous row pointer
            new_row_ptr[i + 1] = new_row_ptr[i] + nnz_in_row;
        }

        TensorCsr {
            num_rows: self.dim_i,
            num_cols: self.dim_j,
            row_ptr: new_row_ptr,
            col_ind: new_col_ind,
            val: new_val,
        }
    }
}
