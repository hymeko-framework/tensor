//! Compressed sparse-row adjacency + bitset helpers.


pub fn build_csr(edges: &[(u32, u32)], n_nodes: usize, directed: bool)
    -> (Vec<u32>, Vec<u32>)
{
    // Tightly-packed CSR. Each row is sorted + dedup'd; row_ptr[i+1] -
    // row_ptr[i] is exactly the unique-neighbour count for vertex i. No
    // u32::MAX sentinel; `neighbours` returns a clean slice in O(1).
    //
    // ``directed = false`` (default): store both (u,v) and (v,u).
    // ``directed = true``: store only (u,v) — out-edges only.
    let mut deg = vec![0u32; n_nodes];
    for &(u, v) in edges {
        deg[u as usize] += 1;
        if !directed { deg[v as usize] += 1; }
    }
    let mut raw_ptr = vec![0u32; n_nodes + 1];
    for i in 0..n_nodes {
        raw_ptr[i + 1] = raw_ptr[i] + deg[i];
    }
    let total = raw_ptr[n_nodes] as usize;
    let mut raw_col = vec![0u32; total];
    let mut cursor = raw_ptr.clone();
    for &(u, v) in edges {
        let pu = cursor[u as usize] as usize;
        raw_col[pu] = v;
        cursor[u as usize] += 1;
        if !directed {
            let pv = cursor[v as usize] as usize;
            raw_col[pv] = u;
            cursor[v as usize] += 1;
        }
    }
    let mut col_idx: Vec<u32> = Vec::with_capacity(total);
    let mut row_ptr: Vec<u32> = Vec::with_capacity(n_nodes + 1);
    row_ptr.push(0);
    for i in 0..n_nodes {
        let s = raw_ptr[i] as usize;
        let e = raw_ptr[i + 1] as usize;
        raw_col[s..e].sort_unstable();
        let mut prev: i64 = -1;
        for &v in &raw_col[s..e] {
            if v as i64 != prev {
                col_idx.push(v);
                prev = v as i64;
            }
        }
        row_ptr.push(col_idx.len() as u32);
    }
    (row_ptr, col_idx)
}

#[inline]
pub fn neighbours<'a>(row_ptr: &'a [u32], col_idx: &'a [u32], v: u32) -> &'a [u32] {
    let s = row_ptr[v as usize] as usize;
    let e = row_ptr[v as usize + 1] as usize;
    &col_idx[s..e]
}

#[inline]
pub fn has_edge(row_ptr: &[u32], col_idx: &[u32], u: u32, v: u32) -> bool {
    neighbours(row_ptr, col_idx, u).binary_search(&v).is_ok()
}

// ============================================================================
// Bitset visited (1 bit/vertex). Replaces Vec<bool> on the exact-DFS hot
// path — 8× smaller, fits in L1 for graphs up to ~500k nodes.
// ============================================================================

#[inline] pub fn bs_words(n: usize) -> usize { n.div_ceil(64) }
#[inline] pub fn bs_get(bits: &[u64], v: u32) -> bool {
    (bits[(v >> 6) as usize] >> (v & 63)) & 1 == 1
}
#[inline] pub fn bs_set(bits: &mut [u64], v: u32) {
    bits[(v >> 6) as usize] |= 1u64 << (v & 63);
}
#[inline] pub fn bs_clear(bits: &mut [u64], v: u32) {
    bits[(v >> 6) as usize] &= !(1u64 << (v & 63));
}
