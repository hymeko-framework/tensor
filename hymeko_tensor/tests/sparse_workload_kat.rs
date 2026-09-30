//! Frozen-identity KAT for the deterministic sparse-CSR workload fixture (P1
//! benchmark-harness consolidation).
//!
//! # Why this test exists
//!
//! The fixture (`Lcg` + `DegreeShape` + `SparseSpec`) existed byte-for-byte in
//! two places — `hymeko_core/benches/sparse_fixture.rs` and
//! `hymeko_wgpu/src/testing.rs` — and P1 consolidates it into a single owner,
//! `hymeko_tensor::sparse_workload`. This KAT pins the *exact bytes* the
//! generator emits for the three presets so the relocation is provably
//! bit-identical: it is committed first against the OLD bench source (via
//! `#[path]`), then rebound to the owner module after the move with the **same**
//! digest constants. A single changed byte anywhere in `row_ptr` / `col_ind` /
//! `val` / `dense_vector` / `dense_block` flips the BLAKE3 digest.
//!
//! Digest domain: `HYMEKO-P1-SPARSE-KAT-v1` then, per preset,
//! `rows, cols, row_ptr[], col_ind[], val-bits[], dense_vector-bits[],
//! (fd, dense_block-bits[]) for fd in {8,32,128}` — all little-endian, `u64`
//! for indices, `f32::to_bits` for weights.

// Bound to the consolidated owner module (commit 2). The digest constants below
// are the ones frozen in commit 1 against the OLD benches/sparse_fixture.rs, so a
// green run here proves the relocation is byte-identical.
use hymeko_tensor::sparse_workload::SparseSpec;

use blake3::Hasher;

/// Canonical `(rows, cols, nnz, hex-digest)` over a spec's full materialisation.
fn identity(spec: &SparseSpec) -> (usize, usize, usize, String) {
    let csr = spec.build();
    let mut h = Hasher::new();
    h.update(b"HYMEKO-P1-SPARSE-KAT-v1");
    h.update(&(csr.num_rows as u64).to_le_bytes());
    h.update(&(csr.num_cols as u64).to_le_bytes());
    for &p in &csr.row_ptr {
        h.update(&(p as u64).to_le_bytes());
    }
    for &c in &csr.col_ind {
        h.update(&(c as u64).to_le_bytes());
    }
    for &v in &csr.val {
        h.update(&v.to_bits().to_le_bytes());
    }
    for &v in &spec.dense_vector() {
        h.update(&v.to_bits().to_le_bytes());
    }
    for &fd in &[8usize, 32, 128] {
        h.update(&(fd as u64).to_le_bytes());
        for &v in &spec.dense_block(fd) {
            h.update(&v.to_bits().to_le_bytes());
        }
    }
    (
        csr.num_rows,
        csr.num_cols,
        csr.col_ind.len(),
        h.finalize().to_hex().to_string(),
    )
}

/// Frozen golden identity per preset. `(rows, cols, nnz, digest)` captured from
/// the byte-identical pre-consolidation fixture on 2026-09-01; any drift in the
/// generator (or the relocation in commit 2) flips the digest and fails here.
fn assert_identity(spec: &SparseSpec, rows: usize, cols: usize, nnz: usize, digest: &str) {
    let got = identity(spec);
    assert_eq!(
        got,
        (rows, cols, nnz, digest.to_string()),
        "sparse-workload fixture identity drifted"
    );
}

#[test]
fn small_identity_is_frozen() {
    assert_identity(
        &SparseSpec::small(),
        1_000,
        1_000,
        7_967,
        "d29a53e99ef4b7ba91a463ddce424d403760666a789ef081b3b64d5633a818ce",
    );
}

#[test]
fn medium_identity_is_frozen() {
    assert_identity(
        &SparseSpec::medium(),
        50_000,
        50_000,
        799_877,
        "311abed61138e62f23c2968f5436469816ff455a0ce49620d32edc554f766ef6",
    );
}

#[test]
fn large_identity_is_frozen() {
    assert_identity(
        &SparseSpec::large(),
        200_000,
        200_000,
        5_442_303,
        "7d764dea1f397ba9e2eca89e4faae941b17496490b41bb1ed96caa4db259c883",
    );
}
