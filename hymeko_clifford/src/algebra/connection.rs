//! A **discrete connection** on a graph: tree gauge fixing, cotree residuals, curvature.
//!
//! A connection assigns a rotor to every edge, transporting a frame from one endpoint to the other.
//! Its curvature is what a frame fails to return to after going round a closed loop — the
//! *holonomy*. This module computes that in one pass, in closed form, from the topology and the
//! connection alone.
//!
//! ```text
//! 1. gauge-fix by tree transport   R_root = 1,  R_node = g · R_parent
//! 2. cotree loop-closure residual  rho_e = R_b^-1 · g_{a→b} · R_a
//! 3. readout                       mean rotation angle of the residuals
//! ```
//!
//! # Why the cotree carries all of it
//!
//! On a spanning tree the edge rotors can be transformed to the identity — that is what step 1 does
//! — and a tree has no cycles, so a gauge in which the tree is flat costs nothing. Every remaining
//! edge closes exactly one fundamental cycle, and the residual there **is** that cycle's holonomy.
//! The curvature of the whole connection therefore lives on the `|E| - |V| + 1` cotree edges and
//! nowhere else, which is the graph's cyclomatic number and the count of independent loops.
//!
//! Two consequences are worth stating because they are what make this worth having:
//!
//! * The estimate is **exact**, not fitted. There is no iteration, no learning rate and no
//!   objective; the cost is `O(|E|)` and the answer is the holonomy.
//! * On a closed surface the product of the plaquette holonomies is the identity **by
//!   construction**, because every interior edge is traversed twice with opposite orientation. A
//!   divergence-free constraint expressed this way is not a constraint to enforce but an identity
//!   the representation already satisfies.
//!
//! # Scope
//!
//! Rotor algebra over an ordered combinatorial structure, and nothing else. The topology arrives as
//! plain slices — edge endpoints, a traversal order, a cotree list — so this crate stays free of any
//! graph type, mesh, dataset or model concept, on the same argument that brought
//! [`super::rotor_holonomy_forward`] here: an ordered product of unit quaternions is general rotor
//! algebra wherever the order comes from.
//!
//! # Provenance
//!
//! Transcribed 2026-08-31 from `nagare`'s `holonomy_estimator.rs`, where it was the measured
//! component behind finding F-HOLO-3 — a closed-form curvature readout that reached oracle-level
//! separation (AUROC 1.000) on a task where a strong MLP reached 0.626. The arithmetic is
//! unchanged; what is removed is the task: the wheel graph, the sampler and the classification
//! metric stay on the application side, and `tests/connection.rs` rebuilds a wheel of its own to
//! check the identity this module is supposed to satisfy.

use super::rotor::{quat_conjugate, quat_mul};

/// The identity rotor.
const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// Read quaternion `i` out of a flat 4-strided buffer.
#[inline]
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// The rotation angle of a unit rotor, in radians, on `[0, π]`.
///
/// Taken through `atan2` of the vector norm against `|w|` rather than `2·acos(w)`: near the identity
/// the latter loses precision to cancellation exactly where a curvature readout needs it most, and
/// the absolute value collapses the double cover so that `q` and `-q` — the same rotation — give the
/// same angle.
pub fn rotor_angle(q: [f32; 4]) -> f32 {
    let vector_norm = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    2.0 * vector_norm.atan2(q[0].abs())
}

/// One step of a spanning-tree traversal: `(node, edge, edge_points_parent_to_node)`.
///
/// The flag is not a convenience. An undirected tree edge is stored once with a fixed orientation,
/// and whether that orientation happens to point from the parent to the child decides whether the
/// rotor transports or its inverse does. Getting it wrong produces frames that are still unit
/// quaternions and still look plausible, so it is stated rather than inferred.
pub type TreeStep = (usize, usize, bool);

/// Step 1 — gauge-fix by transporting frames along a spanning tree from the root.
///
/// `steps` must list every non-root node exactly once, in an order where each node's parent has
/// already been visited (breadth- or depth-first from the root, which is node `0`).
///
/// # Preconditions
/// `edge_q.len() == edges.len() * 4`; every entry of `edge_q` is a unit quaternion.
///
/// # Postconditions
/// The root's frame is the identity, and every frame is a product of unit quaternions.
///
/// # Panics
/// If `edge_q` has the wrong length, or a step names a node or edge that does not exist.
pub fn tree_gauge_frames(
    node_count: usize,
    edges: &[(u32, u32)],
    steps: &[TreeStep],
    edge_q: &[f32],
) -> Vec<f32> {
    assert_eq!(
        edge_q.len(),
        edges.len() * 4,
        "edge_q must hold one quaternion per edge"
    );
    let mut frames = vec![0.0f32; node_count * 4];
    frames[0..4].copy_from_slice(&IDENT);
    for &(node, edge, forward) in steps {
        let (a, b) = edges[edge];
        let rotor = q_at(edge_q, edge);
        // the parent is whichever endpoint is not this node; a reversed edge transports by its
        // inverse
        let (parent, transport) = if forward {
            (a as usize, rotor)
        } else {
            (b as usize, quat_conjugate(rotor))
        };
        debug_assert!(
            (forward && b as usize == node) || (!forward && a as usize == node),
            "tree step orientation disagrees with the edge it names"
        );
        let transported = quat_mul(transport, q_at(&frames, parent));
        frames[node * 4..node * 4 + 4].copy_from_slice(&transported);
    }
    frames
}

/// Step 2 — the loop-closure residual on each cotree edge, `rho = R_b^-1 · g_{a→b} · R_a`.
///
/// One residual per fundamental cycle, flat `(cotree.len(), 4)`. Identity exactly where that cycle
/// is flat, so the deviation from the identity **is** the cycle's holonomy in the tree gauge.
///
/// # Preconditions
/// `frames.len() == node_count * 4` and `edge_q.len() == edges.len() * 4`.
pub fn cotree_residuals(
    edges: &[(u32, u32)],
    cotree: &[usize],
    edge_q: &[f32],
    frames: &[f32],
) -> Vec<f32> {
    assert_eq!(
        edge_q.len(),
        edges.len() * 4,
        "edge_q must hold one quaternion per edge"
    );
    let mut residuals = vec![0.0f32; cotree.len() * 4];
    for (k, &edge) in cotree.iter().enumerate() {
        let (a, b) = edges[edge];
        let rotor = q_at(edge_q, edge);
        let frame_a = q_at(frames, a as usize);
        let frame_b = q_at(frames, b as usize);
        let rho = quat_mul(quat_conjugate(frame_b), quat_mul(rotor, frame_a));
        residuals[k * 4..k * 4 + 4].copy_from_slice(&rho);
    }
    residuals
}

/// Step 3 — the mean rotation angle over the residuals: zero on a flat connection, positive on a
/// curved one.
///
/// An empty set of cycles has no curvature to report and gives zero, which is the right answer for
/// a tree: a graph with no independent loop is flat for want of anywhere to be curved.
pub fn curvature_energy(residuals: &[f32]) -> f32 {
    let count = residuals.len() / 4;
    if count == 0 {
        return 0.0;
    }
    let total: f32 = (0..count).map(|k| rotor_angle(q_at(residuals, k))).sum();
    total / count as f32
}

/// The three steps in one call: gauge-fix, close the cotree loops, read the mean angle.
///
/// `O(|E|)`, one pass, closed form.
pub fn closed_form_curvature(
    node_count: usize,
    edges: &[(u32, u32)],
    steps: &[TreeStep],
    cotree: &[usize],
    edge_q: &[f32],
) -> f32 {
    let frames = tree_gauge_frames(node_count, edges, steps, edge_q);
    let residuals = cotree_residuals(edges, cotree, edge_q, &frames);
    curvature_energy(&residuals)
}
