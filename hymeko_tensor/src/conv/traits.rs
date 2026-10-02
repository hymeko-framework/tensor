use crate::common::Real;
use crate::tensor_val::IncVal;

/// Unifying trait for all hypergraph convolution layers.
///
/// The `forward` method takes node features X (flattened: n × d)
/// and produces output features X' (flattened: n × d_out).
///
/// `params` holds learnable weight matrices as flat arrays.
pub trait HypergraphConv<F: Real> {
    /// Forward pass: X (n × d_in) → X' (n × d_out)
    fn forward(&self, x_in: &[F], d_in: usize, x_out: &mut [F], d_out: usize);

    /// Number of learnable parameters
    fn num_params(&self) -> usize;

    /// Mutable access to parameter slice (for optimizer updates)
    fn params_mut(&mut self) -> &mut [F];

    /// Immutable access to parameter slice
    fn params(&self) -> &[F];
}

/// Degree vector computation modes.
#[derive(Clone, Copy, Debug)]
pub enum DegreeMode {
    /// d(v) = Σ_e |b(v,e)|  (standard vertex degree)
    VertexSum,
    /// d(e) = Σ_v |b(v,e)|  (hyperedge degree = cardinality)
    EdgeSum,
    /// d(v) = Σ_e b(v,e)²   (squared incidence, used in clique_diag)
    VertexSquaredSum,
}

/// Compute a degree vector from a view's edge→node CSR, generic over the stored
/// incidence-coordinate type.
///
/// M2c.1 (`APPROVED-CORE-EDIT: dense-view-index-compute-degree`): the degree
/// arithmetic exists **once** here. `idx` is a statically-dispatched projection
/// from the stored coordinate to its `usize` array index — a canonical `NodeId`
/// (`|n| n.raw()`) or a `hymeko_hnn` dense view
/// coordinate (`|d| d.as_usize()`). The closure is monomorphised (no
/// `Box<dyn Trait>`, no temporary `Vec<NodeId>`/`Vec<usize>`), and the numerical
/// accumulation order is exactly the pre-M2c.1 order.
#[inline]
pub fn compute_degree_by_index<F, V, I, P>(
    edge_offsets: &[usize],
    flat_edge_nodes: &[I],
    flat_edge_w: &[V],
    num_nodes: usize,
    num_edges: usize,
    idx: P,
    mode: DegreeMode,
) -> Vec<F>
where
    F: Real,
    V: IncVal<F>,
    I: Copy,
    P: Fn(I) -> usize,
{
    match mode {
        DegreeMode::VertexSum => {
            let mut deg = vec![F::zero(); num_nodes];
            for e in 0..num_edges {
                let s = edge_offsets[e];
                let eend = edge_offsets[e + 1];
                for p in s..eend {
                    let v = idx(flat_edge_nodes[p]);
                    deg[v] += flat_edge_w[p].degree_mass();
                }
            }
            deg
        }
        DegreeMode::EdgeSum => {
            let mut deg = vec![F::zero(); num_edges];
            for e in 0..num_edges {
                let s = edge_offsets[e];
                let eend = edge_offsets[e + 1];
                for weight in &flat_edge_w[s..eend] {
                    deg[e] += weight.degree_mass();
                }
            }
            deg
        }
        DegreeMode::VertexSquaredSum => {
            let mut deg = vec![F::zero(); num_nodes];
            for e in 0..num_edges {
                let s = edge_offsets[e];
                let eend = edge_offsets[e + 1];
                for p in s..eend {
                    let v = idx(flat_edge_nodes[p]);
                    let w = flat_edge_w[p].degree_mass();
                    deg[v] += w * w;
                }
            }
            deg
        }
    }
}

/// Inverse square root of degree: D^{-1/2}
#[inline]
pub fn inv_sqrt_degree<F: Real>(deg: &[F]) -> Vec<F> {
    deg.iter()
        .map(|&d| {
            if d > F::zero() {
                F::one() / d.sqrt()
            } else {
                F::zero()
            }
        })
        .collect()
}
