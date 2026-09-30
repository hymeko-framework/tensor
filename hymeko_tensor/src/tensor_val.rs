use crate::aggregation::{agg_weight, AggCfg};
use crate::common::Real;

pub trait IncVal<F>: Clone + Send + Sync
where
    F: Real,
{
    fn zero() -> Self;
    /// default value when no explicit weights exist
    fn one() -> Self;

    /// multiply by scalar (arc weight, edge weight, etc.)
    fn scale(&self, k: F) -> Self;

    /// aggregate two values (same (e,n) key)
    fn agg(cfg: &AggCfg, a: &Self, b: &Self) -> Self;

    /// for degree / normalization: return a nonnegative scalar measure
    fn degree_mass(&self) -> F;

    /// Convert to f32 for final output (e.g., for GNN input)
    fn as_scalar(&self) -> F;
}

// Skalár eset: V = F
impl<F: Real> IncVal<F> for F {
    #[inline]
    fn zero() -> Self {
        F::zero()
    }
    #[inline]
    fn one() -> Self {
        F::one()
    }
    #[inline]
    fn scale(&self, k: F) -> Self {
        *self * k
    }
    #[inline]
    fn agg(cfg: &AggCfg, a: &Self, b: &Self) -> Self {
        agg_weight::<F>(cfg, *a, *b) // ezt is generikussá kell tenni (lásd lent)
    }
    #[inline]
    fn degree_mass(&self) -> F {
        self.abs()
    }
    #[inline]
    fn as_scalar(&self) -> F {
        *self
    }
}

pub trait EdgeWeight<V, F>: Clone + Send + Sync
where
    V: IncVal<F>,
    F: Real,
{
    fn one() -> Self; // “1”
    fn scale(&self, x: V) -> V; // ha kell: edge-súlyozás

    fn apply_to(&self, x: V) -> V;
}

#[derive(Clone, Copy, Default)]
pub struct EdgeWScalar<F: Real>(pub F);

impl<F: Real> EdgeWeight<F, F> for EdgeWScalar<F> {
    fn one() -> Self {
        Self(F::one())
    }
    fn scale(&self, x: F) -> F {
        self.0 * x
    }

    #[inline(always)]
    fn apply_to(&self, x: F) -> F {
        self.0 * x
    }
}

pub trait Dot<Rhs> {
    type Out;
    fn dot(self, rhs: Rhs) -> Self::Out;
}

// f32 · f32 -> f32
impl Dot<f32> for f32 {
    type Out = f32;
    #[inline(always)]
    fn dot(self, rhs: f32) -> f32 {
        self * rhs
    }
}
