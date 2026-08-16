//! Cycle / walk pruning trait.
//!
//! The [`CyclePruner`] trait is consulted by the DFS enumerator at
//! two points:
//!
//! 1. **partial-path check** ([`CyclePruner::extend_ok`]) — at every
//!    step of the DFS, before pushing a new vertex onto the path,
//!    the pruner can veto the extension.  This is the *Friedler
//!    pre-check* equivalent: structural infeasibility detected
//!    early prevents materialisation of the dead branch.
//! 2. **closed-cycle emit check** ([`CyclePruner::emit_ok`]) — once
//!    a candidate cycle is closed (or, for walks, once the path
//!    reaches the desired length), the pruner gets the full
//!    sequence of edge signs and can apply tests that require the
//!    completed structure (e.g. cycle-balance product, total
//!    negative count for Davis weak balance).
//!
//! The two-level API mirrors how Friedler's accelerated
//! branch-and-bound works on P-graphs: cheap structural checks
//! during the search, expensive verification once a candidate is
//! ready.

/// Decision returned by a pruner at any check point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrunerDecision {
    /// Continue the DFS / emit the cycle.
    Accept,
    /// Reject this branch / cycle but continue the search.
    Reject,
}

impl PrunerDecision {
    /// `Accept` iff `b` is `true`.
    #[inline]
    pub fn from_bool(b: bool) -> PrunerDecision {
        if b {
            PrunerDecision::Accept
        } else {
            PrunerDecision::Reject
        }
    }

    /// Is this an accept decision?
    #[inline]
    pub fn is_accept(self) -> bool {
        matches!(self, PrunerDecision::Accept)
    }
}

/// Pluggable cycle / walk pruner.
///
/// Methods are pure (no `&mut self`) so the same pruner instance
/// can be shared across rayon worker threads without locking.
pub trait CyclePruner: Sync + Send {
    /// May the DFS extend the partial path `path` by adding `next`?
    ///
    /// Default implementation accepts any extension; override for
    /// structural pre-checks (e.g. bipartite alternation, P-graph
    /// kind invariants, k-color rainbow constraint).
    ///
    /// `path` does **not** yet contain `next`; the caller will push
    /// `next` only if this returns [`PrunerDecision::Accept`].
    #[allow(unused_variables)]
    #[inline]
    fn extend_ok(&self, path: &[u32], next: u32) -> PrunerDecision {
        PrunerDecision::Accept
    }

    /// May the closed candidate cycle `cycle` be emitted?
    ///
    /// `cycle` is the canonical-form vertex sequence (length $k$).
    /// `edge_signs` parallels the cycle's edges in canonical order:
    /// `edge_signs[i]` is the sign of edge `(cycle[i], cycle[(i+1) % k])`.
    /// Length of `edge_signs` equals `cycle.len()` for closed cycles
    /// and `cycle.len() - 1` for open walks (passed via the same
    /// trait when an enumerator calls into this from the walk path).
    ///
    /// Default implementation accepts any cycle; override for
    /// post-completion balance / parity / sign-product tests.
    #[allow(unused_variables)]
    #[inline]
    fn emit_ok(&self, cycle: &[u32], edge_signs: &[i8]) -> PrunerDecision {
        PrunerDecision::Accept
    }
}

/// Trivial pruner — accepts everything.  Equivalent to running the
/// enumerator without pruning.  Default value when no domain-specific
/// pruner is supplied.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpPruner;

impl CyclePruner for NoOpPruner {}

// ─── Instrumentation ──────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

/// Counter pack for one pruner's call/reject totals.  All counters
/// use `Relaxed` ordering — they are diagnostic, not synchronising.
#[derive(Debug, Default)]
pub struct PrunerStats {
    /// Total `extend_ok` invocations.
    pub extend_calls: AtomicU64,
    /// `extend_ok` invocations that returned [`PrunerDecision::Reject`].
    pub extend_rejects: AtomicU64,
    /// Total `emit_ok` invocations.
    pub emit_calls: AtomicU64,
    /// `emit_ok` invocations that returned [`PrunerDecision::Reject`].
    pub emit_rejects: AtomicU64,
}

impl PrunerStats {
    /// Snapshot the four counters as a plain tuple — useful at the
    /// end of a benchmark when the [`AtomicU64`] container is no
    /// longer needed.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            extend_calls: self.extend_calls.load(Ordering::Relaxed),
            extend_rejects: self.extend_rejects.load(Ordering::Relaxed),
            emit_calls: self.emit_calls.load(Ordering::Relaxed),
            emit_rejects: self.emit_rejects.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of [`PrunerStats`] at one moment.
#[derive(Debug, Default, Clone, Copy)]
pub struct StatsSnapshot {
    /// Total `extend_ok` invocations.
    pub extend_calls: u64,
    /// `extend_ok` invocations that returned [`PrunerDecision::Reject`].
    pub extend_rejects: u64,
    /// Total `emit_ok` invocations.
    pub emit_calls: u64,
    /// `emit_ok` invocations that returned [`PrunerDecision::Reject`].
    pub emit_rejects: u64,
}

impl StatsSnapshot {
    /// `extend_rejects / extend_calls` (or 0 when no calls).
    pub fn extend_reject_rate(&self) -> f64 {
        if self.extend_calls == 0 {
            0.0
        } else {
            self.extend_rejects as f64 / self.extend_calls as f64
        }
    }
    /// `emit_rejects / emit_calls` (or 0 when no calls).
    pub fn emit_reject_rate(&self) -> f64 {
        if self.emit_calls == 0 {
            0.0
        } else {
            self.emit_rejects as f64 / self.emit_calls as f64
        }
    }
}

/// Wrap any pruner to count its decisions.  The wrapped pruner's
/// `extend_ok` / `emit_ok` semantics are unchanged; the counts
/// attached to [`CountingPruner::stats`] are the only side effect.
#[derive(Debug, Default)]
pub struct CountingPruner<P: CyclePruner> {
    inner: P,
    /// Live counters; safe to read from any thread (Relaxed).
    pub stats: PrunerStats,
}

impl<P: CyclePruner> CountingPruner<P> {
    /// Wrap `inner`. Stats start at zero.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            stats: PrunerStats::default(),
        }
    }

    /// Borrow the wrapped pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: CyclePruner> CyclePruner for CountingPruner<P> {
    fn extend_ok(&self, path: &[u32], next: u32) -> PrunerDecision {
        self.stats.extend_calls.fetch_add(1, Ordering::Relaxed);
        let d = self.inner.extend_ok(path, next);
        if d == PrunerDecision::Reject {
            self.stats.extend_rejects.fetch_add(1, Ordering::Relaxed);
        }
        d
    }

    fn emit_ok(&self, cycle: &[u32], edge_signs: &[i8]) -> PrunerDecision {
        self.stats.emit_calls.fetch_add(1, Ordering::Relaxed);
        let d = self.inner.emit_ok(cycle, edge_signs);
        if d == PrunerDecision::Reject {
            self.stats.emit_rejects.fetch_add(1, Ordering::Relaxed);
        }
        d
    }
}

/// AND-composition of a named chain of pruners.  At every check
/// point each child pruner is consulted in declaration order; the
/// first to return [`PrunerDecision::Reject`] short-circuits *and*
/// receives the rejection credit. Per-child stats are exposed via
/// [`CompositePruner::child_stats`].
#[derive(Default)]
pub struct CompositePruner {
    /// `(name, pruner)` pairs in evaluation order.
    pub children: Vec<(String, Box<dyn CyclePruner>)>,
    /// Per-child stats array, parallel to `children`.
    pub stats: Vec<PrunerStats>,
}

impl std::fmt::Debug for CompositePruner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositePruner")
            .field(
                "children",
                &self.children.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .field("stats", &self.stats)
            .finish()
    }
}

impl CompositePruner {
    /// Empty composite — accepts everything until children are added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a named pruner to the chain.
    pub fn with(mut self, name: impl Into<String>, p: Box<dyn CyclePruner>) -> Self {
        self.children.push((name.into(), p));
        self.stats.push(PrunerStats::default());
        self
    }

    /// Snapshot stats by child name.
    pub fn child_stats(&self) -> Vec<(String, StatsSnapshot)> {
        self.children
            .iter()
            .zip(self.stats.iter())
            .map(|((name, _), s)| (name.clone(), s.snapshot()))
            .collect()
    }
}

impl CyclePruner for CompositePruner {
    fn extend_ok(&self, path: &[u32], next: u32) -> PrunerDecision {
        for (i, (_, p)) in self.children.iter().enumerate() {
            self.stats[i].extend_calls.fetch_add(1, Ordering::Relaxed);
            let d = p.extend_ok(path, next);
            if d == PrunerDecision::Reject {
                self.stats[i].extend_rejects.fetch_add(1, Ordering::Relaxed);
                return PrunerDecision::Reject;
            }
        }
        PrunerDecision::Accept
    }

    fn emit_ok(&self, cycle: &[u32], edge_signs: &[i8]) -> PrunerDecision {
        for (i, (_, p)) in self.children.iter().enumerate() {
            self.stats[i].emit_calls.fetch_add(1, Ordering::Relaxed);
            let d = p.emit_ok(cycle, edge_signs);
            if d == PrunerDecision::Reject {
                self.stats[i].emit_rejects.fetch_add(1, Ordering::Relaxed);
                return PrunerDecision::Reject;
            }
        }
        PrunerDecision::Accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OnlyEvenLength;
    impl CyclePruner for OnlyEvenLength {
        fn emit_ok(&self, cycle: &[u32], _signs: &[i8]) -> PrunerDecision {
            PrunerDecision::from_bool(cycle.len().is_multiple_of(2))
        }
    }

    #[test]
    fn pruner_emit_filters_by_length() {
        let p = OnlyEvenLength;
        assert_eq!(p.emit_ok(&[0, 1, 2, 3], &[1; 4]), PrunerDecision::Accept);
        assert_eq!(p.emit_ok(&[0, 1, 2], &[1; 3]), PrunerDecision::Reject);
    }

    #[test]
    fn counting_pruner_tracks_calls_and_rejects() {
        let p = CountingPruner::new(OnlyEvenLength);
        // 2 accepts on even, 2 rejects on odd.
        let _ = p.emit_ok(&[0, 1, 2, 3], &[1; 4]);
        let _ = p.emit_ok(&[0, 1, 2], &[1; 3]);
        let _ = p.emit_ok(&[0, 1], &[1; 2]);
        let _ = p.emit_ok(&[0], &[]);
        let snap = p.stats.snapshot();
        assert_eq!(snap.emit_calls, 4);
        assert_eq!(snap.emit_rejects, 2);
    }

    #[test]
    fn composite_pruner_attributes_rejects_to_first_failer() {
        // child A rejects everything via emit; child B accepts.
        struct AlwaysRejectEmit;
        impl CyclePruner for AlwaysRejectEmit {
            fn emit_ok(&self, _: &[u32], _: &[i8]) -> PrunerDecision {
                PrunerDecision::Reject
            }
        }
        let p = CompositePruner::new()
            .with("A_reject", Box::new(AlwaysRejectEmit))
            .with("B_accept", Box::new(NoOpPruner));
        let _ = p.emit_ok(&[0, 1, 2], &[1; 3]);
        let _ = p.emit_ok(&[0, 1, 2, 3], &[1; 4]);
        let s = p.child_stats();
        assert_eq!(s[0].0, "A_reject");
        assert_eq!(s[0].1.emit_rejects, 2);
        // B was never reached because A short-circuited.
        assert_eq!(s[1].1.emit_calls, 0);
    }
}
