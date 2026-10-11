// Generic-arena contract, approved 2026-09-27 (user chat); moved from hymeko_core with the tensor extraction.
//
// The reuse policy of `hymeko_wgpu`'s `TensorArena`, lifted out of the WGPU crate and made generic
// over the execution backend, so that the host and every device backend reuse allocations under one
// contract. `hymeko_tensor` gains no dependency: the backend is a type parameter, the host
// deployment uses `std` only, and nothing here names a `wgpu` type or dispatches dynamically.

//! `Pool<B>`: the reuse of allocations, one policy for every execution backend.
//!
//! # What this is for
//!
//! A step that allocates its large buffers afresh pays for them every time: in `wgpu::Buffer`
//! creation on a device, in page faults on the host. A pool hands out allocations and takes them
//! back, and nothing else. It does not know what an operation is, does not schedule, and does not
//! decide when to submit.
//!
//! # The three states an allocation can be in
//!
//! ```text
//!   free ──acquire──> leased ──release──> retired ──completion──> free
//!    ▲                                                              │
//!    └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! * **free**: owned by the pool, referenced by nothing, reusable at once.
//! * **leased**: moved out of the pool to its tenant, so the pool cannot hand it out twice.
//! * **retired**: released, but work already submitted may still read it. It carries the
//!   [`ArenaBackend::Epoch`] that must complete first. The host's epoch is `()`, complete at every
//!   watermark, so on the host this state lasts no longer than the release.
//!
//! A release while a session is open is held *pending*: the session's commands, not yet submitted,
//! may reference it, so its epoch is the submission's, which [`Pool::submit_session`] assigns.
//!
//! # What is generic and what is deployed
//!
//! [`Pool`] is the policy: free list, pending and retired sets, budgets with deterministic
//! eviction, and exact statistics. It is plain data behind `&mut self`. How it is shared, and so
//! what a lease is, belongs to the deployment:
//!
//! | deployment | sharing | lease |
//! |---|---|---|
//! | `hymeko_wgpu::TensorArena` | `Rc<RefCell<Pool<WgpuBackend>>>`, one thread | borrows the arena, tracked by sessions |
//! | [`HostArena`] | `Arc<Mutex<Pool<Host<R>>>>`, any thread | [`HostLease`], `'static`, returns on drop |
//!
//! A CUDA backend would implement [`ArenaBackend`] with a device allocation as its resource and a
//! stream-event sequence as its epoch. There is none: no workload has measured a need for one.
//!
//! # Allocation happens outside the pool
//!
//! [`Pool::request`] answers from the free list; [`Request::allocate`] creates or grows through the
//! backend without touching the pool; [`Pool::admit`] accounts for the result. A deployment that
//! locks the pool therefore never allocates under its lock. [`Pool::acquire`] is the three together,
//! for a pool that is not shared across threads.

mod host;
mod pool;

pub use host::{Host, HostArena, HostLease, HostResource};
pub use pool::{Allocation, Pool, Request};

use std::fmt;
use std::hash::Hash;

/// When a released allocation is safe to hand out again.
///
/// A device backend dates a release with the submission that may still read it. The host needs no
/// date, so `()` is complete at every watermark.
pub trait ArenaEpoch: Copy {
    /// Complete at every watermark: no submitted work references the allocation.
    const NEVER: Self;

    /// Whether work up to this epoch has completed once the backend reports `watermark`.
    fn is_complete_at(self, watermark: Self) -> bool;
}

impl ArenaEpoch for () {
    const NEVER: Self = ();

    fn is_complete_at(self, _watermark: Self) -> bool {
        true
    }
}

/// What a [`Pool`] needs from an execution backend: how allocations are told apart, and how one is
/// made, measured and, where the backend can, grown.
pub trait ArenaBackend {
    /// One pooled allocation.
    type Resource;
    /// What makes two allocations of equal size incompatible: usage flags on a device, which fix
    /// what a buffer may be bound as; nothing on the host.
    type Class: Copy + Eq + Hash + fmt::Debug;
    /// When a released resource is safe to hand out again.
    type Epoch: ArenaEpoch;

    /// Whether an allocation can grow in place. A device buffer's size is fixed at creation; a host
    /// `Vec` reallocates. Consulted only by [`ReusePolicy::BestFit`].
    const GROWS: bool = false;

    /// A total order on classes, so that eviction picks the same victim in every run.
    fn class_rank(class: Self::Class) -> u64;

    /// A new allocation of `class` that holds at least `bytes`.
    fn create(&self, bytes: u64, class: Self::Class) -> Self::Resource;

    /// The bytes `resource` holds now.
    fn bytes(&self, resource: &Self::Resource) -> u64;

    /// Grow `resource` in place to hold at least `bytes`. Called only when [`Self::GROWS`].
    fn grow(&self, resource: &mut Self::Resource, bytes: u64) {
        let _ = (resource, bytes);
    }
}

/// How a request is matched against cached allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReusePolicy {
    /// Reuse only an allocation of exactly the requested size.
    ///
    /// The default, and the right first choice when sizes repeat: every layer of a fixed feature
    /// dimension asks for the same size, so a layered chain reuses heavily and wastes nothing.
    #[default]
    ExactSize,
    /// Reuse an allocation from the same power-of-two size class.
    ///
    /// Trades internal fragmentation (up to 2x) for hit rate on ragged sizes.
    SizeClass,
    /// Reuse the smallest cached allocation of the class that holds the request.
    ///
    /// When none does and the backend [grows](ArenaBackend::GROWS), the largest cached allocation of
    /// the class is grown instead of a new one being created, so the pool never caches more
    /// allocations than were leased at once. A backend that cannot grow creates a new one.
    ///
    /// For requests whose sizes vary from call to call. Measured on Nagare's HSiKAN host arena
    /// (P41): first-fit reuse cost +26 % peak RSS over fresh allocation, best-fit +7.1 %.
    BestFit,
}

/// Budget and policy for a [`Pool`].
///
/// # Preconditions
///
/// Both limits bound the **free list only**. Leased and in-flight allocations are never evicted,
/// so work that needs more memory than the budget still runs; the pool simply stops caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaConfig {
    /// Ceiling on bytes held in the free list.
    pub max_cached_bytes: u64,
    /// Ceiling on the number of allocations held in the free list.
    pub max_cached_buffers: usize,
    pub reuse_policy: ReusePolicy,
}

impl ArenaConfig {
    /// No budget: the free list keeps everything returned to it.
    #[must_use]
    pub const fn unbounded(reuse_policy: ReusePolicy) -> Self {
        Self {
            max_cached_bytes: u64::MAX,
            max_cached_buffers: usize::MAX,
            reuse_policy,
        }
    }
}

impl Default for ArenaConfig {
    /// 512 MiB / 64 allocations, exact size: the device arena's budget.
    ///
    /// Sized against the measured WGPU workload rather than picked round: the 8-layer chain at
    /// 200 000 x 128 needs two live 102 MiB intermediates, so a 512 MiB cache holds a comfortable
    /// multiple of the working set. A host arena starts from [`Self::unbounded`] instead.
    fn default() -> Self {
        Self {
            max_cached_bytes: 512 * 1024 * 1024,
            max_cached_buffers: 64,
            reuse_policy: ReusePolicy::ExactSize,
        }
    }
}

/// An allocation's size and class: what the free list is indexed by, and what a lease hands back
/// with its resource.
///
/// The class is part of the key, not a compatibility test: two allocations of equal size but
/// different class are different resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationKey<C> {
    bytes: u64,
    class: C,
}

impl<C: Copy> AllocationKey<C> {
    #[must_use]
    pub const fn new(bytes: u64, class: C) -> Self {
        Self { bytes, class }
    }

    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn class(&self) -> C {
        self.class
    }
}

/// Counters describing what a pool has done. Every field is exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// Acquires served by a cached allocation as it was.
    pub reuse_hits: u64,
    /// Acquires that allocated: created a new allocation or grew a cached one.
    pub reuse_misses: u64,
    /// Allocations created over the pool's lifetime.
    pub buffers_created: u64,
    /// Acquires served by growing a cached allocation in place.
    pub resources_grown: u64,
    /// Allocations dropped because the budget was exceeded.
    pub evictions: u64,
    /// Bytes currently held by leases.
    pub active_bytes: u64,
    /// Bytes released but not yet known complete.
    pub in_flight_bytes: u64,
    /// Bytes in the free list.
    pub cached_bytes: u64,
    /// Allocations in the free list.
    pub cached_buffers: usize,
    /// Highest value of `active + in_flight + cached` observed.
    pub peak_arena_bytes: u64,
    /// Bytes the pool itself allocated: every creation, and every growth by its increase. A tenant
    /// that grows a leased resource is not counted here; its bytes enter the state totals when the
    /// resource is released.
    pub total_allocated_bytes: u64,
}

impl ArenaStats {
    /// Bytes the pool currently holds, in any state.
    #[must_use]
    pub const fn device_bytes(&self) -> u64 {
        self.active_bytes + self.in_flight_bytes + self.cached_bytes
    }

    /// Hit rate over all acquires; `None` before the first acquire.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.reuse_hits + self.reuse_misses;
        (total > 0).then(|| self.reuse_hits as f64 / total as f64)
    }
}
