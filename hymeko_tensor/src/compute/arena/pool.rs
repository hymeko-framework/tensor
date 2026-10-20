//! The reuse policy itself: [`Pool`] and the two-phase request that lets a deployment allocate
//! outside whatever guards the pool.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use super::{AllocationKey, ArenaBackend, ArenaConfig, ArenaEpoch, ArenaStats, ReusePolicy};
use crate::compute::ComputeError;

type ResourceWithKey<B> = (
    <B as ArenaBackend>::Resource,
    AllocationKey<<B as ArenaBackend>::Class>,
);

/// One allocation the pool holds.
struct Slab<B: ArenaBackend> {
    resource: B::Resource,
    bytes: u64,
    class: B::Class,
    /// Epoch that must complete before this slab may be handed out again.
    safe_after: B::Epoch,
}

/// A request the free list has answered, before anything is allocated.
///
/// Fulfilled by [`Self::allocate`] outside the pool, then accounted by [`Pool::admit`].
#[must_use = "a request that is dropped loses the cached allocation it took"]
pub struct Request<B: ArenaBackend> {
    key: AllocationKey<B::Class>,
    want: u64,
    cached: Option<Slab<B>>,
}

impl<B: ArenaBackend> Request<B> {
    /// Fulfil the request: the cached allocation as it is, the cached allocation grown, or a new
    /// one. Touches only the backend.
    pub fn allocate(self, backend: &B) -> Allocation<B> {
        let (resource, class, source) = match self.cached {
            Some(slab) if slab.bytes >= self.want => (slab.resource, slab.class, Source::Reused),
            Some(mut short) => {
                backend.grow(&mut short.resource, self.want);
                let from = short.bytes;
                (short.resource, short.class, Source::Grown { from })
            }
            None => {
                let created = backend.create(self.key.bytes, self.key.class);
                (created, self.key.class, Source::Created)
            }
        };
        Allocation {
            resource,
            class,
            want: self.want,
            source,
        }
    }
}

/// A fulfilled request, which [`Pool::admit`] accounts for.
#[must_use = "an allocation that is not admitted is missing from the pool's statistics"]
pub struct Allocation<B: ArenaBackend> {
    resource: B::Resource,
    class: B::Class,
    want: u64,
    source: Source,
}

/// Where a fulfilled request's resource came from.
#[derive(Clone, Copy, Debug)]
enum Source {
    Reused,
    Grown { from: u64 },
    Created,
}

/// The reuse policy over one backend: free list, pending and retired sets, budgets, statistics.
///
/// Plain data behind `&mut self`, created per deployment, never global. How it is shared is the
/// deployment's choice (see the [`arena`](super) documentation).
///
/// # Invariants
///
/// * An allocation is in at most one of: the free list, pending, retired, or a lease.
/// * `stats.cached_bytes` / `cached_buffers` count the free list, `in_flight_bytes` the pending and
///   retired sets, and `active_bytes` the leases, exactly.
/// * A retired allocation re-enters the free list only once its epoch completes.
pub struct Pool<B: ArenaBackend> {
    config: ArenaConfig,
    free: HashMap<AllocationKey<B::Class>, Vec<Slab<B>>>,
    /// Released while a session was open: the epoch is not yet known.
    ///
    /// Each entry records WHICH session was open at release time, because a slab may be re-handed
    /// out inside that same session but not to another one (see [`Pool::take_pending`]). `None`
    /// means the owning session was ambiguous (two were open at once), which disables same-session
    /// reuse.
    pending: Vec<(Option<u64>, Slab<B>)>,
    /// Released with a known epoch, waiting for the watermark to reach it.
    retired: VecDeque<Slab<B>>,
    stats: ArenaStats,
    open_sessions: usize,
    /// The single open session, when exactly one is open.
    current_session: Option<u64>,
    next_session_id: u64,
}

impl<B: ArenaBackend> fmt::Debug for Pool<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pool")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .field("open_sessions", &self.open_sessions)
            .finish_non_exhaustive()
    }
}

impl<B: ArenaBackend> Pool<B> {
    #[must_use]
    pub fn new(config: ArenaConfig) -> Self {
        Self {
            config,
            free: HashMap::new(),
            pending: Vec::new(),
            retired: VecDeque::new(),
            stats: ArenaStats::default(),
            open_sessions: 0,
            current_session: None,
            next_session_id: 0,
        }
    }

    #[must_use]
    pub const fn config(&self) -> ArenaConfig {
        self.config
    }

    /// Current counters.
    #[must_use]
    pub const fn stats(&self) -> ArenaStats {
        self.stats
    }

    /// The key a request of `bytes` is filed under: its size class under
    /// [`ReusePolicy::SizeClass`], the size itself otherwise.
    ///
    /// # Errors
    ///
    /// [`ComputeError::IndexOverflow`] when a size class would exceed the largest representable
    /// power of two. An unchecked `next_power_of_two` wraps to zero there in release builds, which
    /// would hand out a zero-sized allocation for an enormous request.
    pub fn key_for(
        &self,
        bytes: u64,
        class: B::Class,
    ) -> Result<AllocationKey<B::Class>, ComputeError> {
        let bytes = match self.config.reuse_policy {
            ReusePolicy::ExactSize | ReusePolicy::BestFit => bytes,
            ReusePolicy::SizeClass => {
                bytes
                    .checked_next_power_of_two()
                    .ok_or(ComputeError::IndexOverflow {
                        what:
                            "allocation size class exceeds the largest representable power of two",
                        value: usize::try_from(bytes).unwrap_or(usize::MAX),
                    })?
            }
        };
        Ok(AllocationKey { bytes, class })
    }

    /// Whether a cached allocation already holds a request of `bytes`, without taking it.
    ///
    /// For a deployment that does something before a miss, such as polling for completions that
    /// would turn a retired allocation into a hit.
    ///
    /// # Errors
    ///
    /// As [`Self::key_for`].
    pub fn has_fitting(&self, bytes: u64, class: B::Class) -> Result<bool, ComputeError> {
        let key = self.key_for(bytes, class)?;
        Ok(match self.config.reuse_policy {
            ReusePolicy::ExactSize | ReusePolicy::SizeClass => self
                .free
                .get(&key)
                .is_some_and(|bucket| bucket.iter().any(|slab| slab.bytes >= bytes)),
            ReusePolicy::BestFit => self.smallest_fitting(key).is_some(),
        })
    }

    /// Take from the free list what a request of `bytes` will be served by.
    ///
    /// # Errors
    ///
    /// As [`Self::key_for`]. Nothing is taken on an error.
    pub fn request(&mut self, bytes: u64, class: B::Class) -> Result<Request<B>, ComputeError> {
        let key = self.key_for(bytes, class)?;
        let grows_instead = B::GROWS && self.config.reuse_policy == ReusePolicy::BestFit;
        let cached = match self.take_free(key, bytes) {
            Some(slab) => Some(slab),
            None if grows_instead => self.take_largest(class),
            None => None,
        };
        if let Some(slab) = &cached {
            self.stats.cached_bytes -= slab.bytes;
            self.stats.cached_buffers -= 1;
        }
        Ok(Request {
            key,
            want: bytes,
            cached,
        })
    }

    /// Account for a fulfilled request, and hand its resource to the tenant with the key its lease
    /// returns it under.
    ///
    /// # Postconditions
    ///
    /// The resource holds at least the requested bytes, checked in debug builds: a backend whose
    /// `create` or `grow` gives less is at fault.
    pub fn admit(&mut self, backend: &B, allocation: Allocation<B>) -> ResourceWithKey<B> {
        let bytes = backend.bytes(&allocation.resource);
        debug_assert!(
            bytes >= allocation.want,
            "the backend fulfilled a request of {} bytes with {bytes}",
            allocation.want
        );
        match allocation.source {
            Source::Reused => self.stats.reuse_hits += 1,
            Source::Grown { from } => {
                self.stats.reuse_misses += 1;
                self.stats.resources_grown += 1;
                self.stats.total_allocated_bytes += bytes.saturating_sub(from);
            }
            Source::Created => {
                self.stats.reuse_misses += 1;
                self.stats.buffers_created += 1;
                self.stats.total_allocated_bytes += bytes;
            }
        }
        self.stats.active_bytes += bytes;
        self.note_peak();
        (
            allocation.resource,
            AllocationKey {
                bytes,
                class: allocation.class,
            },
        )
    }

    /// [`Self::request`], [`Request::allocate`] and [`Self::admit`] in one call, for a pool that is
    /// not shared across threads.
    ///
    /// # Postconditions
    ///
    /// The returned resource is owned exclusively: no other lease can hold it, because the pool
    /// moved it out. Its contents are whatever its previous tenant left.
    ///
    /// # Errors
    ///
    /// As [`Self::key_for`].
    pub fn acquire(
        &mut self,
        backend: &B,
        bytes: u64,
        class: B::Class,
    ) -> Result<ResourceWithKey<B>, ComputeError> {
        let allocation = self.request(bytes, class)?.allocate(backend);
        Ok(self.admit(backend, allocation))
    }

    /// Take back a leased resource.
    ///
    /// `key` is the one [`Self::admit`] handed out with it. The resource is re-measured, so a tenant
    /// that grew it is accounted at its new size from here on. While a session is open the
    /// resource is held pending, because the session's unsubmitted commands may reference it;
    /// otherwise it retires until `latest_issued` completes.
    pub fn release(
        &mut self,
        backend: &B,
        resource: B::Resource,
        key: AllocationKey<B::Class>,
        latest_issued: B::Epoch,
    ) {
        self.stats.active_bytes -= key.bytes;
        let mut slab = Slab {
            bytes: backend.bytes(&resource),
            resource,
            class: key.class,
            safe_after: B::Epoch::NEVER,
        };
        self.stats.in_flight_bytes += slab.bytes;
        if self.open_sessions > 0 {
            self.pending.push((self.current_session, slab));
        } else {
            slab.safe_after = latest_issued;
            self.retired.push_back(slab);
        }
        self.note_peak();
    }

    /// Move retired allocations whose epoch has completed onto the free list.
    pub fn reclaim_completed(&mut self, watermark: B::Epoch) {
        self.reclaim_completed_budgeted(watermark, usize::MAX);
    }

    /// Check at most `max_checks` retired allocations and return the number reclaimed.
    ///
    /// Incomplete epochs rotate to the back, so repeated bounded calls do not
    /// starve ready resources behind them. Each allocation is checked at most
    /// once per call. Zero budget changes nothing. Pending sessions are untouched.
    ///
    /// The budget bounds epoch checks, not elapsed time: free-list insertion,
    /// eviction and backend destruction retain their existing costs and may
    /// allocate. Existing cache byte/buffer limits still apply to every insertion.
    ///
    /// # Postconditions
    ///
    /// Only completed resources leave retirement, and all byte/state counters
    /// remain exact. A reclaimed resource may be evicted instead of cached.
    pub fn reclaim_completed_budgeted(&mut self, watermark: B::Epoch, max_checks: usize) -> usize {
        let mut reclaimed = 0;
        for _ in 0..max_checks.min(self.retired.len()) {
            // The initial length bounds the loop; every non-reclaimed slab is
            // put back, so this branch cannot exhaust the queue prematurely.
            let Some(slab) = self.retired.pop_front() else {
                break;
            };
            if slab.safe_after.is_complete_at(watermark) {
                self.stats.in_flight_bytes -= slab.bytes;
                self.push_free(slab);
                reclaimed += 1;
            } else {
                self.retired.push_back(slab);
            }
        }
        reclaimed
    }

    /// Drop every cached allocation. Leased and in-flight ones are not the pool's to free.
    pub fn trim(&mut self) {
        let freed = self.stats.cached_buffers;
        self.free.clear();
        self.stats.cached_bytes = 0;
        self.stats.cached_buffers = 0;
        self.stats.evictions += freed as u64;
    }

    /// Open a session; releases from now until it closes are held pending. Returns its id.
    pub fn open_session(&mut self) -> u64 {
        self.open_sessions += 1;
        let id = self.next_session_id;
        self.next_session_id += 1;
        // Same-session reuse is only sound when exactly one session is open: two open sessions
        // may be submitted in either order, so a slab released under one could be written by the
        // other first.
        self.current_session = (self.open_sessions == 1).then_some(id);
        id
    }

    /// Close a session whose commands were submitted as `epoch`: everything pending retires until
    /// `epoch` completes.
    pub fn submit_session(&mut self, epoch: B::Epoch) {
        self.close_session();
        for (_owner, mut slab) in self.pending.drain(..) {
            slab.safe_after = epoch;
            self.retired.push_back(slab);
        }
    }

    /// Close a session whose commands were never submitted. They can never run, so everything
    /// pending needs no epoch: it is reclaimed at the next watermark, whatever it is.
    pub fn abandon_session(&mut self) {
        self.close_session();
        for (_owner, mut slab) in self.pending.drain(..) {
            slab.safe_after = B::Epoch::NEVER;
            self.retired.push_back(slab);
        }
    }

    /// A resource released earlier in `session`, while it is the only open session.
    ///
    /// Sound where [`Self::acquire`] is not, because inside one session every command that
    /// references a released resource was encoded before its release, and one command stream
    /// executes in order. The deployment that owns sessions states the full argument
    /// (`hymeko_wgpu::TrackedSession::acquire`).
    ///
    /// # Errors
    ///
    /// As [`Self::key_for`].
    pub fn take_pending(
        &mut self,
        session: u64,
        bytes: u64,
        class: B::Class,
    ) -> Result<Option<ResourceWithKey<B>>, ComputeError> {
        let key = self.key_for(bytes, class)?;
        let found = self.pending.iter().position(|(owner, slab)| {
            *owner == Some(session) && slab.class == class && slab.bytes >= key.bytes
        });
        Ok(found.map(|at| {
            let (_owner, slab) = self.pending.swap_remove(at);
            self.stats.in_flight_bytes -= slab.bytes;
            self.stats.active_bytes += slab.bytes;
            self.stats.reuse_hits += 1;
            self.note_peak();
            let key = AllocationKey {
                bytes: slab.bytes,
                class: slab.class,
            };
            (slab.resource, key)
        }))
    }

    fn close_session(&mut self) {
        self.open_sessions = self.open_sessions.saturating_sub(1);
        if self.open_sessions == 0 {
            self.current_session = None;
        }
    }

    fn note_peak(&mut self) {
        self.stats.peak_arena_bytes = self.stats.peak_arena_bytes.max(self.stats.device_bytes());
    }

    /// The smallest cached key of `key`'s class that holds `key.bytes`.
    fn smallest_fitting(&self, key: AllocationKey<B::Class>) -> Option<AllocationKey<B::Class>> {
        self.free
            .keys()
            .filter(|cached| cached.class == key.class && cached.bytes >= key.bytes)
            .min_by_key(|cached| cached.bytes)
            .copied()
    }

    /// Take a slab that serves a request of `want` bytes filed under `key`, by the policy.
    fn take_free(&mut self, key: AllocationKey<B::Class>, want: u64) -> Option<Slab<B>> {
        match self.config.reuse_policy {
            ReusePolicy::ExactSize => self.pop_from(key),
            ReusePolicy::SizeClass => {
                // The key's byte count is already class-rounded by `key_for`, so an exact bucket
                // lookup finds the class. The check that the slab really is large enough guards
                // against a rounding change making the two disagree.
                let bucket = self.free.get_mut(&key)?;
                let at = bucket.iter().position(|slab| slab.bytes >= want)?;
                let slab = bucket.swap_remove(at);
                if bucket.is_empty() {
                    self.free.remove(&key);
                }
                Some(slab)
            }
            ReusePolicy::BestFit => self.smallest_fitting(key).and_then(|at| self.pop_from(at)),
        }
    }

    /// The largest cached slab of `class`, for a backend that grows it.
    fn take_largest(&mut self, class: B::Class) -> Option<Slab<B>> {
        let largest = self
            .free
            .keys()
            .filter(|cached| cached.class == class)
            .max_by_key(|cached| cached.bytes)
            .copied()?;
        self.pop_from(largest)
    }

    /// The most recently cached slab under exactly `key`.
    fn pop_from(&mut self, key: AllocationKey<B::Class>) -> Option<Slab<B>> {
        let bucket = self.free.get_mut(&key)?;
        let slab = bucket.pop();
        if bucket.is_empty() {
            self.free.remove(&key);
        }
        slab
    }

    /// Put a slab on the free list, evicting to stay inside the budget.
    ///
    /// Eviction is deterministic: the incoming slab is dropped if it alone exceeds the budget,
    /// otherwise the smallest cached slabs go first. Never touches leased or in-flight memory:
    /// those are not the pool's to free.
    fn push_free(&mut self, slab: Slab<B>) {
        if self.exceeds_budget(slab.bytes) {
            self.evict_until_fits(slab.bytes);
        }
        // Re-check: eviction may not have been able to make room (the free list was already empty
        // and this one slab exceeds the budget). Dropping the incoming slab is correct: nothing
        // references it.
        if self.exceeds_budget(slab.bytes) {
            self.stats.evictions += 1;
            return;
        }
        self.stats.cached_bytes += slab.bytes;
        self.stats.cached_buffers += 1;
        let key = AllocationKey {
            bytes: slab.bytes,
            class: slab.class,
        };
        self.free.entry(key).or_default().push(slab);
    }

    /// Whether caching one more slab of `incoming` bytes would exceed either limit.
    fn exceeds_budget(&self, incoming: u64) -> bool {
        self.stats.cached_bytes.saturating_add(incoming) > self.config.max_cached_bytes
            || self.stats.cached_buffers.saturating_add(1) > self.config.max_cached_buffers
    }

    /// Drop cached slabs until `incoming` bytes would fit, or the free list is empty.
    fn evict_until_fits(&mut self, incoming: u64) {
        while self.exceeds_budget(incoming) && self.stats.cached_buffers > 0 {
            // Deterministic victim: the smallest key, its most recent slab. Choosing by key rather
            // than by HashMap order keeps eviction reproducible across runs.
            let victim = self
                .free
                .keys()
                .copied()
                .min_by_key(|key| (key.bytes, B::class_rank(key.class)));
            let Some(slab) = victim.and_then(|key| self.pop_from(key)) else {
                break;
            };
            self.stats.cached_bytes -= slab.bytes;
            self.stats.cached_buffers -= 1;
            self.stats.evictions += 1;
        }
    }
}
