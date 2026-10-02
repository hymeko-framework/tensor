//! Generic-arena contract (approved 2026-09-27; moved from hymeko_core with the tensor extraction). Tests of
//! `hymeko_tensor::compute::arena`: the generic `Pool<B>` over a deterministic test device, and
//! the host deployment `HostArena`.

use std::cell::Cell;

use hymeko_tensor::compute::ComputeError;
use hymeko_tensor::compute::arena::{
    AllocationKey, ArenaBackend, ArenaConfig, ArenaEpoch, ArenaStats, HostArena, HostResource,
    Pool, ReusePolicy,
};

/// A device whose allocations are numbered in creation order, sized exactly, and classed by a
/// small tag. `GROWS` selects whether it can grow an allocation in place.
#[derive(Default)]
struct Device<const GROWS: bool> {
    created: Cell<u64>,
}

#[derive(Debug, PartialEq, Eq)]
struct Buffer {
    id: u64,
    bytes: u64,
}

/// A submission counter; `Epoch(0)` is complete at every watermark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Epoch(u64);

impl ArenaEpoch for Epoch {
    const NEVER: Self = Self(0);

    fn is_complete_at(self, watermark: Self) -> bool {
        self.0 <= watermark.0
    }
}

impl<const GROWS: bool> ArenaBackend for Device<GROWS> {
    type Resource = Buffer;
    type Class = u8;
    type Epoch = Epoch;

    const GROWS: bool = GROWS;

    fn class_rank(class: u8) -> u64 {
        u64::from(class)
    }

    fn create(&self, bytes: u64, _class: u8) -> Buffer {
        let id = self.created.get() + 1;
        self.created.set(id);
        Buffer { id, bytes }
    }

    fn bytes(&self, resource: &Buffer) -> u64 {
        resource.bytes
    }

    fn grow(&self, resource: &mut Buffer, bytes: u64) {
        resource.bytes = bytes;
    }
}

type Leased = (Buffer, AllocationKey<u8>);

/// A pool over its own test device, with the calls every test makes. `GROWS` selects whether the
/// device can grow an allocation.
struct Rig<const GROWS: bool> {
    pool: Pool<Device<GROWS>>,
    device: Device<GROWS>,
}

type Fixed = Rig<false>;
type Growing = Rig<true>;

impl<const GROWS: bool> Rig<GROWS> {
    fn new(reuse_policy: ReusePolicy) -> Self {
        Self::with(ArenaConfig::unbounded(reuse_policy))
    }

    fn with(config: ArenaConfig) -> Self {
        Self {
            pool: Pool::new(config),
            device: Device::default(),
        }
    }

    fn take(&mut self, bytes: u64, class: u8) -> Leased {
        self.pool
            .acquire(&self.device, bytes, class)
            .expect("an ordinary size has a key")
    }

    fn release(&mut self, (resource, key): Leased, latest_issued: Epoch) {
        self.pool
            .release(&self.device, resource, key, latest_issued);
    }

    /// Release with no submitted work outstanding, and reclaim: the allocation is free at once.
    fn give_back(&mut self, leased: Leased) {
        self.release(leased, Epoch::NEVER);
        self.pool.reclaim_completed(Epoch::NEVER);
    }

    /// Lease every `(bytes, class)` at once, then give them all back in order: they are cached.
    fn cache(&mut self, allocations: &[(u64, u8)]) {
        let leased: Vec<Leased> = allocations
            .iter()
            .map(|&(bytes, class)| self.take(bytes, class))
            .collect();
        for lease in leased {
            self.give_back(lease);
        }
    }

    fn stats(&self) -> ArenaStats {
        self.pool.stats()
    }

    fn fits(&self, bytes: u64, class: u8) -> bool {
        self.pool
            .has_fitting(bytes, class)
            .expect("an ordinary size has a key")
    }
}

fn budget(max_cached_bytes: u64, max_cached_buffers: usize) -> ArenaConfig {
    ArenaConfig {
        max_cached_bytes,
        max_cached_buffers,
        reuse_policy: ReusePolicy::ExactSize,
    }
}

fn assert_states_sum(stats: ArenaStats) {
    assert_eq!(
        stats.device_bytes(),
        stats.active_bytes + stats.in_flight_bytes + stats.cached_bytes
    );
    assert!(stats.peak_arena_bytes >= stats.device_bytes());
}

#[test]
fn a_retired_allocation_is_reused_only_once_its_epoch_completes() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    let first = rig.take(256, 0);
    rig.release(first, Epoch(3));
    assert_eq!(rig.stats().in_flight_bytes, 256);
    assert_eq!(
        rig.take(256, 0).0.id,
        2,
        "a retired allocation is not handed out"
    );

    rig.pool.reclaim_completed(Epoch(2));
    assert_eq!(
        rig.stats().in_flight_bytes,
        256,
        "epoch 3 has not completed"
    );
    rig.pool.reclaim_completed(Epoch(3));
    assert_eq!(rig.stats().cached_buffers, 1);
    assert_eq!(rig.take(256, 0).0.id, 1);
}

#[test]
fn exact_size_reuses_only_the_same_size_and_class() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    rig.cache(&[(256, 0)]);
    assert_eq!(rig.take(128, 0).0.id, 2, "smaller");
    assert_eq!(rig.take(257, 0).0.id, 3, "larger");
    assert_eq!(rig.take(256, 1).0.id, 4, "another class");
    assert_eq!(rig.take(256, 0).0.id, 1);
    let stats = rig.stats();
    assert_eq!((stats.reuse_hits, stats.reuse_misses), (1, 4));
}

#[test]
fn size_class_serves_a_request_from_its_power_of_two_class() {
    let mut rig = Fixed::new(ReusePolicy::SizeClass);
    let first = rig.take(300, 0);
    assert_eq!(first.1.bytes(), 512);
    rig.give_back(first);
    let (reused, key) = rig.take(400, 0);
    assert_eq!((reused.id, key.bytes()), (1, 512));
    assert_eq!(rig.take(600, 0).1.bytes(), 1024);
}

/// Moved from `hymeko_wgpu::sizing` with the rounding it tests. Above 2^63 there is no
/// representable power of two; an unchecked `next_power_of_two` wrapped to 0 there in release
/// builds, which would hand out a zero-sized allocation for an enormous request.
#[test]
fn size_classes_round_up_and_refuse_to_wrap() {
    let rounding = Fixed::new(ReusePolicy::SizeClass);
    let class = |bytes| rounding.pool.key_for(bytes, 0).map(|key| key.bytes());
    assert_eq!(class(1), Ok(1));
    assert_eq!(class(5), Ok(8));
    assert_eq!(class(1 << 20), Ok(1 << 20));
    assert_eq!(class(1 << 63), Ok(1 << 63));
    assert!(matches!(
        class((1 << 63) + 1),
        Err(ComputeError::IndexOverflow { .. })
    ));
    let exact = Fixed::new(ReusePolicy::BestFit);
    assert_eq!(
        exact.pool.key_for(5, 0).map(|key| key.bytes()),
        Ok(5),
        "only SizeClass rounds"
    );
}

#[test]
fn a_request_without_a_size_class_fails_and_takes_nothing() {
    let mut rig = Fixed::new(ReusePolicy::SizeClass);
    rig.cache(&[(1 << 20, 0)]);
    let before = rig.stats();
    assert!(matches!(
        rig.pool.request(u64::MAX, 0),
        Err(ComputeError::IndexOverflow { .. })
    ));
    assert!(rig.pool.has_fitting(u64::MAX, 0).is_err());
    assert_eq!(rig.stats(), before);
}

#[test]
fn best_fit_takes_the_smallest_allocation_that_holds_the_request() {
    let mut rig = Fixed::new(ReusePolicy::BestFit);
    rig.cache(&[(10_000, 0), (1_000, 0), (10, 0)]);
    let (chosen, key) = rig.take(900, 0);
    assert_eq!(chosen.id, 2, "not the large one, not the small one");
    assert_eq!(
        key.bytes(),
        1_000,
        "the lease is accounted at what it holds"
    );
}

#[test]
fn best_fit_grows_the_largest_when_nothing_fits_and_the_backend_grows() {
    let mut rig = Growing::new(ReusePolicy::BestFit);
    rig.cache(&[(10, 0), (100, 0)]);
    let (grown, key) = rig.take(1_000, 0);
    assert_eq!((grown.id, key.bytes()), (2, 1_000));
    let stats = rig.stats();
    assert_eq!(stats.buffers_created, 2, "no third allocation");
    assert_eq!(stats.resources_grown, 1);
    assert_eq!(stats.reuse_misses, 3);
    assert_eq!(stats.total_allocated_bytes, 10 + 100 + 900);
    assert_eq!(stats.cached_buffers, 1, "the small one stays cached");
    assert_states_sum(stats);
}

#[test]
fn best_fit_creates_when_nothing_fits_and_the_backend_cannot_grow() {
    let mut rig = Fixed::new(ReusePolicy::BestFit);
    rig.cache(&[(10, 0), (100, 0)]);
    assert_eq!(rig.take(1_000, 0).0.id, 3);
    assert_eq!(rig.stats().cached_buffers, 2);
}

#[test]
fn best_fit_never_grows_across_classes() {
    let mut rig = Growing::new(ReusePolicy::BestFit);
    rig.cache(&[(100, 1)]);
    assert_eq!(rig.take(1_000, 0).0.id, 2);
    assert_eq!(rig.stats().resources_grown, 0);
}

#[test]
fn eviction_drops_the_smallest_cached_allocation_first_in_every_run() {
    // Fresh pools, so each has its own hash seed; the victim must not depend on it.
    for _ in 0..16 {
        let mut rig = Fixed::with(budget(u64::MAX, 2));
        rig.cache(&[(128, 2), (128, 1), (512, 0)]);
        assert_eq!(rig.stats().evictions, 1);
        assert!(!rig.fits(128, 1), "equal size: the lower class rank goes");
        assert!(rig.fits(128, 2));
        assert!(rig.fits(512, 0));
    }
}

#[test]
fn an_allocation_larger_than_the_byte_budget_is_not_cached() {
    let mut rig = Fixed::with(budget(100, usize::MAX));
    rig.cache(&[(256, 0)]);
    let stats = rig.stats();
    assert_eq!((stats.evictions, stats.cached_buffers), (1, 0));
    assert_eq!(stats.device_bytes(), 0);
}

#[test]
fn a_release_inside_an_open_session_waits_for_that_sessions_submission() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    rig.pool.open_session();
    let leased = rig.take(256, 0);
    rig.release(leased, Epoch(1));
    rig.pool.reclaim_completed(Epoch(100));
    assert_eq!(
        rig.stats().in_flight_bytes,
        256,
        "pending, whatever has completed: its commands are not submitted yet"
    );
    rig.pool.submit_session(Epoch(5));
    rig.pool.reclaim_completed(Epoch(4));
    assert_eq!(rig.stats().in_flight_bytes, 256);
    rig.pool.reclaim_completed(Epoch(5));
    assert_eq!(rig.stats().cached_buffers, 1);
}

#[test]
fn same_session_reuse_takes_only_that_sessions_releases_of_the_class() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    let session = rig.pool.open_session();
    let leased = rig.take(256, 0);
    rig.release(leased, Epoch::NEVER);
    let mut pending = |session, bytes, class| rig.pool.take_pending(session, bytes, class).unwrap();
    assert!(pending(session + 1, 256, 0).is_none());
    assert!(pending(session, 256, 1).is_none());
    assert!(pending(session, 257, 0).is_none());
    let (reused, key) = pending(session, 128, 0).expect("its own release");
    assert_eq!((reused.id, key.bytes()), (1, 256));
    let stats = rig.stats();
    assert_eq!(
        (stats.reuse_hits, stats.in_flight_bytes, stats.active_bytes),
        (1, 0, 256)
    );
}

#[test]
fn a_second_open_session_disables_same_session_reuse() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    let (first, second) = (rig.pool.open_session(), rig.pool.open_session());
    let leased = rig.take(256, 0);
    rig.release(leased, Epoch::NEVER);
    for session in [first, second] {
        assert!(rig.pool.take_pending(session, 256, 0).unwrap().is_none());
    }
}

#[test]
fn an_abandoned_sessions_releases_are_reclaimed_at_any_watermark() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    rig.pool.open_session();
    let leased = rig.take(256, 0);
    rig.release(leased, Epoch(9));
    rig.pool.abandon_session();
    rig.pool.reclaim_completed(Epoch::NEVER);
    assert_eq!(rig.stats().cached_buffers, 1);
}

#[test]
fn every_state_total_is_exact_through_a_lease_cycle() {
    let mut rig = Fixed::new(ReusePolicy::ExactSize);
    let (a, b) = (rig.take(256, 0), rig.take(128, 0));
    assert_eq!(
        (rig.stats().active_bytes, rig.stats().peak_arena_bytes),
        (384, 384)
    );

    rig.release(a, Epoch(1));
    let stats = rig.stats();
    assert_eq!((stats.active_bytes, stats.in_flight_bytes), (128, 256));
    assert_states_sum(stats);

    rig.pool.reclaim_completed(Epoch(1));
    rig.give_back(b);
    assert_eq!(
        (rig.stats().cached_bytes, rig.stats().cached_buffers),
        (384, 2)
    );

    rig.pool.trim();
    let stats = rig.stats();
    assert_eq!(
        (stats.cached_bytes, stats.cached_buffers, stats.evictions),
        (0, 0, 2)
    );
    assert_eq!(stats.peak_arena_bytes, 384);
    assert_eq!(stats.total_allocated_bytes, 384);
    assert_eq!(stats.hit_rate(), Some(0.0));
}

#[test]
fn a_tenant_that_grows_its_resource_is_accounted_at_its_new_size() {
    let mut rig = Fixed::new(ReusePolicy::BestFit);
    let (mut buffer, key) = rig.take(100, 0);
    buffer.bytes = 400;
    rig.give_back((buffer, key));
    let stats = rig.stats();
    assert_eq!((stats.active_bytes, stats.cached_bytes), (0, 400));
    assert_eq!(stats.peak_arena_bytes, 400);
    assert_eq!(
        stats.total_allocated_bytes, 100,
        "the pool allocated only 100"
    );
}

#[test]
fn the_split_request_is_the_same_acquire() {
    let run = |split: bool| {
        let mut rig = Growing::new(ReusePolicy::BestFit);
        let mut ids = Vec::new();
        for bytes in [64, 32, 128, 16, 256] {
            let leased = if split {
                let allocation = rig.pool.request(bytes, 0).unwrap().allocate(&rig.device);
                rig.pool.admit(&rig.device, allocation)
            } else {
                rig.take(bytes, 0)
            };
            ids.push(leased.0.id);
            rig.give_back(leased);
        }
        (ids, rig.stats())
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn has_fitting_probes_without_taking() {
    let mut rig = Fixed::new(ReusePolicy::BestFit);
    rig.cache(&[(1_000, 0)]);
    assert!(rig.fits(900, 0));
    assert!(!rig.fits(2_000, 0));
    assert!(!rig.fits(900, 1));
    assert_eq!(rig.stats().cached_buffers, 1);
}

#[test]
fn the_default_config_is_the_device_budget_and_unbounded_has_none() {
    let device = ArenaConfig::default();
    assert_eq!(device.reuse_policy, ReusePolicy::ExactSize);
    assert_eq!(
        (device.max_cached_bytes, device.max_cached_buffers),
        (512 * 1024 * 1024, 64)
    );
    let host = ArenaConfig::unbounded(ReusePolicy::BestFit);
    assert_eq!(
        (host.max_cached_bytes, host.max_cached_buffers),
        (u64::MAX, usize::MAX)
    );
    assert_eq!(ReusePolicy::default(), ReusePolicy::ExactSize);
}

// ---- HostArena ---------------------------------------------------------------------------

#[test]
fn a_dropped_lease_returns_its_buffer_and_the_next_lease_reuses_it() {
    let arena = HostArena::<Vec<f32>>::default();
    let first = arena.zeroed(1000);
    let address = first.as_ptr();
    assert_eq!(arena.stats().cached_buffers, 0);
    drop(first);
    assert_eq!(arena.stats().cached_buffers, 1);
    let second = arena.for_overwrite(1000);
    assert_eq!(
        second.as_ptr(),
        address,
        "the same allocation, not a new one"
    );
    assert_eq!(arena.stats().cached_buffers, 0);
    assert_eq!(arena.stats().reuse_hits, 1);
}

#[test]
fn a_zeroed_lease_reads_zero_whatever_its_buffer_held() {
    let arena = HostArena::<Vec<f32>>::default();
    let mut dirty = arena.for_overwrite(64);
    dirty.fill(f32::NAN);
    drop(dirty);
    let clean = arena.zeroed(48);
    assert_eq!(clean.len(), 48);
    assert!(clean.iter().all(|&v| v.to_bits() == 0), "+0.0 everywhere");
}

#[test]
fn a_lease_prefers_a_buffer_that_already_has_the_capacity() {
    let arena = HostArena::<Vec<f32>>::default();
    let (small, large) = (arena.zeroed(10), arena.zeroed(10_000));
    let large_at = large.as_ptr();
    drop(large);
    drop(small);
    let lease = arena.for_overwrite(5_000);
    assert_eq!(lease.as_ptr(), large_at);
    assert_eq!(lease.len(), 5_000);
}

#[test]
fn a_lease_takes_the_smallest_buffer_that_fits() {
    let arena = HostArena::<Vec<f32>>::default();
    let (large, medium, small) = (arena.zeroed(10_000), arena.zeroed(1_000), arena.zeroed(10));
    let medium_at = medium.as_ptr();
    drop((large, medium, small));
    let lease = arena.for_overwrite(900);
    assert_eq!(
        lease.as_ptr(),
        medium_at,
        "not the large one, not the small one grown"
    );
}

#[test]
fn a_request_no_free_buffer_fits_grows_the_largest_free_one() {
    let arena = HostArena::<Vec<f32>>::default();
    drop((arena.zeroed(10), arena.zeroed(100)));
    let lease = arena.zeroed(1_000);
    assert_eq!(lease.len(), 1_000);
    assert!(lease.iter().all(|&v| v == 0.0));
    let stats = arena.stats();
    assert_eq!(
        stats.cached_buffers, 1,
        "the small one stays; no third buffer is made"
    );
    assert_eq!((stats.buffers_created, stats.resources_grown), (2, 1));
}

/// A value that is refilled by its tenant rather than sized by its lease, like a spline tape.
#[derive(Clone, Debug, Default, PartialEq)]
struct Tape {
    values: Vec<f32>,
}

impl HostResource for Tape {
    fn bytes(&self) -> u64 {
        (self.values.capacity() * size_of::<f32>()) as u64
    }
}

#[test]
fn a_typed_value_comes_back_in_the_state_its_lease_left_it() {
    let arena = HostArena::<Tape>::default();
    let mut lease = arena.lease();
    lease.values.extend_from_slice(&[1.0, 2.0]);
    let held = lease.bytes();
    drop(lease);
    assert_eq!(
        arena.stats().cached_bytes,
        held,
        "accounted at what it holds on release"
    );
    assert_eq!(
        arena.lease().values,
        [1.0, 2.0],
        "the caller refills a typed value"
    );
    assert_eq!(arena.stats().buffers_created, 1);
}

#[test]
fn a_clone_is_a_separate_lease_holding_a_copy() {
    let arena = HostArena::<Vec<f32>>::default();
    let mut original = arena.zeroed(3);
    original[1] = 2.0;
    let copy = original.clone();
    assert_eq!(*copy, vec![0.0, 2.0, 0.0]);
    assert_ne!(copy.as_ptr(), original.as_ptr());
    drop((original, copy));
    assert_eq!(arena.stats().cached_buffers, 2);
}

#[test]
fn leases_taken_and_dropped_on_several_threads_all_return() {
    let arena = HostArena::<Vec<f32>>::default();
    std::thread::scope(|scope| {
        for t in 0..8 {
            let arena = arena.clone();
            scope.spawn(move || {
                for round in 0..50 {
                    let mut lease = arena.zeroed(100 + t * 10 + round);
                    lease[0] = 1.0;
                }
            });
        }
    });
    let stats = arena.stats();
    assert!((1..=8).contains(&stats.cached_buffers), "{stats:?}");
    assert_eq!((stats.active_bytes, stats.in_flight_bytes), (0, 0));
    assert_states_sum(stats);
}

#[test]
fn a_lease_is_static_and_returns_from_another_thread() {
    fn assert_static_send<T: Send + 'static>(_: &T) {}
    let arena = HostArena::<Vec<f32>>::default();
    let lease = arena.zeroed(16);
    assert_static_send(&lease);
    std::thread::spawn(move || drop(lease)).join().unwrap();
    assert_eq!(arena.stats().cached_buffers, 1);
}

#[test]
fn a_budget_bounds_what_the_host_arena_keeps() {
    let arena = HostArena::<Vec<f32>>::new(ArenaConfig {
        max_cached_bytes: 8_000,
        max_cached_buffers: usize::MAX,
        reuse_policy: ReusePolicy::BestFit,
    });
    let leases: Vec<_> = (0..3).map(|_| arena.for_overwrite(1_000)).collect();
    assert_eq!(
        arena.stats().active_bytes,
        12_000,
        "leases are never evicted"
    );
    drop(leases);
    let stats = arena.stats();
    assert_eq!((stats.cached_bytes, stats.evictions), (8_000, 1));
}

#[test]
fn a_host_arena_clone_shares_its_pool() {
    let arena = HostArena::<Vec<f32>>::default();
    drop(arena.clone().zeroed(8));
    assert_eq!(arena.stats().cached_buffers, 1);
    assert_eq!(arena.config(), ArenaConfig::unbounded(ReusePolicy::BestFit));
    arena.trim();
    assert_eq!(arena.stats().cached_buffers, 0);
}
