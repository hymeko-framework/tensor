//! The host deployment: a [`Pool`] over process memory behind `Arc<Mutex<..>>`, with `'static`
//! leases that return on drop.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{AllocationKey, ArenaBackend, ArenaConfig, ArenaStats, Pool, ReusePolicy};

/// A value the host pools: it answers the heap bytes it holds and, when it is a buffer, makes room
/// for more.
///
/// A value that cannot make room is leased by its state, not its size: its tenant refills it, and
/// every request for it asks for zero bytes.
pub trait HostResource: Default + Send {
    /// Whether [`Self::reserve_bytes`] makes room. See [`ArenaBackend::GROWS`].
    const GROWS: bool = false;

    /// Heap bytes the value holds.
    fn bytes(&self) -> u64;

    /// Make room for at least `bytes`, keeping the contents. Called only when [`Self::GROWS`].
    fn reserve_bytes(&mut self, bytes: u64) {
        let _ = bytes;
    }
}

impl<T: Send> HostResource for Vec<T> {
    const GROWS: bool = true;

    fn bytes(&self) -> u64 {
        (self.capacity() * size_of::<T>()) as u64
    }

    /// `Vec::reserve`'s growth, so that a grown buffer is exactly what a `resize` to the same length
    /// would have made.
    fn reserve_bytes(&mut self, bytes: u64) {
        let elements =
            usize::try_from(bytes.div_ceil(size_of::<T>().max(1) as u64)).unwrap_or(usize::MAX);
        if elements > self.capacity() {
            self.reserve(elements - self.len());
        }
    }
}

/// The host backend: allocations of `R` in process memory. Zero-sized; it needs no device.
pub struct Host<R>(PhantomData<fn() -> R>);

impl<R> Host<R> {
    const BACKEND: Self = Self(PhantomData);
}

impl<R: HostResource> ArenaBackend for Host<R> {
    type Resource = R;
    type Class = ();
    type Epoch = ();

    const GROWS: bool = R::GROWS;

    fn class_rank(_class: ()) -> u64 {
        0
    }

    fn create(&self, bytes: u64, _class: ()) -> R {
        let mut value = R::default();
        if R::GROWS {
            value.reserve_bytes(bytes);
        }
        value
    }

    fn bytes(&self, resource: &R) -> u64 {
        resource.bytes()
    }

    fn grow(&self, resource: &mut R, bytes: u64) {
        resource.reserve_bytes(bytes);
    }
}

/// A pool of host values shared by the leases it hands out, from any thread.
///
/// A clone shares the pool. The default is unbounded with [`ReusePolicy::BestFit`]; a budget is
/// set with [`Self::new`].
pub struct HostArena<R: HostResource> {
    pool: Arc<Mutex<Pool<Host<R>>>>,
}

/// A value leased from a [`HostArena`]; it returns there when dropped.
///
/// It holds its arena by `Arc`, so it is `'static` whenever `R` is, and can sit in a tape that
/// outlives the call that leased it. It is `Send` whenever `R` is.
pub struct HostLease<R: HostResource> {
    /// `Some` for the lease's whole life; taken only by `drop`.
    held: Option<(R, AllocationKey<()>)>,
    pool: Arc<Mutex<Pool<Host<R>>>>,
}

impl<R: HostResource> HostArena<R> {
    #[must_use]
    pub fn new(config: ArenaConfig) -> Self {
        Self {
            pool: Arc::new(Mutex::new(Pool::new(config))),
        }
    }

    /// A cached value, or `R::default()` when none is cached. Its state is whatever its last
    /// tenant left: the caller refills it.
    #[must_use]
    pub fn lease(&self) -> HostLease<R> {
        self.sized(0)
    }

    #[must_use]
    pub fn config(&self) -> ArenaConfig {
        lock(&self.pool).config()
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> ArenaStats {
        lock(&self.pool).stats()
    }

    /// Drop every cached value.
    pub fn trim(&self) {
        lock(&self.pool).trim();
    }

    /// A value whose allocation holds at least `bytes`, when `R` can make room; see
    /// [`HostResource`]. Allocation and growth run outside the lock.
    fn sized(&self, bytes: u64) -> HostLease<R> {
        let want = if R::GROWS { bytes } else { 0 };
        // `want` is at most `isize::MAX` (checked by every caller that passes a size), and the
        // size class of that, 2^63, fits `u64`; with any other policy the key is `want` itself.
        // `request` therefore cannot fail.
        let request = lock(&self.pool)
            .request(want, ())
            .expect("a host request of at most isize::MAX bytes has a size class");
        let allocation = request.allocate(&Host::BACKEND);
        let held = lock(&self.pool).admit(&Host::BACKEND, allocation);
        HostLease {
            held: Some(held),
            pool: Arc::clone(&self.pool),
        }
    }
}

impl<T: Copy + Default + Send> HostArena<Vec<T>> {
    /// `len` copies of `T::default()`, zero for every numeric type, in a buffer this arena may have
    /// held before: for a buffer that is accumulated into.
    ///
    /// # Panics
    ///
    /// When `len` elements exceed `isize::MAX` bytes, as `Vec::with_capacity` does.
    #[must_use]
    pub fn zeroed(&self, len: usize) -> HostLease<Vec<T>> {
        let mut lease = self.for_overwrite(len);
        lease.fill(T::default());
        lease
    }

    /// `len` elements whose values are unspecified: for a buffer the caller writes element by
    /// element before reading any of it.
    ///
    /// # Panics
    ///
    /// When `len` elements exceed `isize::MAX` bytes, as `Vec::with_capacity` does.
    #[must_use]
    pub fn for_overwrite(&self, len: usize) -> HostLease<Vec<T>> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .filter(|&bytes| isize::try_from(bytes).is_ok())
            .expect("capacity overflow");
        let mut lease = self.sized(bytes as u64);
        lease.resize(len, T::default());
        lease
    }
}

impl<R: HostResource> Default for HostArena<R> {
    fn default() -> Self {
        Self::new(ArenaConfig::unbounded(ReusePolicy::BestFit))
    }
}

impl<R: HostResource> Clone for HostArena<R> {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
        }
    }
}

impl<R: HostResource> fmt::Debug for HostArena<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("HostArena")
            .field(&*lock(&self.pool))
            .finish()
    }
}

impl<R: HostResource> Drop for HostLease<R> {
    fn drop(&mut self) {
        if let Some((resource, key)) = self.held.take() {
            let mut pool = lock(&self.pool);
            pool.release(&Host::BACKEND, resource, key, ());
            // The host epoch is complete at every watermark: the release is free at once.
            pool.reclaim_completed(());
        }
    }
}

impl<R: HostResource> Deref for HostLease<R> {
    type Target = R;

    fn deref(&self) -> &R {
        // `held` is `None` only inside `drop`.
        &self.held.as_ref().expect("a live lease holds its value").0
    }
}

impl<R: HostResource> DerefMut for HostLease<R> {
    fn deref_mut(&mut self) -> &mut R {
        // `held` is `None` only inside `drop`.
        &mut self.held.as_mut().expect("a live lease holds its value").0
    }
}

/// A clone is a separate lease from the same arena, holding a copy.
impl<R: HostResource + Clone> Clone for HostLease<R> {
    fn clone(&self) -> Self {
        let arena = HostArena {
            pool: Arc::clone(&self.pool),
        };
        let mut copy = arena.sized(self.bytes());
        // `R`'s `clone_from`, which may reuse the leased value's capacity; the lease's own would
        // call this `clone` again.
        R::clone_from(&mut copy, self);
        copy
    }
}

impl<R: HostResource + fmt::Debug> fmt::Debug for HostLease<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// The pool behind `mutex`. A panic while it was locked leaves a pool whose every allocation is in
/// exactly one place, because no pool method panics between moving an allocation and accounting
/// for it: recover it.
fn lock<B: ArenaBackend>(mutex: &Mutex<Pool<B>>) -> MutexGuard<'_, Pool<B>> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
