//! **Scenario: how the algebra's memory grows with dimension.**
//!
//! `Multivector` stores `2^n` dense `f64`s regardless of how many blades carry a
//! value, and `geo` allocates a fresh output on every call. Wall-clock
//! benchmarks (`nagare/benches/clifford_algebra_bench.rs`) cover the time axis;
//! this covers the allocation axis, which is deterministic and therefore
//! testable rather than merely measurable (CLAUDE.md §3 — timing assertions are
//! flaky, allocation counts are not).
//!
//! Three properties are pinned:
//!
//! * storage is **exactly** `2^n · 8` bytes per multivector — the dense-storage
//!   contract, so a future sparse representation shows up here as a deliberate
//!   change rather than a silent one;
//! * `geo` builds **one** output and no other temporary;
//! * a rotor sandwich allocates a **constant, `n`-independent number** of
//!   buffers — the operation a transport chain repeats must not leak
//!   intermediates as the algebra grows.
//!
//! ## Why the counters are thread-local
//!
//! A process-global counter cannot support a byte-exact assertion here: `libtest`
//! runs test functions concurrently, and a *free* on another thread between the
//! baseline read and the peak read makes the measured peak come out **short**
//! (observed: 4041 B where 4096 B was allocated). A mutex around each test body
//! does not fix it, because the harness's own bookkeeping allocates outside that
//! lock. `libtest` gives each test its own thread, so thread-local counters
//! measure exactly the allocations the test itself caused — no lock, no
//! cross-test contamination, and the numbers become exact rather than
//! approximate.
//!
//! Only `hymeko_clifford` and `std` are involved.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use hymeko_clifford::{Multivector, Signature};

thread_local! {
    /// Live bytes attributable to this thread. `const` initialiser: reading it
    /// must never allocate, or the allocator would recurse.
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Track;

unsafe impl GlobalAlloc for Track {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the layout is forwarded to the system allocator unmodified.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            // `try_with` because TLS is unavailable during thread teardown;
            // dropping those samples is correct, not a silent failure — no
            // measurement is in flight then.
            let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
            let _ = LIVE.try_with(|l| {
                let now = l.get() + layout.size();
                l.set(now);
                let _ = PEAK.try_with(|p| p.set(p.get().max(now)));
            });
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = LIVE.try_with(|l| l.set(l.get().saturating_sub(layout.size())));
        // SAFETY: `ptr` came from `alloc` with this same layout.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOC: Track = Track;

/// Peak bytes and allocation count attributable to `f`, on this thread.
///
/// # Postconditions
/// The returned peak excludes allocations live before the call; `f`'s result is
/// dropped before returning, so the peak covers construction, not teardown.
fn profile<F, R>(f: F) -> (usize, usize)
where
    F: FnOnce() -> R,
{
    let baseline = LIVE.with(|l| l.get());
    PEAK.with(|p| p.set(baseline));
    let before = ALLOCS.with(|a| a.get());
    let out = f();
    let peak = PEAK.with(|p| p.get()).saturating_sub(baseline);
    let allocs = ALLOCS.with(|a| a.get()) - before;
    drop(out);
    (peak, allocs)
}

const F64: usize = std::mem::size_of::<f64>();

/// Bytes a dense multivector of dimension `n` must occupy.
fn dense_bytes(n: usize) -> usize {
    (1usize << n) * F64
}

#[test]
fn multivector_storage_is_exactly_two_to_the_n_doubles() {
    for n in 1..=14usize {
        let (peak, allocs) = profile(|| Multivector::zero(n));
        assert_eq!(
            allocs, 1,
            "Multivector::zero({n}) made {allocs} allocations, expected exactly 1"
        );
        assert_eq!(
            peak,
            dense_bytes(n),
            "Multivector::zero({n}) reserved {peak} B, expected 2^{n}·8 = {} B",
            dense_bytes(n)
        );
    }
}

#[test]
fn geometric_product_allocates_one_output_and_nothing_else() {
    for n in 2..=10usize {
        let sig = Signature::euclidean(n);
        let a = Multivector::zero(n);
        let b = Multivector::zero(n);
        let (peak, allocs) = profile(|| a.geo(&b, &sig));
        assert_eq!(
            allocs, 1,
            "geo at n={n} made {allocs} allocations; the product must build one output"
        );
        assert_eq!(
            peak,
            dense_bytes(n),
            "geo at n={n} peaked at {peak} B, expected one output buffer of {} B",
            dense_bytes(n)
        );
    }
}

#[test]
fn rotor_sandwich_temporaries_do_not_grow_with_dimension() {
    // `R x R̃` is the repeated unit of a transport chain. The *number* of
    // temporaries must be constant; only their size may follow 2^n. A regression
    // that allocated per blade would surface here long before it surfaced as
    // wall time.
    let mut counts = Vec::new();
    for n in 3..=12usize {
        let sig = Signature::euclidean(n);
        let mut r = Multivector::zero(n);
        r.components[0] = 0.8;
        r.components[0b011] = 0.6;
        let mut x = Multivector::zero(n);
        x.components[1] = 1.0;

        let (peak, allocs) = profile(|| r.geo(&x, &sig).geo(&r.reverse(), &sig));
        counts.push(allocs);
        eprintln!(
            "sandwich n={n:>2}: {allocs} allocations, peak {peak:>7} B = {}× the dense buffer",
            peak / dense_bytes(n)
        );
        assert!(
            peak <= 4 * dense_bytes(n),
            "sandwich at n={n} peaked at {peak} B — more than four live buffers of {} B",
            dense_bytes(n)
        );
    }
    let first = counts[0];
    assert!(
        counts.iter().all(|&c| c == first),
        "sandwich allocation count varies with dimension: {counts:?} — it must be \
         constant, with only the buffer size following 2^n"
    );
}

#[test]
fn dense_storage_is_paid_even_for_a_single_populated_blade() {
    // The cost of the dense representation, stated as a test rather than left
    // implicit: a vector with one populated blade still reserves 2^n. This is the
    // measurement that would justify a sparse representation if one is ever
    // wanted, and the test that must be updated deliberately if one lands.
    for n in [8usize, 12, 14] {
        let (peak, _) = profile(|| {
            let mut mv = Multivector::zero(n);
            mv.components[1] = 1.0;
            mv
        });
        assert_eq!(
            peak,
            dense_bytes(n),
            "expected the full dense buffer at n={n}"
        );
        eprintln!(
            "dense-storage overhead at n={n:>2}: {peak:>7} B reserved for {F64} B of data ({}×)",
            dense_bytes(n) / F64
        );
    }
}
