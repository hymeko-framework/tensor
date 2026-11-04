//! Deterministic warm-allocation invariant. No timers, profiling or large inputs.
#[path = "support/gmres_cases.rs"]
mod cases;
use hymeko_numerics::{GmresConfig, GmresTolerance, GmresWorkspace};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}
struct AllocationCounter;
impl AllocationCounter {
    fn record() {
        let _ = COUNT.try_with(|count| {
            if let Some(value) = count.get() {
                count.set(Some(value + 1));
            }
        });
    }
    fn during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                COUNT.with(|c| c.set(None));
            }
        }
        COUNT.with(|c| c.set(Some(0)));
        let guard = Guard;
        let result = operation();
        let count = COUNT.with(|c| c.get().unwrap());
        drop(guard);
        (result, count)
    }
}
// Instrumentation only: every allocation is delegated unchanged to System.
unsafe impl GlobalAlloc for AllocationCounter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        Self::record();
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: AllocationCounter = AllocationCounter;

#[test]
fn warm_restarted_solve_reuses_capacity_across_dimensions_and_rhs_changes() {
    let mut workspace = GmresWorkspace::new();
    let mut case = cases::Case {
        n: 65,
        variant: 1,
        restart: 8,
        budget: 17,
        calls: Cell::new(0),
    };
    let config = GmresConfig {
        tolerance: GmresTolerance::Relative(1e-12),
        restart: 8,
        max_iterations: 17,
    };
    workspace.solve(&case, 0.75, &case.rhs(), &mut case.initial(), config);
    let retained = workspace.retained_bytes();
    assert!(retained >= 65 * 8 * 8);
    for n in [65, 17, 2, 0, 65] {
        case.n = n;
        case.variant ^= 1;
        let rhs = case.rhs();
        let mut x = case.initial();
        let (result, allocations) =
            AllocationCounter::during(|| workspace.solve(&case, 0.75, &rhs, &mut x, config));
        assert_eq!(allocations, 0, "dimension {n}");
        assert!(result.iterations <= config.max_iterations);
        assert_eq!(workspace.retained_bytes(), retained);
    }
}
