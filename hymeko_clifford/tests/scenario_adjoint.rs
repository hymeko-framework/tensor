//! **Scenario: the adjoint sweep — correctness against the reference, and the
//! complexity class proved by counting rather than by clock.**
//!
//! `RotorChain::adjoint` claims `O(k)` where `finite_difference_gradient` is
//! `O(k²)`. Timing can suggest that; it cannot prove it, and timing assertions
//! are flaky (CLAUDE.md §3). Allocation counts are deterministic, and every
//! `Multivector::geo` allocates exactly one buffer — pinned by
//! `scenario_allocation_scaling.rs`. Counting allocations therefore measures
//! *rotor applications* exactly, and the growth ratio across a doubling of `k`
//! is the complexity class itself:
//!
//! * adjoint — one forward and one backward sweep ⇒ ratio ≈ **2**;
//! * central difference — `2k` sweeps of `k` links ⇒ ratio ≈ **4**.
//!
//! Correctness comes first: a fast wrong gradient is worthless, so the adjoint is
//! checked against the finite difference it is replacing, and the finite
//! difference is itself checked against the closed form `ω × w` in
//! `scenario_derivative.rs`.
//!
//! Only `hymeko_clifford` and `std` are involved.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use hymeko_clifford::RotorChain;

thread_local! {
    /// `const` initialiser: reading it must never allocate, or the allocator
    /// would recurse. One counter per test thread, so `libtest`'s concurrency
    /// cannot contaminate a measurement (the lesson from
    /// `scenario_allocation_scaling.rs`).
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Count;

unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: layout forwarded unmodified to the system allocator.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            // `try_with`: TLS is gone during thread teardown, and no measurement
            // is in flight then.
            let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this same layout.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOC: Count = Count;

/// Allocations performed by `f` on this thread.
fn allocations_of<F, R>(f: F) -> usize
where
    F: FnOnce() -> R,
{
    let before = ALLOCS.with(|a| a.get());
    let out = f();
    let n = ALLOCS.with(|a| a.get()) - before;
    drop(out);
    n
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Deterministic axes — no rng dependency, reproducible without a fixture file.
fn chain_of(k: usize) -> RotorChain {
    let axes: Vec<[f64; 3]> = (0..k)
        .map(|i| {
            let t = i as f64;
            unit([
                (0.7 * t + 0.3).sin(),
                (1.1 * t + 0.9).cos(),
                (0.4 * t + 1.7).sin(),
            ])
        })
        .collect();
    RotorChain::new(axes).expect("axes are unit by construction")
}

fn angles_of(k: usize) -> Vec<f64> {
    (0..k).map(|i| 0.11 + 0.07 * i as f64).collect()
}

const V: [f64; 3] = [0.3, -0.5, 0.8];
const SEED: [f64; 3] = [0.6, 0.2, -0.9];

#[test]
fn adjoint_matches_the_finite_difference_across_chain_lengths() {
    let mut worst: f64 = 0.0;
    for k in [1usize, 2, 4, 8, 16] {
        let chain = chain_of(k);
        let angles = angles_of(k);
        let exact = chain.adjoint(&angles, V, SEED).expect("adjoint");
        let reference = chain
            .finite_difference_gradient(&angles, V, SEED, 1e-6)
            .expect("finite difference");

        assert_eq!(exact.grad_angles.len(), k);
        for (i, (&got, &want)) in exact.grad_angles.iter().zip(reference.iter()).enumerate() {
            let d = (got - want).abs();
            worst = worst.max(d);
            assert!(
                d < 1e-7,
                "k={k}, angle {i}: adjoint {got} vs finite difference {want}"
            );
        }
    }
    eprintln!("adjoint vs central difference: worst component error {worst:.3e}");
}

#[test]
fn adjoint_also_returns_the_input_gradient() {
    // `∂L/∂v` falls out of the same backward sweep. It is checked against a
    // finite difference in `v`, which the library does not provide — so this is
    // a genuine external check, not a self-comparison.
    let k = 6;
    let chain = chain_of(k);
    let angles = angles_of(k);
    let out = chain.adjoint(&angles, V, SEED).expect("adjoint");

    let h = 1e-6;
    for axis in 0..3 {
        let mut up = V;
        let mut down = V;
        up[axis] += h;
        down[axis] -= h;
        let dot = |w: [f64; 3]| w[0] * SEED[0] + w[1] * SEED[1] + w[2] * SEED[2];
        let numeric = (dot(chain.apply(&angles, up).expect("apply"))
            - dot(chain.apply(&angles, down).expect("apply")))
            / (2.0 * h);
        assert!(
            (out.grad_input[axis] - numeric).abs() < 1e-7,
            "∂L/∂v component {axis}: adjoint {} vs finite difference {numeric}",
            out.grad_input[axis]
        );
    }
}

#[test]
fn adjoint_value_agrees_with_apply() {
    // The forward sweep inside `adjoint` must not drift from `apply`; a caller
    // that used one for the value and the other for the gradient would otherwise
    // be differentiating a different function than it evaluated.
    for k in [1usize, 3, 9] {
        let chain = chain_of(k);
        let angles = angles_of(k);
        let via_apply = chain.apply(&angles, V).expect("apply");
        let via_adjoint = chain.adjoint(&angles, V, SEED).expect("adjoint").value;
        for i in 0..3 {
            assert!(
                (via_apply[i] - via_adjoint[i]).abs() < 1e-15,
                "k={k} component {i}: apply {via_apply:?} vs adjoint {via_adjoint:?}"
            );
        }
    }
}

#[test]
fn adjoint_is_linear_in_k_and_the_finite_difference_is_quadratic() {
    // The complexity claim, machine-checked. Every `geo` allocates exactly one
    // buffer, so an allocation count is a rotor-application count.
    let mut adjoint_counts = Vec::new();
    let mut fd_counts = Vec::new();
    for k in [4usize, 8, 16, 32] {
        let chain = chain_of(k);
        let angles = angles_of(k);
        let a = allocations_of(|| chain.adjoint(&angles, V, SEED).expect("adjoint"));
        let f = allocations_of(|| {
            chain
                .finite_difference_gradient(&angles, V, SEED, 1e-6)
                .expect("finite difference")
        });
        eprintln!(
            "k={k:>2}: adjoint {a:>6} allocations, central difference {f:>6}  ({:.1}× more)",
            f as f64 / a as f64
        );
        adjoint_counts.push((k, a));
        fd_counts.push((k, f));
    }

    for w in adjoint_counts.windows(2) {
        let ratio = w[1].1 as f64 / w[0].1 as f64;
        assert!(
            (1.8..2.2).contains(&ratio),
            "adjoint allocation growth from k={} to k={} was {ratio:.2}×, expected ≈2 (linear)",
            w[0].0,
            w[1].0
        );
    }
    for w in fd_counts.windows(2) {
        let ratio = w[1].1 as f64 / w[0].1 as f64;
        assert!(
            (3.6..4.4).contains(&ratio),
            "finite-difference allocation growth from k={} to k={} was {ratio:.2}×, \
             expected ≈4 (quadratic)",
            w[0].0,
            w[1].0
        );
    }

    // At the largest size the separation must be substantial, not marginal.
    let (k, a) = *adjoint_counts.last().expect("counts");
    let (_, f) = *fd_counts.last().expect("counts");
    assert!(
        f as f64 / a as f64 > 8.0,
        "at k={k} the finite difference did only {:.1}× the adjoint's work",
        f as f64 / a as f64
    );
}

#[test]
fn the_returned_gradient_is_the_natural_gradient_for_the_group_metric() {
    // §6i: the bi-invariant metric is the identity in these coordinates, so the
    // Euclidean gradient the adjoint returns *is* the Riemannian one — no metric
    // solve. The property that makes it so is that a unit change in any angle is
    // a unit geodesic step, independent of the link's position in the chain.
    // Asserted here so a future reparameterisation cannot quietly break it.
    let k = 8;
    let chain = chain_of(k);
    let angles = angles_of(k);
    let d = 1e-6;

    for i in 0..k {
        // Geodesic distance between the composed rotations, via the angle of the
        // relative rotation applied to an orthonormal frame.
        let mut moved = angles.clone();
        moved[i] += d;
        let mut trace = 0.0;
        for basis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let a = chain.apply(&angles, basis).expect("apply");
            let b = chain.apply(&moved, basis).expect("apply");
            trace += a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }
        let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let step_per_unit = angle / d;
        assert!(
            (step_per_unit - 1.0).abs() < 1e-3,
            "link {i}: a unit change in θ moved the chain by {step_per_unit} geodesic \
             units, expected 1 — the metric is no longer the identity and the \
             returned gradient is no longer the natural one"
        );
    }
}

#[test]
fn a_chain_that_composes_to_the_identity_has_the_expected_gradient_structure() {
    // Out-and-back about one axis: the composed rotation is the identity, so the
    // value is `v` unchanged — but the *gradient* is not zero, because each angle
    // still moves the output. A test asserting "identity ⇒ zero gradient" would
    // be wrong, and this pins the correct behaviour so nobody "fixes" it.
    let axis = unit([0.4, -0.7, 0.6]);
    let chain = RotorChain::new(vec![axis, axis]).expect("unit axes");
    let angles = [0.8, -0.8];
    let out = chain.adjoint(&angles, V, SEED).expect("adjoint");

    for (i, (&got, &want)) in out.value.iter().zip(V.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "the composed rotation should be the identity at component {i}: {:?} vs {V:?}",
            out.value
        );
    }
    // Both angles rotate about the same axis, so their partials are equal.
    assert!(
        (out.grad_angles[0] - out.grad_angles[1]).abs() < 1e-12,
        "equal-axis links should share a partial: {:?}",
        out.grad_angles
    );
    assert!(
        out.grad_angles[0].abs() > 1e-6,
        "the gradient vanished on a non-degenerate objective: {:?}",
        out.grad_angles
    );
}
