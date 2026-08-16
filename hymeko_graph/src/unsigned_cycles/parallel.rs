//! Rayon-parallel cycle enumeration driver (uses CSR + BFS + DFS + Sink).

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::bfs::bfs_distances_into;
use super::csr::{bs_words, neighbours};
use super::dfs::dfs_from_pair;
use super::sink::Sink;

/// per-mode invariant (uniform sample / full enumeration / first-cap).
pub fn merge_sinks(a: Sink, b: Sink, k: usize) -> Sink {
    match (a, b) {
        (Sink::Full(mut buf_a), Sink::Full(buf_b)) => {
            buf_a.extend_from_slice(&buf_b);
            Sink::Full(buf_a)
        }
        (Sink::Reservoir { buf: ba, cap, seen: sa, rng_state: rsa },
         Sink::Reservoir { buf: bb, seen: sb, rng_state: rsb, .. }) => {
            // Stratified merge. Each stratum (a, b) is itself a uniform
            // sample of its observed `seen_t` cycles. We allocate the
            // global cap proportionally: target_a = cap * sa / (sa+sb).
            // Items in a Vitter reservoir are uniformly distributed
            // across reservoir slots, so taking the *first* target_a is
            // a uniform sub-sample of the stratum.
            let new_seen = sa + sb;
            let avail_a = ba.len() / k;
            let avail_b = bb.len() / k;
            let target_a = if new_seen == 0 {
                0
            } else {
                ((cap as u128 * sa as u128) / new_seen as u128) as usize
            }.min(avail_a);
            let target_b = cap.saturating_sub(target_a).min(avail_b);
            // Backfill if one side underdelivered (rare, only at boundary).
            let target_a = (cap.saturating_sub(target_b)).min(avail_a).max(target_a);
            let mut new_buf = Vec::with_capacity((target_a + target_b) * k);
            new_buf.extend_from_slice(&ba[..target_a * k]);
            new_buf.extend_from_slice(&bb[..target_b * k]);
            Sink::Reservoir {
                buf: new_buf,
                cap,
                seen: new_seen,
                rng_state: rsa ^ rsb.rotate_left(13),
            }
        }
        (Sink::EarlyStop { mut buf, cap, global },
         Sink::EarlyStop { buf: bb, .. }) => {
            let needed = cap.saturating_sub(buf.len() / k) * k;
            let take = needed.min(bb.len());
            buf.extend_from_slice(&bb[..take]);
            Sink::EarlyStop { buf, cap, global }
        }
        _ => unreachable!("attempted to merge sinks of mismatched modes"),
    }
}

pub fn make_thread_sink(
    max_cycles: Option<usize>,
    early_stop: Option<&std::sync::Arc<AtomicUsize>>,
    seed: u64,
) -> Sink {
    match (max_cycles, early_stop) {
        (Some(cap), Some(global)) => Sink::new_early_stop(cap, global.clone()),
        (Some(cap), None) => Sink::new_reservoir(cap, seed),
        (None, _) => Sink::new_full(),
    }
}

pub fn make_identity_sink(
    max_cycles: Option<usize>,
    early_stop: Option<&std::sync::Arc<AtomicUsize>>,
) -> Sink {
    // Identity for the parallel `reduce`. Empty buffer, same mode.
    match (max_cycles, early_stop) {
        (Some(cap), Some(global)) => Sink::new_early_stop(cap, global.clone()),
        (Some(cap), None) => Sink::Reservoir {
            buf: Vec::new(),
            cap,
            seen: 0,
            rng_state: 0,
        },
        (None, _) => Sink::new_full(),
    }
}

/// Run DFS in parallel over starting vertices using rayon, then
/// merge per-thread sinks. Falls back to serial when n_threads == 1.
#[allow(clippy::too_many_arguments)]
// Flat CSR + control knobs mirror the PyO3 kwargs surface; a config struct
// would thread through `hymeko_py` call sites — tracked for a follow-up.
pub fn enumerate_parallel(
    row_ptr: &[u32],
    col_idx: &[u32],
    n_nodes: usize,
    k: usize,
    directed: bool,
    max_cycles: Option<usize>,
    seed: u64,
    early_stop: bool,
    n_threads: Option<usize>,
) -> Sink {
    let pool = if let Some(nt) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt.max(1))
            .build()
            .ok()
    } else {
        None
    };

    // Shared atomic counter for early-stop coordination across workers.
    let global_es = if early_stop && max_cycles.is_some() {
        Some(std::sync::Arc::new(AtomicUsize::new(0)))
    } else {
        None
    };

    // Parallelise by `start` so the per-start BFS distances (used for
    // closure-pruning the DFS in the undirected case) are computed once
    // per start and reused across that start's first-hop expansion. For
    // each start we sequentially run all (start, first_hop > start) DFSes;
    // the load-imbalance from heavy vertex 0 is largely offset by the
    // BFS pruning shrinking its DFS tree.
    let starts: Vec<u32> = (0..n_nodes as u32).collect();
    let words = bs_words(n_nodes);

    let do_work = || {
        starts
            .par_iter()
            .copied()
            .fold(
                || {
                    let tid = rayon::current_thread_index().unwrap_or(0) as u64;
                    let s = make_thread_sink(
                        max_cycles, global_es.as_ref(),
                        seed.wrapping_add(tid)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    );
                    // Per-fold-segment scratch: visited bitset, BFS dist
                    // buffer and BFS frontier/next queues. Reused across
                    // every start the segment processes — eliminates the
                    // n_nodes × n_nodes bytes of allocator churn that
                    // dominated wall-clock at high n.
                    let dist: Vec<u8> = if directed {
                        Vec::new()
                    } else {
                        vec![u8::MAX; n_nodes]
                    };
                    (vec![0u64; words],
                     Vec::with_capacity(k),
                     dist,
                     Vec::<u32>::new(),
                     Vec::<u32>::new(),
                     s)
                },
                |(mut visited, mut path, mut dist, mut bfs_a, mut bfs_b, mut sink), start| {
                    // Early-stop short-circuit at the segment level.
                    if let Some((g, cap)) = global_es.as_ref().zip(max_cycles)
                        && g.load(Ordering::Relaxed) >= cap
                    {
                        return (visited, path, dist, bfs_a, bfs_b, sink);
                    }
                    if !directed {
                        bfs_distances_into(
                            row_ptr, col_idx, start, n_nodes, k as u8,
                            &mut dist, &mut bfs_a, &mut bfs_b,
                        );
                    }
                    for &first_hop in neighbours(row_ptr, col_idx, start) {
                        if first_hop <= start { continue; }
                        // Bound check: first_hop must lie within k-1 hops
                        // of start (it does — it's at distance 1 — but the
                        // check is free and guards future API changes).
                        if !directed && dist[first_hop as usize] as usize > k - 1 {
                            continue;
                        }
                        let cont = dfs_from_pair(
                            row_ptr, col_idx, start, first_hop, k, directed,
                            &mut visited, &mut path, &dist, &mut sink,
                        );
                        if !cont { break; }
                    }
                    (visited, path, dist, bfs_a, bfs_b, sink)
                },
            )
            .map(|(_, _, _, _, _, s)| s)
            .reduce(
                || make_identity_sink(max_cycles, global_es.as_ref()),
                |a, b| merge_sinks(a, b, k),
            )
    };

    if let Some(p) = pool {
        p.install(do_work)
    } else {
        do_work()
    }
}
