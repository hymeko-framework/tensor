//! `Sink` — cycle-output collector with full / reservoir / early-stop modes.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cycle output collector — Full | Reservoir (Vitter Algorithm R) | EarlyStop.
pub enum Sink {
    Full(Vec<u32>),
    Reservoir {
        buf: Vec<u32>,
        cap: usize,
        seen: usize,
        rng_state: u64,
    },
    /// EarlyStop now consults a shared atomic counter so all parallel
    /// workers stop as soon as the *global* sample reaches cap, not when
    /// each per-segment sink fills its own cap (which would multiply
    /// the wasted DFS work by n_threads).
    EarlyStop {
        buf: Vec<u32>,
        cap: usize,
        global: std::sync::Arc<AtomicUsize>,
    },
}

impl Sink {
    pub fn new_full() -> Self { Sink::Full(Vec::new()) }
    pub fn new_reservoir(cap: usize, seed: u64) -> Self {
        Sink::Reservoir {
            buf: Vec::new(),
            cap,
            seen: 0,
            rng_state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }
    pub fn new_early_stop(cap: usize, global: std::sync::Arc<AtomicUsize>) -> Self {
        Sink::EarlyStop { buf: Vec::new(), cap, global }
    }

    /// LCG step → next u64. Uses Knuth's MMIX constants.
    #[inline]
    pub fn next_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    /// Returns true if the DFS should keep exploring; false if the
    /// sink is full and further cycles would be discarded (early-stop).
    pub fn offer(&mut self, path: &[u32]) -> bool {
        match self {
            Sink::Full(buf) => {
                buf.extend_from_slice(path);
                true
            }
            Sink::Reservoir { buf, cap, seen, rng_state } => {
                let k = path.len();
                if *seen < *cap {
                    buf.extend_from_slice(path);
                } else {
                    let r = Self::next_u64(rng_state) >> 33;
                    let j = (r as usize) % (*seen + 1);
                    if j < *cap {
                        let dst = j * k;
                        buf[dst..dst + k].copy_from_slice(path);
                    }
                }
                *seen += 1;
                true
            }
            Sink::EarlyStop { buf, cap, global } => {
                // Atomically claim a slot. If the global counter is
                // already >= cap, drop this cycle and tell the DFS to
                // stop. Relaxed ordering is enough — we only need
                // approximate consensus on "have we hit cap yet".
                let claimed = global.fetch_add(1, Ordering::Relaxed);
                if claimed < *cap {
                    buf.extend_from_slice(path);
                    // Keep going if the global counter hasn't yet
                    // committed to cap. After the increment, claimed+1
                    // is what's been issued; if that equals cap, the
                    // very next caller will be told to bail.
                    claimed + 1 < *cap
                } else {
                    false
                }
            }
        }
    }

    pub fn into_flat(self) -> Vec<u32> {
        match self {
            Sink::Full(b) => b,
            Sink::Reservoir { buf, .. } => buf,
            Sink::EarlyStop { buf, .. } => buf,
        }
    }
}
