//! Codata Substrate: the shared memory primitive in which agent contexts live
//! as *active state* rather than as serialized messages traveling through a
//! channel.
//!
//! Phase 1.3 substrate: a lock-free reader/writer pair backed by `arc-swap`
//! plus an atomic length counter. This removes the contention of the original
//! `Mutex<Vec<T>>` while preserving the empirical contract pinned by the
//! Phase 1.2 tests: peek is non-destructive, push overwrites the observed
//! head, and pushing past capacity is a hard error.

use arc_swap::ArcSwapOption;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Bounded, lock-free substrate where the most recently pushed item
/// represents the live "context" of a Nexus agent.
pub struct NexusChannel<T> {
    capacity: usize,
    len: AtomicUsize,
    latest: ArcSwapOption<T>,
}

impl<T> NexusChannel<T>
where
    T: Clone,
{
    /// Allocates a new substrate with the given upper bound on stored items.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            len: AtomicUsize::new(0),
            latest: ArcSwapOption::from(None),
        }
    }

    /// Deposits an item into the substrate. Returns `Err` if the substrate is
    /// already at capacity; the caller is expected to surface the failure to
    /// the FFI boundary (e.g., as a Python `ValueError`).
    pub fn push(&self, item: T) -> Result<(), &'static str> {
        if self.len.load(Ordering::Relaxed) >= self.capacity {
            return Err("nexus channel is at full capacity");
        }
        self.len.fetch_add(1, Ordering::Relaxed);
        self.latest.store(Some(Arc::new(item)));
        Ok(())
    }

    /// Returns a clone of the most recently pushed item without removing it,
    /// modeling the "Codata" active-state read: peeking does not consume
    /// state, it observes it.
    pub fn peek_context(&self) -> Option<T> {
        self.latest.load_full().map(|arc| (*arc).clone())
    }
}
