//! Codata Substrate: the shared memory primitive in which agent contexts live
//! as *active state* rather than as serialized messages traveling through a
//! channel.
//!
//! The first iteration of `NexusChannel<T>` is a bounded, mutex-protected
//! ring of slots. It deliberately favors clarity over peak throughput; the
//! lock-free Phase 1.3 substrate (atomics + crossbeam) will replace the
//! `Mutex` once the contract is empirically pinned by the Phase 1.2 tests.

use std::sync::Mutex;

/// Bounded, thread-safe substrate where the most recently pushed item
/// represents the live "context" of a Nexus agent.
pub struct NexusChannel<T> {
    capacity: usize,
    buffer: Mutex<Vec<T>>,
}

impl<T> NexusChannel<T>
where
    T: Clone,
{
    /// Allocates a new substrate with the given upper bound on stored items.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Mutex::new(Vec::with_capacity(capacity)),
        }
    }

    /// Deposits an item into the substrate. Returns `Err` if the substrate is
    /// already at capacity; the caller is expected to surface the failure to
    /// the FFI boundary (e.g., as a Python `ValueError`).
    pub fn push(&self, item: T) -> Result<(), &'static str> {
        let mut buf = self
            .buffer
            .lock()
            .map_err(|_| "nexus channel mutex was poisoned")?;
        if buf.len() >= self.capacity {
            return Err("nexus channel is at full capacity");
        }
        buf.push(item);
        Ok(())
    }

    /// Returns a clone of the most recently pushed item without removing it,
    /// modeling the "Codata" active-state read: peeking does not consume
    /// state, it observes it.
    pub fn peek_context(&self) -> Option<T> {
        let buf = self.buffer.lock().ok()?;
        buf.last().cloned()
    }
}
