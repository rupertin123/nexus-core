//! PagedAttention substrate: a virtual memory manager for the Gemma 4 KV
//! cache that allocates fixed-size physical blocks instead of contiguous
//! per-sequence buffers, and uses reference counting to support
//! Copy-on-Write sharing of common prefixes (e.g., a heavy system prompt
//! shared by every branch of a Tree-of-Thought search).
//!
//! Phase 2.1 deliberately keeps the data structures plain (`Vec` + linear
//! scan); the lock-free / NUMA-aware allocator arrives in Phase 2.2 once
//! the CoW contract is empirically pinned.

/// Identifier of a physical KV-cache block. Index into
/// `KVCacheAllocator::physical_blocks`.
pub type BlockId = usize;

/// One fixed-size chunk of KV cache. The actual tensor data is owned
/// elsewhere (Phase 1.1 buffer); this struct only tracks the metadata
/// the allocator needs to decide whether a slot is live.
#[derive(Clone)]
pub struct PhysicalBlock {
    /// Number of logical sequences currently mapped to this physical slot.
    /// `0` means the slot is free and can be handed out by `allocate_block`.
    pub ref_count: usize,
}

/// A logical sequence's view of memory: an ordered list of physical block
/// ids. Two sequences may legitimately point at the same `BlockId` — that
/// is precisely how CoW prefix sharing is represented.
pub struct LogicalMemory {
    pub blocks: Vec<BlockId>,
}

/// Physical-block allocator. Owns the table of `PhysicalBlock` slots and
/// the scan/free/duplicate primitives that the higher-level sequence
/// manager composes on top.
pub struct KVCacheAllocator {
    physical_blocks: Vec<PhysicalBlock>,
}

impl KVCacheAllocator {
    /// Reserves `max_blocks` physical slots, all initially free.
    pub fn new(max_blocks: usize) -> Self {
        Self {
            physical_blocks: vec![PhysicalBlock { ref_count: 0 }; max_blocks],
        }
    }

    /// Returns the id of the first free slot, marking it live (`ref_count = 1`).
    /// Linear scan is fine for Phase 2.1; a free-list arrives in Phase 2.2.
    pub fn allocate_block(&mut self) -> Result<BlockId, &'static str> {
        for (id, block) in self.physical_blocks.iter_mut().enumerate() {
            if block.ref_count == 0 {
                block.ref_count = 1;
                return Ok(id);
            }
        }
        Err("no free physical blocks available")
    }

    /// Decrements the ref-count of `id`. If it reaches zero the slot
    /// becomes available to the next `allocate_block`.
    pub fn free_block(&mut self, id: BlockId) {
        if let Some(block) = self.physical_blocks.get_mut(id) {
            if block.ref_count > 0 {
                block.ref_count -= 1;
            }
        }
    }

    /// Increments the ref-count of `id` and returns the same id. This is
    /// the CoW primitive: a "copy" is just a new reference to the same
    /// physical block. An actual deep-copy only happens at write time, in
    /// a later phase, when the writer detects `ref_count > 1`.
    pub fn duplicate_block(&mut self, id: BlockId) -> BlockId {
        if let Some(block) = self.physical_blocks.get_mut(id) {
            block.ref_count += 1;
        }
        id
    }

    /// Number of physical slots currently free.
    pub fn available_blocks(&self) -> usize {
        self.physical_blocks
            .iter()
            .filter(|b| b.ref_count == 0)
            .count()
    }
}
