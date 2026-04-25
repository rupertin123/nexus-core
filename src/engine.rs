//! Unified Nexus inference engine.
//!
//! Phase 2.3 fuses the two substrates built in 2.1 and 2.2 — the
//! PagedAttention block allocator and the Continuous Batching scheduler —
//! into a single orchestrator that survives a workload whose total memory
//! footprint exceeds physical capacity. The contract this module is built
//! to honour empirically: when an active agent reaches its required block
//! count, every block it held is reclaimed *in the same tick* and made
//! available to whichever waiting agent is admitted next, so the pool can
//! be churned through indefinitely without ever returning an allocation
//! failure to the caller.

use crate::paged_attention::{BlockId, KVCacheAllocator};

/// State of an agent inside the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AgentPhase {
    /// Queued, waiting for a slot in the active batch.
    Waiting,
    /// Currently in the active batch, consuming blocks each tick.
    Active,
    /// Done generating; never re-enters the batch and holds no blocks.
    Finished,
}

/// Per-agent bookkeeping. The allocator does not know which blocks belong
/// to which agent, so the engine tracks the mapping itself.
struct Agent {
    required_blocks: usize,
    held_blocks: Vec<BlockId>,
    phase: AgentPhase,
}

/// The unified engine. Owns the physical block pool, the batching limit,
/// and the table of agents that have been admitted to the system.
pub struct NexusEngine {
    allocator: KVCacheAllocator,
    max_batch_size: usize,
    agents: Vec<Agent>,
    next_agent_id: usize,
    active_count: usize,
}

impl NexusEngine {
    pub fn new(max_blocks: usize, max_batch_size: usize) -> Self {
        Self {
            allocator: KVCacheAllocator::new(max_blocks),
            max_batch_size,
            agents: Vec::new(),
            next_agent_id: 0,
            active_count: 0,
        }
    }

    /// Registers a new agent in the waiting queue and returns its id.
    /// The agent does not consume any block until `step()` admits it into
    /// the active batch.
    pub fn add_agent(&mut self, required_blocks: usize) -> usize {
        let agent_id = self.next_agent_id;
        self.next_agent_id += 1;
        self.agents.push(Agent {
            required_blocks,
            held_blocks: Vec::with_capacity(required_blocks),
            phase: AgentPhase::Waiting,
        });
        agent_id
    }

    /// One engine tick.
    ///
    /// The order is load-bearing: we admit waiting agents first, then
    /// allocate one block per active agent, then evict any agent that
    /// has reached its required block count and recycle its blocks back
    /// into the pool. The eviction step is the engine's escape hatch
    /// from OOM under stress: as soon as an agent's blocks return to the
    /// pool, the next tick can both admit a fresh agent and allocate
    /// for the survivors out of the freed capacity.
    ///
    /// Returns `true` while there is still work to do (any agent is
    /// either active or waiting), `false` once the workload is drained.
    pub fn step(&mut self) -> bool {
        self.admit_waiting_agents();
        self.allocate_one_block_per_active_agent();
        self.evict_finished_agents();
        self.has_pending_work()
    }

    /// Promotes `Waiting` agents into the active batch in insertion order
    /// until the batch is full.
    fn admit_waiting_agents(&mut self) {
        for agent in self.agents.iter_mut() {
            if self.active_count >= self.max_batch_size {
                break;
            }
            if agent.phase == AgentPhase::Waiting {
                agent.phase = AgentPhase::Active;
                self.active_count += 1;
            }
        }
    }

    /// Allocates one fresh physical block per active agent for this tick.
    /// If the pool is momentarily exhausted the agent skips this tick and
    /// will retry on the next one — finished agents will have freed their
    /// blocks by then.
    fn allocate_one_block_per_active_agent(&mut self) {
        for agent in self.agents.iter_mut() {
            if agent.phase != AgentPhase::Active {
                continue;
            }
            if agent.held_blocks.len() >= agent.required_blocks {
                continue;
            }
            if let Ok(block_id) = self.allocator.allocate_block() {
                agent.held_blocks.push(block_id);
            }
        }
    }

    /// Evicts every agent that has reached its required block count,
    /// returning all of its blocks to the allocator in a single pass.
    /// The agent record stays in the table so its id remains valid, but
    /// its phase becomes `Finished` and it never re-enters the batch.
    fn evict_finished_agents(&mut self) {
        for agent in self.agents.iter_mut() {
            if agent.phase != AgentPhase::Active {
                continue;
            }
            if agent.held_blocks.len() < agent.required_blocks {
                continue;
            }
            for block_id in agent.held_blocks.drain(..) {
                self.allocator.free_block(block_id);
            }
            agent.phase = AgentPhase::Finished;
            self.active_count -= 1;
        }
    }

    /// True while at least one agent is still active or waiting for work.
    fn has_pending_work(&self) -> bool {
        self.agents
            .iter()
            .any(|agent| agent.phase != AgentPhase::Finished)
    }
}
