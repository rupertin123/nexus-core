//! Continuous Batching Scheduler.
//!
//! Where naive batching collects N agents into a fixed group and waits for
//! the slowest one, this scheduler operates per token-tick: at every
//! `step()` it inspects per-task state, evicts anything that is no longer
//! eligible to consume a batch slot (`Halted` waiting on an MCP tool, or
//! `Finished`), and immediately promotes `Waiting` tasks into the freed
//! seats. The result is a batch that stays saturated even when individual
//! agents stall on tool calls.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentState {
    /// Queued, eligible to be admitted into the active batch on the next tick.
    Waiting,
    /// Currently consuming a batch slot for token generation.
    Active,
    /// Stalled on an external dependency (e.g., an MCP tool round-trip).
    /// Vacates its batch slot until something external resumes it.
    Halted,
    /// Done generating; will never re-enter the batch.
    Finished,
}

/// One agent's slot in the scheduler's task table.
pub struct AgentTask {
    pub task_id: usize,
    pub state: AgentState,
}

/// Per-tick scheduler that maintains the active batch invariant.
pub struct ContinuousScheduler {
    max_batch_size: usize,
    tasks: Vec<AgentTask>,
    next_task_id: usize,
}

impl ContinuousScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            tasks: Vec::new(),
            next_task_id: 0,
        }
    }

    /// Registers a new task in the `Waiting` state and returns its id.
    pub fn add_task(&mut self) -> usize {
        let task_id = self.next_task_id;
        self.next_task_id += 1;
        self.tasks.push(AgentTask {
            task_id,
            state: AgentState::Waiting,
        });
        task_id
    }

    /// One engine tick. Promotes `Waiting` tasks into freed batch slots in
    /// insertion order, then returns the ids of every currently `Active`
    /// task. The returned vector is exactly the batch that the inference
    /// kernel would process for this token cycle.
    pub fn step(&mut self) -> Vec<usize> {
        let mut active_count = self
            .tasks
            .iter()
            .filter(|task| task.state == AgentState::Active)
            .count();

        for task in self.tasks.iter_mut() {
            if active_count >= self.max_batch_size {
                break;
            }
            if task.state == AgentState::Waiting {
                task.state = AgentState::Active;
                active_count += 1;
            }
        }

        self.tasks
            .iter()
            .filter(|task| task.state == AgentState::Active)
            .map(|task| task.task_id)
            .collect()
    }

    /// Mutates the state of an existing task. Returns an error if the id
    /// has not been registered via `add_task`.
    pub fn set_task_state(
        &mut self,
        task_id: usize,
        new_state: AgentState,
    ) -> Result<(), &'static str> {
        for task in self.tasks.iter_mut() {
            if task.task_id == task_id {
                task.state = new_state;
                return Ok(());
            }
        }
        Err("task id not found in scheduler")
    }
}
