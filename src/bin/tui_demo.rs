//! Manual exercise harness for the Nexus-Core TUI.
//!
//! The TUI is too coupled to a real terminal to fit in a pytest
//! assertion, so this binary is the rehearsal stand-in. A background
//! task fakes a token stream from the orchestrator while a status
//! heartbeat fires every ten ticks. Running `cargo run --bin tui_demo`
//! locally lets an operator verify the layout, the scrollback, and
//! the `Esc` / `Ctrl+C` exit path end-to-end. The project's
//! automated check is the compile alone — `cargo build --bin
//! tui_demo` proves the wiring is sound without needing a TTY.

use std::time::Duration;

use nexus_core::tui::{run_interactive_terminal, AgentEvent};
use tokio::sync::mpsc;
use tokio::time;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let (tx, rx) = mpsc::channel::<AgentEvent>(64);

    // Producer: simulates the orchestrator pushing tokens at ~20 Hz
    // and a status heartbeat every ten tokens. The producer exits
    // silently if the TUI receiver hangs up first.
    let producer_tx = tx.clone();
    tokio::spawn(async move {
        let mut tick: u64 = 0;
        loop {
            time::sleep(Duration::from_millis(50)).await;
            if producer_tx
                .send(AgentEvent::Token(" token".into()))
                .await
                .is_err()
            {
                break;
            }
            tick += 1;
            if tick % 10 == 0 {
                if producer_tx
                    .send(AgentEvent::Status("Background MCP Call Executed".into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    });

    // Drop the original sender so the channel closes when the
    // producer task exits, allowing the TUI to detect end-of-stream.
    drop(tx);

    run_interactive_terminal(rx).await
}
