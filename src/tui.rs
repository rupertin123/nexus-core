//! Asynchronous TUI for the Nexus-Core operator console.
//!
//! The contract this module exists to satisfy: a DevOps engineer must be
//! able to watch LLM tokens stream in *and* see live system telemetry
//! without either pane stalling the other. We model that as a `tokio`
//! event loop that multiplexes two independent sources — agent events
//! (tokens and status messages from the orchestrator) and keyboard
//! events (the user's escape hatch) — and redraws on every tick. The
//! terminal is restored deterministically through a Drop guard so a
//! panic anywhere inside the loop still leaves the user with a working
//! shell.

use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use tokio::sync::mpsc;
use tokio::task;

/// Events the orchestrator pushes into the TUI. Two variants suffice
/// for Phase 4.1 — one for the token stream, one for everything else
/// the operator needs to glance at without scrolling. Adding a third
/// variant later does not change the loop's shape.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// One token (or token chunk) appended to the reasoning buffer.
    Token(String),
    /// A one-line system telemetry update displayed in the status pane.
    Status(String),
}

/// RAII guard around the `ratatui` terminal handle. Owning the
/// renderer through `TerminalRenderer` means the `Drop` impl runs on
/// every exit path — clean returns, `?` early returns, *and* panics —
/// so the user is never left with raw mode stuck on after a crash.
pub struct TerminalRenderer {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalRenderer {
    /// Acquires the terminal: enables raw mode and switches into the
    /// alternate screen. If either step fails we unwind whatever we
    /// did manage to set so the user's shell is left in its prior
    /// state.
    pub fn acquire() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(e) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(e);
        }
        let backend = CrosstermBackend::new(stdout);
        match Terminal::new(backend) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(e) => {
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
                let _ = disable_raw_mode();
                Err(e)
            }
        }
    }

    /// Renders one frame: a large reasoning pane on top, a single-line
    /// status bar beneath it. The split is computed every frame because
    /// the user can resize the terminal between ticks.
    fn draw(&mut self, reasoning: &str, status: &str) -> io::Result<()> {
        self.terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.size());

            let reasoning_panel = Paragraph::new(reasoning.to_string())
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .title("Reasoning Stream")
                        .borders(Borders::ALL),
                );

            let status_line = Paragraph::new(Line::from(vec![Span::styled(
                status.to_string(),
                Style::default().add_modifier(Modifier::BOLD),
            )]))
            .block(Block::default().title("Status").borders(Borders::ALL));

            frame.render_widget(reasoning_panel, chunks[0]);
            frame.render_widget(status_line, chunks[1]);
        })?;
        Ok(())
    }
}

impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        // Best-effort cleanup. We deliberately swallow errors because we
        // are already on an exit path and the alternative — panicking
        // inside Drop — would be strictly worse for the operator.
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

/// Drives the operator console until the user asks to exit.
///
/// The loop multiplexes three sources via `tokio::select!`:
///
///   * `agent_rx` carries `AgentEvent`s from the orchestrator. Tokens
///     are concatenated into a reasoning buffer; status updates
///     replace the status line.
///   * `key_rx` carries keyboard events forwarded by a blocking
///     polling task. `Esc` and `Ctrl+C` both terminate the loop.
///   * A short timer triggers a redraw even when neither source
///     produced anything, so a resize event from the user is picked
///     up promptly.
///
/// The keyboard poller runs inside `spawn_blocking` because
/// `crossterm::event::read` is a synchronous syscall; isolating it on
/// the blocking pool means the main runtime is never starved by stdin
/// activity.
pub async fn run_interactive_terminal(
    mut agent_rx: mpsc::Receiver<AgentEvent>,
) -> io::Result<()> {
    let mut renderer = TerminalRenderer::acquire()?;

    let (key_tx, mut key_rx) = mpsc::channel::<KeyEvent>(32);
    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    let key_task = task::spawn_blocking(move || -> io::Result<()> {
        loop {
            if shutdown_rx.try_recv().is_ok() {
                break;
            }
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Release {
                        continue;
                    }
                    if key_tx.blocking_send(key).is_err() {
                        break;
                    }
                }
            }
        }
        Ok(())
    });

    let mut reasoning_buffer = String::new();
    let mut status_line = String::from("idle");
    renderer.draw(&reasoning_buffer, &status_line)?;

    let mut redraw_tick = tokio::time::interval(Duration::from_millis(50));
    redraw_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            maybe_event = agent_rx.recv() => {
                match maybe_event {
                    Some(AgentEvent::Token(token)) => {
                        reasoning_buffer.push_str(&token);
                    }
                    Some(AgentEvent::Status(status)) => {
                        status_line = status;
                    }
                    None => {
                        // Producer hung up. Show a final frame so the
                        // operator sees the terminal state at end-of-stream.
                        renderer.draw(&reasoning_buffer, &status_line)?;
                        break;
                    }
                }
                renderer.draw(&reasoning_buffer, &status_line)?;
            }
            maybe_key = key_rx.recv() => {
                if let Some(key) = maybe_key {
                    if is_quit_key(&key) {
                        break;
                    }
                } else {
                    // Key task exited unexpectedly — treat as quit so
                    // the user is not left without an escape hatch.
                    break;
                }
            }
            _ = redraw_tick.tick() => {
                renderer.draw(&reasoning_buffer, &status_line)?;
            }
        }
    }

    // Signal the key poller to exit and wait for it. We do not
    // propagate its result: the operator is already on the way out.
    let _ = shutdown_tx.send(()).await;
    let _ = key_task.await;
    Ok(())
}

/// `Esc` or `Ctrl+C` terminate the session. The check lives in its
/// own function so the quit policy can grow (e.g. add `q`) without
/// disturbing the select loop.
fn is_quit_key(key: &KeyEvent) -> bool {
    matches!(key.code, KeyCode::Esc)
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
}
