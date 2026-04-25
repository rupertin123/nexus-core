# Contributing to Nexus-Core

Thanks for your interest in moving the project forward. Read the two
short sections below before opening a pull request.

## Open-Core Philosophy

Nexus-Core ships under a deliberate two-tier license model:

- **Apache 2.0 — the core.** Everything currently in this repository
  (`/src`, `/tests`, the PyO3 surface, the CI/CD workflows, the TUI
  demo) is and will remain Apache 2.0. You can read it, fork it,
  embed it in commercial products, and ship derivatives without
  asking for permission.
- **BSL — future enterprise features.** Forthcoming modules targeting
  fleet management, multi-tenant orchestration, hardened compliance
  audits, and managed cloud control planes will be released under the
  Business Source License with a fixed time-delayed conversion to
  Apache 2.0. Those modules will live in clearly marked subtrees and
  will never retroactively re-license existing core code.

If you are unsure which side of the line a change falls on, open an
issue first. We will tell you before you write any code.

## Pull Request Contract

Every PR must satisfy the same gate the CI pipeline enforces. There
are no exceptions — not for "trivial" changes, not for documentation
typos that touch code-adjacent files, not for refactors.

1. **Failing test first.** New behaviour begins with a `pytest` test
   under `/tests` that fails for the right reason. See
   [`.ai/routing.md`](.ai/routing.md) for the full LLM-and-human
   contributor protocol.
2. **Rust implementation.** Heavy logic belongs in `/src`. Python is
   orchestration glue only.
3. **PyO3 surface.** Expose the new behaviour via `src/lib.rs` and
   map Rust `Err` to the most specific Python exception.
4. **Local verification before pushing:**
   ```bash
   .venv/bin/maturin develop
   uv run --no-sync pytest tests/ -v
   ```
   The full suite must pass on your machine before the PR is opened.
5. **CI must be green.** The matrix in `.github/workflows/ci.yml`
   runs the same `uv run maturin develop` + `uv run pytest tests/ -v`
   pipeline on Ubuntu, macOS, and Windows. A red CI is a blocking
   review comment regardless of the change's apparent scope.

Commit messages should describe the behaviour change in the
imperative ("add MCP destructive-prefix interceptor"), not the
mechanics ("edit mcp.rs"). Squash before merging.

## Reporting Security Issues

Do not open a public issue for a security report. Email the
maintainers directly; we will coordinate disclosure.
