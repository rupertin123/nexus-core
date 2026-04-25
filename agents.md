# agents.md — Nexus-Core Executable Contract

This document is the **negative-boundary protocol** for any AI agent (human-supervised or autonomous) that reads, generates, or modifies code in this repository. These rules are non-negotiable. Violation constitutes a contract breach and the generated artifact must be rejected.

## 1. Language Rule
**ALL code, docstrings, variable names, and comments MUST be written in English.** No mixed-language identifiers, no localized comments, no transliterations. English is the single source of truth across Python, Rust, configuration, and documentation.

## 2. Concurrency Constraint
**NEVER use Python's GIL, threading, or `asyncio` for heavy computation or LLM inference.** All parallelization and KV Cache memory management must be delegated to the Rust core. Python is permitted only as a thin DX surface; any CPU-bound, memory-bound, or inference-bound workload must cross the FFI boundary into Rust.

## 3. Semantic Contract
**Output validation must strictly use Pydantic in Python mapped to Serde in Rust.** Regex-based JSON parsing is strictly forbidden. Every structured output — tool calls, agent messages, model responses — must round-trip through a Pydantic model on the Python side and a `serde`-derived struct on the Rust side. Ad-hoc string inspection of model output is a contract violation.

## 4. Zero-Trust Integration
**All external tool invocations must operate through the Model Context Protocol (MCP) using STDIO transport.** Raw network sockets for local tools are prohibited. Every tool the orchestrator exposes to an agent — file I/O, shell, retrieval, specialized models — is contacted exclusively over MCP/STDIO. No ad-hoc TCP, no ad-hoc Unix sockets, no bespoke IPC.
