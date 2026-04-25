//! Zero-Trust MCP client (STDIO transport scaffold).
//!
//! The Model Context Protocol is the agent's only sanctioned egress to
//! the outside world. In Nexus-Core that egress is policed inside Rust:
//! the security policy is consulted *before* any byte hits a transport,
//! so a misconfigured Python caller cannot accidentally bypass the
//! gatekeeper by reaching the transport directly. Phase 3.2 lays the
//! architectural groundwork — the policy matrix, the transport
//! abstraction, and the dispatch surface — while the actual JSON-RPC
//! pipe over child-process STDIO arrives in a later phase.

/// Policy levels enforced by the gatekeeper, ordered from permissive
/// to airtight. The variants are deliberately coarse: a finer policy
/// engine (per-tool ACLs, per-agent caps) will sit *behind* this enum
/// so that the load-bearing audit decision stays a single match arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityLevel {
    /// No gating. Reserved for fully sandboxed environments.
    AllowAll,
    /// Destructive operations require explicit out-of-band approval;
    /// read-only operations pass through.
    RequireApproval,
    /// Hard deny. Used as a kill switch when an agent has been flagged.
    DenyAll,
}

/// Transport abstraction. Today only `Mock` is wired so we can test
/// the gatekeeper deterministically; `Stdio` is reserved so the policy
/// surface above does not change when the real JSON-RPC pipe lands.
pub enum Transport {
    Mock,
    #[allow(dead_code)]
    Stdio { command: String },
}

/// Tool-name prefixes that escalate to "destructive" under
/// `RequireApproval`. The list is intentionally short and explicit —
/// allow-by-default would invert the Zero-Trust posture.
const DESTRUCTIVE_PREFIXES: &[&str] = &["write_", "delete_", "drop_", "update_"];

/// MCP client wrapped around a transport and a security policy. The
/// transport handle is owned (not borrowed) so the same `McpClient`
/// can be reused across many `invoke_tool` calls without re-opening
/// the underlying child process once `Stdio` is wired.
pub struct McpClient {
    transport: Transport,
    security_level: SecurityLevel,
}

impl McpClient {
    pub fn new(transport: Transport, security_level: SecurityLevel) -> Self {
        Self {
            transport,
            security_level,
        }
    }

    /// The gatekeeper. Every code path that reaches a transport goes
    /// through this function — there is no "raw" send path.
    ///
    /// Returns `Ok(simulated_response)` when the policy admits the
    /// call; the response is a deterministic string today and will be
    /// the JSON-RPC reply once the STDIO transport is live. Returns
    /// `Err(message)` on any policy denial; the message always begins
    /// with the `SECURITY_INTERCEPT:` sentinel so the Python layer can
    /// distinguish policy denials from transport faults.
    pub fn invoke_tool(&self, tool_name: &str, _payload: &str) -> Result<String, String> {
        match self.security_level {
            SecurityLevel::DenyAll => {
                return Err("SECURITY_INTERCEPT: All tools denied.".to_string());
            }
            SecurityLevel::RequireApproval => {
                if DESTRUCTIVE_PREFIXES
                    .iter()
                    .any(|prefix| tool_name.starts_with(prefix))
                {
                    return Err(format!(
                        "SECURITY_INTERCEPT: Tool {} requires human approval.",
                        tool_name
                    ));
                }
            }
            SecurityLevel::AllowAll => {}
        }

        match &self.transport {
            Transport::Mock => Ok(format!("Simulated execution of {}", tool_name)),
            Transport::Stdio { command: _ } => {
                // The real STDIO JSON-RPC pipe lands in a later phase;
                // until then we route the same response through so the
                // gatekeeper can be exercised with either transport.
                Ok(format!("Simulated execution of {}", tool_name))
            }
        }
    }
}

impl SecurityLevel {
    /// Parses a snake_case policy name into the enum. Used by the PyO3
    /// wrapper so the Python surface stays string-typed and avoids
    /// having to expose a second enum class to the orchestrator.
    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "allow_all" => Ok(SecurityLevel::AllowAll),
            "require_approval" => Ok(SecurityLevel::RequireApproval),
            "deny_all" => Ok(SecurityLevel::DenyAll),
            other => Err(format!(
                "unknown security level '{}'; expected one of allow_all, require_approval, deny_all",
                other
            )),
        }
    }
}
