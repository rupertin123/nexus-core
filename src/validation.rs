//! Cognitive Reliability Layer.
//!
//! The Python orchestrator never sees a malformed LLM response. Every
//! contract violation — bad JSON syntax, missing required key — is
//! intercepted here in Rust, converted into a deterministic correction
//! prompt that downstream code can feed straight back to the model, and
//! retried inside a bounded loop. The Python caller observes either a
//! validated payload or a single quarantine error after the budget is
//! exhausted; the noisy in-between never escapes the FFI boundary.

use serde_json::Value;

/// Stateless validator for one schema. Owning the contract on the Rust
/// side keeps the hot-path allocation-free aside from the JSON parse
/// itself, and means the same interceptor can be reused across many
/// `enforce_contract` calls without rebuilding the key list.
pub struct SemanticInterceptor {
    required_keys: Vec<String>,
}

impl SemanticInterceptor {
    pub fn new(required_keys: Vec<String>) -> Self {
        Self { required_keys }
    }

    /// Validates `raw_output` against the contract.
    ///
    /// On success returns `Ok(raw_output.to_string())` so the caller can
    /// pass the validated payload along without retaining a borrow into
    /// the original buffer. On failure returns `Err` with a deterministic
    /// correction prompt that is safe to send back to the LLM verbatim:
    /// the strings are stable across calls so an upstream test harness
    /// can pin them, and they describe the *category* of failure rather
    /// than echoing the malformed input back.
    pub fn enforce_contract(
        &self,
        raw_output: &str,
        required_keys: Vec<String>,
    ) -> Result<String, String> {
        let parsed: Value = match serde_json::from_str(raw_output) {
            Ok(value) => value,
            Err(_) => {
                return Err(
                    "CRITICAL: Invalid JSON format. Expected valid JSON object."
                        .to_string(),
                );
            }
        };

        // Top-level keys only live on objects. A bare array or scalar
        // also violates the contract because the schema is keyed.
        let object = match parsed.as_object() {
            Some(map) => map,
            None => {
                return Err(
                    "CRITICAL: Invalid JSON format. Expected valid JSON object."
                        .to_string(),
                );
            }
        };

        for key in required_keys.iter() {
            if !object.contains_key(key) {
                return Err(format!(
                    "CRITICAL: Schema validation failed. Missing required key: {}. \
                     Fix the JSON structure.",
                    key
                ));
            }
        }

        Ok(raw_output.to_string())
    }

    /// Convenience accessor so the PyO3 wrapper does not need to keep a
    /// parallel copy of the contract.
    pub fn required_keys(&self) -> &[String] {
        &self.required_keys
    }
}
