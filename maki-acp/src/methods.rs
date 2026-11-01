use agent_client_protocol_schema::ProtocolVersion;
use agent_client_protocol_schema::v1::{
    AgentCapabilities, Implementation, InitializeResponse, LoadSessionResponse, McpCapabilities,
    NewSessionResponse, PromptCapabilities, SessionCapabilities, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOption, SessionListCapabilities, SessionMode,
    SessionModeId, SessionModeState,
};

use crate::extension;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const MODE_BUILD: &str = "build";
const MODE_PLAN: &str = "plan";

pub const MODEL_CONFIG_ID: &str = "model";

pub fn initialize_response() -> InitializeResponse {
    let mut capabilities = AgentCapabilities::new()
        .load_session(true)
        .prompt_capabilities(PromptCapabilities::new().image(true).embedded_context(true))
        .mcp_capabilities(McpCapabilities::new().http(true))
        .session_capabilities(SessionCapabilities::new().list(SessionListCapabilities::new()));
    // Fork extension: 02.3 advertises allocate+activate (see
    // `extension::capability_meta`); 02.9a/02.10b extend the method list.
    // Stock clients ignore `_meta`, so this stays standard-compatible.
    capabilities.meta = Some(extension::capability_meta());
    InitializeResponse::new(ProtocolVersion::V1)
        .agent_capabilities(capabilities)
        .auth_methods(vec![])
        .agent_info(Implementation::new("maki", VERSION))
}

pub fn mode_state(current: &str) -> SessionModeState {
    SessionModeState::new(
        SessionModeId::from(current.to_string()),
        vec![
            SessionMode::new(SessionModeId::from(MODE_BUILD.to_string()), "Build"),
            SessionMode::new(SessionModeId::from(MODE_PLAN.to_string()), "Plan"),
        ],
    )
}

pub fn new_session_response(session_id: &str) -> NewSessionResponse {
    NewSessionResponse::new(session_id.to_string()).modes(mode_state(MODE_BUILD))
}

pub fn load_session_response() -> LoadSessionResponse {
    LoadSessionResponse::new().modes(mode_state(MODE_BUILD))
}

pub fn model_config_option(current: &str, specs: &[String]) -> SessionConfigOption {
    let mut options: Vec<SessionConfigSelectOption> = specs
        .iter()
        .map(|spec| SessionConfigSelectOption::new(spec.clone(), spec.clone()))
        .collect();
    if !specs.iter().any(|spec| spec == current) {
        options.insert(
            0,
            SessionConfigSelectOption::new(current.to_string(), current.to_string()),
        );
    }
    SessionConfigOption::select(MODEL_CONFIG_ID, "Model", current.to_string(), options)
        .category(SessionConfigOptionCategory::Model)
}

pub fn mode_id_to_agent_mode(mode_id: &str) -> Option<maki_agent::AgentMode> {
    match mode_id {
        MODE_BUILD => Some(maki_agent::AgentMode::Build),
        MODE_PLAN => {
            let storage = maki_storage::StateDir::resolve().ok()?;
            let plan_path = maki_storage::plans::new_plan_path(&storage).ok()?;
            Some(maki_agent::AgentMode::Plan(plan_path))
        }
        _ => None,
    }
}

/// Extension result shapes (02.3). These methods have no schema types, so the
/// results are plain JSON: `{sessionId, turnEpoch}` with the epoch the lease
/// holds (0: no prompt has ever run on an allocation). Release of an
/// unactivated allocation reports the same unmoved epoch; the bumped release
/// epoch in `EXTENSION.md` belongs to 02.10b's active-session release.
pub fn allocate_result(session_id: &str) -> serde_json::Value {
    serde_json::json!({
        "sessionId": session_id,
        "turnEpoch": extension::TurnEpoch::initial().current(),
    })
}

/// Successful activation answers the same shape as allocation: the session
/// starts at epoch 0, and readiness follows on `_maki/session/lifecycle`.
pub fn activate_result(session_id: &str) -> serde_json::Value {
    allocate_result(session_id)
}

/// Release of an unactivated allocation: the lease is gone, nothing ever ran.
pub fn release_allocation_result(session_id: &str) -> serde_json::Value {
    serde_json::json!({
        "sessionId": session_id,
        "released": true,
        "turnEpoch": extension::TurnEpoch::initial().current(),
    })
}

/// Lifecycle params for a session whose activation finished: `active` at epoch
/// 0, usable for prompts. Failed activation sends no lifecycle frame: the
/// activation error itself is the honest terminal, and the lease is gone.
pub fn lifecycle_active_params(session_id: &str) -> serde_json::Value {
    serde_json::json!({
        "sessionId": session_id,
        "state": "active",
        "turnEpoch": extension::TurnEpoch::initial().current(),
        "rawReason": null,
        "usable": true,
    })
}
