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
    // `extension::capability_meta`); 02.9a adds cancel, 02.10b adds release.
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

/// 02.10b active-session release result, sent by the background release task
/// once bounded teardown settles (the routing loop never waits on it). The
/// epoch is the post-release one: the cancel settlement or loss snapshot in
/// the release prologue bumps exactly once, so a racing terminal lands stale.
pub fn release_result(session_id: &str, turn_epoch: u64) -> serde_json::Value {
    serde_json::json!({
        "sessionId": session_id,
        "released": true,
        "turnEpoch": turn_epoch,
    })
}

/// Release of an unactivated allocation: the lease is gone, nothing ever ran.
pub fn release_allocation_result(session_id: &str) -> serde_json::Value {
    serde_json::json!({
        "sessionId": session_id,
        "released": true,
        "turnEpoch": extension::TurnEpoch::initial().current(),
    })
}

/// 04.1a model discovery result: selectable full `provider/model-id` specs
/// from the existing process and provider catalogue. No aliases or tiers;
/// listing a model is not proof an inference call succeeds.
pub fn model_list_result(models: &[String]) -> serde_json::Value {
    serde_json::json!({
        "models": models,
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

/// 02.10a supervised-loss params: same 02.1 lifecycle fields, new `lost`
/// state only. `turnEpoch` is the post-loss bumped epoch, `rawReason` the
/// verbatim cause, `usable` always false. A lost handle never reopens;
/// 02.10b's release reports this prior loss plus its own cleanup result.
pub fn lifecycle_lost_params(
    session_id: &str,
    turn_epoch: u64,
    raw_reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "sessionId": session_id,
        "state": "lost",
        "turnEpoch": turn_epoch,
        "rawReason": raw_reason,
        "usable": false,
    })
}

/// 02.10b successful-release params. `rawReason` echoes the client's release
/// note (`client_done` by default; `shutdown`/`replaced` on the close paths).
/// `usable` is always false: the routing entry is gone and the claim is
/// dropped. A release of a supervised-dead handle carries `priorLoss` with
/// the recorded loss reason/raw, so one frame reports both the prior loss
/// and the cleanup result without reviving the handle.
pub fn lifecycle_released_params(
    session_id: &str,
    turn_epoch: u64,
    reason: &str,
    prior_loss: Option<(&str, &str)>,
) -> serde_json::Value {
    let mut params = serde_json::json!({
        "sessionId": session_id,
        "state": "released",
        "turnEpoch": turn_epoch,
        "rawReason": reason,
        "usable": false,
    });
    if let Some((loss_reason, loss_raw)) = prior_loss {
        params["priorLoss"] = serde_json::json!({
            "reason": loss_reason,
            "rawReason": loss_raw,
        });
    }
    params
}

/// 02.10b failed-teardown params. Same frame shape as success but `state` is
/// `release_failed` and `stage` names the bounded stage that gave up
/// (`hook`, `mcp`, or `release` for the overall cap) with the verbatim cause
/// in `rawReason`. The entry is still dropped and the claim still released —
/// the failure is reported, never retried here — but no `released` frame is
/// ever sent for the session, so no successful release is claimed.
pub fn lifecycle_release_failed_params(
    session_id: &str,
    turn_epoch: u64,
    reason: &str,
    stage: &str,
    raw_reason: &str,
    prior_loss: Option<(&str, &str)>,
) -> serde_json::Value {
    let mut params = serde_json::json!({
        "sessionId": session_id,
        "state": "release_failed",
        "turnEpoch": turn_epoch,
        "rawReason": raw_reason,
        "stage": stage,
        "usable": false,
        "releaseNote": reason,
    });
    if let Some((loss_reason, loss_raw)) = prior_loss {
        params["priorLoss"] = serde_json::json!({
            "reason": loss_reason,
            "rawReason": loss_raw,
        });
    }
    params
}
