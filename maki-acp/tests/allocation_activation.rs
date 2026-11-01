//! Slice 02.3 acceptance: allocate/activate over the live wire, with the
//! requested model fixed before any provider connection.
//!
//! Fake client only, and no prompt is ever sent to a live session: a prompt
//! to an activated session would reach the provider, and fixtures use no
//! keys, inference, or network. What this fixture proves at the wire:
//! - `initialize` advertises allocate+activate+cancel+release;
//! - allocate reserves distinct ids with no provider/task, and a prompt to
//!   an unactivated allocation is rejected without creating either;
//! - activate answers `{sessionId, turnEpoch: 0}` plus a usable `active`
//!   lifecycle frame, with the requested model fixed per session;
//! - failed activation, release of an unactivated allocation, and the lease
//!   lifecycle behave as specified; the peer stays usable throughout;
//! - addressed cancel plus active release (02.9a/02.10b) answer over the
//!   wire with the deferred release outcome and a `released` lifecycle.
//!
//! What it deliberately leaves for named later slices (not silent gaps):
//! - construction-order instrumentation (which spec reached the first
//!   provider connection) is proven at unit level in `server.rs`, where the
//!   starter is a fixture: the live child cannot hold or observe its own
//!   startup without a test hook in the binary;
//! - held-startup responsiveness is proven at unit level with a gated
//!   starter; the live child proves the same dispatch shape with immediate
//!   startup (peer operations correlate by id while both sessions live);
//! - initial cwd/environment content is 02.4, permission-answer routing is
//!   02.8, cancel scoping is 02.9a, supervision/release bounds are 02.10a/b.

mod support;

use maki_acp::extension;
use support::{MakiChild, ProjectPair};

/// Distinct requested models for the two live sessions. Both parse offline
/// (the harness boots on MODEL_A with no keys), and neither prompt is ever
/// sent, so no provider is contacted either way.
const MODEL_A: &str = "openai/gpt-5";
const MODEL_B: &str = "openai/gpt-5.6-sol";

async fn lifecycle_active(child: &mut MakiChild, session: &str) -> serde_json::Value {
    child
        .expect_notification(|frame| {
            frame["method"] == extension::LIFECYCLE_NOTIFICATION
                && frame["params"]["sessionId"] == session
        })
        .await
        .unwrap()
}

/// Allocate on both fixture projects, then activate each with its requested
/// model. Every wait is a read-until-match with the frame bound, never a
/// sleep: correlation itself is the two-session proof.
#[tokio::test]
async fn allocate_then_activate_two_sessions_with_distinct_models() {
    let projects = ProjectPair::create().unwrap();
    let mut child = MakiChild::spawn().await.unwrap();
    let initialize = child.initialize().await.unwrap();
    let advertised = &initialize["result"]["agentCapabilities"]["_meta"]["maki.multiSession"];
    assert_eq!(advertised["version"], 1, "{advertised}");
    let methods: Vec<&str> = advertised["methods"]
        .as_array()
        .expect("methods is an array")
        .iter()
        .filter_map(|m| m.as_str())
        .collect();
    assert_eq!(
        methods,
        [
            extension::ALLOCATE_METHOD,
            extension::ACTIVATE_METHOD,
            extension::CANCEL_METHOD,
            extension::RELEASE_METHOD
        ],
        "02.10b advertises release alongside what 02.3/02.9a answer: {methods:?}"
    );

    let cwd_a = projects.a.to_string_lossy().into_owned();
    let cwd_b = projects.b.to_string_lossy().into_owned();
    let reply_a = child.allocate_raw(&cwd_a).await.unwrap();
    let reply_b = child.allocate_raw(&cwd_b).await.unwrap();
    let session_a = reply_a["result"]["sessionId"].as_str().unwrap();
    let session_b = reply_b["result"]["sessionId"].as_str().unwrap();
    assert_ne!(session_a, session_b, "allocations hold distinct ids");
    assert_eq!(reply_a["result"]["turnEpoch"], 0);
    assert_eq!(reply_b["result"]["turnEpoch"], 0);

    let output_a = projects.a.join("plan-a.md").to_string_lossy().into_owned();
    let output_b = projects.b.join("plan-b.md").to_string_lossy().into_owned();
    let active_a = child
        .activate_raw(session_a, MODEL_A, "build", &cwd_a, &output_a)
        .await
        .unwrap();
    assert_eq!(active_a["result"]["sessionId"], session_a);
    assert_eq!(active_a["result"]["turnEpoch"], 0);
    let active_b = child
        .activate_raw(session_b, MODEL_B, "plan", &cwd_b, &output_b)
        .await
        .unwrap();
    assert_eq!(active_b["result"]["sessionId"], session_b);

    for session in [session_a, session_b] {
        let frame = lifecycle_active(&mut child, session).await;
        assert_eq!(frame["params"]["state"], "active", "{frame}");
        assert_eq!(frame["params"]["turnEpoch"], 0, "{frame}");
        assert_eq!(frame["params"]["usable"], true, "{frame}");
    }

    // Activation does not reserve output: 03's transactional reservation
    // owns that, and nothing here may create the artifact early.
    assert!(
        !projects.a.join("plan-a.md").exists() && !projects.b.join("plan-b.md").exists(),
        "activate fixes the output path without creating it"
    );

    // Both sessions stay usable and fixed: post-activation changes route to
    // the addressed session and reject, and an unknown prompt still rejects
    // without touching either runtime.
    for (session, mode) in [(session_a, "plan"), (session_b, "build")] {
        let reply = child.set_mode_raw(session, mode).await.unwrap();
        assert_eq!(reply["error"]["code"], -32602, "{session}: {reply}");
        assert_eq!(reply["error"]["data"]["reason"], "fixed_at_activation");
    }
    let config = child.set_config_raw(session_a, MODEL_B).await.unwrap();
    assert_eq!(config["error"]["code"], -32602, "{config}");
    let unknown = format!("{session_a}-missing");
    let no_prompt = child.prompt_raw(&unknown, "hello").await.unwrap();
    assert_eq!(no_prompt["error"]["code"], -32002);
    assert_eq!(no_prompt["error"]["data"]["sessionId"], unknown);

    child.shutdown().await.unwrap();
}

/// A prompt to an unactivated allocation is rejected before anything exists
/// for it: no provider is constructed and no task spawns. A bad allocate is
/// rejected the same way, leasing nothing.
#[tokio::test]
async fn prompt_to_an_unactivated_allocation_rejects() {
    let projects = ProjectPair::create().unwrap();
    let mut child = MakiChild::spawn().await.unwrap();
    child.initialize().await.unwrap();

    let reply = child
        .allocate_raw(&projects.a.to_string_lossy())
        .await
        .unwrap();
    let session = reply["result"]["sessionId"].as_str().unwrap();
    let rejected = child.prompt_raw(session, "hello").await.unwrap();
    assert_eq!(rejected["error"]["code"], -32602, "{rejected}");
    assert_eq!(rejected["error"]["data"]["reason"], "not_activated");
    assert_eq!(rejected["error"]["data"]["sessionId"], session);

    let bad = child.allocate_raw("relative/project").await.unwrap();
    assert_eq!(bad["error"]["code"], -32602, "{bad}");
    assert_eq!(bad["error"]["data"]["reason"], "invalid_cwd");

    child.shutdown().await.unwrap();
}

/// A failed activation consumes its lease: the error is honest, and the id
/// is unknown afterwards. The peer (here: the absence of any other session)
/// is unaffected: the server still answers and shuts down cleanly.
#[tokio::test]
async fn failed_activate_consumes_the_lease() {
    let projects = ProjectPair::create().unwrap();
    let mut child = MakiChild::spawn().await.unwrap();
    child.initialize().await.unwrap();
    let cwd = projects.a.to_string_lossy().into_owned();

    let reply = child.allocate_raw(&cwd).await.unwrap();
    let session = reply["result"]["sessionId"].as_str().unwrap().to_owned();
    let output = projects.a.join("plan.md").to_string_lossy().into_owned();
    let failed = child
        .activate_raw(&session, "nope/not-a-provider", "build", &cwd, &output)
        .await
        .unwrap();
    assert_eq!(failed["error"]["code"], -32602, "{failed}");
    assert_eq!(failed["error"]["data"]["model"], "nope/not-a-provider");

    let again = child
        .activate_raw(&session, MODEL_A, "build", &cwd, &output)
        .await
        .unwrap();
    assert_eq!(again["error"]["code"], -32002, "the lease is gone: {again}");
    assert_eq!(again["error"]["data"]["sessionId"], session);
    let no_prompt = child.prompt_raw(&session, "hello").await.unwrap();
    assert_eq!(no_prompt["error"]["code"], -32002);

    child.shutdown().await.unwrap();
}

/// Explicit release forgets an unactivated allocation; releasing twice
/// answers `resource_not_found` (no tombstones in 02.3).
#[tokio::test]
async fn release_forgets_an_unactivated_allocation() {
    let projects = ProjectPair::create().unwrap();
    let mut child = MakiChild::spawn().await.unwrap();
    child.initialize().await.unwrap();

    let reply = child
        .allocate_raw(&projects.a.to_string_lossy())
        .await
        .unwrap();
    let session = reply["result"]["sessionId"].as_str().unwrap().to_owned();
    let released = child.release_raw(&session).await.unwrap();
    assert_eq!(released["result"]["released"], true);
    assert_eq!(released["result"]["sessionId"], session);

    let repeat = child.release_raw(&session).await.unwrap();
    assert_eq!(repeat["error"]["code"], -32002, "{repeat}");
    assert_eq!(repeat["error"]["data"]["sessionId"], session);

    child.shutdown().await.unwrap();
}

/// 02.9a addressed cancel plus 02.10b active release over the live wire:
/// cancel answers on an idle activated session (no prompt is ever sent, so
/// the provider wire stays untouched), then release settles the deferred
/// outcome and reports `released` while the routing entry drops — and an
/// unknown id answers `resource_not_found`.
#[tokio::test]
async fn cancel_answers_and_active_release_reports() {
    let projects = ProjectPair::create().unwrap();
    let mut child = MakiChild::spawn().await.unwrap();
    child.initialize().await.unwrap();
    let cwd = projects.a.to_string_lossy().into_owned();

    let reply = child.allocate_raw(&cwd).await.unwrap();
    let session = reply["result"]["sessionId"].as_str().unwrap().to_owned();
    let output = projects.a.join("plan.md").to_string_lossy().into_owned();
    let active = child
        .activate_raw(&session, MODEL_A, "build", &cwd, &output)
        .await
        .unwrap();
    assert!(active.get("error").is_none(), "{active}");
    lifecycle_active(&mut child, &session).await;

    // Idle cancel: nothing runs, so `cancelled` is false but the epoch still
    // bumps, and the session stays usable afterwards.
    let idle = child.cancel_raw(&session, None).await.unwrap();
    assert_eq!(idle["result"]["sessionId"], session, "{idle}");
    assert_eq!(idle["result"]["cancelled"], false, "{idle}");
    assert_eq!(idle["result"]["turnEpoch"], 1, "{idle}");
    let mode = child.set_mode_raw(&session, "plan").await.unwrap();
    assert_eq!(mode["error"]["data"]["reason"], "fixed_at_activation");

    // Release settles off-loop and answers once teardown reports: the idle
    // settle bumps the epoch once more, the entry drops, and the lifecycle
    // carries the client's release note.
    let released = child.release_raw(&session).await.unwrap();
    assert_eq!(released["result"]["sessionId"], session, "{released}");
    assert_eq!(released["result"]["released"], true, "{released}");
    assert_eq!(released["result"]["turnEpoch"], 2, "{released}");
    let frame = child
        .expect_notification(|frame| {
            frame["method"] == extension::LIFECYCLE_NOTIFICATION
                && frame["params"]["sessionId"] == session
        })
        .await
        .unwrap();
    assert_eq!(frame["params"]["state"], "released", "{frame}");
    assert_eq!(frame["params"]["turnEpoch"], 2, "{frame}");
    assert_eq!(frame["params"]["rawReason"], "client_done", "{frame}");
    assert_eq!(frame["params"]["usable"], false, "{frame}");

    // Re-release re-answers the recorded success; an unknown id is -32002.
    let repeat = child.release_raw(&session).await.unwrap();
    assert_eq!(repeat["result"]["released"], true, "{repeat}");
    assert_eq!(repeat["result"]["turnEpoch"], 2, "{repeat}");
    let unknown = format!("{session}-missing");
    let missing = child.release_raw(&unknown).await.unwrap();
    assert_eq!(missing["error"]["code"], -32002, "{missing}");
    assert_eq!(missing["error"]["data"]["sessionId"], unknown);

    child.shutdown().await.unwrap();
}
