//! Slice 02.1 acceptance checks: the fork extension contract is checked in
//! (`maki-acp/EXTENSION.md`, names in `maki_acp::extension`), and this
//! deterministic harness exercises initialize/capability/correlation,
//! clear rejection of unsupported capabilities, hold-and-observe peer
//! sync, the shared two-session + A/B-project helpers, and blocking-mode
//! launch. Fake client (this harness) and no provider contact: no prompt
//! is ever sent, so nothing here needs keys, inference or network.

mod support;

use maki_acp::extension;
use serde_json::json;
use test_case::test_case;

use support::AGENT_NAME;
use support::AGENT_VERSION;
use support::MakiChild;
use support::PROTOCOL_VERSION;
use support::TwoSession;

/// `initialize` answers are correlated by id for both id shapes, carry
/// protocol 1 and the maki 0.7.0 identity.
#[test_case(serde_json::json!(11) ; "numeric_id")]
#[test_case(serde_json::json!("init-second") ; "string_id")]
#[tokio::test]
async fn initialize_correlates_both_id_shapes(id: serde_json::Value) {
    let mut child = MakiChild::spawn().await.unwrap();
    child
        .send_request(
            "initialize",
            id.clone(),
            json!({"protocolVersion": PROTOCOL_VERSION, "clientCapabilities": {}}),
        )
        .await
        .unwrap();
    let reply = child.expect_response(&id).await.unwrap();
    assert!(reply.get("error").is_none(), "initialize failed: {reply}");
    assert_eq!(reply["id"], id);
    assert_eq!(reply["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(reply["result"]["agentInfo"]["name"], AGENT_NAME);
    assert_eq!(reply["result"]["agentInfo"]["version"], AGENT_VERSION);
    child.shutdown().await.unwrap();
}

/// The multi-session extension is specified but not implemented on this
/// branch, so capability checking must report it absent, and every
/// extension probe must reject with -32601 instead of hanging or
/// answering success.
#[tokio::test]
async fn unsupported_extension_capability_rejects_clearly() {
    let mut child = MakiChild::spawn().await.unwrap();
    let initialize = child.initialize().await.unwrap();
    let capability =
        initialize["result"]["agentCapabilities"]["_meta"].get(extension::CAPABILITY_KEY);
    assert!(
        capability.is_none(),
        "extension is specified-not-implemented here; advertising {capability:?} would lie"
    );
    for method in [
        extension::ALLOCATE_METHOD,
        extension::ACTIVATE_METHOD,
        extension::CANCEL_METHOD,
        extension::RELEASE_METHOD,
        extension::OWNED_DELETION_METHOD,
        &format!("{}list_models", extension::MODEL_DISCOVERY_PREFIX),
    ] {
        let reply = child.probe_unsupported(method).await.unwrap();
        assert_eq!(
            reply["error"]["code"], -32601,
            "{method} must reject as unknown, got: {reply}"
        );
        assert!(
            reply.get("result").is_none(),
            "{method} must not answer success: {reply}"
        );
    }
    child.shutdown().await.unwrap();
}

/// The harness holds one session's setup request outstanding while it
/// drives the peer session to completion, then collects the held reply by
/// id. No sleeps: every wait is a read-until-match with a failure bound.
#[tokio::test]
async fn holds_one_setup_while_observing_peer_responses() {
    let projects = support::ProjectPair::create().unwrap();
    let mut held = MakiChild::spawn().await.unwrap();
    let mut peer = MakiChild::spawn().await.unwrap();
    held.initialize().await.unwrap();
    peer.initialize().await.unwrap();

    let held_id = held.fresh_id();
    held.send_request(
        "session/new",
        held_id.clone(),
        json!({"cwd": projects.a, "mcpServers": []}),
    )
    .await
    .unwrap();

    let peer_session = peer
        .session_new(&projects.b.to_string_lossy())
        .await
        .unwrap();
    let list = peer.session_list(None).await.unwrap();
    assert!(
        list["sessions"].is_array(),
        "peer session/list must answer a session array: {list}"
    );
    peer.set_mode(&peer_session, "plan").await.unwrap();
    let peer_update = peer
        .expect_notification(|frame| {
            frame["method"] == "session/update"
                && frame["params"]["update"]["currentModeId"] == "plan"
        })
        .await
        .unwrap();
    assert_eq!(
        peer_update["params"]["sessionId"].as_str(),
        Some(peer_session.as_str())
    );

    let reply = held.expect_response(&held_id).await.unwrap();
    assert!(reply.get("error").is_none(), "held setup failed: {reply}");
    let held_session = reply["result"]["sessionId"].as_str().unwrap();
    assert_ne!(held_session, peer_session);

    held.shutdown().await.unwrap();
    peer.shutdown().await.unwrap();
}

/// The shared two-session + A/B-project helpers come up with distinct ids
/// on distinct cwds, and each peer independently drives its own session
/// (mode switch with the notification addressed to its session). Fresh
/// sessions persist on their first turn, so cwd-filtered listing proofs
/// wait for the fake-provider slices; the A/B wiring itself (which peer
/// runs on which project) is asserted here.
#[tokio::test]
async fn two_session_and_ab_project_helpers_smoke() {
    let mut sessions = TwoSession::establish().await.unwrap();
    assert_ne!(sessions.a.session_id, sessions.b.session_id);
    assert_eq!(
        sessions.a.cwd,
        sessions.projects.a.to_string_lossy(),
        "peer A must run on project A"
    );
    assert_eq!(
        sessions.b.cwd,
        sessions.projects.b.to_string_lossy(),
        "peer B must run on project B"
    );

    let session_a = sessions.a.session_id.clone();
    let session_b = sessions.b.session_id.clone();
    sessions.a.child.set_mode(&session_a, "plan").await.unwrap();
    sessions
        .b
        .child
        .set_mode(&session_b, "build")
        .await
        .unwrap();
    for (peer, expected) in [
        (&mut sessions.a.child, session_a.as_str()),
        (&mut sessions.b.child, session_b.as_str()),
    ] {
        let update = peer
            .expect_notification(|frame| frame["method"] == "session/update")
            .await
            .unwrap();
        assert_eq!(
            update["params"]["sessionId"].as_str(),
            Some(expected),
            "the mode notification must address its own session"
        );
    }

    sessions.shutdown().await.unwrap();
}

/// Every harness launch pins blocking mode on the wire argv; the
/// `background: false -> BackgroundMode::Blocking` mapping itself is
/// pinned by `blocking_mode_disables_detached_tools` in `server.rs`.
#[tokio::test]
async fn harness_launches_maki_in_blocking_mode() {
    let child = MakiChild::spawn().await.unwrap();
    let argv: Vec<String> = child
        .args()
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let position = argv
        .iter()
        .position(|arg| arg == "--background-timeout")
        .unwrap();
    assert_eq!(argv[position + 1], "0");
    child.shutdown().await.unwrap();
}
