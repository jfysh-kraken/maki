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

/// 02.3 implements allocate/activate, 02.9a implements addressed cancel,
/// 02.10b implements bounded active-session release, and 04.1a implements
/// model discovery, so the capabilities are advertised with exactly those
/// methods. Owned deletion (05.2) stays specified but not implemented: its
/// probe answers -32601 instead of hanging or answering success.
#[tokio::test]
async fn extension_capability_advertises_allocate_activate_cancel_and_release() {
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
        "capability lists only what the server answers: {methods:?}"
    );
    let discovery =
        &initialize["result"]["agentCapabilities"]["_meta"][extension::MODEL_DISCOVERY_CAPABILITY];
    assert_eq!(discovery["version"], 1, "{discovery}");
    let discovery_methods: Vec<&str> = discovery["methods"]
        .as_array()
        .expect("methods is an array")
        .iter()
        .filter_map(|m| m.as_str())
        .collect();
    assert_eq!(
        discovery_methods,
        [extension::MODEL_LIST_METHOD],
        "discovery is advertised only because its method answers: {discovery_methods:?}"
    );
    for method in [extension::OWNED_DELETION_METHOD] {
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
    // Release parses before it routes: empty params fail parsing (-32602),
    // while an unknown id routes past parsing to `resource_not_found`.
    // Active-session release itself is proven in `allocation_activation.rs`.
    let empty = child
        .probe_unsupported(extension::RELEASE_METHOD)
        .await
        .unwrap();
    assert_eq!(empty["error"]["code"], -32602, "{empty}");
    child.shutdown().await.unwrap();
}

/// 04.1a model discovery: `_maki/model/list_models` answers on the existing
/// process with selectable full IDs before any session exists. No second
/// process, no dummy session, no aliases or tiers.
#[tokio::test]
async fn model_discovery_lists_full_ids_without_a_session() {
    let mut child = MakiChild::spawn().await.unwrap();
    child.initialize().await.unwrap();
    let reply = child
        .probe_unsupported(extension::MODEL_LIST_METHOD)
        .await
        .unwrap();
    assert!(
        reply.get("error").is_none(),
        "discovery must answer, not reject: {reply}"
    );
    let models = reply["result"]["models"]
        .as_array()
        .unwrap_or_else(|| panic!("discovery result needs a models array: {reply}"));
    assert!(
        !models.is_empty(),
        "discovery must list at least one selectable model: {reply}"
    );
    let mut previous: Option<&str> = None;
    for model in models {
        let id = model
            .as_str()
            .unwrap_or_else(|| panic!("model IDs are strings: {model}"));
        assert!(!id.is_empty(), "no empty model IDs");
        assert!(
            !id.chars().any(char::is_whitespace),
            "full IDs carry no whitespace: {id}"
        );
        assert!(
            id.contains('/'),
            "selectable full IDs name provider and model: {id}"
        );
        if let Some(prev) = previous {
            assert!(
                prev <= id,
                "discovery answers deterministically sorted: {prev} before {id}"
            );
        }
        previous = Some(id);
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
    let mode_reply = peer.set_mode_raw(&peer_session, "plan").await.unwrap();
    assert!(mode_reply.get("result").is_some(), "{mode_reply}");

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
    let reply_a = sessions
        .a
        .child
        .set_mode_raw(&session_a, "plan")
        .await
        .unwrap();
    assert!(reply_a.get("result").is_some(), "{reply_a}");
    let reply_b = sessions
        .b
        .child
        .set_mode_raw(&session_b, "build")
        .await
        .unwrap();
    assert!(reply_b.get("result").is_some(), "{reply_b}");

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
