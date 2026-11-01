//! Slice 02.2b acceptance: prompt and event routing by session under one Maki
//! PID, with exactly-once settlement and strict identity rejection.
//!
//! Fake client only. No prompt is ever sent to a live session: such a prompt
//! would reach the provider, and fixtures use no keys, inference, or network.
//! Prompt flows (duplicate rejection, terminal settlement, another turn after
//! idle, stale/cross-session answers) are covered at unit level instead, where
//! `server.rs` tests drive real pumps with synthetic agent events. This
//! fixture proves what needs the live child:
//! - interleaved operations on two sessions under one PID correlate by
//!   request id and route to the session each names;
//! - unknown identities reject before anything is queued, without touching
//!   either live runtime;
//! - post-activation model/mode changes route, then reject with
//!   `fixed_at_activation`, leaving the fixed setup and the peer alone;
//! - an idle session stays usable after a cancel dispatch and a rejection;
//! - the extension runtime seams still answer -32601 (02.3/02.9a/02.10b own
//!   them; see also `extension_runtime_seams_still_reject_clearly`).
//!
//! Every wait is a read-until-match with the frame bound, never a sleep:
//! correlation itself is the interleaving proof, since peer traffic buffers
//! separately and never unblocks the wrong waiter.

mod support;

use maki_acp::extension;
use serde_json::json;
use support::SharedSessions;

const FIXED_AT_ACTIVATION: &str = "fixed_at_activation";

fn assert_fixed(reply: &serde_json::Value, session: &str) {
    assert_eq!(
        reply["error"]["code"], -32602,
        "{session} routes its own change: {reply}"
    );
    assert_eq!(
        reply["error"]["data"]["reason"], FIXED_AT_ACTIVATION,
        "{session} rejects post-activation changes: {reply}"
    );
    assert!(
        reply.get("result").is_none(),
        "{session} must not answer success: {reply}"
    );
}

/// Interleaved operations on two sessions under the same PID correlate by
/// request id and route to the session each names: A and B reject their own
/// changes out of order, an unknown prompt rejects in the middle, and an idle
/// cancel changes nothing. No inference runs anywhere in the sequence.
#[tokio::test]
async fn two_sessions_route_independently_under_one_pid() {
    let mut shared = SharedSessions::establish().await.unwrap();
    assert_ne!(shared.a, shared.b, "two runtimes hold distinct ids");

    // Both requests are in flight before either reply is collected, so arrival
    // order cannot fake the correlation: each reply must carry its own id.
    let id_a = shared.child.fresh_id();
    shared
        .child
        .send_request(
            "session/set_mode",
            id_a.clone(),
            json!({"sessionId": &shared.a, "modeId": "plan"}),
        )
        .await
        .unwrap();
    let id_b = shared.child.fresh_id();
    shared
        .child
        .send_request(
            "session/set_config_option",
            id_b.clone(),
            json!({"sessionId": &shared.b, "configId": "model", "value": "openai/gpt-5"}),
        )
        .await
        .unwrap();

    // Collected B-first on purpose: correlation is by id, not arrival.
    let reply_b = shared.child.expect_response(&id_b).await.unwrap();
    assert_eq!(reply_b["id"], id_b, "B's reply carries B's id: {reply_b}");
    assert_fixed(&reply_b, "session B");
    let reply_a = shared.child.expect_response(&id_a).await.unwrap();
    assert_eq!(reply_a["id"], id_a, "A's reply carries A's id: {reply_a}");
    assert_fixed(&reply_a, "session A");

    // An unknown prompt rejects in the middle of live traffic, before any run
    // exists for it, so no provider is ever contacted.
    let unknown = format!("{}-missing", shared.a);
    let no_prompt = shared.child.prompt_raw(&unknown, "hello").await.unwrap();
    assert_eq!(
        no_prompt["error"]["code"], -32002,
        "an unknown prompt rejects: {no_prompt}"
    );
    assert_eq!(no_prompt["error"]["data"]["sessionId"], unknown);

    // An idle cancel dispatches to its session and changes nothing observable:
    // the session routes its next operation exactly as before.
    shared.child.cancel_notify(&shared.a).await.unwrap();
    let again_a = shared.child.set_mode_raw(&shared.a, "build").await.unwrap();
    assert_fixed(&again_a, "session A after an idle cancel");
    let again_b = shared
        .child
        .set_config_raw(&shared.b, "openai/gpt-5")
        .await
        .unwrap();
    assert_fixed(&again_b, "session B after the peer's cancel");

    shared.shutdown().await.unwrap();
}

/// Unknown identities reject without touching either live runtime: unknown
/// prompts, mode/config changes, and cancels all leave both sessions exactly
/// as usable as before.
#[tokio::test]
async fn unknown_identities_reject_without_touching_either_session() {
    let mut shared = SharedSessions::establish().await.unwrap();
    let unknown = format!("{}-missing", shared.b);

    let no_prompt = shared.child.prompt_raw(&unknown, "hello").await.unwrap();
    assert_eq!(no_prompt["error"]["code"], -32002);
    assert_eq!(no_prompt["error"]["data"]["sessionId"], unknown);

    let no_mode = shared.child.set_mode_raw(&unknown, "plan").await.unwrap();
    assert_eq!(
        no_mode["error"]["code"], -32002,
        "an unknown mode change rejects by routing first: {no_mode}"
    );
    assert_eq!(no_mode["error"]["data"]["sessionId"], unknown);

    let no_config = shared
        .child
        .set_config_raw(&unknown, "openai/gpt-5")
        .await
        .unwrap();
    assert_eq!(
        no_config["error"]["code"], -32002,
        "an unknown config change rejects by routing first: {no_config}"
    );
    assert_eq!(no_config["error"]["data"]["sessionId"], unknown);

    // A notification carries no id to answer on, so an unknown cancel is only
    // dropped; the live sessions must be unaffected either way.
    shared.child.cancel_notify(&unknown).await.unwrap();

    for session in [&shared.a, &shared.b] {
        let reply = shared.child.set_mode_raw(session, "plan").await.unwrap();
        assert_fixed(&reply, session);
    }

    shared.shutdown().await.unwrap();
}

/// Post-activation changes address their session, then reject with the
/// contract reason: neither the fixed setup nor the peer moves, and routing
/// an unknown id still answers `resource_not_found` first.
#[tokio::test]
async fn post_activation_changes_route_then_reject() {
    let mut shared = SharedSessions::establish().await.unwrap();

    for session in [&shared.a, &shared.b] {
        let mode = shared.child.set_mode_raw(session, "plan").await.unwrap();
        assert_fixed(&mode, session);
        let config = shared
            .child
            .set_config_raw(session, "openai/gpt-5")
            .await
            .unwrap();
        assert_fixed(&config, session);
    }

    let unknown = format!("{}-missing", shared.a);
    let missing = shared.child.set_mode_raw(&unknown, "plan").await.unwrap();
    assert_eq!(
        missing["error"]["code"], -32002,
        "routing comes before the fixed-setup rule: {missing}"
    );

    // Repeating the same changes rejects identically: the first rejection
    // changed no setup on either session.
    for session in [&shared.a, &shared.b] {
        let mode = shared.child.set_mode_raw(session, "build").await.unwrap();
        assert_fixed(&mode, session);
    }

    // The runtime seams stay specified-but-unimplemented: 02.3/02.9a/02.10b
    // own them, and routing must not have started answering them.
    for method in [
        extension::ALLOCATE_METHOD,
        extension::ACTIVATE_METHOD,
        extension::CANCEL_METHOD,
        extension::RELEASE_METHOD,
    ] {
        let reply = shared.child.probe_unsupported(method).await.unwrap();
        assert_eq!(
            reply["error"]["code"], -32601,
            "{method} must reject as unknown, got: {reply}"
        );
        assert!(
            reply.get("result").is_none(),
            "{method} must not answer success: {reply}"
        );
    }

    shared.shutdown().await.unwrap();
}
