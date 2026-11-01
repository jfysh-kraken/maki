//! Slice 02.2a acceptance: one Maki PID owns two independent session runtimes,
//! each with its own event pump and conversation context, with no
//! current-session slot and no per-session process fallback.
//!
//! Fake client only, and no prompt is ever sent to a live session: no
//! inference, provider keys or network. What this fixture deliberately leaves
//! for later slices (not silent gaps):
//! - prompt/event routing proof (interleaving, exactly-once settlement,
//!   duplicate-prompt and unknown/cross-session rejection) is 02.2b (see
//!   `tests/prompt_routing.rs` and the `server.rs` routing tests);
//! - `_maki/session/allocate` and `_maki/session/activate` runtime behavior
//!   is 02.3 (see `tests/allocation_activation.rs` and the `server.rs`
//!   lease tests); addressed cancel scoping is 02.9a, bounded release is
//!   02.10b: those two probes must still answer -32601 (see
//!   `extension_runtime_seams_still_reject_clearly`).
//!
//! 02.2b changed the per-session observable used here: setup is fixed per
//! session, so a mode change routes to its session and rejects with
//! `fixed_at_activation` instead of emitting a `currentModeId` update. Pump
//! addressing itself stays covered by the `two_runtimes_coexist` unit test,
//! which drives both pumps with synthetic events.

mod support;

use maki_acp::extension;
use support::SharedSessions;

/// Both runtimes coexist under the same Maki PID: opening the second session
/// never closes or disturbs the first, and each session routes only its own
/// operations. One child process by construction, so one PID by construction.
/// 02.2b: setup is fixed per session, so each mode change routes to its
/// session and rejects with `fixed_at_activation`; an unknown id rejects as
/// `resource_not_found` without touching either runtime.
#[tokio::test]
async fn two_sessions_coexist_under_one_pid_with_independent_pumps() {
    let mut shared = SharedSessions::establish().await.unwrap();
    assert_ne!(shared.a, shared.b, "two runtimes hold distinct ids");
    assert_eq!(
        shared.cwd_a,
        shared.projects.a.to_string_lossy(),
        "session A runs on project A"
    );
    assert_eq!(
        shared.cwd_b,
        shared.projects.b.to_string_lossy(),
        "session B runs on project B"
    );

    // A is still usable after B opened: the registry never replaced it, and
    // each session routes its own change to the fixed-at-activation rejection.
    for (session, mode) in [(&shared.a, "plan"), (&shared.b, "build")] {
        let reply = shared.child.set_mode_raw(session, mode).await.unwrap();
        assert_eq!(
            reply["error"]["code"], -32602,
            "{session} routes its own mode change: {reply}"
        );
        assert_eq!(
            reply["error"]["data"]["reason"], "fixed_at_activation",
            "{session} rejects post-activation changes: {reply}"
        );
    }

    // An unknown id rejects without touching either runtime.
    let unknown = format!("{}-missing", shared.a);
    let missing = shared.child.set_mode_raw(&unknown, "plan").await.unwrap();
    assert_eq!(
        missing["error"]["code"], -32002,
        "an unknown session rejects: {missing}"
    );
    assert_eq!(missing["error"]["data"]["sessionId"], unknown);

    // Both sessions still route afterwards: neither rejection disturbed a peer.
    for session in [&shared.a, &shared.b] {
        let reply = shared
            .child
            .set_config_raw(session, "openai/gpt-5")
            .await
            .unwrap();
        assert_eq!(
            reply["error"]["data"]["reason"], "fixed_at_activation",
            "{session} still routes after the peer rejected: {reply}"
        );
    }

    shared.shutdown().await.unwrap();
}

/// Addressed cancel and active-session release stay specified but not
/// implemented: the cancel probe answers -32601 like an unknown method, on a
/// multi-session server too. A bare release probe fails param parsing first
/// (see `extension_contract.rs`); the release seam itself — active sessions
/// staying -32601 — is proven in `allocation_activation.rs`. Allocate and
/// activate moved out of this list in 02.3.
#[tokio::test]
async fn extension_runtime_seams_still_reject_clearly() {
    let mut shared = SharedSessions::establish().await.unwrap();
    let reply = shared
        .child
        .probe_unsupported(extension::CANCEL_METHOD)
        .await
        .unwrap();
    assert_eq!(
        reply["error"]["code"],
        -32601,
        "{} must reject as unknown, got: {reply}",
        extension::CANCEL_METHOD
    );
    assert!(
        reply.get("result").is_none(),
        "{} must not answer success: {reply}",
        extension::CANCEL_METHOD
    );
    shared.shutdown().await.unwrap();
}
