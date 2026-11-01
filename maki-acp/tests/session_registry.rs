//! Slice 02.2a acceptance: one Maki PID owns two independent session runtimes,
//! each with its own event pump and conversation context, with no
//! current-session slot and no per-session process fallback.
//!
//! Fake client only, and no prompt is ever sent: no inference, provider keys
//! or network. What this fixture deliberately leaves for later slices (not
//! silent gaps):
//! - prompt/event routing proof (interleaving, exactly-once settlement,
//!   duplicate-prompt and unknown/cross-session rejection) is 02.2b;
//! - fixed-at-activation route-then-reject for post-activation model/mode
//!   changes is 02.2b (both sessions keep standard per-session behavior here);
//! - `_maki/session/allocate` and `_maki/session/activate` runtime behavior
//!   is 02.3, addressed cancel scoping is 02.9a, bounded release is 02.10b:
//!   all four probes must still answer -32601 (see
//!   `extension_runtime_seams_still_reject_clearly`).

mod support;

use maki_acp::extension;
use support::SharedSessions;

/// Both runtimes coexist under the same Maki PID: opening the second session
/// never closes or disturbs the first, and each pump addresses only its own
/// session. One child process by construction, so one PID by construction.
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

    // A is still usable after B opened: the registry never replaced it.
    shared.child.set_mode(&shared.a, "plan").await.unwrap();
    shared.child.set_mode(&shared.b, "build").await.unwrap();

    let a = shared.a.clone();
    let update_a = shared
        .child
        .expect_notification(|frame| {
            frame["method"] == "session/update" && frame["params"]["sessionId"] == a
        })
        .await
        .unwrap();
    assert_eq!(
        update_a["params"]["update"]["currentModeId"], "plan",
        "A's pump addresses A's own session: {update_a}"
    );

    let b = shared.b.clone();
    let update_b = shared
        .child
        .expect_notification(|frame| {
            frame["method"] == "session/update" && frame["params"]["sessionId"] == b
        })
        .await
        .unwrap();
    assert_eq!(
        update_b["params"]["update"]["currentModeId"], "build",
        "B's pump addresses B's own session: {update_b}"
    );

    shared.shutdown().await.unwrap();
}

/// Allocation, activation, addressed cancel and release stay specified but not
/// implemented: every probe answers -32601 like an unknown method, on a
/// multi-session server too.
#[tokio::test]
async fn extension_runtime_seams_still_reject_clearly() {
    let mut shared = SharedSessions::establish().await.unwrap();
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
