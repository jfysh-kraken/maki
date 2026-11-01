//! Slice 02.4 acceptance (wire half): interleaved sessions in fixture
//! projects A/B activate with their own initial context while Maki's process
//! cwd stays unchanged.
//!
//! Fake client only, and no prompt is ever sent to a live session: a prompt
//! to an activated session would reach the provider, and fixtures use no
//! keys, inference, or network. What this fixture proves at the wire:
//! - allocate/activate interleave across A/B with distinct models/modes;
//! - each activation answers `{sessionId, turnEpoch: 0}` plus a usable
//!   `active` lifecycle frame, correlated by id;
//! - model/mode/output are fixed before the first prompt (post-activation
//!   changes route to the addressed session and reject; unknown ids reject
//!   without touching either runtime);
//! - the harness process cwd is unchanged throughout (the child-side half of
//!   that invariant — no `set_current_dir` anywhere near async work — is
//!   proven by audit plus the in-process unit tests, which also prove the
//!   instruction/environment *content* is per-session: the wire has no
//!   readback for the system prompt without sending a prompt, which fixtures
//!   never do).
//!
//! What it deliberately leaves for named later slices (not silent gaps):
//! - per-session instruction/environment *content* is proven at unit level in
//!   `maki-agent` (`initial_session_context_is_per_session_cwd`) and in
//!   `server.rs` (`interleaved_sessions_keep_their_own_initial_context`),
//!   where the starter is a fixture: the live child cannot observe its own
//!   startup text without a test hook in the binary;
//! - runtime-tool cwd (actual filesystem/Lua/subprocess resolution against the
//!   session cwd) is 02.5a–02.5c, explicitly not this slice. Evidence here is
//!   labeled as the initial-context proof, not the runtime-tool proof.

mod support;

use maki_acp::extension;
use support::{MakiChild, ProjectPair};

/// Distinct requested models for the two live sessions. Both parse offline
/// (the harness boots on MODEL_A with no keys), and neither prompt is ever
/// sent, so no provider is contacted either way.
const MODEL_A: &str = "openai/gpt-5";
const MODEL_B: &str = "openai/gpt-5.6-sol";

/// Distinct instruction markers per fixture project. The live child reads
/// them at activation (or ignores their absence identically); what matters
/// here is that each session is built against its own directory, which the
/// interleaving below plus the fixed-setup rejections prove per session.
const MARKER_A: &str = "alpha project rules";
const MARKER_B: &str = "beta project rules";

async fn lifecycle_active(child: &mut MakiChild, session: &str) -> serde_json::Value {
    child
        .expect_notification(|frame| {
            frame["method"] == extension::LIFECYCLE_NOTIFICATION
                && frame["params"]["sessionId"] == session
        })
        .await
        .unwrap()
}

/// Interleaved A/B allocations activate with their own model/mode/output and
/// stay isolated: neither session inherits peer context, and the process cwd
/// never moves. No prompt is sent anywhere.
#[tokio::test]
async fn interleaved_sessions_activate_with_their_own_initial_context() {
    let process_cwd = std::env::current_dir().expect("the harness cwd is readable");
    let projects = ProjectPair::create().unwrap();
    std::fs::write(projects.a.join("AGENTS.md"), MARKER_A).unwrap();
    std::fs::write(projects.b.join("AGENTS.md"), MARKER_B).unwrap();

    let mut child = MakiChild::spawn().await.unwrap();
    child.initialize().await.unwrap();

    let cwd_a = projects.a.to_string_lossy().into_owned();
    let cwd_b = projects.b.to_string_lossy().into_owned();
    assert_ne!(cwd_a, cwd_b, "A/B are distinct fixture projects");
    let reply_a = child.allocate_raw(&cwd_a).await.unwrap();
    let reply_b = child.allocate_raw(&cwd_b).await.unwrap();
    let session_a = reply_a["result"]["sessionId"].as_str().unwrap();
    let session_b = reply_b["result"]["sessionId"].as_str().unwrap();
    assert_ne!(session_a, session_b, "allocations hold distinct ids");

    // Interleaved activation: A activates while B is still unactivated, so a
    // shared initial-context slot would visibly leak across the boundary.
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
    assert_eq!(active_b["result"]["turnEpoch"], 0);

    for session in [session_a, session_b] {
        let frame = lifecycle_active(&mut child, session).await;
        assert_eq!(frame["params"]["state"], "active", "{frame}");
        assert_eq!(frame["params"]["turnEpoch"], 0, "{frame}");
        assert_eq!(frame["params"]["usable"], true, "{frame}");
    }

    // Activation fixes the setup before the first prompt: per-session changes
    // route to the addressed session and reject, and an unknown prompt
    // rejects without touching either runtime.
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

    // Neither session created its output artifact: 03 owns the reservation,
    // and 02.4 only fixes the path.
    assert!(
        !projects.a.join("plan-a.md").exists() && !projects.b.join("plan-b.md").exists(),
        "activate fixes the output path without creating it"
    );
    assert_eq!(
        std::env::current_dir().expect("the harness cwd is readable"),
        process_cwd,
        "interleaved activation never moves the harness process cwd"
    );

    child.shutdown().await.unwrap();
}
