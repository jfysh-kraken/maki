//! Slice 02.7 acceptance: declared provider refusal/filter codes traverse
//! decoding, agent completion and ACP translation to `refusal`, never
//! collapsing into `end_turn`.
//!
//! No live inference, no keys, no network. The wire codes below are the
//! verbatim strings recorded provider SSE carries (`stop_reason`,
//! `finish_reason`, `finishReason`); decoder unit tests feed them as bytes,
//! and this fixture drives the same codes through the real
//! `StopReason::from_*` classifiers, `DoneReason::from`, and
//! `translate::map_done_reason`, plus the additive `_meta` carrying the raw
//! reason and task usability. Nothing here fabricates an ACP reply: every
//! asserted `stopReason` comes out of the production mapping.
//!
//! Duplicate-terminal settlement (exactly-once through the 02.2b turn-epoch
//! primitive) is proven at unit level in `server.rs`, where real pumps are
//! driven with synthetic agent events; ask retirement stays 02.8's.

use agent_client_protocol_schema::v1::StopReason as AcpStopReason;
use maki_acp::{extension, translate};
use maki_agent::DoneReason;
use maki_providers::StopReason;

/// One recorded wire code, end to end: decode, complete, translate.
fn end_to_end(classify: StopReason) -> (DoneReason, AcpStopReason) {
    let done = DoneReason::from(Some(classify));
    let acp = translate::map_done_reason(done);
    (done, acp)
}

#[test]
fn anthropic_refusal_reaches_acp_refusal() {
    // Recorded `message_delta` code for a refused Anthropic turn.
    let (done, acp) = end_to_end(StopReason::from_anthropic("refusal"));
    assert_eq!(done, DoneReason::Refusal);
    assert_eq!(acp, AcpStopReason::Refusal);
    assert!(!done.usable(), "a refusal banks no usable output");
}

#[test]
fn openai_content_filter_reaches_acp_refusal() {
    // Recorded terminal `finish_reason` of a filtered chat-completions turn.
    let (done, acp) = end_to_end(StopReason::from_openai("content_filter"));
    assert_eq!(done, DoneReason::Refusal);
    assert_eq!(acp, AcpStopReason::Refusal);
    assert!(!done.usable(), "filtered output is not usable");
}

#[test]
fn gemini_safety_block_reaches_acp_refusal() {
    // Recorded Gemini `finishReason` for a safety-blocked turn.
    let (done, acp) = end_to_end(StopReason::from_google("SAFETY"));
    assert_eq!(done, DoneReason::Refusal);
    assert_eq!(acp, AcpStopReason::Refusal);
}

#[test]
fn refusal_like_prose_with_a_normal_code_stays_a_normal_ending() {
    // The classifier never sees text: a normal code beside refusal-like prose
    // ("I can't help with that", "filtered by policy") stays normal.
    for (classify, done, acp) in [
        (
            StopReason::from_anthropic("end_turn"),
            DoneReason::EndTurn,
            AcpStopReason::EndTurn,
        ),
        (
            StopReason::from_openai("stop"),
            DoneReason::EndTurn,
            AcpStopReason::EndTurn,
        ),
        (
            StopReason::from_google("STOP"),
            DoneReason::EndTurn,
            AcpStopReason::EndTurn,
        ),
    ] {
        assert_eq!(DoneReason::from(Some(classify)), done);
        assert_eq!(translate::map_done_reason(done), acp);
        assert!(done.usable());
    }
}

#[test]
fn normal_endings_limits_cancellation_stay_distinguishable() {
    let cases = [
        (DoneReason::EndTurn, AcpStopReason::EndTurn, true),
        (DoneReason::MaxTokens, AcpStopReason::MaxTokens, true),
        (DoneReason::MaxTurns, AcpStopReason::MaxTurnRequests, true),
        (DoneReason::Cancelled, AcpStopReason::Cancelled, false),
        (DoneReason::Refusal, AcpStopReason::Refusal, false),
        (DoneReason::Dropped, AcpStopReason::Refusal, false),
        // Housekeeping has no ACP code of its own and stays an end turn.
        (DoneReason::Compact, AcpStopReason::EndTurn, true),
    ];
    for (done, expected, usable) in cases {
        assert_eq!(translate::map_done_reason(done), expected, "{done:?}");
        assert_eq!(done.usable(), usable, "{done:?}");
    }
}

#[test]
fn unknown_wire_codes_keep_the_compat_fallthrough() {
    // Codes no decoder names still collapse to a normal ending, as before.
    for classify in [
        StopReason::from_anthropic("stop_sequence"),
        StopReason::from_openai("function_call"),
        StopReason::from_google("OTHER"),
    ] {
        let (done, acp) = end_to_end(classify);
        assert_eq!(done, DoneReason::EndTurn);
        assert_eq!(acp, AcpStopReason::EndTurn);
    }
}

#[test]
fn terminal_meta_preserves_raw_reason_and_usability() {
    // What the prompt response carries beside the standard `stopReason`.
    let meta = extension::terminal_meta(0, "content_filter", false);
    assert_eq!(
        meta.get(extension::TURN_EPOCH_META_KEY),
        Some(&serde_json::Value::from(0))
    );
    assert_eq!(
        meta.get(extension::RAW_REASON_META_KEY),
        Some(&serde_json::Value::from("content_filter"))
    );
    assert_eq!(
        meta.get(extension::USABLE_META_KEY),
        Some(&serde_json::Value::from(false))
    );
}

#[test]
fn agent_side_endings_default_the_raw_reason_to_their_own_name() {
    // No wire code exists for endings the agent decided itself; the `Done`
    // name is the honest raw reason, matching the server fallback.
    for reason in [
        DoneReason::Dropped,
        DoneReason::Cancelled,
        DoneReason::MaxTurns,
        DoneReason::Compact,
    ] {
        let raw = reason.to_string();
        let meta = extension::terminal_meta(1, &raw, reason.usable());
        assert_eq!(
            meta.get(extension::RAW_REASON_META_KEY),
            Some(&serde_json::Value::from(raw.as_str()))
        );
    }
    assert_eq!(DoneReason::Refusal.to_string(), "refusal");
}
