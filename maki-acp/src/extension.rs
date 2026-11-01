//! Fork multi-session ACP extension names and the turn-epoch settlement
//! helper. Specified in slice 02.1 (`EXTENSION.md` is normative); the
//! multi-session runtime that wires these in arrives across 02.2a–02.10b.
//! The only executable piece here is [`TurnEpoch`], the single settlement
//! contract every later slice shares: bump on terminal/cancel/loss/release,
//! check late answers and events against it.

/// `_meta` key under `agentCapabilities` advertising the extension. Clients
/// branch on its presence, never on an agent version string.
pub const CAPABILITY_KEY: &str = "maki.multiSession";
/// Informational capability version. Clients MUST NOT gate on it.
pub const CAPABILITY_VERSION: u32 = 1;
/// Reserve an id with no provider spin-up and no task spawn.
pub const ALLOCATE_METHOD: &str = "_maki/session/allocate";
/// Bind an id to a model, fixed mode/cwd/absolute output, and start its task.
pub const ACTIVATE_METHOD: &str = "_maki/session/activate";
/// Session-scoped, optionally turn-pinned cancel.
pub const CANCEL_METHOD: &str = "_maki/session/cancel";
/// Release one session; its peers stay usable.
pub const RELEASE_METHOD: &str = "_maki/session/release";
/// Server-to-client lifecycle notification (`allocated|active|released|lost`).
pub const LIFECYCLE_NOTIFICATION: &str = "_maki/session/lifecycle";
/// Capability key 04.1 model discovery will advertise. Named only.
pub const MODEL_DISCOVERY_CAPABILITY: &str = "maki.modelDiscovery";
/// Method-prefix 04.1 model discovery will own. Named only.
pub const MODEL_DISCOVERY_PREFIX: &str = "_maki/model/";
/// Capability key 05.2 owned transcript deletion will advertise. Named only.
pub const OWNED_DELETION_CAPABILITY: &str = "maki.ownedDeletion";
/// Method 05.2 owned transcript deletion will own. Named only.
pub const OWNED_DELETION_METHOD: &str = "_maki/session/delete";
/// `data.reason` answering a post-activation model/mode change. Setup is
/// fixed per session; the client starts another session for a new setup.
pub const FIXED_AT_ACTIVATION_REASON: &str = "fixed_at_activation";
/// `_meta` key carrying the turn epoch on turn-scoped server-to-client
/// frames (session updates, permission/elicitation asks, prompt responses),
/// so answers and events correlate by request and turn.
pub const TURN_EPOCH_META_KEY: &str = "maki.turnEpoch";
/// `_meta` key carrying the terminal reason verbatim on a prompt response:
/// the provider's wire code (`"refusal"`, `"content_filter"`, ...) when the
/// provider named one, else the `Done` reason name. Never remapped, so later
/// slices keep provider semantics without re-parsing display strings.
pub const RAW_REASON_META_KEY: &str = "maki.rawReason";
/// `_meta` key carrying runtime usability, not usefulness of the answer.
/// A refusal can leave a healthy session reusable; a cancelled turn stays
/// unavailable until cleanup is confirmed.
pub const USABLE_META_KEY: &str = "maki.usable";

/// Advertise the implemented slice of the extension. The contract names four
/// methods, and the capability lists what this binary actually answers:
/// 02.3 implements allocate and activate, 02.9a adds addressed cancel, and
/// 02.10b adds bounded active-session release. Clients MUST probe any method
/// they intend to call: an unlisted method answers `-32601`, which is the
/// "unsupported" signal, not a failure.
pub fn capability_meta() -> agent_client_protocol_schema::v1::Meta {
    agent_client_protocol_schema::v1::Meta::from_iter([(
        CAPABILITY_KEY.to_owned(),
        serde_json::json!({
            "version": CAPABILITY_VERSION,
            "methods": [ALLOCATE_METHOD, ACTIVATE_METHOD, CANCEL_METHOD, RELEASE_METHOD],
        }),
    )])
}

/// Per-session turn epoch: 0 at allocate/activate, bumped on every terminal
/// settlement, on an addressed cancel taking effect, and on handle loss. Late
/// answers and events validate against it through [`TurnEpoch::check`]; the
/// first terminal for an epoch settles the prompt and every later frame for
/// that epoch is dropped, which is the exactly-once rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TurnEpoch(u64);

/// A stale answer names an epoch that is no longer current. Rejected, never
/// applied, so a stale allow cannot green a later turn's tool call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaleEpoch {
    /// Epoch the session had when the stale frame was checked.
    pub expected: u64,
    /// Epoch the stale frame carried.
    pub got: u64,
}

impl std::fmt::Display for StaleEpoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stale turn epoch: session is at {}, frame named {}",
            self.expected, self.got
        )
    }
}

impl std::error::Error for StaleEpoch {}

/// Stamps a turn-scoped frame with the epoch it was issued under. The only
/// production constructor for the key, so every frame names it identically.
pub fn turn_epoch_meta(epoch: u64) -> agent_client_protocol_schema::v1::Meta {
    agent_client_protocol_schema::v1::Meta::from_iter([(
        TURN_EPOCH_META_KEY.to_owned(),
        serde_json::Value::from(epoch),
    )])
}

/// Stamps a prompt response with everything the fork adds to the terminal:
/// the settling epoch plus the verbatim raw reason and task usability. All
/// three keys are additive `_meta`, so stock clients ignore them while fork
/// clients branch on their presence.
pub fn terminal_meta(
    epoch: u64,
    raw_reason: &str,
    usable: bool,
) -> agent_client_protocol_schema::v1::Meta {
    agent_client_protocol_schema::v1::Meta::from_iter([
        (
            TURN_EPOCH_META_KEY.to_owned(),
            serde_json::Value::from(epoch),
        ),
        (
            RAW_REASON_META_KEY.to_owned(),
            serde_json::Value::from(raw_reason),
        ),
        (USABLE_META_KEY.to_owned(), serde_json::Value::from(usable)),
    ])
}

impl TurnEpoch {
    /// Epoch of a freshly allocated or activated session.
    pub fn initial() -> Self {
        Self::default()
    }

    /// Epoch late frames are checked against.
    pub fn current(self) -> u64 {
        self.0
    }

    /// Advance past the settling turn. Returns the new epoch the next turn
    /// runs under.
    pub fn bump(&mut self) -> u64 {
        self.0 += 1;
        self.0
    }

    /// Whether `epoch` is the live one.
    pub fn fresh(self, epoch: u64) -> bool {
        self.0 == epoch
    }

    /// Accept a frame from the current epoch, reject anything older or from
    /// the future: both are stale once interleaved sessions share one loop.
    pub fn check(self, epoch: u64) -> Result<(), StaleEpoch> {
        if self.fresh(epoch) {
            Ok(())
        } else {
            Err(StaleEpoch {
                expected: self.0,
                got: epoch,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::*;

    const BUMPED_ONCE: u64 = 1;
    const STALE_PREVIOUS: u64 = 0;
    const STALE_FUTURE: u64 = 7;

    #[test]
    fn a_new_session_starts_at_epoch_zero() {
        assert_eq!(TurnEpoch::initial().current(), 0);
    }

    #[test]
    fn bump_moves_to_the_next_turn() {
        let mut epoch = TurnEpoch::initial();
        assert_eq!(epoch.bump(), BUMPED_ONCE);
        assert_eq!(epoch.current(), BUMPED_ONCE);
    }

    #[test_case(STALE_PREVIOUS ; "previous_epoch_after_a_terminal")]
    #[test_case(STALE_FUTURE ; "epoch_from_another_session")]
    fn only_the_current_epoch_checks_out(stale: u64) {
        let mut epoch = TurnEpoch::initial();
        epoch.bump();
        assert!(epoch.check(BUMPED_ONCE).is_ok());
        assert_eq!(
            epoch.check(stale),
            Err(StaleEpoch {
                expected: BUMPED_ONCE,
                got: stale
            })
        );
    }

    #[test]
    fn a_stale_rejection_names_both_epochs() {
        let rejection = TurnEpoch::initial().check(STALE_FUTURE).unwrap_err();
        assert_eq!(rejection.expected, 0);
        assert_eq!(rejection.got, STALE_FUTURE);
        assert!(rejection.to_string().contains('7'));
    }

    #[test]
    fn turn_epoch_meta_names_the_epoch() {
        let meta = super::turn_epoch_meta(3);
        assert_eq!(
            meta.get(super::TURN_EPOCH_META_KEY),
            Some(&serde_json::Value::from(3))
        );
        assert_eq!(meta.len(), 1);
    }

    #[test]
    fn terminal_meta_carries_epoch_raw_reason_and_usability() {
        let meta = super::terminal_meta(2, "content_filter", false);
        assert_eq!(
            meta.get(super::TURN_EPOCH_META_KEY),
            Some(&serde_json::Value::from(2))
        );
        assert_eq!(
            meta.get(super::RAW_REASON_META_KEY),
            Some(&serde_json::Value::from("content_filter"))
        );
        assert_eq!(
            meta.get(super::USABLE_META_KEY),
            Some(&serde_json::Value::from(false))
        );
        assert_eq!(meta.len(), 3);
    }

    #[test]
    fn extension_names_share_one_private_prefix() {
        for method in [
            ALLOCATE_METHOD,
            ACTIVATE_METHOD,
            CANCEL_METHOD,
            RELEASE_METHOD,
        ] {
            assert!(method.starts_with("_maki/session/"), "{method}");
        }
        assert!(MODEL_DISCOVERY_PREFIX.starts_with("_maki/"));
        assert_eq!(OWNED_DELETION_METHOD, "_maki/session/delete");
    }

    /// 02.3 advertises exactly the methods it answers: allocate and activate.
    /// 02.9a adds addressed cancel, 02.10b adds bounded active-session
    /// release; advertising an unanswered method would lie, so each slice
    /// extends this list as it lands.
    #[test]
    fn capability_advertises_only_the_implemented_methods() {
        let meta = super::capability_meta();
        let entry = meta.get(super::CAPABILITY_KEY).expect("capability keyed");
        assert_eq!(entry["version"], super::CAPABILITY_VERSION);
        let methods: Vec<&str> = entry["methods"]
            .as_array()
            .expect("methods is an array")
            .iter()
            .filter_map(|m| m.as_str())
            .collect();
        assert_eq!(
            methods,
            [
                super::ALLOCATE_METHOD,
                super::ACTIVATE_METHOD,
                super::CANCEL_METHOD,
                super::RELEASE_METHOD
            ]
        );
    }
}
