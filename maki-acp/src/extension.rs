//! Fork multi-session ACP extension names and the turn-epoch settlement
//! helper. Specified in slice 02.1 (`EXTENSION.md` is normative); the
//! multi-session runtime that wires these in arrives in 02.2a/02.2b. The only
//! executable piece here is [`TurnEpoch`], the single settlement contract
//! every later slice shares: bump on terminal/cancel/loss, check late
//! answers and events against it.

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
}
