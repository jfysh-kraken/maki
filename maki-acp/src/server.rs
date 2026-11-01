use std::collections::HashMap;
use std::io::Write;
use std::iter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_client_protocol_schema::v1::{
    AgentNotification, AgentRequest, AgentResponse, CancelNotification, ConfigOptionUpdate,
    ContentBlock, EmbeddedResourceResource, Error as AcpError, ImageContent, InitializeRequest,
    JsonRpcMessage, ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, McpServer,
    NewSessionRequest, Notification, PromptRequest, PromptResponse, Request, RequestId,
    RequestPermissionRequest, RequestPermissionResponse, Response, SessionId, SessionInfo,
    SessionNotification, SessionUpdate, SetSessionConfigOptionRequest, SetSessionModeRequest,
    StopReason, TextContent,
};
use color_eyre::eyre::Context;
use flume::{Sender, WeakSender};
use futures::future::BoxFuture;
use maki_agent::headless::{self, BackgroundMode, InteractiveHandle, InteractiveParams};
use maki_agent::mcp::config::{RawHttpFields, RawStdioFields, RawTransport};
use maki_agent::mcp::{self, McpHandle};
use maki_agent::permissions::{PermissionAnswer, PluginRuleStore, TaggedAnswer};
use maki_agent::prompt::ResolvedPrompts;
use maki_agent::session::{Resumed, StoredSession};
use maki_agent::tools::{LocalTool, LocalTools, QUESTION_TOOL_NAME, ToolAudience, local_tool};
use maki_agent::types::AgentEvent;
use maki_agent::{
    AgentConfig, AgentInput, AgentMode, CancelToken, CostTally, DoneReason, Envelope,
    ImageMediaType, ImageSource, InputSource, SessionEndReason, SessionEvents,
};
use maki_config::project::{self, TrustAnswer, TrustMode, policy_grant};
use maki_config::{MAX_SERVER_NAME_LEN, ModelPolicy, ProjectConfig, SessionDefaults, TrustConfig};
use maki_providers::model::Model;
use maki_providers::provider::{available_model_specs, fetch_all_models};
use maki_providers::{Timeouts, settle_session};
use maki_storage::StateDir;
use maki_storage::id::{MakiId, SessionRef};
use maki_storage::sessions::{SessionClaim, SessionError};
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::task::{JoinHandle, JoinSet, spawn_blocking};
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::extension::TurnEpoch;
use crate::{AcpParams, SessionEndHook, elicitation, extension, methods, permissions, translate};

const FIRST_OUTGOING_REQUEST_ID: i64 = 1000;
/// Turns already recorded were priced when they ran. `always_fast` is a live
/// config value, so reading it here would reprice history the user paid for at
/// standard rates. New turns still honour it.
const RESTORED_FAST: bool = false;
/// From JSON-RPC's range for application errors. A session another maki holds
/// is not an internal error (`-32603`), nothing broke here, and the message
/// already says which session and why.
const SESSION_BUSY_CODE: i32 = -32000;

/// Ids come from here and are never reused, so a late answer for a closed
/// session cannot match a request of the session that replaced it.
static NEXT_OUTGOING_REQUEST_ID: AtomicI64 = AtomicI64::new(FIRST_OUTGOING_REQUEST_ID);

/// How long a cancelled run may take to report its terminal before the
/// session reports the cleanup as uncertain instead of merely pending (02.9b).
/// The bound only changes the report: a held cleanup stays non-resumable
/// either way, and final acceptance stays gated on 02.11's controlled live
/// evidence. Generous on purpose: a slow tool that still observes its cancel
/// reports late but confirmed, while a tool that never reports at all flips
/// the rejection to uncertain instead of hanging the client on "pending".
const CLEANUP_GRACE: Duration = Duration::from_secs(30);

/// 02.10b bounded release: every per-session teardown stage is capped, so one
/// wedged hook or MCP server cannot stall the routing loop or a shutdown.
/// The session-end hook (the `SessionEnd` callback, invoked per released
/// session) gets its own short bound; the MCP handle keeps its internal
/// 2s shutdown bound with this outer cap as backstop; the overall cap keeps
/// the whole release — hook, task/pump/supervisor reap, MCP shutdown — under
/// the fixture deadlock bound even when every stage hangs at once. Per-stage
/// failures surface as `release_failed` (never as `released`); 02.11 owns
/// tuning these bounds against controlled live evidence.
const RELEASE_HOOK_TIMEOUT: Duration = Duration::from_secs(2);
const RELEASE_MCP_TIMEOUT: Duration = Duration::from_secs(3);
const RELEASE_TIMEOUT: Duration = Duration::from_secs(8);

/// What the client still owes us. Every agent waits on its own answer channel,
/// so a child's permission can be outstanding next to the main agent's. The
/// epoch lives here, under the same guard as the prompt it settles, so
/// terminal settlement is check-and-bump in one lock: the single settlement
/// contract (`EXTENSION.md`), with no second epoch mechanism anywhere.
///
/// Asks are keyed by the globally unique outgoing request id
/// (`NEXT_OUTGOING_REQUEST_ID`), never by session or turn slot, so replies
/// route to the ask they answer even when several are outstanding or another
/// session settled since. Each ask is bound to the session and turn that
/// issued it; unknown, repeated, late, stale-turn and cross-session answers
/// are dropped at routing, never applied.
struct Pending {
    prompt: Option<RequestId>,
    /// Epoch the outstanding prompt runs under. `None` while idle; a terminal
    /// with no turn outstanding settles nothing and is dropped.
    turn: Option<u64>,
    epoch: TurnEpoch,
    asks: HashMap<i64, Ask>,
    /// Whether a settling terminal retires every outstanding ask. Blocking
    /// sessions (`--background-timeout 0`, the harness all fixtures run)
    /// retire: no call can outlive the turn, so a terminal turn cannot retain
    /// unresolved asks and a stale response can never revive one. Detached
    /// sessions retain: a moved call or a still-running child can be parked on
    /// an ask across the turn boundary.
    blocking: bool,
    /// 02.9a: set when an addressed cancel settles a live turn. The session
    /// accepts no further prompt while set (see `handle_prompt`): settlement
    /// alone never reopens the session. 02.9b clears it through
    /// `confirm_cleanup` once the cancelled run's terminal proves its
    /// blocking tools stopped; until then replacement prompts reject with
    /// `cleanup_unconfirmed`.
    awaiting_cleanup: bool,
    /// 02.9b: set when the cancelled run failed instead of stopping (its
    /// terminal was an `Error`, not a `Done`), or observed through
    /// `is_uncertain` once the bounded wait lapses with no terminal at all.
    /// Stays non-resumable either way: the flag only changes the rejection
    /// report from "pending" to "uncertain", never into a retry license.
    cleanup_uncertain: bool,
    /// 02.9b: when the cancelled turn started waiting for its tools to stop.
    /// Bounds the wait (`CLEANUP_GRACE`): a terminal that never arrives is
    /// held cleanup, reported as uncertain, still non-resumable. `None`
    /// while no cleanup is awaited.
    awaiting_since: Option<Instant>,
    /// 02.10a: supervised loss observed post-activation (task/pump death,
    /// stream end, dead input channel, provider-init failure). Non-resumable,
    /// reusing the 02.9b non-resumable mechanics, not infinite work. The typed
    /// reason is always `task_lost`; `loss_raw` carries the verbatim cause.
    /// 02.10b's release reports this prior loss; a dropped handle here is
    /// explicitly not cleanup proof and no teardown runs on this path.
    lost: bool,
    loss_reason: Option<String>,
    loss_raw: Option<String>,
    /// 02.10b: Close dropped the routing entry. Set once under the control
    /// lock in the release prologue, atomically with the cancel settlement or
    /// loss snapshot, so a racing terminal, loss report, answer, or second
    /// settle can never revive the handle, change its outcome, or touch a
    /// peer. Late paths check this flag and settle/send nothing.
    released: bool,
}

impl Default for Pending {
    /// Fixtures mirror the harness's blocking mode: no detached tools, so a
    /// settling terminal retires every outstanding ask.
    fn default() -> Self {
        Self {
            prompt: None,
            turn: None,
            epoch: TurnEpoch::default(),
            asks: HashMap::new(),
            blocking: true,
            awaiting_cleanup: false,
            cleanup_uncertain: false,
            awaiting_since: None,
            lost: false,
            loss_reason: None,
            loss_raw: None,
            released: false,
        }
    }
}

/// How to read the answer and who gets it. A subagent's permission carries the
/// channel its own agent waits on, everything else answers the main agent.
/// Every ask is bound to the session and turn that issued it: stale answers
/// are dropped, never applied. Concurrent-ask routing on top of this binding
/// (several outstanding, task delivery, terminal retirement) is 02.8.
enum Ask {
    /// `request_id` is the agent-side id of the ask the tool is parked on, not
    /// the JSON-RPC id and not the client-facing tool call: the answer is
    /// tagged with it so a waiter on any other ask refuses it.
    Permission {
        request_id: String,
        answer_tx: Option<Sender<String>>,
        session: String,
        epoch: u64,
    },
    Elicitation {
        session: String,
        epoch: u64,
    },
}

impl Ask {
    fn session(&self) -> &str {
        match self {
            Ask::Permission { session, .. } | Ask::Elicitation { session, .. } => session,
        }
    }

    fn epoch(&self) -> u64 {
        match self {
            Ask::Permission { epoch, .. } | Ask::Elicitation { epoch, .. } => *epoch,
        }
    }
}

/// What one addressed cancel did, decided under the control lock. The prompt
/// (if any) was taken exactly once and answers with `cancelled`; the epoch is
/// always the post-cancel one, bumped even when idle so a racing terminal
/// lands stale.
struct CancelSettlement {
    prompt: Option<RequestId>,
    /// The epoch the settled turn ran under (pre-bump). `None` when idle.
    turn: Option<u64>,
    /// Whether a live turn was settled (and the run signalled + asks retired).
    cancelled: bool,
    epoch: u64,
}

impl Pending {
    /// Whether the session accepts another prompt. False after an addressed
    /// cancel settles a live turn, until 02.9b confirms tool cleanup, false
    /// once 02.10a marks supervised loss, and false once 02.10b releases the
    /// routing entry: a released handle never reopens.
    fn is_resumable(&self) -> bool {
        !self.awaiting_cleanup && !self.lost && !self.released
    }

    /// 02.10a: whether supervised loss was already recorded. A lost handle is
    /// unusable afterwards; the client must release it (02.10b) and any
    /// further prompt or addressed call answers `resource_not_found`.
    fn is_lost(&self) -> bool {
        self.lost
    }

    /// 02.10b: whether Close already dropped the routing entry. Late answers,
    /// terminals, loss reports, and second settles no-op against a released
    /// handle: they cannot revive it, change its outcome, or touch a peer.
    fn is_released(&self) -> bool {
        self.released
    }

    /// 02.10b release prologue, under the same control lock as `claim_ask`,
    /// `settle_cancel`, and `settle_loss`: marks the handle released
    /// atomically with the settle/snapshot the caller just took, so a racing
    /// pump terminal or loss report lands on the other side of the mark and
    /// is dropped. Called once; the routing entry is already gone.
    fn mark_released(&mut self) {
        self.released = true;
    }

    /// 02.9b: whether a `cleanup_unconfirmed` rejection must report the
    /// cleanup as uncertain rather than merely pending. True when the
    /// cancelled run failed instead of stopping, or when the bounded wait
    /// lapsed with no terminal at all (held cleanup). Both stay
    /// non-resumable; this only changes the report, never into a retry.
    fn is_uncertain(&self) -> bool {
        self.cleanup_uncertain || self.cleanup_overdue()
    }

    /// The bounded wait lapsed while cleanup was still unconfirmed: the run
    /// may still be stopping its tools, or it may never report at all. Either
    /// way the session cannot claim the tools died, so the rejection says so.
    fn cleanup_overdue(&self) -> bool {
        match self.awaiting_since {
            Some(since) => self.awaiting_cleanup && since.elapsed() >= CLEANUP_GRACE,
            None => false,
        }
    }

    /// 02.9b confirmation: the cancelled turn's tools are recorded as
    /// stopped. Called by the pump when the cancelled run's terminal arrives
    /// while no live turn is parked (see the `Done`/`Error` arms of
    /// `start_event_pump`): a `Done` means the run ended, and under
    /// `--background-timeout 0` no call outlives its run, so the cancelled
    /// turn's blocking tools stopped with it. An `Error` terminal is failure,
    /// not confirmation (see `note_run_ended`). 02.11 owns the controlled
    /// live proof that tools really died; this gate only records what the run
    /// reported — and a process exit or process-group reap alone never proves
    /// an escaped `setsid` tool died (see the escaped-child record in
    /// `tests/cleanup_probes.rs`).
    fn confirm_cleanup(&mut self) {
        self.awaiting_cleanup = false;
        self.cleanup_uncertain = false;
        self.awaiting_since = None;
    }

    /// 02.9b pump observation of the cancelled run's terminal, after
    /// `settle_prompt` found no live turn to settle. `failed == false`
    /// (`Done`, whatever its reason: the run ended, so in blocking mode its
    /// tools stopped with it) confirms cleanup and reopens the session for
    /// exactly one new turn. `failed == true` (`Error`: the run failed
    /// instead of stopping) records the uncertainty and stays non-resumable.
    /// A no-op unless cleanup is awaited, so duplicate or unrelated terminals
    /// can never reopen a session or revive a failed one. Nothing is answered
    /// here: the prompt already settled through `settle_cancel`.
    fn note_run_ended(&mut self, failed: bool) {
        if !self.awaiting_cleanup {
            return;
        }
        if failed {
            self.cleanup_uncertain = true;
        } else {
            self.confirm_cleanup();
        }
    }

    /// Single control-lock claim for an incoming answer: remove-then-validate
    /// under one guard, so a concurrent cancel settlement cannot slip between
    /// the check and the take. Removes the ask either way (exactly-once: a
    /// replay finds an unknown id), and returns it only when it is live for
    /// `owner` under the current epoch. Anything else — unknown,
    /// already-answered, cross-session, stale-turn, or released — is dropped
    /// by the caller without reaching any agent.
    fn claim_ask(&mut self, wire_id: i64, owner: &str) -> Option<Ask> {
        // 02.10b: the routing entry is gone and the asks are retired; a late
        // answer must not claim anything, revive the handle, or touch a peer.
        if self.is_released() {
            return None;
        }
        let ask = self.asks.remove(&wire_id)?;
        if ask.session() == owner && self.epoch.check(ask.epoch()).is_ok() {
            Some(ask)
        } else {
            None
        }
    }

    /// 02.9a addressed-cancel settlement through the 02.2b turn-epoch
    /// primitive: the first taker of a live turn wins (a racing agent
    /// terminal finds no prompt and drops), and the epoch always bumps once,
    /// so a terminal racing the cancel lands stale either way. A settled turn
    /// retires every outstanding ask in both modes — the run is stopped, so
    /// unlike a terminally-settled detached turn nothing can outlive it — and
    /// marks the session non-resumable until the pump observes the cancelled
    /// run's terminal and confirms cleanup through `note_run_ended` (02.9b).
    /// A `Done` terminal confirms; an `Error` terminal or a lapsed bounded
    /// wait records explicitly non-resumable uncertainty. Whether an escaped
    /// tool actually died is 02.11's controlled live evidence, not claimed
    /// here.
    fn settle_cancel(&mut self) -> CancelSettlement {
        // 02.10b: a settle after Close changes nothing — no bump, no take —
        // so a Stop racing a release can never revive the handle or move its
        // epoch. Close itself settles before marking released, under this same
        // guard, so it queues behind an in-flight cancel instead of bypassing.
        if self.is_released() {
            return CancelSettlement {
                prompt: None,
                turn: None,
                cancelled: false,
                epoch: self.epoch.current(),
            };
        }
        let turn = self.epoch.current();
        let live = match (&self.prompt, self.turn) {
            (Some(_), Some(parked)) => self.epoch.check(parked).is_ok(),
            _ => false,
        };
        let epoch = self.epoch.bump();
        self.turn = None;
        if live {
            self.asks.clear();
            self.awaiting_cleanup = true;
            self.awaiting_since = Some(Instant::now());
            CancelSettlement {
                prompt: self.prompt.take(),
                turn: Some(turn),
                cancelled: true,
                epoch,
            }
        } else {
            CancelSettlement {
                prompt: None,
                turn: None,
                cancelled: false,
                epoch,
            }
        }
    }

    /// 02.10a supervised-loss settlement through the 02.2b turn-epoch
    /// primitive: the first observer of a post-activation task/pump death,
    /// stream end, dead input channel, or provider-init failure wins; every
    /// later observer finds `lost` set and settles nothing. The epoch always
    /// bumps once on first loss (even idle), so a racing terminal lands
    /// stale. A live outstanding prompt is taken exactly once; asks retire in
    /// both modes (the handle is dead, nothing can outlive it); the session
    /// stays failed/non-resumable. No teardown runs here: a dropped handle is
    /// explicitly not cleanup proof, and 02.10b's release reports this prior
    /// loss plus its own cleanup result.
    fn settle_loss(&mut self, reason: &str, raw: &str) -> LossSettlement {
        // 02.10b: a loss report after Close is late by definition — no bump,
        // no take, no send — so a racing pump stream-end or supervisor report
        // can never change the released outcome or emit a post-release `lost`.
        if self.is_lost() || self.is_released() {
            return LossSettlement {
                first: false,
                prompt: None,
                epoch: self.epoch.current(),
            };
        }
        let live = match (&self.prompt, self.turn) {
            (Some(_), Some(parked)) => self.epoch.check(parked).is_ok(),
            _ => false,
        };
        let epoch = self.epoch.bump();
        self.turn = None;
        self.asks.clear();
        self.lost = true;
        self.loss_reason = Some(reason.to_owned());
        self.loss_raw = Some(raw.to_owned());
        if live {
            LossSettlement {
                first: true,
                prompt: self.prompt.take(),
                epoch,
            }
        } else {
            LossSettlement {
                first: true,
                prompt: None,
                epoch,
            }
        }
    }
}

/// What one supervised-loss observation did, decided under the control lock.
/// The first observer takes the outstanding prompt (if live) and the bumped
/// epoch; every later observer finds `first == false` and sends nothing.
struct LossSettlement {
    first: bool,
    prompt: Option<RequestId>,
    epoch: u64,
}
/// The per-session control lock. Answer (client response), Stop (addressed
/// cancel), and Close (02.10b release) all serialize through this one mutex,
/// reusing the 02.2b session/turn identity: an answer claims its ask, a cancel
/// settles the turn, and a Close settle-then-marks-released under the same
/// guard, so a queued answer or Close can never bypass an in-flight cancel
/// settlement and a racing terminal or loss report after the mark settles
/// nothing.
type PendingState = Arc<Mutex<Pending>>;

/// One independently owned session runtime. Each entry in the registry owns its
/// agent task, its event pump task, and its conversation context (run handle,
/// storage claim, pending asks, turn epoch). Nothing here is shared between
/// sessions except the outgoing wire sender, so no global serialization and no
/// per-session process fallback exists.
struct SessionRuntime {
    handle: InteractiveHandle,
    mcp: Option<McpHandle>,
    current_mode: AgentMode,
    current_model: String,
    /// Requested session cwd, fixed at activation (02.4). The run, its event
    /// pump, and its permission manager all serve this directory; the Maki
    /// process cwd is never consulted for it. Runtime tool resolution against
    /// this value is 02.5b/c's; here it is retained, not yet consulted.
    cwd: PathBuf,
    /// Absolute output path fixed at activation (`EXTENSION.md`): the plan
    /// artifact in plan mode, the transcript-adjacent run record in build
    /// mode. `None` for standard `session/new`/`session/load` runtimes, which
    /// predate the extension. 03's transactional output reservation consumes
    /// this; 02.3 only carries it through unchanged.
    output: Option<PathBuf>,
    pending: PendingState,
    /// Shared with the run, so reloading the same session reads under this
    /// claim instead of clashing with it.
    claim: SessionClaim,
    /// The runtime owns its pump task; closing the session aborts and reaps it
    /// on the bounded release path.
    pump: JoinHandle<()>,
    /// 02.10a pump-death supervisor, off the shared routing loop. It holds no
    /// pump handle: the pump holds a done-sender and this task awaits its
    /// disconnect, so a panicked or silently-exited pump still reports `lost`
    /// exactly once while idle, without polling the loop. Aborted with the
    /// pump on release/shutdown — after the handle is marked released under
    /// the control lock — so teardown never reports loss. No teardown runs
    /// here; bounded release (`begin_release`) owns it.
    supervisor: JoinHandle<()>,
}

/// An allocated-but-unactivated lease (02.3). The id is reserved and the
/// storage claim held, but no provider exists and no agent task runs: prompts
/// are rejected, and nothing here can reach a provider. Activation consumes
/// the entry (success transfers the claim to the runtime, failure drops it);
/// release and ACP disconnect drop it the same way.
struct Allocation {
    /// Session cwd reserved alongside the id. Activation must name the same
    /// cwd back: the lease covers one (id, cwd) pair, never a floating id a
    /// later call can repoint.
    cwd: PathBuf,
    claim: SessionClaim,
}

/// Everything one activation needs, owned, so the slow half can move to a
/// dedicated task and leave the routing loop responsive. `mcp` starts empty
/// and `project_config` starts as the untrusted base: the task resolves trust
/// and starts MCP before spawning the runtime.
struct ActivationInput {
    spawn: SpawnCtx,
    trust_mode: TrustMode,
    trust_policy: Arc<TrustConfig>,
}

/// A startup that died before the activation result settled. Carries the
/// claim back so the loop releases the lease through the same helper as every
/// other unwind path. The production starter is infallible today (model parse
/// and trust resolution cannot fail, MCP start degrades to `None`); this arm
/// exists for the instrumented fixture and for fallible startup later.
struct ActivationFailure {
    claim: SessionClaim,
    reason: String,
}

type ActivationResult = Result<SessionRuntime, ActivationFailure>;

/// Slow half of activation: trust resolution, provider/MCP startup and agent
/// spawn, off the shared control loop. Production answers with the real
/// runtime; fixtures record the requested model and answer with a stub (or a
/// held/failed outcome), so no test needs provider contact.
type ActivationStarter =
    Arc<dyn Fn(ActivationInput) -> BoxFuture<'static, ActivationResult> + Send + Sync>;

/// Everything `spawn_runtime` needs, owned. The standard paths build one from
/// `AcpParams` on the loop; activation builds one from the requested
/// model/mode/cwd/output and finishes it on a dedicated task.
struct SpawnCtx {
    out_tx: Sender<Value>,
    elicits_form: bool,
    model: Model,
    mode: AgentMode,
    output: Option<PathBuf>,
    cwd: PathBuf,
    resumed: Resumed,
    claim: SessionClaim,
    mcp: Option<McpHandle>,
    project_config: ProjectConfig,
    initial_cost: Option<f64>,
    config: AgentConfig,
    timeouts: Timeouts,
    prompt_slots: Arc<ResolvedPrompts>,
    yolo: bool,
    defaults: SessionDefaults,
    model_policy: Arc<ModelPolicy>,
    plugin_rules: Arc<PluginRuleStore>,
    storage: StateDir,
    background: bool,
}

struct Server {
    out_tx: Sender<Value>,
    model_specs: Vec<String>,
    client_elicits_form: bool,
    defaults: SessionDefaults,
    /// ID-addressed runtime ownership. No current-session slot remains: every
    /// handler resolves its session from the request's session id, and a miss
    /// answers `resource_not_found` without touching peer runtimes. Prompt and
    /// event routing proof on top of this table is 02.2b.
    sessions: HashMap<String, SessionRuntime>,
    /// Unactivated allocation leases, by session id. No prompt is accepted and
    /// no provider/task starts while an id sits here; activation consumes the
    /// entry, release and disconnect drop it (02.3).
    allocations: HashMap<String, Allocation>,
    /// Activations with slow startup in flight, by session id. The value is
    /// the task handle so shutdown can abort a startup the loop will never
    /// settle; presence is the exactly-once guard: the first completion takes
    /// the entry and settles the activation result, a second completion for
    /// the same id finds nothing and is dropped.
    activating: HashMap<String, JoinHandle<()>>,
    /// Slow half of activation (see [`ActivationStarter`]).
    starter: ActivationStarter,
    /// How a finished startup reports back to the loop, which settles the
    /// activation result exactly once via [`settle_activation`]. Weak, like
    /// model discovery: an activation task must never keep the loop alive
    /// once stdin closes, so EOF still disconnects `in_rx` and the loop
    /// exits promptly; the task's report then fails and its pieces drop.
    completion: WeakSender<Incoming>,
    on_session_end: Option<SessionEndHook>,
    /// 02.10b background release tasks (explicit release, load replacement,
    /// and the shutdown sweep). The routing loop never awaits teardown: it
    /// spawns here and moves on, so a held hook or MCP teardown cannot stall
    /// peer cancel/permission routing. Shutdown drains this set, so a release
    /// already in flight still reports its outcome instead of being dropped.
    release_tasks: JoinSet<()>,
    /// 02.10b terminal release outcomes by session id, shared with the
    /// background tasks that record them. A re-release re-answers the recorded
    /// outcome — success stays idempotent, a failed teardown stays honestly
    /// failed — instead of claiming a new release over a dropped claim.
    /// `InFlight` covers the window between the routing-entry drop and the
    /// background task's report.
    release_records: ReleaseRecords,
}

/// Terminal release outcome for one session id. Shared between the routing
/// loop (which files `InFlight` when it drops the entry and reads records
/// for re-releases) and the background task (which replaces it with `Done`).
#[derive(Debug, Clone)]
enum ReleaseRecord {
    InFlight {
        epoch: u64,
    },
    Done {
        epoch: u64,
        failure: Option<ReleaseFailure>,
    },
}

/// Which bounded teardown stage failed, and the verbatim cause. Reported
/// distinctly from success: the release response is an error and the
/// lifecycle frame is `release_failed`, never `released`.
#[derive(Debug, Clone)]
struct ReleaseFailure {
    stage: &'static str,
    raw: String,
}

type ReleaseRecords = Arc<Mutex<HashMap<String, ReleaseRecord>>>;

impl Server {
    /// Look up the runtime a request addresses. A miss rejects without
    /// disturbing peers; there is no current-session fallback (02.2a).
    /// Strict cross-session answer/event identity on top is 02.2b.
    fn session(&self, id: &SessionId) -> Result<&SessionRuntime, AcpError> {
        self.sessions
            .get(&*id.0)
            .ok_or_else(|| unknown_session(&id.0))
    }
}

/// Unknown session id. The contract names the id in `data`; a notification
/// carries no id to answer on, so its caller drops the frame instead.
fn unknown_session(id: &str) -> AcpError {
    AcpError::resource_not_found(None).data(serde_json::json!({"sessionId": id}))
}

impl Server {
    fn respond(&self, id: RequestId, result: Result<AgentResponse, AcpError>) {
        send(&self.out_tx, Response::new(id, result));
    }

    /// Answers an extension method, which has no schema response type: the
    /// result is plain JSON (`methods::*_result`), the error the standard
    /// shape. Used for allocate/activate/release only.
    fn respond_value(&self, id: RequestId, result: Result<Value, AcpError>) {
        send(&self.out_tx, Response::new(id, result));
    }
}

enum Incoming {
    Line(String),
    Models(Vec<String>),
    /// A dedicated activation task finished slow startup. The loop settles
    /// the activation result exactly once from here; the task itself never
    /// answers the client. Boxed: the runtime dwarfs the other variants.
    ActivationDone {
        key: String,
        id: RequestId,
        outcome: Box<ActivationResult>,
    },
}

pub async fn serve(params: AcpParams) -> color_eyre::Result<()> {
    let (out_tx, out_rx) = flume::unbounded::<Value>();

    let writer_task = tokio::spawn(async move {
        let stdout = std::io::stdout();
        while let Ok(msg) = out_rx.recv_async().await {
            let mut handle = stdout.lock();
            if serde_json::to_writer(&mut handle, &msg).is_ok() {
                let _ = handle.write_all(b"\n");
                let _ = handle.flush();
            }
        }
    });

    let (in_tx, in_rx) = flume::unbounded::<Incoming>();
    let mut server = Server {
        out_tx,
        model_specs: available_model_specs(&params.model_policy),
        client_elicits_form: false,
        defaults: params.defaults,
        sessions: HashMap::new(),
        allocations: HashMap::new(),
        activating: HashMap::new(),
        starter: production_starter(),
        completion: in_tx.downgrade(),
        on_session_end: params.on_session_end.clone(),
        release_tasks: JoinSet::new(),
        release_records: ReleaseRecords::default(),
    };

    // Weak, so a discovery still in flight cannot keep the loop alive once stdin closes.
    discover_models(Arc::clone(&params.model_policy), in_tx.downgrade());
    let reader_task = tokio::spawn(read_stdin(in_tx));

    while let Ok(Ok(incoming)) = params.stop.race(in_rx.recv_async()).await {
        match incoming {
            Incoming::Line(line) => handle_line(&mut server, &line, &params).await,
            Incoming::Models(specs) => refresh_models(&mut server, specs),
            // The only settlement site for activation: the first completion
            // takes the in-flight entry and answers, a second finds nothing.
            Incoming::ActivationDone { key, id, outcome } => {
                settle_activation(&mut server, key, id, *outcome);
            }
        }
    }

    close_all_sessions(&mut server, SessionEndReason::Shutdown).await;
    drop(server);
    writer_task.await.context("stdout writer")?;
    // A stop leaves stdin open, so its read would never end.
    if params.stop.is_cancelled() {
        reader_task.abort();
    } else {
        reader_task
            .await
            .context("stdin reader")?
            .context("read stdin")?;
    }

    Ok(())
}

/// Lives in its own task because `read_line` is not cancel safe: the main loop
/// waits on discovery too, and a dropped read would eat half a line.
async fn read_stdin(tx: Sender<Incoming>) -> std::io::Result<()> {
    let mut reader = BufReader::new(tokio::io::stdin());
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        if tx.send_async(Incoming::Line(line)).await.is_err() {
            return Ok(());
        }
    }
}

/// Static manifests miss providers that only list their models over the wire
/// (OpenRouter and friends), so the same discovery the TUI runs happens here,
/// in the background. Each batch leaves the moment it lands: the slowest source
/// is a cold catalog download, and a provider the client could already pick from
/// should not wait behind it.
fn discover_models(policy: Arc<ModelPolicy>, tx: WeakSender<Incoming>) {
    tokio::spawn(async move {
        fetch_all_models(
            &policy,
            |batch| {
                if let Some(tx) = tx.upgrade() {
                    let _ = tx.send(Incoming::Models(batch.models));
                }
            },
            None,
        )
        .await;
    });
}

/// Discovery lands in batches after the client built its selector from the
/// offline list, so every batch that adds something announces the fuller list.
fn refresh_models(srv: &mut Server, batch: Vec<String>) {
    let known = srv.model_specs.len();
    for spec in batch {
        if !srv.model_specs.contains(&spec) {
            srv.model_specs.push(spec);
        }
    }
    if srv.model_specs.len() == known {
        return;
    }
    // Merged into every live runtime, since each session builds its selector
    // from this list. Per-session sends keep pumps independent: one slow or
    // gone peer never blocks another's announcement.
    for runtime in srv.sessions.values() {
        let option = methods::model_config_option(&runtime.current_model, &srv.model_specs);
        let epoch = runtime.pending.lock().unwrap().epoch.current();
        session_update(
            &srv.out_tx,
            &SessionId::from(runtime.handle.session_id.to_string()),
            epoch,
            SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(vec![option])),
        );
    }
}

async fn handle_line(server: &mut Server, line: &str, params: &AcpParams) {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return;
    }

    let raw: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "invalid JSON on stdin");
            server.respond(RequestId::Null, Err(AcpError::parse_error()));
            return;
        }
    };

    let id = raw.get("id").map(request_id);

    if raw.get("result").is_some() || raw.get("error").is_some() {
        handle_incoming_response(server, &raw);
    } else if let Some(method) = raw.get("method").and_then(Value::as_str) {
        match id {
            Some(id) => handle_request(server, method, id, &raw, params).await,
            None => handle_notification(server, method, &raw),
        }
    } else if let Some(id) = id {
        server.respond(id, Err(AcpError::invalid_request()));
    }
}

fn request_id(v: &Value) -> RequestId {
    serde_json::from_value(v.clone()).unwrap_or(RequestId::Null)
}

async fn handle_request(
    srv: &mut Server,
    method: &str,
    id: RequestId,
    raw: &Value,
    params: &AcpParams,
) {
    // Fork extension methods with schema-less results (`EXTENSION.md`).
    // Allocate and release-of-allocation answer synchronously: neither does
    // slow work. Activate validates synchronously and defers the result: the
    // dedicated task settles it exactly once on completion, so slow provider
    // and MCP startup never blocks this loop. Addressed cancel (02.9a)
    // settles synchronously through the turn epoch and answers at once.
    // Active-session release (02.10b) settles synchronously through the same
    // control lock and defers only the answer: the background task reports the
    // bounded teardown outcome, so slow hooks/MCP never block this loop.
    if method == extension::ALLOCATE_METHOD {
        let result = handle_allocate(srv, raw, params);
        srv.respond_value(id, result);
        return;
    }
    if method == extension::ACTIVATE_METHOD {
        if let Some(result) = handle_activate(srv, raw, &id, params) {
            srv.respond_value(id, result);
        }
        return;
    }
    if method == extension::RELEASE_METHOD {
        // `None` means the background release task owns the answer now.
        if let Some(result) = handle_release(srv, raw, id.clone()) {
            srv.respond_value(id, result);
        }
        return;
    }
    if method == extension::CANCEL_METHOD {
        let result = handle_cancel(srv, raw);
        srv.respond_value(id, result);
        return;
    }
    let result = match method {
        "initialize" => {
            srv.client_elicits_form = parse_params::<InitializeRequest>(raw)
                .is_ok_and(|req| elicitation::supports_form(&req.client_capabilities));
            Ok(AgentResponse::InitializeResponse(
                methods::initialize_response(),
            ))
        }
        "session/new" => new_session(srv, raw, params).await,
        "session/load" => load_session(srv, raw, params).await,
        "session/list" => list_sessions(raw, &params.storage).await,
        "session/prompt" => match handle_prompt(srv, raw, &id) {
            Ok(()) => return,
            Err(e) => Err(e),
        },
        "session/set_mode" => handle_set_mode(srv, raw),
        "session/set_config_option" => handle_set_config(srv, raw),
        // Reserved 04.1/05.2 names keep answering -32601 through the
        // catch-all below.
        _ => Err(AcpError::method_not_found()),
    };
    srv.respond(id, result);
}

/// Production slow half of activation: resolve trust, start MCP, then spawn
/// the runtime, all on a dedicated task. Infallible today (trust resolution
/// cannot fail and MCP start degrades to `None`); the `Err` arm exists for
/// the instrumented fixture and for fallible startup later.
async fn production_start(mut input: ActivationInput) -> ActivationResult {
    let project_config = trusted_project_config(
        &input.spawn.cwd,
        &input.spawn.storage,
        input.trust_mode,
        &input.trust_policy,
    );
    let mcp = start_mcp(&input.spawn.cwd, &[], project_config.clone()).await;
    input.spawn.project_config = project_config;
    input.spawn.mcp = mcp;
    Ok(spawn_runtime(input.spawn).1)
}

fn production_starter() -> ActivationStarter {
    Arc::new(|input| Box::pin(production_start(input)))
}

#[derive(Debug, Deserialize)]
struct AllocateParams {
    cwd: PathBuf,
}

/// Reserves an id with no provider spin-up and no task spawn. The storage
/// claim is the lease: holding it keeps the id alive, dropping it releases
/// it, and `SessionClaim::is_held` observes it. Fast path only, so this
/// answers synchronously from the loop.
fn handle_allocate(srv: &mut Server, raw: &Value, params: &AcpParams) -> Result<Value, AcpError> {
    let req: AllocateParams = parse_params(raw)?;
    if !req.cwd.is_absolute() || !req.cwd.is_dir() {
        return Err(AcpError::invalid_params().data(serde_json::json!({
            "reason": "invalid_cwd",
            "cwd": req.cwd,
        })));
    }
    let claim = SessionClaim::fresh(&params.storage);
    let session_id = claim.id().to_string();
    srv.allocations.insert(
        session_id.clone(),
        Allocation {
            cwd: req.cwd,
            claim,
        },
    );
    Ok(methods::allocate_result(&session_id))
}

#[derive(Debug, Deserialize)]
struct ActivateParams {
    #[serde(rename = "sessionId")]
    session_id: String,
    model: String,
    mode: String,
    cwd: PathBuf,
    output: PathBuf,
}

fn already_active(session_id: &str) -> AcpError {
    AcpError::invalid_params().data(serde_json::json!({
        "reason": "already_active",
        "sessionId": session_id,
    }))
}

fn activation_in_progress(session_id: &str) -> AcpError {
    AcpError::invalid_params().data(serde_json::json!({
        "reason": "activation_in_progress",
        "sessionId": session_id,
    }))
}

fn not_activated(session_id: &str) -> AcpError {
    AcpError::invalid_params().data(serde_json::json!({
        "reason": "not_activated",
        "sessionId": session_id,
    }))
}

/// 02.9a gate, 02.9b report: a session whose turn was cancelled stays
/// non-resumable until the pump confirms its tool cleanup. `uncertain` is
/// true when the cancelled run failed instead of stopping, or when the
/// bounded cleanup wait lapsed with no terminal: the rejection then reports
/// the uncertainty explicitly instead of claiming the tools merely pend.
/// Either way no retry path reopens the session; only `confirm_cleanup`
/// (through the pump's terminal observation) does.
fn cleanup_unconfirmed(session_id: &str, uncertain: bool) -> AcpError {
    AcpError::invalid_params().data(serde_json::json!({
        "reason": "cleanup_unconfirmed",
        "sessionId": session_id,
        "uncertain": uncertain,
    }))
}

/// 02.10a supervised loss: typed reason plus verbatim cause plus usability,
/// reusing the 02.1 lost-handle fields, not redefining them. The typed reason
/// is always `task_lost`; the raw cause names what the supervisor observed
/// (`stream_ended`, `pump_dead`, `input_channel_dead`, or the provider error
/// text). `usable` is always false: a lost handle never accepts another
/// prompt. The outstanding prompt (if live) answers with this error exactly
/// once; the lifecycle `lost` frame carries the same raw/usable pair.
const LOSS_TYPED_REASON: &str = "task_lost";

fn loss_error(session_id: &str, raw: &str) -> AcpError {
    AcpError::internal_error().data(serde_json::json!({
        "reason": LOSS_TYPED_REASON,
        "rawReason": raw,
        "usable": false,
        "sessionId": session_id,
    }))
}

/// 02.10a loss settlement shared by the pump (stream end, dead-handle Error),
/// the loop (dead input channel on prompt), and the pump-death supervisor.
/// Non-blocking and peer-free: one control-lock settlement plus at most two
/// wire sends for the addressed session only, so supervision never blocks the
/// shared routing loop. Returns true on the first observation; every later
/// observer for the same session finds `lost` set and sends nothing (exactly
/// once, including the activation-handoff window: the activation result
/// already settled, and task death plus stream end collapse to one `lost`).
/// No teardown runs here: the claim stays held and the dead task/pump handles
/// stay filed for 02.10b's bounded release, which reports this prior loss
/// plus its own cleanup result. A dropped handle is explicitly not cleanup
/// proof.
fn report_loss(
    pending: &PendingState,
    out_tx: &Sender<Value>,
    session_id: &str,
    raw: &str,
) -> bool {
    let settlement = pending.lock().unwrap().settle_loss(LOSS_TYPED_REASON, raw);
    if !settlement.first {
        return false;
    }
    send(
        out_tx,
        Notification {
            method: Arc::from(extension::LIFECYCLE_NOTIFICATION),
            params: Some(methods::lifecycle_lost_params(
                session_id,
                settlement.epoch,
                raw,
            )),
        },
    );
    if let Some(id) = settlement.prompt {
        send(
            out_tx,
            Response::<AgentResponse>::new(id, Err(loss_error(session_id, raw))),
        );
    }
    true
}

/// Binds an allocated id to the requested model, fixed mode/cwd/absolute
/// output, and starts its task. Returns `None` when the activation left on a
/// dedicated task: the loop stays responsive and the task's completion
/// settles the result exactly once via [`settle_activation`]. Any addressed
/// failure consumes the lease (the client allocates again to retry); an
/// unparsable request never reaches the lease.
#[allow(clippy::too_many_lines)]
fn handle_activate(
    srv: &mut Server,
    raw: &Value,
    id: &RequestId,
    params: &AcpParams,
) -> Option<Result<Value, AcpError>> {
    let req: ActivateParams = match parse_params(raw) {
        Ok(req) => req,
        Err(e) => return Some(Err(e)),
    };
    if srv.sessions.contains_key(&req.session_id) {
        return Some(Err(already_active(&req.session_id)));
    }
    if srv.activating.contains_key(&req.session_id) {
        return Some(Err(activation_in_progress(&req.session_id)));
    }
    let allocation = match srv.allocations.remove(&req.session_id) {
        Some(allocation) => allocation,
        None => return Some(Err(unknown_session(&req.session_id))),
    };
    // The lease is claimed from here on: validating, starting, failing, or
    // succeeding, the allocation entry is gone and the claim moves with the
    // activation. Any failure below drops it through `release_lease`, success
    // hands it to the runtime. No path leaks the claim or files two runtimes.
    let key = req.session_id.clone();
    let claim = allocation.claim;
    let model = match Model::from_spec_with_policy(&req.model, &params.model_policy) {
        Ok(model) => model,
        Err(e) => {
            let error = AcpError::invalid_params().data(serde_json::json!({
                "reason": e.to_string(),
                "model": req.model,
            }));
            release_lease(claim);
            return Some(Err(error));
        }
    };
    let mode = match req.mode.as_str() {
        "build" => AgentMode::Build,
        "plan" => AgentMode::Plan(req.output.clone()),
        _ => {
            let error = AcpError::invalid_params().data(serde_json::json!({
                "reason": "invalid_mode",
                "mode": req.mode,
            }));
            release_lease(claim);
            return Some(Err(error));
        }
    };
    if req.cwd != allocation.cwd {
        let error = AcpError::invalid_params().data(serde_json::json!({
            "reason": "cwd_mismatch",
            "cwd": req.cwd,
        }));
        release_lease(claim);
        return Some(Err(error));
    }
    if !req.output.is_absolute() {
        let error = AcpError::invalid_params().data(serde_json::json!({
            "reason": "output_must_be_absolute",
            "output": req.output,
        }));
        release_lease(claim);
        return Some(Err(error));
    }
    // The constructor never creates a default-provider session: the runtime
    // below is built with the requested model directly, and the recorded
    // current model is that same spec. There is no switch to observe.
    //
    // Upgraded while the loop is alive, which it is: the loop runs this
    // handler. The task's clone dies with the task; the weak itself never
    // keeps `in_rx` connected, so EOF still ends the loop promptly.
    let Some(done_tx) = srv.completion.upgrade() else {
        release_lease(claim);
        return Some(Err(AcpError::internal_error().data(serde_json::json!({
            "reason": "server shutting down",
            "sessionId": key,
        }))));
    };
    let session_ref = SessionRef::from(claim.id());
    let input = ActivationInput {
        spawn: SpawnCtx {
            out_tx: srv.out_tx.clone(),
            elicits_form: srv.client_elicits_form,
            model,
            mode: mode.clone(),
            output: Some(req.output.clone()),
            cwd: req.cwd.clone(),
            resumed: Resumed::empty(session_ref),
            claim,
            mcp: None,
            // Untrusted base: the activation task resolves trust against the
            // store before spawning, exactly like the standard paths.
            project_config: ProjectConfig::discover(&req.cwd),
            initial_cost: None,
            config: params.config.clone(),
            timeouts: params.timeouts,
            prompt_slots: Arc::clone(&params.prompt_slots),
            yolo: params.yolo,
            defaults: params.defaults,
            model_policy: Arc::clone(&params.model_policy),
            plugin_rules: Arc::clone(&params.plugin_rules),
            storage: params.storage.clone(),
            background: params.background,
        },
        trust_mode: params.trust_mode,
        trust_policy: Arc::clone(&params.trust_policy),
    };
    let starter = Arc::clone(&srv.starter);
    let id = id.clone();
    // Filed before spawning so a racing completion settles exactly once, and
    // so shutdown can abort a startup the loop will never settle.
    srv.activating.insert(
        key.clone(),
        tokio::spawn(async move {
            let outcome = starter(input).await;
            let _ = done_tx.send(Incoming::ActivationDone {
                key,
                id,
                outcome: Box::new(outcome),
            });
        }),
    );
    None
}

/// Settles an activation completion from the loop: the first completion for
/// an in-flight id takes the entry and answers exactly once; any second
/// completion for the same id finds nothing and is dropped, never answered
/// twice. Success files the runtime and announces `active`; failure unwinds
/// the claim and answers an honest terminal while the peer stays usable.
fn settle_activation(srv: &mut Server, key: String, id: RequestId, outcome: ActivationResult) {
    if srv.activating.remove(&key).is_none() {
        warn!(session = %key, "dropping a duplicate activation completion");
        return;
    }
    match outcome {
        Ok(runtime) => {
            debug_assert_eq!(key, runtime.handle.session_id.to_string());
            // Every activated runtime carries the fixed output 03 reserves
            // against; standard runtimes (None) never pass through here.
            debug_assert!(runtime.output.is_some());
            // 02.4: the runtime retains its requested session cwd. Logged so
            // the activation record names the directory the session serves;
            // runtime-tool resolution against it is 02.5b/c's (this field is
            // also read by the 02.4 focused tests, not by dispatch).
            debug!(
                session = %key,
                cwd = %runtime.cwd.display(),
                model = %runtime.current_model,
                "session activated with its own initial context"
            );
            srv.sessions.insert(key.clone(), runtime);
            maki_otel::emit::session_started(maki_otel::emit::START_FRESH, Some(key.as_str()));
            srv.respond_value(id, Ok(methods::activate_result(&key)));
            send(
                &srv.out_tx,
                Notification {
                    method: Arc::from(extension::LIFECYCLE_NOTIFICATION),
                    params: Some(methods::lifecycle_active_params(&key)),
                },
            );
        }
        Err(failure) => {
            release_lease(failure.claim);
            srv.respond_value(
                id,
                Err(AcpError::internal_error().data(serde_json::json!({
                    "reason": failure.reason,
                    "sessionId": key,
                }))),
            );
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReleaseParams {
    #[serde(rename = "sessionId")]
    session_id: String,
    /// Free-form client note, echoed in the lifecycle frame. Absent on older
    /// clients; defaults to `client_done`.
    reason: Option<String>,
}

/// 02.10b failed-teardown error: which bounded stage gave up and why, plus
/// usability. Never `released`: a failed teardown leaves the runtime failed
/// and non-resumable, and re-releases re-answer this error honestly instead
/// of claiming a success that never happened.
fn release_failed_error(session_id: &str, stage: &str, raw: &str) -> AcpError {
    AcpError::internal_error().data(serde_json::json!({
        "reason": "release_failed",
        "stage": stage,
        "rawReason": raw,
        "usable": false,
        "sessionId": session_id,
    }))
}

/// A duplicate release racing the background task: the routing entry is
/// already gone but the outcome is not recorded yet. Rejects instead of
/// answering a second receipt for one release.
fn release_in_progress(session_id: &str, epoch: u64) -> AcpError {
    AcpError::invalid_params().data(serde_json::json!({
        "reason": "release_in_progress",
        "sessionId": session_id,
        "turnEpoch": epoch,
    }))
}

/// Releases one session. An unactivated allocation drops its lease at once;
/// an id with startup in flight is rejected; an unknown id is
/// `resource_not_found`. An active session settles synchronously through the
/// per-session control lock — Close queues behind an in-flight Answer/Stop,
/// cancels the addressed turn and retires its asks through the 02.9a path —
/// then tears down on a background task and answers there, so slow hooks or
/// MCP shutdown never block the routing loop (`None` = the task owns the
/// answer now). A supervised-dead handle reports its prior loss plus the
/// cleanup result and is never revived. Re-releases re-answer the recorded
/// outcome: success stays idempotent, failure stays honestly failed.
fn handle_release(srv: &mut Server, raw: &Value, id: RequestId) -> Option<Result<Value, AcpError>> {
    let req: ReleaseParams = match parse_params(raw) {
        Ok(req) => req,
        Err(e) => return Some(Err(e)),
    };
    if let Some(allocation) = srv.allocations.remove(&req.session_id) {
        release_lease(allocation.claim);
        return Some(Ok(methods::release_allocation_result(&req.session_id)));
    }
    if srv.activating.contains_key(&req.session_id) {
        return Some(Err(activation_in_progress(&req.session_id)));
    }
    if let Some(record) = srv.release_records.lock().unwrap().get(&req.session_id) {
        let record = record.clone();
        return Some(match record {
            ReleaseRecord::InFlight { epoch } => Err(release_in_progress(&req.session_id, epoch)),
            ReleaseRecord::Done { epoch, failure } => match failure {
                None => Ok(methods::release_result(&req.session_id, epoch)),
                Some(f) => Err(release_failed_error(&req.session_id, f.stage, &f.raw)),
            },
        });
    }
    let Some(runtime) = srv.sessions.remove(&req.session_id) else {
        return Some(Err(unknown_session(&req.session_id)));
    };
    let hook = srv.on_session_end.clone();
    let reason = req.reason.unwrap_or_else(|| "client_done".to_owned());
    // Explicit release closes the session the client is done with.
    begin_release(
        srv,
        runtime,
        req.session_id,
        reason,
        Some(id),
        hook,
        SessionEndReason::Delete,
    );
    None
}

/// What the release prologue settled, decided under the control lock.
struct ReleasePrologue {
    /// Post-release epoch: the ack and lifecycle report it, and a racing
    /// terminal lands stale against it.
    epoch: u64,
    /// Owed prompt and its turn, taken exactly once when a live turn settled.
    prompt: Option<RequestId>,
    turn: Option<u64>,
    cancelled: bool,
    /// Recorded supervised loss, when the handle was already dead. The
    /// release reports it alongside its own cleanup result and never revives
    /// the handle.
    prior_loss: Option<(String, String)>,
}

/// Starts an individual bounded release. The caller already dropped the
/// routing entry, so new input rejects as `resource_not_found` from here on.
/// The prologue below is prompt (control-lock settle/snapshot, prompt answer,
/// run signal, handle aborts — microseconds, no awaits); the spawned task
/// owns the slow teardown and the answer (when `id` is set), then drops the
/// claim as its runtime falls out of scope.
fn begin_release(
    srv: &mut Server,
    runtime: SessionRuntime,
    key: String,
    reason: String,
    id: Option<RequestId>,
    hook: Option<SessionEndHook>,
    end_reason: SessionEndReason,
) {
    let prologue = {
        let mut guard = runtime.pending.lock().unwrap();
        let settled = if guard.is_lost() {
            // Supervised-dead: no cancel settle (that would bump again and
            // mark cleanup-awaited on a dead handle) — snapshot the recorded
            // loss for the report instead.
            ReleasePrologue {
                epoch: guard.epoch.current(),
                prompt: None,
                turn: None,
                cancelled: false,
                prior_loss: Some((
                    guard.loss_reason.clone().unwrap_or_default(),
                    guard.loss_raw.clone().unwrap_or_default(),
                )),
            }
        } else {
            let settlement = guard.settle_cancel();
            ReleasePrologue {
                epoch: settlement.epoch,
                prompt: settlement.prompt,
                turn: settlement.turn,
                cancelled: settlement.cancelled,
                prior_loss: None,
            }
        };
        guard.mark_released();
        settled
    };
    // The 02.9a-cancel shape: a settled prompt answers `cancelled` with its
    // turn, raw reason, and usability, so the client never hangs on it.
    if let (Some(prompt_id), Some(turn)) = (prologue.prompt.clone(), prologue.turn) {
        let reason = DoneReason::Cancelled;
        let raw = reason.to_string();
        let resp = PromptResponse::new(StopReason::Cancelled).meta(extension::terminal_meta(
            turn,
            &raw,
            reason.usable(),
        ));
        send(
            &srv.out_tx,
            Response::<AgentResponse, AcpError>::new(
                prompt_id,
                Ok(AgentResponse::PromptResponse(resp)),
            ),
        );
    }
    // Only the addressed run is signalled — never the shared Maki process,
    // which outlives every per-session release.
    if prologue.cancelled {
        let _ = runtime.handle.cancel_tx.try_send(());
    }
    // Instant: stopping the work now keeps the read-after-close consistent
    // for load replacement and stops spend before the hook runs. The spawned
    // task reaps the handles.
    runtime.handle.task.abort();
    runtime.supervisor.abort();
    runtime.pump.abort();
    srv.release_records.lock().unwrap().insert(
        key.clone(),
        ReleaseRecord::InFlight {
            epoch: prologue.epoch,
        },
    );
    let out_tx = srv.out_tx.clone();
    let records = Arc::clone(&srv.release_records);
    let job = ReleaseJob {
        runtime,
        key,
        prologue,
        reason,
        id,
        hook,
        end_reason,
    };
    srv.release_tasks.spawn(async move {
        run_release(out_tx, records, job).await;
    });
}

/// Everything the background release task owns: the removed runtime plus the
/// prologue outcome and reporting addresses. Bundled so the loop hands the
/// slow half over in one move and never waits on it.
struct ReleaseJob {
    runtime: SessionRuntime,
    key: String,
    prologue: ReleasePrologue,
    reason: String,
    id: Option<RequestId>,
    hook: Option<SessionEndHook>,
    end_reason: SessionEndReason,
}

/// Slow half of an individual release, off the shared routing loop. Best
/// effort through every stage — a failed hook must not skip MCP shutdown —
/// each bounded, then one report: the deferred answer plus a `released` or
/// `release_failed` lifecycle frame, the terminal outcome recorded for
/// re-releases, and the claim dropped as the runtime falls out of scope.
///
/// Teardown targets are the per-session handles in the 02.5a inventory: the
/// session task (its Lua VM, TaskCells, and `JobOwner::Session` jobs die with
/// the abort), its event pump and loss supervisor, its MCP handle, and one
/// session-end hook invocation. Process-global hooks/MCP that cannot be
/// scoped — plugin-global `SlotHook` slot chains, autocmd stores, provider
/// hooks — are not touched here; they stand as recorded blockers, not
/// silently skipped work.
async fn run_release(out_tx: Sender<Value>, records: ReleaseRecords, job: ReleaseJob) {
    let ReleaseJob {
        runtime,
        key,
        prologue,
        reason,
        id,
        hook,
        end_reason,
    } = job;
    let stages = async {
        let mut failure: Option<ReleaseFailure> = None;
        // The end hook first, like the legacy close path: the bill it saves
        // is complete. Bounded with panics contained: a wedged or panicking
        // hook reports `release_failed` instead of stalling peers.
        if let Some(cb) = hook {
            let mut hook_task = tokio::spawn(cb(runtime.handle.session_id.id(), end_reason));
            match timeout(RELEASE_HOOK_TIMEOUT, &mut hook_task).await {
                Ok(Ok(())) => {}
                // Cancelled: only the abort below cancels it, after recording.
                Ok(Err(join)) if !join.is_cancelled() => {
                    failure = failure.or(Some(ReleaseFailure {
                        stage: "hook",
                        raw: "hook_panicked".to_owned(),
                    }));
                }
                Ok(Err(_)) => {}
                Err(_) => {
                    hook_task.abort();
                    failure = failure.or(Some(ReleaseFailure {
                        stage: "hook",
                        raw: "hook_timeout".to_owned(),
                    }));
                }
            }
        }
        // Aborted in the prologue, so these reaps are prompt: a wedged task
        // cannot stall release, and the process is never killed to close one
        // session. Join errors are discarded — a dead handle's panic is its
        // recorded prior loss, not a new cleanup failure.
        let _ = runtime.handle.task.await;
        let _ = runtime.supervisor.await;
        let _ = runtime.pump.await;
        // The per-session MCP handle from the 02.5a inventory. Internally
        // bounded (2s) with this outer cap as backstop; an unacked shutdown
        // reports `release_failed` instead of hanging the release — or
        // claiming a clean shutdown that never confirmed.
        if let Some(mcp) = &runtime.mcp
            && !matches!(timeout(RELEASE_MCP_TIMEOUT, mcp.shutdown()).await, Ok(true))
        {
            warn!(session = %key, "MCP shutdown unconfirmed");
            failure = failure.or(Some(ReleaseFailure {
                stage: "mcp",
                raw: "mcp_shutdown_timeout".to_owned(),
            }));
        }
        failure
    };
    // Backstop over the whole teardown: even a join that never resolves
    // reports instead of leaking the release.
    let failure: Option<ReleaseFailure> = match timeout(RELEASE_TIMEOUT, stages).await {
        Ok(failure) => failure,
        Err(_) => Some(ReleaseFailure {
            stage: "release",
            raw: "release_timeout".to_owned(),
        }),
    };
    let prior = prologue
        .prior_loss
        .as_ref()
        .map(|(reason, raw)| (reason.as_str(), raw.as_str()));
    match &failure {
        None => {
            info!(session = %key, epoch = prologue.epoch, "session released");
            if let Some(id) = id {
                send(
                    &out_tx,
                    Response::<Value, AcpError>::new(
                        id,
                        Ok(methods::release_result(&key, prologue.epoch)),
                    ),
                );
            }
            send(
                &out_tx,
                Notification {
                    method: Arc::from(extension::LIFECYCLE_NOTIFICATION),
                    params: Some(methods::lifecycle_released_params(
                        &key,
                        prologue.epoch,
                        &reason,
                        prior,
                    )),
                },
            );
        }
        Some(f) => {
            warn!(session = %key, stage = f.stage, raw = %f.raw, "session release failed");
            if let Some(id) = id {
                send(
                    &out_tx,
                    Response::<Value, AcpError>::new(
                        id,
                        Err(release_failed_error(&key, f.stage, &f.raw)),
                    ),
                );
            }
            send(
                &out_tx,
                Notification {
                    method: Arc::from(extension::LIFECYCLE_NOTIFICATION),
                    params: Some(methods::lifecycle_release_failed_params(
                        &key,
                        prologue.epoch,
                        &reason,
                        f.stage,
                        &f.raw,
                        prior,
                    )),
                },
            );
        }
    }
    records.lock().unwrap().insert(
        key,
        ReleaseRecord::Done {
            epoch: prologue.epoch,
            failure,
        },
    );
    // `runtime` — and the storage claim it holds — drops here.
}

#[derive(Debug, Deserialize)]
struct CancelParams {
    #[serde(rename = "sessionId")]
    session_id: String,
    /// Optional turn pin: when present and not equal to the session's current
    /// epoch, the cancel is stale and changes nothing.
    #[serde(rename = "turnEpoch")]
    turn_epoch: Option<u64>,
}

/// 02.9a addressed cancel: session-scoped, optionally turn-pinned. Settles
/// the outstanding prompt exactly once through the 02.2b turn-epoch
/// primitive (`settle_cancel`: take the prompt, bump, retire every ask, mark
/// non-resumable) and answers the settled prompt with `cancelled`, preserving
/// reason/raw stop reason/usability. A racing agent terminal finds no prompt
/// and drops; late answers find retired asks and drop without reaching any
/// agent. The run is signalled through its own `cancel_tx` only — never the
/// shared Maki process, which outlives every per-session stop. An idle
/// session still bumps (so a racing terminal lands stale) but stays usable:
/// nothing ran, so there is no cleanup to confirm. Settlement alone never
/// reopens a cancelled session: replacement prompts reject with
/// `cleanup_unconfirmed` until the pump observes the cancelled run's
/// terminal and confirms its tool cleanup (02.9b).
fn handle_cancel(srv: &Server, raw: &Value) -> Result<Value, AcpError> {
    let req: CancelParams = parse_params(raw)?;
    // No runtime exists behind a lease: the cancel names no turn to settle
    // and no agent to signal, so it rejects like every other addressed op.
    if srv.allocations.contains_key(&req.session_id) {
        return Err(not_activated(&req.session_id));
    }
    if srv.activating.contains_key(&req.session_id) {
        return Err(activation_in_progress(&req.session_id));
    }
    let session = srv.session(&SessionId::from(req.session_id.clone()))?;
    // 02.10a: a supervised-dead handle answers `resource_not_found`, like an
    // unknown id. The `lost` lifecycle frame already carried reason/raw/usable.
    if session.pending.lock().unwrap().is_lost() {
        return Err(unknown_session(&req.session_id));
    }
    // The pin is checked before anything moves: a stale cancel changes
    // nothing — no signal, no bump, no settlement.
    let current = session.pending.lock().unwrap().epoch.current();
    if let Some(got) = req.turn_epoch
        && got != current
    {
        return Err(AcpError::invalid_params().data(serde_json::json!({
            "expectedEpoch": current,
            "gotEpoch": got,
        })));
    }
    let settlement = session.pending.lock().unwrap().settle_cancel();
    // Only a settled turn names a run to stop. An idle cancel leaves the
    // channel alone: a queued signal would be consumed by the next turn's
    // stop task and cancel work the client never asked to stop.
    if settlement.cancelled {
        let _ = session.handle.cancel_tx.try_send(());
    }
    if let (Some(id), Some(turn)) = (settlement.prompt, settlement.turn) {
        // `DoneReason::Cancelled` carries the preserved triple: the standard
        // `cancelled` stop reason, the verbatim raw name, and no usability —
        // a cancelled-awaiting-cleanup session banks nothing to build on.
        let reason = DoneReason::Cancelled;
        let raw = reason.to_string();
        let resp = PromptResponse::new(StopReason::Cancelled).meta(extension::terminal_meta(
            turn,
            &raw,
            reason.usable(),
        ));
        srv.respond(id, Ok(AgentResponse::PromptResponse(resp)));
    }
    Ok(serde_json::json!({
        "sessionId": req.session_id,
        "cancelled": settlement.cancelled,
        "turnEpoch": settlement.epoch,
    }))
}

/// Drops an allocation's storage claim, releasing the lease. The single
/// ownership-transfer site for failed activation, explicit release of an
/// unactivated allocation, and the disconnect sweep below, so no unwind path
/// leaks a claim or files a second runtime. 02.10b's EOF sweep reuses this
/// helper.
fn release_lease(claim: SessionClaim) {
    drop(claim);
}

/// Releases every unactivated lease. Runs wherever sessions close: end of
/// input, stop, and `session/load` replacement, so a client that disappears
/// without explicit failure leaves no lease behind. In-flight startups own
/// their claims in their tasks: the completion send fails once the loop is
/// gone and the pieces (claim included) drop with the task.
fn close_all_allocations(srv: &mut Server) {
    for (_, allocation) in srv.allocations.drain() {
        release_lease(allocation.claim);
    }
}

async fn new_session(
    srv: &mut Server,
    raw: &Value,
    params: &AcpParams,
) -> Result<AgentResponse, AcpError> {
    let req: NewSessionRequest = parse_params(raw)?;
    // Coexists with every live runtime: peers are never closed or disturbed.
    // The standard ack stays synchronous for standard-client compatibility;
    // the ack-from-loop plus background-setup rule applies to extension
    // allocate/activate, whose ordering lands in 02.3.
    let project_config = trusted_project_config(
        &req.cwd,
        &params.storage,
        params.trust_mode,
        &params.trust_policy,
    );
    let mcp = start_mcp(&req.cwd, &req.mcp_servers, project_config.clone()).await;
    let claim = SessionClaim::fresh(&params.storage);
    let session_ref = start_session(
        srv,
        params,
        req.cwd,
        Resumed::empty(SessionRef::from(claim.id())),
        claim,
        mcp,
        project_config,
        None,
    );
    maki_otel::emit::session_started(maki_otel::emit::START_FRESH, Some(session_ref.as_str()));
    let spec = params.model.spec();
    let resp = methods::new_session_response(session_ref.as_str())
        .config_options(vec![methods::model_config_option(&spec, &srv.model_specs)]);
    Ok(AgentResponse::NewSessionResponse(resp))
}

async fn load_session(
    srv: &mut Server,
    raw: &Value,
    params: &AcpParams,
) -> Result<AgentResponse, AcpError> {
    let req: LoadSessionRequest = parse_params(raw)?;
    let session_ref: SessionRef = req
        .session_id
        .0
        .parse()
        .map_err(|_| AcpError::resource_not_found(Some(req.session_id.0.to_string())))?;
    // Only the addressed entry is ever replaced. Reloading a session this
    // server runs keeps its claim, so no other process slips in, and closes
    // its run first so the read includes its last turn. Peers are untouched,
    // and a load that fails leaves the whole table alone.
    let held = srv
        .sessions
        .get(session_ref.as_str())
        .map(|runtime| runtime.claim.clone())
        .filter(|claim| claim.id() == session_ref.id());
    let (restored, claim) = match held {
        Some(claim) => {
            close_one_session(srv, session_ref.as_str(), SessionEndReason::Replaced).await;
            (reload_history(&params.storage, &claim)?, claim)
        }
        None => {
            let loaded = claim_history(&params.storage, session_ref.id())?;
            close_one_session(srv, session_ref.as_str(), SessionEndReason::Replaced).await;
            loaded
        }
    };
    let project_config = trusted_project_config(
        &req.cwd,
        &params.storage,
        params.trust_mode,
        &params.trust_policy,
    );
    let mcp = start_mcp(&req.cwd, &req.mcp_servers, project_config.clone()).await;
    let sid = SessionId::from(session_ref.to_string());
    let home = maki_storage::paths::home();
    let replay_cwd = restored.cwd.as_deref().unwrap_or(&req.cwd);
    // The runtime below starts fresh, so its epoch is the initial one; no
    // prompt has run yet for these replayed updates to belong to.
    let replay_epoch = TurnEpoch::initial().current();
    for update in
        translate::replay_history(restored.session.messages(), replay_cwd, home.as_deref())
    {
        session_update(&srv.out_tx, &sid, replay_epoch, update);
    }
    // Priced against the model the session recorded, not the one selected now
    // (which may cost 10x more or less). Later turns add their own exact cost.
    let recorded_model =
        Model::from_spec(&restored.session.model).unwrap_or_else(|_| params.model.clone());
    // Settling writes today's estimate into entries that never recorded a cost.
    // The run saves this session back, so we settle a copy and keep that
    // estimate off disk.
    let mut by_model = restored.session.usage_by_model().clone();
    let restored_cost = settle_session(
        &restored.session.token_usage,
        &mut by_model,
        &recorded_model,
        RESTORED_FAST,
    );
    let started = start_session(
        srv,
        params,
        req.cwd,
        Resumed::stored(session_ref, restored.session),
        claim,
        mcp,
        project_config,
        restored_cost,
    );
    maki_otel::emit::session_started(maki_otel::emit::START_RESUME, Some(started.as_str()));
    let spec = params.model.spec();
    let resp = methods::load_session_response()
        .config_options(vec![methods::model_config_option(&spec, &srv.model_specs)]);
    Ok(AgentResponse::LoadSessionResponse(resp))
}

/// `session/list`: stored sessions, newest first. Reads the store, not
/// the live session, so it works before any session is open. The
/// optional cwd filters by exact match against the stored cwd, like the
/// CLI and TUI pickers: a path spelled differently (symlink, trailing
/// slash) matches nothing. Without a cwd all stored sessions are
/// listed, like `maki session list --global`. `cursor` is ignored; the
/// response is a single page.
async fn list_sessions(raw: &Value, storage: &StateDir) -> Result<AgentResponse, AcpError> {
    let req: ListSessionsRequest = parse_params(raw)?;
    // The scan stats every session file and tail-reads those with a cold
    // cache, so it runs on a blocking thread, like the TUI picker.
    let storage = storage.clone();
    let internal = |e: &dyn std::fmt::Display| AcpError::internal_error().data(json_str(&e));
    let summaries = spawn_blocking(move || match req.cwd {
        Some(cwd) => StoredSession::list(&cwd.to_string_lossy(), &storage),
        None => StoredSession::list_all(&storage),
    })
    .await
    .map_err(|e| internal(&e))?
    .map_err(|e| internal(&e))?;
    let sessions = summaries
        .into_iter()
        .map(|s| {
            let info = SessionInfo::new(s.id.to_string(), s.cwd).title(s.title);
            match iso_from_epoch(s.updated_at) {
                Some(ts) => info.updated_at(ts),
                None => info,
            }
        })
        .collect();
    Ok(AgentResponse::ListSessionsResponse(
        ListSessionsResponse::new(sessions),
    ))
}

/// Maki stores epoch seconds; ACP requires an ISO 8601 string. The
/// store writes 0 when a transcript's tail meta could not be read,
/// meaning "unknown", which must not be rendered as 1970-01-01.
fn iso_from_epoch(secs: u64) -> Option<String> {
    if secs == 0 {
        return None;
    }
    let secs = i64::try_from(secs).ok()?;
    jiff::Timestamp::from_second(secs)
        .ok()
        .map(|t| t.to_string())
}

/// Spawns a session and files its runtime in the registry under its id.
/// Spawning alone is not a useful state: the event stream has exactly one
/// reader, so it must be handed to that session's own pump here rather than
/// travel any further. Peer runtimes are not touched.
#[allow(clippy::too_many_arguments)]
fn start_session(
    srv: &mut Server,
    params: &AcpParams,
    cwd: PathBuf,
    resumed: Resumed,
    claim: SessionClaim,
    mcp: Option<McpHandle>,
    project_config: ProjectConfig,
    initial_cost: Option<f64>,
) -> SessionRef {
    let ctx = SpawnCtx {
        out_tx: srv.out_tx.clone(),
        elicits_form: srv.client_elicits_form,
        model: params.model.clone(),
        mode: AgentMode::Build,
        output: None,
        cwd,
        resumed,
        claim,
        mcp,
        project_config,
        initial_cost,
        config: params.config.clone(),
        timeouts: params.timeouts,
        prompt_slots: Arc::clone(&params.prompt_slots),
        yolo: params.yolo,
        defaults: params.defaults,
        model_policy: Arc::clone(&params.model_policy),
        plugin_rules: Arc::clone(&params.plugin_rules),
        storage: params.storage.clone(),
        background: params.background,
    };
    let (session_ref, runtime) = spawn_runtime(ctx);
    srv.sessions
        .insert(session_ref.as_str().to_owned(), runtime);
    session_ref
}

/// Builds a session runtime: agent task plus its own event pump. Used by the
/// standard paths on the loop and by activation on its dedicated task; both
/// file the finished runtime the same way, so activation constructs the
/// requested model directly instead of starting a default session and
/// switching it.
fn spawn_runtime(ctx: SpawnCtx) -> (SessionRef, SessionRuntime) {
    let pending = PendingState::default();
    // Without form elicitation the question tool would spin forever waiting
    // for a TUI that does not exist, so it is dropped and the model asks in
    // plain text instead.
    let (excluded_tools, local_tools) = if ctx.elicits_form {
        let tool = question_tool(ctx.out_tx.downgrade(), Arc::clone(&pending));
        let map: LocalTools = Arc::new(HashMap::from([(QUESTION_TOOL_NAME.to_owned(), tool)]));
        (Vec::new(), map)
    } else {
        (vec![QUESTION_TOOL_NAME], LocalTools::default())
    };
    // The ACP process cwd owns env, Lua, and application config, but the client
    // picks the session cwd. So permissions, where a saved answer lands, and
    // MCP config all follow the session's project, not ours.
    let project_trusted = ctx.project_config.is_trusted();
    let permissions_config = maki_config::load_permissions(&ctx.project_config);
    let (handle, events) = headless::spawn_interactive(InteractiveParams {
        model: ctx.model.clone(),
        config: ctx.config,
        permissions_config,
        timeouts: ctx.timeouts,
        prompt_slots: ctx.prompt_slots,
        excluded_tools,
        mcp_handle: ctx.mcp.clone(),
        initial_wd: ctx.cwd.clone(),
        resumed: ctx.resumed,
        claim: ctx.claim.clone(),
        storage: ctx.storage,
        yolo: ctx.yolo,
        unattended: false,
        system_prompt_override: None,
        append_system_prompt: None,
        defaults: ctx.defaults,
        model_policy: ctx.model_policy,
        plugin_rules: ctx.plugin_rules,
        project_config: ctx.project_config,
        local_tools,
        // Results ride the next prompt. The client never asked for a turn,
        // so nothing wakes one.
        background: background_mode(ctx.background),
        // `close_all_sessions` ends it, on a stop as on end of input.
        stop: CancelToken::none(),
    });
    let session_ref = handle.session_id.clone();
    // The pump is the session's own task: it only reads this runtime's event
    // stream and only writes frames addressed to this runtime's id, so pumps
    // never serialize against each other. Slow setup already happened in the
    // caller; the loop only files the finished runtime here.
    // The run's blocking mode decides ask lifetime at settlement: a blocking
    // turn cannot outlive its calls, so its terminal retires every ask, while
    // a detached turn may still have moved calls parked on theirs.
    // 02.10a supervision, from activation onward and including while idle:
    // the pump proves the handle live through the input channel (a dead run
    // drops its receiver), reports stream end as `lost`, and takes the loss
    // path on an `Error` the dead handle cannot survive. The supervisor
    // reports a silently-exited or panicked pump as `pump_dead`. Both run off
    // the shared routing loop and touch only this session, so peers stay
    // usable through a sibling's loss.
    let pending = Arc::new(Mutex::new(Pending {
        blocking: !ctx.background,
        ..Pending::default()
    }));
    let live_tx = handle.input_tx.clone();
    let is_live: Liveness = Arc::new(move || !live_tx.is_disconnected());
    let (pump_done_tx, pump_done_rx) = flume::bounded::<()>(1);
    let session_key = session_ref.to_string();
    let pump = start_event_pump(
        events,
        session_ref.clone(),
        ctx.out_tx.clone(),
        Arc::clone(&pending),
        ctx.cwd.clone(),
        maki_storage::paths::home(),
        project_trusted,
        ctx.initial_cost,
        is_live,
        true,
        Some(pump_done_tx),
    );
    let supervisor =
        start_loss_supervisor(Arc::clone(&pending), ctx.out_tx, session_key, pump_done_rx);
    let runtime = SessionRuntime {
        handle,
        mcp: ctx.mcp,
        current_mode: ctx.mode,
        current_model: ctx.model.spec(),
        cwd: ctx.cwd.clone(),
        output: ctx.output,
        pending,
        claim: ctx.claim,
        pump,
        supervisor,
    };
    (session_ref, runtime)
}

/// `false` is a zero `--background-timeout`: no seed is published, so no
/// tool call can move off its turn and every call blocks until it finishes.
fn background_mode(background: bool) -> BackgroundMode {
    if background {
        BackgroundMode::Detached
    } else {
        BackgroundMode::Blocking
    }
}

/// Registers each ask before sending so a response cannot race past it.
fn ask_client(
    out_tx: &Sender<Value>,
    pending: &PendingState,
    ask: Ask,
    request: AgentRequest,
) -> i64 {
    let id = NEXT_OUTGOING_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    pending.lock().unwrap().asks.insert(id, ask);
    send(
        out_tx,
        Request {
            id: RequestId::Number(id),
            method: Arc::from(request.method()),
            params: Some(request),
        },
    );
    id
}

/// Shadows the Lua `question` tool: sends `elicitation/create` to the client
/// and blocks the tool call until the form comes back. Serializes on the same
/// answer channel as main-agent permissions.
/// Weak, because `LocalTools` is captured verbatim by every `LuaCtx` a tool
/// call parks and an idle VM never collects it. A strong clone here outlives
/// `serve()` and leaves `writer_task` waiting on a sender nobody will drop.
fn question_tool(out_tx: WeakSender<Value>, pending: PendingState) -> LocalTool {
    // The audience the Lua `question` tool carries: shadowing a tool must not
    // widen who may call it.
    local_tool(ToolAudience::MAIN, move |input, ctx| {
        let out_tx = out_tx.clone();
        let pending = Arc::clone(&pending);
        Box::pin(async move {
            let session_id = ctx
                .session_id
                .as_ref()
                .map(ToString::to_string)
                .ok_or("no session")?;
            // Batch/code_execution children dispatch with an empty id; a
            // scope pointing at a tool call the client never saw would get
            // the elicitation rejected or dropped.
            let tool_call_id = ctx.tool_use_id.filter(|id| !id.is_empty());
            let out_tx = out_tx.upgrade().ok_or("connection closed")?;
            let request = elicitation::form_request(&session_id, tool_call_id, &input)?;
            let rx = ctx.user_response_rx.as_ref().ok_or("no answer channel")?;

            let guard = rx.lock().await;
            let epoch = pending.lock().unwrap().epoch.current();
            let request = AgentRequest::CreateElicitationRequest(
                request.meta(extension::turn_epoch_meta(epoch)),
            );
            let id = ask_client(
                &out_tx,
                &pending,
                Ask::Elicitation {
                    session: session_id.clone(),
                    epoch,
                },
                request,
            );
            let response = ctx.cancel.race(guard.recv_async()).await;
            // Cleared while still holding the channel, so a stale id cannot
            // clobber whatever ask comes next.
            pending.lock().unwrap().asks.remove(&id);
            drop(guard);

            Ok(match response {
                Ok(Ok(raw)) => elicitation::format_response(&input, &raw),
                _ => elicitation::DISMISSED.to_owned(),
            })
        })
    })
}

/// Servers the client injects on `session/new` and `session/load`. A transport we
/// cannot speak is dropped like a broken `mcp.toml` entry: losing one server beats
/// losing the session.
fn injected_servers(servers: &[McpServer]) -> Vec<(String, RawTransport)> {
    servers
        .iter()
        .filter_map(|server| match server {
            McpServer::Http(http) => Some((
                server_name(&http.name),
                RawTransport::Http(RawHttpFields {
                    url: http.url.clone(),
                    headers: pairs(&http.headers, |h| (&h.name, &h.value)),
                    oauth: None,
                    ca_file: None,
                }),
            )),
            McpServer::Stdio(stdio) => Some((
                server_name(&stdio.name),
                RawTransport::Stdio(RawStdioFields {
                    command: iter::once(stdio.command.to_string_lossy().into_owned())
                        .chain(stdio.args.iter().cloned())
                        .collect(),
                    environment: pairs(&stdio.env, |e| (&e.name, &e.value)),
                }),
            )),
            _ => {
                warn!("ignoring injected MCP server, only http and stdio are supported");
                None
            }
        })
        .collect()
}

/// Clients name their servers freely, maki names them like `mcp.toml` does.
fn server_name(name: &str) -> String {
    name.chars()
        .take(MAX_SERVER_NAME_LEN)
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn pairs<T>(items: &[T], split: impl Fn(&T) -> (&String, &String)) -> HashMap<String, String> {
    items
        .iter()
        .map(|item| {
            let (name, value) = split(item);
            (name.clone(), value.clone())
        })
        .collect()
}

/// MCP is per session: the client picks the cwd and may inject its own servers.
/// Returns as soon as the config is read, the first prompt waits for the tools.
async fn start_mcp(
    cwd: &Path,
    servers: &[McpServer],
    project_config: ProjectConfig,
) -> Option<McpHandle> {
    let (handle, errors) =
        mcp::start_with_extra(cwd, project_config, injected_servers(servers)).await;
    if !errors.is_empty() {
        warn!(%errors, "MCP config errors");
    }
    handle
}

fn trusted_project_config(
    cwd: &Path,
    storage: &StateDir,
    mode: TrustMode,
    policy: &TrustConfig,
) -> ProjectConfig {
    let mut decision = project::resolve(storage, cwd, mode);
    let matched = decision
        .state
        .unanswered()
        .and_then(|question| policy_grant(question, policy));
    if let Some(pattern) = matched {
        // ACP has no card, so policy is the only yes a cwd with no stored
        // decision can get. Recorded like any other yes so `maki trust list`
        // shows what this server trusted on the client's behalf. `unanswered`
        // and not `question`: a recorded `Never` is a stored decision, and
        // granting over it would wipe the rejection out of the store.
        info!(%pattern, cwd = %cwd.display(), "ACP folder trusted by trust.paths policy");
        decision = project::apply_answer(storage, decision, TrustAnswer::Trust);
    }
    // ACP never asks, so the restriction notice is part of what it reports.
    for warning in decision.notices() {
        warn!(%warning, "ACP project configuration trust warning");
    }
    decision.project_config
}

/// Drops one addressed runtime: the only entry `session/load` replaces. The
/// release runs on a background task — aborting the handles is prompt, so
/// the load's read-after-close still finds a stopped run — and a miss is a
/// no-op, so a `session/load` for a stored-but-not-live session only adds
/// its runtime.
async fn close_one_session(srv: &mut Server, key: &str, reason: SessionEndReason) {
    let Some(runtime) = srv.sessions.remove(key) else {
        return;
    };
    let hook = srv.on_session_end.clone();
    // No request id to answer: replacement reports through the lifecycle
    // frame, and the load result answers the client.
    begin_release(
        srv,
        runtime,
        key.to_owned(),
        reason.to_string(),
        None,
        hook,
        reason,
    );
}

/// Drops every runtime, for end of input and stop. Each release runs on its
/// own background task — drained up front so one slow teardown cannot strand
/// a peer — and this await collects every outcome, including replacements
/// already in flight. Unactivated leases drop through the 02.3 helper;
/// in-flight startups are aborted (their completions can never settle once
/// the loop is gone, and each owns its claim in its task, so aborting drops
/// the lease with the pieces). Cleanup failures surface as `release_failed`
/// lifecycle frames, never silently discarded.
async fn close_all_sessions(srv: &mut Server, reason: SessionEndReason) {
    close_all_allocations(srv);
    for (_, task) in srv.activating.drain() {
        task.abort();
        let _ = task.await;
    }
    // Drained up front so one slow teardown cannot strand a peer runtime.
    let runtimes: Vec<(String, SessionRuntime)> =
        std::mem::take(&mut srv.sessions).into_iter().collect();
    for (key, runtime) in runtimes {
        let hook = srv.on_session_end.clone();
        begin_release(srv, runtime, key, reason.to_string(), None, hook, reason);
    }
    // Every release — table entries above plus any replacement already in
    // flight — reaps here, each bounded by its own `RELEASE_TIMEOUT`.
    while srv.release_tasks.join_next().await.is_some() {}
}

#[derive(Debug)]
struct Restored {
    /// Handed to the run whole, so the turns that follow write back into the
    /// session this read rather than into a blank one under the same id.
    session: StoredSession,
    /// Only set when the session recorded an absolute cwd.
    cwd: Option<PathBuf>,
}

/// History plus the absolute cwd the session recorded in its header. Tool
/// inputs from a past run resolve against that cwd, not the client's current
/// one; a non-absolute recording falls back to the caller's cwd.
///
/// Reads from the server's state dir, the one the run writes back to, rather
/// than resolving a second answer to where sessions live.
///
/// Claims before it reads, since the turns that follow replace this transcript.
fn claim_history(
    storage: &StateDir,
    session_id: MakiId,
) -> Result<(Restored, SessionClaim), AcpError> {
    let (session, claim) = StoredSession::claim_and_load(session_id, storage)
        .map_err(|e| load_error(e, session_id))?;
    Ok((Restored::from(session), claim))
}

/// [`claim_history`] for the session this server already holds.
fn reload_history(storage: &StateDir, claim: &SessionClaim) -> Result<Restored, AcpError> {
    StoredSession::load_claimed(claim, storage)
        .map(Restored::from)
        .map_err(|e| load_error(e, claim.id()))
}

fn load_error(e: SessionError, session_id: MakiId) -> AcpError {
    match e.is_busy() {
        // `not found` would send the client looking for a session it can see
        // right there in the list.
        true => AcpError::new(SESSION_BUSY_CODE, e.to_string()),
        false => {
            AcpError::resource_not_found(Some(format!("session/{session_id}"))).data(json_str(&e))
        }
    }
}

impl From<StoredSession> for Restored {
    fn from(session: StoredSession) -> Self {
        let cwd = Path::new(&session.cwd)
            .is_absolute()
            .then(|| PathBuf::from(&session.cwd));
        Self { cwd, session }
    }
}

fn handle_prompt(srv: &mut Server, raw: &Value, id: &RequestId) -> Result<(), AcpError> {
    let req: PromptRequest = parse_params(raw)?;
    // An unactivated allocation holds no provider and no task: the prompt is
    // rejected before anything is queued or started, and neither the lease
    // nor any peer is touched.
    if srv.allocations.contains_key(&*req.session_id.0)
        || srv.activating.contains_key(&*req.session_id.0)
    {
        return Err(not_activated(&req.session_id.0));
    }
    // Addressed to the request's own session; no current-session fallback. An
    // unknown id rejects here, before anything is queued or any peer is
    // touched; duplicate and cross-session rejection keep the same shape.
    // A supervised-dead handle answers the same `resource_not_found`: the
    // 02.1 contract says a `lost` handle is unusable and the client must
    // release it (02.10b drops the routing entry then).
    let session = srv.session(&req.session_id)?;
    if session.pending.lock().unwrap().is_lost() {
        return Err(unknown_session(&req.session_id.0));
    }
    // 02.9a: a cancelled-awaiting-cleanup session accepts nothing until the
    // pump confirms its tool cleanup (02.9b: the cancelled run's terminal).
    // Settlement alone never reopens it; the peer's slot is untouched by
    // this rejection.
    let resumable = session.pending.lock().unwrap().is_resumable();
    if !resumable {
        let uncertain = session.pending.lock().unwrap().is_uncertain();
        return Err(cleanup_unconfirmed(&req.session_id.0, uncertain));
    }
    debug!(
        session = %req.session_id.0,
        epoch = session.pending.lock().unwrap().epoch.current(),
        "prompt addressed"
    );

    let (message, images) = extract_prompt_content(&req.prompt);
    let input = AgentInput::from_defaults(
        message,
        session.current_mode.clone(),
        images,
        srv.defaults,
        InputSource::Acp,
    );

    // One outstanding prompt per session, checked and set under the same
    // guard: `input_tx` is unbounded, so a second prompt would queue happily
    // and overwrite the first id, leaving that request unanswered forever.
    // The turn snapshots the live epoch, so the terminal that settles this
    // prompt can only settle this turn, never the next one.
    // 02.10a: a dead input channel means the supervised task is gone. Report
    // loss exactly once (lifecycle `lost` plus the discovering prompt's loss
    // error, or nothing on a repeat) instead of queueing onto a dead run.
    // Non-blocking and peer-free: one lock plus at most two wire sends for
    // this session only. The claim stays held for 02.10b's bounded release.
    let mut pending = session.pending.lock().unwrap();
    if pending.prompt.is_some() {
        return Err(AcpError::new(-32603, "a prompt is already running"));
    }
    if session.handle.input_tx.send(input).is_err() {
        drop(pending);
        let first = report_loss(
            &session.pending,
            &srv.out_tx,
            &req.session_id.0,
            "input_channel_dead",
        );
        return Err(match first {
            true => loss_error(&req.session_id.0, "input_channel_dead"),
            false => unknown_session(&req.session_id.0),
        });
    }
    pending.turn = Some(pending.epoch.current());
    pending.prompt = Some(id.clone());
    Ok(())
}

/// Setup is fixed per session: route-then-reject for a post-activation
/// model/mode change (`EXTENSION.md`). Routing comes first, so an unknown
/// session still answers `resource_not_found`; a routed one is rejected
/// without applying anything, emitting anything, or touching the peer. Nori
/// starts another session for a different setup. The error shape stays
/// standard, so standard clients keep working.
fn fixed_at_activation() -> AcpError {
    AcpError::invalid_params().data(serde_json::json!({
        "reason": extension::FIXED_AT_ACTIVATION_REASON,
    }))
}

fn handle_set_mode(srv: &Server, raw: &Value) -> Result<AgentResponse, AcpError> {
    let req: SetSessionModeRequest = parse_params(raw)?;
    // An unactivated allocation has no fixed setup yet: the change is
    // rejected as not-activated, not applied, and the lease is untouched.
    // Routing still comes first, so an unknown session answers
    // `resource_not_found`.
    if srv.allocations.contains_key(&*req.session_id.0)
        || srv.activating.contains_key(&*req.session_id.0)
    {
        return Err(not_activated(&req.session_id.0));
    }
    let session = srv.session(&req.session_id)?;
    // 02.10a: a lost handle answers `resource_not_found`, like an unknown id.
    if session.pending.lock().unwrap().is_lost() {
        return Err(unknown_session(&req.session_id.0));
    }
    debug!(
        session = %req.session_id.0,
        mode = ?session.current_mode,
        epoch = session.pending.lock().unwrap().epoch.current(),
        "mode change rejected: setup fixed at activation"
    );
    Err(fixed_at_activation())
}

fn handle_set_config(srv: &Server, raw: &Value) -> Result<AgentResponse, AcpError> {
    let req: SetSessionConfigOptionRequest = parse_params(raw)?;
    // Same lease rule as `handle_set_mode`: no fixed setup exists before
    // activation, so the change is rejected without touching the lease.
    if srv.allocations.contains_key(&*req.session_id.0)
        || srv.activating.contains_key(&*req.session_id.0)
    {
        return Err(not_activated(&req.session_id.0));
    }
    let session = srv.session(&req.session_id)?;
    // 02.10a: a lost handle answers `resource_not_found`, like an unknown id.
    if session.pending.lock().unwrap().is_lost() {
        return Err(unknown_session(&req.session_id.0));
    }
    debug!(
        session = %req.session_id.0,
        model = %session.current_model,
        epoch = session.pending.lock().unwrap().epoch.current(),
        "config change rejected: setup fixed at activation"
    );
    Err(fixed_at_activation())
}

fn handle_notification(srv: &Server, method: &str, raw: &Value) {
    match method {
        "session/cancel" => {
            // Legacy best-effort Stop: identity-check and dispatch only. The
            // notification names its own session, a miss is dropped (no id to
            // answer on), and a hit only signals the run; the run's terminal
            // settles the prompt through `settle_prompt` as usual, so no
            // epoch moves here. Session/turn scoping with synchronous
            // exactly-once settlement and ask retirement is the addressed
            // `_maki/session/cancel` method (`handle_cancel`, 02.9a), which
            // serializes through the same per-session control lock.
            let Ok(note) = parse_params::<CancelNotification>(raw) else {
                return;
            };
            // Asks stay on this legacy path: a moved call outlives the turn
            // and may be parked on one, and a stale answer is tagged, so
            // whoever waits next drops it. Retiring asks here is not built:
            // only agent-settled terminals (Done/Error through
            // `settle_prompt`) and addressed cancels (through
            // `settle_cancel`) retire asks, and only agent terminal
            // settlement honors the blocking/detached distinction (an
            // addressed cancel stops the run, so it retires in both modes).
            // An unactivated allocation or an unknown id names no run to
            // signal, so the notification is dropped the same way.
            if let Ok(session) = srv.session(&note.session_id) {
                let _ = session.handle.cancel_tx.try_send(());
            }
        }
        _ => debug!(method, "unknown notification"),
    }
}

fn handle_incoming_response(srv: &Server, raw: &Value) {
    let id = raw.get("id").map(request_id).unwrap_or(RequestId::Null);
    let Some(want) = ask_id(&id) else {
        warn!(%id, "response for an id no ask can carry");
        return;
    };
    // Ask ids are minted from one process-wide counter, so the owning runtime
    // is found by identity without touching peers. The claim is
    // remove-then-validate under that runtime's control guard — one critical
    // section — so a cancel settling concurrently cannot slip between the
    // check and the take: whichever holds the guard first wins, and the loser
    // finds nothing to act on. A queued answer can never bypass an in-flight
    // cancel settlement this way; 02.10b's Close joins through the same
    // guard.
    let mut found = None;
    for (key, runtime) in srv.sessions.iter() {
        if let Some(ask) = runtime.pending.lock().unwrap().claim_ask(want, key) {
            found = Some((key.clone(), ask));
            break;
        }
    }
    let Some((owner, ask)) = found else {
        warn!(%id, "response for an unknown, answered, or stale ask");
        return;
    };
    let Some(runtime) = srv.sessions.get(&owner) else {
        warn!(%id, owner, "response for a released session");
        return;
    };
    // The claim already validated the binding: the ask named this session and
    // the live turn. Every ask is consumed exactly once on claim, so a
    // repeated answer for the same id finds nothing and is dropped; a settled
    // terminal (blocking mode) or an addressed cancel (both modes) already
    // retired every ask, so a late answer finds nothing either. Nothing
    // rejected here ever reaches an agent, so a stale allow can never green a
    // later turn's tool call.
    let main_answer_tx = runtime.handle.answer_tx.clone();
    let (answer, answer_tx) = match &ask {
        Ask::Permission {
            request_id,
            answer_tx,
            ..
        } => (
            TaggedAnswer::new(request_id.clone(), permission_answer(raw)).encode(),
            answer_tx.as_ref(),
        ),
        // The waiting question tool parses this; an error response decodes to
        // nothing and counts as a dismissal.
        Ask::Elicitation { .. } => (
            raw.get("result")
                .cloned()
                .unwrap_or(Value::Null)
                .to_string(),
            None,
        ),
    };
    let _ = answer_tx.unwrap_or(&main_answer_tx).send(answer);
}

/// Our asks go out with a numeric id, but JSON-RPC ids are `string | number`
/// and a client may echo ours back as a string. Dropping such a response would
/// park the waiting tool forever.
fn ask_id(id: &RequestId) -> Option<i64> {
    match id {
        RequestId::Number(id) => Some(*id),
        RequestId::Str(id) => id.parse().ok(),
        RequestId::Null => None,
    }
}

/// A response we cannot read still has to answer the agent, or the tool waits
/// on a permission that will never come.
fn permission_answer(raw: &Value) -> PermissionAnswer {
    match raw
        .get("result")
        .map(|result| serde_json::from_value::<RequestPermissionResponse>(result.clone()))
    {
        Some(Ok(resp)) => permissions::outcome_to_answer(&resp.outcome),
        _ => PermissionAnswer::Deny,
    }
}

fn extract_prompt_content(blocks: &[ContentBlock]) -> (String, Vec<ImageSource>) {
    let mut text = String::new();
    let mut images = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text(TextContent { text: t, .. }) => append(&mut text, t),
            ContentBlock::Image(ImageContent {
                data, mime_type, ..
            }) => images.push(ImageSource::new(
                image_media_type(mime_type),
                Arc::from(data.as_str()),
            )),
            ContentBlock::Resource(res) => {
                if let EmbeddedResourceResource::TextResourceContents(trc) = &res.resource {
                    append(&mut text, &format!("--- {} ---\n{}", trc.uri, trc.text));
                }
            }
            ContentBlock::ResourceLink(rl) => append(&mut text, &format!("[Resource: {}]", rl.uri)),
            _ => {}
        }
    }

    (text, images)
}

fn append(text: &mut String, part: &str) {
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(part);
}

fn image_media_type(mime: &str) -> ImageMediaType {
    match mime {
        "image/png" => ImageMediaType::Png,
        "image/gif" => ImageMediaType::Gif,
        "image/webp" => ImageMediaType::Webp,
        _ => ImageMediaType::Jpeg,
    }
}

/// Liveness probe the pump consults before trusting an `Error` terminal.
/// Production proves the handle live through its input channel (a dead run
/// drops its receiver); fixtures inject liveness directly, so no test needs a
/// provider, keys, or network.
type Liveness = Arc<dyn Fn() -> bool + Send + Sync>;

#[allow(clippy::too_many_arguments)]
fn start_event_pump(
    mut events: SessionEvents,
    session_id: SessionRef,
    out_tx: Sender<Value>,
    pending: PendingState,
    cwd: PathBuf,
    home: Option<PathBuf>,
    project_trusted: bool,
    initial_cost: Option<f64>,
    is_live: Liveness,
    report_stream_end: bool,
    done: Option<Sender<()>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Held until the pump exits (return, panic unwind, or abort drop):
        // the loss supervisor awaits its disconnect, so a silently-exited or
        // panicked pump still reports `pump_dead` exactly once while idle.
        let _done = done;
        let sid = SessionId::from(session_id.to_string());
        let session_key = sid.to_string();
        let mut cost = CostTally::starting_at(initial_cost);
        // A permission request only carries scopes, which are matching keys and
        // not always paths, so the file context a client needs has to come from
        // the tool call that asked for permission. Kept for the turn, like
        // sdk_mode does: the history holds these inputs anyway.
        let mut tool_inputs: HashMap<String, Value> = HashMap::new();

        while let Some(Envelope {
            event, subagent, ..
        }) = events.next().await
        {
            // A subagent's turn spends session money even though its events
            // stay out of the transcript.
            if let AgentEvent::TurnComplete(tc) = &event {
                cost.turn(tc.cost);
            }

            let update = match event {
                AgentEvent::PermissionRequest {
                    id,
                    tool,
                    scopes,
                    reason,
                } => {
                    let tool = tool.to_string();
                    let scope = format!("{tool}: {}", scopes.join(", "));
                    // A plugin escalated this call, and its reason is what the
                    // user most needs to read.
                    let scope = match reason {
                        Some(reason) => format!("{reason} ({scope})"),
                        None => scope,
                    };
                    // The cache is keyed by the call that asked for permission,
                    // so look it up before the remap below. A subagent is left
                    // out on purpose: nothing makes a child's ids unique
                    // against the parent's, and naming the wrong file in a
                    // security prompt is worse than naming none.
                    let raw_input = subagent.is_none().then(|| tool_inputs.get(&id)).flatten();
                    // A child's own tool call never reached the client, nor
                    // this cache, so its permission rides on the `task` call
                    // the client can see, only the child's channel may take the
                    // answer, and the title is all the dialog gets.
                    let request_id = id.clone();
                    let (id, title, answer_tx) = match subagent {
                        Some(info) => {
                            let Some(answer_tx) = info.answer_tx else {
                                warn!(
                                    subagent = info.name,
                                    parent_tool_use_id = info.parent_tool_use_id,
                                    tool_use_id = id,
                                    scope,
                                    "dropping subagent permission with no answer channel"
                                );
                                continue;
                            };
                            let title = format!("{}: {scope}", info.name);
                            (info.parent_tool_use_id, title, Some(answer_tx))
                        }
                        None => (id, scope, None),
                    };
                    // The ask carries the session and turn that issued it, so the
                    // answer can be gated on the live epoch on the way back;
                    // the frame carries the turn for request/turn correlation.
                    let epoch = pending.lock().unwrap().epoch.current();
                    let request = AgentRequest::RequestPermissionRequest(
                        RequestPermissionRequest::new(
                            sid.clone(),
                            translate::permission_update(
                                id,
                                title,
                                &tool,
                                raw_input,
                                &cwd,
                                home.as_deref(),
                            ),
                            permissions::permission_options(project_trusted),
                        )
                        .meta(extension::turn_epoch_meta(epoch)),
                    );
                    ask_client(
                        &out_tx,
                        &pending,
                        Ask::Permission {
                            request_id,
                            answer_tx,
                            session: sid.to_string(),
                            epoch,
                        },
                        request,
                    );
                    continue;
                }
                _ if subagent.is_some() => continue,
                AgentEvent::TextDelta { text } => translate::text_delta(&text),
                AgentEvent::ThinkingDelta { text } => translate::thinking_delta(&text),
                AgentEvent::ToolPending { id, name } => translate::tool_pending(&id, &name),
                AgentEvent::ToolStart(event) => {
                    let update = translate::tool_start(&event, &cwd, home.as_deref());
                    if let Some(raw_input) = event.raw_input {
                        tool_inputs.insert(event.id, raw_input);
                    }
                    update
                }
                AgentEvent::ToolOutput { id, content } => translate::tool_output(&id, &content),
                AgentEvent::ToolDone(event) => translate::tool_done(&event, &cwd, home.as_deref()),
                AgentEvent::TurnComplete(event) => translate::usage_update(&event, cost.total()),
                AgentEvent::ObservationLine { text, .. } => {
                    translate::notice(&MakiId::generate().to_string(), &text)
                }
                AgentEvent::Done {
                    reason,
                    cost: ledger,
                    raw_stop_reason,
                    ..
                } => {
                    cost.settle(ledger);
                    tool_inputs.clear();
                    if let Some((id, turn)) = settle_prompt(&pending) {
                        // The standard `stopReason` keeps its standard value;
                        // the verbatim wire code (or the `Done` name for
                        // agent-side endings) and task usability ride along as
                        // additive `_meta`, never remapped.
                        let raw = raw_stop_reason.unwrap_or_else(|| reason.to_string());
                        let resp = PromptResponse::new(translate::map_done_reason(reason))
                            .meta(extension::terminal_meta(turn, &raw, reason.usable()));
                        send(
                            &out_tx,
                            Response::<AgentResponse>::new(
                                id,
                                Ok(AgentResponse::PromptResponse(resp)),
                            ),
                        );
                    } else {
                        // 02.9b: no live turn to settle. When an addressed
                        // cancel settled the turn ahead of its run, this
                        // terminal is that run ending: under
                        // `--background-timeout 0` nothing outlives the run,
                        // so its end confirms the cancelled turn's blocking
                        // tools stopped and reopens the session for exactly
                        // one new turn. Nothing is answered: the prompt
                        // already settled through `settle_cancel`.
                        pending.lock().unwrap().note_run_ended(false);
                    }
                    continue;
                }
                AgentEvent::Error {
                    message,
                    auth,
                    totals,
                } => {
                    // A turn that dies on a provider 500 never reaches `Done`,
                    // and the pump outlives the session, so without this the
                    // whole file a `write` was carrying stays pinned forever.
                    tool_inputs.clear();
                    cost.settle(totals.cost);
                    // 02.10a: an `Error` alone is neither normal completion nor
                    // proof the handle accepts another prompt. When the
                    // supervised task/pump proves the handle live (input
                    // channel still connected), settle as a failed turn as
                    // before; otherwise take the loss path (settle once via
                    // the turn epoch, retire asks in both modes, notify
                    // `lost` with reason/raw/usability). Peers are untouched:
                    // only this session's pending/out-tx move, off-loop.
                    if !is_live() {
                        report_loss(&pending, &out_tx, &session_key, &message);
                        continue;
                    }
                    if let Some((id, _turn)) = settle_prompt(&pending) {
                        let error = if auth {
                            AcpError::auth_required()
                        } else {
                            AcpError::internal_error()
                        };
                        let error = error.data(Value::String(message));
                        send(&out_tx, Response::<AgentResponse>::new(id, Err(error)));
                    } else {
                        // 02.9b: the cancelled run failed instead of stopping.
                        // Its tools may have stopped with it, but the abnormal
                        // end proves nothing, so the session records the
                        // uncertainty and stays non-resumable; the retry
                        // rejection reports it. (Task/pump loss itself is
                        // 02.10a's; this arm only records the failed cleanup
                        // when the handle is still proved live.)
                        pending.lock().unwrap().note_run_ended(true);
                    }
                    continue;
                }
                _ => continue,
            };
            let epoch = pending.lock().unwrap().epoch.current();
            session_update(&out_tx, &sid, epoch, update);
        }
        // 02.10a: the event stream ends only when the task drops its guard
        // (return, panic, or provider-init failure that already sent its
        // `Error`): no normal turn ends the stream, since the task outlives
        // every prompt. Report supervised loss exactly once (first observer
        // wins; a racing `Error`-loss or supervisor `pump_dead` collapses to
        // the same single `lost`). Aborted pumps never reach here (abort
        // drops the future), so shutdown/close reports no loss. Throwaway
        // test pumps opt out via `report_stream_end == false`.
        if report_stream_end {
            report_loss(&pending, &out_tx, &session_key, "stream_ended");
        }
    })
}

/// 02.10a pump-death supervisor, off the shared routing loop. The pump holds
/// the done-sender until exit; this task awaits its disconnect, so a pump
/// that panics or silently exits still notifies `lost` exactly once while
/// idle, without polling the loop. The first observer (pump stream-end vs.
/// this supervisor) wins via `report_loss`; the loser sends nothing. Aborted
/// with the pump on release/shutdown — after the handle is marked released
/// under the control lock — so teardown reports no loss. No teardown
/// runs here; bounded release (`begin_release`) owns it.
fn start_loss_supervisor(
    pending: PendingState,
    out_tx: Sender<Value>,
    session_id: String,
    pump_done: flume::Receiver<()>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let _ = pump_done.recv_async().await;
        // A normal return already reported stream-end loss (or never needed
        // to: `Done` turns keep the stream open); only a pump that died
        // without reporting reaches here un-lost.
        if pending.lock().unwrap().is_lost() {
            return;
        }
        report_loss(&pending, &out_tx, &session_id, "pump_dead");
    })
}

/// Settles the outstanding prompt exactly once through the turn epoch: the
/// first terminal for the live turn takes the prompt and bumps the epoch, and
/// answers with the epoch the turn settled under. A terminal with no prompt
/// outstanding, or one whose turn is no longer live, settles nothing and is
/// dropped, so it can never settle the next turn instead.
///
/// A settling terminal also retires every outstanding ask in blocking mode:
/// no call can outlive a blocking turn, so a blocking terminal turn cannot
/// retain unresolved asks and a stale response arriving later finds an
/// unknown id, never a live ask to revive. Detached sessions keep their asks:
/// a call moved to the background can still be parked on one. Ask retirement
/// on the addressed-cancel path is 02.9a's (`settle_cancel`, both modes: the
/// run is stopped, so nothing outlives a cancelled turn); retirement on the
/// handle-loss path is 02.10a's (`settle_loss`, both modes: the handle is
/// dead, so nothing can outlive it).
fn settle_prompt(pending: &PendingState) -> Option<(RequestId, u64)> {
    let mut guard = pending.lock().unwrap();
    let live = match (&guard.prompt, guard.turn) {
        (Some(_), Some(turn)) => guard.epoch.check(turn).is_ok(),
        _ => false,
    };
    if !live {
        return None;
    }
    let turn = guard.epoch.current();
    guard.epoch.bump();
    guard.turn = None;
    if guard.blocking {
        guard.asks.clear();
    }
    guard.prompt.take().map(|id| (id, turn))
}

fn send(out_tx: &Sender<Value>, msg: impl Serialize) {
    if let Ok(json) = serde_json::to_value(JsonRpcMessage::wrap(msg)) {
        let _ = out_tx.send(json);
    }
}

fn session_update(out_tx: &Sender<Value>, sid: &SessionId, epoch: u64, update: SessionUpdate) {
    let notification = AgentNotification::SessionNotification(
        SessionNotification::new(sid.clone(), update).meta(extension::turn_epoch_meta(epoch)),
    );
    send(
        out_tx,
        Notification {
            method: Arc::from("session/update"),
            params: Some(notification),
        },
    );
}

fn parse_params<T: serde::de::DeserializeOwned>(raw: &Value) -> Result<T, AcpError> {
    serde_json::from_value(raw.get("params").cloned().unwrap_or(Value::Null))
        .map_err(|e| AcpError::invalid_params().data(json_str(&e)))
}

fn json_str(e: &impl std::fmt::Display) -> Value {
    Value::String(e.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use maki_agent::headless::SessionEnd;
    use maki_agent::permissions::{PermissionCheck, PermissionError, PermissionManager};
    use maki_agent::tools::PermissionScopes;
    use maki_agent::{
        CancelToken, DoneReason, EventSender, RunTotals, SubagentInfo, ToolStartEvent,
        TurnCompleteEvent,
    };
    use maki_config::project::TrustQuestion;
    use maki_config::{AgentConfig, Effect, ToolKey, TrustFileConfig};
    use maki_providers::{ContentBlock as MsgBlock, Message, Role, Timeouts, TokenUsage};
    use maki_storage::StateDir;
    use maki_storage::sessions::{Session, StoredTokenUsage};
    use maki_storage::trusted_folders::{CanonicalFolder, TrustStatus, TrustedFolders};
    use std::future;
    use std::sync::atomic::AtomicBool;

    use tempfile::TempDir;
    use test_case::test_case;
    use tokio::sync::Notify;
    use tokio::time::timeout;
    use tokio_util::task::AbortOnDropHandle;

    use super::*;

    const ANSWERED_ID: i64 = 1001;
    const UNKNOWN_ID: i64 = 1002;
    const DISCOVERED_SPEC: &str = "openrouter/discovered-model";
    const OFFLINE_SPEC: &str = "openai/gpt-5";
    const SELECTED_SPEC: &str = "openai/gpt-5.6-sol";
    const FAILED_LOAD_KEEPS_SESSION: &str = "a load that failed must not close the running session";
    /// Neither resolves in the price tables, so nothing can re-price a restored
    /// session back onto the recorded number by luck.
    const RETIRED_SPEC: &str = "retired-vendor/retired-model-9000";
    const RETIRED_MODEL_ID: &str = "retired-model-9000";
    const RECORDED_COST: f64 = 1.25;
    /// Generous on purpose: the work under test is a few file reads, so any
    /// wait near this long is the deadlock and not a slow machine.
    const STDIN_DEADLOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    const DENY_SCOPE: &str = "acp-session-trust-boundary-test-deny";
    const ALLOW_SCOPE: &str = "acp-session-trust-boundary-test-allow";
    const POLICY_MATCH_GLOB: &str = "**";
    const POLICY_MISS_GLOB: &str = "/nowhere/*";
    const GATED_INIT_SOURCE: &str = "return {}";
    /// The tool a later turn asks about. Nothing allows `bash` by default, so
    /// an `Ok` here can only come from an answer that was applied.
    const NEXT_TURN_TOOL: &str = "bash";
    const NEXT_TURN_SCOPE: &str = "rm -rf /";
    const NEXT_TURN_TOOL_USE_ID: &str = "toolu_next_turn";
    const STALE_ALLOW: &str = "a stale answer must never allow the next turn's tool";
    const CLOSE_WAITED: &str = "closing a session must not wait for its background work";
    const BLOCKING_MODE: &str = "a zero --background-timeout must leave every tool call blocking";

    /// Where fixture servers claim their placeholder session. One dir is enough,
    /// since fresh ids never collide and nothing is written under them.
    static FIXTURE_CLAIMS: LazyLock<TempDir> = LazyLock::new(|| TempDir::new().unwrap());

    /// The client picks the session cwd, so that folder's stored trust decides
    /// whether its `.maki` may widen permissions. Its deny rules need no trust:
    /// a repository can only narrow what the agent may do.
    #[test]
    fn session_project_config_follows_stored_folder_trust() {
        let state = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        let maki_dir = project.path().join(".maki");
        std::fs::create_dir(project.path().join(".git")).unwrap();
        std::fs::create_dir(&maki_dir).unwrap();
        std::fs::write(
            maki_dir.join("permissions.toml"),
            format!("[bash]\ndeny = [\"{DENY_SCOPE}\"]\nallow = [\"{ALLOW_SCOPE}\"]\n"),
        )
        .unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());

        let untrusted = trusted_project_config(
            project.path(),
            &storage,
            TrustMode::Consult,
            &TrustConfig::default(),
        );
        assert!(!untrusted.is_trusted());
        let rules = maki_config::load_permissions(&untrusted).rules;
        assert!(
            rules
                .iter()
                .any(|rule| bash_rule(rule, DENY_SCOPE, Effect::Deny))
        );
        assert!(
            !rules
                .iter()
                .any(|rule| rule.scope.as_deref() == Some(ALLOW_SCOPE))
        );

        let folder = CanonicalFolder::resolve(project.path()).unwrap();
        project::grant(&storage, &TrustQuestion::for_folder(&folder)).unwrap();

        let trusted = trusted_project_config(
            project.path(),
            &storage,
            TrustMode::Consult,
            &TrustConfig::default(),
        );
        assert!(trusted.is_trusted());
        assert_eq!(
            trusted.config_root(),
            ProjectConfig::for_project(project.path()).config_root()
        );
        let rules = maki_config::load_permissions(&trusted).rules;
        assert!(
            rules
                .iter()
                .any(|rule| bash_rule(rule, DENY_SCOPE, Effect::Deny))
        );
        assert!(
            rules
                .iter()
                .any(|rule| bash_rule(rule, ALLOW_SCOPE, Effect::Allow))
        );
    }

    /// ACP resolves folder trust from its dispatch loop, on the same executor
    /// that another thread is blocking on a whole stdin read. Reaching for
    /// stdin on a path that can never ask a question parked the loop until the
    /// client sent bytes it was only going to send after our answer, so the
    /// first session hung forever.
    #[test]
    fn resolving_trust_without_a_prompt_does_not_wait_for_stdin() {
        let state = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        std::fs::create_dir(project.path().join(".git")).unwrap();
        std::fs::create_dir(project.path().join(".maki")).unwrap();
        std::fs::write(project.path().join(".maki/init.lua"), "return {}").unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let cwd = project.path().to_path_buf();

        let held = std::io::stdin().lock();
        let (done_tx, done_rx) = flume::bounded(1);
        let worker = std::thread::spawn(move || {
            let config =
                trusted_project_config(&cwd, &storage, TrustMode::Consult, &TrustConfig::default());
            let _ = done_tx.send(config.is_trusted());
        });

        let finished = done_rx.recv_timeout(STDIN_DEADLOCK_TIMEOUT);
        drop(held);
        worker.join().unwrap();

        assert_eq!(
            finished,
            Ok(false),
            "a non-interactive trust resolution must not touch stdin"
        );
    }

    /// A session cwd shipping one gated file, so a start there has a real
    /// question for the policy to answer.
    fn gated_project(project: &Path) {
        std::fs::create_dir(project.join(".git")).unwrap();
        std::fs::create_dir(project.join(".maki")).unwrap();
        std::fs::write(project.join(".maki/init.lua"), GATED_INIT_SOURCE).unwrap();
    }

    fn trust_policy(pattern: &str) -> TrustConfig {
        TrustConfig::from_file(TrustFileConfig {
            paths: Some(vec![pattern.to_owned()]),
            prompt: Some(false),
        })
        .expect("valid trust pattern")
    }

    /// The container image's global `init.lua` is the only thing that can
    /// answer for an ACP run, and its answer has to reach the store: the next
    /// start consults that alone, and `maki trust list` has to show it.
    #[test_case(POLICY_MATCH_GLOB, true ; "a_matching_pattern_grants_without_asking")]
    #[test_case(POLICY_MISS_GLOB, false ; "a_non_matching_pattern_leaves_the_folder_untrusted")]
    fn policy_answers_a_session_cwd(pattern: &str, expected_trusted: bool) {
        let state = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        gated_project(project.path());
        let storage = StateDir::from_path(state.path().to_path_buf());

        let config = trusted_project_config(
            project.path(),
            &storage,
            TrustMode::Consult,
            &trust_policy(pattern),
        );

        assert_eq!(config.is_trusted(), expected_trusted);
        let recorded = project::resolve(&storage, project.path(), TrustMode::Consult);
        assert_eq!(recorded.project_config.is_trusted(), expected_trusted);
    }

    /// A recorded `Never` is an answer, and `grant` replaces a rejection in the
    /// store. A policy that overturned one would delete the decision the user
    /// went out of their way to give, on every session the client opens.
    #[test]
    fn policy_does_not_overturn_a_recorded_rejection() {
        let state = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        gated_project(project.path());
        let storage = StateDir::from_path(state.path().to_path_buf());
        let folder = CanonicalFolder::resolve(project.path()).unwrap();
        project::deny(&storage, &TrustQuestion::for_folder(&folder)).unwrap();

        let config = trusted_project_config(
            project.path(),
            &storage,
            TrustMode::Consult,
            &trust_policy(POLICY_MATCH_GLOB),
        );

        assert!(!config.is_trusted());
        assert_eq!(
            TrustedFolders::new(&storage).status(&folder).unwrap(),
            TrustStatus::Rejected,
            "the rejection must survive a matching policy"
        );
    }

    fn bash_rule(rule: &maki_config::PermissionRule, scope: &str, effect: Effect) -> bool {
        rule.tool == ToolKey::native("bash")
            && rule.scope.as_deref() == Some(scope)
            && rule.effect == effect
    }

    fn allow_once(id: i64) -> Value {
        serde_json::json!({
            "id": id,
            "result": { "outcome": { "outcome": "selected", "optionId": "allow_once" } },
        })
    }

    #[test_case(allow_once(ANSWERED_ID), PermissionAnswer::AllowOnce ; "selected_option")]
    #[test_case(serde_json::json!({ "id": ANSWERED_ID, "result": { "outcome": { "outcome": "cancelled" } } }), PermissionAnswer::Deny ; "cancelled_outcome")]
    #[test_case(serde_json::json!({ "id": ANSWERED_ID, "result": { "nonsense": true } }), PermissionAnswer::Deny ; "unparsable_result")]
    #[test_case(serde_json::json!({ "id": ANSWERED_ID, "error": { "code": -32603 } }), PermissionAnswer::Deny ; "jsonrpc_error")]
    fn permission_answer_maps_response(raw: Value, expected: PermissionAnswer) {
        assert_eq!(permission_answer(&raw), expected);
    }

    /// One live runtime for single-session fixtures. Its pump task is already
    /// complete: these fixtures drive real pumps themselves via `run_pump`, so
    /// the owned handle must exist but never run.
    fn fixture_runtime() -> (String, SessionRuntime) {
        let (answer_tx, _) = flume::unbounded();
        let claim = SessionClaim::fresh(&StateDir::from_path(FIXTURE_CLAIMS.path().to_path_buf()));
        let key = SessionRef::from(claim.id()).as_str().to_owned();
        let handle = InteractiveHandle {
            tool_names: Vec::new(),
            input_tx: flume::unbounded().0,
            answer_tx,
            cancel_tx: flume::unbounded().0,
            model_tx: flume::unbounded().0,
            mode_tx: flume::unbounded().0,
            session_id: SessionRef::from(claim.id()),
            permissions: Arc::new(PermissionManager::new(
                maki_config::PermissionsConfig::default(),
                PathBuf::from("/project"),
                ProjectConfig::for_project(Path::new("/project")),
                Arc::default(),
            )),
            task: AbortOnDropHandle::new(tokio::spawn(async { SessionEnd::default() })),
            spend: Arc::default(),
        };
        let runtime = SessionRuntime {
            handle,
            mcp: None,
            current_mode: AgentMode::Build,
            current_model: String::new(),
            // Single-session unit fixture: no activation named a cwd, so the
            // placeholder stands in. Multi-session fixtures address entries by
            // key and set this from the requested cwd instead.
            cwd: PathBuf::from("/project"),
            output: None,
            pending: Arc::default(),
            claim,
            pump: tokio::spawn(async {}),
            // Owned pump never runs in these fixtures (throwaway pumps via
            // `run_pump`/`drive_runtime` do the work with stream-end reporting
            // off), so the supervisor is a no-op handle, never a loss source.
            supervisor: tokio::spawn(async {}),
        };
        (key, runtime)
    }

    fn test_server() -> (Server, flume::Receiver<String>, flume::Receiver<Value>) {
        let (answer_tx, answer_rx) = flume::unbounded();
        let (out_tx, out_rx) = flume::unbounded();
        let (key, mut runtime) = fixture_runtime();
        // The answer channel under test is this runtime's own.
        runtime.handle.answer_tx = answer_tx;
        // Dangled: single-session fixtures never activate, and nothing here
        // upgrades the weak, so the dead link is never observed.
        let (completion_tx, _completion_rx) = flume::unbounded();
        let server = Server {
            out_tx,
            model_specs: Vec::new(),
            client_elicits_form: false,
            defaults: SessionDefaults::default(),
            on_session_end: None,
            sessions: HashMap::from([(key, runtime)]),
            allocations: HashMap::new(),
            activating: HashMap::new(),
            starter: fixture_starter(&StarterHarness::default()),
            completion: completion_tx.downgrade(),
            release_tasks: JoinSet::new(),
            release_records: ReleaseRecords::default(),
        };
        (server, answer_rx, out_rx)
    }

    /// The live epoch of the single runtime single-session fixtures install.
    fn live_epoch(srv: &Server) -> u64 {
        pending(srv).lock().unwrap().epoch.current()
    }

    /// Parks an outstanding prompt for the single runtime, the way
    /// `handle_prompt` leaves it: the request id plus the turn it runs under.
    fn park_prompt(srv: &Server, id: RequestId) {
        let mut guard = pending(srv).lock().unwrap();
        guard.turn = Some(guard.epoch.current());
        guard.prompt = Some(id);
    }

    /// The single runtime single-session fixtures install. Multi-session
    /// fixtures address entries by key instead.
    fn only(srv: &Server) -> &SessionRuntime {
        assert_eq!(
            srv.sessions.len(),
            1,
            "single-session fixture holds one runtime"
        );
        srv.sessions.values().next().unwrap()
    }

    fn only_mut(srv: &mut Server) -> &mut SessionRuntime {
        assert_eq!(
            srv.sessions.len(),
            1,
            "single-session fixture holds one runtime"
        );
        srv.sessions.values_mut().next().unwrap()
    }

    fn acp_params(storage: StateDir) -> AcpParams {
        AcpParams {
            model: Model::from_spec(SELECTED_SPEC).expect("a shipped model"),
            config: AgentConfig::default(),
            timeouts: Timeouts::default(),
            initial_wd: PathBuf::from("/project"),
            prompt_slots: Arc::default(),
            yolo: false,
            defaults: SessionDefaults::default(),
            model_policy: Arc::default(),
            plugin_rules: Arc::default(),
            trust_mode: TrustMode::Consult,
            trust_policy: Arc::default(),
            on_session_end: None,
            storage,
            background: false,
            stop: CancelToken::none(),
        }
    }

    fn load_request(id: MakiId) -> Value {
        serde_json::json!({
            "params": { "sessionId": id.to_string(), "cwd": "/project", "mcpServers": [] }
        })
    }

    /// A load that cannot happen must not take any live runtime down with
    /// it: the client still holds those sessions, and every prompt they send
    /// next must still find their runtimes.
    #[test_case(false ; "a session that does not exist")]
    #[test_case(true ; "a session another maki holds")]
    #[tokio::test]
    async fn a_failed_load_leaves_the_running_session_alone(held_elsewhere: bool) {
        let tmp = TempDir::new().unwrap();
        let storage = StateDir::from_path(tmp.path().to_path_buf());
        let mut stored: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new(SELECTED_SPEC, "/project");
        let claim = SessionClaim::acquire(stored.id, &storage).unwrap();
        stored.save(&claim, &storage).unwrap();
        let _elsewhere = held_elsewhere.then_some(claim);
        let target = match held_elsewhere {
            true => stored.id,
            false => MakiId::generate(),
        };
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let running = only(&srv).handle.session_id.clone();

        let result = load_session(&mut srv, &load_request(target), &acp_params(storage)).await;

        assert!(result.is_err());
        assert_eq!(srv.sessions.len(), 1, "{FAILED_LOAD_KEEPS_SESSION}");
        assert_eq!(
            only(&srv).handle.session_id,
            running,
            "{FAILED_LOAD_KEEPS_SESSION}"
        );
    }

    const TEXT_A: &str = "runtime A speaks";
    const TEXT_B: &str = "runtime B speaks";

    /// Drives one runtime's own pump over a fake stream. No prompts are sent,
    /// so nothing here can reach a provider.
    async fn drive_runtime(srv: &Server, key: &str, feed: impl FnOnce(&EventSender)) {
        let runtime = srv
            .sessions
            .get(key)
            .expect("the addressed runtime is live");
        let (guard, events) = maki_agent::event_stream();
        let pump = start_event_pump(
            events,
            runtime.handle.session_id.clone(),
            srv.out_tx.clone(),
            Arc::clone(&runtime.pending),
            PathBuf::from(PUMP_CWD),
            None,
            PUMP_TRUSTED,
            None,
            live_handle(),
            false,
            None,
        );
        let sender = guard.sender(0);
        feed(&sender);
        drop(guard);
        pump.await.unwrap();
    }

    /// 02.2a acceptance, in-process half: two runtimes share one `Server` (one
    /// PID), each with its own pump task and conversation context, and neither
    /// disturbs the other. No prompts are sent and no inference runs: prompt
    /// and event routing proof, with interleaving and settlement, is 02.2b.
    /// The same-PID process half lives in `tests/session_registry.rs`.
    #[tokio::test]
    async fn two_runtimes_coexist_with_independent_pumps_and_contexts() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (key_b, runtime_b) = fixture_runtime();
        srv.sessions.insert(key_b.clone(), runtime_b);

        assert_ne!(key_a, key_b, "two runtimes hold distinct ids");
        assert_eq!(srv.sessions.len(), 2, "one PID owns both runtimes");
        let sid_a = srv.sessions.get(&key_a).unwrap().handle.session_id.clone();
        let sid_b = srv.sessions.get(&key_b).unwrap().handle.session_id.clone();
        assert_ne!(
            sid_a, sid_b,
            "two runtimes hold distinct conversation contexts"
        );
        assert_ne!(
            srv.sessions.get(&key_a).unwrap().claim.id(),
            srv.sessions.get(&key_b).unwrap().claim.id(),
            "two runtimes hold distinct storage claims"
        );

        // Per-session fixed setup: each runtime keeps its own mode and model.
        only_mut_addr(&mut srv, &key_a).current_model = SELECTED_SPEC.to_owned();
        only_mut_addr(&mut srv, &key_b).current_model = "other-vendor/other-model".to_owned();

        drive_runtime(&srv, &key_a, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: TEXT_A.to_owned(),
                })
                .unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;
        drive_runtime(&srv, &key_b, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: TEXT_B.to_owned(),
                })
                .unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;

        let frames: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(
            frames.len(),
            2,
            "each pump forwards only its own stream: {frames:?}"
        );
        for (text, sid) in [(TEXT_A, sid_a.to_string()), (TEXT_B, sid_b.to_string())] {
            assert!(
                frames
                    .iter()
                    .any(|frame| frame["params"]["sessionId"] == sid
                        && frame["params"]["update"]["content"]["text"] == text),
                "the {sid} pump addresses its own text: {frames:?}"
            );
            assert!(
                frames
                    .iter()
                    .filter(|frame| frame["params"]["sessionId"] == sid)
                    .count()
                    == 1,
                "no pump writes for the peer: {frames:?}"
            );
        }

        // Pending slots and epochs are per runtime: settling A leaves B alone.
        // The parked prompt carries its turn, the way `handle_prompt` leaves
        // it; the terminal below settles it and bumps A's epoch by itself.
        {
            let runtime = srv.sessions.get(&key_a).unwrap();
            let mut guard = runtime.pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(PROMPT_ID));
        }
        drive_runtime(&srv, &key_a, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx.try_recv().expect("A settles its own prompt");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(
            answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the answer carries the turn it settled: {answer}"
        );
        assert!(
            srv.sessions
                .get(&key_b)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .prompt
                .is_none(),
            "settling A never touches B's pending slot"
        );

        assert_eq!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
            1,
            "settling A bumps A's epoch"
        );
        assert!(
            srv.sessions
                .get(&key_b)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .check(0)
                .is_ok(),
            "settling A never moves B's epoch"
        );
        assert_eq!(
            srv.sessions.get(&key_a).unwrap().current_model,
            SELECTED_SPEC,
            "per-session setup survives the peer settling"
        );
        assert_eq!(
            srv.sessions.get(&key_b).unwrap().current_model,
            "other-vendor/other-model",
            "per-session setup survives the peer settling"
        );
    }

    fn only_mut_addr<'a>(srv: &'a mut Server, key: &str) -> &'a mut SessionRuntime {
        srv.sessions
            .get_mut(key)
            .expect("the addressed runtime is live")
    }

    fn pending(srv: &Server) -> &PendingState {
        &only(srv).pending
    }

    fn session_key(srv: &Server) -> String {
        only(srv).handle.session_id.to_string()
    }

    fn permission_ask(srv: &Server, answer_tx: Option<Sender<String>>) -> Ask {
        Ask::Permission {
            request_id: PARENT_TOOL_USE_ID.to_owned(),
            answer_tx,
            session: session_key(srv),
            epoch: live_epoch(srv),
        }
    }

    fn elicitation_ask(srv: &Server) -> Ask {
        Ask::Elicitation {
            session: session_key(srv),
            epoch: live_epoch(srv),
        }
    }

    fn server_with_permission_ask(
        answer_tx: Option<Sender<String>>,
    ) -> (Server, flume::Receiver<String>, flume::Receiver<Value>) {
        let (server, answer_rx, out_rx) = test_server();
        let ask = permission_ask(&server, answer_tx);
        pending(&server)
            .lock()
            .unwrap()
            .asks
            .insert(ANSWERED_ID, ask);
        (server, answer_rx, out_rx)
    }

    fn server_with_elicitation() -> (Server, flume::Receiver<String>, flume::Receiver<Value>) {
        let (server, answer_rx, out_rx) = test_server();
        let ask = elicitation_ask(&server);
        pending(&server)
            .lock()
            .unwrap()
            .asks
            .insert(ANSWERED_ID, ask);
        (server, answer_rx, out_rx)
    }

    const PUMP_CWD: &str = "/project";
    const QUEUED_TEXT: &str = "queued before close";
    const NOTICE_LINE: &str = "Background agent finished: find auth";
    const CHILD_NOTICE_LINE: &str = "a child's own notice";
    const HOST_NOTICE: &str = "the prompt changed, so the cache starts over";
    const PROMPT_ID: i64 = 7;
    const SUBAGENT_COST: f64 = 0.25;
    const TURN_COST: f64 = 0.5;
    const LEDGER_COST: f64 = 0.75;
    const CONTEXT_WINDOW: u32 = 200_000;
    const PARENT_TOOL_USE_ID: &str = "toolu_1";
    /// A string id no ask can ever carry, since ours are minted as numbers.
    const UNANSWERABLE_ID: &str = "not-an-id";
    const SUBAGENT_NAME: &str = "task";
    const CHILD_TOOL_USE_IDS: [&str; 2] = ["child_write_1", "child_write_2"];
    const PERMISSION_TOOL: &str = "write";
    const MAIN_PERMISSION_TITLE: &str = "write: /project";
    const CHILD_PERMISSION_TITLE: &str = "task: write: /project";
    const ESCALATION_REASON: &str = "plugin flagged this write";
    const ESCALATED_PERMISSION_TITLE: &str = "plugin flagged this write (write: /project)";
    const TURN_ERROR: &str = "provider returned 500";
    /// One tool id arriving from two runs of the same session. `openai_compat`
    /// mints unique ids now, but a provider can still repeat one, and the pump
    /// has to keep the runs apart either way.
    const COLLIDING_TOOL_USE_ID: &str = "maki_unnamed_0";
    /// These cover cost and transcript plumbing, not the trust-scoped wording
    /// of permission options.
    const PUMP_TRUSTED: bool = true;
    async fn run_pump(srv: &Server, initial_cost: Option<f64>, feed: impl FnOnce(&EventSender)) {
        let session = only(srv);
        let (guard, events) = maki_agent::event_stream();
        let pump = start_event_pump(
            events,
            session.handle.session_id.clone(),
            srv.out_tx.clone(),
            Arc::clone(&session.pending),
            PathBuf::from(PUMP_CWD),
            None,
            PUMP_TRUSTED,
            initial_cost,
            live_handle(),
            false,
            None,
        );
        let sender = guard.sender(0);
        feed(&sender);
        drop(guard);
        pump.await.unwrap();
    }

    /// 02.10a supervised pump: the throwaway pump reports stream end as `lost`
    /// and consults `is_live` on `Error`, like production. Fixtures inject
    /// liveness directly (dead handle vs. live handle); no provider, keys, or
    /// network are involved.
    async fn run_supervised_pump(
        srv: &Server,
        key: &str,
        is_live: Liveness,
        feed: impl FnOnce(&EventSender),
    ) {
        let runtime = srv
            .sessions
            .get(key)
            .expect("the addressed runtime is live");
        let (guard, events) = maki_agent::event_stream();
        let pump = start_event_pump(
            events,
            runtime.handle.session_id.clone(),
            srv.out_tx.clone(),
            Arc::clone(&runtime.pending),
            PathBuf::from(PUMP_CWD),
            None,
            PUMP_TRUSTED,
            None,
            is_live,
            true,
            None,
        );
        let sender = guard.sender(0);
        feed(&sender);
        drop(guard);
        pump.await.unwrap();
    }

    fn dead_handle() -> Liveness {
        Arc::new(|| false)
    }

    fn live_handle() -> Liveness {
        Arc::new(|| true)
    }

    fn turn_complete(cost: f64) -> Box<TurnCompleteEvent> {
        Box::new(TurnCompleteEvent {
            message: Message::user(String::new()),
            usage: TokenUsage::default(),
            model: SELECTED_SPEC.to_owned(),
            cost: Some(cost),
            subsidised_list_cost: None,
            context_size: None,
            context_window: CONTEXT_WINDOW,
        })
    }

    fn subagent(answer_tx: Option<Sender<String>>) -> SubagentInfo {
        SubagentInfo {
            parent_tool_use_id: PARENT_TOOL_USE_ID.to_owned(),
            name: SUBAGENT_NAME.to_owned(),
            prompt: None,
            model: None,
            opts: None,
            answer_tx,
            inbox: None,
            parent_call: None,
        }
    }

    fn done_event() -> AgentEvent {
        AgentEvent::Done {
            usage: TokenUsage::default(),
            cost: None,
            list_cost: None,
            context_size: 0,
            context_window: CONTEXT_WINDOW,
            num_turns: 1,
            reason: DoneReason::EndTurn,
            raw_stop_reason: None,
        }
    }

    /// What the pump sees when a cancelled run actually stops: `run.rs`
    /// maps `Err(Cancelled)` to `Ok(DoneReason::Cancelled)` and emits it, so
    /// a stopped run always reports `Done`, never `Error`. Only real failures
    /// surface as `AgentEvent::Error`.
    fn cancelled_done_event() -> AgentEvent {
        AgentEvent::Done {
            usage: TokenUsage::default(),
            cost: None,
            list_cost: None,
            context_size: 0,
            context_window: CONTEXT_WINDOW,
            num_turns: 1,
            reason: DoneReason::Cancelled,
            raw_stop_reason: Some("cancelled".to_owned()),
        }
    }

    fn write_tool_start(tool_use_id: &str) -> AgentEvent {
        AgentEvent::ToolStart(Box::new(ToolStartEvent {
            id: tool_use_id.to_owned(),
            tool: Arc::from(PERMISSION_TOOL),
            summary: String::new(),
            render_header: None,
            annotation: None,
            input: None,
            raw_input: Some(serde_json::json!({"path": PUMP_CWD, "content": QUEUED_TEXT})),
            output: None,
        }))
    }

    fn permission_request(tool_use_id: &str) -> AgentEvent {
        AgentEvent::PermissionRequest {
            id: tool_use_id.to_owned(),
            tool: ToolKey::native(PERMISSION_TOOL),
            scopes: vec![PUMP_CWD.to_owned()],
            reason: None,
        }
    }

    #[test_case(false ; "answers_reach_the_agent_that_asked_even_out_of_order")]
    #[test_case(true ; "cancel_keeps_the_asks_a_moved_call_may_wait_on")]
    #[tokio::test]
    async fn concurrent_subagent_permissions(cancel: bool) {
        let (srv, main_rx, out_rx) = test_server();
        let children = [flume::unbounded(), flume::unbounded()];
        run_pump(&srv, None, |sender| {
            for ((answer_tx, _), child_id) in children.iter().zip(CHILD_TOOL_USE_IDS) {
                let info = subagent(Some(answer_tx.clone()));
                for event in [
                    AgentEvent::TextDelta {
                        text: QUEUED_TEXT.to_owned(),
                    },
                    permission_request(child_id),
                ] {
                    sender
                        .send_envelope(Envelope {
                            event,
                            subagent: Some(info.clone()),
                            run_id: 0,
                        })
                        .unwrap();
                }
            }
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
        })
        .await;

        let requests: Vec<_> = out_rx.try_iter().collect();
        assert_eq!(
            requests.len(),
            3,
            "of the subagent events only permissions reach the client"
        );
        for (request, title) in requests.iter().zip([
            CHILD_PERMISSION_TITLE,
            CHILD_PERMISSION_TITLE,
            MAIN_PERMISSION_TITLE,
        ]) {
            assert_eq!(request["method"], "session/request_permission");
            let call = &request["params"]["toolCall"];
            assert_eq!(
                call["toolCallId"], PARENT_TOOL_USE_ID,
                "a child permission rides on the task call the client knows"
            );
            assert_eq!(call["title"], title);
        }

        if cancel {
            let raw = serde_json::json!({
                "params": { "sessionId": only(&srv).handle.session_id.to_string() }
            });
            handle_notification(&srv, "session/cancel", &raw);
        }
        for (request, option) in requests
            .iter()
            .zip(["reject_once", "allow_once", "allow_always"])
            .rev()
        {
            handle_incoming_response(
                &srv,
                &serde_json::json!({
                    "id": request["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": option}},
                }),
            );
        }
        let child_rx = children.map(|(_, rx)| rx);
        for ((rx, expected), asked_id) in child_rx
            .iter()
            .chain([&main_rx])
            .zip([
                PermissionAnswer::Deny,
                PermissionAnswer::AllowOnce,
                PermissionAnswer::AllowSession,
            ])
            .zip(CHILD_TOOL_USE_IDS.iter().chain([&PARENT_TOOL_USE_ID]))
        {
            let expected = TaggedAnswer::new(*asked_id, expected).encode();
            assert_eq!(rx.try_recv().ok(), Some(expected));
            assert!(rx.is_empty(), "each answer is delivered only once");
        }
    }

    /// Without a channel of its own a child's answer would land in the main
    /// agent's, and the main agent's next wait would eat it as its own.
    #[tokio::test]
    async fn a_subagent_permission_without_an_answer_channel_is_never_asked() {
        let (srv, main_rx, out_rx) = test_server();
        run_pump(&srv, None, |sender| {
            sender
                .send_envelope(Envelope {
                    event: permission_request(CHILD_TOOL_USE_IDS[0]),
                    subagent: Some(subagent(None)),
                    run_id: 0,
                })
                .unwrap();
        })
        .await;

        assert!(out_rx.is_empty(), "the client is never asked");
        assert!(main_rx.is_empty(), "the main agent keeps its own turn");
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "nothing is left waiting for an answer"
        );
    }

    /// Runs one permission wait the way a turn does, on the session's own
    /// answer channel. Whatever is queued when this is called is what the
    /// waiter finds, which is the shape of the bug: answers outlive the turn
    /// that asked for them.
    async fn wait_for_permission(
        srv: &Server,
        answer_rx: &flume::Receiver<String>,
        request_id: &str,
    ) -> Result<(), PermissionError> {
        let session = only(srv);
        let (guard, _events) = maki_agent::event_stream();
        let event_tx = guard.sender(0);
        let rx = tokio::sync::Mutex::new(answer_rx.clone());
        session
            .handle
            .permissions
            .enforce(
                &ToolKey::native(NEXT_TURN_TOOL),
                &PermissionScopes::single(NEXT_TURN_SCOPE.to_owned()),
                None,
                &event_tx,
                Some(&rx),
                request_id,
                &CancelToken::none(),
                None,
                None,
                None,
            )
            .await
            // No layers here, so nothing can have answered in a plugin's name.
            .map(|_| ())
    }

    fn queue_answer(srv: &Server, request_id: &str, answer: PermissionAnswer) {
        let session = only(srv);
        session
            .handle
            .answer_tx
            .send(TaggedAnswer::new(request_id, answer).encode())
            .unwrap();
    }

    fn next_turn_is_allowed(srv: &Server) -> bool {
        let session = only(srv);
        matches!(
            session.handle.permissions.check(
                &ToolKey::native(NEXT_TURN_TOOL),
                NEXT_TURN_SCOPE,
                None
            ),
            PermissionCheck::Allowed
        )
    }

    /// The race the ids exist for: the cancel lands while the permission
    /// envelope is still queued in the pump, so the pump still asks and the
    /// client answers it for real. With nobody parked, that answer waits in
    /// the channel for the next turn.
    #[tokio::test]
    async fn an_answer_for_a_cancelled_ask_is_not_consumed_by_the_next_turn() {
        let (srv, answer_rx, out_rx) = test_server();
        let cancel = serde_json::json!({
            "params": { "sessionId": only(&srv).handle.session_id.to_string() }
        });
        handle_notification(&srv, "session/cancel", &cancel);
        run_pump(&srv, None, |sender| {
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
        })
        .await;

        let request = out_rx.try_recv().expect("the pump still asked the client");
        handle_incoming_response(
            &srv,
            &serde_json::json!({
                "id": request["id"],
                "result": {"outcome": {"outcome": "selected", "optionId": "allow_always"}},
            }),
        );

        queue_answer(&srv, NEXT_TURN_TOOL_USE_ID, PermissionAnswer::Deny);
        let outcome = wait_for_permission(&srv, &answer_rx, NEXT_TURN_TOOL_USE_ID).await;

        assert!(outcome.is_err(), "{STALE_ALLOW}");
        assert!(!next_turn_is_allowed(&srv), "{STALE_ALLOW}");
        assert!(
            answer_rx.is_empty(),
            "the stale answer was dropped, not left"
        );
    }

    /// A mismatched answer is dropped without ending the wait: the tool is
    /// still blocked on the ask the user was shown, and only that ask's answer
    /// may end it.
    #[tokio::test]
    async fn an_answer_naming_another_ask_is_discarded_and_the_wait_continues() {
        let (srv, answer_rx, ..) = test_server();
        queue_answer(&srv, PARENT_TOOL_USE_ID, PermissionAnswer::Deny);
        queue_answer(&srv, NEXT_TURN_TOOL_USE_ID, PermissionAnswer::AllowOnce);

        let outcome = wait_for_permission(&srv, &answer_rx, NEXT_TURN_TOOL_USE_ID).await;

        assert!(
            outcome.is_ok(),
            "the deny named another ask and must not have ended this wait"
        );
        assert!(
            answer_rx.is_empty(),
            "both answers were taken off the queue"
        );
    }

    #[test_case(PermissionAnswer::AllowSession, true ; "allow_applies_the_decision")]
    #[test_case(PermissionAnswer::Deny, false ; "deny_applies_the_decision")]
    #[tokio::test]
    async fn a_matching_answer_is_applied(answer: PermissionAnswer, allowed: bool) {
        let (srv, answer_rx, ..) = test_server();
        queue_answer(&srv, NEXT_TURN_TOOL_USE_ID, answer);

        let outcome = wait_for_permission(&srv, &answer_rx, NEXT_TURN_TOOL_USE_ID).await;

        assert_eq!(outcome.is_ok(), allowed);
        assert_eq!(next_turn_is_allowed(&srv), allowed);
    }

    /// The close marker rides the same FIFO as the events, so a turn still
    /// streaming when the session got replaced is reported in full and the
    /// client's outstanding `session/prompt` is answered instead of hanging.
    #[tokio::test]
    async fn event_pump_delivers_everything_queued_before_the_close() {
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_pump(&srv, None, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: QUEUED_TEXT.to_owned(),
                })
                .unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;

        let chunk = out_rx.try_recv().expect("the queued text reaches the wire");
        let update = &chunk["params"]["update"];
        assert_eq!(update["sessionUpdate"], "agent_message_chunk");
        assert_eq!(update["content"]["text"], QUEUED_TEXT);

        let answer = out_rx.try_recv().expect("the pending prompt is answered");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["result"]["stopReason"], "end_turn");
        assert!(pending(&srv).lock().unwrap().prompt.is_none());
    }

    /// A provider refusal settles the prompt once through the turn epoch with
    /// `refusal`, carrying the verbatim wire code and task usability in
    /// additive `_meta`; a duplicate terminal for the same epoch is dropped,
    /// never re-answered, and moves no epoch.
    #[tokio::test]
    async fn event_pump_settles_a_refusal_once_with_raw_reason_and_usability() {
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        let refusal = || AgentEvent::Done {
            usage: TokenUsage::default(),
            cost: None,
            list_cost: None,
            context_size: 0,
            context_window: CONTEXT_WINDOW,
            num_turns: 1,
            reason: DoneReason::Refusal,
            raw_stop_reason: Some("content_filter".to_owned()),
        };
        run_pump(&srv, None, |sender| {
            sender.send(refusal()).unwrap();
            sender.send(refusal()).unwrap();
        })
        .await;

        let answer = out_rx.try_recv().expect("the pending prompt is answered");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["result"]["stopReason"], "refusal");
        assert_eq!(
            answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the answer carries the turn it settled: {answer}"
        );
        assert_eq!(
            answer["result"]["_meta"][extension::RAW_REASON_META_KEY],
            "content_filter",
            "the verbatim wire code survives: {answer}"
        );
        assert_eq!(
            answer["result"]["_meta"][extension::USABLE_META_KEY],
            false,
            "a refusal banks no usable output: {answer}"
        );
        assert!(pending(&srv).lock().unwrap().prompt.is_none());
        assert_eq!(live_epoch(&srv), 1, "one settlement bumps once");
        assert!(
            out_rx.try_recv().is_err(),
            "a duplicate terminal for one epoch settles nothing"
        );
    }

    /// An accepted observation's line arrives as a finished card, in the order
    /// the agent sent it, and the turn's permission and answer routing stay as
    /// they were. A host notice is not sent, and a child's line stays in the
    /// child, like the rest of its stream.
    #[tokio::test]
    async fn an_accepted_notice_reaches_the_client_in_stream_order() {
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_pump(&srv, None, |sender| {
            sender
                .send(AgentEvent::ObservationLine {
                    text: NOTICE_LINE.to_owned(),
                    detail: None,
                    end: None,
                })
                .unwrap();
            sender
                .send(AgentEvent::Notice {
                    text: HOST_NOTICE.to_owned(),
                })
                .unwrap();
            sender
                .send_envelope(Envelope {
                    event: AgentEvent::ObservationLine {
                        text: CHILD_NOTICE_LINE.to_owned(),
                        detail: None,
                        end: None,
                    },
                    subagent: Some(subagent(None)),
                    run_id: 0,
                })
                .unwrap();
            sender
                .send(AgentEvent::TextDelta {
                    text: QUEUED_TEXT.to_owned(),
                })
                .unwrap();
            sender.send(write_tool_start(PARENT_TOOL_USE_ID)).unwrap();
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;

        let wire: Vec<Value> = out_rx.try_iter().collect();
        let text = serde_json::to_value(translate::text_delta(QUEUED_TEXT)).unwrap();
        assert_eq!(wire.len(), 5, "{wire:?}");
        let card = &wire[0]["params"]["update"];
        assert_eq!(card["sessionUpdate"], "tool_call");
        assert_eq!(card["title"], NOTICE_LINE);
        assert_eq!(card["status"], "completed");
        assert_eq!(wire[1]["params"]["update"], text);
        assert_eq!(
            wire[2]["params"]["update"]["sessionUpdate"],
            "tool_call_update"
        );
        assert_eq!(wire[3]["method"], "session/request_permission");
        assert_eq!(
            wire[3]["params"]["toolCall"]["title"],
            MAIN_PERMISSION_TITLE
        );
        assert_eq!(wire[4]["id"], PROMPT_ID);
        assert_eq!(wire[4]["result"]["stopReason"], "end_turn");
    }

    /// Clients only prompt a login on `auth_required`, so a rejected credential
    /// must not come back as a generic failure.
    #[test_case(true, AcpError::auth_required() ; "auth_error_asks_for_a_login")]
    #[test_case(false, AcpError::internal_error() ; "other_error_is_internal")]
    #[tokio::test]
    async fn a_failed_turn_answers_the_prompt_with_its_error_kind(auth: bool, expected: AcpError) {
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_pump(&srv, None, |sender| {
            sender
                .send(AgentEvent::Error {
                    message: TURN_ERROR.to_owned(),
                    auth,
                    totals: RunTotals::default(),
                })
                .unwrap();
        })
        .await;

        let answer = out_rx.try_recv().expect("the pending prompt is answered");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["error"]["code"], i32::from(expected.code));
        assert_eq!(answer["error"]["data"], TURN_ERROR);
    }

    /// A cached input holds the whole file a `write` is about to lay down, and
    /// this pump lives as long as the session does, so a turn that died on a
    /// provider error has to let go of it just like a turn that finished.
    #[test_case(None, true ; "a_live_turn_shows_the_file_its_tool_call_named")]
    #[test_case(Some(done_event()), false ; "a_finished_turn_releases_its_tool_inputs")]
    #[test_case(Some(AgentEvent::Error { message: TURN_ERROR.to_owned(), auth: false, totals: RunTotals::default() }), false ; "a_failed_turn_releases_its_tool_inputs")]
    #[tokio::test]
    async fn a_terminal_event_releases_the_turns_tool_inputs(
        terminal: Option<AgentEvent>,
        keeps_input: bool,
    ) {
        let (srv, .., out_rx) = test_server();
        run_pump(&srv, None, |sender| {
            sender.send(write_tool_start(PARENT_TOOL_USE_ID)).unwrap();
            if let Some(event) = terminal {
                sender.send(event).unwrap();
            }
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
        })
        .await;

        let request = out_rx
            .try_iter()
            .last()
            .expect("the permission request reaches the client");
        let call = &request["params"]["toolCall"];
        assert_eq!(request["method"], "session/request_permission");
        assert_eq!(call["title"], MAIN_PERMISSION_TITLE);
        assert_eq!(!call["rawInput"].is_null(), keeps_input, "{request}");
    }

    /// Ids are only unique inside one run, so a child can ask about a call whose
    /// id the parent already cached. The dialog has to stay title only, or it
    /// would name a file the subagent is not touching.
    #[tokio::test]
    async fn a_subagent_permission_ignores_a_colliding_cached_input() {
        let (srv, .., out_rx) = test_server();
        let (answer_tx, _answer_rx) = flume::unbounded();
        run_pump(&srv, None, |sender| {
            sender
                .send(write_tool_start(COLLIDING_TOOL_USE_ID))
                .unwrap();
            sender
                .send_envelope(Envelope {
                    event: permission_request(COLLIDING_TOOL_USE_ID),
                    subagent: Some(subagent(Some(answer_tx))),
                    run_id: 0,
                })
                .unwrap();
        })
        .await;

        let request = out_rx
            .try_iter()
            .last()
            .expect("the permission request reaches the client");
        let call = &request["params"]["toolCall"];
        assert_eq!(request["method"], "session/request_permission");
        assert_eq!(call["toolCallId"], PARENT_TOOL_USE_ID);
        assert_eq!(call["title"], CHILD_PERMISSION_TITLE);
        for field in ["kind", "locations", "rawInput", "content"] {
            assert!(call[field].is_null(), "{field} must stay unset: {request}");
        }
    }

    #[test_case(Some(ESCALATION_REASON), ESCALATED_PERMISSION_TITLE ; "a_plugin_reason_leads_the_title")]
    #[test_case(None, MAIN_PERMISSION_TITLE ; "no_reason_keeps_the_plain_title")]
    #[tokio::test]
    async fn permission_title_carries_the_escalation_reason(reason: Option<&str>, title: &str) {
        let (srv, .., out_rx) = test_server();
        run_pump(&srv, None, |sender| {
            sender
                .send(AgentEvent::PermissionRequest {
                    id: PARENT_TOOL_USE_ID.to_owned(),
                    tool: ToolKey::native(PERMISSION_TOOL),
                    scopes: vec![PUMP_CWD.to_owned()],
                    reason: reason.map(str::to_owned),
                })
                .unwrap();
        })
        .await;

        let request = out_rx
            .try_recv()
            .expect("the permission request reaches the client");
        assert_eq!(request["method"], "session/request_permission");
        assert_eq!(request["params"]["toolCall"]["title"], title);
    }

    /// A resumed session opens with a bill, and subagent turns spend against it
    /// even though their events never enter the transcript.
    #[tokio::test]
    async fn event_pump_folds_restored_and_subagent_cost_into_the_usage_update() {
        let (srv, .., out_rx) = test_server();
        run_pump(&srv, Some(RECORDED_COST), |sender| {
            sender
                .send_envelope(Envelope {
                    event: AgentEvent::TurnComplete(turn_complete(SUBAGENT_COST)),
                    subagent: Some(subagent(None)),
                    run_id: 0,
                })
                .unwrap();
            sender
                .send(AgentEvent::TurnComplete(turn_complete(TURN_COST)))
                .unwrap();
        })
        .await;

        let usage = out_rx.try_recv().expect("the session's own turn reports");
        let update = &usage["params"]["update"];
        assert_eq!(update["sessionUpdate"], "usage_update");
        assert_eq!(
            update["cost"]["amount"].as_f64(),
            Some(RECORDED_COST + SUBAGENT_COST + TURN_COST)
        );
        assert!(
            out_rx.is_empty(),
            "the subagent turn pays but stays out of the transcript"
        );
    }

    /// A run's `Done` or `Error` carries its ledger, which also bills
    /// compaction and moved work, so it replaces what the run's turns added
    /// up to.
    #[test_case(false ; "a_finished_run")]
    #[test_case(true ; "a_failed_run")]
    #[tokio::test]
    async fn event_pump_settles_a_run_on_its_ledger(failed: bool) {
        let (srv, .., out_rx) = test_server();
        let ledger = RunTotals {
            cost: Some(LEDGER_COST),
            ..RunTotals::default()
        };
        let ending = if failed {
            AgentEvent::Error {
                message: TURN_ERROR.to_owned(),
                auth: false,
                totals: ledger,
            }
        } else {
            AgentEvent::Done {
                usage: ledger.usage,
                cost: ledger.cost,
                list_cost: None,
                context_size: 0,
                context_window: CONTEXT_WINDOW,
                num_turns: 1,
                reason: DoneReason::EndTurn,
                raw_stop_reason: None,
            }
        };
        run_pump(&srv, Some(RECORDED_COST), |sender| {
            sender
                .send(AgentEvent::TurnComplete(turn_complete(TURN_COST)))
                .unwrap();
            sender.send(ending).unwrap();
            sender
                .send(AgentEvent::TurnComplete(turn_complete(TURN_COST)))
                .unwrap();
        })
        .await;

        let last = out_rx.try_iter().last().expect("the next turn reports");
        assert_eq!(
            last["params"]["update"]["cost"]["amount"].as_f64(),
            Some(RECORDED_COST + LEDGER_COST + TURN_COST)
        );
    }

    /// Results ride the next prompt and nothing wakes a turn, so a session
    /// closed with work out stops it rather than waiting on it. The task that
    /// never ends stands in for that work.
    #[tokio::test]
    async fn close_session_returns_at_once_with_background_work_out() {
        let (mut srv, ..) = test_server();
        only_mut(&mut srv).handle.task = AbortOnDropHandle::new(tokio::spawn(future::pending()));

        let closed = timeout(
            STDIN_DEADLOCK_TIMEOUT,
            close_all_sessions(&mut srv, SessionEndReason::Shutdown),
        )
        .await
        .is_ok();

        assert!(closed, "{CLOSE_WAITED}");
    }

    #[test]
    fn blocking_mode_disables_detached_tools() {
        assert!(
            matches!(background_mode(false), BackgroundMode::Blocking),
            "{BLOCKING_MODE}"
        );
        assert!(
            matches!(background_mode(true), BackgroundMode::Detached),
            "a nonzero timeout keeps the detached mode the harness forbids"
        );
    }

    #[tokio::test]
    async fn close_session_awaits_the_session_end_hook() {
        let (mut srv, ..) = test_server();
        let ended = only(&srv).handle.session_id.id();
        let (ended_tx, ended_rx) = flume::bounded(1);
        srv.on_session_end = Some(Arc::new(move |id, reason| {
            let ended_tx = ended_tx.clone();
            Box::pin(async move {
                let _ = ended_tx.send((id, reason));
            })
        }));

        close_all_sessions(&mut srv, SessionEndReason::Replaced).await;

        assert_eq!(
            ended_rx.try_recv().ok(),
            Some((ended, SessionEndReason::Replaced))
        );
        assert!(srv.sessions.is_empty(), "close must take the session");
    }

    /// JSON-RPC ids are `string | number`, and a client is free to echo ours
    /// back in either shape. An answer dropped for its shape leaves the tool
    /// parked forever.
    #[test_case(Value::from(ANSWERED_ID), true ; "a_numeric_id_answers_the_ask")]
    #[test_case(Value::from(ANSWERED_ID.to_string()), true ; "a_string_id_answers_the_same_ask")]
    #[test_case(Value::from(UNANSWERABLE_ID), false ; "an_id_matching_no_ask_answers_nothing")]
    #[tokio::test]
    async fn a_response_is_delivered_whatever_shape_its_id_has(id: Value, delivered: bool) {
        let (srv, answer_rx, ..) = server_with_permission_ask(None);

        handle_incoming_response(
            &srv,
            &serde_json::json!({
                "id": id,
                "result": { "outcome": { "outcome": "selected", "optionId": "allow_once" } },
            }),
        );

        let answered = TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode();
        assert_eq!(answer_rx.try_recv().ok(), delivered.then_some(answered));
        assert_eq!(
            pending(&srv)
                .lock()
                .unwrap()
                .asks
                .contains_key(&ANSWERED_ID),
            !delivered,
            "only the ask that was answered is retired"
        );
    }

    #[tokio::test]
    async fn only_the_outstanding_request_id_is_answered() {
        let (srv, answer_rx, ..) = server_with_permission_ask(None);

        handle_incoming_response(&srv, &allow_once(UNKNOWN_ID));
        assert!(answer_rx.is_empty(), "an unknown id is dropped");

        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert_eq!(
            answer_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode())
        );

        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(
            answer_rx.is_empty(),
            "a replayed answer cannot land on the next request"
        );
    }

    #[tokio::test]
    async fn elicitation_response_forwards_the_raw_result() {
        let (srv, answer_rx, ..) = server_with_elicitation();
        let raw = serde_json::json!({
            "id": ANSWERED_ID,
            "result": { "action": "accept", "content": { "q1": "axum" } },
        });

        handle_incoming_response(&srv, &raw);
        let forwarded = answer_rx.try_recv().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&forwarded).unwrap(),
            raw["result"]
        );
    }

    /// 02.8 concurrent permission requests and answer routing. Asks keep
    /// their request/session/turn identities under globally unique ids, so
    /// answers route to the task that asked, never the currently active slot;
    /// unknown, repeated, late, stale-turn and cross-session answers reject;
    /// and a blocking terminal retires every outstanding ask. No prompt here
    /// reaches a provider: interleaving is driven with synthetic agent events
    /// plus the real pump, and no keys or network are involved.
    const SECOND_ASK_ID: i64 = 1010;
    const THIRD_ASK_ID: i64 = 1011;

    fn done_with(reason: DoneReason) -> AgentEvent {
        AgentEvent::Done {
            usage: TokenUsage::default(),
            cost: None,
            list_cost: None,
            context_size: 0,
            context_window: CONTEXT_WINDOW,
            num_turns: 1,
            reason,
            raw_stop_reason: None,
        }
    }

    /// Every agent-settled terminal (the 02.7 sources) settles the prompt and
    /// retires every outstanding ask in blocking mode: permission and
    /// elicitation alike. A late answer afterwards finds an unknown id, never
    /// a live ask to revive.
    #[test_case(DoneReason::EndTurn ; "end_turn")]
    #[test_case(DoneReason::MaxTokens ; "max_tokens")]
    #[test_case(DoneReason::MaxTurns ; "max_turns")]
    #[test_case(DoneReason::Cancelled ; "agent_decided_cancel")]
    #[test_case(DoneReason::Refusal ; "refusal")]
    #[test_case(DoneReason::Dropped ; "dropped")]
    #[test_case(DoneReason::Compact ; "compact")]
    #[tokio::test]
    async fn terminal_settlement_retires_pending_asks(reason: DoneReason) {
        let (srv, answer_rx, out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        // Built before the insert takes the guard: the mutex is not
        // reentrant, so reading the epoch under a held guard would park here.
        let main_ask = permission_ask(&srv, None);
        let form_ask = elicitation_ask(&srv);
        {
            let mut guard = pending(&srv).lock().unwrap();
            guard.asks.insert(ANSWERED_ID, main_ask);
            guard.asks.insert(SECOND_ASK_ID, form_ask);
        }
        run_pump(&srv, None, |sender| {
            sender.send(done_with(reason)).unwrap();
        })
        .await;

        let answer = out_rx.try_recv().expect("the pending prompt is answered");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(live_epoch(&srv), 1, "settling bumps exactly once");
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "no ask survives a blocking terminal: {reason:?}"
        );
        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(
            answer_rx.is_empty(),
            "a late answer cannot revive a retired ask"
        );
    }

    /// A failed turn settles through the same path, so its asks retire the
    /// same way.
    #[tokio::test]
    async fn error_settlement_retires_pending_asks() {
        let (srv, answer_rx, out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        // Built before the insert takes the guard: the mutex is not
        // reentrant, so reading the epoch under a held guard would park here.
        let ask = permission_ask(&srv, None);
        pending(&srv).lock().unwrap().asks.insert(ANSWERED_ID, ask);
        run_pump(&srv, None, |sender| {
            sender
                .send(AgentEvent::Error {
                    message: TURN_ERROR.to_owned(),
                    auth: false,
                    totals: RunTotals::default(),
                })
                .unwrap();
        })
        .await;

        let answer = out_rx.try_recv().expect("the pending prompt is answered");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["error"]["data"], TURN_ERROR);
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "no ask survives a failed turn"
        );
        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(
            answer_rx.is_empty(),
            "a late answer cannot revive a retired ask"
        );
    }

    /// A repeated answer finds its ask consumed; after the terminal retires
    /// the rest and the next turn starts, late answers still reject while the
    /// next turn's own ask answers normally.
    #[tokio::test]
    async fn repeated_and_late_answers_reject_across_turns() {
        let (srv, answer_rx, ..) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        // Built before the insert takes the guard: the mutex is not
        // reentrant, so reading the epoch under a held guard would park here.
        let first = permission_ask(&srv, None);
        let second = permission_ask(&srv, None);
        {
            let mut guard = pending(&srv).lock().unwrap();
            guard.asks.insert(ANSWERED_ID, first);
            guard.asks.insert(SECOND_ASK_ID, second);
        }

        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert_eq!(
            answer_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode())
        );
        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(
            answer_rx.is_empty(),
            "a replayed answer cannot land on the next request"
        );

        assert!(
            matches!(
                settle_prompt(pending(&srv)),
                Some((RequestId::Number(PROMPT_ID), 0))
            ),
            "the terminal settles the turn it ran under"
        );
        assert_eq!(live_epoch(&srv), 1);
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "settlement retires the still-outstanding ask"
        );
        handle_incoming_response(&srv, &allow_once(SECOND_ASK_ID));
        assert!(
            answer_rx.is_empty(),
            "a late answer for a retired ask is dropped"
        );

        park_prompt(&srv, RequestId::Number(8));
        // Built before the insert takes the guard (see above).
        let next = permission_ask(&srv, None);
        pending(&srv)
            .lock()
            .unwrap()
            .asks
            .insert(THIRD_ASK_ID, next);
        handle_incoming_response(&srv, &allow_once(THIRD_ASK_ID));
        assert_eq!(
            answer_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode()),
            "retirement never poisons the next turn"
        );
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "every ask is consumed exactly once"
        );
    }

    /// Several outstanding asks keep their request/session/turn identities,
    /// and answers arriving newest-first each reach their own task with their
    /// own decision: nothing lands on the currently active slot instead.
    #[tokio::test]
    async fn out_of_order_answers_reach_their_own_tasks() {
        let (srv, main_rx, ..) = test_server();
        let (child0_tx, child0_rx) = flume::unbounded();
        let (child1_tx, child1_rx) = flume::unbounded();
        let session = session_key(&srv);
        let epoch = live_epoch(&srv);
        {
            let mut guard = pending(&srv).lock().unwrap();
            guard.asks.insert(
                SECOND_ASK_ID,
                Ask::Permission {
                    request_id: CHILD_TOOL_USE_IDS[0].to_owned(),
                    answer_tx: Some(child0_tx),
                    session: session.clone(),
                    epoch,
                },
            );
            guard.asks.insert(
                THIRD_ASK_ID,
                Ask::Permission {
                    request_id: CHILD_TOOL_USE_IDS[1].to_owned(),
                    answer_tx: Some(child1_tx),
                    session: session.clone(),
                    epoch,
                },
            );
            guard.asks.insert(
                ANSWERED_ID,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session,
                    epoch,
                },
            );
        }

        for (id, option) in [
            (ANSWERED_ID, "allow_once"),
            (THIRD_ASK_ID, "allow_always"),
            (SECOND_ASK_ID, "reject_once"),
        ] {
            handle_incoming_response(
                &srv,
                &serde_json::json!({
                    "id": id,
                    "result": {"outcome": {"outcome": "selected", "optionId": option}},
                }),
            );
        }

        assert_eq!(
            main_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode())
        );
        assert!(main_rx.is_empty(), "each answer is delivered only once");
        assert_eq!(
            child1_rx.try_recv().ok(),
            Some(TaggedAnswer::new(CHILD_TOOL_USE_IDS[1], PermissionAnswer::AllowSession).encode())
        );
        assert!(child1_rx.is_empty(), "each answer is delivered only once");
        assert_eq!(
            child0_rx.try_recv().ok(),
            Some(TaggedAnswer::new(CHILD_TOOL_USE_IDS[0], PermissionAnswer::Deny).encode())
        );
        assert!(child0_rx.is_empty(), "each answer is delivered only once");
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "every ask is consumed exactly once"
        );
    }

    /// Detached sessions keep their asks across settlement: a moved call or a
    /// still-running child can be parked on one past the turn boundary.
    #[tokio::test]
    async fn detached_sessions_keep_asks_across_settlement() {
        let (srv, ..) = test_server();
        pending(&srv).lock().unwrap().blocking = false;
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        // Built before the insert takes the guard: the mutex is not
        // reentrant, so reading the epoch under a held guard would park here.
        let ask = permission_ask(&srv, None);
        pending(&srv).lock().unwrap().asks.insert(ANSWERED_ID, ask);

        assert!(
            matches!(
                settle_prompt(pending(&srv)),
                Some((RequestId::Number(PROMPT_ID), 0))
            ),
            "the terminal still settles the prompt"
        );
        assert_eq!(live_epoch(&srv), 1);
        assert_eq!(
            pending(&srv).lock().unwrap().asks.len(),
            1,
            "detached keeps asks for calls that outlive the turn"
        );
    }

    /// 02.8 synchronised fixture: one PID owns a blocked session A and an
    /// advancing session B. Each session's tool asks through its own pump, so
    /// the frames name their session and carry globally unique ids from one
    /// process-wide counter. B runs a full turn and settles without touching
    /// A's parked prompt, ask, or epoch; then A's answer still routes to A,
    /// and B's retired ask rejects.
    #[tokio::test]
    async fn blocked_session_stays_put_while_its_peer_advances() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (answer_a_tx, answer_a_rx) = flume::unbounded();
        srv.sessions.get_mut(&key_a).unwrap().handle.answer_tx = answer_a_tx;
        let (key_b, mut runtime_b) = fixture_runtime();
        let (answer_b_tx, answer_b_rx) = flume::unbounded();
        runtime_b.handle.answer_tx = answer_b_tx;
        srv.sessions.insert(key_b.clone(), runtime_b);
        assert_ne!(key_a, key_b, "two sessions hold distinct ids");

        drive_runtime(&srv, &key_a, |sender| {
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
        })
        .await;
        drive_runtime(&srv, &key_b, |sender| {
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
        })
        .await;
        let frames: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(frames.len(), 2, "each pump asks through its own session");
        let id_a = frames[0]["id"].clone();
        let id_b = frames[1]["id"].clone();
        assert_ne!(id_a, id_b, "ask ids are globally unique");
        for (frame, key) in frames.iter().zip([&key_a, &key_b]) {
            assert_eq!(frame["method"], "session/request_permission");
            let sid = srv.sessions.get(key).unwrap().handle.session_id.to_string();
            assert_eq!(frame["params"]["sessionId"], sid);
        }

        for (key, id) in [(&key_a, PROMPT_ID), (&key_b, 8)] {
            let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(id));
        }

        drive_runtime(&srv, &key_b, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: TEXT_B.to_owned(),
                })
                .unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;
        let settled: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(
            settled.len(),
            2,
            "B's text and B's prompt answer: {settled:?}"
        );
        assert_eq!(settled[1]["id"], 8);
        assert_eq!(settled[1]["result"]["stopReason"], "end_turn");
        {
            let runtime_b = srv.sessions.get(&key_b).unwrap();
            let guard = runtime_b.pending.lock().unwrap();
            assert!(guard.prompt.is_none(), "B settled its own prompt");
            assert!(guard.asks.is_empty(), "B retired its ask on settlement");
            assert_eq!(guard.epoch.current(), 1, "B bumped its own epoch");
        }
        {
            let runtime_a = srv.sessions.get(&key_a).unwrap();
            let guard = runtime_a.pending.lock().unwrap();
            assert_eq!(
                guard.prompt,
                Some(RequestId::Number(PROMPT_ID)),
                "A's prompt stays parked while B settles"
            );
            assert_eq!(guard.asks.len(), 1, "A's blocked ask stays put");
            assert_eq!(guard.epoch.current(), 0, "B never moves A's epoch");
        }

        handle_incoming_response(
            &srv,
            &serde_json::json!({
                "id": id_a,
                "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
            }),
        );
        assert_eq!(
            answer_a_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode()),
            "A's answer still routes to A after B settled"
        );
        assert!(answer_b_rx.is_empty(), "nothing lands on the peer");
        handle_incoming_response(
            &srv,
            &serde_json::json!({
                "id": id_b,
                "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
            }),
        );
        assert!(
            answer_b_rx.is_empty(),
            "B's retired ask rejects after B settled"
        );

        drive_runtime(&srv, &key_a, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx
            .try_iter()
            .find(|frame| frame.get("id").is_some() && frame["id"] == PROMPT_ID)
            .expect("A still settles afterwards");
        assert_eq!(answer["result"]["stopReason"], "end_turn");
        assert_eq!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
            1
        );
    }

    /// 02.9a addressed-cancellation fixture: one PID owns a working session A
    /// and a working session B. Cancelling A settles its outstanding prompt
    /// exactly once through the turn epoch and retires all its asks, while B
    /// continues untouched. Late answers and terminal events from the
    /// cancelled turn change neither the settled outcome nor B; replacement
    /// prompts for A reject until 02.9b confirms cleanup, while B stays
    /// responsive. No prompt here reaches a provider: prompts park via
    /// `handle_prompt` against rewired live channels, and interleaving is
    /// driven with synthetic agent events.
    #[tokio::test]
    async fn addressed_cancel_settles_once_retires_asks_and_leaves_the_peer_alone() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        // A's channels are observable: answers land here, the cancel signal
        // is read off the live cancel channel, and the prompt send has a live
        // receiver so parking never fails on a dropped pipe.
        let (answer_a_tx, answer_a_rx) = flume::unbounded();
        srv.sessions.get_mut(&key_a).unwrap().handle.answer_tx = answer_a_tx;
        let (_input_a_rx, cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (answer_b_tx, answer_b_rx) = flume::unbounded();
        runtime_b.handle.answer_tx = answer_b_tx;
        // B's input stays live so a later prompt proves the peer responsive.
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        assert_ne!(key_a, key_b, "two sessions hold distinct ids");

        // Both sessions work a turn; A also waits on two asks (permission +
        // elicitation), B on one.
        for (key, id) in [(&key_a, PROMPT_ID), (&key_b, 8)] {
            let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(id));
        }
        let sid_a = srv
            .sessions
            .get(&key_a)
            .unwrap()
            .handle
            .session_id
            .to_string();
        let sid_b = srv
            .sessions
            .get(&key_b)
            .unwrap()
            .handle
            .session_id
            .to_string();
        {
            let mut guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            guard.asks.insert(
                ANSWERED_ID,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session: sid_a.clone(),
                    epoch: 0,
                },
            );
            guard.asks.insert(
                SECOND_ASK_ID,
                Ask::Elicitation {
                    session: sid_a.clone(),
                    epoch: 0,
                },
            );
        }
        {
            let mut guard = srv.sessions.get(&key_b).unwrap().pending.lock().unwrap();
            guard.asks.insert(
                THIRD_ASK_ID,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session: sid_b.clone(),
                    epoch: 0,
                },
            );
        }

        // Stop A, addressed, no pin: settles A's turn and retires its asks.
        let result = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_a } }),
        )
        .unwrap();
        assert_eq!(result["sessionId"], key_a);
        assert_eq!(result["cancelled"], true);
        assert_eq!(result["turnEpoch"], 1);
        assert!(
            cancel_a_rx.try_recv().is_ok(),
            "the addressed cancel signals A's run, never the shared process"
        );

        let answer = out_rx.try_recv().expect("A's prompt settles once");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["result"]["stopReason"], "cancelled");
        assert_eq!(
            answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the answer settles the turn it ran under: {answer}"
        );
        assert_eq!(
            answer["result"]["_meta"][extension::RAW_REASON_META_KEY],
            "cancelled",
            "the verbatim reason survives: {answer}"
        );
        assert_eq!(
            answer["result"]["_meta"][extension::USABLE_META_KEY],
            false,
            "a cancelled turn banks no usable output: {answer}"
        );
        {
            let guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            assert!(guard.prompt.is_none(), "A settled its own prompt");
            assert!(guard.turn.is_none());
            assert!(guard.asks.is_empty(), "A retired every ask on cancel");
            assert_eq!(guard.epoch.current(), 1, "A bumped its own epoch");
            assert!(!guard.is_resumable(), "settlement alone never reopens A");
        }
        {
            let guard = srv.sessions.get(&key_b).unwrap().pending.lock().unwrap();
            assert_eq!(guard.prompt, Some(RequestId::Number(8)));
            assert_eq!(guard.asks.len(), 1, "B's ask stays put");
            assert_eq!(guard.epoch.current(), 0, "B never moves");
            assert!(guard.is_resumable());
        }

        // A late answer for a retired ask drops without reaching any agent.
        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(answer_a_rx.is_empty(), "a late answer cannot revive A");
        assert!(answer_b_rx.is_empty(), "nothing lands on the peer");

        // Settlement alone never reopens A: a replacement prompt rejects
        // while cleanup is unconfirmed. Positive retry proof is 02.9b's.
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "too soon"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::invalid_params().code);
        assert_eq!(
            err.data,
            Some(
                serde_json::json!({"reason": "cleanup_unconfirmed", "sessionId": key_a, "uncertain": false})
            )
        );

        // A late terminal from the cancelled turn settles nothing: the prompt
        // is gone and the epoch stands, so it can never settle a later turn.
        assert_eq!(
            settle_prompt(&srv.sessions.get(&key_a).unwrap().pending),
            None,
            "the stale terminal settles nothing"
        );
        drive_runtime(&srv, &key_a, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        assert!(
            out_rx.try_recv().is_err(),
            "no second answer for the settled turn"
        );
        // 02.9b: that same terminal is the cancelled run ending, so in
        // blocking mode it confirms the tools stopped with it — settlement
        // still settled nothing, but the session is resumable now.
        assert!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .is_resumable(),
            "the run's terminal confirms its cleanup"
        );

        // B runs a full turn afterwards: its text and prompt answer carry its
        // own epoch, and A's settled state stands.
        drive_runtime(&srv, &key_b, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: TEXT_B.to_owned(),
                })
                .unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;
        let settled: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(
            settled.len(),
            2,
            "B's text and B's prompt answer: {settled:?}"
        );
        assert_eq!(settled[1]["id"], 8);
        assert_eq!(settled[1]["result"]["stopReason"], "end_turn");
        assert_eq!(
            settled[1]["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "B settles the turn it ran under"
        );
        assert_eq!(
            srv.sessions
                .get(&key_b)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
            1
        );
        // B settling moves only B: A's epoch stands and no frame names A.
        assert_eq!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
            1,
            "B settling never moves A's epoch"
        );

        // B stays responsive on its next turn. A's replacement/retry race is
        // 02.9b's fixture below, on the confirmation this test just proved.
        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays responsive");
    }

    /// 02.9b headline fixture: Stop A, confirm its turn/tool cleanup through
    /// the cancelled run's terminal, then accept another A prompt while B
    /// continues under the same Maki PID (one `Server`, two runtimes — no
    /// second process). Late answers and duplicate terminals from the
    /// cancelled turn cannot affect A's replacement turn, and B's traffic
    /// never leaks into A. No prompt here reaches a provider: turns are
    /// parked by hand and terminals are fed as synthetic agent events, so no
    /// keys or network are involved.
    #[tokio::test]
    async fn stop_confirm_cleanup_then_retry_while_the_peer_continues() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (answer_a_tx, answer_a_rx) = flume::unbounded();
        srv.sessions.get_mut(&key_a).unwrap().handle.answer_tx = answer_a_tx;
        let (input_a_rx, cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (answer_b_tx, answer_b_rx) = flume::unbounded();
        runtime_b.handle.answer_tx = answer_b_tx;
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        assert_ne!(key_a, key_b, "two sessions hold distinct ids");

        // Both sessions work a turn; A also waits on a permission ask.
        for (key, id) in [(&key_a, PROMPT_ID), (&key_b, 8)] {
            let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(id));
        }
        let sid_a = srv
            .sessions
            .get(&key_a)
            .unwrap()
            .handle
            .session_id
            .to_string();
        {
            let mut guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            guard.asks.insert(
                ANSWERED_ID,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session: sid_a.clone(),
                    epoch: 0,
                },
            );
        }

        // Stop A: settles its turn, retires its ask, signals its run.
        let result = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_a } }),
        )
        .unwrap();
        assert_eq!(result["cancelled"], true);
        assert_eq!(result["turnEpoch"], 1);
        assert!(
            cancel_a_rx.try_recv().is_ok(),
            "the addressed cancel signals A's run"
        );
        let answer = out_rx.try_recv().expect("A's prompt settles once");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["result"]["stopReason"], "cancelled");

        // Premature retry is rejected before anything is queued...
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "too soon"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(
            err.data,
            Some(
                serde_json::json!({"reason": "cleanup_unconfirmed", "sessionId": key_a, "uncertain": false})
            )
        );
        assert!(input_a_rx.is_empty(), "a rejected retry queues no input");

        // ...until the cancelled run's terminal confirms its tools stopped.
        // The confirmation answers nothing: the prompt already settled.
        drive_runtime(&srv, &key_a, |sender| {
            sender.send(cancelled_done_event()).unwrap();
        })
        .await;
        assert!(
            out_rx.is_empty(),
            "the confirmation terminal answers nothing"
        );
        assert!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .is_resumable(),
            "a stopped run reopens its session"
        );
        assert_eq!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
            1,
            "confirming cleanup moves no epoch"
        );

        // The retry is accepted onto the confirmed session...
        handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "again"),
            &RequestId::Number(9),
        )
        .expect("confirmed cleanup reopens A");
        assert!(
            input_a_rx.try_recv().is_ok(),
            "the replacement turn is queued on A's run"
        );
        // ...for exactly one turn: a second prompt still rejects as busy.
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "overlap"),
            &RequestId::Number(10),
        )
        .unwrap_err();
        assert_eq!(i32::from(err.code), -32603);
        assert_eq!(err.message, PROMPT_RUNNING);

        // A late answer for the retired ask drops without reaching any agent...
        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(answer_a_rx.is_empty(), "a late answer cannot revive A");
        assert!(answer_b_rx.is_empty(), "nothing lands on the peer");

        // ...and the replacement turn settles exactly once under its own epoch.
        drive_runtime(&srv, &key_a, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx.try_recv().expect("the replacement turn settles");
        assert_eq!(answer["id"], 9);
        assert_eq!(
            answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            1,
            "the replacement settles the turn it ran under"
        );
        // A duplicate terminal after the replacement settled is dropped: the
        // session is not awaiting cleanup, so it confirms nothing and answers
        // nothing.
        drive_runtime(&srv, &key_a, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        assert!(
            out_rx.is_empty(),
            "no second answer for the settled replacement"
        );

        // B ran undisturbed throughout and settles under its own epoch.
        {
            let guard = srv.sessions.get(&key_b).unwrap().pending.lock().unwrap();
            assert_eq!(guard.prompt, Some(RequestId::Number(8)));
            assert_eq!(guard.epoch.current(), 0);
        }
        drive_runtime(&srv, &key_b, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx.try_recv().expect("B settles its own prompt");
        assert_eq!(answer["id"], 8);
        assert_eq!(
            answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "B settles the turn it ran under"
        );
    }

    /// 02.9b held cleanup: with no terminal from the cancelled run the
    /// session stays non-resumable, the rejection reports the uncertainty
    /// explicitly once the bounded wait lapses, and B stays responsive. A
    /// late terminal still confirms: overdue is a report, not a verdict.
    #[tokio::test]
    async fn held_cleanup_prevents_retry_reports_uncertainty_and_leaves_the_peer_alive() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (_input_a_rx, _cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);

        for (key, id) in [(&key_a, PROMPT_ID), (&key_b, 8)] {
            let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(id));
        }
        handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_a } }),
        )
        .unwrap();
        let answer = out_rx.try_recv().expect("the cancelled prompt settles");
        assert_eq!(answer["id"], PROMPT_ID);
        assert!(
            !srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .is_resumable()
        );

        // Premature retry rejects as pending, not uncertain...
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "too soon"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(
            err.data,
            Some(
                serde_json::json!({"reason": "cleanup_unconfirmed", "sessionId": key_a, "uncertain": false})
            )
        );

        // ...while B runs its turn to completion under the same PID.
        drive_runtime(&srv, &key_b, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx.try_recv().expect("B settles its own prompt");
        assert_eq!(answer["id"], 8);
        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays responsive on its next turn");

        // The bounded wait lapses with no terminal: held cleanup, reported.
        {
            let mut guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            guard.awaiting_since = Some(Instant::now() - CLEANUP_GRACE - Duration::from_secs(1));
            assert!(guard.is_uncertain(), "a lapsed bound reports uncertainty");
        }
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "still soon"),
            &RequestId::Number(11),
        )
        .unwrap_err();
        assert_eq!(
            err.data,
            Some(
                serde_json::json!({"reason": "cleanup_unconfirmed", "sessionId": key_a, "uncertain": true})
            ),
            "held cleanup reports uncertainty, still non-resumable"
        );

        // A late terminal still confirms: the run ended, so in blocking mode
        // its tools stopped with it — overdue never hardens into failure.
        drive_runtime(&srv, &key_a, |sender| {
            sender.send(cancelled_done_event()).unwrap();
        })
        .await;
        assert!(
            srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .is_resumable()
        );
        handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "late but confirmed"),
            &RequestId::Number(12),
        )
        .expect("a late terminal confirms the held cleanup");
    }

    /// 02.9b failed cleanup: an `Error` terminal from the cancelled run is
    /// failure, not confirmation. The session reports the uncertainty
    /// explicitly, stays non-resumable, and never self-heals into retry; B is
    /// unaffected throughout.
    #[tokio::test]
    async fn failed_cleanup_reports_uncertainty_and_stays_non_resumable() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (_input_a_rx, _cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);

        for (key, id) in [(&key_a, PROMPT_ID), (&key_b, 8)] {
            let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(id));
        }
        handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_a } }),
        )
        .unwrap();
        let answer = out_rx.try_recv().expect("the cancelled prompt settles");
        assert_eq!(answer["id"], PROMPT_ID);

        // The cancelled run fails instead of stopping: no answer (the prompt
        // already settled), no resumability, recorded uncertainty.
        drive_runtime(&srv, &key_a, |sender| {
            sender
                .send(AgentEvent::Error {
                    message: TURN_ERROR.to_owned(),
                    auth: false,
                    totals: RunTotals::default(),
                })
                .unwrap();
        })
        .await;
        assert!(out_rx.is_empty(), "a failed cleanup answers nothing");
        {
            let guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            assert!(
                !guard.is_resumable(),
                "failed cleanup never reopens the session"
            );
            assert!(guard.is_uncertain(), "failed cleanup reports uncertainty");
        }
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "retry"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(
            err.data,
            Some(
                serde_json::json!({"reason": "cleanup_unconfirmed", "sessionId": key_a, "uncertain": true})
            )
        );

        // A repeated failure changes nothing: uncertainty never self-heals.
        drive_runtime(&srv, &key_a, |sender| {
            sender
                .send(AgentEvent::Error {
                    message: TURN_ERROR.to_owned(),
                    auth: false,
                    totals: RunTotals::default(),
                })
                .unwrap();
        })
        .await;
        assert!(
            !srv.sessions
                .get(&key_a)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .is_resumable()
        );
        assert!(out_rx.is_empty(), "a repeated failure answers nothing");

        // B is unaffected: it settles and stays responsive.
        drive_runtime(&srv, &key_b, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: TEXT_B.to_owned(),
                })
                .unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;
        let settled: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(
            settled.len(),
            2,
            "B's text and B's prompt answer: {settled:?}"
        );
        assert_eq!(settled[1]["id"], 8);
        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays responsive");
    }

    /// 02.9b gate matrix on `Pending` itself: an observation with nothing
    /// awaited changes nothing, a `Done` confirms, an `Error` records
    /// uncertainty. Overdue is report-only: it never appears without a wait.
    #[test]
    fn cleanup_confirmation_marks_exactly_what_the_run_reported() {
        let mut pending = Pending::default();
        pending.note_run_ended(false);
        assert!(
            pending.is_resumable(),
            "no awaited cleanup, nothing to confirm"
        );
        assert!(!pending.is_uncertain(), "no wait lapses without one");

        pending.turn = Some(pending.epoch.current());
        pending.prompt = Some(RequestId::Number(PROMPT_ID));
        assert!(pending.settle_cancel().cancelled);
        assert!(!pending.is_resumable(), "a settled cancel awaits cleanup");
        assert!(
            !pending.is_uncertain(),
            "a fresh cancel is pending, not uncertain"
        );

        pending.note_run_ended(true);
        assert!(
            !pending.is_resumable(),
            "failed cleanup never reopens the session"
        );
        assert!(pending.is_uncertain(), "failed cleanup reports uncertainty");

        pending.confirm_cleanup();
        assert!(
            pending.is_resumable(),
            "a confirmed cleanup reopens the session"
        );
        assert!(!pending.is_uncertain());
    }

    /// 02.10a loss gate matrix on `Pending` itself: first loss wins, bumps
    /// once, retires asks in both modes, and marks the handle failed; a
    /// second loss settles nothing and moves no epoch. No teardown runs here:
    /// the claim stays held for 02.10b, and a dropped handle is explicitly
    /// not cleanup proof.
    #[test]
    fn loss_settlement_marks_non_resumable_retires_asks_and_bumps_once() {
        let mut pending = Pending::default();
        pending.turn = Some(pending.epoch.current());
        pending.prompt = Some(RequestId::Number(PROMPT_ID));
        pending.asks.insert(
            ANSWERED_ID,
            Ask::Elicitation {
                session: "s".to_owned(),
                epoch: 0,
            },
        );
        assert!(!pending.is_lost());

        let first = pending.settle_loss("task_lost", "task_dead");
        assert!(first.first);
        assert_eq!(first.prompt, Some(RequestId::Number(PROMPT_ID)));
        assert_eq!(first.epoch, 1);
        assert!(pending.is_lost());
        assert!(!pending.is_resumable(), "loss never reopens the session");
        assert!(pending.prompt.is_none());
        assert!(pending.turn.is_none());
        assert!(pending.asks.is_empty(), "loss retires asks in both modes");
        assert_eq!(pending.loss_reason.as_deref(), Some("task_lost"));
        assert_eq!(pending.loss_raw.as_deref(), Some("task_dead"));

        let second = pending.settle_loss("task_lost", "stream_ended");
        assert!(!second.first);
        assert!(second.prompt.is_none());
        assert_eq!(second.epoch, 1, "a second loss moves no epoch");
        assert_eq!(pending.loss_raw.as_deref(), Some("task_dead"));
    }

    /// 02.10a loss matrix without a parked prompt: an idle loss still bumps
    /// (so a racing terminal lands stale), notifies `lost`, and stays
    /// failed/non-resumable with no prompt to answer.
    #[test]
    fn idle_loss_bumps_without_a_prompt_to_settle() {
        let mut pending = Pending::default();
        let first = pending.settle_loss("task_lost", "stream_ended");
        assert!(first.first);
        assert!(first.prompt.is_none());
        assert_eq!(first.epoch, 1);
        assert!(pending.is_lost());
        assert!(!pending.is_resumable());
        let second = pending.settle_loss("task_lost", "pump_dead");
        assert!(!second.first);
        assert_eq!(second.epoch, 1);
    }

    /// 02.10a headline: task loss during a prompt settles once via the turn
    /// epoch, retires asks, notifies the 02.1 lost-handle fields
    /// (reason/rawReason/usable), and leaves the peer usable. A second loss
    /// observation (stream end racing the task death) sends nothing. No
    /// prompt reaches a provider and no keys or network are involved.
    #[tokio::test]
    async fn task_loss_during_prompt_settles_once_asks_retire_peers_usable() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (_input_a_rx, _cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);

        for (key, id) in [(&key_a, PROMPT_ID), (&key_b, 8)] {
            let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(id));
        }
        let sid_a = srv
            .sessions
            .get(&key_a)
            .unwrap()
            .handle
            .session_id
            .to_string();
        {
            let mut guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            guard.asks.insert(
                ANSWERED_ID,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session: sid_a.clone(),
                    epoch: 0,
                },
            );
        }

        // Injected task death: the supervisor observes it post-activation.
        let first = report_loss(
            &srv.sessions.get(&key_a).unwrap().pending,
            &srv.out_tx,
            &key_a,
            "task_dead",
        );
        assert!(first, "the first loss observation reports");
        let lifecycle = out_rx.try_recv().expect("loss notifies `lost`");
        assert_eq!(lifecycle["method"], extension::LIFECYCLE_NOTIFICATION);
        assert_eq!(lifecycle["params"]["sessionId"], key_a);
        assert_eq!(lifecycle["params"]["state"], "lost");
        assert_eq!(lifecycle["params"]["turnEpoch"], 1);
        assert_eq!(lifecycle["params"]["rawReason"], "task_dead");
        assert_eq!(lifecycle["params"]["usable"], false);
        let answer = out_rx
            .try_recv()
            .expect("the outstanding prompt settles once");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["error"]["code"], -32603);
        assert_eq!(answer["error"]["data"]["reason"], "task_lost");
        assert_eq!(answer["error"]["data"]["rawReason"], "task_dead");
        assert_eq!(answer["error"]["data"]["usable"], false);
        assert_eq!(answer["error"]["data"]["sessionId"], key_a);
        {
            let guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            assert!(guard.is_lost());
            assert!(!guard.is_resumable(), "a lost handle never reopens");
            assert!(guard.asks.is_empty(), "loss retires asks in both modes");
            assert_eq!(guard.epoch.current(), 1);
        }

        // Stream end racing the task death collapses to the same single loss.
        let second = report_loss(
            &srv.sessions.get(&key_a).unwrap().pending,
            &srv.out_tx,
            &key_a,
            "stream_ended",
        );
        assert!(!second, "a second loss sends nothing");
        assert!(out_rx.is_empty(), "exactly-once loss: no double-settle");

        // A late answer for the retired ask drops; further prompts and
        // addressed calls answer `resource_not_found` per the 02.1 contract.
        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));
        assert!(out_rx.is_empty());
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "retry"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::resource_not_found(None).code);
        let cancel = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_a } }),
        )
        .unwrap_err();
        assert_eq!(cancel.code, AcpError::resource_not_found(None).code);

        // B runs undisturbed and settles under its own epoch (unsupervised
        // throwaway pump: production keeps the stream open after `Done`, so
        // no `lost` follows a normal turn).
        drive_runtime(&srv, &key_b, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx.try_recv().expect("B settles its own prompt");
        assert_eq!(answer["id"], 8);
        assert_eq!(answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY], 0);
        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays usable through a sibling's loss");
    }

    /// 02.10a idle loss: pump death between prompts notifies `lost` with no
    /// prompt to settle; the next prompt rejects as `resource_not_found`
    /// while the peer continues. Supervision work stays off-loop (one lock
    /// plus wire sends for the addressed session only).
    #[tokio::test]
    async fn pump_loss_between_prompts_notifies_loss_future_prompt_rejects() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (_input_a_rx, _cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        {
            let mut guard = srv.sessions.get(&key_b).unwrap().pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(8));
        }

        // Injected pump death while A is idle: no prompt outstanding.
        let (pump_done_tx, pump_done_rx) = flume::bounded::<()>(1);
        let supervisor = start_loss_supervisor(
            Arc::clone(&srv.sessions.get(&key_a).unwrap().pending),
            srv.out_tx.clone(),
            key_a.clone(),
            pump_done_rx,
        );
        drop(pump_done_tx);
        supervisor.await.unwrap();
        let lifecycle = out_rx.try_recv().expect("idle loss still notifies `lost`");
        assert_eq!(lifecycle["params"]["state"], "lost");
        assert_eq!(lifecycle["params"]["sessionId"], key_a);
        assert_eq!(lifecycle["params"]["turnEpoch"], 1);
        assert_eq!(lifecycle["params"]["rawReason"], "pump_dead");
        assert_eq!(lifecycle["params"]["usable"], false);
        {
            let guard = srv.sessions.get(&key_a).unwrap().pending.lock().unwrap();
            assert!(guard.is_lost());
            assert_eq!(guard.epoch.current(), 1, "idle loss still bumps");
        }

        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key_a, "next"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::resource_not_found(None).code);

        drive_runtime(&srv, &key_b, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let answer = out_rx.try_recv().expect("B settles under its own epoch");
        assert_eq!(answer["id"], 8);
    }

    /// 02.10a stream-end honesty: a stream that ends with an outstanding
    /// prompt settles it once as `lost`; an idle stream end notifies `lost`
    /// with no answer. Both retire asks and bump once.
    #[tokio::test]
    async fn stream_end_settles_honestly_with_and_without_a_prompt() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_supervised_pump(&srv, &key_a, dead_handle(), |_| {}).await;
        let lifecycle = out_rx.try_recv().expect("stream end notifies `lost`");
        assert_eq!(lifecycle["params"]["state"], "lost");
        assert_eq!(lifecycle["params"]["rawReason"], "stream_ended");
        assert_eq!(lifecycle["params"]["usable"], false);
        let answer = out_rx
            .try_recv()
            .expect("the outstanding prompt settles once");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["error"]["data"]["reason"], "task_lost");
        assert!(out_rx.is_empty(), "no double-settle");

        // Idle stream end: lifecycle only, still bumps.
        let (srv, .., out_rx) = test_server();
        let key = session_key(&srv);
        run_supervised_pump(&srv, &key, dead_handle(), |_| {}).await;
        let lifecycle = out_rx.try_recv().expect("idle stream end notifies `lost`");
        assert_eq!(lifecycle["params"]["state"], "lost");
        assert_eq!(
            srv.sessions
                .get(&key)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
            1
        );
        assert!(out_rx.try_recv().is_err(), "idle loss answers no prompt");
    }

    /// 02.10a dead-channel honesty: a prompt sent on a dead input channel
    /// reports loss once (lifecycle plus the discovering prompt's loss
    /// error); the next prompt answers `resource_not_found`. Peers are
    /// untouched.
    #[tokio::test]
    async fn dead_channel_on_prompt_settles_honestly() {
        let (mut srv, key, input_rx, out_rx) = server_with_live_input();
        let (other_key, other) = fixture_runtime();
        srv.sessions.insert(other_key.clone(), other);
        drop(input_rx);

        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key, "hello"),
            &RequestId::Number(PROMPT_ID),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::internal_error().code);
        assert_eq!(
            err.data,
            Some(serde_json::json!({
                "reason": "task_lost",
                "rawReason": "input_channel_dead",
                "usable": false,
                "sessionId": key,
            }))
        );
        let lifecycle = out_rx.try_recv().expect("dead channel notifies `lost`");
        assert_eq!(lifecycle["params"]["state"], "lost");
        assert_eq!(lifecycle["params"]["rawReason"], "input_channel_dead");
        let repeat = handle_prompt(
            &mut srv,
            &prompt_request(&key, "again"),
            &RequestId::Number(8),
        )
        .unwrap_err();
        assert_eq!(repeat.code, AcpError::resource_not_found(None).code);
        assert!(
            srv.sessions.contains_key(&key),
            "the claim stays held for 02.10b"
        );
        assert!(srv.sessions.contains_key(&other_key));
    }

    /// 02.10a `Error`-alone rule: an `Error` on a proved-live handle settles
    /// as a failed turn (session stays resumable); an `Error` on a dead
    /// handle takes the loss path (settles once as `lost`, retires asks,
    /// non-resumable). Neither disturbs the peer.
    #[tokio::test]
    async fn error_takes_loss_path_only_when_the_handle_is_dead() {
        // Live handle: normal failed-turn settlement, still resumable.
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_supervised_pump(&srv, &session_key(&srv), live_handle(), |sender| {
            sender
                .send(AgentEvent::Error {
                    message: TURN_ERROR.to_owned(),
                    auth: false,
                    totals: RunTotals::default(),
                })
                .unwrap();
        })
        .await;
        // The `Error` settles as a failed turn plus the trailing stream-end
        // loss: exactly one prompt answer (the failure), then one `lost`
        // lifecycle. The failure itself proves nothing about the next prompt;
        // the stream end proves the handle dead.
        let frames: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(frames.len(), 2, "{frames:?}");
        assert_eq!(frames[0]["id"], PROMPT_ID);
        assert_eq!(frames[0]["error"]["data"], TURN_ERROR);
        assert_eq!(frames[1]["params"]["state"], "lost");
        assert_eq!(frames[1]["params"]["rawReason"], "stream_ended");

        // Dead handle: the `Error` itself takes the loss path.
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_supervised_pump(&srv, &session_key(&srv), dead_handle(), |sender| {
            sender
                .send(AgentEvent::Error {
                    message: TURN_ERROR.to_owned(),
                    auth: false,
                    totals: RunTotals::default(),
                })
                .unwrap();
        })
        .await;
        let frames: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(frames.len(), 2, "{frames:?}");
        assert_eq!(frames[0]["params"]["state"], "lost");
        assert_eq!(frames[0]["params"]["rawReason"], TURN_ERROR);
        assert_eq!(frames[1]["id"], PROMPT_ID);
        assert_eq!(frames[1]["error"]["data"]["reason"], "task_lost");
        assert!(
            srv.sessions
                .values()
                .next()
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .is_lost()
        );
    }

    /// 02.10a activation-handoff post-half: 02.3 proved the pre-activation
    /// half (slow-startup failure settles the activation result once). Here
    /// activation succeeds and files the runtime, then the fresh task dies in
    /// the handoff window: loss settles exactly once (lifecycle plus no
    /// prompt, since none exists yet) with no double-settle, and the peer
    /// stays usable. Failed activation still consumes its lease (02.3), and
    /// no teardown runs on the loss path (02.10b owns release).
    #[tokio::test]
    async fn handoff_window_failure_after_activation_settles_loss_once() {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, completion_rx, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage.clone());
        let (peer_key, mut peer) = fixture_runtime();
        let (_peer_input, _peer_cancel) = rewire_live(&mut peer);
        srv.sessions.insert(peer_key.clone(), peer);

        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;
        let output = project
            .path()
            .join("plan.md")
            .to_string_lossy()
            .into_owned();
        let cwd = project.path().to_string_lossy().into_owned();
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            RequestId::Number(2),
            &activate_request(&session, "openai/gpt-5", "build", &cwd, &output),
            &params,
        )
        .await;
        let (key, id, outcome) = next_completion(&completion_rx).await;
        settle_activation(&mut srv, key.clone(), id, outcome);
        let reply = out_rx
            .try_recv()
            .expect("activation settles its result once");
        assert_eq!(reply["result"]["sessionId"], session);
        let lifecycle = out_rx.try_recv().expect("activation announces active");
        assert_eq!(lifecycle["params"]["state"], "active");

        // The fresh task dies in the handoff window with no prompt yet.
        let pending = Arc::clone(&srv.sessions.get(&session).unwrap().pending);
        assert!(report_loss(&pending, &srv.out_tx, &session, "task_dead"));
        assert!(!report_loss(
            &pending,
            &srv.out_tx,
            &session,
            "stream_ended"
        ));
        let lost = out_rx
            .try_recv()
            .expect("handoff loss notifies `lost` once");
        assert_eq!(lost["params"]["state"], "lost");
        assert_eq!(lost["params"]["rawReason"], "task_dead");
        assert!(
            out_rx.is_empty(),
            "no prompt exists, so no answer; no double-settle"
        );
        {
            let guard = srv.sessions.get(&session).unwrap().pending.lock().unwrap();
            assert!(guard.is_lost());
            assert_eq!(guard.epoch.current(), 1);
        }

        // The peer routes while the handoff loss stands.
        handle_prompt(
            &mut srv,
            &prompt_request(&peer_key, "hello"),
            &RequestId::Number(3),
        )
        .expect("the peer stays usable through handoff loss");
        let failed_id: MakiId = session.parse().unwrap();
        assert!(
            SessionClaim::is_held(failed_id, &storage),
            "the claim stays held: loss performs no teardown (02.10b owns release)"
        );
    }

    /// A stale turn pin changes nothing: no signal, no bump, no settlement,
    /// and the parked prompt stays for the live turn. The matching pin
    /// settles; an unknown session rejects without touching the live one.
    #[tokio::test]
    async fn a_stale_cancel_pin_changes_nothing() {
        let (srv, .., out_rx) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        let key = session_key(&srv);

        let stale = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key, "turnEpoch": 99 } }),
        )
        .unwrap_err();
        assert_eq!(stale.code, AcpError::invalid_params().code);
        assert_eq!(
            stale.data,
            Some(serde_json::json!({"expectedEpoch": 0, "gotEpoch": 99}))
        );
        {
            let guard = pending(&srv).lock().unwrap();
            assert_eq!(guard.prompt, Some(RequestId::Number(PROMPT_ID)));
            assert_eq!(guard.epoch.current(), 0, "a stale cancel moves no epoch");
        }
        assert!(out_rx.is_empty(), "a stale cancel settles nothing");

        let unknown = handle_cancel(
            &srv,
            &serde_json::json!({
                "params": { "sessionId": MakiId::generate().to_string() }
            }),
        )
        .unwrap_err();
        assert_eq!(unknown.code, AcpError::resource_not_found(None).code);
        assert_eq!(
            pending(&srv).lock().unwrap().prompt,
            Some(RequestId::Number(PROMPT_ID))
        );

        let result = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key, "turnEpoch": 0 } }),
        )
        .unwrap();
        assert_eq!(result["cancelled"], true);
        assert_eq!(result["turnEpoch"], 1);
        let answer = out_rx
            .try_recv()
            .expect("the pinned cancel settles the prompt");
        assert_eq!(answer["id"], PROMPT_ID);
        assert_eq!(answer["result"]["stopReason"], "cancelled");
    }

    /// Cancelling an idle session still bumps the epoch, so a terminal racing
    /// the cancel lands stale — but with no turn to clean up the session
    /// stays usable and the next prompt starts a new turn under the bumped
    /// epoch. No run is signalled for work that never started.
    #[tokio::test]
    async fn an_idle_cancel_bumps_without_blocking_the_next_turn() {
        let (mut srv, key, _input_rx, out_rx) = server_with_live_input();
        let (cancel_tx, cancel_rx) = flume::unbounded();
        srv.sessions.get_mut(&key).unwrap().handle.cancel_tx = cancel_tx;

        let result =
            handle_cancel(&srv, &serde_json::json!({ "params": { "sessionId": key } })).unwrap();
        assert_eq!(result["cancelled"], false);
        assert_eq!(result["turnEpoch"], 1);
        assert!(out_rx.is_empty(), "an idle cancel answers no prompt");
        assert!(cancel_rx.is_empty(), "nothing runs, so no run is signalled");
        assert!(pending(&srv).lock().unwrap().is_resumable());

        handle_prompt(
            &mut srv,
            &prompt_request(&key, "next"),
            &RequestId::Number(8),
        )
        .expect("an idle-cancelled session stays usable");
        assert_eq!(
            pending(&srv).lock().unwrap().turn,
            Some(1),
            "the next prompt starts a new turn under the bumped epoch"
        );
    }

    /// Answer and Stop serialize through the one per-session control lock:
    /// claims are remove-then-validate under a single guard, and the cancel
    /// settle takes the prompt, bumps, and retires under the same guard, so a
    /// queued answer can never bypass an in-flight cancel settlement. Racing
    /// threads prove the invariant structurally: every wire id is claimed at
    /// most once, the prompt settles at most once, and a cross-session claim
    /// never delivers. (02.10b routes Close through this same lock; the
    /// Close half is proven in `close_queues_behind_in_flight_cancel_settlement`.)
    #[test]
    fn concurrent_answers_never_bypass_in_flight_cancel_settlement() {
        use std::sync::Barrier;

        const ASKS: i64 = 32;
        let pending = PendingState::default();
        {
            let mut guard = pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(PROMPT_ID));
            for n in 0..ASKS {
                guard.asks.insert(
                    ANSWERED_ID + n,
                    Ask::Permission {
                        request_id: format!("toolu_{n}"),
                        answer_tx: None,
                        session: "s".to_owned(),
                        epoch: 0,
                    },
                );
            }
            // Bound to another session: a live id with the wrong owner that
            // must never claim for this session, however raced.
            guard.asks.insert(
                ANSWERED_ID + ASKS,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session: "other".to_owned(),
                    epoch: 0,
                },
            );
        }

        let barrier = Arc::new(Barrier::new(5));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let control = Arc::clone(&pending);
            let gate = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                gate.wait();
                let mut won = Vec::new();
                for n in 0..=ASKS {
                    if control
                        .lock()
                        .unwrap()
                        .claim_ask(ANSWERED_ID + n, "s")
                        .is_some()
                    {
                        won.push(n);
                    }
                }
                won
            }));
        }
        barrier.wait();
        // The Stop lands while answers race it: one guard, one settlement.
        let settlement = pending.lock().unwrap().settle_cancel();

        let mut won: Vec<i64> = Vec::new();
        for handle in handles {
            won.extend(handle.join().expect("no control-lock deadlock"));
        }
        won.sort_unstable();
        let mut dedup = won.clone();
        dedup.dedup();
        assert_eq!(
            won, dedup,
            "every wire id claims at most once, however raced"
        );
        assert!(
            !won.contains(&ASKS),
            "a cross-session ask never claims for this session"
        );
        assert!(settlement.cancelled);
        assert_eq!(settlement.epoch, 1, "one Stop bumps exactly once");
        assert_eq!(settlement.prompt, Some(RequestId::Number(PROMPT_ID)));
        assert!(pending.lock().unwrap().asks.is_empty());
        assert_eq!(
            settle_prompt(&pending),
            None,
            "the settled prompt never answers twice"
        );
    }

    /// Answer handling never waits on another session's slow startup: while
    /// A's activation is parked on the gate, the peer's answer routes through
    /// the same line path the loop uses. The release-time half — peer work
    /// while a hook/MCP teardown is held — is proven by the 02.10b release
    /// tests.
    #[tokio::test]
    async fn peer_answer_routes_while_activation_is_held() {
        let harness = StarterHarness::default();
        harness.hold.store(true, Ordering::SeqCst);
        let (mut srv, out_rx, completion_rx, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage);
        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;
        let output = project
            .path()
            .join("plan.md")
            .to_string_lossy()
            .into_owned();
        let cwd = project.path().to_string_lossy().into_owned();
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            RequestId::Number(2),
            &activate_request(&session, "openai/gpt-5", "build", &cwd, &output),
            &params,
        )
        .await;
        assert!(
            srv.activating.contains_key(&session),
            "startup is still held"
        );

        let (peer_key, mut peer) = fixture_runtime();
        let (answer_tx, answer_rx) = flume::unbounded();
        peer.handle.answer_tx = answer_tx;
        srv.sessions.insert(peer_key.clone(), peer);
        // Built before the insert takes the guard: the mutex is not
        // reentrant, so reading the epoch under a held guard would park here.
        let ask = Ask::Permission {
            request_id: PARENT_TOOL_USE_ID.to_owned(),
            answer_tx: None,
            session: peer_key.clone(),
            epoch: srv
                .sessions
                .get(&peer_key)
                .unwrap()
                .pending
                .lock()
                .unwrap()
                .epoch
                .current(),
        };
        srv.sessions
            .get(&peer_key)
            .unwrap()
            .pending
            .lock()
            .unwrap()
            .asks
            .insert(ANSWERED_ID, ask);

        let line = serde_json::to_string(&allow_once(ANSWERED_ID)).unwrap();
        timeout(
            STDIN_DEADLOCK_TIMEOUT,
            handle_line(&mut srv, &line, &params),
        )
        .await
        .expect("answer handling never waits on startup");
        assert_eq!(
            answer_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode())
        );
        assert!(
            srv.activating.contains_key(&session),
            "startup is still held afterwards"
        );

        harness.gate.notify_one();
        let (key, id, outcome) = next_completion(&completion_rx).await;
        settle_activation(&mut srv, key.clone(), id, outcome);
        assert!(
            srv.sessions.contains_key(&key),
            "the held activation still settles normally"
        );
    }

    /// 02.2b routing: one active prompt per session, strict identity
    /// rejection, fixed-at-activation setup, and exactly-once settlement
    /// through the turn epoch. No prompt here reaches a provider: prompts go
    /// through `handle_prompt` against runtimes whose channels lead nowhere,
    /// and interleaving is driven with synthetic agent events.
    const FIXED_AT_ACTIVATION: &str = "fixed_at_activation";
    const PROMPT_RUNNING: &str = "a prompt is already running";

    fn prompt_request(session: &str, text: &str) -> Value {
        serde_json::json!({
            "params": {
                "sessionId": session,
                "prompt": [{ "type": "text", "text": text }],
            }
        })
    }

    fn mode_request(session: &str, mode: &str) -> Value {
        serde_json::json!({ "params": { "sessionId": session, "modeId": mode } })
    }

    fn config_request(session: &str) -> Value {
        serde_json::json!({
            "params": { "sessionId": session, "configId": "model", "value": "openai/gpt-5" }
        })
    }

    /// A server whose runtime can actually take a prompt: the fixture's input
    /// channel is rewired to a live one, so an accepted prompt sends into the
    /// void instead of failing on a dropped receiver. Nothing runs, so no
    /// provider, keys, or network are involved.
    fn server_with_live_input() -> (
        Server,
        String,
        flume::Receiver<AgentInput>,
        flume::Receiver<Value>,
    ) {
        let (mut server, _answer_rx, out_rx) = test_server();
        let (input_tx, input_rx) = flume::unbounded();
        let key = server.sessions.keys().next().unwrap().clone();
        server.sessions.get_mut(&key).unwrap().handle.input_tx = input_tx;
        (server, key, input_rx, out_rx)
    }

    /// A second prompt on a busy session rejects without replacing the
    /// outstanding request or starting a new turn.
    #[tokio::test]
    async fn a_second_prompt_rejects_without_replacing_the_outstanding_request() {
        let (mut srv, key, _input_rx, ..) = server_with_live_input();
        let first = RequestId::Number(PROMPT_ID);
        handle_prompt(&mut srv, &prompt_request(&key, "first"), &first).unwrap();

        let err = handle_prompt(
            &mut srv,
            &prompt_request(&key, "second"),
            &RequestId::Number(8),
        )
        .unwrap_err();
        assert_eq!(i32::from(err.code), -32603);
        assert_eq!(err.message, PROMPT_RUNNING);
        let guard = pending(&srv).lock().unwrap();
        assert_eq!(
            guard.prompt,
            Some(first),
            "a duplicate prompt never overwrites the outstanding request"
        );
        assert_eq!(
            guard.turn,
            Some(0),
            "a duplicate prompt never starts another turn"
        );
    }

    /// A prompt naming an unknown session rejects before anything is queued,
    /// and neither live runtime's pending request, turn, nor epoch moves.
    #[tokio::test]
    async fn a_prompt_for_an_unknown_session_rejects_without_touching_the_peer() {
        let (mut srv, ..) = server_with_live_input();
        let (other_key, other) = fixture_runtime();
        srv.sessions.insert(other_key, other);

        let unknown = MakiId::generate().to_string();
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&unknown, "hello"),
            &RequestId::Number(9),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::resource_not_found(None).code);
        assert_eq!(err.data, Some(serde_json::json!({"sessionId": unknown})));
        for runtime in srv.sessions.values() {
            let guard = runtime.pending.lock().unwrap();
            assert!(
                guard.prompt.is_none() && guard.turn.is_none(),
                "an unknown prompt parks nothing"
            );
            assert_eq!(guard.epoch.current(), 0, "an unknown prompt moves no epoch");
        }
    }

    /// The first terminal for the live turn settles it and bumps the epoch; a
    /// second terminal for the same epoch is dropped without another bump,
    /// and the next prompt starts a new turn under the new epoch.
    #[tokio::test]
    async fn a_terminal_settles_once_and_the_next_prompt_starts_a_new_turn() {
        let (mut srv, key, _input_rx, ..) = server_with_live_input();
        handle_prompt(
            &mut srv,
            &prompt_request(&key, "first"),
            &RequestId::Number(7),
        )
        .unwrap();
        assert!(
            matches!(
                settle_prompt(pending(&srv)),
                Some((RequestId::Number(7), 0))
            ),
            "the terminal settles the turn it ran under"
        );
        assert_eq!(live_epoch(&srv), 1);
        assert_eq!(
            settle_prompt(pending(&srv)),
            None,
            "a second terminal for one epoch is dropped"
        );
        assert_eq!(live_epoch(&srv), 1, "a dropped terminal never bumps");

        handle_prompt(
            &mut srv,
            &prompt_request(&key, "second"),
            &RequestId::Number(8),
        )
        .unwrap();
        assert!(
            matches!(
                settle_prompt(pending(&srv)),
                Some((RequestId::Number(8), 1))
            ),
            "another prompt after idle starts a new turn"
        );
        assert_eq!(live_epoch(&srv), 2);
    }

    /// A terminal from a turn that is no longer live settles nothing: the
    /// parked prompt stays for the path that owns the new epoch (an addressed
    /// cancel in 02.9a), and the epoch it bumped to stands.
    #[tokio::test]
    async fn a_stale_terminal_cannot_settle_another_turn() {
        let (srv, ..) = test_server();
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        pending(&srv).lock().unwrap().epoch.bump();

        assert_eq!(
            settle_prompt(pending(&srv)),
            None,
            "the stale terminal settles nothing"
        );
        let guard = pending(&srv).lock().unwrap();
        assert_eq!(
            guard.prompt,
            Some(RequestId::Number(PROMPT_ID)),
            "the prompt stays parked for the path that owns the new epoch"
        );
        assert_eq!(guard.epoch.current(), 1);
    }

    /// Post-activation setup changes route to the addressed session, then
    /// reject with the contract reason: nothing is applied and nothing is
    /// emitted. An unknown session rejects with `resource_not_found` first.
    #[tokio::test]
    async fn post_activation_mode_and_config_changes_route_then_reject() {
        let (srv, ..) = test_server();
        let key = session_key(&srv);

        for raw in [mode_request(&key, "plan"), config_request(&key)] {
            let err = if raw["params"].get("modeId").is_some() {
                handle_set_mode(&srv, &raw).unwrap_err()
            } else {
                handle_set_config(&srv, &raw).unwrap_err()
            };
            assert_eq!(err.code, AcpError::invalid_params().code);
            assert_eq!(
                err.data,
                Some(serde_json::json!({"reason": FIXED_AT_ACTIVATION})),
                "the rejection names the fixed-at-activation reason: {raw}"
            );
        }
        assert_eq!(
            only(&srv).current_mode,
            AgentMode::Build,
            "the fixed mode is unchanged"
        );
        assert!(
            only(&srv).current_model.is_empty(),
            "the fixed model is unchanged"
        );

        let unknown = MakiId::generate().to_string();
        for raw in [mode_request(&unknown, "plan"), config_request(&unknown)] {
            let err = if raw["params"].get("modeId").is_some() {
                handle_set_mode(&srv, &raw).unwrap_err()
            } else {
                handle_set_config(&srv, &raw).unwrap_err()
            };
            assert_eq!(err.code, AcpError::resource_not_found(None).code);
            assert_eq!(err.data, Some(serde_json::json!({"sessionId": unknown})));
        }
    }

    /// Rejecting on one session emits nothing and leaves the peer's pending
    /// request, turn, epoch, and setup alone.
    #[tokio::test]
    async fn a_fixed_rejection_leaves_the_peer_alone() {
        let (mut srv, _, out_rx) = test_server();
        let peer = session_key(&srv);
        let (other_key, other) = fixture_runtime();
        srv.sessions.insert(other_key.clone(), other);
        srv.sessions.get_mut(&other_key).unwrap().current_model =
            "other-vendor/other-model".to_owned();

        for raw in [mode_request(&peer, "plan"), config_request(&peer)] {
            let rejected = if raw["params"].get("modeId").is_some() {
                handle_set_mode(&srv, &raw).is_err()
            } else {
                handle_set_config(&srv, &raw).is_err()
            };
            assert!(rejected);
        }
        assert!(out_rx.is_empty(), "a rejection emits no session update");
        let runtime = srv.sessions.get(&other_key).unwrap();
        let guard = runtime.pending.lock().unwrap();
        assert!(
            guard.prompt.is_none() && guard.turn.is_none(),
            "the peer parks nothing"
        );
        assert_eq!(guard.epoch.current(), 0, "the peer epoch never moves");
        assert_eq!(
            runtime.current_model, "other-vendor/other-model",
            "the peer setup is unchanged"
        );
    }

    /// A stale answer is dropped, never applied: the waiting turn keeps
    /// waiting and the next turn's tool stays denied.
    #[tokio::test]
    async fn a_stale_answer_is_dropped_and_never_applied() {
        let (srv, answer_rx, ..) = server_with_permission_ask(None);
        pending(&srv).lock().unwrap().epoch.bump();

        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));

        assert!(
            answer_rx.is_empty(),
            "the stale answer was dropped: {STALE_ALLOW}"
        );
        assert!(!next_turn_is_allowed(&srv), "{STALE_ALLOW}");
        assert!(
            pending(&srv).lock().unwrap().asks.is_empty(),
            "a consumed answer cannot replay onto the next turn"
        );
    }

    /// The ask binding names its session and turn; an answer bound to another
    /// session is dropped even though its id matches a live ask.
    #[tokio::test]
    async fn an_answer_bound_to_another_session_is_dropped() {
        let (srv, answer_rx, ..) = test_server();
        // Built before the insert takes the guard: the mutex is not
        // reentrant, so reading the epoch under a held guard would park here.
        let foreign = Ask::Permission {
            request_id: PARENT_TOOL_USE_ID.to_owned(),
            answer_tx: None,
            session: "another-session".to_owned(),
            epoch: live_epoch(&srv),
        };
        pending(&srv)
            .lock()
            .unwrap()
            .asks
            .insert(ANSWERED_ID, foreign);

        handle_incoming_response(&srv, &allow_once(ANSWERED_ID));

        assert!(
            answer_rx.is_empty(),
            "a cross-session answer is never applied"
        );
    }

    /// Turn-scoped frames carry the turn they were issued under: updates and
    /// asks name the session and epoch, and the prompt answer settles the
    /// turn it ran under while bumping to the next.
    #[tokio::test]
    async fn outgoing_turn_frames_carry_session_and_epoch() {
        let (srv, .., out_rx) = test_server();
        let sid = session_key(&srv);
        park_prompt(&srv, RequestId::Number(PROMPT_ID));
        run_pump(&srv, None, |sender| {
            sender
                .send(AgentEvent::TextDelta {
                    text: QUEUED_TEXT.to_owned(),
                })
                .unwrap();
            sender.send(permission_request(PARENT_TOOL_USE_ID)).unwrap();
            sender.send(done_event()).unwrap();
        })
        .await;

        let frames: Vec<Value> = out_rx.try_iter().collect();
        assert_eq!(
            frames.len(),
            3,
            "text, ask, then the settling answer: {frames:?}"
        );
        assert_eq!(frames[0]["params"]["sessionId"], sid);
        assert_eq!(
            frames[0]["params"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the update carries its turn: {frames:?}"
        );
        assert_eq!(frames[1]["method"], "session/request_permission");
        assert_eq!(frames[1]["params"]["sessionId"], sid);
        assert_eq!(
            frames[1]["params"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the ask carries its turn: {frames:?}"
        );
        assert_eq!(frames[2]["id"], PROMPT_ID);
        assert_eq!(frames[2]["result"]["stopReason"], "end_turn");
        assert_eq!(
            frames[2]["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the answer settles the turn it ran under: {frames:?}"
        );
        assert_eq!(live_epoch(&srv), 1, "settling bumps to the next turn");
    }

    /// A cancel naming an unknown session is dropped: notifications carry no
    /// id to answer on, and no live runtime is disturbed.
    #[tokio::test]
    async fn a_cancel_for_an_unknown_session_is_dropped() {
        let (srv, ..) = test_server();
        let raw = serde_json::json!({
            "params": { "sessionId": MakiId::generate().to_string() }
        });
        handle_notification(&srv, "session/cancel", &raw);

        let guard = pending(&srv).lock().unwrap();
        assert!(guard.prompt.is_none() && guard.turn.is_none());
        assert_eq!(guard.epoch.current(), 0);
    }

    #[tokio::test]
    async fn discovered_models_are_pushed_to_the_client() {
        let (mut srv, .., out_rx) = test_server();
        srv.model_specs = vec![OFFLINE_SPEC.to_owned()];
        let batch = vec![DISCOVERED_SPEC.to_owned()];

        refresh_models(&mut srv, batch.clone());
        let update = out_rx.try_recv().expect("the fuller list is announced");
        let option = &update["params"]["update"]["configOptions"][0];
        assert_eq!(option["id"], methods::MODEL_CONFIG_ID);
        let selectable: Vec<&str> = option["options"]
            .as_array()
            .expect("the option is a select")
            .iter()
            .filter_map(|o| o["value"].as_str())
            .collect();
        assert!(
            selectable.contains(&OFFLINE_SPEC) && selectable.contains(&DISCOVERED_SPEC),
            "a batch is merged into the offline list, not swapped for it: {selectable:?}"
        );

        refresh_models(&mut srv, batch);
        assert!(out_rx.is_empty(), "a batch adding nothing is not announced");
    }

    /// Editor clients call `session/list` to offer resuming a session. The
    /// response must contain only the requested project's stored sessions,
    /// with the fields the picker renders (title, ISO 8601 updated_at).
    #[tokio::test]
    async fn list_sessions_returns_stored_sessions_for_their_cwd() {
        use std::str::FromStr as _;
        let tmp = TempDir::new().unwrap();
        let dir = StateDir::from_path(tmp.path().to_path_buf());
        let mut here: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new(SELECTED_SPEC, "/project");
        here.replace_messages(vec![Message::user("keep".into())]);
        here.save(&SessionClaim::acquire(here.id, &dir).unwrap(), &dir)
            .unwrap();
        let mut elsewhere: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new(SELECTED_SPEC, "/other");
        elsewhere.replace_messages(vec![Message::user("away".into())]);
        elsewhere
            .save(&SessionClaim::acquire(elsewhere.id, &dir).unwrap(), &dir)
            .unwrap();

        let raw = serde_json::json!({ "params": { "cwd": "/project" } });
        let AgentResponse::ListSessionsResponse(list) = list_sessions(&raw, &dir).await.unwrap()
        else {
            panic!("session/list must answer ListSessionsResponse");
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains(&here.id.to_string()), "{json}");
        assert!(!json.contains(&elsewhere.id.to_string()), "{json}");
        assert_eq!(list.sessions.len(), 1);
        assert_eq!(list.sessions[0].title.as_deref(), Some("New session"));
        // Any u64 formats to some string; parsing is the real check.
        let ts = list.sessions[0]
            .updated_at
            .clone()
            .expect("a saved session has an updated_at");
        assert!(jiff::Timestamp::from_str(&ts).is_ok(), "{ts}");
    }

    /// `session/list` must work with no live session. An omitted cwd
    /// applies no filter: sessions of every project are listed.
    #[tokio::test]
    async fn session_list_dispatches_with_no_session_and_no_cwd() {
        let tmp = TempDir::new().unwrap();
        let storage = StateDir::from_path(tmp.path().to_path_buf());
        let cwd = std::env::current_dir().unwrap();
        let mut session: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new(SELECTED_SPEC, &cwd.to_string_lossy());
        session.replace_messages(vec![Message::user("here".into())]);
        session
            .save(
                &SessionClaim::acquire(session.id, &storage).unwrap(),
                &storage,
            )
            .unwrap();
        let mut foreign: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new(SELECTED_SPEC, "/somewhere-else");
        foreign.replace_messages(vec![Message::user("there".into())]);
        foreign
            .save(
                &SessionClaim::acquire(foreign.id, &storage).unwrap(),
                &storage,
            )
            .unwrap();

        let (mut srv, _answer_rx, out_rx) = test_server();
        srv.sessions.clear();
        let params = acp_params(storage);
        let raw = serde_json::json!({ "params": {} });
        handle_request(
            &mut srv,
            "session/list",
            RequestId::Number(7),
            &raw,
            &params,
        )
        .await;
        let resp = out_rx.recv().unwrap();
        let resp = resp.to_string();
        assert!(
            resp.contains(&session.id.to_string()),
            "session/list with no cwd must include the process cwd's session: {resp}"
        );
        assert!(
            resp.contains(&foreign.id.to_string()),
            "an omitted cwd applies no filter, so other projects are listed: {resp}"
        );
    }

    #[test]
    fn load_history_round_trips_stored_messages() {
        let tmp = TempDir::new().unwrap();
        let dir = StateDir::from_path(tmp.path().to_path_buf());
        let messages = vec![
            Message::user("rename foo to bar".into()),
            Message {
                role: Role::Assistant,
                content: vec![MsgBlock::Text {
                    text: "done".into(),
                }],
                display_text: None,
                ..Default::default()
            },
        ];
        let mut session: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new("anthropic/test-model", "/project");
        session.replace_messages(messages.clone());
        session.token_usage = TokenUsage {
            input: 1_000,
            output: 200,
            ..Default::default()
        };
        session
            .save(&SessionClaim::acquire(session.id, &dir).unwrap(), &dir)
            .unwrap();

        let id: MakiId = session.id;
        let restored = claim_history(&dir, id).unwrap().0;
        assert_eq!(restored.session.model, "anthropic/test-model");
        assert_eq!(
            serde_json::to_value(restored.session.messages()).unwrap(),
            serde_json::to_value(&messages).unwrap()
        );
        assert_eq!(restored.cwd, Some(PathBuf::from("/project")));
        assert_eq!(restored.session.token_usage, session.token_usage);
    }

    /// Resuming must bill what the session actually paid. If the per-model
    /// breakdown came back empty or lost its recorded costs, ACP would re-price
    /// the restored total against today's table and disagree with the TUI.
    #[test]
    fn load_history_prices_a_resumed_session_at_what_it_paid() {
        let tmp = TempDir::new().unwrap();
        let dir = StateDir::from_path(tmp.path().to_path_buf());
        let mut session: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new(RETIRED_SPEC, "/project");
        session.token_usage = TokenUsage {
            input: 1_000_000,
            output: 200_000,
            ..Default::default()
        };
        session.add_model_usage(
            RETIRED_MODEL_ID,
            StoredTokenUsage {
                input: 1_000_000,
                output: 200_000,
                cost: Some(RECORDED_COST),
                ..Default::default()
            },
        );
        session
            .save(&SessionClaim::acquire(session.id, &dir).unwrap(), &dir)
            .unwrap();

        let restored = claim_history(&dir, session.id).unwrap().0;
        let mut by_model = restored.session.usage_by_model().clone();
        assert_eq!(
            by_model[RETIRED_MODEL_ID].cost,
            Some(RECORDED_COST),
            "the per-model breakdown survives the file"
        );

        // Mirrors `load_session`: the recorded spec no longer parses, so the
        // selected model stands in, and that must not change the bill.
        let recorded_model = Model::from_spec(&restored.session.model)
            .unwrap_or_else(|_| Model::from_spec(SELECTED_SPEC).expect("a shipped model"));
        assert_eq!(
            settle_session(
                &restored.session.token_usage,
                &mut by_model,
                &recorded_model,
                false
            ),
            Some(RECORDED_COST)
        );
    }

    #[test]
    fn load_history_records_absolute_cwd_only() {
        let tmp = TempDir::new().unwrap();
        let dir = StateDir::from_path(tmp.path().to_path_buf());
        let mut session: Session<Message, TokenUsage, maki_agent::ToolOutput> =
            Session::new("anthropic/test-model", "relative/project");
        session
            .save(&SessionClaim::acquire(session.id, &dir).unwrap(), &dir)
            .unwrap();
        assert_eq!(claim_history(&dir, session.id).unwrap().0.cwd, None);
    }

    #[test]
    fn load_missing_session_is_resource_not_found() {
        let tmp = TempDir::new().unwrap();
        let dir = StateDir::from_path(tmp.path().to_path_buf());
        let err = claim_history(&dir, MakiId::generate()).unwrap_err();
        assert_eq!(err.code, AcpError::resource_not_found(None).code);
    }

    #[test]
    fn converts_injected_mcp_servers() {
        let raw = serde_json::json!({
            "params": {
                "sessionId": MakiId::generate().to_string(),
                "cwd": "/project",
                "mcpServers": [
                    {
                        "type": "http",
                        "name": "kan.dev/mcp",
                        "url": "http://127.0.0.1:41012",
                        "headers": [{ "name": "Authorization", "value": "Bearer abc" }]
                    },
                    {
                        "name": "local",
                        "command": "/usr/bin/mcp",
                        "args": ["--stdio"],
                        "env": [{ "name": "TOKEN", "value": "t" }]
                    },
                    {
                        "type": "sse",
                        "name": "legacy",
                        "url": "http://127.0.0.1:41013",
                        "headers": []
                    }
                ]
            }
        });

        let req: LoadSessionRequest = parse_params(&raw).unwrap();
        let servers = injected_servers(&req.mcp_servers);
        assert_eq!(servers.len(), 2, "sse is dropped, not converted");

        let (name, RawTransport::Http(http)) = &servers[0] else {
            panic!("expected http transport");
        };
        assert_eq!(name, "kan-dev-mcp", "wire names are coerced to valid ones");
        assert_eq!(http.url, "http://127.0.0.1:41012");
        assert_eq!(
            http.headers.get("Authorization").map(String::as_str),
            Some("Bearer abc")
        );

        let (name, RawTransport::Stdio(stdio)) = &servers[1] else {
            panic!("expected stdio transport");
        };
        assert_eq!(name, "local");
        assert_eq!(stdio.command, ["/usr/bin/mcp", "--stdio"]);
        assert_eq!(
            stdio.environment.get("TOKEN").map(String::as_str),
            Some("t")
        );
    }

    /// 02.3 allocation/activation: lease ownership, initial model selection,
    /// pre-activation startup failure, and loop responsiveness under held
    /// startup. The starter is instrumented: it records the model each
    /// activation was constructed with (the session's very first provider
    /// connection), and per test it can hold startup on a gate or fail it.
    /// No prompt is ever sent in these fixtures, so nothing reaches a
    /// provider: no keys, no network.
    #[derive(Clone, Default)]
    struct StarterHarness {
        /// `(session_key, model_spec)` in construction order.
        started: Arc<Mutex<Vec<(String, String)>>>,
        /// What each activation was constructed with (02.4): the requested
        /// session cwd, mode, and absolute output alongside the model spec.
        /// Model/mode/output inputs are available before the first prompt.
        built: Arc<Mutex<Vec<BuiltActivation>>>,
        /// Live input receivers, one per constructed stub: kept so an
        /// accepted prompt has somewhere to send.
        inputs: Arc<Mutex<Vec<flume::Receiver<AgentInput>>>>,
        /// Live cancel receivers, one per constructed stub: `session/cancel`
        /// dispatch is observed here.
        cancels: Arc<Mutex<Vec<flume::Receiver<()>>>>,
        /// When set, the starter parks on `gate` before constructing.
        hold: Arc<AtomicBool>,
        gate: Arc<Notify>,
        /// When set, the starter fails with this reason instead of building.
        fail: Arc<Mutex<Option<String>>>,
    }

    /// What one activation carried into construction (02.4 initial context).
    #[derive(Clone, Debug, PartialEq)]
    struct BuiltActivation {
        session: String,
        model: String,
        mode: String,
        cwd: PathBuf,
        output: PathBuf,
    }

    /// Fixture slow half of activation: records the requested model, then
    /// answers with a stub runtime (or a held/failed outcome). The stub
    /// carries the requested model/mode/cwd/output directly: no default-provider
    /// session is ever built, so there is nothing to switch.
    fn fixture_starter(harness: &StarterHarness) -> ActivationStarter {
        let harness = harness.clone();
        Arc::new(
            move |input: ActivationInput| -> BoxFuture<'static, ActivationResult> {
                let harness = harness.clone();
                Box::pin(async move {
                    if harness.hold.load(Ordering::SeqCst) {
                        harness.gate.notified().await;
                    }
                    let spec = input.spawn.model.spec();
                    harness
                        .started
                        .lock()
                        .unwrap()
                        .push((input.spawn.claim.id().to_string(), spec.clone()));
                    harness.built.lock().unwrap().push(BuiltActivation {
                        session: input.spawn.claim.id().to_string(),
                        model: spec.clone(),
                        mode: format!("{:?}", input.spawn.mode),
                        cwd: input.spawn.cwd.clone(),
                        output: input.spawn.output.clone().unwrap_or_default(),
                    });
                    if let Some(reason) = harness.fail.lock().unwrap().clone() {
                        return Err(ActivationFailure {
                            claim: input.spawn.claim,
                            reason,
                        });
                    }
                    let (input_tx, input_rx) = flume::unbounded();
                    let (cancel_tx, cancel_rx) = flume::unbounded();
                    harness.inputs.lock().unwrap().push(input_rx);
                    harness.cancels.lock().unwrap().push(cancel_rx);
                    let handle = InteractiveHandle {
                        tool_names: Vec::new(),
                        input_tx,
                        answer_tx: flume::unbounded().0,
                        cancel_tx,
                        model_tx: flume::unbounded().0,
                        mode_tx: flume::unbounded().0,
                        session_id: SessionRef::from(input.spawn.claim.id()),
                        permissions: Arc::new(PermissionManager::new(
                            maki_config::PermissionsConfig::default(),
                            // The requested session cwd, like production:
                            // tool contexts built from this handle serve
                            // their own project, never the process cwd.
                            input.spawn.cwd.clone(),
                            ProjectConfig::for_project(&input.spawn.cwd),
                            Arc::default(),
                        )),
                        task: AbortOnDropHandle::new(tokio::spawn(async { SessionEnd::default() })),
                        spend: Arc::default(),
                    };
                    Ok(SessionRuntime {
                        handle,
                        mcp: None,
                        current_mode: input.spawn.mode.clone(),
                        current_model: spec,
                        // Retained from the activation request, like the real
                        // `spawn_runtime` path: the runtime serves this cwd.
                        cwd: input.spawn.cwd.clone(),
                        output: input.spawn.output.clone(),
                        pending: Arc::default(),
                        claim: input.spawn.claim,
                        pump: tokio::spawn(async {}),
                        // Fixture stubs own no supervised pump (activation
                        // fixtures settle the result, not the stream); the
                        // supervisor is a no-op handle, never a loss source.
                        supervisor: tokio::spawn(async {}),
                    })
                })
            },
        )
    }

    fn server_with_harness(
        harness: &StarterHarness,
    ) -> (
        Server,
        flume::Receiver<Value>,
        flume::Receiver<Incoming>,
        // Keepalive: the server holds only a weak completion link (so tests
        // never inherit the loop-liveness deadlock), and the weak upgrades
        // only while this sender lives.
        flume::Sender<Incoming>,
    ) {
        let (out_tx, out_rx) = flume::unbounded();
        let (completion_tx, completion_rx) = flume::unbounded();
        let server = Server {
            out_tx,
            model_specs: Vec::new(),
            client_elicits_form: false,
            defaults: SessionDefaults::default(),
            on_session_end: None,
            sessions: HashMap::new(),
            allocations: HashMap::new(),
            activating: HashMap::new(),
            starter: fixture_starter(harness),
            completion: completion_tx.downgrade(),
            release_tasks: JoinSet::new(),
            release_records: ReleaseRecords::default(),
        };
        (server, out_rx, completion_rx, completion_tx)
    }

    fn allocate_request(cwd: &str) -> Value {
        serde_json::json!({ "params": { "cwd": cwd } })
    }

    fn activate_request(session: &str, model: &str, mode: &str, cwd: &str, output: &str) -> Value {
        serde_json::json!({
            "params": {
                "sessionId": session,
                "model": model,
                "mode": mode,
                "cwd": cwd,
                "output": output,
            }
        })
    }

    /// Next completion from an activation task, with the deadlock bound. A
    /// timeout here is the loop blocked on slow startup.
    async fn next_completion(
        rx: &flume::Receiver<Incoming>,
    ) -> (String, RequestId, ActivationResult) {
        let incoming = timeout(STDIN_DEADLOCK_TIMEOUT, rx.recv_async())
            .await
            .expect("an activation task must report back without the loop")
            .expect("the completion channel stays open");
        match incoming {
            Incoming::ActivationDone { key, id, outcome } => (key, id, *outcome),
            // No model discovery runs in these fixtures, so any other
            // variant is a bug in the test, not timing.
            _ => panic!("expected an activation completion"),
        }
    }

    /// Allocate through the real dispatch and read the leased id.
    async fn allocate_id(
        srv: &mut Server,
        out_rx: &flume::Receiver<Value>,
        params: &AcpParams,
        cwd: &std::path::Path,
        id: RequestId,
    ) -> String {
        handle_request(
            srv,
            extension::ALLOCATE_METHOD,
            id.clone(),
            &allocate_request(&cwd.to_string_lossy()),
            params,
        )
        .await;
        let reply = out_rx.try_recv().expect("allocate answers synchronously");
        assert_eq!(reply["id"], serde_json::to_value(&id).unwrap());
        assert_eq!(reply["result"]["turnEpoch"], 0);
        reply["result"]["sessionId"]
            .as_str()
            .expect("a leased id")
            .to_owned()
    }

    /// Rewire a fixture runtime's channels to live ones, so prompts and
    /// cancels addressed to it are observable. Returns the receivers.
    fn rewire_live(
        runtime: &mut SessionRuntime,
    ) -> (flume::Receiver<AgentInput>, flume::Receiver<()>) {
        let (input_tx, input_rx) = flume::unbounded();
        let (cancel_tx, cancel_rx) = flume::unbounded();
        runtime.handle.input_tx = input_tx;
        runtime.handle.cancel_tx = cancel_tx;
        (input_rx, cancel_rx)
    }

    /// Allocate reserves an (id, cwd) lease with no provider and no task: the
    /// claim is held, nothing was constructed, and the sessions table is
    /// untouched.
    #[tokio::test]
    async fn allocate_reserves_a_lease_without_provider_or_task() {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage.clone());

        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;

        assert!(srv.sessions.is_empty(), "allocation spawns no runtime");
        assert!(
            harness.started.lock().unwrap().is_empty(),
            "allocation constructs no provider"
        );
        let id: MakiId = session.parse().expect("a Maki id");
        assert!(
            SessionClaim::is_held(id, &storage),
            "the allocation holds its storage lease"
        );
    }

    #[test_case("relative/project" ; "a_relative_cwd")]
    #[tokio::test]
    async fn allocate_rejects_a_cwd_that_is_not_an_absolute_directory(relative: &str) {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let params = acp_params(storage);

        handle_request(
            &mut srv,
            extension::ALLOCATE_METHOD,
            RequestId::Number(1),
            &allocate_request(relative),
            &params,
        )
        .await;
        let reply = out_rx.try_recv().expect("allocate answers synchronously");
        assert_eq!(reply["error"]["code"], -32602);
        assert_eq!(reply["error"]["data"]["reason"], "invalid_cwd");
        assert!(
            srv.allocations.is_empty(),
            "a rejected allocate leases nothing"
        );
    }

    /// A prompt, mode change, or config change naming an unactivated
    /// allocation is rejected before anything starts: no provider is
    /// constructed, no task spawns, and the lease is untouched.
    #[tokio::test]
    async fn operations_on_an_unactivated_allocation_reject_without_starting_anything() {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage.clone());
        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;

        let err = handle_prompt(
            &mut srv,
            &prompt_request(&session, "hello"),
            &RequestId::Number(2),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::invalid_params().code);
        assert_eq!(
            err.data,
            Some(serde_json::json!({"reason": "not_activated", "sessionId": session}))
        );
        for raw in [mode_request(&session, "plan"), config_request(&session)] {
            let err = if raw["params"].get("modeId").is_some() {
                handle_set_mode(&srv, &raw).unwrap_err()
            } else {
                handle_set_config(&srv, &raw).unwrap_err()
            };
            assert_eq!(
                err.data,
                Some(serde_json::json!({"reason": "not_activated", "sessionId": session}))
            );
        }

        assert!(
            harness.started.lock().unwrap().is_empty(),
            "a rejected prompt constructs no provider"
        );
        assert!(srv.sessions.is_empty(), "a rejected prompt spawns no task");
        let id: MakiId = session.parse().unwrap();
        assert!(
            SessionClaim::is_held(id, &storage),
            "rejected operations keep the lease"
        );
        assert!(out_rx.is_empty(), "rejections emit no session updates");
    }

    /// Activation constructs each session with its requested model directly:
    /// the starter log (construction order) names each requested spec once,
    /// each runtime files that same spec, and the default `params.model` is
    /// never constructed unless requested.
    #[tokio::test]
    async fn activation_constructs_each_session_with_its_requested_model() {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, completion_rx, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let projects = [TempDir::new().unwrap(), TempDir::new().unwrap()];
        let params = acp_params(storage);
        let specs = ["openai/gpt-5", "openai/gpt-5.6-sol"];
        assert_ne!(specs[0], specs[1]);

        let mut sessions = Vec::new();
        for (n, project) in projects.iter().enumerate() {
            sessions.push((
                specs[n].to_owned(),
                allocate_id(
                    &mut srv,
                    &out_rx,
                    &params,
                    project.path(),
                    RequestId::Number(n as i64 + 1),
                )
                .await,
            ));
        }
        for (n, ((spec, session), project)) in sessions.iter().zip(projects.iter()).enumerate() {
            let output = project.path().join("plan.md");
            handle_request(
                &mut srv,
                extension::ACTIVATE_METHOD,
                RequestId::Number(n as i64 + 10),
                &activate_request(
                    session,
                    spec,
                    "build",
                    &project.path().to_string_lossy(),
                    &output.to_string_lossy(),
                ),
                &params,
            )
            .await;
        }
        // Completions can land in any order; each is settled under its own
        // key, and each runtime must file the model its own session asked for.
        let expected: HashMap<&str, &str> = sessions
            .iter()
            .map(|(spec, session)| (session.as_str(), spec.as_str()))
            .collect();
        for _ in 0..sessions.len() {
            let (key, id, outcome) = next_completion(&completion_rx).await;
            settle_activation(&mut srv, key.clone(), id, outcome);
            let reply = out_rx.try_recv().expect("activation settles its result");
            assert_eq!(reply["result"]["sessionId"], key);
            assert_eq!(reply["result"]["turnEpoch"], 0);
            let lifecycle = out_rx.try_recv().expect("activation announces active");
            assert_eq!(lifecycle["method"], extension::LIFECYCLE_NOTIFICATION);
            assert_eq!(lifecycle["params"]["sessionId"], key);
            assert_eq!(lifecycle["params"]["state"], "active");
            assert_eq!(lifecycle["params"]["turnEpoch"], 0);
            assert_eq!(lifecycle["params"]["usable"], true);
            let runtime = srv.sessions.get(&key).expect("the runtime is filed");
            assert_eq!(&runtime.current_model, expected[&key.as_str()]);
            assert_eq!(runtime.current_mode, AgentMode::Build);
            assert!(runtime.output.is_some(), "activation fixes the output");
        }

        let started = harness.started.lock().unwrap().clone();
        assert_eq!(
            started
                .iter()
                .map(|(_, spec)| spec.as_str())
                .collect::<Vec<_>>(),
            specs,
            "each session is constructed once, with its requested model"
        );
        assert_eq!(srv.allocations.len(), 0, "activation consumes the lease");
        assert!(
            srv.activating.is_empty(),
            "settling clears the in-flight entry"
        );
    }

    /// 02.4 initial session context: interleaved sessions in fixture projects
    /// A/B keep their own requested cwd, model, mode, and absolute output.
    /// Each runtime retains its own cwd; neither session inherits peer context
    /// or the launch directory; the Maki process cwd never moves; and the
    /// fixed setup is observable before the first prompt (mode/config changes
    /// route to the addressed session and reject). No prompt is ever sent, so
    /// nothing reaches a provider: no keys, no inference, no network.
    #[tokio::test]
    async fn interleaved_sessions_keep_their_own_initial_context() {
        let process_cwd = std::env::current_dir().expect("the process cwd is readable");
        let harness = StarterHarness::default();
        let (mut srv, out_rx, completion_rx, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project_a = TempDir::new().unwrap();
        let project_b = TempDir::new().unwrap();
        // Distinct project markers: a process-global read (launch dir, peer
        // project, or a shared default) can only produce one answer, so two
        // correct per-session answers exclude it.
        std::fs::write(project_a.path().join("AGENTS.md"), "alpha project rules").unwrap();
        std::fs::write(project_b.path().join("AGENTS.md"), "beta project rules").unwrap();
        let params = acp_params(storage);
        let cwd_a = project_a.path().to_string_lossy().into_owned();
        let cwd_b = project_b.path().to_string_lossy().into_owned();
        assert_ne!(cwd_a, cwd_b);

        let session_a = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project_a.path(),
            RequestId::Number(1),
        )
        .await;
        let session_b = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project_b.path(),
            RequestId::Number(2),
        )
        .await;
        assert_ne!(session_a, session_b);

        let output_a = project_a.path().join("plan-a.md");
        let output_b = project_b.path().join("plan-b.md");
        // Interleaved activation: A starts while B is still unactivated, so a
        // shared slot would visibly leak.
        for (n, (session, cwd, output, model, mode)) in [
            (&session_a, &cwd_a, &output_a, "openai/gpt-5", "build"),
            (&session_b, &cwd_b, &output_b, "openai/gpt-5.6-sol", "plan"),
        ]
        .iter()
        .enumerate()
        {
            handle_request(
                &mut srv,
                extension::ACTIVATE_METHOD,
                RequestId::Number(n as i64 + 10),
                &activate_request(session, model, mode, cwd, &output.to_string_lossy()),
                &params,
            )
            .await;
        }
        for _ in 0..2 {
            let (key, id, outcome) = next_completion(&completion_rx).await;
            settle_activation(&mut srv, key, id, outcome);
            let reply = out_rx.try_recv().expect("activation settles its result");
            assert_eq!(reply["result"]["turnEpoch"], 0);
            let lifecycle = out_rx.try_recv().expect("activation announces active");
            assert_eq!(lifecycle["params"]["state"], "active");
        }

        // Each runtime retains its own requested cwd, model, mode, and output.
        let runtime_a = srv.sessions.get(&session_a).expect("A is filed");
        let runtime_b = srv.sessions.get(&session_b).expect("B is filed");
        assert_eq!(runtime_a.cwd, project_a.path());
        assert_eq!(runtime_b.cwd, project_b.path());
        assert_eq!(runtime_a.current_model, "openai/gpt-5");
        assert_eq!(runtime_b.current_model, "openai/gpt-5.6-sol");
        assert_eq!(runtime_a.current_mode, AgentMode::Build);
        assert!(
            matches!(runtime_b.current_mode, AgentMode::Plan(_)),
            "B runs in plan mode: {:?}",
            runtime_b.current_mode
        );
        assert_eq!(runtime_a.output.as_deref(), Some(output_a.as_path()));
        assert_eq!(runtime_b.output.as_deref(), Some(output_b.as_path()));

        // Construction saw the same per-session values: the very first
        // provider connection for each session names its requested model, and
        // the requested cwd/mode/output rode along before any prompt.
        let mut built = harness.built.lock().unwrap().clone();
        built.sort_by(|x, y| x.session.cmp(&y.session));
        let mut expected = vec![
            BuiltActivation {
                session: session_a.clone(),
                model: "openai/gpt-5".to_owned(),
                mode: format!("{:?}", AgentMode::Build),
                cwd: project_a.path().to_path_buf(),
                output: output_a.clone(),
            },
            BuiltActivation {
                session: session_b.clone(),
                model: "openai/gpt-5.6-sol".to_owned(),
                mode: format!("{:?}", runtime_b.current_mode),
                cwd: project_b.path().to_path_buf(),
                output: output_b.clone(),
            },
        ];
        expected.sort_by(|x, y| x.session.cmp(&y.session));
        assert_eq!(built, expected, "construction keeps each session's context");

        // Fixed setup is observable before the first prompt: changes route to
        // the addressed session and reject without touching the peer.
        for (session, mode) in [(&session_a, "plan"), (&session_b, "build")] {
            let reply = handle_set_mode(
                &srv,
                &serde_json::json!({
                    "params": { "sessionId": session, "modeId": mode }
                }),
            )
            .unwrap_err();
            assert_eq!(
                reply.data,
                Some(serde_json::json!({
                    "reason": extension::FIXED_AT_ACTIVATION_REASON
                })),
                "{session}: the mode is fixed before the first prompt"
            );
        }
        let unknown = format!("{session_a}-missing");
        let no_prompt = handle_prompt(
            &mut srv,
            &prompt_request(&unknown, "hello"),
            &RequestId::Number(7),
        )
        .unwrap_err();
        assert_eq!(no_prompt.code, AcpError::resource_not_found(None).code);
        assert_eq!(srv.sessions.len(), 2, "rejections touch no runtime");

        assert_eq!(
            std::env::current_dir().expect("the process cwd is readable"),
            process_cwd,
            "activation never moves the Maki process cwd"
        );
        assert!(
            !process_cwd.starts_with(project_a.path())
                && !process_cwd.starts_with(project_b.path()),
            "neither session inherits the launch directory: {process_cwd:?}"
        );
    }

    /// An activation that fails fast (unparsable or disallowed model, bad
    /// mode, mismatched cwd, relative output) consumes its lease and answers
    /// without starting anything: the id is unknown afterwards, and no task
    /// or provider was ever constructed.
    #[tokio::test]
    async fn failed_activation_releases_the_lease_before_answering() {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage.clone());
        let cwd = project.path().to_string_lossy().into_owned();

        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;
        let output = project
            .path()
            .join("plan.md")
            .to_string_lossy()
            .into_owned();
        let bad_model = activate_request(&session, "nope/not-a-provider", "build", &cwd, &output);
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            RequestId::Number(2),
            &bad_model,
            &params,
        )
        .await;
        let reply = out_rx.try_recv().expect("a fast failure answers at once");
        assert_eq!(reply["error"]["code"], -32602);
        assert_eq!(reply["error"]["data"]["model"], "nope/not-a-provider");

        assert!(srv.allocations.is_empty(), "the failure consumes the lease");
        assert!(srv.activating.is_empty(), "no startup ever left the loop");
        assert!(srv.sessions.is_empty(), "no runtime is filed");
        assert!(
            harness.started.lock().unwrap().is_empty(),
            "validation runs before any construction"
        );
        let id: MakiId = session.parse().unwrap();
        assert!(!SessionClaim::is_held(id, &storage), "the claim is dropped");
        let err = handle_prompt(
            &mut srv,
            &prompt_request(&session, "hello"),
            &RequestId::Number(3),
        )
        .unwrap_err();
        assert_eq!(err.code, AcpError::resource_not_found(None).code);
    }

    #[test_case("build", "openai/gpt-5", "plan.md", "relative/output.md" ; "relative_output")]
    #[test_case("build", "openai/gpt-5", "elsewhere", "plan.md" ; "mismatched_cwd")]
    #[test_case("write", "openai/gpt-5", "plan.md", "plan.md" ; "unknown_mode")]
    #[tokio::test]
    async fn activation_validates_mode_cwd_and_output(
        mode: &str,
        model: &str,
        cwd_leaf: &str,
        output_leaf: &str,
    ) {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let elsewhere = TempDir::new().unwrap();
        let params = acp_params(storage.clone());
        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;
        let cwd = if cwd_leaf == "elsewhere" {
            elsewhere.path().to_string_lossy().into_owned()
        } else {
            project.path().to_string_lossy().into_owned()
        };
        let output = if output_leaf == "relative/output.md" {
            output_leaf.to_owned()
        } else {
            project
                .path()
                .join(output_leaf)
                .to_string_lossy()
                .into_owned()
        };
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            RequestId::Number(2),
            &activate_request(&session, model, mode, &cwd, &output),
            &params,
        )
        .await;
        let reply = out_rx.try_recv().expect("a fast failure answers at once");
        assert_eq!(reply["error"]["code"], -32602);
        assert!(srv.allocations.is_empty(), "the failure consumes the lease");
        let id: MakiId = session.parse().unwrap();
        assert!(!SessionClaim::is_held(id, &storage), "the claim is dropped");
    }

    /// Pre-activation half of the handoff test: slow startup fails before the
    /// activation result settles it once, and settling the same completion
    /// again answers nothing. The failure is confined to its session: the
    /// peer runtime is untouched and still routes prompts. Post-activation
    /// loss belongs to 02.10a and is not built here.
    #[tokio::test]
    async fn slow_startup_failure_settles_once_and_leaves_the_peer_usable() {
        let harness = StarterHarness::default();
        harness
            .fail
            .lock()
            .unwrap()
            .replace("provider exploded".to_owned());
        let (mut srv, out_rx, completion_rx, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage.clone());
        // A live peer runtime with observable channels.
        let (peer_key, mut peer) = fixture_runtime();
        let (_peer_input, _peer_cancel) = rewire_live(&mut peer);
        srv.sessions.insert(peer_key, peer);

        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;
        let output = project
            .path()
            .join("plan.md")
            .to_string_lossy()
            .into_owned();
        let cwd = project.path().to_string_lossy().into_owned();
        let request_id = RequestId::Number(2);
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            request_id.clone(),
            &activate_request(&session, "openai/gpt-5", "build", &cwd, &output),
            &params,
        )
        .await;
        assert!(
            out_rx.is_empty(),
            "slow startup answers nothing synchronously"
        );

        let (key, id, outcome) = next_completion(&completion_rx).await;
        assert_eq!(key, session);
        let outcome2 = match &outcome {
            Err(failure) => Err(ActivationFailure {
                claim: failure.claim.clone(),
                reason: failure.reason.clone(),
            }),
            Ok(_) => panic!("the starter was armed to fail"),
        };
        settle_activation(&mut srv, key.clone(), id.clone(), outcome);
        let reply = out_rx.try_recv().expect("the failure settles the result");
        assert_eq!(reply["id"], serde_json::to_value(&id).unwrap());
        assert_eq!(reply["error"]["code"], -32603);
        assert_eq!(reply["error"]["data"]["reason"], "provider exploded");
        assert_eq!(reply["error"]["data"]["sessionId"], session);

        // The pre-activation half of the handoff: settling the same
        // completion again must not answer a second time.
        settle_activation(&mut srv, key, id, outcome2);
        assert!(out_rx.is_empty(), "a second settle answers nothing");

        assert_eq!(srv.sessions.len(), 1, "the failure files no runtime");
        // The peer runtime is untouched: its prompt still routes.
        let peer_key = srv.sessions.keys().next().unwrap().clone();
        handle_prompt(
            &mut srv,
            &prompt_request(&peer_key, "hello"),
            &RequestId::Number(3),
        )
        .unwrap();
        let failed_id: MakiId = session.parse().unwrap();
        assert!(
            !SessionClaim::is_held(failed_id, &storage),
            "the failed lease is dropped"
        );
    }

    /// Slow startup parks on a gate while the loop keeps routing: a peer
    /// cancel dispatches and a peer prompt routes before the held activation
    /// completes. Full permission-answer routing is 02.8; this proves the
    /// dispatch path never waits on startup.
    #[tokio::test]
    async fn held_startup_leaves_peer_cancel_routing_responsive() {
        let harness = StarterHarness::default();
        harness.hold.store(true, Ordering::SeqCst);
        let (mut srv, out_rx, completion_rx, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage);
        let (peer_key, mut peer) = fixture_runtime();
        let (_peer_input, peer_cancel) = rewire_live(&mut peer);
        srv.sessions.insert(peer_key.clone(), peer);

        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;
        let output = project
            .path()
            .join("plan.md")
            .to_string_lossy()
            .into_owned();
        let cwd = project.path().to_string_lossy().into_owned();
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            RequestId::Number(2),
            &activate_request(&session, "openai/gpt-5", "build", &cwd, &output),
            &params,
        )
        .await;
        assert!(out_rx.is_empty(), "held startup answers nothing yet");

        // The peer stays routable while the activation task is parked: its
        // cancel dispatches and its prompt is accepted, both bounded by the
        // deadlock timeout rather than by the held gate.
        handle_notification(
            &srv,
            "session/cancel",
            &serde_json::json!({ "params": { "sessionId": peer_key } }),
        );
        timeout(STDIN_DEADLOCK_TIMEOUT, peer_cancel.recv_async())
            .await
            .expect("peer cancel routes while startup is held")
            .expect("the cancel channel stays open");
        handle_prompt(
            &mut srv,
            &prompt_request(&peer_key, "hello"),
            &RequestId::Number(3),
        )
        .unwrap();

        harness.gate.notify_one();
        let (key, id, outcome) = next_completion(&completion_rx).await;
        settle_activation(&mut srv, key.clone(), id, outcome);
        let reply = out_rx
            .try_recv()
            .expect("releasing the gate settles the result");
        assert_eq!(reply["result"]["sessionId"], session);
        let lifecycle = out_rx.try_recv().expect("activation announces active");
        assert_eq!(lifecycle["params"]["state"], "active");
        assert!(srv.sessions.contains_key(&session));
    }

    /// A client that disappears without explicit failure leaves leases this
    /// slice releases on disconnect: the sweep drops every unactivated claim
    /// and aborts every in-flight startup, and the ids are unknown afterwards.
    #[tokio::test]
    async fn disconnect_releases_abandoned_allocations() {
        let harness = StarterHarness::default();
        harness.hold.store(true, Ordering::SeqCst);
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let projects = [TempDir::new().unwrap(), TempDir::new().unwrap()];
        let params = acp_params(storage.clone());

        let mut sessions = Vec::new();
        for (n, project) in projects.iter().enumerate() {
            sessions.push(
                allocate_id(
                    &mut srv,
                    &out_rx,
                    &params,
                    project.path(),
                    RequestId::Number(n as i64 + 1),
                )
                .await,
            );
        }
        // One activation in flight, parked on the gate.
        let output = projects[0]
            .path()
            .join("plan.md")
            .to_string_lossy()
            .into_owned();
        let cwd = projects[0].path().to_string_lossy().into_owned();
        handle_request(
            &mut srv,
            extension::ACTIVATE_METHOD,
            RequestId::Number(10),
            &activate_request(&sessions[0], "openai/gpt-5", "build", &cwd, &output),
            &params,
        )
        .await;
        assert_eq!(srv.activating.len(), 1);

        close_all_sessions(&mut srv, SessionEndReason::Shutdown).await;

        assert!(srv.allocations.is_empty(), "the sweep drops every lease");
        assert!(srv.activating.is_empty(), "in-flight startup is aborted");
        for session in &sessions {
            let id: MakiId = session.parse().unwrap();
            assert!(
                !SessionClaim::is_held(id, &storage),
                "no claim survives disconnect"
            );
            let err = handle_prompt(
                &mut srv,
                &prompt_request(session, "hello"),
                &RequestId::Number(20),
            )
            .unwrap_err();
            assert_eq!(err.code, AcpError::resource_not_found(None).code);
        }
    }

    /// Explicit release forgets an unactivated allocation: the lease drops and
    /// the id is unknown afterwards. Active-session release defers its answer
    /// to the background task; see the 02.10b release tests.
    #[tokio::test]
    async fn release_forgets_an_unactivated_allocation() {
        let harness = StarterHarness::default();
        let (mut srv, out_rx, _, _keepalive) = server_with_harness(&harness);
        let state = TempDir::new().unwrap();
        let storage = StateDir::from_path(state.path().to_path_buf());
        let project = TempDir::new().unwrap();
        let params = acp_params(storage.clone());
        let session = allocate_id(
            &mut srv,
            &out_rx,
            &params,
            project.path(),
            RequestId::Number(1),
        )
        .await;

        let raw = serde_json::json!({ "params": { "sessionId": session } });
        let result = handle_release(&mut srv, &raw, RequestId::Number(2))
            .expect("allocation release answers synchronously")
            .unwrap();
        assert_eq!(result["released"], true);
        assert_eq!(result["sessionId"], session);
        assert_eq!(result["turnEpoch"], 0);
        let id: MakiId = session.parse().unwrap();
        assert!(
            !SessionClaim::is_held(id, &storage),
            "release drops the lease"
        );

        let repeat = handle_release(&mut srv, &raw, RequestId::Number(3))
            .expect("repeat release answers synchronously")
            .unwrap_err();
        assert_eq!(repeat.code, AcpError::resource_not_found(None).code);
        assert_eq!(repeat.data, Some(serde_json::json!({"sessionId": session})));
        let unknown =
            serde_json::json!({ "params": { "sessionId": MakiId::generate().to_string() } });
        assert_eq!(
            handle_release(&mut srv, &unknown, RequestId::Number(4))
                .expect("unknown release answers synchronously")
                .unwrap_err()
                .code,
            AcpError::resource_not_found(None).code
        );
    }

    /// 02.10b helpers: a release request plus the deferred-outcome collector.
    /// The loop only does the fast prologue; the background task reports, so
    /// every release test collects the answer and lifecycle off `out_rx` with
    /// the deadlock bound instead of hanging on them.
    const ASK_RELEASE_A: i64 = 2001;
    const ASK_RELEASE_B: i64 = 2011;
    const RELEASE_ID: i64 = 50;

    fn release_request(session: &str) -> Value {
        serde_json::json!({
            "params": { "sessionId": session, "reason": "client_done" }
        })
    }

    fn park_prompt_for(srv: &Server, key: &str, id: RequestId) {
        let mut guard = srv.sessions.get(key).unwrap().pending.lock().unwrap();
        guard.turn = Some(guard.epoch.current());
        guard.prompt = Some(id);
    }

    fn park_permission_ask_for(srv: &Server, key: &str, wire_id: i64) -> flume::Receiver<String> {
        let (answer_tx, answer_rx) = flume::unbounded();
        let epoch = srv
            .sessions
            .get(key)
            .unwrap()
            .pending
            .lock()
            .unwrap()
            .epoch
            .current();
        srv.sessions
            .get(key)
            .unwrap()
            .pending
            .lock()
            .unwrap()
            .asks
            .insert(
                wire_id,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: Some(answer_tx),
                    session: key.to_owned(),
                    epoch,
                },
            );
        answer_rx
    }

    fn answer_for(wire_id: i64) -> Value {
        serde_json::json!({
            "id": wire_id,
            "result": { "outcome": { "outcome": "selected", "optionId": "allow_once" } },
        })
    }

    /// Drains `out_rx` until the deferred release answer for `id` and the
    /// first lifecycle frame for `session` both arrive. Returns every frame
    /// seen, so tests can also assert what never arrived.
    async fn await_release_outcome(
        out_rx: &flume::Receiver<Value>,
        id: RequestId,
        session: &str,
    ) -> Vec<Value> {
        let want_id = serde_json::to_value(&id).unwrap();
        let mut seen = Vec::new();
        let mut answered = false;
        let mut lifecycled = false;
        timeout(STDIN_DEADLOCK_TIMEOUT, async {
            while !answered || !lifecycled {
                let frame = out_rx.recv_async().await.expect("release always reports");
                if frame.get("id") == Some(&want_id) {
                    answered = true;
                }
                if frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
                    && frame["params"]["sessionId"] == *session
                {
                    lifecycled = true;
                }
                seen.push(frame);
            }
        })
        .await
        .expect("bounded teardown always reports");
        seen
    }

    fn fixture_claims() -> StateDir {
        StateDir::from_path(FIXTURE_CLAIMS.path().to_path_buf())
    }

    async fn await_claim_dropped(id: MakiId, storage: &StateDir) {
        timeout(STDIN_DEADLOCK_TIMEOUT, async {
            while SessionClaim::is_held(id, storage) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("release drops the claim");
    }

    /// 02.10b acceptance, working session: Close on A while B continues under
    /// the same PID. A's turn settles through the 02.9a cancel shape and its
    /// asks retire; only A's runtime ends; slow teardown is bounded and off
    /// the loop; the deferred answer and `released` lifecycle report exactly
    /// once; the claim drops; B stays responsive throughout. No network,
    /// keys, or providers are involved.
    #[tokio::test]
    async fn release_working_session_cancels_retires_and_reports_while_peer_continues() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        let (_input_a_rx, cancel_a_rx) = rewire_live(srv.sessions.get_mut(&key_a).unwrap());
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        assert_ne!(key_a, key_b, "two sessions hold distinct ids");

        // Both sessions work a turn; A also waits on a permission ask.
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        park_prompt_for(&srv, &key_b, RequestId::Number(8));
        let answer_a_rx = park_permission_ask_for(&srv, &key_a, ASK_RELEASE_A);
        let answer_b_rx = park_permission_ask_for(&srv, &key_b, ASK_RELEASE_B);
        // A's run is wedged in background work: the abort in the prologue
        // still reaps it promptly — no task ever stalls a release, and the
        // shared process is never killed to close one session.
        only_mut_addr(&mut srv, &key_a).handle.task =
            AbortOnDropHandle::new(tokio::spawn(future::pending()));

        // Close is deferred: the loop does the fast prologue and moves on.
        let deferred = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID),
        );
        assert!(
            deferred.is_none(),
            "the background task owns the answer, not the loop"
        );
        assert!(
            !srv.sessions.contains_key(&key_a),
            "Close rejects new input at once"
        );
        assert!(
            srv.sessions.contains_key(&key_b),
            "only the addressed runtime ends"
        );
        assert!(
            cancel_a_rx.try_recv().is_ok(),
            "the addressed Close signals A's run, never the shared process"
        );

        let frames = await_release_outcome(&out_rx, RequestId::Number(RELEASE_ID), &key_a).await;
        let answer = frames
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(PROMPT_ID)))
            .expect("A's prompt settles once");
        assert_eq!(answer["result"]["stopReason"], "cancelled");
        assert_eq!(
            answer["result"]["_meta"][extension::TURN_EPOCH_META_KEY],
            0,
            "the answer settles the turn it ran under: {answer}"
        );
        assert_eq!(
            answer["result"]["_meta"][extension::RAW_REASON_META_KEY],
            "cancelled"
        );
        assert_eq!(answer["result"]["_meta"][extension::USABLE_META_KEY], false);
        let result = frames
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(RELEASE_ID)))
            .expect("the deferred release answers");
        assert_eq!(result["result"]["sessionId"], key_a);
        assert_eq!(result["result"]["released"], true);
        assert_eq!(result["result"]["turnEpoch"], 1);
        let lifecycle = frames
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
            })
            .expect("teardown reports a lifecycle");
        assert_eq!(lifecycle["params"]["sessionId"], key_a);
        assert_eq!(lifecycle["params"]["state"], "released");
        assert_eq!(lifecycle["params"]["turnEpoch"], 1);
        assert_eq!(lifecycle["params"]["rawReason"], "client_done");
        assert_eq!(lifecycle["params"]["usable"], false);
        assert!(
            lifecycle["params"].get("priorLoss").is_none(),
            "a live release reports no prior loss: {lifecycle}"
        );
        assert!(
            !frames
                .iter()
                .any(|frame| frame["params"]["state"] == "release_failed"),
            "no failure is claimed on a clean teardown: {frames:?}"
        );

        // A late answer for the retired ask drops without reaching any agent.
        handle_incoming_response(&srv, &answer_for(ASK_RELEASE_A));
        assert!(answer_a_rx.is_empty(), "a late answer cannot revive A");
        assert!(answer_b_rx.is_empty(), "nothing lands on the peer");
        // New input for A rejects; re-release re-answers the recorded success.
        for err in [
            handle_prompt(
                &mut srv,
                &prompt_request(&key_a, "retry"),
                &RequestId::Number(51),
            )
            .unwrap_err(),
            handle_cancel(
                &srv,
                &serde_json::json!({ "params": { "sessionId": key_a } }),
            )
            .unwrap_err(),
        ] {
            assert_eq!(err.code, AcpError::resource_not_found(None).code);
        }
        let repeat = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID + 1),
        )
        .expect("recorded releases answer synchronously")
        .unwrap();
        assert_eq!(repeat["released"], true);
        assert_eq!(repeat["turnEpoch"], 1);

        let id_a: MakiId = key_a.parse().unwrap();
        await_claim_dropped(id_a, &fixture_claims()).await;

        // B runs undisturbed and settles under its own epoch.
        drive_runtime(&srv, &key_b, |sender| {
            sender.send(done_event()).unwrap();
        })
        .await;
        let settled = out_rx.try_recv().expect("B settles its own prompt");
        assert_eq!(settled["id"], 8);
        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays usable through a sibling's release");
    }

    /// 02.10b control serialization, structural half: Close's settle runs
    /// through the same per-session lock as Answer claims and Stop settles.
    /// Racing threads prove the invariant: every wire id claims at most once,
    /// the prompt settles exactly once across the Stop and the Close, the
    /// Close finds no live turn left, and every settle after the release mark
    /// changes nothing — no bump, no take, no revival.
    #[test]
    fn close_queues_behind_in_flight_cancel_settlement() {
        use std::sync::Barrier;

        const ASKS: i64 = 32;
        let pending = PendingState::default();
        {
            let mut guard = pending.lock().unwrap();
            guard.turn = Some(guard.epoch.current());
            guard.prompt = Some(RequestId::Number(PROMPT_ID));
            for n in 0..ASKS {
                guard.asks.insert(
                    ANSWERED_ID + n,
                    Ask::Permission {
                        request_id: format!("toolu_{n}"),
                        answer_tx: None,
                        session: "s".to_owned(),
                        epoch: 0,
                    },
                );
            }
            guard.asks.insert(
                ANSWERED_ID + ASKS,
                Ask::Permission {
                    request_id: PARENT_TOOL_USE_ID.to_owned(),
                    answer_tx: None,
                    session: "other".to_owned(),
                    epoch: 0,
                },
            );
        }

        let barrier = Arc::new(Barrier::new(5));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let control = Arc::clone(&pending);
            let gate = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                gate.wait();
                let mut won = Vec::new();
                for n in 0..=ASKS {
                    if control
                        .lock()
                        .unwrap()
                        .claim_ask(ANSWERED_ID + n, "s")
                        .is_some()
                    {
                        won.push(n);
                    }
                }
                won
            }));
        }
        barrier.wait();
        // The Stop lands while answers race it; the Close queues behind the
        // Stop on the same guard and marks released atomically with its own
        // settle, exactly like the release prologue.
        let (stop, close) = {
            let mut guard = pending.lock().unwrap();
            let stop = guard.settle_cancel();
            let close = guard.settle_cancel();
            guard.mark_released();
            (stop, close)
        };

        let mut won: Vec<i64> = Vec::new();
        for handle in handles {
            won.extend(handle.join().expect("no control-lock deadlock"));
        }
        won.sort_unstable();
        let mut dedup = won.clone();
        dedup.dedup();
        assert_eq!(
            won, dedup,
            "every wire id claims at most once, however raced"
        );
        assert!(
            !won.contains(&ASKS),
            "a cross-session ask never claims for this session"
        );
        assert!(stop.cancelled, "the first settle takes the live turn");
        assert_eq!(stop.prompt, Some(RequestId::Number(PROMPT_ID)));
        assert!(
            !close.cancelled,
            "the queued Close finds no live turn left to settle"
        );
        assert_eq!(close.prompt, None);
        assert_eq!(stop.epoch, 1, "one Stop bumps exactly once");
        assert_eq!(close.epoch, 2, "the queued Close bumps behind it");
        assert!(pending.lock().unwrap().is_released());
        assert!(pending.lock().unwrap().asks.is_empty());

        // Everything after the mark is a late frame: no bump, no take, no
        // revival — the released outcome can never change.
        let epoch = pending.lock().unwrap().epoch.current();
        let late_cancel = pending.lock().unwrap().settle_cancel();
        assert!(!late_cancel.cancelled);
        assert_eq!(late_cancel.prompt, None);
        let late_loss = pending.lock().unwrap().settle_loss("task_dead", "x");
        assert!(!late_loss.first);
        assert!(
            pending
                .lock()
                .unwrap()
                .claim_ask(ANSWERED_ID, "s")
                .is_none()
        );
        assert_eq!(
            pending.lock().unwrap().epoch.current(),
            epoch,
            "late settles never move the released epoch"
        );
        assert!(
            !pending.lock().unwrap().is_resumable(),
            "a released handle never reopens"
        );
    }

    /// 02.10b hung-hook half: a session-end hook that never resolves bounds
    /// the release instead of stalling it, and the failure is visible. Peers
    /// prove the routing loop never waited: while A's teardown is held, B's
    /// cancel routes and B's permission answer delivers — both land before
    /// A's `release_failed` lifecycle does. No keys, network, or providers.
    #[tokio::test]
    async fn hung_hook_release_is_bounded_visible_and_peer_responsive() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        let answer_a_rx = park_permission_ask_for(&srv, &key_a, ASK_RELEASE_A);
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        park_prompt_for(&srv, &key_b, RequestId::Number(8));
        let answer_b_rx = park_permission_ask_for(&srv, &key_b, ASK_RELEASE_B);

        let gate = Arc::new(Notify::new());
        let held = Arc::clone(&gate);
        srv.on_session_end = Some(Arc::new(move |_, _| {
            let held = Arc::clone(&held);
            Box::pin(async move {
                held.notified().await;
            })
        }));

        let deferred = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID),
        );
        assert!(deferred.is_none(), "teardown never runs on the loop");

        // Peer work routes while the hook teardown is held: both land before
        // A's lifecycle does (the release cannot finish before the hook
        // bound, so an arrived lifecycle here would mean the loop stalled).
        // The answer goes first: B's ask is bound to the current epoch and
        // the cancel below bumps it.
        handle_incoming_response(&srv, &answer_for(ASK_RELEASE_B));
        assert_eq!(
            answer_b_rx.try_recv().ok(),
            Some(TaggedAnswer::new(PARENT_TOOL_USE_ID, PermissionAnswer::AllowOnce).encode()),
            "the peer answer delivers while A's teardown is held"
        );
        let cancel = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_b } }),
        )
        .unwrap();
        assert_eq!(cancel["cancelled"], true);
        // Drained, not dropped: B's cancel settled its prompt while held, and
        // A's prompt answer (prologue) is in here too.
        let drained: Vec<Value> = out_rx.try_iter().collect();
        assert!(
            drained
                .iter()
                .all(|frame| frame["params"]["state"] != "release_failed"),
            "A's outcome cannot arrive before the hook bound"
        );
        let peer_settled = drained
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(8)))
            .expect("B's prompt settles while A's teardown is held");
        assert_eq!(peer_settled["result"]["stopReason"], "cancelled");
        let owed = drained
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(PROMPT_ID)))
            .expect("A's prompt is answered in the prologue");
        assert_eq!(owed["result"]["stopReason"], "cancelled");

        let frames = await_release_outcome(&out_rx, RequestId::Number(RELEASE_ID), &key_a).await;
        let result = frames
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(RELEASE_ID)))
            .expect("the deferred release answers with its failure");
        assert_eq!(result["error"]["code"], -32603);
        assert_eq!(result["error"]["data"]["reason"], "release_failed");
        assert_eq!(result["error"]["data"]["stage"], "hook");
        assert_eq!(result["error"]["data"]["rawReason"], "hook_timeout");
        assert_eq!(result["error"]["data"]["usable"], false);
        let lifecycle = frames
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
            })
            .expect("the failure is visible as a lifecycle");
        assert_eq!(lifecycle["params"]["state"], "release_failed");
        assert_eq!(lifecycle["params"]["stage"], "hook");
        assert_eq!(lifecycle["params"]["usable"], false);
        assert!(
            !frames
                .iter()
                .any(|frame| frame["params"]["state"] == "released"
                    && frame["params"]["sessionId"] == key_a),
            "no successful release is claimed after a failed teardown: {frames:?}"
        );

        // Late answers still drop; new input still rejects; re-release stays
        // honestly failed instead of flipping to success.
        handle_incoming_response(&srv, &answer_for(ASK_RELEASE_A));
        assert!(answer_a_rx.is_empty(), "a late answer cannot revive A");
        for err in [
            handle_prompt(
                &mut srv,
                &prompt_request(&key_a, "retry"),
                &RequestId::Number(51),
            )
            .unwrap_err(),
            handle_cancel(
                &srv,
                &serde_json::json!({ "params": { "sessionId": key_a } }),
            )
            .unwrap_err(),
        ] {
            assert_eq!(err.code, AcpError::resource_not_found(None).code);
        }
        let repeat = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID + 1),
        )
        .expect("recorded releases answer synchronously")
        .unwrap_err();
        assert_eq!(repeat.data.unwrap()["reason"], "release_failed");

        let id_a: MakiId = key_a.parse().unwrap();
        await_claim_dropped(id_a, &fixture_claims()).await;
        // B cancelled its own turn mid-test, so its 02.9b gate is undisturbed
        // by the sibling's failed release: still held, still explicit.
        let retry = handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .unwrap_err();
        assert_eq!(retry.data.unwrap()["reason"], "cleanup_unconfirmed");
    }

    /// 02.10b held-MCP half: an MCP `Shutdown` the server never acks bounds
    /// the release at the MCP stage and reports it distinctly. The held
    /// command proves the teardown targeted the per-session handle from the
    /// 02.5a inventory, and peers stay responsive while it is held.
    #[tokio::test]
    async fn held_mcp_release_is_bounded_visible_and_peer_responsive() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        let (held_mcp, cmd_rx) = maki_agent::mcp::test_support::held_handle();
        only_mut(&mut srv).mcp = Some(held_mcp);
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        park_prompt_for(&srv, &key_b, RequestId::Number(8));
        let answer_b_rx = park_permission_ask_for(&srv, &key_b, ASK_RELEASE_B);

        let deferred = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID),
        );
        assert!(deferred.is_none(), "teardown never runs on the loop");

        // The background task targets A's own MCP handle first (joins are
        // prompt post-abort), while the peer keeps routing.
        let cmd = timeout(STDIN_DEADLOCK_TIMEOUT, cmd_rx.recv_async())
            .await
            .expect("teardown targets the per-session MCP handle")
            .expect("the command channel stays open");
        assert!(
            matches!(cmd, maki_agent::mcp::McpCommand::Shutdown { .. }),
            "teardown shuts the session MCP down, nothing else"
        );
        // The peer answer goes first: its ask is bound to the current epoch
        // and the cancel below bumps it.
        handle_incoming_response(&srv, &answer_for(ASK_RELEASE_B));
        assert!(
            answer_b_rx.try_recv().is_ok(),
            "the peer answer delivers while MCP teardown is held"
        );
        let cancel = handle_cancel(
            &srv,
            &serde_json::json!({ "params": { "sessionId": key_b } }),
        )
        .unwrap();
        assert_eq!(cancel["cancelled"], true);

        let frames = await_release_outcome(&out_rx, RequestId::Number(RELEASE_ID), &key_a).await;
        let result = frames
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(RELEASE_ID)))
            .expect("the deferred release answers with its failure");
        assert_eq!(result["error"]["data"]["reason"], "release_failed");
        assert_eq!(result["error"]["data"]["stage"], "mcp");
        assert_eq!(result["error"]["data"]["rawReason"], "mcp_shutdown_timeout");
        let lifecycle = frames
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
            })
            .expect("the failure is visible as a lifecycle");
        assert_eq!(lifecycle["params"]["state"], "release_failed");
        assert_eq!(lifecycle["params"]["stage"], "mcp");
        assert!(
            !frames
                .iter()
                .any(|frame| frame["params"]["state"] == "released"
                    && frame["params"]["sessionId"] == key_a),
            "no successful release is claimed after a failed teardown: {frames:?}"
        );

        let repeat = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID + 1),
        )
        .expect("recorded releases answer synchronously")
        .unwrap_err();
        assert_eq!(repeat.data.unwrap()["reason"], "release_failed");
        let id_a: MakiId = key_a.parse().unwrap();
        await_claim_dropped(id_a, &fixture_claims()).await;
    }

    /// 02.10b late-frame rejection: after a clean release, permission answers
    /// and terminal/loss events for the released session are dropped — they
    /// cannot revive the handle, change its outcome, or affect the peer.
    #[tokio::test]
    async fn late_answers_and_terminals_are_rejected_after_release() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        let answer_a_rx = park_permission_ask_for(&srv, &key_a, ASK_RELEASE_A);
        let pending_a = Arc::clone(&srv.sessions.get(&key_a).unwrap().pending);
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);
        let epoch_before = pending_a.lock().unwrap().epoch.current();

        assert!(
            handle_release(
                &mut srv,
                &release_request(&key_a),
                RequestId::Number(RELEASE_ID),
            )
            .is_none()
        );
        await_release_outcome(&out_rx, RequestId::Number(RELEASE_ID), &key_a).await;
        let framed = out_rx.try_iter().count();

        // A late permission answer for the retired ask is dropped.
        handle_incoming_response(&srv, &answer_for(ASK_RELEASE_A));
        assert!(answer_a_rx.is_empty(), "late answers never reach an agent");
        // Late terminal and loss events settle/send nothing.
        assert_eq!(settle_prompt(&pending_a), None);
        assert!(!report_loss(&pending_a, &srv.out_tx, &key_a, "task_dead"));
        assert_eq!(
            pending_a.lock().unwrap().epoch.current(),
            epoch_before + 1,
            "only the release prologue moved the epoch"
        );
        assert!(pending_a.lock().unwrap().is_released());
        assert_eq!(
            out_rx.try_iter().count(),
            0,
            "no late frame is emitted after release (had {framed} at release)"
        );
        // New addressed input rejects; the peer is untouched.
        for err in [
            handle_prompt(
                &mut srv,
                &prompt_request(&key_a, "retry"),
                &RequestId::Number(51),
            )
            .unwrap_err(),
            handle_cancel(
                &srv,
                &serde_json::json!({ "params": { "sessionId": key_a } }),
            )
            .unwrap_err(),
        ] {
            assert_eq!(err.code, AcpError::resource_not_found(None).code);
        }
        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays usable after a sibling's release");
    }

    /// 02.10b supervised-dead release: releasing a `lost` handle is bounded,
    /// awaits the dead handles, and reports the prior loss plus the release
    /// cleanup result — with the claim dropped and the handle never revived.
    /// A second leg proves a failing MCP teardown on a dead handle still
    /// reports the prior loss alongside its own failure.
    #[tokio::test]
    async fn supervised_dead_release_reports_prior_loss_and_cleanup() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        let (key_b, mut runtime_b) = fixture_runtime();
        let (_input_b_rx, _cancel_b_rx) = rewire_live(&mut runtime_b);
        srv.sessions.insert(key_b.clone(), runtime_b);

        // Injected task death, observed pre-release like the 02.10a path.
        let pending_a = Arc::clone(&srv.sessions.get(&key_a).unwrap().pending);
        assert!(
            report_loss(&pending_a, &srv.out_tx, &key_a, "task_dead"),
            "the first loss observation reports"
        );
        let lost = out_rx.try_recv().expect("loss notifies `lost`");
        assert_eq!(lost["params"]["state"], "lost");

        assert!(
            handle_release(
                &mut srv,
                &release_request(&key_a),
                RequestId::Number(RELEASE_ID),
            )
            .is_none(),
            "even a dead handle releases off-loop"
        );
        let frames = await_release_outcome(&out_rx, RequestId::Number(RELEASE_ID), &key_a).await;
        let result = frames
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(RELEASE_ID)))
            .expect("the deferred release answers");
        assert_eq!(result["result"]["released"], true);
        assert_eq!(result["result"]["turnEpoch"], 1);
        let lifecycle = frames
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
                    && frame["params"]["state"] == "released"
                    && frame["params"]["sessionId"] == key_a
            })
            .expect("release reports cleanup after the prior loss");
        assert_eq!(lifecycle["params"]["turnEpoch"], 1);
        assert_eq!(
            lifecycle["params"]["priorLoss"],
            serde_json::json!({"reason": "task_lost", "rawReason": "task_dead"}),
            "the release carries the recorded prior loss: {lifecycle}"
        );
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame["params"]["state"] == "lost")
                .count(),
            0,
            "no second `lost` is emitted by the release path (it predates it)"
        );

        // The handle stays failed/non-resumable: never revived, never reused.
        assert!(!report_loss(
            &pending_a,
            &srv.out_tx,
            &key_a,
            "stream_ended"
        ));
        assert_eq!(settle_prompt(&pending_a), None);
        assert!(pending_a.lock().unwrap().is_lost());
        assert!(pending_a.lock().unwrap().is_released());
        for err in [
            handle_prompt(
                &mut srv,
                &prompt_request(&key_a, "retry"),
                &RequestId::Number(51),
            )
            .unwrap_err(),
            handle_cancel(
                &srv,
                &serde_json::json!({ "params": { "sessionId": key_a } }),
            )
            .unwrap_err(),
        ] {
            assert_eq!(err.code, AcpError::resource_not_found(None).code);
        }
        let repeat = handle_release(
            &mut srv,
            &release_request(&key_a),
            RequestId::Number(RELEASE_ID + 1),
        )
        .expect("recorded releases answer synchronously")
        .unwrap();
        assert_eq!(repeat["released"], true, "success stays idempotent");
        let id_a: MakiId = key_a.parse().unwrap();
        await_claim_dropped(id_a, &fixture_claims()).await;

        // Second leg: a dead handle whose MCP teardown also fails reports
        // both — the prior loss and the cleanup failure, never `released`.
        let (key_c, mut runtime_c) = fixture_runtime();
        let (_held_c, cmd_c_rx) = maki_agent::mcp::test_support::held_handle();
        runtime_c.mcp = Some(_held_c);
        srv.sessions.insert(key_c.clone(), runtime_c);
        let pending_c = Arc::clone(&srv.sessions.get(&key_c).unwrap().pending);
        assert!(report_loss(&pending_c, &srv.out_tx, &key_c, "pump_dead"));
        let lost_c = out_rx.try_recv().expect("loss notifies `lost`");
        assert_eq!(lost_c["params"]["sessionId"], key_c);
        assert!(
            handle_release(
                &mut srv,
                &release_request(&key_c),
                RequestId::Number(RELEASE_ID + 2),
            )
            .is_none()
        );
        let _ = timeout(STDIN_DEADLOCK_TIMEOUT, cmd_c_rx.recv_async())
            .await
            .expect("teardown targets the dead session's MCP handle");
        let frames_c =
            await_release_outcome(&out_rx, RequestId::Number(RELEASE_ID + 2), &key_c).await;
        let failed = frames_c
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
                    && frame["params"]["sessionId"] == key_c
            })
            .expect("a dead handle with failing cleanup still reports");
        assert_eq!(failed["params"]["state"], "release_failed");
        assert_eq!(failed["params"]["stage"], "mcp");
        assert_eq!(
            failed["params"]["priorLoss"],
            serde_json::json!({"reason": "task_lost", "rawReason": "pump_dead"}),
            "the failure still carries the prior loss: {failed}"
        );

        handle_prompt(
            &mut srv,
            &prompt_request(&key_b, "b-next"),
            &RequestId::Number(10),
        )
        .expect("the peer stays usable through dead-handle releases");
    }

    /// 02.10b shutdown: EOF releases everything — every runtime with a
    /// visible per-session result, every unactivated lease through the 02.3
    /// drop-claim helper, every in-flight startup aborted — and a slow
    /// teardown on one session cannot strand its peer.
    #[tokio::test]
    async fn shutdown_releases_everything_with_visible_results() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        let (key_b, runtime_b) = fixture_runtime();
        srv.sessions.insert(key_b.clone(), runtime_b);
        // One unactivated lease plus one startup in flight, like a client
        // that disappeared mid-activation.
        let lease_storage = TempDir::new().unwrap();
        let lease_state = StateDir::from_path(lease_storage.path().to_path_buf());
        let lease_claim = SessionClaim::fresh(&lease_state);
        let lease_id = lease_claim.id();
        srv.allocations.insert(
            "lease-id".to_owned(),
            Allocation {
                cwd: PathBuf::from("/project"),
                claim: lease_claim,
            },
        );
        srv.activating.insert(
            "activating-id".to_owned(),
            tokio::spawn(future::pending::<()>()),
        );

        timeout(
            STDIN_DEADLOCK_TIMEOUT,
            close_all_sessions(&mut srv, SessionEndReason::Shutdown),
        )
        .await
        .expect("shutdown stays bounded");
        assert!(srv.sessions.is_empty(), "every runtime is released");
        assert!(srv.allocations.is_empty(), "every lease is released");
        assert!(srv.activating.is_empty(), "every startup is aborted");
        let id_a: MakiId = key_a.parse().unwrap();
        let id_b: MakiId = key_b.parse().unwrap();
        assert!(
            !SessionClaim::is_held(id_a, &fixture_claims()),
            "A's claim drops"
        );
        assert!(
            !SessionClaim::is_held(id_b, &fixture_claims()),
            "B's claim drops"
        );
        assert!(
            !SessionClaim::is_held(lease_id, &lease_state),
            "unactivated leases drop through the 02.3 helper"
        );

        let frames: Vec<Value> = out_rx.try_iter().collect();
        for key in [&key_a, &key_b] {
            let lifecycle = frames
                .iter()
                .find(|frame| {
                    frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
                        && frame["params"]["sessionId"] == *key
                })
                .unwrap_or_else(|| panic!("shutdown reports {key}: {frames:?}"));
            assert_eq!(lifecycle["params"]["state"], "released", "{lifecycle}");
            assert_eq!(lifecycle["params"]["rawReason"], "shutdown", "{lifecycle}");
        }
        let owed = frames
            .iter()
            .find(|frame| frame.get("id") == Some(&serde_json::json!(PROMPT_ID)))
            .expect("the owed prompt is answered, never hung");
        assert_eq!(owed["result"]["stopReason"], "cancelled");
    }

    /// 02.10b shutdown under fault: a hung hook plus a held MCP teardown on
    /// one session bounds the whole sweep and stays visible per session — the
    /// failed session reports `release_failed` while its peer still reports
    /// `released`, and every claim still drops.
    #[tokio::test]
    async fn shutdown_with_held_teardown_is_bounded_and_visible() {
        let (mut srv, _answer_rx, _out_rx) = test_server();
        let (out_tx, out_rx) = flume::unbounded();
        srv.out_tx = out_tx;
        let key_a: String = srv.sessions.keys().next().unwrap().clone();
        park_prompt_for(&srv, &key_a, RequestId::Number(PROMPT_ID));
        let (held_mcp, _cmd_rx) = maki_agent::mcp::test_support::held_handle();
        only_mut(&mut srv).mcp = Some(held_mcp);
        let gate = Arc::new(Notify::new());
        let held = Arc::clone(&gate);
        // The hook hangs only the addressed session; the peer's own
        // invocation returns at once. A global hook that hung for every
        // session would fail every release — honestly, since it is part of
        // each teardown — but this test needs the peer distinction.
        let victim: MakiId = key_a.parse().unwrap();
        srv.on_session_end = Some(Arc::new(move |id, _| {
            let held = Arc::clone(&held);
            Box::pin(async move {
                if id == victim {
                    held.notified().await;
                }
            })
        }));
        let (key_b, runtime_b) = fixture_runtime();
        srv.sessions.insert(key_b.clone(), runtime_b);

        timeout(
            STDIN_DEADLOCK_TIMEOUT,
            close_all_sessions(&mut srv, SessionEndReason::Shutdown),
        )
        .await
        .expect("a held teardown cannot stall shutdown");
        assert!(srv.sessions.is_empty(), "every runtime is released");

        let frames: Vec<Value> = out_rx.try_iter().collect();
        let failed = frames
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
                    && frame["params"]["sessionId"] == key_a
            })
            .expect("the held session reports its failure");
        assert_eq!(failed["params"]["state"], "release_failed", "{failed:?}");
        assert_eq!(failed["params"]["stage"], "hook", "{failed:?}");
        let peer = frames
            .iter()
            .find(|frame| {
                frame.get("method") == Some(&Value::from(extension::LIFECYCLE_NOTIFICATION))
                    && frame["params"]["sessionId"] == key_b
            })
            .expect("the peer still reports its own outcome");
        assert_eq!(peer["params"]["state"], "released", "{peer:?}");
        let id_a: MakiId = key_a.parse().unwrap();
        let id_b: MakiId = key_b.parse().unwrap();
        assert!(!SessionClaim::is_held(id_a, &fixture_claims()));
        assert!(!SessionClaim::is_held(id_b, &fixture_claims()));
    }
}

#[cfg(test)]
mod iso_tests {
    use super::iso_from_epoch;
    use test_case::test_case;

    #[test_case(0, None ; "unknown_sentinel_is_no_timestamp")]
    #[test_case(1_700_000_000, Some("2023-11-14T22:13:20Z") ; "epoch_seconds_to_utc")]
    #[test_case(u64::MAX, None ; "out_of_range_drops_the_timestamp")]
    fn iso_from_epoch_converts(secs: u64, expected: Option<&str>) {
        assert_eq!(iso_from_epoch(secs).as_deref(), expected);
    }
}
