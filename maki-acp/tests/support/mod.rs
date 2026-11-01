//! Deterministic child-process harness for the fork multi-session ACP
//! extension (slice 02.1). Later slices (02.2a-02.10b) reuse these helpers
//! instead of reinventing harnesses.
//!
//! Shape of every fixture here:
//! - The Maki child under test is spawned exactly like the maintained 01
//!   helper launches it: `maki --background-timeout 0 acp --model
//!   openai/gpt-5`, with a cleared environment carrying only the documented
//!   minimum (HOME, PATH, TMPDIR, NO_COLOR, XDG_* roots under an isolated
//!   fixture home). No provider keys, no `.env` loader settings, no developer
//!   XDG paths are inherited. `--background-timeout 0` is blocking mode, so
//!   detached background tools are disabled for every later slice.
//! - Fake provider/client only: the client is this harness, and no prompt is
//!   ever sent, so the provider wire is untouched and no inference, keys or
//!   network are needed. Provider-side fakes arrive with the slices that
//!   need prompts.
//! - Synchronisation is read-until-match on requests/events with generous
//!   timeouts as failure backstops, never sleeps. Responses are correlated
//!   by JSON-RPC id (numbers and strings); notifications buffer separately
//!   so a peer session's traffic never unblocks the wrong waiter.

// Shared across integration-test binaries where each uses a subset (02.1
// fixtures use the two-PID helpers, 02.2a+ the single-PID ones), so
// per-binary unused items are expected reuse surface, not dead code.
#![allow(dead_code)]

use std::collections::HashMap;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tempfile::TempDir;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::ChildStderr;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::timeout;

/// Explicit model the child boots with. The base accepts it without checking
/// provider readiness, so startup needs no keys and no network.
pub const MODEL_SPEC: &str = "openai/gpt-5";
/// ACP protocol version the fixtures speak.
pub const PROTOCOL_VERSION: u64 = 1;
/// Agent identity the fixtures assert on initialize.
pub const AGENT_NAME: &str = "maki";
/// Pinned agent version, same as the 01 smoke helper.
pub const AGENT_VERSION: &str = "0.7.0";
/// Bound for one initialize reply, same as the 01 helper.
pub const INITIALIZE_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound for any later correlated reply.
pub const FRAME_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound for EOF shutdown, same as the 01 helper.
pub const EXIT_TIMEOUT: Duration = Duration::from_secs(10);
/// Bound for reaping a hung child, same as the 01 helper.
pub const KILL_TIMEOUT: Duration = Duration::from_secs(5);
/// Bound for draining a pipe that no longer advances, same as the 01 helper.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_FRAME_BYTES: u64 = 1024 * 1024;
const STDERR_TAIL_BYTES: usize = 64 * 1024;
const BINARY_ENV: &str = "MAKI_UNDER_TEST";
const NO_BINARY: &str = "cannot find the maki binary under test: build it first with \
    `rustup run 1.99.0 cargo build -p maki --locked`, or set MAKI_UNDER_TEST to its path";

/// Isolated roots for one child: its HOME/XDG dirs and its fixture cwd.
/// Mirrors the 01 helper's layout.
pub struct ChildFixture {
    _tmp: TempDir,
    /// Fixture root; the child's TMPDIR.
    pub root: PathBuf,
    /// Isolated HOME; XDG roots live under it.
    pub home: PathBuf,
    /// Child working directory at spawn.
    pub workspace: PathBuf,
}

impl ChildFixture {
    pub fn create(prefix: &str) -> io::Result<Self> {
        let tmp = TempDir::with_prefix(prefix)?;
        let root = tmp.path().to_path_buf();
        let home = root.join("home");
        let workspace = root.join("workspace");
        for path in [&home, &workspace] {
            std::fs::create_dir_all(path)?;
        }
        for relative in [".config", ".local/share", ".local/state", ".cache"] {
            std::fs::create_dir_all(home.join(relative))?;
        }
        Ok(Self {
            _tmp: tmp,
            root,
            home,
            workspace,
        })
    }
}

/// Two fixture projects with distinct cwds for A/B session fixtures.
pub struct ProjectPair {
    _tmp: TempDir,
    /// First project cwd.
    pub a: PathBuf,
    /// Second project cwd.
    pub b: PathBuf,
}

impl ProjectPair {
    pub fn create() -> io::Result<Self> {
        let tmp = TempDir::with_prefix("maki-acp-projects-")?;
        let a = tmp.path().join("project-a");
        let b = tmp.path().join("project-b");
        for path in [&a, &b] {
            std::fs::create_dir_all(path)?;
        }
        Ok(Self { _tmp: tmp, a, b })
    }
}

fn maki_binary() -> PathBuf {
    if let Some(path) = std::env::var_os(BINARY_ENV) {
        return PathBuf::from(path);
    }
    let exe = std::env::current_exe().expect("test executable path");
    for ancestor in exe.ancestors() {
        if ancestor.file_name().is_some_and(|name| name == "deps")
            && let Some(bin) = ancestor.parent()
        {
            let candidate = bin.join(format!("maki{}", std::env::consts::EXE_SUFFIX));
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    panic!("{NO_BINARY}");
}

/// One Maki child under test plus its JSON-RPC framing. argv always carries
/// `--background-timeout 0` (blocking mode); see [`MakiChild::args`].
pub struct MakiChild {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr_log: Arc<Mutex<(u64, VecDeque<u8>)>>,
    stderr_task: Option<JoinHandle<io::Result<()>>>,
    argv: Vec<OsString>,
    next_id: u64,
    _fixture: ChildFixture,
    /// Responses that arrived while another id was awaited (peer traffic on
    /// a shared pipe once one process hosts both sessions).
    stashed: HashMap<String, Value>,
    /// Notifications (frames without an id) buffered for later matching.
    notifications: Vec<Value>,
}

impl MakiChild {
    /// Spawn the child in blocking mode under an isolated fixture home.
    pub async fn spawn() -> io::Result<Self> {
        let fixture = ChildFixture::create("maki-acp-child-")?;
        let argv: Vec<OsString> = ["--background-timeout", "0", "acp", "--model", MODEL_SPEC]
            .iter()
            .map(OsString::from)
            .collect();
        let mut command = Command::new(maki_binary());
        command
            .args(&argv)
            .env_clear()
            .env("HOME", &fixture.home)
            .env("PATH", "/usr/bin:/bin")
            .env("TMPDIR", &fixture.root)
            .env("NO_COLOR", "1")
            .env("XDG_CONFIG_HOME", fixture.home.join(".config"))
            .env("XDG_DATA_HOME", fixture.home.join(".local/share"))
            .env("XDG_STATE_HOME", fixture.home.join(".local/state"))
            .env("XDG_CACHE_HOME", fixture.home.join(".cache"))
            // 02.10a supervision: the supervised task connects its provider
            // at startup, so without a key every activation reports `lost`
            // (honest provider-init failure) and routing fixtures that never
            // prompt would see `resource_not_found` instead of the routing
            // rejection they prove. The dummy key keeps construction offline
            // (no prompt is ever sent from these fixtures, so no network and
            // no inference occur); supervision itself is proven via injected
            // fixtures in `server.rs` with no keys and no network.
            .env("OPENAI_API_KEY", "test-dummy-no-network")
            .current_dir(&fixture.workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stderr_log = Arc::new(Mutex::new((
            0_u64,
            VecDeque::with_capacity(STDERR_TAIL_BYTES),
        )));
        let stderr_task = Some(tokio::spawn(drain_stderr(
            stderr.expect("child stderr"),
            Arc::clone(&stderr_log),
        )));
        let peer = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout.expect("child stdout")),
            stderr_log,
            stderr_task,
            argv,
            next_id: 1,
            _fixture: fixture,
            stashed: HashMap::new(),
            notifications: Vec::new(),
        };
        Ok(peer)
    }

    /// argv the child was spawned with. The blocking-mode check asserts on this.
    pub fn args(&self) -> &[OsString] {
        &self.argv
    }

    /// Next fresh numeric request id.
    pub fn fresh_id(&mut self) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        Value::from(id)
    }

    /// Send one JSON-RPC frame.
    pub async fn send(&mut self, frame: &Value) -> io::Result<()> {
        let mut bytes = serde_json::to_vec(frame).expect("frame serializes");
        bytes.push(b'\n');
        let stdin = self.stdin.as_mut().expect("child stdin");
        stdin.write_all(&bytes).await?;
        stdin.flush().await
    }

    /// Send a request with explicit id and params.
    pub async fn send_request(&mut self, method: &str, id: Value, params: Value) -> io::Result<()> {
        self.send(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await
    }

    /// Read one raw frame with the frame bound. Oversized or unterminated
    /// frames are errors, same as the 01 helper.
    pub async fn next_frame(&mut self) -> io::Result<Value> {
        let mut frame = Vec::new();
        let count = timeout(
            FRAME_TIMEOUT,
            (&mut self.stdout)
                .take(MAX_FRAME_BYTES + 1)
                .read_until(b'\n', &mut frame),
        )
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "no ACP frame arrived"))?
        .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e))?;
        if count == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "ACP EOF"));
        }
        if count as u64 > MAX_FRAME_BYTES || frame.last() != Some(&b'\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "incomplete or oversized ACP frame",
            ));
        }
        serde_json::from_slice(&frame).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Correlated reply for `id`, buffering anything else. Never sleeps:
    /// every wait is a read-until-match with the frame bound.
    pub async fn expect_response(&mut self, id: &Value) -> io::Result<Value> {
        let key = id.to_string();
        if let Some(reply) = self.stashed.remove(&key) {
            return Ok(reply);
        }
        loop {
            let frame = self.next_frame().await?;
            match frame.get("id") {
                Some(frame_id) if frame_id == id => return Ok(frame),
                Some(frame_id) => {
                    self.stashed.insert(frame_id.to_string(), frame);
                }
                None => self.notifications.push(frame),
            }
        }
    }

    /// Next buffered or incoming notification satisfying `want`.
    pub async fn expect_notification(
        &mut self,
        want: impl Fn(&Value) -> bool,
    ) -> io::Result<Value> {
        if let Some(index) = self.notifications.iter().position(&want) {
            return Ok(self.notifications.remove(index));
        }
        loop {
            let frame = self.next_frame().await?;
            if frame.get("id").is_none() {
                if want(&frame) {
                    return Ok(frame);
                }
                self.notifications.push(frame);
            } else {
                let key = frame["id"].to_string();
                self.stashed.insert(key, frame);
            }
        }
    }

    /// Initialize handshake: correlated reply, protocol 1, maki identity.
    pub async fn initialize(&mut self) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "initialize",
            id.clone(),
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {"name": "maki-acp-harness", "version": "0.1.0"},
            }),
        )
        .await?;
        let reply = timeout(INITIALIZE_TIMEOUT, self.expect_response(&id))
            .await
            .map_err(|_| {
                io::Error::new(io::ErrorKind::TimedOut, "initialize exceeded 30 seconds")
            })??;
        assert_no_error(&reply);
        assert_eq!(reply["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(reply["result"]["agentInfo"]["name"], AGENT_NAME);
        assert_eq!(reply["result"]["agentInfo"]["version"], AGENT_VERSION);
        Ok(reply)
    }

    /// Open a session on `cwd`. Returns the fork-assigned session id.
    pub async fn session_new(&mut self, cwd: &str) -> io::Result<String> {
        let id = self.fresh_id();
        self.send_request(
            "session/new",
            id.clone(),
            serde_json::json!({"cwd": cwd, "mcpServers": []}),
        )
        .await?;
        let reply = self.expect_response(&id).await?;
        assert_no_error(&reply);
        reply["result"]["sessionId"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "session/new hid its sessionId")
            })
    }

    /// Stored sessions, newest first, optionally filtered to one cwd.
    pub async fn session_list(&mut self, cwd: Option<&str>) -> io::Result<Value> {
        let id = self.fresh_id();
        let mut params = serde_json::json!({});
        if let Some(cwd) = cwd {
            params["cwd"] = Value::from(cwd);
        }
        self.send_request("session/list", id.clone(), params)
            .await?;
        let reply = self.expect_response(&id).await?;
        assert_no_error(&reply);
        Ok(reply["result"].clone())
    }

    /// Switch a session's mode. The reply plus the following
    /// `session/update` notification let fixtures observe notifications
    /// without any provider contact. Only valid while the server applies
    /// mode changes; 02.2b fixes setup per session, so 02.2b fixtures use
    /// [`MakiChild::set_mode_raw`] and assert the rejection instead.
    pub async fn set_mode(&mut self, session_id: &str, mode: &str) -> io::Result<Value> {
        let reply = self.set_mode_raw(session_id, mode).await?;
        assert_no_error(&reply);
        Ok(reply["result"].clone())
    }

    /// Switch a session's mode without asserting success. 02.2b rejects with
    /// `invalid_params`/`fixed_at_activation`, so the reply itself is the
    /// assertion surface; correlation by id still holds via
    /// `expect_response`.
    pub async fn set_mode_raw(&mut self, session_id: &str, mode: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "session/set_mode",
            id.clone(),
            serde_json::json!({"sessionId": session_id, "modeId": mode}),
        )
        .await?;
        self.expect_response(&id).await
    }

    /// Set a session config option without asserting success (see
    /// [`MakiChild::set_mode_raw`]: the rejection is the assertion surface).
    pub async fn set_config_raw(&mut self, session_id: &str, value: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "session/set_config_option",
            id.clone(),
            serde_json::json!({"sessionId": session_id, "configId": "model", "value": value}),
        )
        .await?;
        self.expect_response(&id).await
    }

    /// Send a prompt without asserting success. Only used against sessions
    /// that reject before any run exists (unknown ids): a prompt to a live
    /// session would reach the provider, which fixtures never do.
    pub async fn prompt_raw(&mut self, session_id: &str, text: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "session/prompt",
            id.clone(),
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": text }],
            }),
        )
        .await?;
        self.expect_response(&id).await
    }

    /// 02.9b scaffolding for 02.11's controlled live probes: send a
    /// `session/prompt` without waiting for its answer and return the id to
    /// correlate later with [`MakiChild::expect_response`]. 02.11 drives the
    /// live sequence off this primitive — send, read-until-match the
    /// tool-call update, cancel mid-tool, read the `cancelled` answer plus
    /// the run's terminal, confirm the retry is accepted — so no 02.9b test
    /// calls it (nothing here may reach a provider: no keys, no network).
    /// Process exit or process-group reaping alone never proves an escaped
    /// `setsid` tool died; see `tests/cleanup_probes.rs` for the executable
    /// record 02.11 must not convert into cleanup proof.
    pub async fn send_prompt(&mut self, session_id: &str, text: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "session/prompt",
            id.clone(),
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": text }],
            }),
        )
        .await?;
        Ok(id)
    }

    /// Reserve an id without starting anything (02.3 allocate). The reply
    /// itself is the assertion surface: `{sessionId, turnEpoch}` on success,
    /// a JSON-RPC error otherwise.
    pub async fn allocate_raw(&mut self, cwd: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "_maki/session/allocate",
            id.clone(),
            serde_json::json!({"cwd": cwd}),
        )
        .await?;
        self.expect_response(&id).await
    }

    /// Bind an allocated id to a model, mode, cwd, and absolute output (02.3
    /// activate). The reply is the assertion surface; readiness follows on a
    /// `_maki/session/lifecycle` notification.
    pub async fn activate_raw(
        &mut self,
        session_id: &str,
        model: &str,
        mode: &str,
        cwd: &str,
        output: &str,
    ) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "_maki/session/activate",
            id.clone(),
            serde_json::json!({
                "sessionId": session_id,
                "model": model,
                "mode": mode,
                "cwd": cwd,
                "output": output,
            }),
        )
        .await?;
        self.expect_response(&id).await
    }

    /// Release a session without asserting success (02.3 implements the
    /// unactivated-allocation half; active sessions stay the 02.10b seam).
    pub async fn release_raw(&mut self, session_id: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(
            "_maki/session/release",
            id.clone(),
            serde_json::json!({"sessionId": session_id, "reason": "client_done"}),
        )
        .await?;
        self.expect_response(&id).await
    }

    /// Addressed turn cancel (02.9a `_maki/session/cancel`). The reply is the
    /// assertion surface: `{sessionId, cancelled, turnEpoch}` on success, a
    /// JSON-RPC error otherwise. `turn_epoch` is the optional pin: omitted
    /// cancels whatever turn runs. No prompt is ever sent here, so the
    /// provider wire stays untouched: no inference, keys or network.
    pub async fn cancel_raw(
        &mut self,
        session_id: &str,
        turn_epoch: Option<u64>,
    ) -> io::Result<Value> {
        let id = self.fresh_id();
        let mut params = serde_json::json!({"sessionId": session_id});
        if let Some(epoch) = turn_epoch {
            params["turnEpoch"] = Value::from(epoch);
        }
        self.send_request("_maki/session/cancel", id.clone(), params)
            .await?;
        self.expect_response(&id).await
    }

    /// Fire a `session/cancel` notification. One-way: a notification carries
    /// no id to answer on, so there is no reply to correlate.
    pub async fn cancel_notify(&mut self, session_id: &str) -> io::Result<()> {
        self.send(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": { "sessionId": session_id },
        }))
        .await
    }

    /// Probe an extension method the current server must reject clearly.
    pub async fn probe_unsupported(&mut self, method: &str) -> io::Result<Value> {
        let id = self.fresh_id();
        self.send_request(method, id.clone(), serde_json::json!({}))
            .await?;
        self.expect_response(&id).await
    }

    /// Close stdin and require a clean exit within the exit bound, then
    /// reap the bounded stderr drain. A hung child is killed and reaped
    /// within the kill bound, same as the 01 helper.
    pub async fn shutdown(mut self) -> io::Result<()> {
        drop(self.stdin.take());
        let status = match timeout(EXIT_TIMEOUT, self.child.wait()).await {
            Ok(status) => status?,
            Err(_) => {
                self.child.start_kill()?;
                timeout(KILL_TIMEOUT, self.child.wait())
                    .await
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::TimedOut, "failed to reap killed Maki child")
                    })??
            }
        };
        if let Some(task) = self.stderr_task.take() {
            let _ = timeout(DRAIN_TIMEOUT, task).await;
        }
        if status.success() {
            return Ok(());
        }
        let mut log = self.stderr_log.lock().await;
        Err(io::Error::other(format!(
            "Maki exited unsuccessfully: {status}; stderr: {}",
            String::from_utf8_lossy(log.1.make_contiguous())
        )))
    }
}

/// Bounded stderr drain, same shape as the 01 helper: every byte counts
/// toward the total while only the tail is kept.
async fn drain_stderr(
    mut stderr: ChildStderr,
    log: Arc<Mutex<(u64, VecDeque<u8>)>>,
) -> io::Result<()> {
    let mut buffer = [0_u8; 8192];
    loop {
        let count = stderr.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        let mut log = log.lock().await;
        log.0 += count as u64;
        for byte in &buffer[..count] {
            if log.1.len() == STDERR_TAIL_BYTES {
                log.1.pop_front();
            }
            log.1.push_back(*byte);
        }
    }
}

fn assert_no_error(reply: &Value) {
    assert!(
        reply.get("error").is_none(),
        "expected success, got: {reply}"
    );
}

/// One initialized session: the child plus its fork-assigned id and cwd.
pub struct SessionPeer {
    pub child: MakiChild,
    pub session_id: String,
    pub cwd: String,
}

impl SessionPeer {
    pub async fn establish(cwd: &std::path::Path) -> io::Result<Self> {
        let mut child = MakiChild::spawn().await?;
        child.initialize().await?;
        let cwd = cwd.to_string_lossy().into_owned();
        let session_id = child.session_new(&cwd).await?;
        Ok(Self {
            child,
            session_id,
            cwd,
        })
    }

    pub async fn shutdown(self) -> io::Result<()> {
        self.child.shutdown().await
    }
}

/// Two synchronised sessions for reuse by 02.2a-02.10b. Each session is its
/// own child PID, so this struct also covers cross-process sync checks.
/// Single-process coexistence (one PID owning both sessions, 02.2a onward)
/// uses [`SharedSessions`] below instead.
pub struct TwoSession {
    /// Distinct fixture project cwds backing the two sessions.
    pub projects: ProjectPair,
    pub a: SessionPeer,
    pub b: SessionPeer,
}

impl TwoSession {
    /// Bring both sessions up deterministically: each peer is initialized
    /// and holds its own session id before either proceeds, so no fixture
    /// step depends on a sleep or on the other's timing.
    pub async fn establish() -> io::Result<Self> {
        let projects = ProjectPair::create()?;
        let a = SessionPeer::establish(&projects.a).await?;
        let b = SessionPeer::establish(&projects.b).await?;
        assert_ne!(
            a.session_id, b.session_id,
            "two sessions must hold distinct ids"
        );
        Ok(Self { projects, a, b })
    }

    pub async fn shutdown(self) -> io::Result<()> {
        self.a.shutdown().await?;
        self.b.shutdown().await
    }
}

/// Two synchronised sessions hosted by ONE child PID (02.2a registry): the
/// child is initialized once and holds a session on each fixture project.
/// No prompt is ever sent here, so the provider wire stays untouched: no
/// inference, keys or network. Prompt/event routing proof is 02.2b.
pub struct SharedSessions {
    /// The single Maki child owning both sessions: one process by construction.
    pub child: MakiChild,
    /// Distinct fixture project cwds backing the two sessions.
    pub projects: ProjectPair,
    /// Fork-assigned id of the session on project A.
    pub a: String,
    /// Cwd backing session A.
    pub cwd_a: String,
    /// Fork-assigned id of the session on project B.
    pub b: String,
    /// Cwd backing session B.
    pub cwd_b: String,
}

impl SharedSessions {
    /// Bring both sessions up deterministically: initialize once, then open a
    /// session per project before either proceeds, so no step depends on a
    /// sleep or on the other's timing.
    pub async fn establish() -> io::Result<Self> {
        let projects = ProjectPair::create()?;
        let mut child = MakiChild::spawn().await?;
        child.initialize().await?;
        let cwd_a = projects.a.to_string_lossy().into_owned();
        let cwd_b = projects.b.to_string_lossy().into_owned();
        let a = child.session_new(&cwd_a).await?;
        let b = child.session_new(&cwd_b).await?;
        assert_ne!(a, b, "two sessions must hold distinct ids");
        Ok(Self {
            child,
            projects,
            a,
            cwd_a,
            b,
            cwd_b,
        })
    }

    pub async fn shutdown(self) -> io::Result<()> {
        self.child.shutdown().await
    }
}
