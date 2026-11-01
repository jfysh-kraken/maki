//! Slice 02.11 acceptance: the integrated live-provider proof.
//!
//! What this proves, with one unchanged Maki PID hosting two concurrent
//! sessions end to end:
//! - interleaved live turns on both sessions (tool-executing and
//!   text-only), correlated by id, with neither peer disturbed;
//! - a real tool executed by the provider's script (bash writes a marker
//!   file the harness reads back: the tool result is on disk, not just in
//!   an ACP frame), behind a real `session/request_permission` ask the
//!   harness answers `allow_once`;
//! - mid-stream cancel of a held provider stream plus retry after the
//!   cancelled run's terminal reopens the session (02.9b criterion, live);
//! - mid-tool cancel of a running `sleep`, with the marker that must never
//!   appear afterwards;
//! - a provider refusal/filter code (`content_filter`) reaching ACP
//!   `refusal` with the verbatim raw reason and a reusable handle (02.7,
//!   live);
//! - independent release of each session and EOF shutdown of the same PID.
//!
//! What the provider is, honestly: a deterministic in-process stub speaking
//! OpenAI-compatible SSE over real HTTP on 127.0.0.1 (ephemeral port). Every
//! layer above the model weights is live: real TCP, real HTTP framing, real
//! SSE decode, real agent pumps/tasks, real bash execution, real permission
//! routing, real cancel propagation, real epoch settlement. What is faked
//! is only the inference itself (scripted streams keyed off prompt
//! markers). No API keys exist in this file (the child gets
//! `OPENAI_API_KEY=test-dummy-no-network`), no external network is touched
//! (the child env is cleared; the only reachable host is the loopback
//! stub), and nothing here can bill anyone.
//!
//! Synchronisation is channels/notifications throughout: the stub signals
//! request arrival over `Notify`, the harness correlates every ACP frame by
//! id, and every wait is a bounded read-until-match. The two stated
//! timing-sensitive parts are the 02.9b retry poll (bounded deadline, each
//! attempt is itself a channel wait) and the mid-tool `sleep` window (the
//! tool's own clock, bounded by the drive timeout).
//!
//! Plumbing note: every ACP frame is read through one unified reader
//! ([`Live::next`]) that auto-allows permission asks inline, so nothing is
//! ever buffered invisibly: the support harness's `expect_response`
//! stash/buffer split would hide peer notifications from the waiters that
//! need them, and this proof needs them all.
//!
//! Reused, not rebuilt: the Lua A/B, nested plan-write, and refusal/filter
//! wire suites run as their own targets (see the 02.11 evidence record);
//! this file only drives them live where the gate needs a wire.

mod support;

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::{MakiChild, ProjectPair};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

/// Bound for one correlated drive: every wait inside is a read-until-match,
/// this is only the failure backstop.
const DRIVE_TIMEOUT: Duration = Duration::from_secs(120);
/// Bound for the stub to observe a request the harness just caused.
const STUB_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound for a cancelled session to confirm cleanup and accept a retry.
/// Under the 02.9b 30s grace: a clean agent abort reports in ~seconds, so a
/// deadline lapse here is the finding, not a slow machine.
const RETRY_DEADLINE: Duration = Duration::from_secs(60);
/// The mid-tool probe's own clock. Success (kill on cancel) returns in
/// seconds; the full window only elapses if the tool ignored its cancel.
const SLEEP_SECS: u64 = 30;
/// How long the held stream stays open after the harness cancels. The client
/// is long gone; this only keeps the stub task tidy, and the server handle
/// is aborted at test end either way.
const HOLD_LINGER_SECS: u64 = 120;

/// Scripted OpenAI-compatible SSE stub. Bound to 127.0.0.1 only; counts
/// every request so the test can prove all provider traffic stayed local.
struct Stub {
    port: u16,
    config: Arc<StubConfig>,
    task: tokio::task::JoinHandle<()>,
}

struct StubConfig {
    marker_a: PathBuf,
    marker_sleep: PathBuf,
    chat_total: AtomicUsize,
    hold_total: AtomicUsize,
    models_total: AtomicUsize,
    hold_arrived: Notify,
    log: std::sync::Mutex<Vec<String>>,
}

impl Stub {
    async fn start(marker_a: PathBuf, marker_sleep: PathBuf) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let config = Arc::new(StubConfig {
            marker_a,
            marker_sleep,
            chat_total: AtomicUsize::new(0),
            hold_total: AtomicUsize::new(0),
            models_total: AtomicUsize::new(0),
            hold_arrived: Notify::new(),
            log: std::sync::Mutex::new(Vec::new()),
        });
        let task_config = Arc::clone(&config);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let config = Arc::clone(&task_config);
                tokio::spawn(async move {
                    serve_conn(stream, &config).await;
                });
            }
        });
        Ok(Self { port, config, task })
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn chats(&self) -> usize {
        self.config.chat_total.load(Ordering::SeqCst)
    }

    fn holds(&self) -> usize {
        self.config.hold_total.load(Ordering::SeqCst)
    }

    fn models(&self) -> usize {
        self.config.models_total.load(Ordering::SeqCst)
    }

    fn stop(self) {
        self.task.abort();
    }
}

fn sse_body(chunks: &[Value]) -> Vec<u8> {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str("data: ");
        out.push_str(&chunk.to_string());
        out.push_str("\n\n");
    }
    out.push_str("data: [DONE]\n\n");
    out.into_bytes()
}

fn text_stream(text: &str) -> Vec<u8> {
    sse_body(&[
        json!({"choices": [{"delta": {"content": text}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ])
}

fn refusal_stream() -> Vec<u8> {
    // The verbatim recorded wire code for a filtered chat-completions turn.
    sse_body(&[json!({"choices": [{"delta": {}, "finish_reason": "content_filter"}]})])
}

fn tool_stream(tag: &str, command: &str) -> Vec<u8> {
    let arguments = json!({"command": command, "description": "live stub probe"}).to_string();
    sse_body(&[
        json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": format!("call_live_{tag}"),
             "function": {"name": "bash", "arguments": arguments}}
        ]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ])
}

/// Route one chat request. A genuine follow-up (tool results back in
/// history, no newer user message) closes the loop with final text; every
/// new turn runs the script for its latest user message. History carries
/// every earlier marker AND every earlier tool result, so both decisions
/// are positional: a user message after the last tool result is a new turn
/// even when the transcript holds old tool traffic.
fn chat_body(body: &str, config: &StubConfig) -> ChatRoute {
    let last_user = last_user_text(body);
    let tag = if last_user.contains("TOOL_PROBE_A") {
        "A"
    } else if last_user.contains("SLEEP_PROBE") {
        "S"
    } else if last_user.contains("TEXT_PROBE_B") {
        "B"
    } else if last_user.contains("TEXT_PROBE_A") {
        "A"
    } else {
        "?"
    };
    if is_followup(body) {
        return ChatRoute::Reply(text_stream(&format!("LIVE_TOOL_DONE_{tag}")));
    }
    if last_user.contains("HOLD_PROBE") {
        return ChatRoute::Hold;
    }
    if last_user.contains("REFUSAL_PROBE") {
        return ChatRoute::Reply(refusal_stream());
    }
    if last_user.contains("TOOL_PROBE_A") {
        let command = format!("printf 'live-tool-A' > '{}'", config.marker_a.display());
        return ChatRoute::Reply(tool_stream("A", &command));
    }
    if last_user.contains("SLEEP_PROBE") {
        let command = format!(
            "sleep {SLEEP_SECS} && printf 'SHOULD_NOT_EXIST' > '{}'",
            config.marker_sleep.display()
        );
        return ChatRoute::Reply(tool_stream("S", &command));
    }
    if last_user.contains("TEXT_PROBE") {
        return ChatRoute::Reply(text_stream(&format!("live-text-{tag} witnessed")));
    }
    ChatRoute::Reply(text_stream("stub-default-done"))
}

/// True when the request answers tool calls rather than starting a turn:
/// a tool result exists and no user message is newer than it. New turns
/// after past tool use carry a user message past the last tool result.
fn is_followup(body: &str) -> bool {
    let Ok(payload) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let Some(messages) = payload.get("messages").and_then(Value::as_array) else {
        return false;
    };
    let mut last_user: Option<usize> = None;
    let mut last_tool: Option<usize> = None;
    for (index, message) in messages.iter().enumerate() {
        match message.get("role").and_then(Value::as_str) {
            Some("user") => last_user = Some(index),
            Some("tool") => last_tool = Some(index),
            _ => {}
        }
    }
    match (last_user, last_tool) {
        (Some(user), Some(tool)) => user < tool,
        _ => false,
    }
}

/// Latest user text in the request, falling back to the whole body when it
/// does not parse (never expected; keeps the stub total).
fn last_user_text(body: &str) -> String {
    let Ok(payload) = serde_json::from_str::<Value>(body) else {
        return body.to_owned();
    };
    let Some(messages) = payload.get("messages").and_then(Value::as_array) else {
        return body.to_owned();
    };
    let mut last = String::new();
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        match message.get("content") {
            Some(Value::String(text)) => last = text.clone(),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    if let Some(piece) = part.get("text").and_then(Value::as_str) {
                        text.push_str(piece);
                    }
                }
                last = text;
            }
            _ => {}
        }
    }
    if last.is_empty() {
        body.to_owned()
    } else {
        last
    }
}

enum ChatRoute {
    Reply(Vec<u8>),
    Hold,
}

async fn serve_conn(stream: TcpStream, config: &StubConfig) {
    let mut reader = BufReader::new(stream);
    let mut head = String::new();
    loop {
        let mut line = String::new();
        let Ok(n) = reader.read_line(&mut line).await else {
            return;
        };
        if n == 0 {
            return;
        }
        head.push_str(&line);
        if head.ends_with("\r\n\r\n") {
            break;
        }
        if head.len() > 64 * 1024 {
            return;
        }
    }
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("").to_owned();
    let mut content_length = 0_usize;
    for line in lines {
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = value.trim().parse().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0_u8; content_length.min(8 * 1024 * 1024)];
    if !body.is_empty() && reader.read_exact(&mut body).await.is_err() {
        return;
    }
    let body_text = String::from_utf8_lossy(&body).into_owned();
    let is_get_models = request_line.starts_with("GET")
        && (request_line.contains("/models") || request_line.contains("models?"));
    let stream = reader.into_inner();

    config.log.lock().unwrap().push(format!(
        "{request_line} chat#{}",
        config.chat_total.load(Ordering::SeqCst)
    ));
    if is_get_models {
        config.models_total.fetch_add(1, Ordering::SeqCst);
        let payload = br#"{"data":[{"id":"gpt-5","object":"model"}]}"#;
        let _ = write_response(stream, "application/json", payload).await;
        return;
    }
    if !request_line.contains("/chat/completions") {
        let payload = br#"{"error":{"message":"stub: unknown path"}}"#;
        let _ = write_raw(stream, "404 Not Found", "application/json", payload).await;
        return;
    }
    config.chat_total.fetch_add(1, Ordering::SeqCst);
    let route_name = match chat_body(&body_text, config) {
        ChatRoute::Reply(_) => "reply",
        ChatRoute::Hold => "hold",
    };
    config.log.lock().unwrap().push(format!(
        "ROUTE {route_name} last_user={:?}",
        last_user_text(&body_text)
    ));
    match chat_body(&body_text, config) {
        ChatRoute::Reply(payload) => {
            let _ = write_response(stream, "text/event-stream", &payload).await;
        }
        ChatRoute::Hold => {
            config.hold_total.fetch_add(1, Ordering::SeqCst);
            // Headers first: the provider request is observably in flight
            // before the harness cancels it.
            let payload = text_stream("held-then-cancelled");
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                payload.len()
            );
            let mut stream = stream;
            if stream.write_all(header.as_bytes()).await.is_err() {
                return;
            }
            if stream.flush().await.is_err() {
                return;
            }
            config.hold_arrived.notify_one();
            tokio::time::sleep(Duration::from_secs(HOLD_LINGER_SECS)).await;
            let _ = stream.write_all(&payload).await;
        }
    }
}

async fn write_response(stream: TcpStream, content_type: &str, payload: &[u8]) -> io::Result<()> {
    write_raw(stream, "200 OK", content_type, payload).await
}

async fn write_raw(
    mut stream: TcpStream,
    status: &str,
    content_type: &str,
    payload: &[u8],
) -> io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(payload).await?;
    stream.flush().await
}

/// One Maki child read through a single unified reader: every frame passes
/// [`Live::next`], which answers `session/request_permission` with
/// `allow_once` inline, so waiters correlate by id and stash the rest.
/// Nothing is ever buffered invisibly.
struct Live {
    child: MakiChild,
    notifications: Vec<Value>,
    permissions_answered: usize,
}

impl Live {
    async fn next(&mut self) -> io::Result<Value> {
        loop {
            let frame = self.child.next_frame().await?;
            if frame.get("method").and_then(Value::as_str) == Some("session/request_permission") {
                let id = frame["id"].clone();
                self.child
                    .send(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
                    }))
                    .await?;
                self.permissions_answered += 1;
                continue;
            }
            return Ok(frame);
        }
    }

    async fn request(&mut self, method: &str, params: Value) -> io::Result<Value> {
        let id = self.child.fresh_id();
        self.child.send_request(method, id.clone(), params).await?;
        let key = id.to_string();
        loop {
            let frame = self.next().await?;
            if frame.get("id").map(ToString::to_string).as_ref() == Some(&key) {
                return Ok(frame);
            }
            self.notifications.push(frame);
        }
    }

    /// Wait for a `_maki/session/lifecycle` frame naming this session and
    /// state; anything else is stashed for later inspection. The stash is
    /// scanned first: an earlier `request` may already have filed it.
    async fn lifecycle(&mut self, session: &str, state: &str) -> io::Result<Value> {
        if let Some(pos) = self.notifications.iter().position(|frame| {
            frame["method"] == "_maki/session/lifecycle"
                && frame["params"]["sessionId"] == session
                && frame["params"]["state"] == state
        }) {
            return Ok(self.notifications.remove(pos));
        }
        loop {
            let frame = self.next().await?;
            if frame["method"] == "_maki/session/lifecycle"
                && frame["params"]["sessionId"] == session
                && frame["params"]["state"] == state
            {
                return Ok(frame);
            }
            assert!(
                frame.get("id").is_none(),
                "lifecycle waiter owns no answer: {frame}"
            );
            self.notifications.push(frame);
        }
    }

    /// Drive a set of in-flight turns (prompt ids) plus one optional cancel
    /// to settlement. Read-until-match; never sleeps.
    async fn drive(
        &mut self,
        prompts: &mut HashMap<String, String>,
        cancel: &mut Option<String>,
    ) -> io::Result<(HashMap<String, Value>, Option<Value>)> {
        let mut answers = HashMap::new();
        let mut cancel_answer = None;
        while !prompts.is_empty() || cancel.is_some() {
            let frame = self.next().await?;
            if let Some(id) = frame.get("id") {
                let key = id.to_string();
                if cancel.as_ref() == Some(&key) {
                    cancel_answer = Some(frame);
                    *cancel = None;
                    continue;
                }
                if prompts.remove(&key).is_some() {
                    answers.insert(key, frame);
                    continue;
                }
                panic!("drive saw an uncorrelated id-carrying frame: {frame}");
            }
            self.notifications.push(frame);
        }
        Ok((answers, cancel_answer))
    }

    async fn drive_bounded(
        &mut self,
        prompts: &mut HashMap<String, String>,
        cancel: &mut Option<String>,
    ) -> io::Result<(HashMap<String, Value>, Option<Value>)> {
        tokio::time::timeout(DRIVE_TIMEOUT, self.drive(prompts, cancel))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "drive exceeded its bound"))?
    }

    async fn prompt(&mut self, session: &str, text: &str) -> io::Result<Value> {
        let id = self.child.fresh_id();
        self.child
            .send_request(
                "session/prompt",
                id.clone(),
                json!({
                    "sessionId": session,
                    "prompt": [{"type": "text", "text": text}],
                }),
            )
            .await?;
        Ok(id)
    }

    async fn prompt_answer(&mut self, session: &str, text: &str) -> io::Result<Value> {
        let id = self.prompt(session, text).await?;
        let mut prompts = HashMap::from([(id.to_string(), session.to_owned())]);
        let (mut answers, _) = self.drive_bounded(&mut prompts, &mut None).await?;
        Ok(answers.remove(&id.to_string()).expect("drive resolves it"))
    }

    /// 02.9b retry path, live: a cancelled session rejects until its run's
    /// terminal confirms cleanup. Each attempt is a channel wait; the loop
    /// only bounds the total. Timing-sensitive by nature (the terminal
    /// arrives on the agent's clock), so the deadline is generous and the
    /// outcome explicit.
    async fn prompt_until_accepted(
        &mut self,
        session: &str,
        text: &str,
    ) -> io::Result<(Value, usize)> {
        let deadline = Instant::now() + RETRY_DEADLINE;
        let mut attempts = 0_usize;
        loop {
            attempts += 1;
            let answer = self.prompt_answer(session, text).await?;
            if answer.get("error").is_none() {
                return Ok((answer, attempts));
            }
            assert_eq!(
                answer["error"]["data"]["reason"], "cleanup_unconfirmed",
                "a gated retry rejects only with cleanup_unconfirmed: {answer}"
            );
            if Instant::now() >= deadline {
                panic!("retry still gated after {RETRY_DEADLINE:?}: {answer}");
            }
        }
    }

    /// Wait for the stub to observe the held request while pumping ACP
    /// frames: auxiliary provider traffic (session titles and the like ride
    /// the same stub) must not win the race; only the hold prompt's own
    /// answer (a rejection, not a start) or the stub signal ends the wait.
    /// Bounded; either signal wins.
    async fn await_hold_while_pumping(&mut self, stub: &Stub, hold_id: &Value) {
        let key = hold_id.to_string();
        let arrived = tokio::time::timeout(STUB_TIMEOUT, async {
            loop {
                tokio::select! {
                    _ = stub.config.hold_arrived.notified() => break,
                    res = self.next() => {
                        let frame = res.expect("child stays up");
                        if frame.get("id").map(ToString::to_string).as_ref() == Some(&key) {
                            panic!(
                                "the hold prompt was rejected, not started: {frame}\nstub log:\n{}",
                                stub.config.log.lock().unwrap().join("\n"),
                            );
                        }
                        self.notifications.push(frame);
                    }
                }
            }
        })
        .await;
        if arrived.is_err() {
            panic!(
                "hold never reached the stub. stub log:\n{}\nnotifications: {}",
                stub.config.log.lock().unwrap().join("\n"),
                self.notifications.len(),
            );
        }
    }

    fn updates_for(&self, session: &str) -> Vec<&Value> {
        self.notifications
            .iter()
            .filter(|frame| {
                frame["method"] == "session/update" && frame["params"]["sessionId"] == session
            })
            .collect()
    }
}

fn stop_reason(answer: &Value) -> &str {
    answer["result"]["stopReason"]
        .as_str()
        .unwrap_or("<missing>")
}

fn meta<'a>(answer: &'a Value, key: &str) -> &'a Value {
    &answer["result"]["_meta"][key]
}

/// Two live provider sessions under one unchanged Maki PID, with interleaved
/// progress and independent cancel/retry/close. See the module docs for what
/// is live and what is scripted.
#[tokio::test]
async fn two_live_sessions_share_one_pid_with_independent_controls() {
    let scratch = tempfile::TempDir::new().expect("a scratch dir");
    let marker_a = scratch.path().join("marker_a.txt");
    let marker_sleep = scratch.path().join("marker_sleep.txt");
    let stub = Stub::start(marker_a.clone(), marker_sleep.clone())
        .await
        .expect("the loopback stub binds");
    let base_url = stub.base_url();

    let projects = ProjectPair::create().expect("fixture projects");
    let child = MakiChild::spawn_with_env(&[
        ("OPENAI_API_KEY", "test-dummy-no-network"),
        ("OPENAI_BASE_URL", &base_url),
    ])
    .await
    .expect("the Maki child spawns");
    let mut live = Live {
        child,
        notifications: Vec::new(),
        permissions_answered: 0,
    };
    let pid = live.child.pid().expect("the child has a pid");

    let initialize = live
        .request(
            "initialize",
            json!({
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "maki-acp-harness", "version": "0.1.0"},
            }),
        )
        .await
        .expect("initialize");
    assert!(initialize.get("error").is_none(), "{initialize}");
    let advertised = &initialize["result"]["agentCapabilities"]["_meta"]["maki.multiSession"];
    for method in [
        "_maki/session/allocate",
        "_maki/session/activate",
        "_maki/session/cancel",
        "_maki/session/release",
    ] {
        assert!(
            advertised["methods"]
                .as_array()
                .is_some_and(|m| m.iter().any(|v| v == method)),
            "extension advertises {method}: {advertised}"
        );
    }

    let cwd_a = projects.a.to_string_lossy().into_owned();
    let cwd_b = projects.b.to_string_lossy().into_owned();
    let alloc_a = live
        .request("_maki/session/allocate", json!({"cwd": cwd_a}))
        .await
        .expect("allocate A");
    let alloc_b = live
        .request("_maki/session/allocate", json!({"cwd": cwd_b}))
        .await
        .expect("allocate B");
    let session_a = alloc_a["result"]["sessionId"]
        .as_str()
        .expect("A id")
        .to_owned();
    let session_b = alloc_b["result"]["sessionId"]
        .as_str()
        .expect("B id")
        .to_owned();
    assert_ne!(session_a, session_b, "two sessions hold distinct ids");

    let output_a = projects.a.join("plan-a.md").to_string_lossy().into_owned();
    let output_b = projects.b.join("plan-b.md").to_string_lossy().into_owned();
    for (session, cwd, output) in [
        (session_a.clone(), cwd_a.clone(), output_a),
        (session_b.clone(), cwd_b.clone(), output_b),
    ] {
        let active = live
            .request(
                "_maki/session/activate",
                json!({
                    "sessionId": session,
                    "model": "openai/gpt-5",
                    "mode": "build",
                    "cwd": cwd,
                    "output": output,
                }),
            )
            .await
            .expect("activate");
        assert!(active.get("error").is_none(), "{session}: {active}");
        assert_eq!(active["result"]["turnEpoch"], 0, "{active}");
    }
    for session in [&session_a, &session_b] {
        let frame = live
            .lifecycle(session, "active")
            .await
            .expect("active lifecycle");
        assert_eq!(frame["params"]["usable"], true, "{frame}");
        assert_eq!(frame["params"]["turnEpoch"], 0, "{frame}");
    }
    assert_eq!(live.child.pid(), Some(pid), "one PID owns both sessions");

    // Phase 1: interleaved live turns. Both prompts are in flight before
    // either answer is collected; correlation by id is the interleave proof.
    let id_a = live
        .prompt(&session_a, "TOOL_PROBE_A run the bash step")
        .await
        .expect("prompt A sends");
    let id_b = live
        .prompt(&session_b, "TEXT_PROBE_B say hello")
        .await
        .expect("prompt B sends");
    let mut prompts = HashMap::from([
        (id_a.to_string(), session_a.clone()),
        (id_b.to_string(), session_b.clone()),
    ]);
    let (answers, _) = live
        .drive_bounded(&mut prompts, &mut None)
        .await
        .expect("both turns settle");
    let answer_a = &answers[&id_a.to_string()];
    let answer_b = &answers[&id_b.to_string()];
    assert!(
        answer_a.get("error").is_none(),
        "A's live turn succeeds: {answer_a}"
    );
    assert_eq!(stop_reason(answer_a), "end_turn", "{answer_a}");
    assert_eq!(meta(answer_a, "maki.turnEpoch"), 0, "{answer_a}");
    assert!(
        answer_b.get("error").is_none(),
        "B's live turn succeeds: {answer_b}"
    );
    assert_eq!(stop_reason(answer_b), "end_turn", "{answer_b}");
    assert_eq!(meta(answer_b, "maki.turnEpoch"), 0, "{answer_b}");
    assert!(
        live.permissions_answered >= 1,
        "the tool turn asked permission and the harness allowed it"
    );
    assert!(
        !live.updates_for(&session_a).is_empty(),
        "A emitted session updates while B ran"
    );
    assert!(
        !live.updates_for(&session_b).is_empty(),
        "B emitted session updates while A ran"
    );
    assert_eq!(
        std::fs::read_to_string(&marker_a)
            .expect("the tool wrote its marker")
            .as_str(),
        "live-tool-A",
        "bash really ran: the marker is on disk with the stub's token"
    );
    assert_eq!(
        live.child.pid(),
        Some(pid),
        "still one PID after interleaved turns"
    );

    // Phase 2: live refusal on B. The filtered turn carries the verbatim
    // wire code; the healthy B handle stays usable for the very next turn.
    let refused = live
        .prompt_answer(&session_b, "REFUSAL_PROBE blocked text")
        .await
        .expect("refusal turn settles");
    assert!(refused.get("error").is_none(), "{refused}");
    assert_eq!(stop_reason(&refused), "refusal", "{refused}");
    assert_eq!(
        meta(&refused, "maki.rawReason"),
        "content_filter",
        "{refused}"
    );
    assert_eq!(meta(&refused, "maki.usable"), true, "{refused}");
    assert_eq!(meta(&refused, "maki.turnEpoch"), 1, "{refused}");
    let recovered = live
        .prompt_answer(&session_b, "TEXT_PROBE_B after refusal")
        .await
        .expect("B recovers");
    assert_eq!(stop_reason(&recovered), "end_turn", "{recovered}");
    assert_eq!(meta(&recovered, "maki.turnEpoch"), 2, "{recovered}");
    assert_eq!(
        live.child.pid(),
        Some(pid),
        "still one PID after refusal and recovery"
    );

    // Phase 3: mid-stream cancel on A while B keeps moving. The hold request
    // reaching the stub (channel signal) is what arms the cancel, and B's
    // text turn completes first, so independence is observed, not assumed.
    let id_hold = live
        .prompt(&session_a, "HOLD_PROBE never mind")
        .await
        .expect("hold prompt sends");
    live.await_hold_while_pumping(&stub, &id_hold).await;
    let b_during = live
        .prompt_answer(&session_b, "TEXT_PROBE_B during A hold")
        .await
        .expect("B progresses while A is held");
    assert_eq!(stop_reason(&b_during), "end_turn", "{b_during}");
    let cancel_id = live.child.fresh_id();
    live.child
        .send_request(
            "_maki/session/cancel",
            cancel_id.clone(),
            json!({"sessionId": session_a}),
        )
        .await
        .expect("cancel sends");
    let mut prompts = HashMap::from([(id_hold.to_string(), session_a.clone())]);
    let mut cancel = Some(cancel_id.to_string());
    let (answers, cancel_answer) = live
        .drive_bounded(&mut prompts, &mut cancel)
        .await
        .expect("cancel settles the held turn");
    let cancel_reply = cancel_answer.expect("cancel answers");
    assert_eq!(
        cancel_reply["result"]["sessionId"], session_a,
        "{cancel_reply}"
    );
    assert_eq!(cancel_reply["result"]["cancelled"], true, "{cancel_reply}");
    assert_eq!(cancel_reply["result"]["turnEpoch"], 2, "{cancel_reply}");
    let held_answer = &answers[&id_hold.to_string()];
    assert_eq!(stop_reason(held_answer), "cancelled", "{held_answer}");
    assert_eq!(meta(held_answer, "maki.turnEpoch"), 1, "{held_answer}");
    // The cancelled run's terminal reopens the session (02.9b, live): the
    // retry poll only passes once cleanup is confirmed.
    let (retried, attempts) = live
        .prompt_until_accepted(&session_a, "TEXT_PROBE_A after cancel")
        .await
        .expect("retry accepted after confirmed cleanup");
    assert_eq!(stop_reason(&retried), "end_turn", "{retried}");
    assert_eq!(meta(&retried, "maki.turnEpoch"), 2, "{retried}");
    eprintln!("02.11 live: retry after mid-stream cancel accepted after {attempts} attempt(s)");
    assert_eq!(
        live.child.pid(),
        Some(pid),
        "still one PID after cancel and retry"
    );

    // Phase 4: mid-tool cancel on B. The sleep only ends early if the cancel
    // really stopped the tool; its marker must never appear.
    let id_sleep = live
        .prompt(&session_b, "SLEEP_PROBE stall then mark")
        .await
        .expect("sleep prompt sends");
    let tool_started = Instant::now();
    loop {
        let frame = tokio::time::timeout(DRIVE_TIMEOUT, live.next())
            .await
            .expect("tool frames arrive within the drive bound")
            .expect("child stays up");
        if frame.get("id").is_some() {
            panic!("no prompt answer is due before the cancel: {frame}");
        }
        let is_tool_progress = frame["method"] == "session/update"
            && frame["params"]["sessionId"] == session_b
            && ["tool_call", "tool_call_update"].contains(
                &frame["params"]["update"]["sessionUpdate"]
                    .as_str()
                    .unwrap_or(""),
            );
        if is_tool_progress {
            live.notifications.push(frame);
            break;
        }
        live.notifications.push(frame);
    }
    eprintln!(
        "02.11 live: tool start observed after {:?}; cancelling mid-tool",
        tool_started.elapsed()
    );
    let cancel_b = live.child.fresh_id();
    live.child
        .send_request(
            "_maki/session/cancel",
            cancel_b.clone(),
            json!({"sessionId": session_b}),
        )
        .await
        .expect("cancel B sends");
    let mut prompts = HashMap::from([(id_sleep.to_string(), session_b.clone())]);
    let mut cancel = Some(cancel_b.to_string());
    let (answers, cancel_answer) = live
        .drive_bounded(&mut prompts, &mut cancel)
        .await
        .expect("mid-tool cancel settles");
    let cancel_b_reply = cancel_answer.expect("cancel B answers");
    assert_eq!(
        cancel_b_reply["result"]["cancelled"], true,
        "{cancel_b_reply}"
    );
    let sleep_answer = &answers[&id_sleep.to_string()];
    assert_eq!(stop_reason(sleep_answer), "cancelled", "{sleep_answer}");
    let (b_recovered, b_attempts) = live
        .prompt_until_accepted(&session_b, "TEXT_PROBE_B after tool cancel")
        .await
        .expect("B accepts a retry after its tool was cancelled");
    assert_eq!(stop_reason(&b_recovered), "end_turn", "{b_recovered}");
    eprintln!("02.11 live: retry after mid-tool cancel accepted after {b_attempts} attempt(s)");
    assert!(
        !marker_sleep.exists(),
        "the killed sleep never reached its marker: the tool really stopped"
    );
    assert_eq!(
        live.child.pid(),
        Some(pid),
        "still one PID after mid-tool cancel"
    );

    // Phase 5: independent release, then EOF shutdown on the same PID.
    for session in [&session_a, &session_b] {
        let released = live
            .request(
                "_maki/session/release",
                json!({"sessionId": session, "reason": "client_done"}),
            )
            .await
            .expect("release");
        assert_eq!(released["result"]["released"], true, "{released}");
        let frame = live
            .lifecycle(session, "released")
            .await
            .expect("released lifecycle");
        assert_eq!(frame["params"]["usable"], false, "{frame}");
    }
    let again = live
        .request(
            "_maki/session/release",
            json!({"sessionId": session_a, "reason": "client_done"}),
        )
        .await
        .expect("re-release");
    assert_eq!(
        again["result"]["released"], true,
        "release is idempotent per outcome: {again}"
    );
    assert_eq!(live.child.pid(), Some(pid), "one PID through every control");
    let chats = stub.chats();
    let holds = stub.holds();
    let models = stub.models();
    eprintln!(
        "02.11 live: stub served {chats} chat requests ({holds} held) and {models} catalog requests, all on 127.0.0.1"
    );
    assert!(
        chats >= 9,
        "every live turn above crossed the real HTTP wire: {chats}"
    );
    assert_eq!(holds, 1, "exactly the scripted hold stayed open: {holds}");
    stub.stop();
    live.child
        .shutdown()
        .await
        .expect("EOF shuts the child down cleanly");
    eprintln!("02.11 live: single PID was {pid}; shutdown clean");
}
