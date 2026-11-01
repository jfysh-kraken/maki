use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use maki_agent::agent::tool_dispatch::silent_mutation_check;
use maki_agent::cancel::CancelToken;
use maki_agent::tools::{
    Deadline, FileKey, MAIN_TASK_ID, ToolAudience, ToolContext, ToolLive, ToolOutputMode,
};
use maki_agent::{AgentEvent, AgentSeed, ToolPart};
use maki_config::{AgentConfig, ToolOutputLines};
use maki_storage::id::{MakiId, SessionRef};
use mlua::{
    AnyUserData, Function, LuaSerdeExt, MultiValue, Table, UserData, UserDataMethods,
    Value as LuaValue,
};
use tokio::task::spawn_blocking;

use crate::api::tool::ToolCallReply;
use crate::api::ui::buf::BufHandle;
use crate::api::util::convert::{TableExt, json_to_lua};
use crate::api::util::pair::{Pair, err_pair, pair};
use crate::runtime::{
    MoveOutcome, RestoreReason, SUBAGENT_SELF_MOVE_MSG, TaskHandle, active_task, lock_cell,
    move_self,
};

const DEADLINE_ALREADY_SET_MSG: &str = "ctx:set_deadline() already called";
const WRITE_TOOL: &str = "write";
const BACKGROUND_HOOK_ALREADY_SET_MSG: &str = "ctx:on_background() already called";
const NOT_A_MODEL_CALL_MSG: &str =
    "on_background: only the handler of a model's tool call can move to the background";
/// Model-facing: bash and task quote it when `background = true` cannot be
/// honoured.
const NESTED_CALL_MSG: &str = "background is ignored for a call inside batch or another \
     tool; call the tool directly to run it in the background";
const NO_BACKGROUND_HOOK_MSG: &str = "set_movable: call ctx:on_background() first";
const NO_BACKGROUND_HOOK_FOR_MOVE_MSG: &str = "background: call ctx:on_background() first";
const NESTED_PARTS_MSG: &str =
    "set_parts: only a model's own call is a task, a call nested in another has no parts";
const PART_STATUS_ERR: &str =
    "set_parts: status must be \"working\", \"done\", \"error\" or \"cancelled\"";
const EXPIRE_BACKGROUND: &str = "background";
const EXPIRE_TIMEOUT: &str = "timeout";

/// What `ctx:set_deadline` does when the deadline passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeadlineExpire {
    Timeout,
    /// Move the call to the background instead, falling back to a timeout
    /// when it cannot move.
    Background,
}

impl DeadlineExpire {
    fn parse(opts: Option<&Table>) -> mlua::Result<Self> {
        let Some(opts) = opts else {
            return Ok(Self::Timeout);
        };
        match opts.opt::<String>("expire")?.as_deref() {
            None | Some(EXPIRE_TIMEOUT) => Ok(Self::Timeout),
            Some(EXPIRE_BACKGROUND) => Ok(Self::Background),
            Some(other) => Err(mlua::Error::runtime(format!(
                "set_deadline: unknown expire {other:?}, expected \"{EXPIRE_TIMEOUT}\" or \"{EXPIRE_BACKGROUND}\""
            ))),
        }
    }
}

/// Tells whoever draws the call whether a Ctrl+B press would move it now: the
/// row's hint for a top-level call, the calling tool for a nested one. A row
/// the press cannot move gets no hint at all.
fn announce_movable(handle: &TaskHandle, agent: &AgentContext, movable: bool) {
    let (live, sink) = {
        let cell = lock_cell(handle);
        (cell.live.clone(), cell.live_sink.clone())
    };
    if let Some(live) = live {
        if could_move(agent) {
            let _ = live.event_tx.send(AgentEvent::ToolMovable {
                id: live.tool_use_id,
                movable,
            });
        }
    } else if let Some(sink) = sink {
        let _ = sink.send(ToolLive::Movable(movable));
    }
}

/// What the runtime asks before a move: a run to report to, and a live chat
/// whose task tabs show the work.
fn could_move(agent: &AgentContext) -> bool {
    agent.mailbox.is_some()
        && agent
            .session_id
            .as_ref()
            .is_some_and(|session| AgentSeed::live(session.id()).is_some())
}

fn send_live_buf(lua: &mlua::Lua, buf: &mlua::AnyUserData) -> mlua::Result<()> {
    let shared = buf.borrow::<BufHandle>().map(|h| Arc::clone(&h.buf))?;
    let task = active_task(lua);
    let (live, sink) = {
        let mut cell = lock_cell(&task);
        cell.root_buf = Some(Arc::clone(&shared));
        (cell.live.clone(), cell.live_sink.clone())
    };
    if let Some(live) = live {
        let _ = live.event_tx.send(maki_agent::AgentEvent::LiveToolBuf {
            id: live.tool_use_id.clone(),
            body: Arc::clone(&shared),
        });
    }
    if let Some(sink) = sink {
        let _ = sink.send(ToolLive::Buf(shared));
    }
    Ok(())
}

/// Captured snapshot of the parent `ToolContext`. Per-call state (deadline,
/// output lines) is reset so child calls start clean. The instruction handles
/// stay: a nested call is still the same session and the same model call.
///
/// Routing state is kept verbatim, `local_tools` included: a name a Lua tool
/// dispatches must land where the same name lands when the model calls it, or
/// shadowing quietly inverts for nested calls. A child session does not inherit
/// them anyway, `session()` always sets its own.
#[derive(Clone)]
pub(crate) struct AgentContext(ToolContext);

impl From<&ToolContext> for AgentContext {
    fn from(ctx: &ToolContext) -> Self {
        let mut c = ctx.clone();
        c.deadline = Deadline::default();
        c.tool_output_lines = ToolOutputLines::default();
        Self(c)
    }
}

impl Deref for AgentContext {
    type Target = ToolContext;
    fn deref(&self) -> &ToolContext {
        &self.0
    }
}

impl AgentContext {
    /// Drops `tool_use_id` so an inner tool never emits UI events under the
    /// outer call's id, and `live_sink` so a grandchild never streams into
    /// a sink meant for its parent.
    pub(crate) fn to_tool_context(&self) -> ToolContext {
        let mut c = self.0.clone();
        c.tool_use_id = None;
        c.live_sink = None;
        c
    }
}

/// One ctx type for handler, `start`, and restore invocations. Each kind's
/// capabilities live in its `Caps` variant, so a capability exists exactly
/// when its data does. Methods a kind lacks return `(nil, err)` instead of
/// not existing, so callers can probe without pcall.
pub(crate) struct LuaCtx {
    caps: Caps,
    pub(crate) cancel: CancelToken,
    tool_output_lines: ToolOutputLines,
    /// Which chat the run serves, see [`ToolContext::session_id`] and
    /// [`ToolContext::task_id`]. Restore takes both from the chat being
    /// re-rendered.
    session_id: Option<SessionRef>,
    task_id: Option<Arc<str>>,
    /// The owning session's cwd (02.5a carries it; 02.5b resolves against
    /// it, 02.5c owns subprocess/instruction behavior). `None` when no
    /// session owns this context: one-shots, restores (whose item names no
    /// cwd), and session-detached spawns keep the process-cwd fallback.
    cwd: Option<PathBuf>,
    pub(crate) finish_tx: Option<flume::Sender<ToolCallReply>>,
}

enum Caps {
    Handler {
        agent: Box<AgentContext>,
    },
    /// `start` runs before permission checks: it reads config and publishes
    /// previews, but dispatching tools is structurally impossible.
    Start {
        config: AgentConfig,
        workflow: bool,
        audience: ToolAudience,
        output_mode: ToolOutputMode,
    },
    Restore {
        state: Option<serde_json::Value>,
        reason: RestoreReason,
    },
}

/// What a restore run knows about the call it re-renders: the host fills it
/// from a `RestoreItem`, batch from the plain table it hands its children.
#[derive(Default)]
pub(crate) struct RestoreCtx {
    pub(crate) tool_output_lines: ToolOutputLines,
    pub(crate) state: Option<serde_json::Value>,
    pub(crate) session_id: Option<SessionRef>,
    pub(crate) task_id: Option<Arc<str>>,
    pub(crate) reason: RestoreReason,
}

impl LuaCtx {
    fn new(ctx: &ToolContext, caps: Caps) -> Self {
        Self {
            caps,
            cancel: ctx.cancel.clone(),
            tool_output_lines: ctx.tool_output_lines,
            session_id: ctx.session_id.clone(),
            task_id: ctx.task_id.clone(),
            // A session-less context (the `maki index` one-shot, a detached
            // spawn) owns no project: its cwd is the process stand-in
            // `interpreter_ctx` records, never an owning session's.
            cwd: ctx.session_id.is_some().then(|| ctx.cwd.clone()),
            finish_tx: None,
        }
    }

    pub(crate) fn handler(ctx: &ToolContext) -> Self {
        Self::new(
            ctx,
            Caps::Handler {
                agent: Box::new(AgentContext::from(ctx)),
            },
        )
    }

    pub(crate) fn start(ctx: &ToolContext) -> Self {
        Self::new(
            ctx,
            Caps::Start {
                config: ctx.config.clone(),
                workflow: ctx.workflow,
                audience: ctx.audience,
                output_mode: ctx.output_mode,
            },
        )
    }

    pub(crate) fn restore(ctx: RestoreCtx) -> Self {
        Self {
            caps: Caps::Restore {
                state: ctx.state,
                reason: ctx.reason,
            },
            cancel: CancelToken::none(),
            tool_output_lines: ctx.tool_output_lines,
            session_id: ctx.session_id,
            task_id: ctx.task_id,
            // 02.5b decision: a restore names its session but carries no cwd
            // (`RestoreItem` records none), and its task runs without a
            // session lineage, so restore callbacks keep the process-cwd
            // fallback. Re-rendering a saved transcript must not resolve
            // fresh paths against a possibly-moved session directory.
            cwd: None,
            finish_tx: None,
        }
    }

    pub(crate) fn session(&self) -> Option<MakiId> {
        self.session_id.as_ref().map(SessionRef::id)
    }

    /// The owning session's cwd, so a call site can resolve against its own
    /// project instead of the process cwd. `None` keeps the process-cwd
    /// fallback (one-shots, restores, detached spawns).
    pub(crate) fn cwd(&self) -> Option<PathBuf> {
        self.cwd.clone()
    }

    /// Dispatch capability: only handler ctxs can call `maki.agent.*`.
    pub(crate) fn agent(&self) -> Option<&AgentContext> {
        match &self.caps {
            Caps::Handler { agent } => Some(agent),
            _ => None,
        }
    }

    fn config(&self) -> Option<&AgentConfig> {
        match &self.caps {
            Caps::Handler { agent } => Some(&agent.config),
            Caps::Start { config, .. } => Some(config),
            Caps::Restore { .. } => None,
        }
    }

    fn workflow(&self) -> Option<bool> {
        match &self.caps {
            Caps::Handler { agent } => Some(agent.workflow),
            Caps::Start { workflow, .. } => Some(*workflow),
            Caps::Restore { .. } => None,
        }
    }

    fn audience(&self) -> Option<ToolAudience> {
        match &self.caps {
            Caps::Handler { agent } => Some(agent.audience),
            Caps::Start { audience, .. } => Some(*audience),
            Caps::Restore { .. } => None,
        }
    }

    fn restore_reason(&self) -> Option<RestoreReason> {
        match &self.caps {
            Caps::Restore { reason, .. } => Some(*reason),
            _ => None,
        }
    }

    fn output_mode(&self) -> ToolOutputMode {
        match &self.caps {
            Caps::Handler { agent } => agent.output_mode,
            Caps::Start { output_mode, .. } => *output_mode,
            Caps::Restore { .. } => ToolOutputMode::Compact,
        }
    }

    fn task_id(&self) -> &str {
        self.task_id.as_deref().unwrap_or(MAIN_TASK_ID)
    }

    fn state(&self) -> Option<&serde_json::Value> {
        match &self.caps {
            Caps::Restore { state, .. } => state.as_ref(),
            _ => None,
        }
    }

    fn kind(&self) -> &'static str {
        match self.caps {
            Caps::Handler { .. } => "handler",
            Caps::Start { .. } => "start",
            Caps::Restore { .. } => "restore",
        }
    }

    pub(crate) fn cap_err(&self, method: &str) -> String {
        format!("{method} not available in {} ctx", self.kind())
    }

    pub(crate) fn cap_err_pair<T>(&self, method: &str) -> Pair<T> {
        (None, Some(self.cap_err(method)))
    }
}

impl UserData for LuaCtx {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("cancelled", |_, this, ()| Ok(this.cancel.is_cancelled()));

        methods.add_method("output_mode", |_, this, ()| Ok(this.output_mode().as_str()));

        methods.add_method("workflow", |_, this, ()| {
            let Some(workflow) = this.workflow() else {
                return Ok(this.cap_err_pair("workflow"));
            };
            Ok((Some(workflow), None))
        });

        methods.add_method("audience", |_, this, ()| {
            let Some(audience) = this.audience() else {
                return Ok(this.cap_err_pair("audience"));
            };
            Ok((Some(audience.name().unwrap_or("main").to_string()), None))
        });

        // The session that called this tool, or whose transcript a restore
        // re-renders, which under concurrent sessions is not always the
        // focused one `maki.session.current()` reports. Nil when the run has
        // no session, as in the `maki index` one-shot.
        methods.add_method("session_id", |_, this, ()| {
            Ok(this.session_id.as_ref().map(|id| id.id().to_string()))
        });

        methods.add_method("task_id", |_, this, ()| Ok(this.task_id().to_owned()));

        methods.add_method("tool_use_id", |_, this, ()| {
            Ok(this.agent().and_then(|agent| agent.tool_use_id.clone()))
        });

        methods.add_method("restore_reason", |_, this, ()| {
            let Some(reason) = this.restore_reason() else {
                return Ok(this.cap_err_pair("restore_reason"));
            };
            Ok((Some(<&str>::from(reason)), None))
        });

        methods.add_method("live_buf", |lua, this, buf: mlua::AnyUserData| {
            if matches!(this.caps, Caps::Restore { .. }) {
                return Ok(this.cap_err_pair("live_buf"));
            }
            send_live_buf(lua, &buf)?;
            Ok((Some(true), None))
        });

        methods.add_method("config", |lua, this, args: MultiValue| {
            let Some(config) = this.config() else {
                return Ok(this.cap_err_pair("config"));
            };
            let config_val = lua.to_value(config)?;
            if args.is_empty() {
                return Ok((Some(config_val), None));
            }
            let key: String = lua.from_value(args[0].clone())?;
            let default = args.get(1).cloned().unwrap_or(LuaValue::Nil);
            let val = match config_val {
                LuaValue::Table(ref tbl) => {
                    let val = tbl.raw_get::<LuaValue>(key.as_str())?;
                    if matches!(val, LuaValue::Nil) {
                        default
                    } else {
                        val
                    }
                }
                _ => default,
            };
            Ok((Some(val), None))
        });

        methods.add_method("tool_output_lines", |lua, this, ()| {
            lua.to_value(&this.tool_output_lines)
        });

        methods.add_method("state", |lua, this, ()| match this.state() {
            Some(v) => json_to_lua(lua, v),
            None => Ok(LuaValue::Nil),
        });

        methods.add_method(
            "on_background",
            |lua, this, hook: Function| -> mlua::Result<Pair<bool>> {
                let Some(agent) = this.agent() else {
                    return Ok(this.cap_err_pair("on_background"));
                };
                let Some(call) = &agent.call else {
                    return Ok(err_pair(NOT_A_MODEL_CALL_MSG));
                };
                let handle = active_task(lua);
                {
                    let cell = lock_cell(&handle);
                    if !cell.call.as_ref().is_some_and(|own| Arc::ptr_eq(own, call)) {
                        return Ok(err_pair(NOT_A_MODEL_CALL_MSG));
                    }
                    if cell.background_hook.is_some() {
                        return Err(mlua::Error::runtime(BACKGROUND_HOOK_ALREADY_SET_MSG));
                    }
                }
                // Made outside the lock: allocating can run the GC, and a
                // finalizer may want the same cell.
                let key = lua.create_registry_value(hook)?;
                lock_cell(&handle).background_hook = Some(key);
                announce_movable(&handle, agent, true);
                Ok((Some(true), None))
            },
        );

        // A hook that would decline right now (a batch waiting on an edit)
        // hides the hint until it would accept again.
        methods.add_method("set_movable", |lua, this, movable: bool| {
            let Some(agent) = this.agent() else {
                return Ok(this.cap_err_pair("set_movable"));
            };
            let handle = active_task(lua);
            if lock_cell(&handle).background_hook.is_none() {
                return Ok(err_pair(NO_BACKGROUND_HOOK_MSG));
            }
            announce_movable(&handle, agent, movable);
            Ok((Some(true), None))
        });

        // Parts hang under the model's own call, the only call with a tab in
        // `/tasks`. A nested call has no tab, so it gets an error instead.
        methods.add_method("set_parts", |lua, this, parts: Vec<Table>| {
            let Some(agent) = this.agent() else {
                return Ok(this.cap_err_pair("set_parts"));
            };
            let Some(id) = agent.tool_use_id.clone() else {
                return Ok(err_pair(NESTED_PARTS_MSG));
            };
            let parts = parts
                .iter()
                .map(|part| {
                    Ok(ToolPart {
                        id: part.get::<String>("id")?,
                        name: part.get::<String>("name")?,
                        tool: part.get::<String>("tool")?.into(),
                        status: lua
                            .from_value(part.get::<LuaValue>("status")?)
                            .map_err(|_| mlua::Error::runtime(PART_STATUS_ERR))?,
                        body: part
                            .opt::<AnyUserData>("body")?
                            .map(|body| body.borrow::<BufHandle>().map(|h| Arc::clone(&h.buf)))
                            .transpose()?,
                    })
                })
                .collect::<mlua::Result<Vec<_>>>()?;
            let _ = agent.event_tx.send(AgentEvent::ToolParts { id, parts });
            Ok((Some(true), None))
        });

        // The call moves itself through its own `on_background` hook, as if
        // the user pressed Ctrl+B. Gated on the plugin's `agent` permission
        // by the runtime.
        methods.add_async_method("background", |lua, this, ()| async move {
            let Some(agent) = this.agent() else {
                return Ok(this.cap_err_pair("background"));
            };
            if agent.task_id.is_some() {
                return Ok(err_pair(SUBAGENT_SELF_MOVE_MSG));
            }
            let Some(call) = agent.call.clone() else {
                return Ok(err_pair(NOT_A_MODEL_CALL_MSG));
            };
            // A nested call shares its parent's scope but has no id of its own.
            let Some(id) = agent.tool_use_id.clone() else {
                return Ok(err_pair(NESTED_CALL_MSG));
            };
            // Nothing may hold the ctx borrow across the wait: the hook runs
            // meanwhile and may call ctx methods.
            drop(this);
            let outcome = move_self(&lua, &id).await;
            // Moved by a Ctrl+B press, before or during the wait: done all
            // the same.
            if call.home().is_some() {
                return Ok((Some(true), None));
            }
            Ok(match outcome {
                MoveOutcome::Moved => (Some(true), None),
                MoveOutcome::NoHook => err_pair(NO_BACKGROUND_HOOK_FOR_MOVE_MSG),
                MoveOutcome::NotMovable(reason) => (None, Some(reason)),
            })
        });

        methods.add_method("backgrounded", |_, this, ()| {
            Ok(this
                .agent()
                .and_then(|agent| agent.call.as_ref())
                .is_some_and(|call| call.home().is_some()))
        });

        methods.add_method(
            "set_deadline",
            |lua, this, (secs, opts): (u64, Option<Table>)| {
                if !matches!(this.caps, Caps::Handler { .. }) {
                    return Ok(this.cap_err_pair("set_deadline"));
                }
                let expire = DeadlineExpire::parse(opts.as_ref())?;
                let handle = active_task(lua);
                let cell = lock_cell(&handle);
                if cell.deadline_secs.get().is_some() {
                    return Err(mlua::Error::runtime(DEADLINE_ALREADY_SET_MSG));
                }
                // Only a model call can move, anything else just times out.
                if cell.call.is_some() {
                    cell.deadline_expire.set(expire);
                }
                cell.deadline_secs.set(Some(secs));
                cell.deadline
                    .set(Some(Instant::now() + Duration::from_secs(secs)));
                cell.deadline_changed.notify(usize::MAX);
                Ok((Some(true), None))
            },
        );

        // The matching check before a write is not exposed: the dispatcher
        // runs it under the file lock for every tool declaring `mutable_path`,
        // and a handler-side copy would race its own sibling.
        methods.add_method("record_read", |_, this, path: String| {
            let Some(agent) = this.agent() else {
                return Ok(this.cap_err_pair("record_read"));
            };
            agent
                .file_access
                .record_read(&FileKey::new(Path::new(&path)));
            Ok((Some(true), None))
        });

        // A plugin that changes the disk itself has no tool call to hang a
        // prompt on, so this says yes only where the write tool would run
        // without asking anyone.
        methods.add_method("can_write", |_, this, path: String| {
            let Some(agent) = this.agent() else {
                return Ok(this.cap_err_pair("can_write"));
            };
            Ok(pair(
                silent_mutation_check(agent, WRITE_TOOL, &path).map(|()| true),
            ))
        });

        methods.add_async_method(
            "load_instructions",
            |_, this, dir_path: String| async move {
                let Some(agent) = this.agent() else {
                    return Ok(this.cap_err_pair("load_instructions"));
                };
                let loaded = agent.loaded_instructions.clone();
                let call = agent.call_instructions.clone();
                let files = Arc::clone(&agent.instruction_files);
                // 02.5c: later subdirectory-instruction discovery walks from
                // the owning session's directory (`agent.cwd`, carried since
                // 02.4 and resolved since 02.5b), not the process cwd or a
                // peer project. Attached agents inherit the parent cwd via
                // 02.5b propagation, so their walks land in the parent
                // project too. The lineage is unreachable off the executor,
                // so capture before the blocking thread takes over.
                let session_cwd = agent.cwd.clone();
                // Nothing may hold the ctx borrow across the wait: a cancel
                // hook firing meanwhile needs `ctx:finish`, which takes it
                // mutably.
                drop(this);
                let blocks = spawn_blocking(move || {
                    let abs = resolve_abs_with_cwd(dir_path, &session_cwd);
                    maki_agent::find_subdirectory_instructions(&abs, &session_cwd, &loaded, &files)
                })
                .await
                .map_err(mlua::Error::external)?;
                call.record(blocks);
                Ok((Some(true), None))
            },
        );

        methods.add_method("is_instruction_file", |_, this, name: String| {
            Ok(this.agent().is_some_and(|agent| {
                maki_agent::is_instruction_file(&agent.instruction_files, &name)
            }))
        });

        methods.add_method_mut("finish", |lua, this, val: LuaValue| {
            if !matches!(this.caps, Caps::Handler { .. }) {
                return Ok(this.cap_err_pair("finish"));
            }
            let tx = this
                .finish_tx
                .take()
                .ok_or_else(|| mlua::Error::runtime("ctx:finish() already called"))?;

            if let Some(buf) = crate::api::ui::buf::buf_from_reply(&val) {
                lock_cell(&active_task(lua)).root_buf = Some(buf);
            }
            let _ = tx.send(ToolCallReply::from_lua_value(lua, &val));
            Ok((Some(true), None))
        });
    }
}

fn resolve_abs_with_cwd(path: String, cwd: &Path) -> PathBuf {
    if Path::new(&path).is_absolute() {
        path.into()
    } else {
        cwd.join(&path)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use maki_agent::tools::test_support::stub_ctx_with;
    use maki_agent::tools::{LocalTool, ToolAudience};
    use maki_agent::{AgentMode, InstructionBlock};
    use mlua::Lua;
    use tempfile::TempDir;
    use test_case::test_case;

    use super::*;
    use crate::runtime::test_support::call_scope;
    use crate::runtime::{TaskCell, TaskScope};

    const TOOL_USE_ID: &str = "tu-1";
    const INSTRUCTION_PATH: &str = "/tmp/nested/AGENTS.md";
    const LOCAL_TOOL_NAME: &str = "sess_tool";
    /// Arbitrary ids are rejected: `SessionRef` parses base58 or a uuid.
    const SESSION_ID: &str = "01965087-4c71-7f00-8000-000000000000";
    const SUBAGENT_TASK_ID: &str = "toolu_task";

    fn session_ref() -> SessionRef {
        SESSION_ID.parse().expect("valid session id")
    }

    fn populated_ctx() -> ToolContext {
        let mut ctx = stub_ctx_with(&AgentMode::Build, None, Some(TOOL_USE_ID));
        ctx.session_id = Some(session_ref());
        ctx.deadline = Deadline::after(Duration::from_secs(60));
        ctx.tool_output_lines = ToolOutputLines {
            bash: 999,
            ..ToolOutputLines::default()
        };
        assert!(
            !ctx.loaded_instructions
                .contains_or_insert(PathBuf::from(INSTRUCTION_PATH))
        );
        ctx.call_instructions.record(vec![InstructionBlock {
            path: INSTRUCTION_PATH.into(),
            content: String::new(),
        }]);
        let mut tools: HashMap<String, LocalTool> = HashMap::new();
        tools.insert(
            LOCAL_TOOL_NAME.into(),
            maki_agent::tools::local_tool(ToolAudience::MODEL, |_, _| {
                Box::pin(async { Ok(String::new()) })
            }),
        );
        ctx.local_tools = Arc::new(tools);
        ctx.live_sink = Some(flume::unbounded().0);
        ctx
    }

    #[test]
    fn agent_context_keeps_tool_use_id_and_resets_per_call_state() {
        let agent = AgentContext::from(&populated_ctx());
        assert_eq!(agent.tool_use_id.as_deref(), Some(TOOL_USE_ID));
        assert_eq!(
            agent.session_id,
            Some(session_ref()),
            "the session owns the whole run, so it is not per-call state"
        );
        assert!(agent.deadline.at().is_none());
        assert_eq!(agent.tool_output_lines, ToolOutputLines::default());
        assert!(
            agent.local_tools.contains_key(LOCAL_TOOL_NAME),
            "a nested call must route names exactly like the model's own call"
        );
        assert!(
            agent
                .loaded_instructions
                .contains_or_insert(PathBuf::from(INSTRUCTION_PATH)),
            "session state, shared with nested calls"
        );
        assert_eq!(
            agent.call_instructions.take().len(),
            1,
            "per model call, shared with nested calls"
        );
    }

    #[test]
    fn agent_context_to_tool_context_drops_tool_use_id_and_sink() {
        let agent = AgentContext::from(&populated_ctx());
        assert!(
            agent.live_sink.is_some(),
            "the sink set by the caller must survive into AgentContext"
        );
        let inner = agent.to_tool_context();
        assert_eq!(inner.tool_use_id, None);
        assert!(inner.live_sink.is_none(), "sink must not be inherited");
        assert_eq!(
            agent.tool_use_id.as_deref(),
            Some(TOOL_USE_ID),
            "dropping must not disturb the caller's own context"
        );
        assert_eq!(
            inner.session_id,
            Some(session_ref()),
            "a dispatched child runs in the same session, unlike tool_use_id"
        );
    }

    /// 02.4: the Lua handler context keeps the owning session's cwd through
    /// every clone (nested calls, attached agents), never the process cwd.
    /// Resolution behavior is 02.5b/c's; this only proves the value is
    /// carried.
    #[test]
    fn agent_context_carries_the_session_cwd() {
        let ctx = populated_ctx();
        let agent = AgentContext::from(&ctx);
        assert_eq!(agent.cwd, ctx.cwd, "the handler keeps its session");
        let inner = agent.to_tool_context();
        assert_eq!(
            inner.cwd, ctx.cwd,
            "a dispatched child serves the same session cwd"
        );
    }

    /// 02.5a: every Lua ctx kind carries its owning session's cwd (`None`
    /// with no owning session), while the Maki process cwd stays unchanged.
    /// Resolution behavior is 02.5b/c's; this only proves the value is
    /// carried.
    #[test]
    fn lua_ctx_carries_each_sessions_cwd() {
        let process_cwd = std::env::current_dir().expect("the process cwd is readable");
        let project_a = TempDir::new().unwrap();
        let project_b = TempDir::new().unwrap();
        assert_ne!(project_a.path(), project_b.path());

        let mut ctx_a = populated_ctx();
        ctx_a.cwd = project_a.path().to_path_buf();
        let mut ctx_b = populated_ctx();
        ctx_b.cwd = project_b.path().to_path_buf();

        for (ctx, project) in [(&ctx_a, &project_a), (&ctx_b, &project_b)] {
            assert_eq!(
                LuaCtx::handler(ctx).cwd().as_deref(),
                Some(project.path()),
                "the handler keeps its session"
            );
            assert_eq!(
                LuaCtx::start(ctx).cwd().as_deref(),
                Some(project.path()),
                "start previews keep the session"
            );
        }

        // No owning session: the one-shot stand-in is the process cwd, but
        // it is not an owning session's, so the ctx carries `None` and 02.5b
        // keeps the process-cwd fallback there.
        let mut lone = populated_ctx();
        lone.session_id = None;
        assert_eq!(LuaCtx::handler(&lone).cwd(), None);
        assert_eq!(LuaCtx::start(&lone).cwd(), None);
        // A restore names its session but carries no cwd (02.5b decides).
        assert_eq!(LuaCtx::restore(restore_ctx(&ctx_a)).cwd(), None);

        assert_eq!(
            std::env::current_dir().expect("the process cwd is readable"),
            process_cwd,
            "carrying session cwds never moves the Maki process cwd"
        );
    }

    fn restore_ctx(ctx: &ToolContext) -> RestoreCtx {
        RestoreCtx {
            session_id: ctx.session_id.clone(),
            task_id: ctx.task_id.clone(),
            ..RestoreCtx::default()
        }
    }

    /// Restore has no `ToolContext`, so its session comes stamped on the item.
    #[test]
    fn session_id_reaches_every_ctx_kind() {
        let ctx = populated_ctx();
        for lua_ctx in [
            LuaCtx::handler(&ctx),
            LuaCtx::start(&ctx),
            LuaCtx::restore(restore_ctx(&ctx)),
        ] {
            assert_eq!(
                lua_ctx.session_id,
                Some(session_ref()),
                "{}",
                lua_ctx.kind()
            );
        }
    }

    #[test_case(None, MAIN_TASK_ID ; "session_owner")]
    #[test_case(Some(SUBAGENT_TASK_ID), SUBAGENT_TASK_ID ; "subagent")]
    fn task_id_reaches_every_ctx_kind(task_id: Option<&str>, expected: &str) {
        let mut ctx = populated_ctx();
        ctx.task_id = task_id.map(Arc::from);
        assert_eq!(LuaCtx::handler(&ctx).task_id(), expected);
        assert_eq!(LuaCtx::start(&ctx).task_id(), expected);
        assert_eq!(LuaCtx::restore(restore_ctx(&ctx)).task_id(), expected);
    }

    #[test]
    fn a_second_on_background_raises() {
        let lua = Lua::new();
        let (call, _run, _) = call_scope(false);
        let mut ctx = populated_ctx();
        ctx.call = Some(Arc::clone(&call));
        let mut cell = TaskCell::new(call.cancel().clone(), None, None);
        cell.call = Some(call);
        let scope = TaskScope::new(&lua, cell);
        let _active = scope.enter();
        let handler = lua.create_userdata(LuaCtx::handler(&ctx)).unwrap();

        let (ok, err): (bool, String) = lua
            .load(
                r#"
                local ctx = ...
                local hook = function() end
                ctx:on_background(hook)
                local ok, err = pcall(ctx.on_background, ctx, hook)
                return ok, tostring(err)
                "#,
            )
            .call(handler)
            .unwrap();

        assert!(!ok);
        assert!(err.contains(BACKGROUND_HOOK_ALREADY_SET_MSG), "{err}");
    }

    /// A plain-table ctx that names no reason must read as a rerender: side
    /// effects on load are opt-in, never the fallback.
    #[test]
    fn restore_reason_defaults_to_rerender_and_is_restore_only() {
        let ctx = populated_ctx();
        assert_eq!(
            LuaCtx::restore(RestoreCtx::default()).restore_reason(),
            Some(RestoreReason::Rerender)
        );
        assert_eq!(LuaCtx::handler(&ctx).restore_reason(), None);
        assert_eq!(LuaCtx::start(&ctx).restore_reason(), None);
    }

    /// One fixture project per session, each with a subdirectory instruction
    /// file the other lacks.
    struct SessionProject {
        _tmp: TempDir,
        dir: PathBuf,
        content: &'static str,
    }

    impl SessionProject {
        fn new(content: &'static str) -> Self {
            let tmp = TempDir::new().unwrap();
            std::fs::create_dir(tmp.path().join("sub")).unwrap();
            std::fs::write(tmp.path().join("sub").join("AGENTS.md"), content).unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            Self {
                _tmp: tmp,
                dir,
                content,
            }
        }
    }

    fn session_tool_ctx(dir: &Path) -> ToolContext {
        let mut ctx = populated_ctx();
        ctx.cwd = dir.to_path_buf();
        ctx.session_id = Some(session_ref());
        ctx.loaded_instructions = maki_agent::agent::LoadedInstructions::new();
        ctx.call_instructions = maki_agent::agent::CallInstructions::default();
        ctx.instruction_files = Arc::new(maki_agent::agent::default_instruction_files());
        ctx
    }

    /// 02.5c: `ctx:load_instructions` walks from the owning session's
    /// directory, not the process cwd or a peer project. Real Lua
    /// `ctx:load_instructions("sub")` through the real handler path.
    #[tokio::test]
    async fn load_instructions_walk_the_owning_session() {
        let process_cwd = std::env::current_dir().expect("the process cwd is readable");
        let project = SessionProject::new("session-instructions-here");
        assert_ne!(
            project.dir,
            process_cwd.canonicalize().unwrap(),
            "the fixture must not sit in the process cwd"
        );

        let ctx = session_tool_ctx(&project.dir);
        let lua = Lua::new();
        let handler = lua.create_userdata(LuaCtx::handler(&ctx)).unwrap();
        let (ok, err): (Option<bool>, Option<String>) = lua
            .load(r#"local ctx = ...; return ctx:load_instructions("sub")"#)
            .call_async(handler)
            .await
            .unwrap();
        assert_eq!(err, None, "later discovery must resolve");
        assert_eq!(ok, Some(true));

        let blocks = ctx.call_instructions.take();
        assert!(
            blocks.iter().any(|block| block.content == project.content),
            "the session's subdirectory instructions must load, got {blocks:?}"
        );
        assert_eq!(
            std::env::current_dir().expect("the process cwd is readable"),
            process_cwd,
            "instruction discovery never moves the Maki process cwd"
        );
    }

    /// 02.5c runtime-tool cwd proof, instruction half (distinct from 02.4's
    /// initial-context proof): interleaved later subdirectory-instruction
    /// loads in separate A/B fixture projects associate with the correct
    /// session, with the process cwd unchanged throughout.
    #[tokio::test]
    async fn interleaved_later_instructions_serve_their_own_session() {
        async fn run_session(project: &SessionProject, label: &'static str) -> Vec<String> {
            let ctx = session_tool_ctx(&project.dir);
            let lua = Lua::new();
            let handler = lua.create_userdata(LuaCtx::handler(&ctx)).unwrap();
            // Yield between sessions exactly like concurrent tool calls on
            // the shared executor.
            tokio::task::yield_now().await;
            let (ok, err): (Option<bool>, Option<String>) = lua
                .load(r#"local ctx = ...; return ctx:load_instructions("sub")"#)
                .call_async(handler)
                .await
                .unwrap();
            assert_eq!(err, None, "{label}: later discovery must resolve");
            assert_eq!(ok, Some(true), "{label}: load must answer true");
            tokio::task::yield_now().await;
            let blocks = ctx.call_instructions.take();
            assert!(
                blocks.iter().any(|block| block.content == project.content),
                "{label}: must load its own instructions, got {blocks:?}"
            );
            blocks.into_iter().map(|block| block.content).collect()
        }

        let process_cwd = std::env::current_dir().expect("the process cwd is readable");
        let project_a = SessionProject::new("session-a-instructions");
        let project_b = SessionProject::new("session-b-peer-instructions");
        assert_ne!(project_a.dir, project_b.dir);

        let (contents_a, contents_b) = tokio::join!(
            run_session(&project_a, "session-a"),
            run_session(&project_b, "session-b"),
        );
        assert!(
            contents_a.iter().any(|c| c == project_a.content),
            "session A loads A, got {contents_a:?}"
        );
        assert!(
            contents_b.iter().any(|c| c == project_b.content),
            "session B loads B, got {contents_b:?}"
        );
        assert!(
            !contents_a.iter().any(|c| c == project_b.content),
            "session A must not load B's instructions"
        );
        assert!(
            !contents_b.iter().any(|c| c == project_a.content),
            "session B must not load A's instructions"
        );
        assert_eq!(
            std::env::current_dir().expect("the process cwd is readable"),
            process_cwd,
            "instruction discovery never moves the Maki process cwd"
        );
    }

    /// 02.5c attached-agent proof: a child session opened from a parent
    /// inherits the parent cwd (02.5b propagation), so its later
    /// `ctx:load_instructions` walks the parent project. Real Lua handler
    /// path under the child's tool context.
    #[tokio::test]
    async fn attached_later_instructions_retain_the_parent_cwd() {
        let process_cwd = std::env::current_dir().expect("the process cwd is readable");
        let project = SessionProject::new("parent-project-instructions");
        assert_ne!(
            project.dir,
            process_cwd.canonicalize().unwrap(),
            "the fixture must not sit in the process cwd"
        );

        // The child's tool context carries the parent cwd the way a real
        // activation builds attached permissions from the parent session
        // (02.4 carries it, 02.5b resolves it, 02.5c walks instructions).
        let ctx = session_tool_ctx(&project.dir);
        let lua = Lua::new();
        let handler = lua.create_userdata(LuaCtx::handler(&ctx)).unwrap();
        let (ok, err): (Option<bool>, Option<String>) = lua
            .load(r#"local ctx = ...; return ctx:load_instructions("sub")"#)
            .call_async(handler)
            .await
            .unwrap();
        assert_eq!(err, None);
        assert_eq!(ok, Some(true));

        let blocks = ctx.call_instructions.take();
        assert!(
            blocks.iter().any(|block| block.content == project.content),
            "the attached walk must land in the parent project, got {blocks:?}"
        );
        assert_eq!(
            std::env::current_dir().expect("the process cwd is readable"),
            process_cwd,
            "attached discovery never moves the Maki process cwd"
        );
    }
}
