# Maki multi-session ACP extension (fork contract, slice 02.1)

Status: partially implemented on branch `nori/mvp`. The single-session
server answered every `_maki/` method with `-32601 Method not found`; slice
02.2b wired prompt/event routing against this contract, slice 02.3 wired
`_maki/session/allocate` and `_maki/session/activate` (runtime in
`maki-acp/src/server.rs`, lease/model/failure/responsiveness fixtures in
`server.rs` tests and `tests/allocation_activation.rs`), slice 02.7 wired
terminal reasons, slice 02.8 wired concurrent permission routing
(several outstanding asks under globally unique ids, task delivery,
blocking-mode terminal retirement), slice 02.9a wired addressed turn
cancellation (exactly-once settlement through the turn epoch, retirement of
the cancelled turn's asks in both modes, `cleanup_unconfirmed` retry gating;
fixtures in `server.rs` tests), and slice 02.9b wires cleanup confirmation
and retry (the cancelled run's terminal confirms in blocking mode, failure
or a lapsed bounded wait reports explicitly non-resumable uncertainty;
fixtures in `server.rs` tests, escaped-child limitations recorded in
`tests/cleanup_probes.rs`). Ask retirement on the handle-loss (02.10a)
path is downstream debt, not built here. Active-session release (02.10b)
still answers `-32601`, as do the reserved 04.1 and 05.2 names. This document
stays normative; the one deliberate delta is the advertised method list: the
capability only names the methods the binary answers (allocate+activate+cancel
today; release joins in 02.10b), so probing an unlisted method keeps reporting
"unsupported".

Scope: one patched Maki PID owns independent concurrent ACP sessions with
separate cwd/model/plan context, permission routing and supervised lifecycle.
Out of scope here: the runtime itself (02.2a onward), model discovery (04.1)
and owned transcript deletion (05.2); both reserve names below and nothing
else.

## Capability advertisement

Standard ACP methods keep their standard shapes (`initialize`, `session/new`,
`session/load`, `session/list`, `session/prompt`, `session/set_mode`,
`session/set_config_option`, `session/cancel`). Extension support is a
capability, never a version-string comparison: the client checks for the key,
and probes any method it intends to call. An unknown or unimplemented method
answers `-32601`, which the client treats as "unsupported", not as a failure.

`initialize` result, `agentCapabilities._meta`:

```json
{
  "maki.multiSession": {
    "version": 1,
    "methods": [
      "_maki/session/allocate",
      "_maki/session/activate",
      "_maki/session/cancel",
      "_maki/session/release"
    ]
  }
}
```

`_meta` is the ACP-sanctioned extension point
(`agent-client-protocol-schema` types it as a free-form JSON map), so stock
clients ignore the key while fork clients branch on its presence. `version`
is informational; clients MUST NOT gate on it. Reserved keys
`maki.modelDiscovery` (04.1) and `maki.ownedDeletion` (05.2) are absent until
those slices land.

## Identity

- `sessionId`: fork-allocated opaque id, `MakiId` text form. Never reused
  within a process; a late answer for a closed session can never name a live
  one.
- `turnId`: `t-<n>`, `n` counting `session/prompt` calls per session from 0.
  Diagnostic only; settlement keys on the epoch, not the turn id.
- `turnEpoch`: `u64` per session, 0 at allocate/activate. Every
  server-to-client turn-scoped frame (session updates for that session,
  `session/request_permission`, the prompt response) carries the epoch it was
  issued under in `_meta: {"maki.turnEpoch": n}`; every client answer carries
  the epoch it answers. See "Settlement" below.

## Methods

### `_maki/session/allocate`

Reserves an id with no provider spin-up and no task spawn, so the routing
table can name a session before anything slow exists. Setup that can block
(provider construction, MCP start) happens in `activate`, off the routing
loop.

Request params:

```json
{ "cwd": "/absolute/fixture/project-a" }
```

- `cwd`: absolute path, SHOULD exist. A missing folder is `invalid_params`,
  not a prompt: the server never asks.

Result:

```json
{ "sessionId": "CgbPLtxCvEPvJvwaGMUWo", "turnEpoch": 0 }
```

### `_maki/session/activate`

Binds the allocated id to a provider, a fixed mode/cwd/absolute output path,
and starts its task. Mode, cwd and output are fixed at activation; later
`session/set_mode` / `session/set_config_option` for this session answer
`invalid_params` with `data: {"reason": "fixed_at_activation"}` (standard
single-session clients keep the old behavior; this rule is extension-only).

Request params:

```json
{
  "sessionId": "CgbPLtxCvEPvJvwaGMUWo",
  "model": "openai/gpt-5",
  "mode": "build",
  "cwd": "/absolute/fixture/project-a",
  "output": "/absolute/fixture/project-a/.maki/plan.md"
}
```

- `model`: full `provider/model-id` spec, required. The requested model is
  constructed before any provider startup, so allocation order never depends
  on which provider answers first.
- `mode`: `build` or `plan`.
- `cwd`: absolute session directory.
- `output`: absolute path receiving the plan artifact (`plan` mode) or the
  transcript-adjacent run record (`build` mode). Relative paths are
  `invalid_params`; the server resolves nothing against its own process cwd.

Result: `{ "sessionId": "...", "turnEpoch": 0 }`, followed by a
`_maki/session/lifecycle` notification with `state: "active"` once the task
is up. Unknown `sessionId` answers `resource_not_found` (`-32002`) with
`data: {"sessionId": ...}`. A disallowed or unparsable model answers
`invalid_params` with the policy reason in `data`.

### `_maki/session/cancel`

Addressed cancel: session-scoped, optionally turn-pinned. This replaces the
unaddressed `session/cancel` notification for fork clients; the standard
notification is retained and applies to the most recently activated session,
documented as best-effort legacy behavior.

Request params:

```json
{ "sessionId": "CgbPLtxCvEPvJvwaGMUWo", "turnEpoch": 0 }
```

- `turnEpoch` (optional pin): when present and not equal to the session's
  current epoch, the cancel is stale: it answers `invalid_params` with
  `data: {"expectedEpoch": ..., "gotEpoch": ...}` and changes nothing.
  Without the pin, the cancel applies to whatever turn is running.

Result: `{ "sessionId": "...", "cancelled": true, "turnEpoch": 1 }`.
`cancelled: false` means no turn was running; the epoch still bumps (see
Settlement), so a racing terminal cannot settle afterwards. Cancelling an
unknown session is `resource_not_found`.

### `_maki/session/release`

Releases one session; its peers stay usable. The server acks immediately and
tears down in the background: the ack MUST NOT wait for task join, MCP
shutdown or transcript flush (bounded teardown is slice 02.10b).

Request params: `{ "sessionId": "...", "reason": "client_done" }`
(`reason` is a free-form client note, echoed in the lifecycle event).
Result: `{ "sessionId": "...", "released": true, "turnEpoch": 1 }`.
Release is idempotent: releasing an already-released id re-answers success
with the same shape. Releasing an unknown (never-allocated) id is
`resource_not_found`.

## Lifecycle and lost handles

Server-to-client notification `_maki/session/lifecycle`:

```json
{
  "sessionId": "...",
  "state": "allocated",
  "turnEpoch": 0,
  "rawReason": null,
  "usable": true
}
```

`state` is one of `allocated`, `active`, `released`, `lost`. Transitions
`allocated -> active -> released` are client-driven; `lost` is server-driven
only: the session's task or event pump died outside a client release (panic,
supervisor kill, EOF teardown race). A `lost` handle is unusable afterwards;
the client MUST `release` it to drop the routing entry, and any further
`session/prompt` or addressed call on it answers `resource_not_found`.

`rawReason` carries the terminal reason verbatim (provider refusal/filter
text, `Done` reason, loss cause); it is never remapped, so later slices can
preserve provider semantics (02.7) without re-parsing display strings.
`usable` tells the client whether another prompt will be accepted: `false`
for `released`/`lost` and for non-retryable terminals, `true` otherwise. The
standard `PromptResponse.stopReason` keeps its standard values; `rawReason`
/ `usable` are the fork's additive signal.

## Settlement: exactly-once prompts and the turn epoch

  One outstanding prompt per session, several outstanding permission asks.
  Settlement is exactly-once: the first terminal frame for the current epoch
 settles the prompt; every later frame for that epoch is dropped, never
  answered twice. A settling terminal also retires every outstanding ask in
  blocking mode (see 02.8): the ask ids become unknown, so a late answer is
  rejected rather than revived. An addressed cancel (02.9a) retires every
  outstanding ask of the cancelled turn in both modes — the run is stopped, so
  unlike a terminally-settled detached turn nothing outlives it — and marks
  the session non-resumable until 02.9b confirms tool cleanup (see "Cleanup
 confirmation" below). Detached sessions keep their asks across terminal
 settlement; retirement on handle loss (02.10a) is downstream debt.

`TurnEpoch` (`src/extension.rs`) is the single implementation all slices
share:

- `initial()` is 0 at allocate/activate.
- `bump()` runs on every terminal settlement (`Done` / `Error` prompt
  response sent), on an addressed cancel taking effect, and on handle loss.
  Cancelling an idle session still bumps, so a terminal racing the cancel
  lands on a stale epoch and is dropped.
- `check(epoch)` validates late answers and events against the current
  epoch: equal passes, anything else is stale. Stale *events* are dropped
  silently; stale *answers* (permission/elicitation replies, late prompt
  responses) are rejected, never applied, so a stale allow can never green
  a later turn's tool call.

Error rules:

| Condition | Answer |
|---|---|
| Unknown method (incl. all `_maki/` on the current server, reserved 04.1/05.2 names) | `-32601 Method not found`, id-correlated |
| Unknown `sessionId` | `-32002 Resource not found`, `data: {"sessionId"}` |
| Answer/event naming another live session (cross-ID) | `-32602 Invalid params`, `data: {"expectedSession": ..., "gotSession": ...}`; the named session is unaffected |
| Stale epoch on an answer | `-32602 Invalid params`, `data: {"expectedEpoch": ..., "gotEpoch": ...}`; nothing applied |
| Replay of an already-settled answer | dropped; exactly-once preserved |
| Second terminal for one epoch | dropped; the first settlement stands |
| Prompt while a turn runs on that session | `-32603`-class "a prompt is already running" (existing rule, unchanged) |

Wire example (stale cancel pin):

```json
// -> {"jsonrpc":"2.0","id":9,"method":"_maki/session/cancel",
//     "params":{"sessionId":"CgbPLtxCvEPvwaGMUWo","turnEpoch":0}}
// <- {"jsonrpc":"2.0","id":9,"error":{"code":-32602,"message":"Invalid params",
//     "data":{"expectedEpoch":1,"gotEpoch":0}}}
```

## Cleanup confirmation (02.9b)

A cancelled session accepts another prompt only after its addressed turn and
blocking tools have confirmed cleanup; unconfirmed cleanup stays explicitly
non-resumable. The criterion is the cancelled run's terminal as observed by
the session's event pump, and it holds under `--background-timeout 0`
(blocking mode) only: with no seed published no call can move off its turn,
so when the run ends its blocking tools ended with it.

- The addressed cancel settles the prompt at once (answer `cancelled`) and
  marks the session non-resumable. Replacement prompts reject with
  `invalid_params` / `data: {"reason": "cleanup_unconfirmed",
  "sessionId": ..., "uncertain": false}` before anything is queued; the peer
  is untouched. Runtime usability is the send itself: a dead run reports
  "session ended" instead of queueing onto it.
- The cancelled run's `Done` terminal (whatever its reason) confirms cleanup:
  the run ended, so in blocking mode its tools stopped with it. The terminal
  answers nothing — the prompt already settled — and reopens the session for
  exactly one new turn. A duplicate terminal afterwards confirms nothing and
  answers nothing.
- The cancelled run's `Error` terminal is failure, not confirmation: the tools
  may have stopped with it, but the abnormal end proves nothing. The session
  records the uncertainty and stays non-resumable; the rejection then carries
  `"uncertain": true`. Uncertainty never self-heals into retry.
- Bounded wait: a cancelled run that reports no terminal within 30 seconds
  (held cleanup) flips the rejection to `"uncertain": true`. Overdue is a
  report, not a verdict — a late terminal still confirms — and the session
  stays non-resumable either way.
- Late answers for the cancelled turn's retired asks are rejected by the same
  epoch check as ever (stale-turn and cross-session answers never apply), so
  they cannot affect the replacement turn; the replacement's own terminal
  settles exactly once under its own epoch.
- What this criterion does NOT prove: that an escaped tool actually died.
  Process exit or process-group reaping alone never proves a `setsid`-escaped
  tool died — it left the group before the kill. The executable record of
  that limitation is `tests/cleanup_probes.rs`; controlled live proof that
  tools really die is 02.11's, and final independent-cancellation acceptance
  stays gated on it.
- `session/load` replaces the runtime wholesale (abort, not confirmation) and
  is not a confirmation path; its teardown bounds belong to 02.10a/02.10b.

## Routing-loop rule

Slow setup (provider construction, MCP start, history replay) and slow
release (task join, MCP shutdown, flush) MUST NOT block the shared routing
loop: allocate/activate/release ack from the loop and do the slow work on
spawned tasks, reporting readiness via `_maki/session/lifecycle`. The loop
itself only matches ids and epochs.

## Reservations (named only, not implemented)

- 04.1 model discovery: capability `_meta["maki.modelDiscovery"]`, methods
  under `_maki/model/`. Probing any of them today answers `-32601`.
- 05.2 owned transcript deletion: capability `_meta["maki.ownedDeletion"]`,
  method `_maki/session/delete`. Probing it today answers `-32601`.

## Deferred to later slices

02.2a registry/ownership, 02.2b routing, 02.3 allocation-vs-activation
ordering, 02.4 session context, 02.5a–c runtime cwd, 02.6 plan inheritance,
 02.7 terminal reasons, 02.8 permission routing, 02.9a addressed
  cancellation, 02.9b cleanup confirmation and retry, 02.10a–b
 supervision/release bounds, 02.11 engine acceptance. Of the above, slices
 through 02.9b are implemented (active-session release still answers
 `-32601`, as do the reserved 04.1/05.2 names); the harness in `tests/`
 speaks standard ACP plus `-32601` probes so the same fixtures run against
 the extension as each slice lands.
