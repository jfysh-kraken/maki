//! 02.9b escaped-child limitations and process-control scaffolding for 02.11.
//!
//! What this proves, with the real [`ChildGuard`] primitive and only local
//! `sleep`/`sh` processes (no provider, no keys, no network):
//!
//! - killing a child's process group reaps a grouped child;
//! - the same kill does NOT reach a grandchild that escaped via `setsid`.
//!
//! That second bullet is the recorded limitation behind 02.9b's cleanup
//! criterion: the pump confirms cleanup when the cancelled run's terminal
//! arrives, but process exit or process-group reaping alone never proves an
//! escaped `setsid` tool died — it left the group before the kill. 02.11
//! must not convert such a reap into cleanup proof; its controlled live
//! evidence has to account for escapees explicitly (direct-pid lifecycle,
//! not group membership).
//!
//! 02.11 reuse: `spawn_grouped`, `is_alive` and `wait_for_death` are the
//! process-control half of the controlled live tool/cancellation probes —
//! start a controlled tool, cancel mid-tool, assert the lifecycle by pid.
//! The ACP half (send prompt, cancel mid-turn, correlate the `cancelled`
//! answer and the run's terminal, retry) rides on
//! `support::MakiChild::send_prompt`, which 02.9b added without calling.

#![cfg(unix)]

use std::os::unix::process::CommandExt;
use std::time::{Duration, Instant};

use maki_agent::ChildGuard;
use rustix::process::{Pid, Signal, kill_process, setsid, test_kill_process};

/// Generous on purpose, matching the repo's fixture style: the work under
/// test is local process control, so any wait near this long is the bug and
/// not a slow machine.
const PROBE_DEADLINE: Duration = Duration::from_secs(10);
const POLL_STEP: Duration = Duration::from_millis(10);

fn raw(pid: u32) -> Option<Pid> {
    i32::try_from(pid).ok().and_then(Pid::from_raw)
}

fn is_alive(pid: u32) -> bool {
    raw(pid).is_some_and(|pid| test_kill_process(pid).is_ok())
}

/// `kill(pid, 0)` still succeeds on a zombie, so await between polls: the
/// runtime reaps the child once it is dead.
async fn wait_for_death(pid: u32, what: &str) {
    let deadline = Instant::now() + PROBE_DEADLINE;
    while Instant::now() < deadline {
        if !is_alive(pid) {
            return;
        }
        tokio::time::sleep(POLL_STEP).await;
    }
    panic!("{what}: process {pid} still alive after {PROBE_DEADLINE:?}");
}

/// Spawn a program as a session/group leader, the way the MCP transport
/// spawns its servers (see `maki-agent/src/mcp/transport.rs`): the child
/// starts its own session, so the group's kill is scoped to it.
fn spawn_grouped(argv0: &str, args: &[&str]) -> tokio::process::Child {
    let mut std_cmd = std::process::Command::new(argv0);
    std_cmd.args(args);
    std_cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    unsafe {
        std_cmd.pre_exec(|| {
            let _ = setsid();
            Ok(())
        });
    }
    tokio::process::Command::from(std_cmd)
        .spawn()
        .unwrap_or_else(|e| panic!("cannot spawn {argv0} for the cleanup probe: {e}"))
}

/// The controlled case 02.11 may rely on: a grouped child dies with its
/// group and is reaped exactly once.
#[tokio::test]
async fn process_group_kill_reaps_a_grouped_child() {
    let child = spawn_grouped("sleep", &["60"]);
    let pid = child.id().expect("a spawned child has a pid");
    assert!(is_alive(pid), "the probe child starts alive");

    let mut guard = ChildGuard::new(child);
    guard.kill_and_reap().await;
    wait_for_death(pid, "group-killed sleep").await;
    assert!(
        guard.status().await.is_err(),
        "the group kill reaps the child exactly once"
    );
}

/// The recorded limitation 02.11 must not reason around: a tool that escapes
/// via `setsid` leaves the process group before the kill, so reaping the
/// group proves nothing about it. The inner `sh` prints its own pid and
/// `exec`s into `sleep`, so the pid file names the escapee deterministically
/// (no fork in between: `exec` keeps the pid).
#[tokio::test]
async fn a_setsid_escaped_grandchild_survives_the_group_kill() {
    let dir = tempfile::TempDir::new().expect("a probe scratch dir");
    let pidfile = dir.path().join("escaped.pid");
    let script = format!(
        "setsid sh -c 'echo $$; exec sleep 60' > {} ; wait",
        pidfile.display()
    );
    let child = spawn_grouped("sh", &["-c", &script]);
    let pid = child.id().expect("a spawned shell has a pid");

    let escaped = {
        let deadline = Instant::now() + PROBE_DEADLINE;
        loop {
            if let Ok(text) = std::fs::read_to_string(&pidfile)
                && let Ok(escaped) = text.trim().parse::<u32>()
            {
                break escaped;
            }
            if Instant::now() >= deadline {
                panic!("the escaped tool never reported its pid; the probe script failed");
            }
            tokio::time::sleep(POLL_STEP).await;
        }
    };
    assert_ne!(escaped, pid, "the escapee is not the grouped child");
    assert!(is_alive(escaped), "the escapee starts alive");

    // The recorded kill primitive reaps the whole group...
    let mut guard = ChildGuard::new(child);
    guard.kill_and_reap().await;
    wait_for_death(pid, "group-killed shell").await;

    // ...but the escapee, reparented into its own session, is untouched.
    // Process exit / group reaping is NOT cleanup proof for escaped tools.
    assert!(
        is_alive(escaped),
        "an escaped setsid tool survives process-group reaping"
    );

    // Hygiene by direct pid: precise, and proves nothing about groups.
    kill_process(raw(escaped).expect("a live pid"), Signal::KILL).expect("kill the escapee");
    wait_for_death(escaped, "direct-killed escapee").await;
}
