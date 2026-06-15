"""Agent daemon wrapper around a Hermes AIAgent instance."""

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from collections import deque

from swarm_server.config import (
    AUTONOMOUS_HEARTBEAT_SECONDS,
    DEFAULT_MAX_ITERATIONS,
    HEARTBEAT_BACKOFF_MAX_DOUBLINGS,
    LITELLM_API_BASE,
    LLM_API_KEY,
    LLM_ERROR_EMIT_THROTTLE_SECONDS,
    MAX_BATCH_SIZE,
    MAX_TASK_RETRIES,
    SELF_LOOP_COOLDOWN_SECONDS,
    SELF_LOOP_REPEATS,
    SELF_LOOP_WINDOW,
    SUPERVISOR_FEED_CHAR_CAP,
    SUPERVISOR_SWEEP_CHAR_CAP,
    SUPERVISOR_SWEEP_FORCE_OPEN_AGE_SECONDS,
    SUPERVISOR_SWEEP_INTERVAL_MINUTES,
    SUPERVISOR_SWEEP_MAX_IDLE_SKIPS,
    SUPERVISOR_SWEEP_PER_PEER_FLOOR,
    SWEEP_INTERVAL_SECONDS,
    SWEEP_PAYLOAD_AGE_ENABLED,
    SWEEP_PAYLOAD_KEEP_RECENT,
    SWEEP_PAYLOAD_MIN_CHARS,
    TOOL_RESULT_AGE_ENABLED,
    TOOL_RESULT_AGE_KEEP_MESSAGES,
    TOOL_RESULT_AGE_MIN_CHARS,
    TOOL_RESULT_AGE_QUANTUM,
    _derive_workspace_path,
    _ensure_project_dir,
    load_agents_config,
    save_agent_config,
    write_agent_hermes_config,
)
from swarm_server.prompts import (
    AUTONOMOUS_HEARTBEAT_PROMPT,
    CRON_WAKEUP_PROMPT,
    DIRECTIVE_HEARTBEAT_PROMPT,
    MISSING_RESULT_NUDGE,
    SELF_LOOP_NUDGE,
    SUPERVISOR_SWEEP_PROMPT,
    TEXT_ONLY_TURN_NUDGE,
    age_stale_sweep_payloads,
    age_stale_tool_results,
    compose_agent_soul,
    compose_live_context,
    compose_soul_identity,
    strip_stale_live_context,
)
from swarm_server.browser_pool import team_browser_manager
from swarm_server.monitoring import monitor_db
from swarm_server.queue import TaskQueue
from swarm_server.tools import (
    _ASK_HUMAN_TOOL_SCHEMA,
    _SEND_PEER_MESSAGE_TOOL_SCHEMA,
    _daemon_registry,
    _register_custom_tools,
)
from swarm_server.websocket import _agent_init_lock, _broadcast, ws_broadcaster

log = logging.getLogger("swarm.agent")

AGENT_STATE_IDLE = "idle"
AGENT_STATE_BUSY = "busy"
AGENT_STATE_ASKING_HUMAN = "asking_human"
AGENT_STATE_PAUSED = "paused"

# Stable substring of every task-prompt preamble. Used to locate this turn's
# output boundary in the returned message list (see _process_tasks_batch), so it
# must stay in sync with the `combined` preamble that carries it.
_TASK_PROMPT_MARKER = "new message(s) to process"

# Backoff between retries of a turn that failed for INFRA reasons (provider down,
# billing, network). Grows per consecutive miss so a sustained outage doesn't
# spin the sweep loop; capped so recovery is still picked up promptly.
INFRA_RETRY_BACKOFF_BASE_SECONDS = 15.0
INFRA_RETRY_BACKOFF_MAX_SECONDS = 300.0

# How long a stop waits for an interrupted turn's worker thread to unwind before
# giving up and proceeding. interrupt() only lands at the next tool boundary, so
# a turn wedged in a long blocking tool call can exceed this; we bound the wait
# so the operator's stop never hangs.
STOP_DRAIN_TIMEOUT_SECONDS = 30.0

# Senders that are control-plane injections, not real work arriving. A message
# from anyone else (a peer, a human) is "real activity" and resets the idle
# heartbeat backoff.
_SYSTEM_SENDERS = frozenset(
    {"autonomous", "cron", "turn-guard", "self-loop-guard", "supervisor-feed",
     "supervisor-sweep", "loop_detector"}
)


def _turn_tool_signatures(turn_messages: List[Dict[str, Any]]) -> List[str]:
    """Normalized signatures of every tool call an assistant made this turn.

    'name(sorted-args-prefix)' — stable across formatting noise so the same
    call with the same arguments compares equal across turns."""
    sigs: List[str] = []
    for m in turn_messages or []:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            name = fn.get("name") or "?"
            args = fn.get("arguments")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, sort_keys=True, default=str)
                except Exception:
                    args = str(args)
            norm = " ".join((args or "").split())[:160]
            sigs.append(f"{name}({norm})")
    return sigs


def detect_repeated_signature(
    turn_sig_history, repeats: int = SELF_LOOP_REPEATS, window: int = SELF_LOOP_WINDOW,
) -> Optional[str]:
    """The self-loop check: a tool signature occurring in >= `repeats` of the
    last `window` turns (counted once per turn). Within-turn repeats don't
    count — retrying inside one turn is normal; re-issuing the identical call
    turn after turn is the degenerate loop the team-level detectors can't see.
    Returns the most-repeated signature, or None."""
    recent = [set(s) for s in list(turn_sig_history)[-window:]]
    if len(recent) < repeats:
        return None
    counts: Dict[str, int] = {}
    for sigset in recent:
        for s in sigset:
            counts[s] = counts.get(s, 0) + 1
    best: Optional[str] = None
    best_n = 0
    for s, c in counts.items():
        if c >= repeats and c > best_n:
            best, best_n = s, c
    return best

def _age_short(seconds: float) -> str:
    """'47m' / '3h12m' — compact age for supervisor-sweep ledger lines."""
    s = int(max(0, seconds))
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def compose_sweep_sections(peer_data: List[Dict[str, Any]], char_cap: int,
                           per_peer_floor: int) -> str:
    """Render the per-agent sections of a supervisor sweep.

    ``peer_data``: one dict per linked peer — {peer, state, pending, transcript,
    signal, messages, tokens}. The total char budget is split across the peers
    that were ACTIVE this window (each keeps its most-recent slice, truncation
    marked); silent peers cost a header line and are reported explicitly, so
    silence-while-owing-work is visible instead of missing. Pure function —
    unit-testable without a daemon.
    """
    active = [d for d in peer_data if d.get("transcript")]
    slice_cap = max(per_peer_floor, char_cap // max(1, len(active)))
    out = []
    for d in peer_data:
        state = (d.get("state") or "?").upper()
        if not d.get("transcript"):
            queued = " — but it HAS queued work waiting" if d.get("pending") else ""
            out.append(
                f"=== {d['peer']} — NO ACTIVITY this window [{state}]{queued} ===\n"
                "(nothing since your last sweep — if the LEDGER shows it owes an "
                "open delegation, that silence IS the finding)"
            )
            continue
        text = d["transcript"]
        if len(text) > slice_cap:
            text = ("[…older activity in this window truncated…]\n\n"
                    + text[-slice_cap:])
        mid_turn = (" [BUSY — may be MID-TURN; its partial turn so far is included]"
                    if state == AGENT_STATE_BUSY.upper() else f" [{state}]")
        out.append(
            f"=== {d['peer']} — {d.get('messages', 0)} message(s), "
            f"~{d.get('tokens', 0)} tokens this window{mid_turn} ===\n"
            f"{d.get('signal') or ''}\n\n{text}"
        )
    return "\n\n".join(out)


def _ensure_hermes_on_path() -> None:
    """Make the Hermes package importable (pip install / PYTHONPATH / checkout).

    Delegates to config.ensure_hermes_importable so resolution is defined in one
    place (see its docstring for the lookup order)."""
    from swarm_server.config import ensure_hermes_importable

    ensure_hermes_importable()


def _set_hermes_home_override(home: Any) -> Optional[Any]:
    """Set the context-local HERMES_HOME override on the *current* thread.

    Hermes exposes a ContextVar-based override (set/reset) whose whole purpose is
    in-process, per-task scoping that — unlike os.environ — is NOT shared across
    threads. We set it on each agent's dedicated worker thread so concurrent
    run_conversation calls resolve get_hermes_home() to their own home instead of
    racing on the process-global env var. Returns a reset token, or None if the
    Hermes API is unavailable (in which case we fall back to the os.environ value
    set during init).
    """
    _ensure_hermes_on_path()
    try:
        from hermes_constants import set_hermes_home_override

        return set_hermes_home_override(str(home))
    except Exception as e:  # pragma: no cover - depends on Hermes version
        log.debug("HERMES_HOME context override unavailable: %s", e)
        return None


def _reset_hermes_home_override(token: Optional[Any]) -> None:
    if token is None:
        return
    try:
        from hermes_constants import reset_hermes_home_override

        reset_hermes_home_override(token)
    except Exception:  # pragma: no cover
        pass


# ── Per-thread TERMINAL_CWD ───────────────────────────────────────────────
# Hermes resolves the terminal/file working directory from os.getenv(
# "TERMINAL_CWD") at *tool-call time* (terminal_tool, tool_executor, file_tools)
# and — unlike HERMES_HOME — exposes NO ContextVar override for it. A single
# process-global env var means the last agent to set it wins, so a peer agent's
# terminal/file ops would run in the wrong team's repo while another team's turn
# is in flight. We make TERMINAL_CWD resolve *per worker thread* instead: each
# agent's dedicated worker thread sets its own value (see _run_conversation_
# blocking), and every os.getenv/os.environ.get read returns that thread's value.
_terminal_cwd_tls = threading.local()
_TERMINAL_CWD_KEY = "TERMINAL_CWD"


class _ThreadAwareEnviron(type(os.environ)):
    """os.environ that resolves TERMINAL_CWD from a per-thread override.

    All Hermes readers use ``.get()`` / ``os.getenv()`` (never subscript), both of
    which funnel through ``__getitem__``; overriding it covers every reader while
    leaving writes, subprocess inheritance, and all other keys untouched.
    """

    def __getitem__(self, key):
        if key == _TERMINAL_CWD_KEY:
            v = getattr(_terminal_cwd_tls, "value", None)
            if v is not None:
                return v
        return super().__getitem__(key)


def _install_thread_aware_terminal_cwd() -> None:
    cur = os.environ
    if isinstance(cur, _ThreadAwareEnviron):
        return
    try:
        os.environ = _ThreadAwareEnviron(
            cur._data, cur.encodekey, cur.decodekey, cur.encodevalue, cur.decodevalue
        )
    except Exception as e:  # pragma: no cover - defensive; keep the global env usable
        log.warning("Could not install thread-aware TERMINAL_CWD (%s)", e)


_install_thread_aware_terminal_cwd()


def _set_terminal_cwd_override(path: Any) -> None:
    """Pin TERMINAL_CWD for the current (worker) thread only."""
    _terminal_cwd_tls.value = str(path)


def _reset_terminal_cwd_override() -> None:
    _terminal_cwd_tls.value = None


class AgentDaemon:
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.state = AGENT_STATE_IDLE
        self._lock = threading.Lock()

        workspace_dir = _derive_workspace_path(cfg.get("team_id", "default"), name)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = workspace_dir / f"{name}_queue.db"
        self.queue = TaskQueue(db_path)
        # Recover tasks stranded 'processing' by a previous run (crash/restart).
        recovered = self.queue.recover_processing()
        if recovered:
            log.info("[%s] Recovered %d in-flight task(s) from previous run", name, recovered)

        # Each agent gets its own isolated Hermes home
        self._hermes_home = workspace_dir / ".hermes"
        self._hermes_home.mkdir(parents=True, exist_ok=True)

        self._ai_agent = None
        # Set by config/soul changes to force a CLEAN re-init at the START of the
        # next turn (on the worker thread) instead of nulling _ai_agent from the
        # event-loop thread mid-turn — which would crash the in-flight turn at
        # self._ai_agent.run_conversation and lose a compaction session rotation.
        self._reinit_requested = False
        # Static half of the ephemeral system prompt; set in _ensure_agent and
        # combined with per-turn live context before each run.
        self._base_ephemeral: Optional[str] = None
        # Guard so the "you ended in the chat" corrective fires at most once per
        # occurrence (never loops): set when nudged, cleared when a turn delivers.
        self._text_only_nudged: bool = False
        # Delegated-task ids already nudged for a missing RESULT (once each).
        self._result_nudged_ids: set = set()
        self._sweep_task: Optional[asyncio.Task] = None
        # Event-driven wake: ingest_task signals this so the sweep loop processes
        # immediately instead of waiting out the poll interval. Created in
        # start_sweep() where the running loop is available.
        self._wake: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Each agent runs its (blocking) Hermes conversation on its OWN single
        # worker thread. This (a) isolates a blocking ask_human wait to this one
        # agent so it can never starve the shared default thread pool that other
        # agents rely on, and (b) gives this agent a stable thread whose
        # contextvars (HERMES_HOME override) are independent of every other
        # agent. max_workers=1 also serializes this agent's own runs.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"agent-{name}"
        )

        self.human_event = threading.Event()
        self.human_response = None
        self.next_sweep_at = 0.0
        self._stop_requested = False
        # Emergency brake. When True the sweep loop processes NO work and fires no
        # crons/heartbeat, but — unlike stop — the pending queue is PRESERVED, so
        # the agent resumes exactly where it was. Set by a supervisor's pause_agent
        # tool or a human via /agent/{name}/pause; cleared by resume.
        self._paused = False
        self._pause_reason = ""
        self._paused_by = ""
        # Hermes reports session-CUMULATIVE token counts; we track the last
        # totals so each batch can log its real per-turn deltas (not just a
        # char estimate). All four counters share one reset rule: a shrinking
        # cumulative total means the session rotated/compacted.
        self._last_total_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_cache_read_tokens = 0
        # Resolved model name (set in _ensure_agent) — logged with every
        # token_usage event so cost attribution survives mid-life model swaps.
        self._current_model = ""
        # 24/7 autonomy: when True, this daemon self-injects a continue-mission
        # task after AUTONOMOUS_HEARTBEAT_SECONDS of idle (empty queue). Set per
        # agent via cfg["autonomous"] — typically only the team coordinator.
        self._autonomous = bool(cfg.get("autonomous", False))
        # Wall-clock of the last time this agent actually did work. Seeds to now
        # so a freshly-started autonomous agent waits one full interval before
        # its first self-driven cycle (gives a human time to send the opener).
        self._last_active = time.time()
        # Throttle for the "LLM provider unreachable" UI message so a sustained
        # outage doesn't post one error per 10s sweep tick.
        self._last_llm_error_emit = 0.0
        # Infra-failure retry backoff. requeue_no_penalty doesn't burn the retry
        # budget, but the sweep's "more pending? wake now" path would otherwise
        # re-attempt instantly — a tight loop hammering a down provider. Hold off
        # draining until this wall-clock time; the wait grows per consecutive miss.
        self._infra_hold_until = 0.0
        self._infra_misses = 0
        # Per-agent sweep interval (falls back to the global default). Lets the UI
        # tune how often a specific agent polls its queue.
        self._sweep_interval = self._resolve_sweep_interval(cfg)
        # Per-agent autonomous-heartbeat interval (falls back to the global
        # default). Configurable from the UI so each agent's 24/7 cadence is tuned
        # independently of the launch-time SWARM_HEARTBEAT_SECONDS.
        self._heartbeat_seconds = self._resolve_heartbeat_interval(cfg)
        # Scheduled cron wake-ups. self._crons is the config list; self._cron_next
        # maps cron-id -> next fire timestamp; self._cron_last -> last fire ts;
        # self._cron_sched remembers each cron's schedule so a config reload only
        # recomputes the next-fire time for schedules that actually changed.
        self._crons: List[Dict[str, Any]] = []
        self._cron_next: Dict[str, float] = {}
        self._cron_last: Dict[str, float] = {}
        self._cron_sched: Dict[str, str] = {}
        # Fire count this process (for the supervisor's cron view); bounded crons
        # also persist their count via config.record_cron_fire so a one-shot stays
        # done across restarts.
        self._cron_runs: Dict[str, int] = {}
        self._load_crons(cfg)
        # Monotonic sequence for ephemeral live-execution events (exec_*). Lets the
        # dashboard order/dedupe the streamed thinking/tool/answer steps per turn.
        self._exec_seq = 0
        # Latest runtime telemetry (context usage, token spend, window, threshold),
        # refreshed at the end of each turn and surfaced read-only in the UI.
        self._telemetry: Dict[str, Any] = {}
        # Hash of the last injected system context we logged, so we record it to
        # the transcript only when it actually changes (not on every turn).
        self._last_sysctx_hash: Optional[int] = None
        # Supervisor agents only: per-linked-peer high-water mark (last monitoring
        # message id already covered by a sweep). Lazy-initialized to the peer's
        # current latest id, so a supervisor covers NEW activity only and never
        # gets a peer's entire backlog dumped into its queue at once.
        self._sup_watermark: Dict[str, int] = {}
        # Supervisor sweep clock: the interval-sweep mechanism replaces the old
        # token-threshold reviews. Seeded to "now" so the first sweep covers
        # daemon-start → first tick, never history.
        self._last_sweep_ts = time.time()
        # Idle-skip state. _last_sweep_check_ts is the cadence clock (a skip
        # must wait a full interval before re-checking — without it a skip
        # re-evaluates every daemon tick and fires seconds after the first new
        # message); _last_sweep_ts stays the WINDOW anchor so a post-skip sweep
        # honestly reports the longer window. _sup_idle_skips counts consecutive
        # skips toward the forced idle backstop; _last_delegation_force_ts
        # rate-limits the stale-delegation early fire.
        self._last_sweep_check_ts = 0.0
        self._sup_idle_skips = 0
        self._last_delegation_force_ts = 0.0
        # Idle-heartbeat backoff: consecutive heartbeat turns that produced no
        # concrete action. Effective interval = base * 2**min(misses, cap), so a
        # genuinely-idle 24/7 agent costs exponentially less instead of burning
        # a full turn every interval forever; any real work or inbound message
        # resets it to the base cadence.
        self._hb_misses = 0
        # Active time-boxed directive (set via POST /agent/{name}/task with
        # duration_minutes). While set and unexpired, idle heartbeats re-present
        # it at the BASE cadence (no backoff) instead of the generic check-in.
        # In-memory only: a server restart ends the push, which is acceptable.
        self._directive: Optional[Dict[str, Any]] = None
        # Cross-turn repetition guard: per-turn tool-call signature sets for the
        # last few turns, plus the last time a corrective was injected.
        self._turn_sigs: deque = deque(maxlen=max(SELF_LOOP_WINDOW, 8))
        self._last_self_loop_nudge = 0.0
        # Passive-message delivery watermark (events.id). Seeded to "now" so the
        # agent's next turn delivers only STATUS/FYI that arrive from here on,
        # never a historical backlog.
        try:
            self._passive_watermark = monitor_db.get_latest_event_id()
        except Exception:
            self._passive_watermark = 0

    @staticmethod
    def _resolve_sweep_interval(cfg: Dict[str, Any]) -> float:
        try:
            v = float(cfg.get("sweep_interval") or 0)
            return v if v >= 1 else float(SWEEP_INTERVAL_SECONDS)
        except (TypeError, ValueError):
            return float(SWEEP_INTERVAL_SECONDS)

    @staticmethod
    def _resolve_heartbeat_interval(cfg: Dict[str, Any]) -> float:
        """Per-agent idle-heartbeat interval, falling back to the global default.
        Clamped to a 60s floor so a typo can't spin the agent every few seconds."""
        try:
            v = float(cfg.get("heartbeat_seconds") or 0)
            return v if v >= 60 else float(AUTONOMOUS_HEARTBEAT_SECONDS)
        except (TypeError, ValueError):
            return float(AUTONOMOUS_HEARTBEAT_SECONDS)

    def _load_crons(self, cfg: Dict[str, Any]) -> None:
        """(Re)load cron wake-ups from config, preserving the next-fire time of any
        schedule that didn't change so an unrelated config save can't reset timers."""
        from swarm_server.cron import cron_next

        crons = list(cfg.get("crons") or [])
        now = time.time()
        new_next: Dict[str, float] = {}
        new_sched: Dict[str, str] = {}
        for c in crons:
            cid = c.get("id")
            sched = c.get("schedule") or ""
            if not cid or not c.get("enabled", True):
                continue
            new_sched[cid] = sched
            # Keep the existing next-fire time iff this schedule is unchanged.
            if cid in self._cron_next and self._cron_sched.get(cid) == sched:
                new_next[cid] = self._cron_next[cid]
                continue
            try:
                nxt = cron_next(sched, now)
            except Exception as e:  # noqa: BLE001 — a bad schedule must not crash load
                log.warning("[%s] cron '%s' (%s) skipped: %s", self.name, cid, sched, e)
                nxt = None
            if nxt is not None:
                new_next[cid] = nxt
        self._crons = crons
        self._cron_next = new_next
        self._cron_sched = new_sched
        # Drop last-fired entries for crons that no longer exist.
        self._cron_last = {k: v for k, v in self._cron_last.items() if k in new_sched}

    def crons_runtime(self) -> List[Dict[str, Any]]:
        """Cron entries enriched with live next/last-fire timestamps, for the UI."""
        out = []
        for c in self._crons:
            cid = c.get("id")
            out.append({
                **c,
                "next_fire_at": self._cron_next.get(cid),
                "last_fired_at": self._cron_last.get(cid),
                # Live fire count (persisted value if present, else this process's).
                "runs": int(c.get("runs", 0)) or self._cron_runs.get(cid, 0),
            })
        return out

    @staticmethod
    def _is_infra_failure(err: str) -> bool:
        """True when a failed turn is environmental (provider down, billing,
        timeout) rather than the task's fault — these should wait for recovery
        instead of burning the retry budget and dead-lettering during an outage."""
        e = (err or "").lower()
        # Deterministic, task-caused failures are excluded FIRST even when their
        # text happens to contain an infra-ish word (e.g. 'LLM call failed after 3
        # attempts: maximum context length exceeded') — otherwise a poison batch
        # would be requeued penalty-free forever and never reach the dead-letter.
        deterministic = (
            "context length", "context_length", "maximum context", "too many tokens",
            "max_tokens", "string too long", "invalid request", "invalid_request",
            "bad request", "400", "401", "403", "404", "422",
            "content policy", "content_policy", "content filter", "moderation",
            "unsupported", "not found", "does not exist", "no such model",
        )
        if any(s in e for s in deterministic):
            return False
        return any(s in e for s in (
            "connection error", "apiconnection", "connection refused",
            "billing or credits", "credits exhausted", "timeout", "timed out",
            "max retries", "failed after", "service unavailable", "502", "503", "504",
            "rate limit", "overloaded", "temporarily unavailable", "econnreset",
        ))

    def _write_soul_md(self, content: str) -> None:
        """Atomically write this agent's SOUL.md (its lead identity block).

        Overwrites the generic SOUL.md Hermes auto-seeds into a fresh
        HERMES_HOME so the cached system prompt leads with the agent's ROLE
        instead of the stock "You are Hermes Agent…" template. Atomic so a
        concurrent AIAgent init can never read a half-written file.
        """
        soul_path = self._hermes_home / "SOUL.md"
        try:
            self._hermes_home.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._hermes_home), prefix=".SOUL.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, soul_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            log.warning("[%s] Could not write SOUL.md: %s", self.name, e)

    def _ensure_agent(self):
        # A pending re-init request (config/soul change) rebuilds the agent here,
        # at the start of the turn on the worker thread, rather than the requester
        # nulling _ai_agent mid-turn from another thread.
        if self._ai_agent is not None and not self._reinit_requested:
            return
        with _agent_init_lock:
            if self._ai_agent is not None and not self._reinit_requested:
                return
            self._reinit_requested = False
            self._ai_agent = None
            try:
                _ensure_hermes_on_path()
                from run_agent import AIAgent

                os.environ["HERMES_HOME"] = str(self._hermes_home)

                # Bring up this team's shared, persistent browser and point this
                # agent at it via browser.cdp_url. All agents in the team share
                # one Chrome (same cookies/logins) whose profile lives on disk,
                # so the session survives restarts. Best-effort: if no Chromium
                # is available, cdp_url stays None and the agent falls back to
                # the per-agent local browser.
                team_id = self.cfg.get("team_id", "default")

                # Point file tools + terminal at the team's ONE shared project
                # directory. TERMINAL_CWD is what the terminal tool and relative
                # file ops resolve against (Hermes reads it at tool-call time), so
                # every agent on the team operates in the same real repo instead of
                # a private folder. Also creates + git-inits the dir on first use.
                # Scoped per worker thread (see _set_terminal_cwd_override) so a
                # concurrent peer turn can't redirect this agent's ops; pinned again
                # here for the AIAgent init that reads it (agent_init.working_dir).
                project_dir = _ensure_project_dir(team_id)
                self._project_dir = project_dir
                _set_terminal_cwd_override(project_dir)

                cdp_url = team_browser_manager.ensure_team_browser(team_id)

                # Per-agent model + sampling knobs (configurable from the UI;
                # fall back to the proxy default / Hermes defaults when unset).
                # Effective backend: per-agent override → swarm default → proxy.
                from swarm_server.model_config import resolve_model

                eff = resolve_model(self.cfg)
                model = eff["model"]
                self._current_model = model
                _eff_base = eff["base_url"] or None
                _eff_key = eff["api_key"] or None
                _eff_provider = eff["provider"] or None
                log.info("[%s] model: %s (provider=%s, source=%s)",
                         self.name, model, eff["provider"], eff["source"])
                _ct = self.cfg.get("compression_threshold")
                try:
                    compression_threshold = float(_ct) if _ct not in (None, "") else None
                except (TypeError, ValueError):
                    compression_threshold = None

                # Enable Hermes' built-in context/session compression for this
                # agent. Must be written before AIAgent() so init_agent picks it
                # up from {HERMES_HOME}/config.yaml. The resolved backend is pinned
                # here too so the default + auxiliary tasks match the agent's model.
                write_agent_hermes_config(
                    self._hermes_home, cdp_url=cdp_url, model=model,
                    compression_threshold=compression_threshold,
                    provider=eff["provider"], base_url=eff["base_url"], api_key=eff["api_key"],
                    is_supervisor=bool(self.cfg.get("is_supervisor")),
                )

                # CRITICAL: bind the session DB to THIS agent's home explicitly.
                # Hermes computes `DEFAULT_DB_PATH = get_hermes_home()/state.db`
                # ONCE at module-import time, and `SessionDB()` with no arg uses
                # that frozen constant — it ignores both os.environ["HERMES_HOME"]
                # AND the ContextVar override. Without an explicit path, every
                # agent in a server run writes to whichever home was active at
                # first import, so sessions cross-contaminate and an agent loses
                # its history on restart (observed: content-writer and
                # social-media-manager had NO state.db of their own — their turns
                # were stranded in cmo's/seo's DBs). Passing session_db pins each
                # agent to its own state.db, race-free.
                agent_session_db = None
                try:
                    from hermes_state import SessionDB

                    agent_session_db = SessionDB(db_path=self._hermes_home / "state.db")
                except Exception as e:
                    log.error(
                        "[%s] Could not open isolated SessionDB (falling back to default): %s",
                        self.name, e,
                    )

                # Prepare config with agent_id for soul composition
                soul_cfg = dict(self.cfg)
                soul_cfg["agent_id"] = self.name
                full_cfg = load_agents_config()

                # Write this agent's ROLE as SOUL.md so Hermes loads it as the
                # lead identity block of the (cached) system prompt — replacing
                # the generic auto-seeded "You are Hermes Agent…" template that
                # Hermes drops into every fresh HERMES_HOME. Must be written
                # before AIAgent() so load_soul_md() picks it up. The role is
                # therefore NOT repeated in the ephemeral prompt (include_role
                # =False) to avoid duplicating it in every turn.
                self._write_soul_md(compose_soul_identity(soul_cfg))

                # Static half of the ephemeral prompt (soul rules + team org +
                # inlined workspace.md). Cached here; the dynamic half (dir tree +
                # recent peer messages) is appended fresh each turn in
                # _run_conversation_blocking so it stays current without rebuilding
                # the (cached) stable system prompt.
                self._base_ephemeral = compose_agent_soul(
                    soul_cfg, full_cfg, include_role=False
                )
                # Advanced sampling knobs — only passed when explicitly set so an
                # unset value keeps Hermes' own default. temperature rides through
                # request_overrides (Hermes clamps/omits it per-model as needed);
                # reasoning_effort maps to Hermes' reasoning_config dict.
                extra_kwargs: Dict[str, Any] = {}
                mt = self.cfg.get("max_tokens")
                if mt:
                    try:
                        extra_kwargs["max_tokens"] = int(mt)
                    except (TypeError, ValueError):
                        pass
                temp = self.cfg.get("temperature")
                if temp is not None and temp != "":
                    try:
                        extra_kwargs["request_overrides"] = {"temperature": float(temp)}
                    except (TypeError, ValueError):
                        pass
                effort = (self.cfg.get("reasoning_effort") or "").strip().lower()
                if effort in ("low", "medium", "high"):
                    extra_kwargs["reasoning_config"] = {"enabled": True, "effort": effort}
                elif effort == "off":
                    extra_kwargs["reasoning_config"] = {"enabled": False}

                # provider/base_url/api_key come from the resolver (eff), set below.
                mi = self.cfg.get("max_iterations")
                if mi:
                    try:
                        extra_kwargs["max_iterations"] = int(mi)
                    except (TypeError, ValueError):
                        pass
                # No per-agent override -> apply the swarm-wide turn ceiling.
                # Unbounded turns were observed running 49 tool calls in one go,
                # monopolizing the agent's thread and dodging between-turn review.
                if "max_iterations" not in extra_kwargs and DEFAULT_MAX_ITERATIONS > 0:
                    extra_kwargs["max_iterations"] = DEFAULT_MAX_ITERATIONS
                # Toolset whitelists/blacklists — accept a list or comma string.
                def _as_list(v):
                    if isinstance(v, list):
                        return [str(x).strip() for x in v if str(x).strip()]
                    if isinstance(v, str) and v.strip():
                        return [s.strip() for s in v.split(",") if s.strip()]
                    return None
                en_ts = _as_list(self.cfg.get("enabled_toolsets"))
                if en_ts:
                    extra_kwargs["enabled_toolsets"] = en_ts
                dis_ts = _as_list(self.cfg.get("disabled_toolsets"))
                # ALWAYS deny the Architect's swarm_master toolset to team agents.
                # It is registered in the shared Hermes registry, so a team agent
                # initialized without an enabled_toolsets whitelist would otherwise
                # auto-load it (get_tool_definitions(None) = all toolsets) and gain
                # team/agent-mutating powers. The master's caller-guard is a second
                # line of defense; this is the first.
                dis_ts = (dis_ts or []) + ["swarm_master"]
                extra_kwargs["disabled_toolsets"] = dis_ts

                if _eff_provider:
                    extra_kwargs["provider"] = _eff_provider
                self._ai_agent = AIAgent(
                    base_url=_eff_base,
                    api_key=_eff_key,
                    model=model,
                    session_id=self.cfg["session_id"],
                    skip_memory=False,
                    skip_context_files=False,
                    quiet_mode=True,
                    ephemeral_system_prompt=self._base_ephemeral,
                    session_db=agent_session_db,
                    **extra_kwargs,
                )
                self._wire_live_callbacks()
                _register_custom_tools()

                existing_names = {
                    t.get("function", {}).get("name") for t in (self._ai_agent.tools or [])
                }
                # Optional swarm tools can be turned off per agent via
                # disabled_toolsets (the dashboard picker lists them). send_peer_message
                # is REQUIRED (turn-ending relies on it) and is never disabled.
                # Reuse the normalized list (dis_ts) so a comma-string config form
                # isn't turned into a set of individual characters by set("a,b").
                disabled = set(dis_ts or [])
                if "send_peer_message" not in existing_names:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_SEND_PEER_MESSAGE_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("send_peer_message")
                if "ask_human" not in existing_names and "ask_human" not in disabled:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_ASK_HUMAN_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("ask_human")
                # Browser handoff: agent hands the live browser to a human for a
                # login / CAPTCHA / verification step, then resumes (see tools.py).
                if "request_human_takeover" not in existing_names and "request_human_takeover" not in disabled:
                    from swarm_server.tools import _REQUEST_HUMAN_TAKEOVER_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_REQUEST_HUMAN_TAKEOVER_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("request_human_takeover")
                if "log_decision" not in existing_names and "log_decision" not in disabled:
                    from swarm_server.tools import _LOG_DECISION_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_LOG_DECISION_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("log_decision")
                # Strategy memory: search the team's past decisions on demand to
                # review what's been tried and pivot (complements the 20-decision
                # window auto-injected each turn).
                if "recall_decisions" not in existing_names and "recall_decisions" not in disabled:
                    from swarm_server.tools import _RECALL_DECISIONS_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_RECALL_DECISIONS_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("recall_decisions")
                if "log_action" not in existing_names and "log_action" not in disabled:
                    from swarm_server.tools import _LOG_ACTION_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_LOG_ACTION_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("log_action")
                # Ledger hygiene: dismiss orphaned OPEN delegations (work that
                # completed via a different chain, so no RESULT will ever close
                # them). All agents — delegators close their own orphans,
                # supervisors close any.
                if ("close_ledger_entry" not in existing_names
                        and "close_ledger_entry" not in disabled):
                    from swarm_server.tools import _CLOSE_LEDGER_ENTRY_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_CLOSE_LEDGER_ENTRY_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("close_ledger_entry")
                # Batch file reads: one call for 2-8 files instead of N single
                # read_file round trips (each round trip re-bills the whole
                # context — measured zero model-side batching, so the batching
                # has to live in the tool). Not for supervisors (no file work).
                if (not self.cfg.get("is_supervisor")
                        and "read_files" not in existing_names
                        and "read_files" not in disabled):
                    from swarm_server.tools import _READ_FILES_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_READ_FILES_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("read_files")
                # Self-awareness: read own config/telemetry + PROPOSE changes
                # (human approves in the UI — agents cannot self-apply).
                from swarm_server.tools import (
                    _GET_SELF_CONFIG_TOOL_SCHEMA,
                    _REQUEST_CONFIG_CHANGE_TOOL_SCHEMA,
                )
                if "get_self_config" not in existing_names and "get_self_config" not in disabled:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_GET_SELF_CONFIG_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("get_self_config")
                if "request_config_change" not in existing_names and "request_config_change" not in disabled:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_REQUEST_CONFIG_CHANGE_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("request_config_change")
                # Cron self-scheduling: agents create/cancel their own recurring
                # wake-ups (managed + visible in the dashboard). Registered in the
                # Hermes registry above; exposed to the LLM here like the others.
                from swarm_server.tools import (
                    _SCHEDULE_WAKEUP_TOOL_SCHEMA,
                    _CANCEL_WAKEUP_TOOL_SCHEMA,
                )
                if "schedule_wakeup" not in existing_names and "schedule_wakeup" not in disabled:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_SCHEDULE_WAKEUP_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("schedule_wakeup")
                if "cancel_wakeup" not in existing_names and "cancel_wakeup" not in disabled:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_CANCEL_WAKEUP_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("cancel_wakeup")
                # Emergency brake — SUPERVISORS ONLY. pause_agent/resume_agent let
                # the overseer freeze a peer mid-turn (e.g. about to damage prod)
                # and lift it once safe. Never exposed to non-supervisors.
                if self.cfg.get("is_supervisor"):
                    from swarm_server.tools import (
                        _PAUSE_AGENT_TOOL_SCHEMA,
                        _RESUME_AGENT_TOOL_SCHEMA,
                    )
                    if "pause_agent" not in existing_names and "pause_agent" not in disabled:
                        self._ai_agent.tools = list(self._ai_agent.tools or [])
                        self._ai_agent.tools.append(_PAUSE_AGENT_TOOL_SCHEMA)
                        self._ai_agent.valid_tool_names.add("pause_agent")
                    if "resume_agent" not in existing_names and "resume_agent" not in disabled:
                        self._ai_agent.tools = list(self._ai_agent.tools or [])
                        self._ai_agent.tools.append(_RESUME_AGENT_TOOL_SCHEMA)
                        self._ai_agent.valid_tool_names.add("resume_agent")

                # Credentials registry: fetch/list per-team secrets by site key
                # with an explicit purpose, instead of secrets riding inline in
                # workspace.md prompts. Not for supervisors (they do no project
                # work and never authenticate anywhere).
                if not self.cfg.get("is_supervisor"):
                    from swarm_server.credentials import (
                        GET_CREDENTIAL_TOOL_SCHEMA,
                        LIST_CREDENTIALS_TOOL_SCHEMA,
                    )
                    for _cred_schema in (GET_CREDENTIAL_TOOL_SCHEMA,
                                         LIST_CREDENTIALS_TOOL_SCHEMA):
                        _cred_name = _cred_schema["function"]["name"]
                        if _cred_name not in existing_names and _cred_name not in disabled:
                            self._ai_agent.tools = list(self._ai_agent.tools or [])
                            self._ai_agent.tools.append(_cred_schema)
                            self._ai_agent.valid_tool_names.add(_cred_name)

                # GUI-grade browser tools (keys/hover/drag/click_xy/screenshot/
                # locate). Only where the Hermes browser toolset itself is live
                # (browser_navigate present) — they drive the same session — and
                # NEVER for supervisors, whose browser access is stripped.
                if (not self.cfg.get("is_supervisor")
                        and "browser_navigate" in existing_names):
                    from swarm_server.browser_gui_tools import GUI_BROWSER_TOOL_SCHEMAS
                    for _gui_schema in GUI_BROWSER_TOOL_SCHEMAS:
                        _gui_name = _gui_schema["function"]["name"]
                        if _gui_name not in existing_names and _gui_name not in disabled:
                            self._ai_agent.tools = list(self._ai_agent.tools or [])
                            self._ai_agent.tools.append(_gui_schema)
                            self._ai_agent.valid_tool_names.add(_gui_name)

                # Force tool-use enforcement guidance — agents must end their turn
                # with a tool call (send_peer_message / ask_human) rather than
                # silently stopping with a text response.
                self._ai_agent._tool_use_enforcement = True

                # Eagerly init session DB while HERMES_HOME is locked to this agent
                try:
                    sd = self._ai_agent._get_session_db_for_recall()
                    if sd is None:
                        log.error("[%s] _get_session_db_for_recall() returned None", self.name)
                    else:
                        log.info("[%s] SessionDB created at %s", self.name, getattr(sd, "_db_path", "?"))
                except Exception as e:
                    log.error("[%s] _get_session_db_for_recall() failed: %s", self.name, e)
                self._ai_agent._ensure_db_session()

                log.info(
                    "[%s] Hermes AIAgent initialised (session=%s, home=%s)",
                    self.name,
                    self.cfg["session_id"],
                    self._hermes_home,
                )
            except Exception as exc:
                log.error("[%s] Failed to init AIAgent: %s", self.name, exc)
                raise

    def _load_session_from_db(self) -> List[Dict[str, Any]]:
        """Load conversation history from agent's own isolated Hermes session DB."""
        if self._ai_agent is None:
            log.debug("[%s] _load_session_from_db: _ai_agent is None", self.name)
            return []
        session_db = getattr(self._ai_agent, "_session_db", None)
        if session_db is None:
            log.warning("[%s] _load_session_from_db: _session_db is None", self.name)
            return []
        try:
            current_sid = getattr(self._ai_agent, "session_id", None) or self.cfg["session_id"]
            # include_ancestors=False is deliberate. When Hermes auto-compacts it
            # ROTATES session_id: the summary + recent tail are written to a new
            # child session, while the raw pre-compaction turns stay in the
            # parent. Pulling ancestors here would re-load those raw turns every
            # sweep and defeat compaction entirely (unbounded growth). The child
            # session already carries the summary, so the current session alone is
            # the compacted, bounded view we want to replay.
            msgs = session_db.get_messages_as_conversation(current_sid, include_ancestors=False)
            # History hygiene: every stored user turn embeds the live-context
            # snapshot that was current WHEN THAT TURN RAN. Replayed as-is, a
            # long session shows the model N conflicting copies of the brief /
            # ledger / decision log with no way to tell which is true — observed
            # as agents acting on stale ledger state (re-delegating closed
            # work), and it is the largest per-turn token line item. Strip the
            # expired copies on replay; only the CURRENT turn carries live state.
            cleaned = []
            for m in msgs:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    stripped = strip_stale_live_context(m["content"])
                    if stripped != m["content"]:
                        m = {**m, "content": stripped}
                cleaned.append(m)
            msgs = cleaned
            # Second hygiene pass: age out OLD, LARGE tool results. They are
            # 50-60% of a working agent's history but almost never read again —
            # and unlike compaction, a stub is recoverable (re-run the tool).
            if TOOL_RESULT_AGE_ENABLED:
                msgs = age_stale_tool_results(
                    msgs,
                    keep_recent=TOOL_RESULT_AGE_KEEP_MESSAGES,
                    min_chars=TOOL_RESULT_AGE_MIN_CHARS,
                    quantum=TOOL_RESULT_AGE_QUANTUM,
                )
            # Supervisor counterpart: stub all but the newest sweep payloads
            # (the user-side team feeds that dominate a supervisor's history;
            # its own verdicts stay verbatim as the durable record).
            if SWEEP_PAYLOAD_AGE_ENABLED and self.cfg.get("is_supervisor"):
                msgs = age_stale_sweep_payloads(
                    msgs,
                    keep_recent=SWEEP_PAYLOAD_KEEP_RECENT,
                    min_chars=SWEEP_PAYLOAD_MIN_CHARS,
                )
            log.debug("[%s] Loaded %d messages from session %s", self.name, len(msgs), current_sid)
            return msgs
        except Exception as e:
            log.warning("[%s] Failed to load session from DB: %s", self.name, e)
            return []

    def _persist_session_id_if_rotated(self) -> None:
        """If a compaction rotated the live Hermes session_id, persist it.

        Hermes rotates session_id when it auto-compacts (the summary lives in a
        new child session). We mirror that id into the agent's stored config so
        a future re-init or process restart resumes from the COMPACTED session
        instead of replaying the original full-history root session. No-op on the
        common path where nothing rotated.
        """
        if self._ai_agent is None:
            return
        live_sid = getattr(self._ai_agent, "session_id", None)
        if not live_sid or live_sid == self.cfg.get("session_id"):
            return
        old_sid = self.cfg.get("session_id")
        with self._lock:
            self.cfg["session_id"] = live_sid
        try:
            save_agent_config(self.name, self.cfg)
        except Exception as e:
            log.warning("[%s] Failed to persist rotated session_id: %s", self.name, e)
        log.info("[%s] Context compacted — session rotated %s -> %s", self.name, old_sid, live_sid)
        # Recover the summary Hermes wrote at the head of the rotated (child)
        # session — it's the compacted stand-in for everything before this point.
        # Persist it as a 'compaction_summary' message so it shows up inline in
        # the History transcript as a checkpoint (the dashboard anchors the
        # viewport to the latest one).
        summary_text = ""
        try:
            msgs = self._load_session_from_db()  # current_sid is the new session now
            if msgs:
                head = msgs[0]
                summary_text = str(head.get("content") or "").strip()[:20000]
        except Exception as e:
            log.debug("[%s] compaction summary recovery failed: %s", self.name, e)
        if summary_text:
            monitor_db.log_message(self.name, "compaction_summary", summary_text)
        monitor_db.log_event(
            self.name, "context_compacted",
            data={"old_session": old_sid, "new_session": live_sid,
                  "summary_preview": summary_text[:200]},
        )
        _broadcast("context_compacted", {
            "agent_name": self.name,
            "old_session": old_sid,
            "new_session": live_sid,
            "summary": summary_text,
            "timestamp": time.time(),
        })

    async def stop_execution(self) -> None:
        """Halt the agent's current sweep, drain tasks, and restart the loop.

        Cancels the in-flight sweep task (the result of any ongoing
        run_conversation call in the executor is discarded on restart),
        marks all pending tasks done so they are not re-processed, resets
        state to idle, and starts a fresh sweep loop with a new executor.
        """
        log.info("[%s] Stop execution requested", self.name)
        with self._lock:
            self._stop_requested = True

        # ACTUALLY halt the in-flight turn. Cancelling the asyncio sweep task and
        # shutting the executor below do NOT stop the Hermes turn — it runs on a
        # worker thread Python cannot kill, so without this the agent keeps going
        # to completion (the bug). Hermes' interrupt() sets _interrupt_requested
        # (checked at each tool-loop iteration) and thread-scopes a tool abort, so
        # the turn unwinds at the next boundary. We also release a turn parked
        # inside ask_human (it blocks on human_event).
        try:
            agent = self._ai_agent
            if agent is not None and hasattr(agent, "interrupt"):
                agent.interrupt("Execution stopped by the operator.")
                log.info("[%s] Sent interrupt() to in-flight Hermes turn", self.name)
        except Exception as e:
            log.debug("[%s] interrupt() failed: %s", self.name, e)
        self.human_event.set()

        # Cancel the sweep task. If run_conversation is in-flight the
        # thread keeps going, but the sweep coroutine never handles the
        # result — any post-run work is skipped because _stop_requested
        # is True.
        if self._sweep_task and not self._sweep_task.done():
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass

        # Drain every pending task so they do not come back.
        drained = self.queue.drain_pending(limit=9999)
        for t in drained:
            self.queue.mark_done(t["id"])
        if drained:
            log.info("[%s] Drained %d pending task(s) on stop", self.name, len(drained))

        # Replace the executor so future sweeps run on a clean thread, then WAIT
        # for the old worker thread to actually unwind before continuing. Without
        # the wait, the interrupted turn (interrupt() only lands at the next tool
        # boundary) could still be writing the shared session DB — and would see
        # _stop_requested flipped back to False below — while a new turn starts on
        # the same session, interleaving history. Offloaded to a default thread so
        # the event loop isn't blocked; bounded so a turn wedged in a long blocking
        # tool call can't hang the stop (it then finishes harmlessly in the bg).
        old_executor = self._executor
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"agent-{self.name}"
        )
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, old_executor.shutdown, True
                ),
                timeout=STOP_DRAIN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.warning("[%s] In-flight turn still unwinding at stop — proceeding; "
                        "it will finish in the background", self.name)
        except Exception as e:
            log.debug("[%s] executor drain failed: %s", self.name, e)

        # Clear the in-flight batch (status='processing') so the stopped work is
        # NOT resurrected by recover_processing() on the next restart. (A crash
        # intentionally requeues such rows; an explicit stop must not.)
        cleared = self.queue.mark_processing_done()
        if cleared:
            log.info("[%s] Cleared %d in-flight task(s) on stop", self.name, cleared)

        # Reset flags and state. Drop the cached AIAgent so the NEXT turn re-inits
        # with a clean interrupt state (the interrupt flag we just set would
        # otherwise persist and abort the next turn immediately). Re-init reloads
        # history from the session DB, so nothing is lost.
        with self._lock:
            self._stop_requested = False
            # A stop is a full reset — it also LIFTS any pause. Otherwise a
            # pause-then-stop leaves _paused=True while state shows idle, and the
            # restarted sweep loop silently refuses to drain the queue (the agent
            # looks idle but is frozen, with no Resume button to recover it).
            self._paused = False
            self._pause_reason = ""
            self._paused_by = ""
            self._ai_agent = None
            self.state = AGENT_STATE_IDLE
            self.next_sweep_at = time.time() + self._sweep_interval
        self._emit_state_change()

        # Restart the sweep loop on the same event loop
        if self._loop is not None:
            self._sweep_task = self._loop.create_task(self.sweep_loop())
            self._wake = asyncio.Event()
            # Wake immediately so the loop is active
            self._wake.set()

        log.info("[%s] Execution stopped and sweep loop restarted", self.name)
        monitor_db.log_event(self.name, "execution_stopped", data={"tasks_drained": len(drained)})
        _broadcast("execution_stopped", {
            "agent_name": self.name,
            "tasks_drained": len(drained),
            "timestamp": time.time(),
        })

    def pause_execution(self, reason: str = "", by: str = "") -> None:
        """Emergency brake: interrupt any in-flight turn and freeze the agent.

        Differs from stop_execution in two ways that matter for an *emergency*:
        (1) the pending queue is PRESERVED — nothing is drained, so the held work
        resumes intact on resume; (2) the sweep loop keeps running but processes
        nothing (the _paused guard in _sweep), so the daemon stays alive and can
        be un-frozen instantly. Synchronous and thread-safe: callable from a
        supervisor's tool handler (worker thread) or a server endpoint (loop).
        """
        with self._lock:
            if self._paused:
                return
            self._paused = True
            self._pause_reason = (reason or "")[:500]
            self._paused_by = by or ""
        # Halt the in-flight Hermes turn NOW so the current (possibly dangerous)
        # action unwinds at the next tool boundary instead of running to the end.
        try:
            agent = self._ai_agent
            if agent is not None and hasattr(agent, "interrupt"):
                agent.interrupt(f"PAUSED by {by or 'operator'}: {reason}"[:200])
                log.info("[%s] Sent interrupt() to in-flight turn (pause)", self.name)
        except Exception as e:
            log.debug("[%s] pause interrupt() failed: %s", self.name, e)
        # Release a turn parked inside ask_human so it unwinds too.
        self.human_event.set()
        with self._lock:
            # Drop the cached AIAgent so the post-resume turn re-inits with a clean
            # interrupt flag (re-init reloads history from the session DB — no loss).
            self._ai_agent = None
            self.state = AGENT_STATE_PAUSED
        self._emit_state_change()
        log.warning("[%s] PAUSED by '%s': %s", self.name, by or "operator", reason)
        monitor_db.log_event(
            self.name, "execution_paused", from_agent=(by or None),
            data={"reason": reason, "by": by},
        )
        _broadcast("execution_paused", {
            "agent_name": self.name,
            "reason": reason,
            "by": by,
            "timestamp": time.time(),
        })

    def resume_execution(self, by: str = "") -> None:
        """Lift a pause: re-enable processing and wake the loop immediately.

        Held queue items are picked up on the next sweep. No-op if not paused.
        """
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            prev_reason = self._pause_reason
            self._pause_reason = ""
            self._paused_by = ""
            self.state = AGENT_STATE_IDLE
        self._emit_state_change()
        # asyncio.Event is not thread-safe — set it on the owning loop thread.
        try:
            if self._wake is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(self._wake.set)
        except Exception as e:
            log.debug("[%s] resume wake failed: %s", self.name, e)
        log.info("[%s] RESUMED by '%s' (was paused: %s)", self.name, by or "operator", prev_reason)
        monitor_db.log_event(
            self.name, "execution_resumed", from_agent=(by or None),
            data={"by": by, "was_paused_for": prev_reason},
        )
        _broadcast("execution_resumed", {
            "agent_name": self.name,
            "by": by,
            "timestamp": time.time(),
        })

    def ingest_task(self, from_agent: str, payload: str) -> str:
        # Real inbound work (a peer or human, not a control-plane injection)
        # snaps the idle-heartbeat backoff to the base cadence.
        if from_agent not in _SYSTEM_SENDERS and self._hb_misses:
            log.info("[%s] Real message from '%s' — heartbeat backoff reset", self.name, from_agent)
            self._hb_misses = 0
        task_id = self.queue.enqueue(from_agent, payload)
        log.info("[%s] Task queued from '%s': %s", self.name, from_agent, payload[:80])
        monitor_db.log_event(
            self.name,
            "task_enqueued",
            from_agent=from_agent,
            task_id=task_id,
            data={"payload_preview": payload[:100]},
        )
        _broadcast("queue_updated", {
            "agent_name": self.name,
            "pending_count": self.queue.get_pending_count(),
            "timestamp": time.time(),
        })
        self._signal_wake()
        return task_id

    def _signal_wake(self) -> None:
        """Wake the sweep loop now (thread-safe; ingest may run on a worker thread)."""
        loop, wake = self._loop, self._wake
        if loop is None or wake is None:
            return
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            pass

    def _emit_state_change(self) -> None:
        _broadcast("state_change", {
            "agent_name": self.name,
            "state": self.state,
            "timestamp": time.time(),
            "next_sweep_at": self.next_sweep_at,
        })
        monitor_db.log_event(self.name, "state_change", data={"new_state": self.state})

    async def sweep_loop(self):
        log.info("[%s] Sweep loop started (interval=%ss, event-driven)", self.name, self._sweep_interval)
        while True:
            # Wake on a new task, the periodic safety tick, or — when crons are
            # scheduled — early enough that the soonest cron fires within ~a tick
            # of its time even if the sweep interval is long.
            timeout = self._sweep_interval
            due_in = self._next_cron_due_in()
            if due_in is not None:
                timeout = max(1.0, min(timeout, due_in))
            # When holding after an infra failure, re-check right when the hold
            # expires (don't sleep a full sweep interval past recovery).
            hold_in = self._infra_hold_until - time.time()
            if hold_in > 0:
                timeout = max(1.0, min(timeout, hold_in))
            self.next_sweep_at = time.time() + timeout
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            await self._sweep()
            # A paused agent does NOTHING off-cycle either — no crons, no
            # autonomous heartbeat, no supervisor reviews — until resumed. A team
            # over its daily budget is held the same way (no new self-inflicted
            # work piling up) until the cap is raised / overridden / rolls over.
            if not self._paused and not self._budget_blocked():
                self._maybe_fire_crons()
                self._maybe_autonomous_heartbeat()
                self._maybe_feed_supervisor()

    async def _sweep(self):
        # Frozen by an emergency pause — process nothing and leave the queue
        # intact (work resumes untouched on resume). Return BEFORE draining.
        if self._paused:
            return
        # Holding off after an infra-failure (provider outage) — don't re-claim
        # the batch until the backoff expires, so we don't hammer a down provider.
        if time.time() < self._infra_hold_until:
            return
        # Team over its daily budget — HOLD the queue (tasks kept, not failed),
        # same shape as the infra hold. Resumes automatically at UTC rollover, or
        # immediately when the human raises the cap / clicks "resume anyway".
        if self._budget_blocked():
            return
        # Claim a bounded batch first. If there's nothing to do, stay idle
        # SILENTLY — no state flip, no broadcast, no event. This is what keeps
        # an idle 24/7 swarm from drowning the monitoring log in busy/idle churn.
        tasks = self.queue.drain_pending(limit=MAX_BATCH_SIZE)
        if not tasks:
            return

        with self._lock:
            self.state = AGENT_STATE_BUSY
        self._emit_state_change()

        log.info("[%s] Sweep: processing %d task(s) in batch", self.name, len(tasks))
        monitor_db.log_event(self.name, "task_dequeued", data={"count": len(tasks)})
        _broadcast("task_dequeued", {
            "agent_name": self.name,
            "count": len(tasks),
            "timestamp": time.time(),
        })

        try:
            await self._process_tasks_batch(tasks)
        finally:
            # Mark activity so the autonomous heartbeat measures idle time from
            # the end of real work, not from server start.
            self._last_active = time.time()
            with self._lock:
                # If a pause landed mid-turn, the interrupted turn unwinds into
                # here — honor the freeze instead of flipping back to idle.
                self.state = AGENT_STATE_PAUSED if self._paused else AGENT_STATE_IDLE
                self.next_sweep_at = time.time() + self._sweep_interval
            self._emit_state_change()
            # More queued while we were busy? Wake immediately rather than wait
            # (but never while paused — held work waits for resume — and never
            # during an infra hold, which would re-spin against a down provider).
            try:
                if (not self._paused
                        and time.time() >= self._infra_hold_until
                        and self.queue.get_pending_count() > 0
                        and self._wake is not None):
                    self._wake.set()
            except Exception:
                pass

    def _budget_blocked(self) -> bool:
        """True when this agent's team is over its daily spend cap. Cheap
        (in-memory dict lookups) — safe to call every sweep tick."""
        try:
            from swarm_server.budget import budget_tracker
            return budget_tracker.is_blocked(self.cfg.get("team_id", "default"))
        except Exception:
            return False

    def _maybe_autonomous_heartbeat(self) -> None:
        """Self-inject a continue-mission task when idle (24/7 autonomy).

        Fires only for agents flagged autonomous (typically the team
        coordinator), and only when: not busy, the queue is empty, and at least
        AUTONOMOUS_HEARTBEAT_SECONDS have elapsed since the last real work. The
        coordinator then reviews the mission + what's done and delegates the
        next increment, which keeps the whole team working without a human in
        the loop. Resetting _last_active here prevents back-to-back refiring.
        """
        if not self._autonomous or self._stop_requested:
            return
        if self.state == AGENT_STATE_BUSY:
            return
        try:
            if self.queue.get_pending_count() > 0:
                return
        except Exception:
            return
        now = time.time()
        directive = self._directive
        if directive and now >= float(directive.get("until_ts", 0)):
            monitor_db.log_event(self.name, "directive_expired",
                                 data={"from_agent": directive.get("from_agent"),
                                       "preview": str(directive.get("payload", ""))[:150]})
            log.info("[%s] Time-boxed directive expired — back to normal idle cadence",
                     self.name)
            self._directive = directive = None
        # Adaptive cadence: each consecutive no-op heartbeat doubles the wait
        # (capped), so a 24/7 agent with genuinely nothing to do costs
        # exponentially less instead of inventing busywork every interval.
        # While a directive is ACTIVE the backoff is suspended: an idle agent
        # mid-push is owing work, not resting.
        misses = 0 if directive else self._hb_misses
        effective = self.effective_heartbeat_interval(self._heartbeat_seconds, misses)
        if now - self._last_active < effective:
            return
        self._last_active = now
        log.info("[%s] Autonomous heartbeat — injecting continue-mission task (backoff x%d)",
                 self.name, 2 ** min(misses, HEARTBEAT_BACKOFF_MAX_DOUBLINGS))
        monitor_db.log_event(self.name, "autonomous_heartbeat",
                             data={"misses": self._hb_misses, "effective_interval": effective,
                                   "directive": bool(directive)})
        import datetime
        if directive:
            remaining = max(0.0, float(directive["until_ts"]) - now)
            prompt = DIRECTIVE_HEARTBEAT_PROMPT.format(
                remaining=_age_short(remaining),
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                from_agent=directive.get("from_agent", "human"),
                directive=str(directive.get("payload", ""))[:1500],
            )
        else:
            prompt = AUTONOMOUS_HEARTBEAT_PROMPT.format(
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            )
        self.ingest_task("autonomous", prompt)

    def set_directive(self, payload: str, duration_minutes: float,
                      from_agent: str = "human") -> None:
        """Arm a time-boxed directive: until it expires, idle heartbeats
        re-present `payload` at base cadence instead of the generic check-in,
        so a multi-hour push survives the queue going empty."""
        try:
            minutes = float(duration_minutes)
        except (TypeError, ValueError):
            return
        if minutes <= 0:
            return
        self._directive = {
            "payload": str(payload or ""),
            "until_ts": time.time() + minutes * 60.0,
            "from_agent": from_agent,
        }
        monitor_db.log_event(self.name, "directive_set",
                             data={"from_agent": from_agent,
                                   "duration_minutes": minutes,
                                   "preview": str(payload or "")[:150]})
        log.info("[%s] Time-boxed directive armed for %.0f min (from %s)",
                 self.name, minutes, from_agent)

    def _apply_turn_output_guards(self, tasks: List[Dict[str, Any]],
                                  turn_messages: List[Dict[str, Any]]) -> List[str]:
        """Post-turn output guards. Returns the turn's tool-call names (the
        heartbeat bookkeeping that follows in the run loop reuses them).

        1. Missing-RESULT handoff: a peer/human TASK or QUESTION turn that ends
           without ANY send_peer_message left the delegator blind — whether the
           turn was truncated by the iteration budget (empty final) or just
           stopped after a free-text status. Force ONE status RESULT per task
           id; sweeps and ledger ages remain the backstop beyond it. (Observed:
           a worker silently stalled twice this way for 95 minutes.)
        2. Text-only turn guard: fires ONLY when the turn ACTED on nothing —
           its tool calls (if any) were all read-only, then it wrote a summary.
           Notification-only turns (heartbeat, guard follow-up, incoming
           RESULT/STATUS/FYI) are EXEMPT: they legitimately end in "looked
           around, nothing to do, summarized". (Observed: 4 false nudges in one
           night, every one on a heartbeat-analysis or RESULT-acknowledgment
           turn, ~50–230k tokens each.) Capped at one nudge per occurrence via
           _text_only_nudged; cleared once a turn acts. Both guards are skipped
           for supervisors and on stop."""
        assts = [m for m in turn_messages if m.get("role") == "assistant"]
        tool_names = [
            tc.get("function", {}).get("name")
            for m in assts for tc in (m.get("tool_calls") or [])
        ]
        acted = any(n and n not in self._READONLY_TOOLS for n in tool_names)
        wrote_summary = (
            bool(assts)
            and not assts[-1].get("tool_calls")
            and len((assts[-1].get("content") or "").strip()) > 40
        )

        def _is_notification(t: Dict[str, Any]) -> bool:
            if t.get("from_agent") in ("autonomous", "turn-guard"):
                return True
            p = str(t.get("payload") or "").lstrip()
            return p.startswith(("[RESULT", "[STATUS", "[FYI"))

        notification_only = bool(tasks) and all(
            _is_notification(t) for t in tasks)

        missing_result_fired = False
        if not self.cfg.get("is_supervisor") and not self._stop_requested:
            delegated = [
                t for t in tasks
                if str(t.get("payload") or "").lstrip().startswith(
                    ("[TASK", "[QUESTION"))
                and t.get("from_agent") not in ("turn-guard", "autonomous")
            ]
            if delegated and "send_peer_message" not in tool_names:
                for t in delegated:
                    m = re.search(r"id=([0-9a-fA-F-]{6,})",
                                  str(t.get("payload") or ""))
                    short_id = m.group(1) if m else t.get("id", "")[:8]
                    if short_id in self._result_nudged_ids:
                        continue
                    self._result_nudged_ids.add(short_id)
                    if len(self._result_nudged_ids) > 100:
                        self._result_nudged_ids = set(
                            list(self._result_nudged_ids)[-50:])
                    missing_result_fired = True
                    self.ingest_task("turn-guard", MISSING_RESULT_NUDGE.format(
                        task_id=short_id, from_agent=t.get("from_agent", "?")))
                    monitor_db.log_event(
                        self.name, "missing_result_nudge",
                        data={"task_id": short_id,
                              "from_agent": t.get("from_agent")})
                    log.info("[%s] turn on task %s ended without a RESULT "
                             "— forced one status handoff", self.name, short_id)

        if acted:
            self._text_only_nudged = False
        elif (wrote_summary and not self._text_only_nudged
              and not missing_result_fired
              and not notification_only
              and not self._stop_requested
              and not self.cfg.get("is_supervisor")):
            self._text_only_nudged = True
            self.ingest_task("turn-guard", TEXT_ONLY_TURN_NUDGE)
            log.info("[%s] read-only/no-op turn ended in summary — "
                     "enqueued one corrective", self.name)
        return tool_names

    @staticmethod
    def effective_heartbeat_interval(base: float, misses: int) -> float:
        """Idle-heartbeat interval after `misses` consecutive no-op heartbeats:
        base * 2**misses, capped at 2**HEARTBEAT_BACKOFF_MAX_DOUBLINGS."""
        return float(base) * (2 ** min(max(0, int(misses)), HEARTBEAT_BACKOFF_MAX_DOUBLINGS))

    def _supervisor_interval_seconds(self) -> int:
        """Sweep cadence: per-agent `supervisor_interval_minutes` (agent
        settings) over the global default; floored at 2 minutes."""
        try:
            v = float(self.cfg.get("supervisor_interval_minutes") or 0)
        except (TypeError, ValueError):
            v = 0
        minutes = v if v > 0 else SUPERVISOR_SWEEP_INTERVAL_MINUTES
        return max(120, int(minutes * 60))

    def _render_feed_transcript(self, msgs: List[Dict[str, Any]],
                                char_cap: int = SUPERVISOR_FEED_CHAR_CAP) -> str:
        """Render a peer's new messages oldest-first, capped to the most-recent
        slice (a marker notes any truncation — no silent loss)."""
        lines = []
        for m in msgs:
            role = (m.get("role") or "?").upper()
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"[{role}] {content}")
        text = "\n\n".join(lines)
        if len(text) > char_cap:
            text = ("[…older activity in this window truncated…]\n\n"
                    + text[-char_cap:])
        return text

    # Tool calls that do NOT advance the mission by themselves — coordination and
    # bookkeeping. Anything else (web/file/terminal/browser/email/code/git…) is a
    # "concrete action" that touches the real world.
    _NONACTION_TOOLS = {
        "send_peer_message", "log_decision", "log_action", "todo", "memory",
        "get_self_config", "ask_human", "request_human_takeover",
        "request_config_change", "schedule_wakeup", "cancel_wakeup",
    }

    # Purely READ-ONLY tools. Every turn ends with a stop-reason text message (the
    # API guarantees that), so "ended with text" is NOT a useful signal. What
    # matters is whether the turn DID anything besides look around. The text-only
    # turn guard fires only when a turn's tool calls were ALL read-only (or there
    # were none) and it then wrote a prose summary — i.e. it investigated/mused and
    # narrated instead of acting. Anything NOT in this set (terminal, code, write,
    # patch, send_peer_message, log_decision, browser_*, deploy, …) counts as
    # acting, so a real deploy-via-terminal turn is never falsely nudged.
    _READONLY_TOOLS = {
        "read_file", "read_files", "search_files", "web_search", "web_extract",
        "get_self_config", "list_files", "list_dir", "grep", "glob", "ls",
        "todo", "memory", "get_messages",
    }

    @staticmethod
    def _norm_msg(text: str) -> str:
        """Normalize an assistant message to its intent so near-duplicate status
        re-confirmations collapse to one key. Drops the tool RESULT (everything
        after '→'), lowercases, keeps alnum+space."""
        t = text.split("→", 1)[0]
        t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
        return re.sub(r"\s+", " ", t).strip()[:200]

    def _assess_peer_progress(self, peer: str, msgs: List[Dict[str, Any]]) -> str:
        """Deterministically classify a peer's review window so the supervisor
        judges PROGRESS, not token volume. Computed in code (not left to the
        model's eye) so the no-progress ACK loop the supervisors kept missing is
        now an explicit flag on every review."""
        actions: List[str] = []
        reports = logs = prose = 0
        norm_keys: List[str] = []
        for m in msgs:
            if (m.get("role") or "") != "assistant":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            # Two producer formats must both parse, or a real working turn is
            # mis-scored as zero-action: the LIVE per-step render "🛠️ name(args)"
            # and the end-of-turn fallback "🛠️ Tool Calls: name() | name2()".
            tools = re.findall(r"🛠️\s*([a-z_][a-z0-9_]*)\(", content)
            if not tools:
                m = re.search(r"🛠️\s*Tool Calls:\s*([^\n]*)", content)
                if m:
                    tools = re.findall(r"([a-z_][a-z0-9_]*)\(", m.group(1))
            if not tools:
                prose += 1  # free-text only — reaches nobody
            for t in tools:
                if t == "send_peer_message":
                    reports += 1
                elif t == "log_decision":
                    logs += 1
                elif t not in self._NONACTION_TOOLS:
                    actions.append(t)
            norm_keys.append(self._norm_msg(content))

        n = len(norm_keys)
        concrete = len(actions)
        seen: Dict[str, int] = {}
        for k in norm_keys:
            seen[k] = seen.get(k, 0) + 1
        dup_turns = sum(c - 1 for c in seen.values() if c > 1)
        top_repeat = max(seen.values()) if seen else 0

        parts = [
            f"PROGRESS SIGNAL (computed, not prose): {n} turns this window — "
            f"{concrete} concrete external action(s)"
            + (f" ({', '.join(sorted(set(actions))[:6])})" if actions else "")
            + f", {reports} peer-message(s), {logs} log(s), {prose} free-text-only."
        ]
        if n >= 3 and concrete == 0 and (dup_turns >= 2 or top_repeat >= 3):
            parts.append(
                f"⚠ NO-PROGRESS LOOP: {dup_turns} of {n} turns are near-duplicate "
                f"re-confirmations of an earlier turn and ZERO concrete actions were "
                f"taken. {peer} is repeating itself, not progressing — this is exactly "
                f"the drift you must break. Do NOT acknowledge it; steer it to ONE "
                f"concrete next action (or escalate)."
            )
        elif concrete == 0 and n >= 2:
            parts.append(
                f"⚠ NO CONCRETE ACTION this window — {peer} only coordinated/logged. "
                f"If it had a real task, that task did not move."
            )
        try:  # latest out-of-band digest — the 'stuck' signal, now wired in
            d = monitor_db.get_last_digest(peer)
            if d:
                s = json.loads(d.get("summary") or "{}")
                risk = s.get("risk_level")
                if risk and risk != "ok":
                    parts.append(
                        f"DIGEST: risk={risk}; {s.get('headline', '')}"
                        + (f"; blocked_on={s.get('blocked_on')}"
                           if s.get("blocked_on") else "")
                    )
        except Exception:
            pass
        return "\n".join(parts)

    def _peer_runtime_state(self, peer: str) -> tuple:
        """(state, pending_count) for a linked peer — live from its daemon."""
        try:
            daemon = _daemon_registry.get(peer)
            if daemon is not None:
                pending = 0
                try:
                    pending = daemon.queue.get_pending_count()
                except Exception:
                    pass
                return (getattr(daemon, "state", None) or "?", pending)
        except Exception:
            pass
        return ("?", 0)

    def _sweep_ledger(self, peers: List[str],
                      by_peer: Dict[str, Dict[str, Any]], now: float) -> str:
        """Compact team-state header for a sweep: agent states, open
        delegations involving the watched agents (with ages + overdue flags),
        and pending human questions. Scoped to THIS supervisor's peers — a
        team can run several supervisors over different subsets."""
        bits = []
        for p in peers:
            d = by_peer.get(p, {})
            state = (d.get("state") or "?").upper()
            q = d.get("pending") or 0
            bits.append(f"{p}: {state}" + (f" ({q} queued)" if q else ""))
        lines = ["Agents — " + " · ".join(bits)]
        try:
            dels = monitor_db.get_open_delegations(
                team_id=self.cfg.get("team_id"), limit=30)
        except Exception:
            dels = []
        mine = [d for d in dels
                if d.get("to_agent") in peers or d.get("from_agent") in peers]
        if mine:
            for d in mine[:10]:
                age = max(0.0, now - float(d.get("timestamp") or now))
                flag = (" ⚠ overdue — chase the owner or reassign"
                        if age > 7200 else "")
                lines.append(
                    f"OPEN [{(d.get('msg_id') or '')[:8]}] "
                    f"{d.get('from_agent')}→{d.get('to_agent')} "
                    f"open {_age_short(age)}{flag}: "
                    f"{(d.get('summary') or '').strip()[:80]}")
            lines.append(
                "An OPEN entry verified complete/obsolete via another chain is an "
                "orphan — close it with close_ledger_entry(msg_id, reason) instead "
                "of re-flagging it every sweep.")
        else:
            lines.append("Open delegations involving your agents: none")
        try:
            from swarm_server.tools import get_pending_questions

            qs = [q for q in get_pending_questions()
                  if q.get("status") == "pending" and q.get("agent_name") in peers]
            for q in qs[:5]:
                age = max(0.0, now - float(q.get("timestamp") or now))
                lines.append(
                    f"WAITING ON HUMAN {_age_short(age)} — {q.get('agent_name')}: "
                    f"{(q.get('question') or '').strip()[:90]}")
        except Exception:
            pass

        # Active cron wake-ups per watched agent. A recurring schedule that was
        # really a one-time task fires forever and quietly monopolises the agent
        # — surface these so the supervisor can catch a runaway and tell the
        # owner to cancel_wakeup it (or re-create it with max_runs).
        cron_lines = []
        for p in peers:
            daemon = _daemon_registry.get(p)
            if daemon is None:
                continue
            try:
                crons = [c for c in daemon.crons_runtime() if c.get("enabled", True)]
            except Exception:
                crons = []
            if not crons:
                continue
            parts = []
            for c in crons[:6]:
                runs = c.get("runs") or 0
                mr = c.get("max_runs")
                age = _age_short(max(0.0, now - float(c.get("created_at") or now)))
                tag = (f", fired {runs}×" if runs else "")
                tag += (f"/{mr}" if mr else (" recurring" if not runs else "×"))
                parts.append(
                    f"[{(c.get('id') or '')[:6]}] {c.get('schedule')} "
                    f"(set {age} ago{tag}): {(c.get('instruction') or '').strip()[:70]}")
            cron_lines.append(f"CRONS {p}: " + " | ".join(parts))
        if cron_lines:
            lines.extend(cron_lines)
            lines.append(
                "Review CRONS: a one-time task scheduled as a recurring wake-up "
                "(high fire count, no max_runs, same instruction repeating) will "
                "derail its owner — message the agent to cancel_wakeup it.")
        return "\n".join(lines)

    def _oldest_open_delegation_age(self, peers: List[str],
                                    now: float) -> Optional[float]:
        """Age (seconds) of the oldest open delegation involving a watched
        peer, or None. Same scoping as _sweep_ledger."""
        try:
            dels = monitor_db.get_open_delegations(
                team_id=self.cfg.get("team_id"), limit=30)
        except Exception:
            return None
        ages = [max(0.0, now - float(d.get("timestamp") or now)) for d in dels
                if d.get("to_agent") in peers or d.get("from_agent") in peers]
        return max(ages) if ages else None

    def _maybe_feed_supervisor(self) -> None:
        """Supervisor agents only: every sweep interval, push ONE task into
        THIS agent's own queue carrying everything every linked peer did since
        the previous sweep — straight from the live monitoring DB, so an agent
        mid-turn contributes its partial turn up to this moment — plus a team
        ledger (states, open delegations with ages, human blocks).

        Daemon-side and automatic — the supervisor never calls a tool to fetch;
        the sweep arrives as a queued task, like any peer message. Replaces the
        retired token-threshold reviews, which were volume-gated, single-peer,
        and silence-blind (an idle agent owing work never generated tokens, so
        it was never reviewed). Interval: per-agent `supervisor_interval_minutes`
        over SUPERVISOR_SWEEP_INTERVAL_MINUTES. If the supervisor is busy when
        the interval elapses, the sweep fires on the next idle tick and the
        window simply covers the longer span — watermarks keep it gapless.

        IDLE-SKIP: when no watched peer logged a single message this window,
        the sweep is skipped (a cheap per-peer COUNT instead of a measured
        ~85k-token all-IDLE LLM turn) — but never silence-blind: every
        SUPERVISOR_SWEEP_MAX_IDLE_SKIPS-th consecutive idle interval forces a
        full review (the idle-but-OWING guarantee this sweep design exists
        for), and an open delegation older than
        SUPERVISOR_SWEEP_FORCE_OPEN_AGE_SECONDS forces one early, rate-limited
        to once per age window. Skips never advance watermarks, so the next
        real sweep covers the whole quiet span.
        """
        if not self.cfg.get("is_supervisor") or self._stop_requested:
            return
        if self.state == AGENT_STATE_BUSY:
            return
        try:
            if self.queue.get_pending_count() > 0:
                return  # a sweep (or other task) is already waiting — never pile up
        except Exception:
            return
        now = time.time()
        # A skip waits a full interval before re-checking; _last_sweep_ts alone
        # would re-evaluate every daemon tick and fire seconds after the first
        # new message.
        if (now - max(self._last_sweep_ts, self._last_sweep_check_ts)
                < self._supervisor_interval_seconds()):
            return
        peers = [p for p in (self.cfg.get("allowed_peers") or []) if p != self.name]
        if not peers:
            self._last_sweep_ts = now
            return
        try:
            # Cheap idle probe: one aggregate row per peer, no content pulled.
            # First-seen peers are anchored here (same as the build path did)
            # and count as "no activity" — the next window covers them fully.
            new_total = 0
            for peer in peers:
                wm = self._sup_watermark.get(peer)
                if wm is None:
                    # First sight of this peer (fresh daemon or newly linked):
                    # anchor to its current latest id. Never dump history.
                    try:
                        wm = monitor_db.get_new_activity(peer, 0)["max_id"]
                    except Exception:
                        wm = 0
                    self._sup_watermark[peer] = wm
                    continue
                try:
                    new_total += int(monitor_db.get_new_activity(peer, wm)["count"])
                except Exception:
                    new_total += 1  # fail OPEN: a probe error must not mute oversight

            force_reason = ""
            if new_total == 0:
                if self._sup_idle_skips + 1 >= SUPERVISOR_SWEEP_MAX_IDLE_SKIPS:
                    force_reason = "idle_backstop"  # idle-but-owing review
                else:
                    age = self._oldest_open_delegation_age(peers, now)
                    if (age is not None
                            and age >= SUPERVISOR_SWEEP_FORCE_OPEN_AGE_SECONDS
                            and now - self._last_delegation_force_ts
                            >= SUPERVISOR_SWEEP_FORCE_OPEN_AGE_SECONDS):
                        force_reason = "stale_delegation"
                        self._last_delegation_force_ts = now

            if new_total == 0 and not force_reason:
                self._sup_idle_skips += 1
                self._last_sweep_check_ts = now
                monitor_db.log_event(
                    self.name, "supervisor_sweep_skipped",
                    data={"peers": len(peers),
                          "consecutive_skips": self._sup_idle_skips,
                          "forced_after": SUPERVISOR_SWEEP_MAX_IDLE_SKIPS,
                          "window_seconds": int(now - self._last_sweep_ts)},
                )
                log.info("[%s] sweep skipped — no peer activity (%d/%d before "
                         "forced review)", self.name, self._sup_idle_skips,
                         SUPERVISOR_SWEEP_MAX_IDLE_SKIPS)
                return

            peer_data: List[Dict[str, Any]] = []
            new_marks: Dict[str, int] = {}
            for peer in peers:
                wm = self._sup_watermark[peer]
                msgs = monitor_db.get_messages_since(peer, wm)
                new_marks[peer] = max([int(m["id"]) for m in msgs], default=wm)
                state, pending = self._peer_runtime_state(peer)
                peer_data.append({
                    "peer": peer, "state": state, "pending": pending,
                    # Render uncapped here; compose_sweep_sections owns the
                    # per-peer budget so one noisy peer can't eat the sweep.
                    "transcript": (self._render_feed_transcript(
                        msgs, char_cap=SUPERVISOR_SWEEP_CHAR_CAP) if msgs else ""),
                    "signal": (self._assess_peer_progress(peer, msgs)
                               if msgs else ""),
                    "messages": len(msgs),
                    "tokens": sum(int(m.get("tokens") or 0) for m in msgs),
                })
            ledger = self._sweep_ledger(
                peers, {d["peer"]: d for d in peer_data}, now)
            sections = compose_sweep_sections(
                peer_data, SUPERVISOR_SWEEP_CHAR_CAP, SUPERVISOR_SWEEP_PER_PEER_FLOOR)
            prompt = SUPERVISOR_SWEEP_PROMPT.format(
                window_minutes=max(1, round((now - self._last_sweep_ts) / 60)),
                peer_count=len(peers), ledger=ledger, sections=sections)
            monitor_db.log_event(
                self.name, "supervisor_sweep",
                data={"peers": len(peers),
                      "active": sum(1 for d in peer_data if d["messages"]),
                      "chars": len(prompt),
                      "forced": force_reason or None,
                      "window_seconds": int(now - self._last_sweep_ts)},
            )
            self.ingest_task("supervisor-sweep", prompt)
            # Commit watermarks + clocks only after the sweep is safely queued,
            # so a failure above retries the same window on the next tick.
            self._sup_watermark.update(new_marks)
            self._last_sweep_ts = now
            self._last_sweep_check_ts = now
            self._sup_idle_skips = 0
        except Exception as e:
            log.warning("[%s] supervisor sweep failed: %s", self.name, e)

    def _maybe_fire_crons(self) -> None:
        """Inject the instruction of any cron wake-up whose time has come.

        Runs every sweep tick. Each due cron enqueues its instruction as a task
        (regardless of autonomous flag — a cron is an explicit schedule), records
        the fire time, and rolls forward to its next occurrence. Independent of
        the idle heartbeat: a cron fires on time even while other work is queued.
        """
        if self._stop_requested or not self._cron_next:
            return
        from swarm_server.cron import cron_next

        now = time.time()
        # Snapshot ids so rescheduling inside the loop can't churn the dict iter.
        due = [cid for cid, ts in list(self._cron_next.items()) if ts is not None and now >= ts]
        if not due:
            return
        by_id = {c.get("id"): c for c in self._crons}
        for cid in due:
            c = by_id.get(cid)
            if c is None or not c.get("enabled", True):
                self._cron_next.pop(cid, None)
                continue
            sched = c.get("schedule") or ""
            import datetime
            created_at = c.get("created_at")
            scheduled_ago = (
                _age_short(max(0, now - created_at)) if created_at else "a while"
            )
            prompt = CRON_WAKEUP_PROMPT.format(
                schedule=sched,
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                scheduled_ago=scheduled_ago,
                instruction=c.get("instruction", ""),
            )
            log.info("[%s] Cron '%s' (%s) fired — injecting scheduled task", self.name, cid, sched)
            monitor_db.log_event(self.name, "cron_fired", data={"cron_id": cid, "schedule": sched})
            _broadcast("cron_fired", {
                "agent_name": self.name,
                "cron_id": cid,
                "schedule": sched,
                "timestamp": now,
            })
            self.ingest_task("cron", prompt)
            self._cron_last[cid] = now
            self._cron_runs[cid] = self._cron_runs.get(cid, 0) + 1

            # Bounded wake-up (max_runs): persist the fire and stop it for good
            # once it has run its course. This is what keeps a one-time task
            # (max_runs=1) from recurring forever and monopolising the agent.
            if c.get("max_runs"):
                try:
                    from swarm_server.config import record_cron_fire
                    res = record_cron_fire(self.name, cid)
                except Exception as e:  # noqa: BLE001 — never let bookkeeping abort a fire
                    log.warning("[%s] record_cron_fire failed for %s: %s", self.name, cid, e)
                    res = {"completed": False}
                if res.get("completed"):
                    c["enabled"] = False
                    self._cron_next.pop(cid, None)
                    log.info("[%s] Cron '%s' completed (%s run(s)) — auto-stopped",
                             self.name, cid, res.get("runs"))
                    monitor_db.log_event(self.name, "cron_completed",
                                         data={"cron_id": cid, "schedule": sched,
                                               "runs": res.get("runs")})
                    _broadcast("cron_updated", {"agent_name": self.name,
                                                "action": "completed", "timestamp": now})
                    continue

            # Roll forward to the next occurrence (strictly after now).
            try:
                self._cron_next[cid] = cron_next(sched, now)
            except Exception:
                self._cron_next.pop(cid, None)

    def _next_cron_due_in(self) -> Optional[float]:
        """Seconds until the soonest pending cron fire, or None if no crons."""
        if not self._cron_next:
            return None
        soonest = min((ts for ts in self._cron_next.values() if ts is not None), default=None)
        if soonest is None:
            return None
        return max(0.0, soonest - time.time())

    # ------------------------------------------------------------------
    # Live execution streaming (ephemeral; dashboard-only)
    # ------------------------------------------------------------------
    def _emit_exec(self, kind: str, data: Dict[str, Any]) -> None:
        """Broadcast one ephemeral live-execution step (thinking/tool/answer).

        Purely for the dashboard's real-time trace — NOT persisted to
        monitoring.db (the final answer + tool calls are still logged at the end
        of the batch as before). Skipped entirely when no dashboard is connected
        so a 24/7 swarm pays nothing for this when nobody is watching (matters for
        the high-frequency token stream). Runs on the agent's worker thread;
        _broadcast hops to the event loop thread-safely.
        """
        try:
            if not ws_broadcaster.clients:
                return
            self._exec_seq += 1
            payload = {
                "agent_name": self.name,
                "seq": self._exec_seq,
                "kind": kind,
                "timestamp": time.time(),
            }
            payload.update(data)
            _broadcast("agent_exec", payload)
        except Exception:
            pass

    def _collect_telemetry(self) -> Dict[str, Any]:
        """Snapshot runtime telemetry from the live AIAgent (read-only, defensive).

        Read after each turn for the UI: current context occupancy, cumulative
        token spend + cost, the model's context window, and the compaction
        trigger. All getattr-guarded so a Hermes version difference degrades to
        a partial dict instead of raising.
        """
        a = self._ai_agent
        if a is None:
            return {}
        t: Dict[str, Any] = {}
        try:
            cc = getattr(a, "context_compressor", None)
            if cc is not None:
                t["context_tokens"] = int(getattr(cc, "last_prompt_tokens", 0) or 0)
                t["context_window"] = int(getattr(cc, "context_length", 0) or 0)
                t["compress_threshold_tokens"] = int(getattr(cc, "threshold_tokens", 0) or 0)
            t["session_total_tokens"] = int(getattr(a, "session_total_tokens", 0) or 0)
            t["session_input_tokens"] = int(getattr(a, "session_input_tokens", 0) or 0)
            t["session_output_tokens"] = int(getattr(a, "session_output_tokens", 0) or 0)
            t["session_cost_usd"] = round(float(getattr(a, "session_estimated_cost_usd", 0.0) or 0.0), 4)
            t["max_iterations"] = int(getattr(a, "max_iterations", 0) or 0)
            t["model"] = getattr(a, "model", None)
            t["provider"] = getattr(a, "provider", None)
            if t.get("context_window"):
                t["context_pct"] = round(100.0 * t.get("context_tokens", 0) / t["context_window"], 1)
        except Exception as e:
            log.debug("[%s] telemetry collect failed: %s", self.name, e)
        self._telemetry = t
        return t

    def _wire_live_callbacks(self) -> None:
        """Attach Hermes' progress callbacks to the live-exec broadcaster.

        Hermes invokes these on the conversation thread as the turn unfolds:
        thinking (status pulse), reasoning (chain-of-thought text), tool start /
        complete, and stream_delta (final-answer tokens). We forward each as an
        'agent_exec' WS event so the UI can render the turn as it happens.
        """
        a = self._ai_agent
        if a is None:
            return

        def on_thinking(text: str = "") -> None:
            self._emit_exec("thinking", {"text": (text or "")[:200]})

        def on_reasoning(text: str = "") -> None:
            if text:
                self._emit_exec("reasoning", {"text": str(text)[:4000]})

        def on_tool_start(tool_call_id, name, args) -> None:
            try:
                args_str = args if isinstance(args, str) else json.dumps(args, default=str)
            except Exception:
                args_str = str(args)
            self._emit_exec("tool_start", {
                "id": str(tool_call_id), "name": str(name), "args": (args_str or "")[:1500],
            })

        def on_tool_complete(tool_call_id, name, args, result) -> None:
            self._emit_exec("tool_result", {
                "id": str(tool_call_id), "name": str(name),
                "result": ("" if result is None else str(result))[:2000],
            })
            # Persist a compact step to monitoring.db AS IT HAPPENS (independent of
            # whether a dashboard is connected). Without this, a turn's activity is
            # invisible to digests + the supervisor until it COMPLETES — so a long
            # runaway turn (or one that gets interrupted) evaded all oversight.
            try:
                try:
                    args_s = args if isinstance(args, str) else json.dumps(args, default=str)
                except Exception:
                    args_s = str(args)
                res = "" if result is None else str(result)
                content = f"🛠️ {name}({(args_s or '')[:300]}) → {res[:600]}"
                tids = ",".join(getattr(self, "_current_task_ids", []) or [])
                monitor_db.log_message(self.name, "assistant", content, tids)
                try:
                    self._live_logged_tool_ids.add(str(tool_call_id))
                except Exception:
                    pass
                _broadcast("message_logged", {
                    "agent_name": self.name, "role": "assistant",
                    "content": content, "task_id": tids, "timestamp": time.time(),
                })
            except Exception as e:
                log.debug("[%s] live tool persist failed: %s", self.name, e)

        def on_stream_delta(chunk) -> None:
            # None is Hermes' flush/end sentinel — ignore it; only forward text.
            if chunk:
                self._emit_exec("token", {"text": str(chunk)})

        a.thinking_callback = on_thinking
        a.reasoning_callback = on_reasoning
        a.tool_start_callback = on_tool_start
        a.tool_complete_callback = on_tool_complete
        a.stream_delta_callback = on_stream_delta

    def _run_conversation_blocking(self, combined: str) -> Dict[str, Any]:
        """Synchronous body executed on this agent's dedicated worker thread.

        Sets a context-local HERMES_HOME override (scoped to this thread) before
        doing any Hermes work — init, history load, and the run itself — so a
        concurrently-running peer agent cannot clobber this agent's home via the
        process-global env var. Any ask_human blocking also happens here, on this
        agent's own thread, so it cannot starve other agents.
        """
        token = _set_hermes_home_override(self._hermes_home)
        # Pin this team's project dir on THIS worker thread for the whole turn, so
        # every terminal/file tool call resolves TERMINAL_CWD to our repo even while
        # a peer agent's turn runs concurrently on its own thread.
        _set_terminal_cwd_override(
            getattr(self, "_project_dir", None)
            or _ensure_project_dir(self.cfg.get("team_id", "default"))
        )
        try:
            self._ensure_agent()
            # Heal a crashed team browser before the turn. Relaunch reuses the
            # same port, so the cdp_url already in config.yaml stays valid — no
            # rewrite needed on the happy path (this is just a health probe).
            team_browser_manager.ensure_team_browser(self.cfg.get("team_id", "default"))
            # Build the dynamic per-turn live context (project tree + last 10 peer
            # messages + minute-precise time). CRITICAL: this changes every turn,
            # so it must NOT go into the system message. The system message sits at
            # position 0, ahead of the whole conversation history; mutating its tail
            # ends the upstream prefix-cache match there and forces the ENTIRE
            # history to be re-billed as uncached input every turn. Instead we keep
            # ephemeral_system_prompt pinned to the STABLE base (so [system + tools +
            # history] is a byte-stable cacheable prefix) and prepend the volatile
            # live context to the FINAL user turn, where it costs only its own tokens.
            try:
                base = getattr(self, "_base_ephemeral", None)
                if base is not None:
                    # Pin the system prompt to the stable base (it may have been left
                    # as base+live by an older build / prior turn).
                    if self._ai_agent.ephemeral_system_prompt != base:
                        self._ai_agent.ephemeral_system_prompt = base
                    live = compose_live_context(
                        self.cfg.get("team_id", "default"), self.name, load_agents_config()
                    )
                    if live:
                        combined = f"{live}\n\n{combined}"
            except Exception as e:
                log.debug("[%s] live-context refresh failed: %s", self.name, e)
            history = self._load_session_from_db()
            # Show the actual inputs to this turn in the live trace, above the
            # thinking/tools/answer the model produces: the injected system
            # context first, then the user/task prompt. Ephemeral (ws-gated, not
            # persisted) — the History tab persists these separately.
            try:
                sysctx = getattr(self._ai_agent, "ephemeral_system_prompt", "") or ""
                if sysctx:
                    self._emit_exec("system", {"text": sysctx[:6000]})
                self._emit_exec("user", {"text": (combined or "")[:6000]})
            except Exception:
                pass
            return self._ai_agent.run_conversation(
                user_message=combined,
                task_id=f"agent_name:{self.name}",
                conversation_history=history,
            )
        finally:
            _reset_hermes_home_override(token)
            _reset_terminal_cwd_override()

    async def _process_tasks_batch(self, tasks: List[Dict[str, Any]]):
        task_ids = [t["id"] for t in tasks]
        task_preview = ", ".join([t["id"][:8] for t in tasks])
        # Per-turn state for LIVE transcript persistence: tool steps are written
        # to monitoring.db as they happen (see on_tool_complete) so a long or
        # interrupted turn is visible to digests + the supervisor, which only read
        # monitoring.db. Reset each turn; the end-of-turn logging below dedups
        # against this set so a COMPLETED turn isn't double-written.
        self._current_task_ids = task_ids
        self._live_logged_tool_ids = set()
        log.info("[%s] Processing batch: %s", self.name, task_preview)
        _broadcast("conversation_start", {
            "agent_name": self.name,
            "task_count": len(tasks),
            "task_ids": task_ids,
            "timestamp": time.time(),
        })

        combined = f"You have {len(tasks)} {_TASK_PROMPT_MARKER}:\n\n"
        for i, task in enumerate(tasks, 1):
            combined += f"--- [{i}] from {task['from_agent']} ---\n{task['payload']}\n\n"

        # Deliver passive STATUS/FYI addressed to this agent since its last
        # delivery. These never wake anyone (that's the point), but parking them
        # solely in the rolling team feed meant they scrolled away unseen on a
        # busy team — and senders escalated to waking TASKs just to be heard.
        # Delivered as a clearly-non-actionable trailer; watermark advances only
        # after the turn succeeds, so a failed turn redelivers rather than drops.
        passive_max_id = self._passive_watermark
        try:
            res = monitor_db.get_passive_messages_for(
                self.name, self._passive_watermark, limit=12
            )
            passive_max_id = res.get("max_id", self._passive_watermark)
            pmsgs = res.get("messages") or []
            if pmsgs:
                plines = []
                for p in pmsgs:
                    stamp = ""
                    try:
                        import datetime as _dt
                        stamp = _dt.datetime.fromtimestamp(p["timestamp"]).strftime("%H:%M")
                    except Exception:
                        pass
                    plines.append(f"  [{stamp}] {p['from_agent']} ({p['kind']}): {p['text']}")
                combined += (
                    "--- PASSIVE UPDATES addressed to you while idle (STATUS/FYI — "
                    "informational only; you owe NO reply and must not answer them) ---\n"
                    + "\n".join(plines) + "\n\n"
                )
        except Exception as e:
            log.debug("[%s] passive delivery failed: %s", self.name, e)

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                self._executor, self._run_conversation_blocking, combined
            )
            # If a stop was requested while the conversation was in-flight,
            # discard the result entirely so no state is mutated.
            if self._stop_requested:
                log.info("[%s] Stop requested while conversation in-flight — discarding result", self.name)
                return
            # Compaction during the run rotates session_id; persist it so the
            # next sweep (and a restart) resumes from the compacted session.
            self._persist_session_id_if_rotated()

            # The turn produced a response (provider is reachable) — clear any
            # infra-outage backoff so subsequent work runs at the normal cadence.
            if not response.get("failed"):
                self._infra_misses = 0
                self._infra_hold_until = 0.0

            # A hard LLM failure (proxy down, billing exhausted, repeated stream
            # drops) does NOT raise — Hermes returns failed=True with an empty or
            # partial turn. The old code fell straight through the success path:
            # it logged nothing to the UI (so the agent just flickered busy->idle
            # with no message) and marked the task DONE, silently consuming the
            # work. Surface it as a visible error + requeue so it isn't lost.
            if response.get("failed"):
                err = str(response.get("error") or response.get("final_response") or "LLM call failed")
                log.error("[%s] LLM turn failed (no response produced): %s", self.name, err[:200])
                infra = self._is_infra_failure(err)
                # Surface it in the UI, but throttle during a sustained outage so
                # we don't spam the monitoring log with one error per 10s tick.
                now = time.time()
                if now - self._last_llm_error_emit >= LLM_ERROR_EMIT_THROTTLE_SECONDS:
                    self._last_llm_error_emit = now
                    if infra:
                        err_content = (
                            f"⚠️ LLM provider unreachable — turn produced no response. "
                            f"Holding work until it recovers (auto-resumes). Detail: {err}"
                        )
                    else:
                        err_content = f"⚠️ LLM call failed — no response produced this turn: {err}"
                    monitor_db.log_message(self.name, "system", err_content, ",".join(task_ids))
                    _broadcast("message_logged", {
                        "agent_name": self.name,
                        "role": "system",
                        "content": err_content,
                        "task_id": task_preview,
                        "timestamp": time.time(),
                    })
                monitor_db.log_event(
                    self.name, "error",
                    data={"error": err[:500], "task_ids": task_ids,
                          "kind": "llm_infra" if infra else "llm_failure"},
                )
                _broadcast("error", {
                    "agent_name": self.name,
                    "task_ids": task_ids,
                    "error": err[:500],
                    "timestamp": time.time(),
                })
                if infra:
                    # Not the task's fault — wait for recovery without burning the
                    # retry budget. Set an exponential hold so the sweep's
                    # "more pending? wake now" path doesn't re-attempt instantly
                    # and spin against a down provider; the loop re-checks when the
                    # hold expires, so work still resumes promptly once it's back.
                    self._infra_misses += 1
                    backoff = min(
                        INFRA_RETRY_BACKOFF_BASE_SECONDS * (2 ** (self._infra_misses - 1)),
                        INFRA_RETRY_BACKOFF_MAX_SECONDS,
                    )
                    self._infra_hold_until = time.time() + backoff
                    self.queue.requeue_no_penalty(task_ids)
                    log.warning("[%s] Held %d task(s) for provider recovery (no penalty)",
                                self.name, len(task_ids))
                else:
                    self._requeue_or_deadletter(tasks)
                return

            # Record REAL token usage. Hermes returns session-cumulative counts,
            # so we log this batch's per-turn deltas plus the running totals —
            # actual numbers from the provider, not the char-based message
            # estimate. (Hermes' estimated_cost_usd is always 0 for proxy
            # models, so pricing lives swarm-side: see MODEL_PRICES_PER_MILLION
            # and the /teams/{id}/costs endpoint.)
            try:
                total = int(response.get("total_tokens", 0) or 0)
                inp = int(response.get("input_tokens", 0) or 0)
                outp = int(response.get("output_tokens", 0) or 0)
                cache = int(response.get("cache_read_tokens", 0) or 0)
                if total < self._last_total_tokens:
                    # Session rotated/compacted → all cumulative counters reset.
                    self._last_total_tokens = 0
                    self._last_input_tokens = 0
                    self._last_output_tokens = 0
                    self._last_cache_read_tokens = 0
                delta = total - self._last_total_tokens
                turn_in = max(0, inp - self._last_input_tokens)
                turn_out = max(0, outp - self._last_output_tokens)
                turn_cache = max(0, cache - self._last_cache_read_tokens)
                self._last_total_tokens = total
                self._last_input_tokens = inp
                self._last_output_tokens = outp
                self._last_cache_read_tokens = cache
                monitor_db.log_event(
                    self.name, "token_usage",
                    data={
                        "model": self._current_model,
                        "delta_tokens": delta,
                        "total_tokens": total,
                        "input_tokens": inp,
                        "output_tokens": outp,
                        "cache_read_tokens": cache,
                        "turn_input_tokens": turn_in,
                        "turn_output_tokens": turn_out,
                        "turn_cache_read_tokens": turn_cache,
                        "estimated_cost_usd": response.get("estimated_cost_usd", 0),
                    },
                )
                # Meter the team's daily spend (auto-pauses the team if over cap).
                from swarm_server.budget import budget_tracker
                budget_tracker.record_turn(
                    self.cfg.get("team_id", "default"), self._current_model,
                    turn_in, turn_out, turn_cache)
            except Exception as e:
                log.debug("[%s] token usage logging failed: %s", self.name, e)

            # Refresh + broadcast read-only runtime telemetry for the UI.
            try:
                tel = self._collect_telemetry()
                if tel:
                    _broadcast("telemetry", {
                        "agent_name": self.name,
                        "telemetry": tel,
                        "timestamp": time.time(),
                    })
            except Exception as e:
                log.debug("[%s] telemetry broadcast failed: %s", self.name, e)

            new_messages = response.get("messages", [])
            final = str(response.get("final_response", ""))
            log.info("[%s] Batch complete. Response: %s", self.name, final[:200])

            # This turn's output is everything after OUR task-prompt user message.
            # Anchor on the task-prompt marker, NOT merely the last user-role
            # message: Hermes injects mid-turn user-role messages (length
            # continuations, tool-use enforcement nudges), and slicing after the
            # last of those would drop this turn's earlier assistant tool calls
            # from the transcript, the self-loop tool signatures, and the
            # text-only ("did it act?") guard. The marker matches the preamble
            # built in `combined` below.
            last_user_idx = -1
            for i, msg in enumerate(new_messages):
                if msg.get("role") == "user" and _TASK_PROMPT_MARKER in (msg.get("content") or ""):
                    last_user_idx = i
            turn_messages = new_messages[last_user_idx + 1 :] if last_user_idx >= 0 else new_messages

            # Record the actual turn INPUTS so History is a faithful transcript,
            # not just the agent's replies: the injected system context (only when
            # it changed — it's large and mostly static) followed by the user/task
            # prompt. These precede the assistant/tool messages logged below.
            try:
                sysctx = getattr(self._ai_agent, "ephemeral_system_prompt", "") or ""
                if sysctx:
                    # Gate on the STABLE part of the prompt (the brief/soul), not
                    # the full text: the live-context section embeds a per-turn
                    # timestamp + team state, so hashing the whole thing would
                    # re-log this large block every single turn. Logging on
                    # stable-change records it ~once per session (and again only
                    # if the brief actually changes), with a current snapshot of
                    # the live context included for completeness.
                    stable = getattr(self, "_base_ephemeral", "") or sysctx
                    h = hash(stable)
                    if h != self._last_sysctx_hash:
                        self._last_sysctx_hash = h
                        monitor_db.log_message(self.name, "system", sysctx, ",".join(task_ids))
                        _broadcast("message_logged", {
                            "agent_name": self.name, "role": "system",
                            "content": sysctx, "task_id": task_preview,
                            "timestamp": time.time(),
                        })
            except Exception as e:
                log.debug("[%s] system-context logging failed: %s", self.name, e)
            monitor_db.log_message(self.name, "user", combined, ",".join(task_ids))
            _broadcast("message_logged", {
                "agent_name": self.name, "role": "user",
                "content": combined, "task_id": task_preview,
                "timestamp": time.time(),
            })

            # Tool steps were persisted live this turn (on_tool_complete); when
            # that happened, skip re-logging them here so the transcript isn't
            # doubled. Pure-text assistant replies + the final answer still log.
            live = bool(getattr(self, "_live_logged_tool_ids", None))
            for msg in turn_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "tool":
                    if live:
                        continue  # already persisted live as it completed
                    tc_id = msg.get("tool_call_id", "?")
                    content = f"📤 Tool Result [{tc_id}]: {content}"
                elif role == "assistant" and msg.get("tool_calls"):
                    if live:
                        # The tool calls are already in the live trace; keep only
                        # any accompanying assistant text, and drop empty ones.
                        if not (content or "").strip():
                            continue
                    else:
                        tcs = msg["tool_calls"]
                        tool_summary = " | ".join([
                            f"{tc.get('function', {}).get('name', '?')}()" for tc in tcs
                        ])
                        content = f"🛠️ Tool Calls: {tool_summary}\n\n{content or ''}"

                monitor_db.log_message(self.name, role, content, ",".join(task_ids))
                _broadcast("message_logged", {
                    "agent_name": self.name,
                    "role": role,
                    "content": content,
                    "task_id": task_preview,
                    "timestamp": time.time(),
                })

            monitor_db.log_event(self.name, "conversation_complete", data={"response_preview": final[:200]})
            _broadcast("conversation_complete", {
                "agent_name": self.name,
                "task_count": len(tasks),
                "response_preview": final[:200],
                "timestamp": time.time(),
            })
            for t in tasks:
                self.queue.mark_done(t["id"])

            # Passive STATUS/FYI delivered in this turn are now consumed.
            self._passive_watermark = passive_max_id

            # --- Text-only turn guard ----------------------------------------
            # Every turn ends with a stop-reason text message (API guarantee), so
            # that alone is not a fault. Fire ONLY when the turn ACTED on nothing:
            # its tool calls (if any) were all read-only, then it wrote a summary —
            # i.e. it investigated/mused and narrated instead of doing/delegating.
            # A turn that ran terminal, wrote a file, sent a peer message, logged a
            # decision, etc. counts as acting and is left alone (no false nudge on
            # a real deploy). Capped at one nudge per occurrence via
            # _text_only_nudged so it can never loop; cleared once a turn acts.
            # Skipped for supervisors (their feed prompt governs them) and on stop.
            try:
                tool_names = self._apply_turn_output_guards(tasks, turn_messages)

                # --- Idle-heartbeat backoff bookkeeping ----------------------
                # CONCRETE means the turn touched something outside coordination
                # AND outside pure reads — the same bar the supervisor signal
                # uses. A heartbeat turn that produced nothing concrete is a
                # miss (interval doubles); any concrete turn resets the cadence.
                concrete = any(
                    n and n not in self._NONACTION_TOOLS and n not in self._READONLY_TOOLS
                    for n in tool_names
                )
                hb_batch = any(t.get("from_agent") == "autonomous" for t in tasks)
                if concrete:
                    self._hb_misses = 0
                elif hb_batch:
                    self._hb_misses += 1
                    log.info("[%s] Heartbeat produced no concrete action (miss #%d) — backing off",
                             self.name, self._hb_misses)
                    monitor_db.log_event(self.name, "heartbeat_noop",
                                         data={"misses": self._hb_misses})

                # --- Cross-turn repetition guard (self-loop) ------------------
                # The pair/team loop detectors can't see ONE agent re-issuing the
                # identical tool call turn after turn (re-verifying, re-reading,
                # re-sending). Track per-turn signature sets; when one signature
                # recurs in SELF_LOOP_REPEATS of the last SELF_LOOP_WINDOW turns,
                # inject ONE corrective naming the exact call (cooldown-capped)
                # and reset the window so it can't immediately re-fire.
                self._turn_sigs.append(_turn_tool_signatures(turn_messages))
                rep = detect_repeated_signature(self._turn_sigs)
                now_ts = time.time()
                if rep and (now_ts - self._last_self_loop_nudge) >= SELF_LOOP_COOLDOWN_SECONDS:
                    self._last_self_loop_nudge = now_ts
                    self._turn_sigs.clear()
                    log.warning("[%s] SELF-LOOP detected — repeated across turns: %s",
                                self.name, rep[:140])
                    monitor_db.log_event(self.name, "self_loop_detected",
                                         data={"signature": rep[:300]})
                    _broadcast("self_loop_detected", {
                        "agent_name": self.name, "signature": rep[:300],
                        "timestamp": now_ts,
                    })
                    self.ingest_task("self-loop-guard", SELF_LOOP_NUDGE.format(signature=rep))
            except Exception as e:
                log.debug("[%s] post-turn guards failed: %s", self.name, e)
        except Exception as exc:
            log.error("[%s] Batch failed: %s", self.name, exc)
            monitor_db.log_event(self.name, "error", data={"error": str(exc), "task_ids": task_ids})
            _broadcast("error", {
                "agent_name": self.name,
                "task_ids": task_ids,
                "error": str(exc),
                "timestamp": time.time(),
            })
            # Don't strand tasks in 'processing' forever (the old zombie bug).
            # Requeue for another attempt; dead-letter once retries are exhausted.
            self._requeue_or_deadletter(tasks)

    def _requeue_or_deadletter(self, tasks: List[Dict[str, Any]]) -> None:
        """Requeue failed tasks for another attempt; dead-letter once retries
        are exhausted. Shared by the exception path and the failed-turn path so
        a batch is never silently consumed (the old zombie/lost-work bug)."""
        retry_ids = [t["id"] for t in tasks if int(t.get("retries", 0)) + 1 <= MAX_TASK_RETRIES]
        dead_ids = [t["id"] for t in tasks if int(t.get("retries", 0)) + 1 > MAX_TASK_RETRIES]
        if retry_ids:
            self.queue.requeue(retry_ids)
            log.warning("[%s] Requeued %d task(s) for retry", self.name, len(retry_ids))
            self._signal_wake()
        if dead_ids:
            self.queue.mark_failed(dead_ids)
            log.error("[%s] %d task(s) exhausted retries -> dead-letter", self.name, len(dead_ids))
            monitor_db.log_event(self.name, "task_failed", data={"task_ids": dead_ids})
            _broadcast("task_failed", {
                "agent_name": self.name,
                "task_ids": dead_ids,
                "timestamp": time.time(),
            })

    def start_sweep(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._wake = asyncio.Event()
        # Wake immediately if tasks were recovered or arrived before the loop ran.
        if self.queue.get_pending_count() > 0:
            self._wake.set()
        self._sweep_task = loop.create_task(self.sweep_loop())
