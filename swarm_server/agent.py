"""Agent daemon wrapper around a Hermes AIAgent instance."""

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm_server.config import (
    AUTONOMOUS_HEARTBEAT_PROMPT,
    AUTONOMOUS_HEARTBEAT_SECONDS,
    CRON_WAKEUP_PROMPT,
    LITELLM_API_BASE,
    LLM_API_KEY,
    LLM_ERROR_EMIT_THROTTLE_SECONDS,
    MAX_BATCH_SIZE,
    MAX_TASK_RETRIES,
    SUPERVISOR_FEED_CHAR_CAP,
    SUPERVISOR_FEED_PROMPT,
    SUPERVISOR_TOKEN_THRESHOLD,
    SWEEP_INTERVAL_SECONDS,
    _derive_workspace_path,
    _ensure_project_dir,
    compose_agent_soul,
    compose_live_context,
    compose_soul_identity,
    load_agents_config,
    save_agent_config,
    write_agent_hermes_config,
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
        # Static half of the ephemeral system prompt; set in _ensure_agent and
        # combined with per-turn live context before each run.
        self._base_ephemeral: Optional[str] = None
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
        # total so each batch can log its real delta (not just a char estimate).
        self._last_total_tokens = 0
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
        # message id already fed for review). Lazy-initialized to the peer's
        # current latest id, so a supervisor reviews NEW activity only and never
        # gets a peer's entire backlog dumped into its queue at once.
        self._sup_watermark: Dict[str, int] = {}

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
            })
        return out

    @staticmethod
    def _is_infra_failure(err: str) -> bool:
        """True when a failed turn is environmental (provider down, billing,
        timeout) rather than the task's fault — these should wait for recovery
        instead of burning the retry budget and dead-lettering during an outage."""
        e = (err or "").lower()
        return any(s in e for s in (
            "connection error", "apiconnection", "connection refused",
            "billing or credits", "credits exhausted", "timeout", "timed out",
            "max retries", "failed after", "service unavailable", "502", "503", "504",
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
        if self._ai_agent is not None:
            return
        with _agent_init_lock:
            if self._ai_agent is not None:
                return
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
                project_dir = _ensure_project_dir(team_id)
                os.environ["TERMINAL_CWD"] = str(project_dir)

                cdp_url = team_browser_manager.ensure_team_browser(team_id)

                # Per-agent model + sampling knobs (configurable from the UI;
                # fall back to the proxy default / Hermes defaults when unset).
                # Effective backend: per-agent override → swarm default → proxy.
                from swarm_server.model_config import resolve_model

                eff = resolve_model(self.cfg)
                model = eff["model"]
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
                if dis_ts:
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
                disabled = set(self.cfg.get("disabled_toolsets") or [])
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
                if "log_changes" not in existing_names and "log_changes" not in disabled:
                    from swarm_server.tools import _LOG_CHANGES_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_LOG_CHANGES_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("log_changes")
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

        # Replace the executor so future sweeps run on a clean thread.
        old_executor = self._executor
        old_executor.shutdown(wait=False)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"agent-{self.name}"
        )

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
            self.next_sweep_at = time.time() + timeout
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            await self._sweep()
            # A paused agent does NOTHING off-cycle either — no crons, no
            # autonomous heartbeat, no supervisor reviews — until resumed.
            if not self._paused:
                self._maybe_fire_crons()
                self._maybe_autonomous_heartbeat()
                self._maybe_feed_supervisor()

    async def _sweep(self):
        # Frozen by an emergency pause — process nothing and leave the queue
        # intact (work resumes untouched on resume). Return BEFORE draining.
        if self._paused:
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
            # (but never while paused — held work waits for resume).
            try:
                if not self._paused and self.queue.get_pending_count() > 0 and self._wake is not None:
                    self._wake.set()
            except Exception:
                pass

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
        if time.time() - self._last_active < self._heartbeat_seconds:
            return
        self._last_active = time.time()
        log.info("[%s] Autonomous heartbeat — injecting continue-mission task", self.name)
        monitor_db.log_event(self.name, "autonomous_heartbeat", data={})
        import datetime
        prompt = AUTONOMOUS_HEARTBEAT_PROMPT.format(
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        )
        self.ingest_task("autonomous", prompt)

    def _supervisor_threshold(self) -> int:
        """New-token threshold before a linked peer's activity is fed for review."""
        try:
            v = int(self.cfg.get("supervisor_token_threshold") or 0)
            return v if v > 0 else SUPERVISOR_TOKEN_THRESHOLD
        except (TypeError, ValueError):
            return SUPERVISOR_TOKEN_THRESHOLD

    def _render_feed_transcript(self, msgs: List[Dict[str, Any]]) -> str:
        """Render a peer's new messages oldest-first, capped to the most-recent
        slice (a marker notes any truncation — no silent loss)."""
        lines = []
        for m in msgs:
            role = (m.get("role") or "?").upper()
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"[{role}] {content}")
        text = "\n\n".join(lines)
        if len(text) > SUPERVISOR_FEED_CHAR_CAP:
            text = ("[…older activity in this window truncated…]\n\n"
                    + text[-SUPERVISOR_FEED_CHAR_CAP:])
        return text

    def _maybe_feed_supervisor(self) -> None:
        """Supervisor agents only: push each linked peer's recent transcript into
        THIS agent's own queue once the peer accrues enough new activity.

        Daemon-side and automatic — the supervisor never calls a tool to fetch;
        the review simply arrives as a queued task, like any peer message. One
        feed per tick (others follow on later ticks) and only while not busy /
        not backed up, so reviews don't pile up.
        """
        if not self.cfg.get("is_supervisor") or self._stop_requested:
            return
        if self.state == AGENT_STATE_BUSY:
            return
        try:
            if self.queue.get_pending_count() > 0:
                return  # already has a review (or other task) queued — wait
        except Exception:
            return
        peers = self.cfg.get("allowed_peers") or []
        threshold = self._supervisor_threshold()
        # Gather every peer with a reviewable backlog, then feed the one with the
        # MOST unreviewed activity. Feeding the first qualifier in list order (and
        # returning) let chatty peers early in allowed_peers permanently starve a
        # busy/looping peer later in the list — so the agent most likely in trouble
        # (it generates the most transcript) was the one never reviewed. Picking the
        # largest backlog both prevents starvation and prioritizes the loudest agent.
        candidates = []  # (tokens, peer)
        for peer in peers:
            try:
                wm = self._sup_watermark.get(peer)
                activity = monitor_db.get_new_activity(peer, wm or 0)
                if wm is None:
                    # First time watching this peer — start from "now" so we don't
                    # dump its whole history; review only what happens next.
                    self._sup_watermark[peer] = activity["max_id"]
                    continue
                if activity["count"] <= 0 or activity["tokens"] < threshold:
                    continue
                candidates.append((activity["tokens"], peer))
            except Exception as e:
                log.debug("[%s] supervisor scan for %s failed: %s", self.name, peer, e)
        if not candidates:
            return
        tokens, peer = max(candidates)  # highest unreviewed token backlog wins
        try:
            wm = self._sup_watermark.get(peer)
            msgs = monitor_db.get_messages_since(peer, wm)
            if not msgs:
                return
            max_id = max(int(m["id"]) for m in msgs)
            transcript = self._render_feed_transcript(msgs)
            prompt = SUPERVISOR_FEED_PROMPT.format(
                peer=peer, tokens=tokens, transcript=transcript
            )
            self._sup_watermark[peer] = max_id
            monitor_db.log_event(
                self.name, "supervisor_feed", to_agent=peer,
                data={"tokens": tokens, "messages": len(msgs)},
            )
            self.ingest_task("supervisor-feed", prompt)  # one review per tick
        except Exception as e:
            log.debug("[%s] supervisor feed for %s failed: %s", self.name, peer, e)

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
            prompt = CRON_WAKEUP_PROMPT.format(
                schedule=sched,
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
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
        try:
            self._ensure_agent()
            # Heal a crashed team browser before the turn. Relaunch reuses the
            # same port, so the cdp_url already in config.yaml stays valid — no
            # rewrite needed on the happy path (this is just a health probe).
            team_browser_manager.ensure_team_browser(self.cfg.get("team_id", "default"))
            # Refresh the dynamic half of the ephemeral system prompt (live project
            # tree + last 10 peer messages). Injected at API-call time, so updating
            # it here keeps the context current WITHOUT invalidating the cached
            # stable/context/volatile system prompt.
            try:
                base = getattr(self, "_base_ephemeral", None)
                if base is not None:
                    live = compose_live_context(
                        self.cfg.get("team_id", "default"), self.name, load_agents_config()
                    )
                    self._ai_agent.ephemeral_system_prompt = base + "\n\n" + live
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

        combined = f"You have {len(tasks)} new message(s) to process:\n\n"
        for i, task in enumerate(tasks, 1):
            combined += f"--- [{i}] from {task['from_agent']} ---\n{task['payload']}\n\n"

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
                    # retry budget. Retries on the natural sweep tick (no wake), so
                    # the work resumes the moment the provider is back.
                    self.queue.requeue_no_penalty(task_ids)
                    log.warning("[%s] Held %d task(s) for provider recovery (no penalty)",
                                self.name, len(task_ids))
                else:
                    self._requeue_or_deadletter(tasks)
                return

            # Record REAL token usage. Hermes returns session-cumulative counts,
            # so we log this batch's delta plus the running total + cost — actual
            # numbers from the provider, not the char-based message estimate.
            try:
                total = int(response.get("total_tokens", 0) or 0)
                delta = total - self._last_total_tokens
                if delta < 0:  # session rotated/compacted -> counter reset
                    delta = total
                self._last_total_tokens = total
                monitor_db.log_event(
                    self.name, "token_usage",
                    data={
                        "delta_tokens": delta,
                        "total_tokens": total,
                        "input_tokens": int(response.get("input_tokens", 0) or 0),
                        "output_tokens": int(response.get("output_tokens", 0) or 0),
                        "cache_read_tokens": int(response.get("cache_read_tokens", 0) or 0),
                        "estimated_cost_usd": response.get("estimated_cost_usd", 0),
                    },
                )
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

            last_user_idx = -1
            for i, msg in enumerate(new_messages):
                if msg.get("role") == "user":
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
