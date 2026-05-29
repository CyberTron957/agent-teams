"""Agent daemon wrapper around a Hermes AIAgent instance."""

import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm_server.config import (
    LITELLM_API_BASE,
    MAX_BATCH_SIZE,
    MAX_TASK_RETRIES,
    SWEEP_INTERVAL_SECONDS,
    _derive_workspace_path,
    compose_agent_soul,
    load_agents_config,
    save_agent_config,
    write_agent_hermes_config,
)
from swarm_server.monitoring import monitor_db
from swarm_server.queue import TaskQueue
from swarm_server.tools import (
    _ASK_HUMAN_TOOL_SCHEMA,
    _SEND_PEER_MESSAGE_TOOL_SCHEMA,
    _daemon_registry,
    _register_custom_tools,
)
from swarm_server.websocket import _agent_init_lock, _broadcast

log = logging.getLogger("swarm.agent")

AGENT_STATE_IDLE = "idle"
AGENT_STATE_BUSY = "busy"
AGENT_STATE_ASKING_HUMAN = "asking_human"

HERMES_AGENT_PATH = "/Users/pradhyun/.hermes/hermes-agent"


def _ensure_hermes_on_path() -> None:
    """Add the Hermes package dir to sys.path exactly once (no duplicate growth)."""
    if HERMES_AGENT_PATH not in sys.path:
        sys.path.insert(0, HERMES_AGENT_PATH)


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

                # Enable Hermes' built-in context/session compression for this
                # agent. Must be written before AIAgent() so init_agent picks it
                # up from {HERMES_HOME}/config.yaml.
                write_agent_hermes_config(self._hermes_home)

                # Prepare config with agent_id for soul composition
                soul_cfg = dict(self.cfg)
                soul_cfg["agent_id"] = self.name
                full_cfg = load_agents_config()

                self._ai_agent = AIAgent(
                    base_url=LITELLM_API_BASE,
                    api_key="sk-1234",
                    model="litellm-model",
                    session_id=self.cfg["session_id"],
                    skip_memory=False,
                    skip_context_files=False,
                    quiet_mode=True,
                    ephemeral_system_prompt=compose_agent_soul(soul_cfg, full_cfg),
                )
                _register_custom_tools()

                existing_names = {
                    t.get("function", {}).get("name") for t in (self._ai_agent.tools or [])
                }
                if "send_peer_message" not in existing_names:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_SEND_PEER_MESSAGE_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("send_peer_message")
                if "ask_human" not in existing_names:
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_ASK_HUMAN_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("ask_human")
                if "log_changes" not in existing_names:
                    from swarm_server.tools import _LOG_CHANGES_TOOL_SCHEMA
                    self._ai_agent.tools = list(self._ai_agent.tools or [])
                    self._ai_agent.tools.append(_LOG_CHANGES_TOOL_SCHEMA)
                    self._ai_agent.valid_tool_names.add("log_changes")

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
        monitor_db.log_event(
            self.name, "context_compacted",
            data={"old_session": old_sid, "new_session": live_sid},
        )
        _broadcast("context_compacted", {
            "agent_name": self.name,
            "old_session": old_sid,
            "new_session": live_sid,
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
        log.info("[%s] Sweep loop started (interval=%ds, event-driven)", self.name, SWEEP_INTERVAL_SECONDS)
        while True:
            self.next_sweep_at = time.time() + SWEEP_INTERVAL_SECONDS
            # Wake on a new task, or fall through on the periodic safety tick.
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=SWEEP_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            await self._sweep()

    async def _sweep(self):
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
            with self._lock:
                self.state = AGENT_STATE_IDLE
                self.next_sweep_at = time.time() + SWEEP_INTERVAL_SECONDS
            self._emit_state_change()
            # More queued while we were busy? Wake immediately rather than wait.
            try:
                if self.queue.get_pending_count() > 0 and self._wake is not None:
                    self._wake.set()
            except Exception:
                pass

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
            history = self._load_session_from_db()
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
            # Compaction during the run rotates session_id; persist it so the
            # next sweep (and a restart) resumes from the compacted session.
            self._persist_session_id_if_rotated()
            new_messages = response.get("messages", [])
            final = str(response.get("final_response", ""))
            log.info("[%s] Batch complete. Response: %s", self.name, final[:200])

            last_user_idx = -1
            for i, msg in enumerate(new_messages):
                if msg.get("role") == "user":
                    last_user_idx = i
            turn_messages = new_messages[last_user_idx + 1 :] if last_user_idx >= 0 else new_messages

            for msg in turn_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "assistant" and msg.get("tool_calls"):
                    tcs = msg["tool_calls"]
                    tool_summary = " | ".join([
                        f"{tc.get('function', {}).get('name', '?')}()" for tc in tcs
                    ])
                    content = f"🛠️ Tool Calls: {tool_summary}\n\n{content or ''}"

                if role == "tool":
                    tc_id = msg.get("tool_call_id", "?")
                    content = f"📤 Tool Result [{tc_id}]: {content}"

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
