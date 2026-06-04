"""Custom tool schemas, registry, and handlers for P2P communication."""

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List

from swarm_server.monitoring import monitor_db
from swarm_server.websocket import _broadcast

log = logging.getLogger("swarm.tools")

# How long ask_human blocks the worker thread waiting for an in-turn answer.
# On elapse the question stays PENDING (not failed) and the answer is re-delivered
# later as a task, so the agent resumes even if the human was away for hours.
# Default 6h: the human usually answers in-turn (smooth resume) rather than the
# agent ending its turn and resuming later via the re-delivered task.
import os as _os
_ASK_HUMAN_WAIT_SECONDS = int(_os.environ.get("SWARM_ASK_HUMAN_WAIT_SECONDS", "21600"))

# Maps agent_name -> AgentDaemon instance (populated at runtime by server.py)
_daemon_registry: Dict[str, Any] = {}

# Global Human Inbox Registry — tracks active/ historical human questions
_pending_human_questions: Dict[str, Dict[str, Any]] = {}
_pending_lock = threading.Lock()


def add_pending_question(agent_name: str, question: str) -> str:
    """Register a new human question and return its ID."""
    from swarm_server.config import MAX_PENDING_QUESTIONS

    qid = str(uuid.uuid4())
    with _pending_lock:
        _pending_human_questions[qid] = {
            "id": qid,
            "agent_name": agent_name,
            "question": question,
            "timestamp": time.time(),
            "status": "pending",
            "response": None,
        }
        # Bound the in-memory registry: drop oldest RESOLVED questions past the
        # cap (never drop pending ones — an agent may still be waiting on them).
        if len(_pending_human_questions) > MAX_PENDING_QUESTIONS:
            resolved = sorted(
                (q for q in _pending_human_questions.values() if q["status"] != "pending"),
                key=lambda q: q["timestamp"],
            )
            for q in resolved[: len(_pending_human_questions) - MAX_PENDING_QUESTIONS]:
                _pending_human_questions.pop(q["id"], None)
    return qid


def get_pending_questions() -> List[Dict[str, Any]]:
    """Return a copy of all questions (pending / answered / timed_out)."""
    with _pending_lock:
        return [dict(q) for q in _pending_human_questions.values()]


def answer_question(qid: str, response: str) -> bool:
    """Mark a question as answered with the given response text."""
    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if not q:
            return False
        q["status"] = "answered"
        q["response"] = response
        q["answered_at"] = time.time()
        return True


def mark_timed_out(qid: str) -> None:
    """Mark a pending question as timed out."""
    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if q and q["status"] == "pending":
            q["status"] = "timed_out"


# ---------------------------------------------------------------------------
# Self-config proposals — an agent can PROPOSE a change to its own config;
# the human approves/rejects it in the dashboard (read-only self-awareness).
# Mirrors the human-question inbox above.
# ---------------------------------------------------------------------------
# Editable keys an agent may propose (kept in sync with server's editable set).
SELF_CONFIG_ALLOWED_KEYS = (
    "model", "provider", "temperature", "max_tokens", "reasoning_effort",
    "sweep_interval", "max_iterations", "enabled_toolsets", "disabled_toolsets",
    "compression_threshold", "autonomous", "heartbeat_seconds",
)

_pending_config_proposals: Dict[str, Dict[str, Any]] = {}
_proposals_lock = threading.Lock()


def add_config_proposal(agent_name: str, changes: Dict[str, Any], reason: str) -> str:
    pid = str(uuid.uuid4())
    with _proposals_lock:
        _pending_config_proposals[pid] = {
            "id": pid,
            "agent_name": agent_name,
            "changes": changes,
            "reason": reason,
            "timestamp": time.time(),
            "status": "pending",
        }
        # Bound the registry: drop oldest RESOLVED proposals past the cap.
        if len(_pending_config_proposals) > 200:
            resolved = sorted(
                (p for p in _pending_config_proposals.values() if p["status"] != "pending"),
                key=lambda p: p["timestamp"],
            )
            for p in resolved[: len(_pending_config_proposals) - 200]:
                _pending_config_proposals.pop(p["id"], None)
    return pid


def get_config_proposals() -> List[Dict[str, Any]]:
    with _proposals_lock:
        return [dict(p) for p in _pending_config_proposals.values()]


def resolve_config_proposal(pid: str, status: str):
    """Mark a proposal approved/rejected; returns the (copied) proposal or None."""
    with _proposals_lock:
        p = _pending_config_proposals.get(pid)
        if not p:
            return None
        p["status"] = status
        p["resolved_at"] = time.time()
        return dict(p)

# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------
_SEND_PEER_MESSAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_peer_message",
        "description": (
            "Send a message to another agent in the swarm. The target will pick it up "
            "on its next sweep and process it. Use this to chat, pass results, or delegate work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Name of the target agent."},
                "message": {"type": "string", "description": "The message to send."},
            },
            "required": ["to_agent", "message"],
        },
    },
}

_ASK_HUMAN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_human",
        "description": "Ask a human for clarification. This call blocks until the human responds.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to present to the human."},
            },
            "required": ["question"],
        },
    },
}

_REQUEST_HUMAN_TAKEOVER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_human_takeover",
        "description": (
            "Hand the live browser to a human when you hit something you can't or "
            "shouldn't do yourself: a login wall, a CAPTCHA, an SMS/email/2FA "
            "verification code, a consent prompt, or any manual click-through. The "
            "browser is moved onto the human's real screen so they act in the SAME "
            "session (cookies persist), and YOUR run blocks until they finish. When "
            "they're done you resume exactly where you left off, now past the "
            "obstacle. This is NOT for questions — use ask_human for those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "What the human must do and on which site/URL, e.g. "
                        "'Log in to linkedin.com' or 'Solve the captcha on the "
                        "checkout page'."
                    ),
                },
            },
            "required": ["reason"],
        },
    },
}

_LOG_CHANGES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "log_changes",
        "description": (
            "Log an important event, status update, or completed task to the shared team activity log. "
            "Use this after completing work, making decisions, or when something notable happens. "
            "This helps the whole team stay informed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "The log entry text. Be concise but informative.",
                },
            },
            "required": ["entry"],
        },
    },
}


_GET_SELF_CONFIG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_self_config",
        "description": (
            "Read YOUR OWN current configuration and live runtime telemetry: model, "
            "provider, allowed/disabled toolsets, sweep interval, max iterations, "
            "reasoning effort, context window size + current context usage, tokens "
            "spent this session, and the compression threshold. Use it to understand "
            "how you are set up before deciding whether a change is worth proposing."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_REQUEST_CONFIG_CHANGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_config_change",
        "description": (
            "PROPOSE a change to your own configuration. You CANNOT change your own "
            "settings directly — this sends the proposal to the human operator, who "
            "approves or rejects it in the dashboard. The change takes effect ONLY "
            "after approval. Use it when you believe a different model, tool set, or "
            "setting would help you do your job better; always explain why."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "object",
                    "description": (
                        "Map of setting -> new value. Allowed keys: model, provider, "
                        "temperature, max_tokens, reasoning_effort, sweep_interval, "
                        "max_iterations, enabled_toolsets, disabled_toolsets, "
                        "compression_threshold, autonomous, heartbeat_seconds."
                    ),
                },
                "reason": {"type": "string", "description": "Why this change is warranted."},
            },
            "required": ["changes", "reason"],
        },
    },
}


_SCHEDULE_WAKEUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "schedule_wakeup",
        "description": (
            "Schedule a recurring future wake-up for YOURSELF. At each scheduled "
            "time the swarm injects your 'instruction' to you as a new task and you "
            "act on it — use this for anything periodic (a 9am competitor check, an "
            "hourly metrics pull, a Monday digest) so it happens on time without a "
            "human. The instruction must be SELF-CONTAINED: when it fires you may "
            "have no other context. Check your current wake-ups (listed in the live "
            "team context) first so you don't create duplicates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "schedule": {
                    "type": "string",
                    "description": (
                        "When to fire. Standard 5-field cron 'min hour day month weekday' "
                        "(e.g. '0 9 * * 1-5' = 9am weekdays, '*/30 * * * *' = every 30 min), "
                        "a macro (@hourly, @daily, @weekly, @monthly), or an interval "
                        "(@every 30m, @every 2h). Server local time."
                    ),
                },
                "instruction": {
                    "type": "string",
                    "description": "The self-contained task to run each time it fires.",
                },
            },
            "required": ["schedule", "instruction"],
        },
    },
}

_CANCEL_WAKEUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cancel_wakeup",
        "description": (
            "Cancel one of your scheduled cron wake-ups by its id. The ids of your "
            "current wake-ups are listed in the live team context. Use this to remove "
            "a schedule you no longer need or one you created by mistake."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cron_id": {"type": "string", "description": "The id of the wake-up to cancel."},
            },
            "required": ["cron_id"],
        },
    },
}


_PAUSE_AGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pause_agent",
        "description": (
            "SUPERVISOR EMERGENCY BRAKE. Immediately freeze one agent you watch — "
            "interrupting its in-flight turn at the next tool boundary — when it is "
            "doing something genuinely dangerous or irreversible (e.g. destroying "
            "production infra, killing processes it doesn't own, deleting data, "
            "leaking secrets, or stuck in a destructive loop). Its pending work is "
            "PRESERVED and resumes on resume_agent. This is disruptive: a paused "
            "agent does nothing until lifted, so use it ONLY for real emergencies "
            "where a warning message would arrive too late — not for slow, "
            "low-quality, or merely off-track work (message them instead). After "
            "pausing, a human is notified to review."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent to pause (must be one you watch).",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Specific justification for the emergency stop — what "
                        "dangerous action you observed and why it can't wait. "
                        "Required, and recorded for the human."
                    ),
                },
            },
            "required": ["agent", "reason"],
        },
    },
}

_RESUME_AGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "resume_agent",
        "description": (
            "Lift a pause you (or a human) placed on an agent, once the danger has "
            "passed — e.g. the agent acknowledged the issue, a human approved, or "
            "you've confirmed it's safe to continue. The agent picks up its held "
            "queue where it left off. Only resume when you are confident the unsafe "
            "condition is resolved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the paused agent to resume.",
                },
            },
            "required": ["agent"],
        },
    },
}


# Swarm-native tools (registered on every agent AFTER Hermes init, so they
# survive any toolset whitelist). `send_peer_message` is REQUIRED (agents end
# their turn with it) and cannot be disabled; the rest are optional and can be
# turned off per agent via `disabled_toolsets`. pause_agent/resume_agent are
# registered ONLY on supervisor agents (see agent.py).
REQUIRED_SWARM_TOOLS = ("send_peer_message",)
_SWARM_TOOL_SCHEMAS = (
    _SEND_PEER_MESSAGE_TOOL_SCHEMA,
    _ASK_HUMAN_TOOL_SCHEMA,
    _LOG_CHANGES_TOOL_SCHEMA,
    _GET_SELF_CONFIG_TOOL_SCHEMA,
    _REQUEST_CONFIG_CHANGE_TOOL_SCHEMA,
    _SCHEDULE_WAKEUP_TOOL_SCHEMA,
    _CANCEL_WAKEUP_TOOL_SCHEMA,
    _PAUSE_AGENT_TOOL_SCHEMA,
    _RESUME_AGENT_TOOL_SCHEMA,
)


def list_swarm_tools():
    """The swarm's own tools (not Hermes toolsets) as [{name, description, required}]
    for the dashboard's enabled/disabled-toolsets picker."""
    out = []
    for s in _SWARM_TOOL_SCHEMAS:
        fn = (s or {}).get("function", {}) if isinstance(s, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": str(fn.get("description") or "")[:160],
            "required": name in REQUIRED_SWARM_TOOLS,
        })
    return out


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------
def _send_peer_message_handler(args: dict, **kwargs) -> str:
    to_agent = args.get("to_agent", "")
    message = args.get("message", "")
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    from swarm_server.config import load_agents_config, peer_allowed

    cfg = load_agents_config()

    target = _daemon_registry.get(to_agent)
    if target is None:
        known = list(_daemon_registry.keys())
        return json.dumps({"success": False, "error": f"Unknown agent '{to_agent}'. Known: {known}"})

    if not peer_allowed(cfg, caller, to_agent):
        caller_team = cfg["agents"].get(caller, {}).get("team_id", "?")
        target_team = cfg["agents"].get(to_agent, {}).get("team_id", "?")
        reason = (
            "cross-team communication" 
            if caller_team != target_team else 
            "not in allowed_peers"
        )
        log.warning(
            "[send_peer_message] DENIED %s -> %s (%s)", caller, to_agent, reason
        )
        monitor_db.log_event(
            caller, "link_violation",
            to_agent=to_agent,
            data={"reason": reason, "target_team": target_team},
        )
        _broadcast("link_violation", {
            "from_agent": caller,
            "to_agent": to_agent,
            "reason": reason,
            "timestamp": __import__("time").time(),
        })
        return json.dumps({
            "success": False,
            "error": (
                f"Messaging to '{to_agent}' denied ({reason}). "
                f"You are only linked to: {cfg['agents'].get(caller, {}).get('allowed_peers', [])}"
            ),
        })

    task_id = target.ingest_task(from_agent=caller, payload=message)
    log.info("[send_peer_message] %s -> %s | task_id=%s", caller, to_agent, task_id[:8])

    # Persist the peer message (not just broadcast it) so historical/REST queries
    # can reconstruct the conversation graph, not only the live dashboard.
    monitor_db.log_event(
        caller, "message_sent",
        from_agent=caller, to_agent=to_agent, task_id=task_id,
        data={"message_preview": message[:120]},
    )

    _broadcast("message_sent", {
        "from_agent": caller,
        "to_agent": to_agent,
        "task_id": task_id,
        "message_preview": message[:120],
        "timestamp": __import__("time").time(),
    })

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "message": f"Message enqueued to '{to_agent}' successfully.",
    })


def _ask_human_handler(args: dict, **kwargs) -> str:
    question = args.get("question", "")
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    daemon = _daemon_registry.get(caller)
    if daemon is None:
        return json.dumps({"error": f"Caller agent '{caller}' not registered."})

    log.info("[%s] [ask_human] Question: %s", daemon.name, question)

    # Register in global inbox
    qid = add_pending_question(caller, question)
    daemon.human_question_id = qid

    from swarm_server.agent import AGENT_STATE_ASKING_HUMAN, AGENT_STATE_BUSY

    monitor_db.log_event(caller, "human_waiting", data={"question": question, "question_id": qid})
    _broadcast("human_waiting", {
        "agent_name": caller,
        "question": question,
        "question_id": qid,
        "timestamp": time.time(),
    })

    with daemon._lock:
        daemon.state = AGENT_STATE_ASKING_HUMAN

    daemon.human_event.clear()
    daemon.human_response = None
    # Block the worker thread for this long waiting for an in-turn answer (smooth
    # path when a human is present). If it elapses, we DON'T fail — the question
    # stays pending and a late answer is re-delivered as a task. Kept modest so a
    # single blocked agent never parks its thread for hours.
    daemon.human_event.wait(timeout=_ASK_HUMAN_WAIT_SECONDS)

    with daemon._lock:
        daemon.state = AGENT_STATE_BUSY

    # Check if a response was provided via the API
    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if q and q["status"] == "answered":
            daemon.human_response = q["response"]
        # On timeout we deliberately LEAVE the question 'pending' (do NOT mark it
        # timed_out): it stays open in the human's inbox indefinitely, and when
        # they eventually answer, the inbox endpoint re-delivers the response to
        # this agent as a new task (see respond_to_human_question). This is what
        # makes a 24/7 swarm survive a human who is away for hours.

    if not daemon.human_event.is_set():
        log.info(
            "[%s] [ask_human] No response in %ds — question stays pending; agent "
            "ends turn and will be re-notified when answered", daemon.name, _ASK_HUMAN_WAIT_SECONDS,
        )
        # NOTE: deliberately do NOT set state to "idle" here. This handler runs
        # *inside* run_conversation on the agent's worker thread, and that
        # conversation is still in flight. The sweep loop's finally-block is the
        # single owner of the busy→idle transition.
        return json.dumps({
            "success": True,
            "status": "waiting_for_human",
            "message": (
                "Your question is saved in the human's inbox (id=" + qid + ") and "
                "will NOT expire. The moment the human answers, "
                "you will receive their response as a new task and resume exactly "
                "where you left off (e.g. log in and publish). Stop here."
            ),
        })

    log.info("[%s] [ask_human] Response received: %s", daemon.name, daemon.human_response)
    monitor_db.log_event(caller, "human_responded", data={"question": question, "response": daemon.human_response})
    _broadcast("human_responded", {
        "agent_name": caller,
        "question": question,
        "response": daemon.human_response,
        "question_id": qid,
        "timestamp": time.time(),
    })

    return json.dumps({"success": True, "response": daemon.human_response})


def _request_human_takeover_handler(args: dict, **kwargs) -> str:
    """Move the team browser onto the human's screen, block until they finish, then
    send it back to the hidden display and resume. Mirrors ask_human's blocking but
    brackets it with begin_takeover/end_takeover so the handoff is seamless."""
    reason = args.get("reason", "")
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    daemon = _daemon_registry.get(caller)
    if daemon is None:
        return json.dumps({"error": f"Caller agent '{caller}' not registered."})
    team_id = (daemon.cfg or {}).get("team_id", "default")

    # Bring the browser onto the human's real screen (same persistent profile).
    try:
        from swarm_server.browser_pool import team_browser_manager
        team_browser_manager.begin_takeover(team_id)
    except Exception as e:
        log.error("[%s] [takeover] begin_takeover failed: %s", caller, e)

    prompt = (
        "🙋 BROWSER TAKEOVER requested by '" + caller + "':\n" + reason +
        "\n\nThe agent's browser is now visible on your screen. Do the step above "
        "in that window, then reply 'done' here to hand control back."
    )
    log.info("[%s] [takeover] %s", daemon.name, reason)

    from swarm_server.agent import AGENT_STATE_ASKING_HUMAN, AGENT_STATE_BUSY

    qid = add_pending_question(caller, prompt)
    daemon.human_question_id = qid
    monitor_db.log_event(caller, "human_waiting",
                         data={"question": prompt, "question_id": qid, "kind": "takeover"})
    _broadcast("human_waiting", {
        "agent_name": caller, "question": prompt, "question_id": qid,
        "kind": "takeover", "timestamp": time.time(),
    })

    with daemon._lock:
        daemon.state = AGENT_STATE_ASKING_HUMAN
    daemon.human_event.clear()
    daemon.human_response = None
    daemon.human_event.wait(timeout=_ASK_HUMAN_WAIT_SECONDS)
    with daemon._lock:
        daemon.state = AGENT_STATE_BUSY

    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if q and q["status"] == "answered":
            daemon.human_response = q["response"]

    if not daemon.human_event.is_set():
        # Human not back yet: leave the browser ON their screen (they still need to
        # act) and the request pending. When they answer, the inbox endpoint sends
        # the browser back to the hidden display and re-delivers the response.
        return json.dumps({
            "success": True,
            "status": "waiting_for_human",
            "message": (
                "Takeover request saved in the human's inbox (id=" + qid + "). The "
                "browser is on their screen waiting and will NOT expire — when they "
                "finish and reply, you resume as a new task, past the obstacle. Stop here."
            ),
        })

    # Human finished in-turn — send the browser back to the hidden display, resume.
    try:
        from swarm_server.browser_pool import team_browser_manager
        team_browser_manager.end_takeover(team_id)
    except Exception as e:
        log.error("[%s] [takeover] end_takeover failed: %s", caller, e)
    monitor_db.log_event(caller, "human_responded",
                         data={"question": prompt, "response": daemon.human_response, "kind": "takeover"})
    _broadcast("human_responded", {
        "agent_name": caller, "response": daemon.human_response,
        "question_id": qid, "kind": "takeover", "timestamp": time.time(),
    })
    return json.dumps({
        "success": True, "status": "completed", "response": daemon.human_response,
        "message": (
            "Human finished the manual step; the browser session is now "
            "authenticated/unblocked. Continue your task from here."
        ),
    })


def _log_changes_handler(args: dict, **kwargs) -> str:
    import time
    from datetime import datetime

    entry = args.get("entry", "")
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    if not entry.strip():
        return json.dumps({"success": False, "error": "Empty log entry."})

    # Get team_id from config
    from swarm_server.config import (
        load_agents_config,
        _get_team_workspace_path,
        _derive_workspace_path,
    )

    cfg = load_agents_config()
    caller_cfg = cfg["agents"].get(caller, {})
    team_id = caller_cfg.get("team_id", "default")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"# [{timestamp}] {caller}: {entry}\n\n"

    def _append(path, header: str) -> None:
        """Append the entry to a log file, seeding a header if it's new."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, "a", encoding="utf-8") as f:
                f.write(log_line)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(log_line)

    # Write to BOTH the shared team log (the canonical project changelog every
    # teammate reads) AND the caller's own per-agent log (their personal
    # activity trail). One log_changes call -> two destinations.
    team_log = _get_team_workspace_path(team_id) / "agent_log.md"
    agent_log = _derive_workspace_path(team_id, caller) / "agent_log.md"
    try:
        _append(team_log, "# Team Activity Log\n\n")
        _append(agent_log, f"# {caller} Activity Log\n\n")
        log.info("[log_changes] %s logged (team + self): %s", caller, entry[:80])
    except Exception as e:
        log.warning("[log_changes] Failed to write log for %s: %s", caller, e)
        return json.dumps({"success": False, "error": f"Failed to write log: {e}"})

    # Also log to monitoring
    monitor_db.log_event(caller, "agent_log", data={"entry": entry})
    _broadcast("log_changes", {
        "agent_name": caller,
        "entry": entry,
        "timestamp": time.time(),
    })

    return json.dumps({"success": True, "message": "Log entry recorded."})


def _caller_from_kwargs(kwargs: dict) -> str:
    task_id_arg = kwargs.get("task_id", "")
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        return task_id_arg.split(":", 1)[1]
    return "unknown"


def _get_self_config_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    from swarm_server.config import load_agents_config, DEFAULT_MODEL

    cfg = load_agents_config()
    a = cfg["agents"].get(caller)
    if a is None:
        return json.dumps({"success": False, "error": f"Agent '{caller}' not found."})
    editable = {k: a.get(k) for k in SELF_CONFIG_ALLOWED_KEYS}
    editable["model"] = editable.get("model") or DEFAULT_MODEL
    daemon = _daemon_registry.get(caller)
    telemetry = dict(getattr(daemon, "_telemetry", {}) or {}) if daemon else {}
    crons = daemon.crons_runtime() if daemon else list(a.get("crons") or [])
    return json.dumps({
        "success": True,
        "agent_name": caller,
        "team_id": a.get("team_id"),
        "config": editable,
        "allowed_peers": a.get("allowed_peers", []),
        "crons": crons,
        "telemetry": telemetry,
    }, default=str)


def _request_config_change_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    changes = args.get("changes") or {}
    reason = (args.get("reason") or "").strip()
    if not isinstance(changes, dict) or not changes:
        return json.dumps({"success": False, "error": "'changes' must be a non-empty object."})
    filtered = {k: v for k, v in changes.items() if k in SELF_CONFIG_ALLOWED_KEYS}
    if not filtered:
        return json.dumps({
            "success": False,
            "error": f"No allowed keys in 'changes'. Allowed: {list(SELF_CONFIG_ALLOWED_KEYS)}",
        })

    pid = add_config_proposal(caller, filtered, reason)
    log.info("[%s] [request_config_change] proposal=%s changes=%s", caller, pid[:8], filtered)
    monitor_db.log_event(
        caller, "config_proposal",
        data={"proposal_id": pid, "changes": filtered, "reason": reason[:300]},
    )
    _broadcast("config_proposal", {
        "agent_name": caller,
        "proposal_id": pid,
        "changes": filtered,
        "reason": reason[:200],
        "timestamp": time.time(),
    })
    return json.dumps({
        "success": True,
        "proposal_id": pid,
        "status": "pending",
        "message": (
            "Proposal sent to the human operator for approval. It will take effect "
            "ONLY if they approve it in the dashboard. Do not assume it is applied."
        ),
    })


def reload_daemon_crons(agent_name: str) -> None:
    """Re-read an agent's crons from saved config into its running daemon.

    Shared by the schedule_wakeup / cancel_wakeup tools and the REST endpoints so
    a cron change takes effect immediately without re-initialising the AIAgent.
    """
    from swarm_server.config import load_agents_config

    daemon = _daemon_registry.get(agent_name)
    if daemon is None:
        return
    entry = load_agents_config()["agents"].get(agent_name)
    if entry is None:
        return
    with daemon._lock:
        daemon.cfg = entry
        daemon._load_crons(entry)


def _schedule_wakeup_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    schedule = (args.get("schedule") or "").strip()
    instruction = (args.get("instruction") or "").strip()
    from swarm_server.config import load_agents_config, add_agent_cron
    from swarm_server.cron import cron_next, cron_describe

    cfg = load_agents_config()
    if caller not in cfg["agents"]:
        return json.dumps({"success": False, "error": f"Agent '{caller}' not found."})
    try:
        entry = add_agent_cron(cfg, caller, schedule, instruction, created_by="agent")
    except ValueError as e:
        return json.dumps({"success": False, "error": str(e)})

    reload_daemon_crons(caller)
    nxt = None
    try:
        nxt_ts = cron_next(entry["schedule"], time.time())
        if nxt_ts:
            from datetime import datetime
            nxt = datetime.fromtimestamp(nxt_ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    log.info("[%s] [schedule_wakeup] id=%s schedule=%s", caller, entry["id"][:8], entry["schedule"])
    monitor_db.log_event(caller, "cron_created", data={"cron_id": entry["id"], "schedule": entry["schedule"]})
    _broadcast("cron_updated", {"agent_name": caller, "action": "created", "timestamp": time.time()})
    return json.dumps({
        "success": True,
        "cron_id": entry["id"],
        "schedule": entry["schedule"],
        "describe": cron_describe(entry["schedule"]),
        "next_fire_at": nxt,
        "message": f"Wake-up scheduled ({cron_describe(entry['schedule'])}). Next fire: {nxt or 'unknown'}.",
    })


def _cancel_wakeup_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    cron_id = (args.get("cron_id") or "").strip()
    from swarm_server.config import load_agents_config, remove_agent_cron

    cfg = load_agents_config()
    if caller not in cfg["agents"]:
        return json.dumps({"success": False, "error": f"Agent '{caller}' not found."})
    try:
        removed = remove_agent_cron(cfg, caller, cron_id)
    except ValueError as e:
        return json.dumps({"success": False, "error": str(e)})
    if not removed:
        return json.dumps({"success": False, "error": f"No wake-up with id '{cron_id}'."})

    reload_daemon_crons(caller)
    log.info("[%s] [cancel_wakeup] removed id=%s", caller, cron_id[:8])
    monitor_db.log_event(caller, "cron_deleted", data={"cron_id": cron_id})
    _broadcast("cron_updated", {"agent_name": caller, "action": "deleted", "timestamp": time.time()})
    return json.dumps({"success": True, "message": f"Wake-up '{cron_id}' cancelled."})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _pause_agent_handler(args: dict, **kwargs) -> str:
    target_name = (args.get("agent") or args.get("agent_name") or "").strip()
    reason = (args.get("reason") or "").strip()
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    from swarm_server.config import load_agents_config
    cfg = load_agents_config()

    # Gate 1: only a supervisor may wield the brake.
    if not cfg["agents"].get(caller, {}).get("is_supervisor"):
        return json.dumps({"success": False, "error": "Only a supervisor agent can pause another agent."})
    # Gate 2: a real justification is mandatory (deters casual use).
    if len(reason) < 10:
        return json.dumps({"success": False, "error": "A specific reason (>=10 chars) is required — pausing is disruptive; justify the emergency."})
    if not target_name:
        return json.dumps({"success": False, "error": "Missing 'agent' (who to pause)."})
    if target_name == caller:
        return json.dumps({"success": False, "error": "Refusing to pause yourself."})
    # Gate 3: only peers the supervisor actually watches.
    watch = cfg["agents"].get(caller, {}).get("allowed_peers", []) or []
    if target_name not in watch:
        return json.dumps({"success": False, "error": f"'{target_name}' is not in your watch list {watch}."})

    target = _daemon_registry.get(target_name)
    if target is None:
        return json.dumps({"success": False, "error": f"Unknown agent '{target_name}'. Known: {list(_daemon_registry.keys())}"})

    try:
        target.pause_execution(reason=reason, by=caller)
    except Exception as e:
        log.warning("[pause_agent] %s -> %s failed: %s", caller, target_name, e)
        return json.dumps({"success": False, "error": f"pause failed: {e}"})

    log.warning("[pause_agent] %s PAUSED %s: %s", caller, target_name, reason)
    monitor_db.log_event(caller, "agent_paused", to_agent=target_name, data={"reason": reason})
    _broadcast("agent_paused", {
        "by": caller, "agent": target_name, "reason": reason, "timestamp": time.time(),
    })
    # Surface to the human inbox so a person knows an emergency stop happened and
    # can review/resume — a paused agent is frozen until lifted.
    try:
        add_pending_question(
            caller,
            f"[EMERGENCY PAUSE] Supervisor '{caller}' paused '{target_name}'.\n\n"
            f"Reason: {reason}\n\n"
            f"'{target_name}' is frozen (its queued work is preserved). Resume it from the "
            f"dashboard, or tell me to resume it, once it's safe.",
        )
    except Exception as e:
        log.debug("[pause_agent] human notify failed: %s", e)

    return json.dumps({
        "success": True,
        "message": (
            f"'{target_name}' PAUSED — its in-flight turn was interrupted and its queue is held. "
            f"A human has been notified to review. Resume it (resume_agent) only once the danger "
            f"is resolved. Do not pause other agents unless they too are in genuine danger."
        ),
    })


def _resume_agent_handler(args: dict, **kwargs) -> str:
    target_name = (args.get("agent") or args.get("agent_name") or "").strip()
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    from swarm_server.config import load_agents_config
    cfg = load_agents_config()
    if not cfg["agents"].get(caller, {}).get("is_supervisor"):
        return json.dumps({"success": False, "error": "Only a supervisor agent can resume another agent."})
    if not target_name:
        return json.dumps({"success": False, "error": "Missing 'agent' (who to resume)."})

    target = _daemon_registry.get(target_name)
    if target is None:
        return json.dumps({"success": False, "error": f"Unknown agent '{target_name}'. Known: {list(_daemon_registry.keys())}"})
    if not getattr(target, "_paused", False):
        return json.dumps({"success": True, "message": f"'{target_name}' is not paused — nothing to do."})

    try:
        target.resume_execution(by=caller)
    except Exception as e:
        return json.dumps({"success": False, "error": f"resume failed: {e}"})

    log.info("[resume_agent] %s RESUMED %s", caller, target_name)
    monitor_db.log_event(caller, "agent_resumed", to_agent=target_name, data={})
    _broadcast("agent_resumed", {"by": caller, "agent": target_name, "timestamp": time.time()})
    return json.dumps({"success": True, "message": f"'{target_name}' resumed — it will pick up its held queue."})


def _register_custom_tools():
    try:
        from swarm_server.config import ensure_hermes_importable

        ensure_hermes_importable()
        from tools.registry import registry

        if "send_peer_message" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="send_peer_message",
                toolset="custom",
                schema=_SEND_PEER_MESSAGE_TOOL_SCHEMA["function"],
                handler=_send_peer_message_handler,
                description="Send a message to another swarm agent.",
            )
            log.info("[send_peer_message] Registered")
        if "ask_human" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="ask_human",
                toolset="custom",
                schema=_ASK_HUMAN_TOOL_SCHEMA["function"],
                handler=_ask_human_handler,
                description="Ask a human for clarification.",
            )
            log.info("[ask_human] Registered")
        if "request_human_takeover" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="request_human_takeover",
                toolset="custom",
                schema=_REQUEST_HUMAN_TAKEOVER_TOOL_SCHEMA["function"],
                handler=_request_human_takeover_handler,
                description="Hand the live browser to a human for a login/CAPTCHA/verification step.",
            )
            log.info("[request_human_takeover] Registered")
        if "log_changes" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="log_changes",
                toolset="custom",
                schema=_LOG_CHANGES_TOOL_SCHEMA["function"],
                handler=_log_changes_handler,
                description="Log important events to the shared team activity log.",
            )
            log.info("[log_changes] Registered")
        if "get_self_config" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="get_self_config",
                toolset="custom",
                schema=_GET_SELF_CONFIG_TOOL_SCHEMA["function"],
                handler=_get_self_config_handler,
                description="Read your own configuration and runtime telemetry.",
            )
            log.info("[get_self_config] Registered")
        if "request_config_change" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="request_config_change",
                toolset="custom",
                schema=_REQUEST_CONFIG_CHANGE_TOOL_SCHEMA["function"],
                handler=_request_config_change_handler,
                description="Propose a change to your own config (human approves).",
            )
            log.info("[request_config_change] Registered")
        if "schedule_wakeup" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="schedule_wakeup",
                toolset="custom",
                schema=_SCHEDULE_WAKEUP_TOOL_SCHEMA["function"],
                handler=_schedule_wakeup_handler,
                description="Schedule a recurring cron wake-up for yourself.",
            )
            log.info("[schedule_wakeup] Registered")
        if "cancel_wakeup" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="cancel_wakeup",
                toolset="custom",
                schema=_CANCEL_WAKEUP_TOOL_SCHEMA["function"],
                handler=_cancel_wakeup_handler,
                description="Cancel one of your scheduled cron wake-ups by id.",
            )
            log.info("[cancel_wakeup] Registered")
        if "pause_agent" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="pause_agent",
                toolset="custom",
                schema=_PAUSE_AGENT_TOOL_SCHEMA["function"],
                handler=_pause_agent_handler,
                description="Supervisor emergency brake: freeze a watched agent mid-turn.",
            )
            log.info("[pause_agent] Registered")
        if "resume_agent" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="resume_agent",
                toolset="custom",
                schema=_RESUME_AGENT_TOOL_SCHEMA["function"],
                handler=_resume_agent_handler,
                description="Lift a pause on an agent once it's safe to continue.",
            )
            log.info("[resume_agent] Registered")
    except Exception as exc:
        log.warning("[Custom Tools] Could not register in Hermes registry: %s", exc)
