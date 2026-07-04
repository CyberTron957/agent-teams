"""Custom tool schemas, registry, and handlers for P2P communication."""

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List

from teams_server.monitoring import monitor_db
from teams_server.websocket import _broadcast

log = logging.getLogger("teams.tools")

# How long ask_human blocks the worker thread waiting for an in-turn answer.
# On elapse the question stays PENDING (not failed) and the answer is re-delivered
# later as a task, so the agent resumes even if the human was away for hours.
# Default 6h: the human usually answers in-turn (smooth resume) rather than the
# agent ending its turn and resuming later via the re-delivered task.
import os as _os
_ASK_HUMAN_WAIT_SECONDS = int(_os.environ.get("TEAMS_ASK_HUMAN_WAIT_SECONDS", "21600"))

# Maps agent_name -> AgentDaemon instance (populated at runtime by server.py)
_daemon_registry: Dict[str, Any] = {}

# Global Human Inbox Registry — tracks active/ historical human questions
_pending_human_questions: Dict[str, Dict[str, Any]] = {}
_pending_lock = threading.Lock()


def add_pending_question(
    agent_name: str, question: str, *, waiting_in_turn: bool = False, kind: str = "question"
) -> str:
    """Register a new human question and return its ID.

    ``waiting_in_turn`` marks that the agent's worker thread is blocking in-turn on
    this exact question RIGHT NOW — set atomically at registration so there is no
    window where the question is answerable but the live-waiter flag is still
    unset (which would let an answer be both delivered in-turn AND re-queued as a
    task). ``kind`` distinguishes a browser takeover from an ordinary question so
    only real takeovers trigger end_takeover on answer.
    """
    from teams_server.config import MAX_PENDING_QUESTIONS

    # Resolve the asking agent's team so the inbox can show WHICH team a request
    # belongs to — agent names can collide across teams, so the team is what
    # disambiguates them. Best-effort: a missing config/agent leaves it blank.
    team_id = ""
    try:
        from teams_server.config import load_agents_config

        team_id = load_agents_config()["agents"].get(agent_name, {}).get("team_id", "") or ""
    except Exception:
        pass

    qid = str(uuid.uuid4())
    with _pending_lock:
        _pending_human_questions[qid] = {
            "id": qid,
            "agent_name": agent_name,
            "team_id": team_id,
            "question": question,
            "timestamp": time.time(),
            "status": "pending",
            "response": None,
            "waiting_in_turn": waiting_in_turn,
            "kind": kind,
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


def deliver_human_answer(qid: str, response: str) -> Dict[str, Any]:
    """Atomically record a human's answer and decide HOW it reaches the agent.

    This is the single synchronization point for every answer path. Under
    ``_pending_lock`` it marks the question answered and, in the SAME critical
    section, reads whether a live in-turn waiter is parked on this exact question
    (``waiting_in_turn``). That atomicity is what makes the two outcomes mutually
    exclusive — an answer is delivered in-turn XOR re-queued as a task, never both
    and never neither:

      • a live waiter  → wake it in place (set its event); the blocked handler
        re-reads the registry and returns the response. No task is queued.
      • no live waiter → the caller must enqueue a resume task (the agent already
        ended its turn, or this is a stale/old question the agent isn't on).

    Because the decision is per-question, answering an OLD question never feeds its
    answer to an agent currently blocked on a DIFFERENT one, and a pause/stop that
    merely sets the event (without answering) leaves status 'pending' so this never
    misreports a wake as an answer.

    Returns ``{"ok", "delivery": "in_turn"|"task", "agent", "question", "kind"}``
    or ``{"ok": False, "error"}`` when the qid is unknown.
    """
    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if not q:
            return {"ok": False, "error": "question not found"}
        q["status"] = "answered"
        q["response"] = response
        q["answered_at"] = time.time()
        agent_name = q["agent_name"]
        info = {
            "ok": True,
            "agent": agent_name,
            "question": q.get("question", ""),
            "kind": q.get("kind", "question"),
        }
        if q.get("waiting_in_turn"):
            daemon = _daemon_registry.get(agent_name)
            if daemon is not None:
                daemon.human_response = response
                daemon.human_event.set()
                info["delivery"] = "in_turn"
                return info
        info["delivery"] = "task"
        return info


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
    "supervisor_interval_minutes",
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
            "Send a message to a linked peer. The `kind` controls whether it WAKES the "
            "recipient — choose it deliberately, it is how the teams avoids endless "
            "status ping-pong:\n"
            "• TASK — delegate concrete work. WAKES them; they owe you a RESULT. Returns a "
            "task_id.\n"
            "• QUESTION — ask something you need answered to proceed. WAKES them. Returns a "
            "task_id.\n"
            "• RESULT — report a finished deliverable back to whoever delegated it. WAKES "
            "that one peer and CLOSES the task. Pass `reply_to` = the task_id you were given.\n"
            "• STATUS — a progress update. Does NOT wake anyone; it just appears in the "
            "team's recent-messages feed. Use this instead of a TASK when you have nothing "
            "for them to DO.\n"
            "• FYI — informational note. Does NOT wake anyone.\n"
            "NEVER send a TASK/QUESTION just to acknowledge or confirm — that creates a "
            "loop. If you have no concrete ask, use STATUS/FYI (or send nothing)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Name of the target linked peer."},
                "message": {"type": "string", "description": "The message body."},
                "kind": {
                    "type": "string",
                    "enum": ["TASK", "QUESTION", "RESULT", "STATUS", "FYI"],
                    "description": (
                        "Message type. TASK/QUESTION/RESULT wake the recipient; STATUS/FYI "
                        "do not. Defaults to TASK if omitted."
                    ),
                },
                "reply_to": {
                    "type": "string",
                    "description": (
                        "For kind=RESULT only: the task_id of the TASK/QUESTION you are "
                        "answering, so the system can close it."
                    ),
                },
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

_LOG_DECISION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "log_decision",
        "description": (
            "Append ONE significant decision to the shared team decision log. The last "
            "20 entries are auto-injected into every agent's prompt each turn, so this is "
            "how the team stays aware without reading a file. Use SPARINGLY — only for "
            "decisions others must know (e.g. 'Switched checkout to live_mode', 'Verified "
            "signup flow returns 200'). Do NOT log trivial actions, acknowledgements, or "
            "status chatter. Entries are append-only and read-only once written."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "description": (
                        "The decision as a SINGLE line of plain text (no newlines, no "
                        "markdown). Be specific and verifiable."
                    ),
                },
            },
            "required": ["decision"],
        },
    },
}

_RECALL_DECISIONS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_decisions",
        "description": (
            "Search your team's PAST decision log so you can change course instead "
            "of repeating yourself. Your prompt always carries the last ~20 "
            "decisions; use THIS to reach further back or to grep the history of a "
            "strategy by keyword. Use it when your current approach is stalling or "
            "not working — review what was already tried, what was verified, and "
            "what was ruled out, then pivot. Returns matching decisions newest "
            "first with who logged them and when."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Optional keyword/substring to filter decisions (e.g. "
                        "'pricing', 'checkout', 'onboarding'). Omit to get the most "
                        "recent decisions across all topics."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max decisions to return (default 40, max 200).",
                },
            },
            "required": [],
        },
    },
}

_CLOSE_LEDGER_ENTRY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "close_ledger_entry",
        "description": (
            "Permanently close ONE orphaned OPEN delegation in the team ledger — an "
            "entry whose work you have VERIFIED was already delivered or made obsolete "
            "through a different task chain, so no RESULT carrying its id will ever "
            "arrive. Use it ONCE per entry instead of re-analyzing the same stale row "
            "on every check-in. Do NOT use it to dodge work you still owe: closing an "
            "entry asserts the underlying work is done/obsolete, and the closure is "
            "logged with your name and reason. Only the delegator, the delegate, or a "
            "supervisor may close an entry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "msg_id": {
                    "type": "string",
                    "description": (
                        "The delegation id exactly as shown in the ledger (the 8-char "
                        "prefix from 'OPEN [xxxxxxxx]' or 'id=...' lines is enough)."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "ONE line: how you verified the work is complete/obsolete "
                        "(e.g. 'delivered via task 9d2c41ab, RESULT logged 06:14 Jun 11')."
                    ),
                },
            },
            "required": ["msg_id", "reason"],
        },
    },
}

_READ_FILES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_files",
        "description": (
            "Read SEVERAL files in ONE call. Strongly preferred over a chain of "
            "single read_file calls: every extra tool round trip re-sends your "
            "entire conversation to the model, so reading 5 files one-by-one "
            "costs ~5x what this does. Use whenever you already know 2+ paths "
            "you need (a module plus its template, a config plus the code that "
            "reads it, …). Returns each file under a '=== path ===' header; "
            "missing files report their error inline without failing the rest."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "2-8 file paths (relative to your workspace, or absolute)."
                    ),
                },
            },
            "required": ["paths"],
        },
    },
}

_READ_FILES_MAX = 8
_READ_FILES_CHARS_EACH = 24_000


def _read_files_handler(args: dict, **kwargs) -> str:
    from pathlib import Path

    from teams_server.config import _derive_workspace_path, load_agents_config

    caller = _caller_from_kwargs(kwargs)
    paths = args.get("paths") or []
    if not isinstance(paths, list) or not paths:
        return json.dumps({"error": "paths must be a non-empty array of strings."})
    if len(paths) > _READ_FILES_MAX:
        return json.dumps({"error": f"At most {_READ_FILES_MAX} files per call "
                                    f"(got {len(paths)}). Split the batch."})
    cfg = load_agents_config()
    team_id = cfg["agents"].get(caller, {}).get("team_id", "default")
    try:
        ws = Path(_derive_workspace_path(team_id, caller))
    except Exception:
        ws = Path(".")

    chunks = []
    for raw in paths:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = ws / p
        try:
            text = p.read_text(errors="replace")
            note = ""
            if len(text) > _READ_FILES_CHARS_EACH:
                text = text[:_READ_FILES_CHARS_EACH]
                note = f"\n…[truncated at {_READ_FILES_CHARS_EACH} chars — use read_file for the rest]"
            chunks.append(f"=== {raw} ({len(text)} chars) ===\n{text}{note}")
        except Exception as e:
            chunks.append(f"=== {raw} ===\n[unreadable: {e}]")
    return "\n\n".join(chunks)


_LOG_ACTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "log_action",
        "description": (
            "Claim/record a side-effecting EXTERNAL action (sending an email, publishing "
            "a post, deploying, taking a payment) under a stable idempotency_key so two "
            "agents can't double-do it. Call it RIGHT BEFORE you perform the action: if it "
            "returns duplicate=true, someone already did this exact thing — DO NOT repeat "
            "it; use their recorded outcome. If duplicate=false you hold the claim — do the "
            "action, then optionally call again with the same key plus the outcome/verified "
            "proof. The idempotency_key must uniquely identify the action (e.g. "
            "'email:jane@acme.com:launch-invite', 'publish:linkedin:2026-06-08-post')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": "Kind of action, e.g. email_send, publish, deploy, payment.",
                },
                "idempotency_key": {
                    "type": "string",
                    "description": "Stable unique id for THIS action so it's never done twice.",
                },
                "target": {
                    "type": "string",
                    "description": "What it acts on (recipient, URL, service). Optional.",
                },
                "outcome": {
                    "type": "string",
                    "description": "Verifiable result/proof once done (message-id, live URL). Optional.",
                },
                "verified": {
                    "type": "boolean",
                    "description": "True only if you hold machine-verifiable proof it succeeded.",
                },
            },
            "required": ["action_type", "idempotency_key"],
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
            "Schedule a future wake-up for YOURSELF — recurring OR one-time. At each "
            "scheduled time the teams injects your 'instruction' to you as a new task "
            "and you act on it. Use it for anything periodic (a 9am competitor check, "
            "an hourly metrics pull, a Monday digest) so it happens on time without a "
            "human. IMPORTANT: for a task that should run only ONCE at a future time, "
            "set max_runs=1 — otherwise the schedule recurs forever and keeps pulling "
            "you back to it instead of letting you move on to new work. Keep the "
            "instruction GOAL-LEVEL and short (max 600 chars): what to achieve, where "
            "the data lives, when to escalate. Do NOT inline a step-by-step script — "
            "the world changes and a frozen script rots; put detailed procedures in a "
            "workspace runbook file (e.g. docs/<name>-runbook.md) and reference its "
            "path so each firing follows the file's LATEST version. Check your current "
            "wake-ups (listed in the live team context) first so you don't duplicate."
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
                    "description": (
                        "Goal-level task to run each time it fires (max 600 chars). "
                        "Reference a workspace runbook file for detailed steps."
                    ),
                },
                "max_runs": {
                    "type": "integer",
                    "description": (
                        "Optional. Total times to fire before the wake-up auto-stops. "
                        "Set max_runs=1 for a ONE-TIME future task. Omit for an "
                        "indefinitely recurring schedule."
                    ),
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


# Teams-native tools (registered on every agent AFTER Hermes init, so they
# survive any toolset whitelist). `send_peer_message` is REQUIRED (agents end
# their turn with it) and cannot be disabled; the rest are optional and can be
# turned off per agent via `disabled_toolsets`. pause_agent/resume_agent are
# registered ONLY on supervisor agents (see agent.py).
REQUIRED_TEAMS_TOOLS = ("send_peer_message",)
_TEAMS_TOOL_SCHEMAS = (
    _SEND_PEER_MESSAGE_TOOL_SCHEMA,
    _ASK_HUMAN_TOOL_SCHEMA,
    _LOG_DECISION_TOOL_SCHEMA,
    _LOG_ACTION_TOOL_SCHEMA,
    _CLOSE_LEDGER_ENTRY_TOOL_SCHEMA,
    _READ_FILES_TOOL_SCHEMA,
    _GET_SELF_CONFIG_TOOL_SCHEMA,
    _REQUEST_CONFIG_CHANGE_TOOL_SCHEMA,
    _SCHEDULE_WAKEUP_TOOL_SCHEMA,
    _CANCEL_WAKEUP_TOOL_SCHEMA,
    _PAUSE_AGENT_TOOL_SCHEMA,
    _RESUME_AGENT_TOOL_SCHEMA,
)


def list_teams_tools():
    """The teams's own tools (not Hermes toolsets) as [{name, description, required}]
    for the dashboard's enabled/disabled-toolsets picker."""
    out = []
    for s in _TEAMS_TOOL_SCHEMAS:
        fn = (s or {}).get("function", {}) if isinstance(s, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": str(fn.get("description") or "")[:160],
            "required": name in REQUIRED_TEAMS_TOOLS,
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

    from teams_server.config import load_agents_config, peer_allowed

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

    import hashlib as _hashlib
    import time as _time

    # Typed messages: kind decides whether the recipient is WOKEN. STATUS/FYI are
    # passive (they surface in the recipient's recent-messages feed but create no
    # task), which is the structural cure for the acknowledge/status ping-pong —
    # an agent literally cannot wake a peer just to confirm a status.
    kind = (args.get("kind") or "TASK").strip().upper()
    if kind not in ("TASK", "QUESTION", "RESULT", "STATUS", "FYI"):
        kind = "TASK"
    reply_to = (args.get("reply_to") or "").strip()
    team_id = cfg["agents"].get(caller, {}).get("team_id")
    waking = kind in ("TASK", "QUESTION", "RESULT")
    # DETERMINISTIC correlation id: derive it from the message identity, NOT a
    # fresh uuid. A random id was embedded in the header, so two identical TASK
    # sends produced different payloads and slipped past the queue's byte-identical
    # pending-dedup — waking the recipient twice and opening two delegations of
    # which only one ever closed (a phantom "overdue" forever). With a content
    # hash, a duplicate send is byte-identical → deduped to one wake/one delegation.
    msg_id = _hashlib.sha1(
        f"{caller}|{to_agent}|{kind}|{reply_to}|{message}".encode("utf-8")
    ).hexdigest()[:8]

    task_id = None
    if waking:
        # Embed a correlation header so the recipient can reference this thread in
        # its RESULT (the from_agent is already shown by the batch builder).
        if kind in ("TASK", "QUESTION"):
            header = (
                f"[{kind} · id={msg_id} · from {caller} — when done, reply with "
                f"send_peer_message(to_agent=\"{caller}\", kind=\"RESULT\", "
                f"reply_to=\"{msg_id}\")]\n"
            )
        else:  # RESULT
            header = f"[RESULT · from {caller}" + (f" · re {reply_to}" if reply_to else "") + "]\n"
        task_id = target.ingest_task(from_agent=caller, payload=header + message)
        if kind in ("TASK", "QUESTION"):
            monitor_db.open_delegation(msg_id, caller, to_agent, kind,
                                       summary=message[:160], team_id=team_id)
        elif kind == "RESULT" and reply_to:
            monitor_db.answer_delegation(reply_to, by_agent=caller)
        log.info("[send_peer_message] %s -%s-> %s | id=%s task=%s",
                 caller, kind, to_agent, msg_id, (task_id or "")[:8])
    else:
        # STATUS / FYI — recorded for awareness, but NOT enqueued and NOT woken.
        log.info("[send_peer_message] %s -%s-> %s | id=%s (passive, no wake)",
                 caller, kind, to_agent, msg_id)

    # Persist the peer message (not just broadcast it) so historical/REST queries
    # can reconstruct the conversation graph, not only the live dashboard.
    # Passive kinds also carry the FULL body (capped): they are never enqueued,
    # so this event is the only place the recipient's daemon can read the text
    # from when it delivers the passive backlog into the next turn.
    _evt_data = {"message_preview": message[:120], "kind": kind, "msg_id": msg_id,
                 "waking": waking}
    if not waking:
        _evt_data["message_full"] = message[:2000]
    monitor_db.log_event(
        caller, "message_sent",
        from_agent=caller, to_agent=to_agent, task_id=task_id,
        data=_evt_data,
    )

    _broadcast("message_sent", {
        "from_agent": caller,
        "to_agent": to_agent,
        "task_id": task_id,
        "kind": kind,
        "waking": waking,
        "message_preview": message[:120],
        "timestamp": _time.time(),
    })

    if not waking:
        return json.dumps({
            "success": True,
            "kind": kind,
            "delivered": "passive",
            "message": (
                f"{kind} recorded for '{to_agent}' (passive — it was NOT woken and owes "
                f"no reply; the full text will be shown to it at the start of its next "
                f"turn). Do not follow up with a TASK just to make sure it was seen."
            ),
        })

    return json.dumps({
        "success": True,
        "kind": kind,
        "task_id": task_id,
        "msg_id": msg_id,
        "message": f"{kind} enqueued to '{to_agent}' successfully.",
    })


def _block_on_human(daemon, caller: str, prompt: str, *, kind: str = "question"):
    """Register a human question, block the worker thread until it's answered or
    the wait elapses, and report the outcome as ``(qid, answered, response)``.

    Shared by ask_human and the browser takeover so the subtle correctness
    ordering lives in ONE place:

      • the wake channel is ARMED (event cleared, response nulled) BEFORE the
        question is exposed, and the question is registered with
        waiting_in_turn=True atomically — so an answer landing the instant it's
        answerable is never lost nor double-delivered (see deliver_human_answer);
      • on wake, the answered/not decision is read from the REGISTRY status, not
        the event flag — pause_execution/stop_execution also set human_event to
        break the wait, and that must NOT be mistaken for a human answer.

    On timeout/pause/stop the question is LEFT pending, so a later answer is
    re-delivered as a task by the REST endpoint.
    """
    from teams_server.agent import AGENT_STATE_ASKING_HUMAN, AGENT_STATE_BUSY

    # Arm the wake channel before the question becomes answerable.
    daemon.human_event.clear()
    daemon.human_response = None
    with daemon._lock:
        daemon.state = AGENT_STATE_ASKING_HUMAN
    qid = add_pending_question(caller, prompt, waiting_in_turn=True, kind=kind)
    daemon.human_question_id = qid

    monitor_db.log_event(caller, "human_waiting",
                         data={"question": prompt, "question_id": qid, "kind": kind})
    _broadcast("human_waiting", {
        "agent_name": caller, "question": prompt, "question_id": qid,
        "kind": kind, "timestamp": time.time(),
    })

    # Mirror the question out to a configured messaging gateway (Telegram/…), so
    # the human can be notified and reply from chat without the dashboard open.
    # Fire-and-forget on its own thread and fully guarded — the gateway must never
    # delay or break the ask path (the question already lives in the inbox).
    try:
        from teams_server.gateway_bridge import forward_question
        forward_question(qid, caller, prompt, kind)
    except Exception as e:
        log.debug("[%s] gateway forward skipped: %s", caller, e)

    # Block the worker thread waiting for an in-turn answer. If it elapses we DON'T
    # fail — the question stays pending and a late answer is re-delivered as a task.
    daemon.human_event.wait(timeout=_ASK_HUMAN_WAIT_SECONDS)

    with daemon._lock:
        daemon.state = AGENT_STATE_BUSY

    answered = False
    response = None
    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if q:
            # Clear the live-waiter flag in the SAME critical section we read the
            # status, so a concurrent deliver_human_answer either delivered before
            # this (status=='answered' here) or now takes the task path.
            q["waiting_in_turn"] = False
            if q["status"] == "answered":
                answered = True
                response = q["response"]
        # On timeout we deliberately LEAVE the question 'pending' (do NOT mark it
        # timed_out): it stays open in the human's inbox, and a later answer is
        # re-delivered as a new task — this is what makes a 24/7 teams survive a
        # human who is away for hours.
    if answered:
        daemon.human_response = response
    return qid, answered, response


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

    qid, answered, response = _block_on_human(daemon, caller, question, kind="question")

    if not answered:
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

    log.info("[%s] [ask_human] Response received: %s", daemon.name, response)
    monitor_db.log_event(caller, "human_responded", data={"question": question, "response": response})
    _broadcast("human_responded", {
        "agent_name": caller,
        "question": question,
        "response": response,
        "question_id": qid,
        "timestamp": time.time(),
    })

    return json.dumps({"success": True, "response": response})


def _request_human_takeover_handler(args: dict, **kwargs) -> str:
    """Open the team browser as a real window on the human's screen, block until
    they finish, then relaunch it headless and resume. Mirrors ask_human's blocking
    but brackets it with begin_takeover/end_takeover so the handoff is seamless."""
    reason = args.get("reason", "")
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    daemon = _daemon_registry.get(caller)
    if daemon is None:
        return json.dumps({"error": f"Caller agent '{caller}' not registered."})
    team_id = (daemon.cfg or {}).get("team_id", "default")

    from teams_server.config import TEAMS_TAKEOVER_MODE
    embedded = TEAMS_TAKEOVER_MODE != "window"

    shown = False
    if embedded:
        # Keep the team browser HEADLESS; the human drives it through the live
        # view embedded in the dashboard. Just make sure a browser exists so the
        # stream has something to attach to.
        try:
            from teams_server.browser_pool import team_browser_manager
            shown = bool(team_browser_manager.ensure_team_browser(team_id))
        except Exception as e:
            log.error("[%s] [takeover] ensure_team_browser failed: %s", caller, e)
        access = (
            "\n\n👉 Open the **Browser** panel in the dashboard (the takeover item "
            "in your inbox has an 'Open browser' button) to see and control the "
            "page that needs you, then complete the step there."
            if shown else
            "\n\n⚠️ No browser is available on this host (install Chrome, or run "
            "`npx playwright install chromium`)."
        )
    else:
        # Legacy: bring the team browser onto the host's screen as a real,
        # visible Chrome window, opened on the page the agent was blocked on.
        try:
            from teams_server.browser_pool import team_browser_manager
            shown = team_browser_manager.begin_takeover(team_id)
        except Exception as e:
            log.error("[%s] [takeover] begin_takeover failed: %s", caller, e)
        access = (
            "\n\n👉 A Chrome window has just opened on your screen, on the page "
            "that needs you. Complete the step there."
            if shown else
            "\n\n⚠️ Couldn't open the browser window automatically (no display on "
            "this host?). The browser session is the team's persistent profile."
        )
    prompt = (
        "🙋 BROWSER TAKEOVER requested by '" + caller + "':\n" + reason + access +
        "\n\nWhen you're done, reply 'done' here to hand control back to the agent."
    )
    log.info("[%s] [takeover] %s | window_shown=%s", daemon.name, reason, shown)

    qid, answered, response = _block_on_human(daemon, caller, prompt, kind="takeover")

    if not answered:
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
        from teams_server.browser_pool import team_browser_manager
        team_browser_manager.end_takeover(team_id)
    except Exception as e:
        log.error("[%s] [takeover] end_takeover failed: %s", caller, e)
    monitor_db.log_event(caller, "human_responded",
                         data={"question": prompt, "response": response, "kind": "takeover"})
    _broadcast("human_responded", {
        "agent_name": caller, "response": response,
        "question_id": qid, "kind": "takeover", "timestamp": time.time(),
    })
    return json.dumps({
        "success": True, "status": "completed", "response": response,
        "message": (
            "Human finished the manual step; the browser session is now "
            "authenticated/unblocked. Continue your task from here."
        ),
    })


def _log_decision_handler(args: dict, **kwargs) -> str:
    import time

    decision = args.get("decision", "") or args.get("entry", "")  # accept legacy key
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]

    # Collapse to a single line — the decision log is a scannable one-liner feed.
    one_line = " ".join((decision or "").split()).strip()
    if not one_line:
        return json.dumps({"success": False, "error": "Empty decision."})

    from teams_server.config import load_agents_config

    cfg = load_agents_config()
    team_id = cfg["agents"].get(caller, {}).get("team_id", "default")

    # Single destination: the team-scoped decision table. The last 20 entries are
    # injected into every agent's prompt each turn (compose_live_context), so there
    # is no file to write — this replaces the old agent_log.md / log_changes trail.
    monitor_db.log_decision(caller, one_line, team_id=team_id)
    log.info("[log_decision] %s: %s", caller, one_line[:120])
    _broadcast("decision_logged", {
        "agent_name": caller,
        "team_id": team_id,
        "decision": one_line,
        "timestamp": time.time(),
    })

    return json.dumps({"success": True, "message": "Decision recorded."})


def _recall_decisions_handler(args: dict, **kwargs) -> str:
    from datetime import datetime

    caller = _caller_from_kwargs(kwargs)
    query = (args.get("query") or "").strip() or None
    try:
        limit = int(args.get("limit") or 40)
    except (TypeError, ValueError):
        limit = 40

    from teams_server.config import load_agents_config

    cfg = load_agents_config()
    team_id = cfg["agents"].get(caller, {}).get("team_id", "default")

    rows = monitor_db.search_decisions(team_id=team_id, query=query, limit=limit)
    # Surface the rolled-up milestone too, so "further back" really reaches the
    # long-term memory that has scrolled out of the live window.
    milestone = None
    try:
        ms = monitor_db.get_latest_milestone(team_id)
        if ms and ms.get("summary"):
            milestone = ms["summary"]
    except Exception:
        pass

    out = []
    for r in rows:
        try:
            stamp = datetime.fromtimestamp(r.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            stamp = "?"
        out.append({"when": stamp, "agent": r.get("agent_name"),
                    "decision": r.get("decision")})

    return json.dumps({
        "success": True,
        "count": len(out),
        "query": query,
        "milestone": milestone,
        "decisions": out,
        "message": (f"{len(out)} decision(s)"
                    + (f" matching '{query}'" if query else "")
                    + (" (none found — nothing logged yet on this)" if not out else "")),
    })


def _close_ledger_entry_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    msg_id = (args.get("msg_id") or "").strip()
    reason = " ".join((args.get("reason") or "").split()).strip()
    if not reason:
        return json.dumps({"success": False,
                           "error": "A one-line verification reason is required."})

    from teams_server.config import load_agents_config

    cfg = load_agents_config()
    agent_cfg = cfg["agents"].get(caller, {})
    team_id = agent_cfg.get("team_id", "default")
    is_supervisor = bool(agent_cfg.get("is_supervisor"))

    result = monitor_db.close_delegation_manual(
        msg_id, by_agent=caller, reason=reason, team_id=team_id,
        require_participant=not is_supervisor)
    if result.get("success"):
        log.info("[close_ledger_entry] %s closed %s: %s",
                 caller, result.get("msg_id", "?")[:8], reason[:120])
    return json.dumps(result)


def _log_action_handler(args: dict, **kwargs) -> str:
    import time

    action_type = (args.get("action_type") or "").strip()
    key = (args.get("idempotency_key") or "").strip()
    target = (args.get("target") or "").strip()
    outcome = (args.get("outcome") or "").strip()
    verified = bool(args.get("verified"))
    task_id_arg = kwargs.get("task_id", "")
    caller = "unknown"
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        caller = task_id_arg.split(":", 1)[1]
    if not action_type or not key:
        return json.dumps({"success": False, "error": "action_type and idempotency_key are required."})

    from teams_server.config import load_agents_config
    cfg = load_agents_config()
    team_id = cfg["agents"].get(caller, {}).get("team_id", "default")

    res = monitor_db.record_action(caller, action_type, key, target=target,
                                   outcome=outcome, verified=verified, team_id=team_id)
    if res.get("duplicate"):
        ex = res.get("existing") or {}
        log.info("[log_action] %s DUP key=%s (already by %s)", caller, key, ex.get("agent_name"))
        return json.dumps({
            "success": True, "duplicate": True,
            "message": (
                f"ALREADY DONE — '{action_type}' with key '{key}' was recorded by "
                f"{ex.get('agent_name')} (outcome: {ex.get('outcome') or 'n/a'}). "
                f"Do NOT repeat it."
            ),
            "existing": {"agent_name": ex.get("agent_name"), "outcome": ex.get("outcome"),
                         "verified": bool(ex.get("verified"))},
        })
    log.info("[log_action] %s recorded %s key=%s verified=%s", caller, action_type, key, verified)
    _broadcast("action_logged", {
        "agent_name": caller, "team_id": team_id, "action_type": action_type,
        "target": target, "verified": verified, "timestamp": time.time(),
    })
    return json.dumps({
        "success": True, "duplicate": False,
        "message": f"Claim recorded for '{action_type}' (key '{key}'). You may proceed.",
    })


def _caller_from_kwargs(kwargs: dict) -> str:
    task_id_arg = kwargs.get("task_id", "")
    if task_id_arg and task_id_arg.startswith("agent_name:"):
        return task_id_arg.split(":", 1)[1]
    return "unknown"


def _get_self_config_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    from teams_server.config import load_agents_config
    from teams_server.model_config import resolve_model

    cfg = load_agents_config()
    a = cfg["agents"].get(caller)
    if a is None:
        return json.dumps({"success": False, "error": f"Agent '{caller}' not found."})
    editable = {k: a.get(k) for k in SELF_CONFIG_ALLOWED_KEYS}
    # Show the effective model: explicit per-agent override, else the resolved
    # default (teams default / ~/.hermes), else empty (unconfigured).
    editable["model"] = editable.get("model") or resolve_model(a).get("model") or ""
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
    from teams_server.config import load_agents_config

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
    from teams_server.config import load_agents_config, add_agent_cron
    from teams_server.cron import cron_next, cron_describe

    cfg = load_agents_config()
    if caller not in cfg["agents"]:
        return json.dumps({"success": False, "error": f"Agent '{caller}' not found."})
    max_runs = args.get("max_runs")
    if max_runs in ("", None):
        max_runs = None
    # Snapshot existing cron ids so we can tell whether add_agent_cron actually
    # created one or collapsed a duplicate (it returns the pre-existing entry).
    _prior_ids = {c.get("id") for c in cfg["agents"][caller].get("crons", [])}
    try:
        entry = add_agent_cron(cfg, caller, schedule, instruction,
                               created_by="agent", max_runs=max_runs)
    except ValueError as e:
        return json.dumps({"success": False, "error": str(e)})

    # Idempotent re-schedule: the agent already has this exact wake-up active.
    # Tell it so plainly (don't log a phantom cron_created) so it stops re-issuing
    # and moves on instead of stacking duplicates.
    if entry.get("id") in _prior_ids:
        return json.dumps({
            "success": True,
            "deduplicated": True,
            "cron_id": entry["id"],
            "schedule": entry["schedule"],
            "max_runs": entry.get("max_runs"),
            "message": (
                f"You already have an active wake-up for this schedule "
                f"({cron_describe(entry['schedule'])}) — kept the existing one, "
                "did not create a duplicate. No further action needed."
            ),
        })

    reload_daemon_crons(caller)
    nxt = None
    try:
        nxt_ts = cron_next(entry["schedule"], time.time())
        if nxt_ts:
            from datetime import datetime
            nxt = datetime.fromtimestamp(nxt_ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    log.info("[%s] [schedule_wakeup] id=%s schedule=%s max_runs=%s",
             caller, entry["id"][:8], entry["schedule"], entry.get("max_runs"))
    monitor_db.log_event(caller, "cron_created", data={"cron_id": entry["id"],
                         "schedule": entry["schedule"], "max_runs": entry.get("max_runs")})
    _broadcast("cron_updated", {"agent_name": caller, "action": "created", "timestamp": time.time()})
    mr = entry.get("max_runs")
    bound = (" Fires once, then auto-stops." if mr == 1
             else f" Fires {mr}× then auto-stops." if mr else " Recurring.")
    return json.dumps({
        "success": True,
        "cron_id": entry["id"],
        "schedule": entry["schedule"],
        "describe": cron_describe(entry["schedule"]),
        "max_runs": mr,
        "next_fire_at": nxt,
        "message": f"Wake-up scheduled ({cron_describe(entry['schedule'])}).{bound} Next fire: {nxt or 'unknown'}.",
    })


def _cancel_wakeup_handler(args: dict, **kwargs) -> str:
    caller = _caller_from_kwargs(kwargs)
    cron_id = (args.get("cron_id") or "").strip()
    from teams_server.config import load_agents_config, remove_agent_cron

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

    from teams_server.config import load_agents_config
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

    from teams_server.config import load_agents_config
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
        from teams_server.config import ensure_hermes_importable

        ensure_hermes_importable()
        from tools.registry import registry

        if "send_peer_message" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="send_peer_message",
                toolset="custom",
                schema=_SEND_PEER_MESSAGE_TOOL_SCHEMA["function"],
                handler=_send_peer_message_handler,
                description="Send a message to another teams agent.",
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
        if "log_decision" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="log_decision",
                toolset="custom",
                schema=_LOG_DECISION_TOOL_SCHEMA["function"],
                handler=_log_decision_handler,
                description="Append one significant decision to the shared decision log.",
            )
            log.info("[log_decision] Registered")
        if "recall_decisions" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="recall_decisions",
                toolset="custom",
                schema=_RECALL_DECISIONS_TOOL_SCHEMA["function"],
                handler=_recall_decisions_handler,
                description="Search the team's past decision log to review and pivot strategy.",
            )
            log.info("[recall_decisions] Registered")
        if "close_ledger_entry" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="close_ledger_entry",
                toolset="custom",
                schema=_CLOSE_LEDGER_ENTRY_TOOL_SCHEMA["function"],
                handler=_close_ledger_entry_handler,
                description="Close one verified-complete/obsolete OPEN ledger entry.",
            )
            log.info("[close_ledger_entry] Registered")
        if "read_files" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="read_files",
                toolset="custom",
                schema=_READ_FILES_TOOL_SCHEMA["function"],
                handler=_read_files_handler,
                description="Read several files in one call (batch read_file).",
            )
            log.info("[read_files] Registered")
        if "log_action" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="log_action",
                toolset="custom",
                schema=_LOG_ACTION_TOOL_SCHEMA["function"],
                handler=_log_action_handler,
                description="Claim/record a side-effecting action with an idempotency key.",
            )
            log.info("[log_action] Registered")
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

        # Per-team credentials registry — secrets live OUTSIDE the prompt
        # stream, each under a site key with an explicit purpose, so agents
        # stop reusing e.g. an SMTP app password as a website login.
        try:
            from teams_server.credentials import (
                GET_CREDENTIAL_TOOL_SCHEMA, LIST_CREDENTIALS_TOOL_SCHEMA,
                get_credential_handler, list_credentials_handler,
            )

            if "get_credential" not in (registry.get_tool_to_toolset_map() or {}):
                registry.register(
                    name="get_credential",
                    toolset="custom",
                    schema=GET_CREDENTIAL_TOOL_SCHEMA["function"],
                    handler=get_credential_handler,
                    description="Fetch one stored team credential by site key.",
                )
                log.info("[get_credential] Registered")
            if "list_credentials" not in (registry.get_tool_to_toolset_map() or {}):
                registry.register(
                    name="list_credentials",
                    toolset="custom",
                    schema=LIST_CREDENTIALS_TOOL_SCHEMA["function"],
                    handler=list_credentials_handler,
                    description="List stored team credentials (no secrets).",
                )
                log.info("[list_credentials] Registered")
        except Exception as exc:  # noqa: BLE001
            log.warning("[Custom Tools] credentials tools skipped: %s", exc)

        # GUI-grade browser tools (real keystrokes, hover, drag, coordinate
        # click, screenshot, vision-grounded locate) — the agent-browser CLI
        # commands Hermes doesn't expose. Schemas are appended per-agent in
        # agent.py (never for supervisors, only when browser is active).
        try:
            from teams_server.browser_gui_tools import register_gui_browser_tools

            register_gui_browser_tools(registry)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Custom Tools] GUI browser tools skipped: %s", exc)

        # Override built-in web_search / web_extract with crawl4ai-backed
        # handlers (ddgs / httpx fallback) so every agent uses crawl4ai as the
        # primary web search + fetch engine. Reuses the existing schemas.
        try:
            from teams_server.web_crawl4ai import install_crawl4ai_web_tools

            install_crawl4ai_web_tools(registry)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Custom Tools] crawl4ai web override skipped: %s", exc)
    except Exception as exc:
        log.warning("[Custom Tools] Could not register in Hermes registry: %s", exc)
