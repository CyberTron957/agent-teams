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
    daemon.human_event.wait(timeout=300)  # 5 min timeout

    with daemon._lock:
        daemon.state = AGENT_STATE_BUSY

    # Check if a response was provided via the API
    with _pending_lock:
        q = _pending_human_questions.get(qid)
        if q and q["status"] == "answered":
            daemon.human_response = q["response"]
        elif q and q["status"] == "pending":
            q["status"] = "timed_out"

    if not daemon.human_event.is_set():
        log.warning("[%s] [ask_human] Timeout — no human response in 300s", daemon.name)
        # NOTE: deliberately do NOT set state to "idle" here. This handler runs
        # *inside* run_conversation on the agent's worker thread, and that
        # conversation is still in flight (the LLM receives this error and keeps
        # going). State was restored to "busy" just above; leaving it that way
        # prevents the sweep loop from launching a SECOND concurrent
        # run_conversation on the same agent. The sweep loop's finally-block is
        # the single owner of the busy→idle transition and resets it once the
        # run actually returns.
        return json.dumps({
            "success": False,
            "error": "No human responded within 5 minutes. Proceed with your best judgment or retry later.",
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
    from swarm_server.config import load_agents_config, _get_team_workspace_path

    cfg = load_agents_config()
    caller_cfg = cfg["agents"].get(caller, {})
    team_id = caller_cfg.get("team_id", "default")

    # Append to team agent_log.md
    team_ws = _get_team_workspace_path(team_id)
    agent_log = team_ws / "agent_log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"# [{timestamp}] {caller}: {entry}\n\n"

    try:
        if agent_log.exists():
            with open(agent_log, "a", encoding="utf-8") as f:
                f.write(log_line)
        else:
            with open(agent_log, "w", encoding="utf-8") as f:
                f.write("# Team Activity Log\n\n")
                f.write(log_line)
        log.info("[log_changes] %s logged: %s", caller, entry[:80])
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _register_custom_tools():
    try:
        import sys

        sys.path.insert(0, "/Users/pradhyun/.hermes/hermes-agent")
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
        if "log_changes" not in (registry.get_tool_to_toolset_map() or {}):
            registry.register(
                name="log_changes",
                toolset="custom",
                schema=_LOG_CHANGES_TOOL_SCHEMA["function"],
                handler=_log_changes_handler,
                description="Log important events to the shared team activity log.",
            )
            log.info("[log_changes] Registered")
    except Exception as exc:
        log.warning("[Custom Tools] Could not register in Hermes registry: %s", exc)
