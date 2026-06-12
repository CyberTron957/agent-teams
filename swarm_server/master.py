"""Master "Architect" agent — a teamless helper that designs and builds swarm teams.

The Architect is NOT a member of any team and NOT a daemon: it has no sweeps,
heartbeats, crons, or monitoring. It is a single Hermes ``AIAgent`` the human
chats with from the dashboard. Its only powers are the ``swarm_master`` tools —
inspect the swarm, create teams/agents, write souls, wire links, seed workspace
files, kick off the team, and (with explicit confirmation) tear things down.

It runs on its own isolated ``HERMES_HOME`` (``DATA_ROOT/master/.hermes``) so its
conversation never mixes with team agents, and it talks to the human directly
through its chat replies (no ``send_peer_message`` / ``ask_human``).

Design notes:
  * The master tools are registered in the shared Hermes registry under the
    toolset ``swarm_master``. Team agents are explicitly denied this toolset (see
    ``AgentDaemon._ensure_agent``), and every handler ALSO verifies the caller is
    "master" — defense in depth against the schema leaking into a team agent.
  * Daemon spawn/despawn/update must run on the server's event loop, but the
    master turn runs on its own worker thread. The server injects thread-safe
    hooks via ``set_master_hooks`` at startup; master.py never imports server.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("swarm.master")

MASTER_NAME = "master"
MASTER_TASK_ID = f"agent_name:{MASTER_NAME}"
_MASTER_MAX_ITERATIONS = 40
# Cap on how much file content the master reads/writes in one tool call, so a
# runaway read can't blow the turn's context.
_MAX_FILE_CHARS = 60_000
_LIST_FILES_MAX = 400
_LIST_FILES_MAX_DEPTH = 4
_HISTORY_DEFAULT = 120


# ---------------------------------------------------------------------------
# Daemon lifecycle hooks (injected by server at startup; thread-safe wrappers
# around register_agent_daemon / delete-drain / _update_daemon_cfg).
# ---------------------------------------------------------------------------
_HOOKS: Dict[str, Any] = {"spawn": None, "despawn": None, "update": None}


def set_master_hooks(spawn=None, despawn=None, update=None) -> None:
    """Wire the daemon lifecycle callbacks. Each takes a single agent_name (or,
    for despawn, an agent_name) and is safe to call from the master worker
    thread — the server's wrappers marshal onto the event loop."""
    if spawn is not None:
        _HOOKS["spawn"] = spawn
    if despawn is not None:
        _HOOKS["despawn"] = despawn
    if update is not None:
        _HOOKS["update"] = update


def _hook(name: str, *args) -> None:
    fn = _HOOKS.get(name)
    if fn is None:
        log.debug("[master] hook '%s' not wired — skipping", name)
        return
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001 — a hook failure must not break the tool
        log.warning("[master] hook '%s' failed: %s", name, e)


_toolsets_cache: Optional[str] = None


def _available_toolsets_str() -> str:
    """Comma-joined names of the Hermes toolsets an agent can be given, so the
    Architect can set enabled/disabled_toolsets knowledgeably. Cached (the set is
    static for a deployment); swarm_master is hidden (Architect-only)."""
    global _toolsets_cache
    if _toolsets_cache is not None:
        return _toolsets_cache
    try:
        from swarm_server.config import ensure_hermes_importable

        ensure_hermes_importable()
        from tools.registry import registry

        names = sorted(n for n in (registry.get_registered_toolset_names() or [])
                       if n and n != "swarm_master")
        _toolsets_cache = ", ".join(names) if names else "(unavailable)"
    except Exception:  # noqa: BLE001
        _toolsets_cache = "(unavailable)"
    return _toolsets_cache


def _broadcast(event_type: str, payload: dict) -> None:
    try:
        from swarm_server.websocket import _broadcast as ws_broadcast

        ws_broadcast(event_type, payload)
    except Exception as e:  # noqa: BLE001
        log.debug("[master] broadcast %s failed: %s", event_type, e)


# ---------------------------------------------------------------------------
# Framework manual + behavior contract (the Architect's identity / SOUL.md)
# ---------------------------------------------------------------------------
MASTER_SOUL = """\
You are the ARCHITECT — the master builder of this multi-agent swarm framework.
You are NOT part of any team. You are a helper assistant to the human operator:
you understand the framework completely and you BUILD agent teams for them, so
they never have to hand-craft agents, souls, links, or briefs in the raw UI.

# Your environment (where you live)

You run INSIDE a live swarm server — a long-running process that hosts every
team. The human talks to you through a chat panel in the swarm DASHBOARD (a web
UI that also shows every team as a live canvas of agents, their messages, queues,
and costs). You are a singleton: one Architect for the whole server, with your
own private conversation that no team agent can see.

Your actions take effect on the LIVE system immediately. When you create an
agent it spins up as a running daemon right away; when you write workspace.md it
becomes that team's brief on the agents' very next turn; when you send a task it
lands in a real queue. There is no "draft" or "deploy" step — building IS doing.
So design carefully and propose before you build.

The current swarm state (which teams and agents exist, the model agents run on,
and the toolsets available to agents) is given to you at the top of each message
in a [SWARM STATE] line — read it. Use master_overview / master_get_team for
detail. You also have web_search and web_extract to research a domain, product,
competitor, or best practices before proposing a team — use them when the goal
involves a space you should understand first.

# What the framework is

A "team" is a named group of agents working a shared goal. Each AGENT is an
autonomous Hermes LLM with:
  - role_soul: its identity/system prompt — the single most important field. It
    defines who the agent is, what it owns, and how it behaves. Write souls that
    are crisp (≈100–200 words), in the second person ("You are the …"), with
    NON-OVERLAPPING mandates so two agents never fight over the same job. State
    what it owns, who it delegates to, what "done" means, and when to escalate.
  - allowed_peers: who it may message. Links are BIDIRECTIONAL and SAME-TEAM
    only (linking A→B also links B→A). An agent can only send_peer_message to a
    linked peer. Design the topology deliberately: a hub-and-spoke around a lead
    (founder/manager) is usually right; avoid a fully-connected mesh.
  - is_supervisor: a supervisor receives automated periodic SWEEPS of everything
    its linked agents did, and nudges/redirects them. It does not do the work.
    Give a team ONE supervisor (often an "overseer"/"manager") linked to all
    workers. Supervisors have a sensible default soul if you don't supply one.
  - autonomous: when true, the agent self-drives the mission when its queue is
    idle (heartbeat). Recommend EXACTLY ONE autonomous driver per team (the
    lead) so the team has a single engine; workers are reactive (autonomous off)
    and act when delegated to. Too many autonomous agents = runaway token spend.
  - model / sampling: leave unset to inherit the swarm default model — only
    override when a role genuinely needs a different model.
  - crons (scheduled wake-ups): for periodic routines (a 9am check-in). Keep the
    instruction GOAL-LEVEL and short (≤600 chars); put detailed steps in a
    workspace runbook file the cron references — never inline a long frozen
    script (it rots).

Agents automatically get the swarm coordination tools (send_peer_message,
ask_human, log_decision, …) plus file/terminal/browser toolsets. You don't wire
those; you design roles, souls, links, and the brief.

# Workspace conventions

Every team has a shared brief at workspace.md (area "workspace", path
"workspace.md"). It is injected into EVERY agent's prompt each turn — it is the
single source of truth for the goal, north-star metric, constraints, key
decisions, and shared conventions. Write it carefully and keep it tight (it is
re-sent every turn — bloat is a direct token cost). Use master_write_file to
create/edit it.

Teams also have a shared PROJECT directory (area "project") — the real working
surface where agents read/write code and deliverables together. Seed starter
files there when useful (a README, a skeleton, a spec).

# Cost reality

Input tokens dominate cost: every tool-call iteration re-sends the agent's whole
context. So keep souls and the brief lean, prefer one supervisor and one
autonomous driver, and don't create more agents than the goal needs. A focused
3–5 agent team almost always beats a sprawling one.

# How you operate

1. INTERVIEW first. Ask the human focused questions (in small batches, not a
   wall) about: the goal and success metric, the domain, any deployment/credentials/
   tools involved, how much autonomy they want, and any roles they already have in
   mind. Don't over-ask — once you understand the goal, move on.
2. PROPOSE a concrete plan in chat BEFORE building: the team name, each agent
   (name, one-line role, supervisor/autonomous flags), the link topology, and a
   draft of workspace.md. Let the human correct it.
3. BUILD only after the human approves. Use the tools: create the team, create
   each agent with its full soul, set the links, write workspace.md and any seed
   files. Create the supervisor LAST (after the agents it watches exist) and link
   it to them.
4. VERIFY with master_get_team and report what you built in plain language.
   Offer to kick the team off with master_send_task (usually a first directive to
   the lead/autonomous agent).
5. For EDITS to existing teams ("add a QA agent", "change the brief"), inspect
   first with master_overview / master_get_team, then make the targeted change.

# Safety

- NEVER invent credentials, API keys, URLs, or external accounts. If the team
  will need them, ASK the human to provide them and record where they live in
  workspace.md (never paste secrets into souls or committed files).
- DELETIONS are destructive. Propose the deletion, get an explicit "yes", and
  only then call the delete tool with confirm=true.
- You only have the swarm_master tools. You cannot browse, run code, or message
  team agents — you act through your tools and you talk to the human through your
  replies. Be concrete and concise; the human reads your messages in a chat panel.
"""

# Short operating reminder pinned as the ephemeral system prompt (the manual
# above lives in SOUL.md as the lead identity block).
MASTER_EPHEMERAL = """\
Operating rules: interview → propose in chat → build only on explicit approval →
verify → offer to kick off. Create supervisors after their workers exist. Keep
souls and workspace.md lean. Never invent credentials. Confirm before any delete.
Each reply is shown to the human in a chat panel — be concrete and concise.
"""


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
def _fn(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_AREA_PROP = {
    "type": "string",
    "enum": ["workspace", "project"],
    "description": "'workspace' = team metadata dir (where workspace.md lives); "
    "'project' = the shared working/code directory.",
}

_MASTER_TOOL_SCHEMAS: List[dict] = [
    _fn("master_overview",
        "List every team and its agents (roles, links, supervisor/autonomous flags, "
        "model) plus each team's workspace.md. Call this first when you need current state.",
        {}, []),
    _fn("master_get_team",
        "Full detail for ONE team: every agent's soul, links, flags, crons, plus the "
        "team's workspace.md and a tree of its workspace + project files.",
        {"team_id": {"type": "string"}}, ["team_id"]),
    _fn("master_create_team",
        "Create a new empty team. Returns success; add agents next.",
        {"team_id": {"type": "string", "description": "slug: letters/digits/_/- only"},
         "name": {"type": "string", "description": "human-readable display name"}},
        ["team_id", "name"]),
    _fn("master_create_agent",
        "Create one agent in a team. Create the agents a supervisor watches BEFORE the "
        "supervisor, then link them. Its daemon starts immediately.",
        {"team_id": {"type": "string"},
         "agent_name": {"type": "string", "description": "slug, unique across the whole swarm"},
         "display_name": {"type": "string"},
         "role_soul": {"type": "string", "description": "the agent's identity/system prompt (~100–200 words)"},
         "allowed_peers": {"type": "array", "items": {"type": "string"},
                           "description": "agent_names in the SAME team to link bidirectionally (optional)"},
         "is_supervisor": {"type": "boolean", "description": "default false"},
         "autonomous": {"type": "boolean", "description": "self-drives when idle; recommend exactly one per team"},
         "model": {"type": "string", "description": "optional model override; omit to inherit the swarm default"}},
        ["team_id", "agent_name", "display_name", "role_soul"]),
    _fn("master_update_agent",
        "Patch an existing agent's editable fields (role_soul, autonomous, is_supervisor, "
        "model, max_iterations, enabled_toolsets, disabled_toolsets, …). Re-inits the agent "
        "on its next turn. Pass only the fields to change.",
        {"agent_name": {"type": "string"},
         "fields": {"type": "object", "description": "map of field → new value"}},
        ["agent_name", "fields"]),
    _fn("master_set_links",
        "Set an agent's complete peer list (replaces it). Bidirectional + same-team enforced.",
        {"agent_name": {"type": "string"},
         "peers": {"type": "array", "items": {"type": "string"}}},
        ["agent_name", "peers"]),
    _fn("master_write_file",
        "Create or overwrite a team file. Use this for workspace.md (area=workspace, "
        "path=workspace.md) and to seed project files (area=project).",
        {"team_id": {"type": "string"}, "area": _AREA_PROP,
         "path": {"type": "string", "description": "relative path within the area, no '..'"},
         "content": {"type": "string"}},
        ["team_id", "area", "path", "content"]),
    _fn("master_read_file",
        "Read a team file's contents.",
        {"team_id": {"type": "string"}, "area": _AREA_PROP,
         "path": {"type": "string"}},
        ["team_id", "area", "path"]),
    _fn("master_list_files",
        "List files under a team area (workspace or project).",
        {"team_id": {"type": "string"}, "area": _AREA_PROP},
        ["team_id", "area"]),
    _fn("master_send_task",
        "Send a task/directive to an agent (as if from the human operator) — use this to "
        "kick a freshly-built team off via its lead/autonomous agent.",
        {"agent_name": {"type": "string"}, "message": {"type": "string"}},
        ["agent_name", "message"]),
    _fn("master_delete_agent",
        "DELETE an agent and its workspace. Destructive — propose it and get an explicit "
        "human yes first, then call with confirm=true.",
        {"agent_name": {"type": "string"}, "confirm": {"type": "boolean"}},
        ["agent_name", "confirm"]),
    _fn("master_delete_team",
        "DELETE an entire team, all its agents and files. Destructive — explicit human "
        "confirmation required, then confirm=true.",
        {"team_id": {"type": "string"}, "confirm": {"type": "boolean"}},
        ["team_id", "confirm"]),
]


# ---------------------------------------------------------------------------
# Tool handlers (caller-guarded; call config.py helpers directly)
# ---------------------------------------------------------------------------
def _caller(kwargs: dict) -> str:
    tid = kwargs.get("task_id", "")
    if isinstance(tid, str) and tid.startswith("agent_name:"):
        return tid.split(":", 1)[1]
    return "unknown"


def _err(msg: str) -> str:
    return json.dumps({"success": False, "error": msg})


def _ok(**data) -> str:
    return json.dumps({"success": True, **data}, default=str)


def _guard(kwargs: dict) -> Optional[str]:
    if _caller(kwargs) != MASTER_NAME:
        return _err("This tool is restricted to the Architect (master) agent.")
    return None


def _agent_brief(name: str, a: dict) -> dict:
    return {
        "agent_name": name,
        "display_name": a.get("name"),
        "is_supervisor": bool(a.get("is_supervisor")),
        "autonomous": bool(a.get("autonomous")),
        "model": a.get("model") or "(default)",
        "allowed_peers": a.get("allowed_peers", []),
        "crons": len(a.get("crons") or []),
    }


def _read_workspace_md(team_id: str) -> str:
    from swarm_server.config import _get_team_workspace_path

    p = _get_team_workspace_path(team_id) / "workspace.md"
    try:
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception as e:  # noqa: BLE001
        return f"[unreadable: {e}]"


def _master_overview_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.config import load_agents_config

    cfg = load_agents_config()
    teams = []
    for tid, t in (cfg.get("teams") or {}).items():
        members = [
            _agent_brief(n, a) for n, a in cfg["agents"].items()
            if a.get("team_id") == tid
        ]
        teams.append({
            "team_id": tid,
            "name": t.get("name"),
            "agents": members,
            "workspace_md": _read_workspace_md(tid),
        })
    return _ok(teams=teams, team_count=len(teams))


def _file_tree(base: Path, max_entries: int = _LIST_FILES_MAX) -> List[str]:
    out: List[str] = []
    if not base.exists():
        return out
    base = base.resolve()
    for root, dirs, files in os.walk(base):
        # Skip VCS / hermes internals and cap depth.
        rel_root = Path(root).resolve().relative_to(base)
        depth = 0 if str(rel_root) == "." else len(rel_root.parts)
        if depth >= _LIST_FILES_MAX_DEPTH:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in (".git", ".hermes", "node_modules", "__pycache__")]
        for f in sorted(files):
            rel = (rel_root / f) if str(rel_root) != "." else Path(f)
            out.append(str(rel))
            if len(out) >= max_entries:
                return out
    return out


def _master_get_team_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.config import (
        load_agents_config, _get_team_workspace_path, _get_project_dir,
    )

    team_id = (args.get("team_id") or "").strip()
    cfg = load_agents_config()
    if team_id not in (cfg.get("teams") or {}):
        return _err(f"Team '{team_id}' not found.")
    agents = []
    for n, a in cfg["agents"].items():
        if a.get("team_id") != team_id:
            continue
        agents.append({
            **_agent_brief(n, a),
            "role_soul": a.get("role_soul", ""),
            "crons_detail": [
                {"schedule": c.get("schedule"), "instruction": c.get("instruction")}
                for c in (a.get("crons") or [])
            ],
        })
    return _ok(
        team_id=team_id,
        name=cfg["teams"][team_id].get("name"),
        agents=agents,
        workspace_md=_read_workspace_md(team_id),
        workspace_files=_file_tree(_get_team_workspace_path(team_id)),
        project_files=_file_tree(_get_project_dir(team_id, cfg)),
    )


def _master_create_team_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.config import load_agents_config, create_team

    team_id = (args.get("team_id") or "").strip()
    name = (args.get("name") or "").strip()
    if not name:
        return _err("A team needs a display name.")
    cfg = load_agents_config()
    try:
        team = create_team(cfg, team_id, name)
    except ValueError as e:
        return _err(str(e))
    log.info("[master] created team '%s'", team_id)
    _broadcast("team_created", {"team_id": team_id, "name": name, "timestamp": time.time()})
    return _ok(team_id=team_id, team=team)


def _master_create_agent_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.config import (
        load_agents_config, create_agent, save_agent_config, set_agent_peers,
    )

    team_id = (args.get("team_id") or "").strip()
    agent_name = (args.get("agent_name") or "").strip()
    display_name = (args.get("display_name") or "").strip()
    role_soul = args.get("role_soul") or ""
    is_supervisor = bool(args.get("is_supervisor"))
    peers = args.get("allowed_peers") or []
    if not display_name:
        return _err("An agent needs a display_name.")
    if not isinstance(peers, list):
        return _err("allowed_peers must be a list of agent_names.")

    cfg = load_agents_config()
    try:
        # Create with no peers first; set_agent_peers below validates same-team
        # membership now that the agent exists.
        create_agent(
            cfg, name=agent_name, team_id=team_id, display_name=display_name,
            allowed_peers=None, role_soul=role_soul, is_supervisor=is_supervisor,
        )
    except ValueError as e:
        return _err(str(e))

    # Optional extra fields not covered by create_agent's signature.
    extra: Dict[str, Any] = {}
    if args.get("autonomous") is not None:
        extra["autonomous"] = bool(args.get("autonomous"))
    model = (args.get("model") or "").strip()
    if model:
        extra["model"] = model
    if extra:
        cfg = load_agents_config()
        cfg["agents"][agent_name].update(extra)
        save_agent_config(agent_name, cfg["agents"][agent_name])

    # Wire links (validated: same-team, existing). Non-fatal if it fails — the
    # agent is already created and the human can fix links after.
    link_warning = None
    if peers:
        try:
            set_agent_peers(load_agents_config(), agent_name, [str(p) for p in peers])
        except ValueError as e:
            link_warning = f"agent created but links not set: {e}"

    _hook("spawn", agent_name)
    log.info("[master] created agent '%s' in '%s' (supervisor=%s)", agent_name, team_id, is_supervisor)
    _broadcast("agent_created", {"agent_name": agent_name, "team_id": team_id, "timestamp": time.time()})
    out = {"agent_name": agent_name, "team_id": team_id}
    if link_warning:
        out["warning"] = link_warning
    return _ok(**out)


def _master_update_agent_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.config import load_agents_config, save_agent_config
    # Reuse the SAME whitelist + normalization the REST PATCH endpoint uses.
    from swarm_server.server import _apply_config_fields

    agent_name = (args.get("agent_name") or "").strip()
    fields = args.get("fields") or {}
    if not isinstance(fields, dict) or not fields:
        return _err("'fields' must be a non-empty object of editable fields.")
    cfg = load_agents_config()
    a = cfg["agents"].get(agent_name)
    if a is None:
        return _err(f"Agent '{agent_name}' not found.")
    updated = _apply_config_fields(a, fields)
    save_agent_config(agent_name, updated)
    _hook("update", agent_name)
    log.info("[master] updated agent '%s' fields=%s", agent_name, list(fields.keys()))
    _broadcast("agent_config_updated", {"agent_name": agent_name, "timestamp": time.time()})
    return _ok(agent_name=agent_name, config=updated)


def _master_set_links_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.config import load_agents_config, set_agent_peers

    agent_name = (args.get("agent_name") or "").strip()
    peers = args.get("peers")
    if not isinstance(peers, list):
        return _err("'peers' must be a list of agent_names.")
    try:
        set_agent_peers(load_agents_config(), agent_name, [str(p) for p in peers])
    except ValueError as e:
        return _err(str(e))
    # Links are read live from config in every agent's prompt/peer-check, so no
    # re-init is required — but refresh the daemon cfg view for tidiness.
    _hook("update", agent_name)
    log.info("[master] set links for '%s': %s", agent_name, peers)
    _broadcast("agent_config_updated", {"agent_name": agent_name, "timestamp": time.time()})
    return _ok(agent_name=agent_name, allowed_peers=peers)


def _resolve_team_file(team_id: str, area: str, rel_path: str):
    """Return (base, resolved_path) or raise ValueError on an unsafe/unknown path."""
    from swarm_server.config import (
        load_agents_config, _get_team_workspace_path, _ensure_project_dir,
    )

    cfg = load_agents_config()
    if team_id not in (cfg.get("teams") or {}):
        raise ValueError(f"Team '{team_id}' not found.")
    if area == "workspace":
        base = _get_team_workspace_path(team_id)
    elif area == "project":
        base = _ensure_project_dir(team_id, cfg)
    else:
        raise ValueError("area must be 'workspace' or 'project'.")
    rel_path = (rel_path or "").strip().lstrip("/")
    if not rel_path:
        raise ValueError("path is required.")
    if ".." in Path(rel_path).parts:
        raise ValueError("path may not contain '..'.")
    base = base.resolve()
    target = (base / rel_path).resolve()
    if base != target and base not in target.parents:
        raise ValueError("path escapes the team directory.")
    return base, target


def _master_write_file_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    content = args.get("content")
    if not isinstance(content, str):
        return _err("'content' must be a string.")
    if len(content) > _MAX_FILE_CHARS:
        return _err(f"content too long ({len(content)} > {_MAX_FILE_CHARS} chars).")
    try:
        base, target = _resolve_team_file(
            (args.get("team_id") or "").strip(), (args.get("area") or "").strip(),
            args.get("path") or "",
        )
    except ValueError as e:
        return _err(str(e))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return _err(f"write failed: {e}")
    rel = target.relative_to(base)
    log.info("[master] wrote %s/%s (%d chars)", args.get("area"), rel, len(content))
    return _ok(path=str(rel), bytes=len(content))


def _master_read_file_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    try:
        base, target = _resolve_team_file(
            (args.get("team_id") or "").strip(), (args.get("area") or "").strip(),
            args.get("path") or "",
        )
    except ValueError as e:
        return _err(str(e))
    if not target.exists() or not target.is_file():
        return _err("file not found.")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return _err(f"read failed: {e}")
    truncated = len(text) > _MAX_FILE_CHARS
    return _ok(path=str(target.relative_to(base)),
               content=text[:_MAX_FILE_CHARS], truncated=truncated)


def _master_list_files_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    try:
        base, _ = _resolve_team_file(
            (args.get("team_id") or "").strip(), (args.get("area") or "").strip(), ".",
        )
    except ValueError as e:
        return _err(str(e))
    files = _file_tree(base)
    return _ok(files=files, count=len(files), truncated=len(files) >= _LIST_FILES_MAX)


def _master_send_task_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    from swarm_server.tools import _daemon_registry

    agent_name = (args.get("agent_name") or "").strip()
    message = (args.get("message") or "").strip()
    if not message:
        return _err("A task needs a message.")
    daemon = _daemon_registry.get(agent_name)
    if daemon is None:
        return _err(f"Agent '{agent_name}' has no running daemon (created but not started?).")
    try:
        task_id = daemon.ingest_task("human", message)
    except Exception as e:  # noqa: BLE001
        return _err(f"could not enqueue task: {e}")
    log.info("[master] sent kickoff task to '%s'", agent_name)
    return _ok(agent_name=agent_name, task_id=task_id)


def _master_delete_agent_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    if not bool(args.get("confirm")):
        return _err("Refused: set confirm=true ONLY after the human explicitly approved the deletion.")
    from swarm_server.config import load_agents_config

    agent_name = (args.get("agent_name") or "").strip()
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return _err(f"Agent '{agent_name}' not found.")
    # The despawn hook drains the daemon AND calls delete_agent (which rmtrees the
    # workspace) on the event loop, in the correct order. See server wiring.
    _hook("despawn", agent_name)
    log.info("[master] deleted agent '%s'", agent_name)
    _broadcast("agent_deleted", {"agent_name": agent_name, "timestamp": time.time()})
    return _ok(agent_name=agent_name, deleted=True)


def _master_delete_team_handler(args: dict, **kwargs) -> str:
    if (g := _guard(kwargs)):
        return g
    if not bool(args.get("confirm")):
        return _err("Refused: set confirm=true ONLY after the human explicitly approved the deletion.")
    from swarm_server.config import load_agents_config

    team_id = (args.get("team_id") or "").strip()
    cfg = load_agents_config()
    if team_id not in (cfg.get("teams") or {}):
        return _err(f"Team '{team_id}' not found.")
    members = [n for n, a in cfg["agents"].items() if a.get("team_id") == team_id]
    # Despawn every member's daemon first (drains + deletes the agent), then drop
    # the now-empty team. The despawn hook handles each agent; the team-drop hook
    # removes the team record + remaining files.
    for n in members:
        _hook("despawn", n)
    _hook("despawn", f"__team__:{team_id}")
    log.info("[master] deleted team '%s' (%d agents)", team_id, len(members))
    _broadcast("team_deleted", {"team_id": team_id, "timestamp": time.time()})
    return _ok(team_id=team_id, deleted=True, agents_removed=len(members))


_MASTER_HANDLERS = {
    "master_overview": _master_overview_handler,
    "master_get_team": _master_get_team_handler,
    "master_create_team": _master_create_team_handler,
    "master_create_agent": _master_create_agent_handler,
    "master_update_agent": _master_update_agent_handler,
    "master_set_links": _master_set_links_handler,
    "master_write_file": _master_write_file_handler,
    "master_read_file": _master_read_file_handler,
    "master_list_files": _master_list_files_handler,
    "master_send_task": _master_send_task_handler,
    "master_delete_agent": _master_delete_agent_handler,
    "master_delete_team": _master_delete_team_handler,
}


def register_master_tools() -> None:
    """Register the swarm_master toolset in the shared Hermes registry (idempotent)."""
    from swarm_server.config import ensure_hermes_importable

    ensure_hermes_importable()
    from tools.registry import registry

    existing = registry.get_tool_to_toolset_map() or {}
    for schema in _MASTER_TOOL_SCHEMAS:
        fn = schema["function"]
        name = fn["name"]
        if name in existing:
            continue
        registry.register(
            name=name,
            toolset="swarm_master",
            schema=fn,
            handler=_MASTER_HANDLERS[name],
            description=fn["description"][:120],
        )
    log.info("[master] swarm_master toolset registered (%d tools)", len(_MASTER_TOOL_SCHEMAS))


# ---------------------------------------------------------------------------
# MasterAgent — the chat wrapper
# ---------------------------------------------------------------------------
class MasterAgent:
    def __init__(self) -> None:
        from swarm_server.config import DATA_ROOT

        self._home = Path(DATA_ROOT) / "master"
        self._hermes_home = self._home / ".hermes"
        self._hermes_home.mkdir(parents=True, exist_ok=True)
        self._state_path = self._home / "master.json"
        self._log_path = self._home / "chat_log.jsonl"
        self._session_id = self._load_session_id()
        self._ai_agent = None
        self._lock = threading.Lock()
        self._busy = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="master")
        self._current_model: Optional[str] = None

    # -- persistence -------------------------------------------------------
    def _load_session_id(self) -> str:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                sid = data.get("session_id")
                if sid:
                    return sid
        except Exception:  # noqa: BLE001
            pass
        return "master-session-v1"

    def _save_session_id(self) -> None:
        try:
            self._state_path.write_text(
                json.dumps({"session_id": self._session_id}), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            log.warning("[master] could not persist session id: %s", e)

    def _append_log(self, role: str, content: str) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "content": content, "ts": time.time()}) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("[master] could not append chat log: %s", e)

    def history(self, limit: int = _HISTORY_DEFAULT) -> List[Dict[str, Any]]:
        if not self._log_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        continue
        except Exception as e:  # noqa: BLE001
            log.warning("[master] could not read chat log: %s", e)
        return out[-limit:]

    # -- model status ------------------------------------------------------
    def model(self) -> str:
        from swarm_server.model_config import resolve_model
        from swarm_server.config import get_global_settings

        override = (get_global_settings().get("master_model") or "").strip()
        eff = resolve_model({"model": override} if override else {})
        return eff.get("model") or "(unconfigured)"

    def is_busy(self) -> bool:
        return self._busy

    def is_configured(self) -> bool:
        from swarm_server.model_config import is_model_configured

        try:
            return bool(is_model_configured())
        except Exception:  # noqa: BLE001
            return True

    # -- agent construction (mirrors AgentDaemon._ensure_agent, trimmed) ----
    def _ensure_agent(self):
        if self._ai_agent is not None:
            return
        from swarm_server.agent import (
            _ensure_hermes_on_path, _agent_init_lock,
        )
        from swarm_server.config import write_agent_hermes_config, get_global_settings
        from swarm_server.model_config import resolve_model

        with _agent_init_lock:
            if self._ai_agent is not None:
                return
            _ensure_hermes_on_path()
            from run_agent import AIAgent
            from hermes_state import SessionDB

            os.environ["HERMES_HOME"] = str(self._hermes_home)

            override = (get_global_settings().get("master_model") or "").strip()
            eff = resolve_model({"model": override} if override else {})
            model = eff["model"]
            self._current_model = model
            write_agent_hermes_config(
                self._hermes_home, cdp_url=None, model=model,
                provider=eff["provider"], base_url=eff["base_url"], api_key=eff["api_key"],
            )

            # Register the master toolset BEFORE init so enabled_toolsets finds it.
            register_master_tools()

            session_db = None
            try:
                session_db = SessionDB(db_path=self._hermes_home / "state.db")
            except Exception as e:  # noqa: BLE001
                log.error("[master] isolated SessionDB failed: %s", e)

            self._write_soul_md(f"You are the Architect.\n\n{MASTER_SOUL}")

            # The Architect gets its own swarm_master tools plus web search/extract
            # (web_search, web_extract) so it can research a domain while designing
            # a team. It deliberately gets NOTHING else — no terminal, browser, or
            # file tools beyond its scoped master_* file tools.
            extra_kwargs: Dict[str, Any] = {"enabled_toolsets": ["swarm_master", "web"]}
            if eff.get("provider"):
                extra_kwargs["provider"] = eff["provider"]

            self._ai_agent = AIAgent(
                base_url=eff["base_url"] or None,
                api_key=eff["api_key"] or None,
                model=model,
                session_id=self._session_id,
                skip_memory=False,
                skip_context_files=False,
                quiet_mode=True,
                ephemeral_system_prompt=MASTER_EPHEMERAL,
                session_db=session_db,
                max_iterations=_MASTER_MAX_ITERATIONS,
                **extra_kwargs,
            )
            # Defensive: ensure every master tool is present even if the toolset
            # filter missed one (ordering / version differences).
            self._attach_master_tools()
            self._wire_callbacks()
            log.info("[master] agent ready (model=%s)", model)

    def _write_soul_md(self, content: str) -> None:
        try:
            self._hermes_home.mkdir(parents=True, exist_ok=True)
            (self._hermes_home / "SOUL.md").write_text(content, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("[master] could not write SOUL.md: %s", e)

    def _attach_master_tools(self) -> None:
        a = self._ai_agent
        if a is None:
            return
        if a.tools is None:
            a.tools = []
        if getattr(a, "valid_tool_names", None) is None:
            a.valid_tool_names = set()
        present = {t.get("function", {}).get("name") for t in a.tools}
        for schema in _MASTER_TOOL_SCHEMAS:
            name = schema["function"]["name"]
            if name not in present:
                a.tools.append(schema)
                a.valid_tool_names.add(name)

    def _wire_callbacks(self) -> None:
        a = self._ai_agent
        if a is None:
            return

        def on_tool_start(tool_call_id, name, targs):  # noqa: ANN001
            _broadcast("master_exec", {"kind": "tool_start", "name": str(name), "timestamp": time.time()})

        def on_tool_complete(tool_call_id, name, targs, result):  # noqa: ANN001
            _broadcast("master_exec", {"kind": "tool_result", "name": str(name), "timestamp": time.time()})

        def on_thinking(text: str = "") -> None:
            if text:
                _broadcast("master_exec", {"kind": "thinking", "text": str(text)[:160], "timestamp": time.time()})

        for attr, cb in (
            ("tool_start_callback", on_tool_start),
            ("tool_complete_callback", on_tool_complete),
            ("thinking_callback", on_thinking),
        ):
            try:
                setattr(a, attr, cb)
            except Exception:  # noqa: BLE001
                pass

    def _load_history(self) -> List[Dict[str, Any]]:
        if self._ai_agent is None:
            return []
        session_db = getattr(self._ai_agent, "_session_db", None)
        if session_db is None:
            return []
        try:
            from swarm_server.prompts import age_stale_tool_results
            from swarm_server.config import (
                TOOL_RESULT_AGE_ENABLED, TOOL_RESULT_AGE_KEEP_MESSAGES,
                TOOL_RESULT_AGE_MIN_CHARS, TOOL_RESULT_AGE_QUANTUM,
            )

            sid = getattr(self._ai_agent, "session_id", None) or self._session_id
            msgs = session_db.get_messages_as_conversation(sid, include_ancestors=False)
            if TOOL_RESULT_AGE_ENABLED:
                msgs = age_stale_tool_results(
                    msgs, keep_recent=TOOL_RESULT_AGE_KEEP_MESSAGES,
                    min_chars=TOOL_RESULT_AGE_MIN_CHARS, quantum=TOOL_RESULT_AGE_QUANTUM,
                )
            return msgs
        except Exception as e:  # noqa: BLE001
            log.warning("[master] history load failed: %s", e)
            return []

    def _state_header(self) -> str:
        """A short, current snapshot prepended to the user turn so the master
        always knows what exists (teams, the agent model, available toolsets)
        without spending a tool call on trivia."""
        try:
            from swarm_server.config import load_agents_config

            cfg = load_agents_config()
            teams = cfg.get("teams") or {}
            if not teams:
                line = "none yet (greenfield)"
            else:
                parts = []
                for tid in teams:
                    n = sum(1 for a in cfg["agents"].values() if a.get("team_id") == tid)
                    parts.append(f"{tid} ({n} agents)")
                line = "; ".join(parts)
            model = self._current_model or self.model()
            header = (
                f"[SWARM STATE] teams: {line}. Agents run on model: {model}. "
                f"Toolsets you can enable on an agent: {_available_toolsets_str()}."
            )
            return header
        except Exception:  # noqa: BLE001
            return ""

    # -- the turn ----------------------------------------------------------
    def submit(self, message: str) -> bool:
        """Schedule a chat turn on the master's worker thread. Returns False if
        already busy. The reply arrives via the 'master_message' WS event."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._append_log("user", message)
        self._executor.submit(self._run_turn, message)
        return True

    def _run_turn(self, message: str) -> None:
        from swarm_server.agent import (
            _set_hermes_home_override, _reset_hermes_home_override,
        )

        _broadcast("master_state", {"busy": True, "timestamp": time.time()})
        token = _set_hermes_home_override(self._hermes_home)
        reply = ""
        try:
            self._ensure_agent()
            header = self._state_header()
            combined = f"{header}\n\n{message}" if header else message
            history = self._load_history()
            resp = self._ai_agent.run_conversation(
                user_message=combined,
                task_id=MASTER_TASK_ID,
                conversation_history=history,
            )
            reply = (resp or {}).get("final_response", "") or "(no reply)"
            # Persist a compaction-rotated session id so the next turn resumes
            # from the compacted session, like AgentDaemon does.
            live_sid = getattr(self._ai_agent, "session_id", None)
            if live_sid and live_sid != self._session_id:
                self._session_id = live_sid
                self._save_session_id()
        except Exception as e:  # noqa: BLE001
            log.exception("[master] turn failed")
            reply = f"⚠️ The Architect hit an error: {e}"
        finally:
            _reset_hermes_home_override(token)
            self._append_log("assistant", reply)
            with self._lock:
                self._busy = False
            _broadcast("master_message", {"role": "assistant", "content": reply, "timestamp": time.time()})
            _broadcast("master_state", {"busy": False, "timestamp": time.time()})

    def reload(self) -> None:
        """Drop the cached AIAgent so the next turn rebuilds it (e.g. after the
        master_model setting changed). Keeps the session + chat log intact."""
        with self._lock:
            if not self._busy:
                self._ai_agent = None

    def reset(self) -> None:
        """Start a fresh conversation (new session) without losing the chat log."""
        with self._lock:
            self._session_id = f"master-session-{int(time.time())}"
            self._ai_agent = None
            self._save_session_id()
        self._append_log("system", "— conversation reset —")
        _broadcast("master_state", {"busy": False, "reset": True, "timestamp": time.time()})


# Singleton accessor ---------------------------------------------------------
_master_singleton: Optional[MasterAgent] = None
_singleton_lock = threading.Lock()


def get_master() -> MasterAgent:
    global _master_singleton
    if _master_singleton is None:
        with _singleton_lock:
            if _master_singleton is None:
                _master_singleton = MasterAgent()
    return _master_singleton
