"""Configuration constants and agent config management."""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("swarm.config")

# Serializes all reads/writes of agents_config.json across threads. Reentrant
# because the read path (load_agents_config) may trigger a migration write
# (_save_full_config) within the same call.
_config_lock = threading.RLock()

# In-process cache of the parsed config, keyed on the file's (mtime_ns, size).
# load_agents_config() is on the hot path (every peer message); without this it
# re-reads + re-parses + re-runs migration scans on every call. Callers mutate
# the returned dict, so reads always hand back a deep copy.
_config_cache: Optional[Dict[str, Any]] = None
_config_cache_key: Optional[tuple] = None


def _config_file_key() -> Optional[tuple]:
    try:
        st = AGENTS_CONFIG_PATH.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
AGENTS_CONFIG_PATH = DATA_ROOT / "agents_config.json"
MONITORING_DB = DATA_ROOT / "monitoring.db"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

WORKSPACE_ROOT = DATA_ROOT / "teams"

# ---------------------------------------------------------------------------
# Network / Runtime
# ---------------------------------------------------------------------------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
LITELLM_API_BASE = f"http://{SERVER_HOST}:4000/v1"
SWEEP_INTERVAL_SECONDS = 10

# ---------------------------------------------------------------------------
# Context / session compression (Hermes built-in ContextCompressor)
# ---------------------------------------------------------------------------
# Hermes auto-compacts a conversation once its token estimate crosses
# context_window * threshold; the middle turns are summarized by the auxiliary
# model and the session is split in SQLite (a new child session holds the
# summary + recent tail). These settings are written into each agent's isolated
# {HERMES_HOME}/config.yaml so compaction triggers deterministically rather than
# relying on provider auto-detection — which cannot see the real model behind a
# LiteLLM proxy and would otherwise fall back to a 256K guess.
#
# CONTEXT window is pinned conservatively: setting it BELOW the real backing
# model only compacts sooner (cheap, safe); setting it ABOVE risks a hard
# context-overflow error before Hermes ever compacts. Tune to match (or sit
# just under) your LiteLLM backing model's real window. Must be >= 64000
# (Hermes' MINIMUM_CONTEXT_LENGTH) or AIAgent init raises.
COMPRESSION_ENABLED = True
AGENT_CONTEXT_WINDOW = 128000
COMPRESSION_THRESHOLD = 0.5          # compact at ~50% of the window
COMPRESSION_TARGET_RATIO = 0.20      # summary budget as a fraction of compacted content
COMPRESSION_PROTECT_FIRST_N = 3      # head turns kept verbatim (besides system prompt)
COMPRESSION_PROTECT_LAST_N = 12      # recent tail turns kept verbatim

# ---------------------------------------------------------------------------
# Queue / sweep behavior
# ---------------------------------------------------------------------------
MAX_BATCH_SIZE = 10          # max tasks pulled into one LLM turn (backpressure)
MAX_TASK_RETRIES = 3         # batch failures requeue up to N times, then -> 'failed' (DLQ)

# ---------------------------------------------------------------------------
# Monitoring retention — rolling cap so monitoring.db stays bounded over 24/7 runs
# ---------------------------------------------------------------------------
MONITORING_MAX_EVENTS = 50000
MONITORING_MAX_MESSAGES = 20000
MONITORING_PRUNE_INTERVAL_SECONDS = 300

# Human-inbox registry cap (in-memory) — drop oldest resolved questions past this.
MAX_PENDING_QUESTIONS = 500

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
# The dashboard is served same-origin, so it needs no CORS at all; restricting to
# localhost drops the spec-invalid wildcard+credentials combo. Override with the
# SWARM_CORS_ORIGINS env var (comma-separated) if hosting the UI elsewhere.
CORS_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "SWARM_CORS_ORIGINS",
        f"http://{SERVER_HOST}:{SERVER_PORT},http://localhost:{SERVER_PORT}",
    ).split(",")
    if o.strip()
]
# Optional bearer token guarding mutating endpoints. Unset => auth disabled
# (relies on localhost binding). Set SWARM_API_KEY to require it.
SWARM_API_KEY = os.environ.get("SWARM_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Common SOUL — shared by all agents in the framework
# Injected at runtime with: {agent_name}, {team_id}, {allowed_peers_list},
# {workspace_path}, {org_diagram}, {all_team_members}
# ---------------------------------------------------------------------------
COMMON_SOUL_TEMPLATE = (
    "You are an autonomous agent in a multi-agent swarm working on a shared project.\n"
    "You operate within an async system: tasks arrive in batches and are processed "
    "every ~{sweep_interval}s. Responses are not immediate.\n\n"
    "--- SWARM RULES ---\n"
    "1. NEVER end your turn silently. Always conclude by calling a tool.\n"
    "   - Prefer 'send_peer_message' to delegate work and keep your context short.\n"
    "   - Use 'ask_human' only when genuinely stuck, instructions are ambiguous, "
    "     or a task requires human judgment (approvals, subjective choices).\n"
    "   - Use 'log_changes' after completing tasks or when something important happens.\n"
    "2. After calling 'send_peer_message' or 'ask_human', STOP calling tools and end your turn.\n"
    "3. Process tasks autonomously without asking for permission.\n"
    "4. Report completions, blockers, or delegation decisions back to the sender.\n"
    "5. Prefer DELEGATING tasks to other agents over doing everything yourself.\n"
    "   Offload work to keep your context window short and focused.\n"
    "6. Keep responses concise — other agents read them in batch.\n\n"
    "--- WORKSPACE RULES ---\n"
    "Your dedicated workspace: {workspace_path}\n"
    "You may ONLY read/write files within this directory or the shared team workspace.\n"
    "When delegating a file-based task, specify the relative path so the recipient "
    "knows exactly where to read/write.\n"
    "When receiving file-based tasks, check your workspace for files at the provided path.\n"
    "Do NOT write files outside your workspace.\n"
    "Team workspace.md: {workspace_path}/../workspace.md\n"
    "  (Read this for project overview and shared context)\n"
    "Team activity log: {workspace_path}/../agent_log.md\n"
    "  (Append important events here using log_changes tool)\n\n"
    "--- COMMUNICATION PROTOCOL ---\n"
    "Use these formats when messaging peers (soft recommendation, not strict):\n"
    "  TASK: [description] | OUTPUT: [where to write results] — assign a task\n"
    "  STATUS: [what you did] | BLOCKERS: [any] — progress update\n"
    "  RESULT: [output] | NEXT: [recommended action] — deliverable complete\n"
    "  HELP: [what you need] | CONTEXT: [background] — request assistance\n\n"
    "Always make it clear: what needs doing and where output should go.\n\n"
    "--- MEMORY USAGE ---\n"
    "Use the 'memory' tool to store important facts, decisions, and context:\n"
    "- Research findings and key data\n"
    "- Decisions made and their rationale\n"
    "- Configuration details, API endpoints, credentials\n"
    "- Reusable patterns or common code snippets\n"
    "When you learn something important, save it to memory immediately.\n"
    "When starting a task, search memory for relevant context first.\n\n"
    "--- PEERS ---\n"
    "Your agent name: {agent_name}\n"
    "Your team: {team_id}\n"
    "Peers you are linked to: {allowed_peers_list}\n"
    "You may ONLY send messages to agents in your linked peers list.\n\n"
    "--- TOOL GUIDANCE ---\n"
    "You have access to the full Hermes tool suite (web search, terminal, file ops, "
    "browser, todo, memory, code execution, etc.). Use tools relevant to the task.\n"
    "You do NOT need permission to:\n"
    "  - Read/write files in your workspace\n"
    "  - Run terminal commands in your workspace\n"
    "  - Search the web\n"
    "  - Delegate to linked peers\n"
    "You MUST ask a human for approval before:\n"
    "  - Actions affecting systems outside your workspace\n"
    "  - Irreversible operations (deleting critical files, pushing code, spending resources)\n"
)


# ---------------------------------------------------------------------------
# New config schema helpers
# ---------------------------------------------------------------------------
def _derive_workspace_path(team_id: str, agent_name: str) -> Path:
    """Return the on-disk workspace directory for an agent."""
    return WORKSPACE_ROOT / team_id / "workspace" / agent_name


def _get_team_workspace_path(team_id: str) -> Path:
    """Return the shared team workspace directory."""
    return WORKSPACE_ROOT / team_id / "workspace"


def write_agent_hermes_config(hermes_home: Path) -> None:
    """Write/merge the Hermes config.yaml under an agent's isolated HERMES_HOME.

    Enables and tunes the built-in ContextCompressor so long-running agents
    auto-compact instead of growing their conversation unbounded. Hermes reads
    {HERMES_HOME}/config.yaml at AIAgent init (cached on the file's mtime+size,
    keyed per path), so this must be written BEFORE the agent is constructed.

    Existing keys are preserved; only the compression-relevant sections are
    (re)written so the values stay in sync with the swarm constants above.
    """
    import yaml

    hermes_home.mkdir(parents=True, exist_ok=True)
    cfg_path = hermes_home / "config.yaml"

    existing: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception as e:
            log.warning("Could not parse existing %s (%s) — rewriting", cfg_path, e)

    model_section = existing.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
    model_section["context_length"] = AGENT_CONTEXT_WINDOW
    existing["model"] = model_section

    existing["compression"] = {
        "enabled": COMPRESSION_ENABLED,
        "threshold": COMPRESSION_THRESHOLD,
        "target_ratio": COMPRESSION_TARGET_RATIO,
        "protect_first_n": COMPRESSION_PROTECT_FIRST_N,
        "protect_last_n": COMPRESSION_PROTECT_LAST_N,
        "abort_on_summary_failure": False,
    }

    aux_section = existing.get("auxiliary")
    if not isinstance(aux_section, dict):
        aux_section = {}
    aux_comp = aux_section.get("compression")
    if not isinstance(aux_comp, dict):
        aux_comp = {}
    # The summarizer reuses the main model via the LiteLLM proxy (no override),
    # so it shares the same window; tell the feasibility check about it.
    aux_comp["context_length"] = AGENT_CONTEXT_WINDOW
    aux_section["compression"] = aux_comp
    existing["auxiliary"] = aux_section

    # Atomic write so a concurrent AIAgent init never reads a half-written file.
    fd, tmp_path = tempfile.mkstemp(dir=str(hermes_home), prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(existing, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cfg_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _build_org_diagram(cfg: Dict[str, Any], team_id: str, current_agent: str) -> str:
    """Build an ASCII org chart for all agents in the team."""
    team_agents = {
        name: a for name, a in cfg.get("agents", {}).items()
        if a.get("team_id") == team_id
    }
    if not team_agents:
        return "(no other team members)"

    lines = []
    for name, agent_cfg in sorted(team_agents.items()):
        peers = agent_cfg.get("allowed_peers", [])
        role = agent_cfg.get("role_soul", f"You are the {agent_cfg.get('name', name)}.").split('\n')[0]
        is_you = " ← YOU" if name == current_agent else ""
        peer_str = ", ".join(peers) if peers else "no links"
        lines.append(f"  {name}: {role[:60]}{is_you}")
        lines.append(f"    → links: {peer_str}")

    return "\n".join(lines)


def _build_team_members_list(cfg: Dict[str, Any], team_id: str) -> str:
    """Build a list of all team members with their roles."""
    team_agents = {
        name: a for name, a in cfg.get("agents", {}).items()
        if a.get("team_id") == team_id
    }
    if not team_agents:
        return "(no other team members)"

    lines = []
    for name, agent_cfg in sorted(team_agents.items()):
        role = agent_cfg.get("role_soul", f"You are the {agent_cfg.get('name', name)}.").split('\n')[0]
        lines.append(f"  - {name}: {role}")

    return "\n".join(lines)


def _migrate_legacy_config(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old flat agent config format -> new {teams, agents} format."""
    log.info("Migrating legacy flat agent config -> new teams schema")
    default_team_id = "default"
    migrated = {
        "teams": {
            default_team_id: {"name": "Default Team", "created_at": 0},
        },
        "agents": {},
    }
    for name, cfg in legacy.items():
        old_ws = cfg.get("workspace", name)
        new_path = _derive_workspace_path(default_team_id, name)
        old_path = DATA_ROOT / old_ws

        # Move existing disk data into new path
        if old_path.exists() and old_path != new_path:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                shutil.rmtree(new_path)
            shutil.move(str(old_path), str(new_path))
            log.info("  Moved %s -> %s", old_path, new_path)

        # v2 migration: extract role-specific part from old monolithic soul
        old_soul = cfg.get("soul", "")
        # If old soul looks like the old default template, extract just the identity line
        if "You are the" in old_soul and "send_peer_message" in old_soul:
            # Extract the first sentence as role identity
            lines = old_soul.split('\n')
            role_lines = []
            for line in lines:
                if line.startswith("You are the") or line.startswith("You are a"):
                    role_lines.append(line)
                elif role_lines and line.strip() and not any(x in line for x in ["send_peer_message", "ask_human", "operate autonomously"]):
                    role_lines.append(line)
            role_soul = '\n'.join(role_lines) if role_lines else old_soul.split('.')[0] + "."
        else:
            role_soul = old_soul

        migrated["agents"][name] = {
            "team_id": default_team_id,
            "name": cfg.get("name", name.capitalize() + " Agent"),
            "session_id": cfg.get("session_id", f"{name}-master-session-v1"),
            "allowed_peers": [],
            "role_soul": role_soul,
        }
    return migrated


def _migrate_v1_to_v2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate v1 config (has 'soul' field) to v2 config (has 'role_soul' field)."""
    migrated_any = False
    for name, agent_cfg in cfg.get("agents", {}).items():
        if "soul" in agent_cfg and "role_soul" not in agent_cfg:
            old_soul = agent_cfg.pop("soul")
            # Extract role-specific part from old monolithic soul
            lines = old_soul.split('\n')
            role_lines = []
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if any(x in line_stripped for x in ["send_peer_message", "ask_human", "operate autonomously",
                                                      "swarm rules", "communication protocol", "tool guidance",
                                                      "You are an autonomous agent", "NEVER end your turn"]):
                    continue
                role_lines.append(line_stripped)
            agent_cfg["role_soul"] = '\n'.join(role_lines) if role_lines else f"You are the {agent_cfg.get('name', name)}."
            migrated_any = True
    if migrated_any:
        log.info("Migrated v1 config (soul field) -> v2 config (role_soul field)")
        _save_full_config(cfg)
    return cfg


def _deep_copy_config(src: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(src))


def _default_config() -> Dict[str, Any]:
    default_team_id = "default"
    return {
        "teams": {
            default_team_id: {"name": "Default Team", "created_at": int(time.time())},
        },
        "agents": {},
    }


# ---------------------------------------------------------------------------
# Soul composition — combines common + role-specific parts
# ---------------------------------------------------------------------------
def compose_agent_soul(agent_cfg: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None) -> str:
    """Build the full ephemeral system prompt for an agent."""
    agent_name = agent_cfg.get("name", "Agent")
    team_id = agent_cfg.get("team_id", "default")
    peers = agent_cfg.get("allowed_peers", [])
    peers_str = ", ".join(peers) if peers else "(none — you cannot message any peers yet)"
    workspace_path = str(_derive_workspace_path(team_id, agent_cfg.get("agent_id", "unknown")))

    # Build org diagram if we have full config
    org_diagram = "(config not available for diagram)"
    all_members = "(config not available for member list)"
    if full_config:
        org_diagram = _build_org_diagram(full_config, team_id, agent_cfg.get("agent_id", "unknown"))
        all_members = _build_team_members_list(full_config, team_id)

    common = COMMON_SOUL_TEMPLATE.format(
        agent_name=agent_name,
        team_id=team_id,
        allowed_peers_list=peers_str,
        sweep_interval=SWEEP_INTERVAL_SECONDS,
        workspace_path=workspace_path,
    )

    role = agent_cfg.get("role_soul", f"You are the {agent_name}.")

    return (
        f"{common}\n"
        f"--- TEAM ORGANIZATION ---\n"
        f"Your team: {team_id}\n\n"
        f"Agent connections and roles:\n"
        f"{org_diagram}\n\n"
        f"All team members:\n"
        f"{all_members}\n\n"
        f"{'=' * 60}\n"
        f"YOUR ROLE\n"
        f"{'=' * 60}\n"
        f"{role}\n"
    )


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------
def load_agents_config() -> Dict[str, Any]:
    global _config_cache, _config_cache_key
    with _config_lock:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        if not AGENTS_CONFIG_PATH.exists():
            default_cfg = _default_config()
            _save_full_config(default_cfg)  # populates the cache
            return _deep_copy_config(default_cfg)

        # Fast path: file unchanged since we last parsed it.
        key = _config_file_key()
        if key is not None and key == _config_cache_key and _config_cache is not None:
            return _deep_copy_config(_config_cache)

        try:
            with open(AGENTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            log.error("Failed to load agents config: %s. Returning default.", e)
            return _default_config()

        # Detect legacy flat format (has string keys at top level with agent dict values)
        if "teams" not in raw and "agents" not in raw:
            migrated = _migrate_legacy_config(raw)
            _save_full_config(migrated)  # refreshes cache + key
            return _deep_copy_config(migrated)

        # Ensure both keys exist even if someone corrupted the file
        if "teams" not in raw:
            raw["teams"] = {}
        if "agents" not in raw:
            raw["agents"] = {}

        # Migrate v1 -> v2 if needed (writes file + refreshes cache when it changes)
        raw = _migrate_v1_to_v2(raw)

        # Cache the parsed result keyed on the file's current stat.
        _config_cache = _deep_copy_config(raw)
        _config_cache_key = _config_file_key()
        return _deep_copy_config(raw)


def _save_full_config(cfg: Dict[str, Any]) -> None:
    """Atomically persist the full config.

    Writes to a temp file in the same directory and os.replace()s it into
    place so a concurrent reader (or a crash mid-write) can never observe a
    half-written / truncated agents_config.json. The lock serializes writers
    so two concurrent saves cannot interleave.
    """
    with _config_lock:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(DATA_ROOT), prefix=".agents_config.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, AGENTS_CONFIG_PATH)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        # Keep the read cache hot and consistent with what we just wrote.
        global _config_cache, _config_cache_key
        _config_cache = _deep_copy_config(cfg)
        _config_cache_key = _config_file_key()


def save_agent_config(agent_name: str, cfg: Dict[str, Any]) -> None:
    full = load_agents_config()
    full["agents"][agent_name] = cfg
    _save_full_config(full)


def save_all_config(cfg: Dict[str, Any]) -> None:
    _save_full_config(cfg)


# ---------------------------------------------------------------------------
# Team CRUD
# ---------------------------------------------------------------------------
def create_team(cfg: Dict[str, Any], team_id: str, name: str) -> Dict[str, Any]:
    if team_id in cfg["teams"]:
        raise ValueError(f"Team '{team_id}' already exists")
    cfg["teams"][team_id] = {"name": name, "created_at": int(time.time())}
    # Ensure workspace directory exists
    team_ws = WORKSPACE_ROOT / team_id / "workspace"
    team_ws.mkdir(parents=True, exist_ok=True)
    # Create shared workspace.md
    workspace_md = team_ws / "workspace.md"
    if not workspace_md.exists():
        workspace_md.write_text(
            f"# Project: {name}\n\n"
            "## Description\n"
            "Describe the project here...\n\n"
            "## Key Decisions\n"
            "- Decision 1\n\n"
            "## Active Tasks\n"
            "- [PENDING] Task description (agent_name)\n\n"
            "## Shared Files\n"
            "- path/to/file: description\n",
            encoding="utf-8",
        )
    # Create shared agent_log.md
    agent_log = team_ws / "agent_log.md"
    if not agent_log.exists():
        agent_log.write_text(
            "# Team Activity Log\n\n"
            "Format: `[YYYY-MM-DD HH:MM:SS] agent_name: message`\n\n",
            encoding="utf-8",
        )
    _save_full_config(cfg)
    return cfg["teams"][team_id]


def delete_team(cfg: Dict[str, Any], team_id: str) -> bool:
    if team_id not in cfg["teams"]:
        return False
    # Remove every agent in this team
    agents_to_remove = [
        name for name, a in cfg["agents"].items() if a.get("team_id") == team_id
    ]
    for name in agents_to_remove:
        del cfg["agents"][name]
    del cfg["teams"][team_id]
    # Nuke disk workspace
    team_dir = WORKSPACE_ROOT / team_id
    if team_dir.exists():
        shutil.rmtree(team_dir)
    _save_full_config(cfg)
    return True


def list_teams(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"id": tid, "name": t["name"], "agent_count": sum(
            1 for a in cfg["agents"].values() if a.get("team_id") == tid
        )}
        for tid, t in cfg["teams"].items()
    ]


# ---------------------------------------------------------------------------
# Agent CRUD
# ---------------------------------------------------------------------------
def create_agent(
    cfg: Dict[str, Any],
    name: str,
    team_id: str,
    display_name: str,
    allowed_peers: Optional[List[str]] = None,
    role_soul: Optional[str] = None,
) -> Dict[str, Any]:
    if name in cfg["agents"]:
        raise ValueError(f"Agent '{name}' already exists")
    if team_id not in cfg["teams"]:
        raise ValueError(f"Team '{team_id}' does not exist")

    agent_cfg = {
        "team_id": team_id,
        "name": display_name,
        "session_id": f"{name}-master-session-v1",
        "allowed_peers": list(allowed_peers or []),
        "role_soul": role_soul or f"You are the {display_name}.",
    }
    cfg["agents"][name] = agent_cfg
    # Prepare workspace dirs
    ws = _derive_workspace_path(team_id, name)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "context").mkdir(exist_ok=True)
    # Create agent-specific log
    agent_log = ws / "agent_log.md"
    if not agent_log.exists():
        agent_log.write_text(
            f"# {display_name} Activity Log\n\n",
            encoding="utf-8",
        )
    _save_full_config(cfg)
    return agent_cfg


def delete_agent(cfg: Dict[str, Any], name: str) -> bool:
    if name not in cfg["agents"]:
        return False
    team_id = cfg["agents"][name].get("team_id", "default")
    del cfg["agents"][name]

    # Remove workspace data permanently
    ws = _derive_workspace_path(team_id, name)
    if ws.exists():
        shutil.rmtree(ws)

    # Also prune this agent from every other agent's allowed_peers list
    for a_cfg in cfg["agents"].values():
        if name in a_cfg.get("allowed_peers", []):
            a_cfg["allowed_peers"].remove(name)

    _save_full_config(cfg)
    return True


def set_agent_peers(cfg: Dict[str, Any], name: str, peers: List[str]) -> Dict[str, Any]:
    if name not in cfg["agents"]:
        raise ValueError(f"Agent '{name}' not found")
    name_team = cfg["agents"][name].get("team_id")
    for peer in peers:
        if peer not in cfg["agents"]:
            raise ValueError(f"Peer agent '{peer}' not found")
        peer_team = cfg["agents"][peer].get("team_id")
        if name_team and peer_team and name_team != peer_team:
            raise ValueError(f"Cross-team peer links are blocked: '{name}' (team={name_team}) → '{peer}' (team={peer_team})")
    cfg["agents"][name]["allowed_peers"] = list(peers)
    _save_full_config(cfg)
    return cfg["agents"][name]


def add_agent_peer(cfg: Dict[str, Any], name: str, peer: str) -> Dict[str, Any]:
    """Add a peer link. Links are ALWAYS bidirectional."""
    if name not in cfg["agents"]:
        raise ValueError(f"Agent '{name}' not found")
    if peer not in cfg["agents"]:
        raise ValueError(f"Peer agent '{peer}' not found")
    name_team = cfg["agents"][name].get("team_id")
    peer_team = cfg["agents"][peer].get("team_id")
    if name_team and peer_team and name_team != peer_team:
        raise ValueError(f"Cross-team peer links are blocked: '{name}' (team={name_team}) → '{peer}' (team={peer_team})")

    # Add name -> peer
    peers = cfg["agents"][name].get("allowed_peers", [])
    if peer not in peers:
        peers.append(peer)
        cfg["agents"][name]["allowed_peers"] = peers

    # Add peer -> name (bidirectional)
    peer_peers = cfg["agents"][peer].get("allowed_peers", [])
    if name not in peer_peers:
        peer_peers.append(name)
        cfg["agents"][peer]["allowed_peers"] = peer_peers

    _save_full_config(cfg)
    return cfg["agents"][name]


def remove_agent_peer(cfg: Dict[str, Any], name: str, peer: str) -> Dict[str, Any]:
    """Remove a peer link. Links are ALWAYS bidirectional."""
    if name not in cfg["agents"]:
        raise ValueError(f"Agent '{name}' not found")

    # Remove name -> peer
    peers = cfg["agents"][name].get("allowed_peers", [])
    if peer in peers:
        peers.remove(peer)
        cfg["agents"][name]["allowed_peers"] = peers

    # Remove peer -> name (bidirectional)
    if peer in cfg["agents"]:
        peer_peers = cfg["agents"][peer].get("allowed_peers", [])
        if name in peer_peers:
            peer_peers.remove(name)
            cfg["agents"][peer]["allowed_peers"] = peer_peers

    _save_full_config(cfg)
    return cfg["agents"][name]


# ---------------------------------------------------------------------------
# Team isolation helpers
# ---------------------------------------------------------------------------
def get_agent_team(cfg: Dict[str, Any], name: str) -> Optional[str]:
    agent = cfg["agents"].get(name)
    return agent["team_id"] if agent else None


def get_team_agents(cfg: Dict[str, Any], team_id: str) -> Dict[str, Any]:
    return {name: a for name, a in cfg["agents"].items() if a.get("team_id") == team_id}


def peer_allowed(cfg: Dict[str, Any], caller: str, target: str) -> bool:
    """Return True if caller is explicitly linked to target AND same team."""
    caller_cfg = cfg["agents"].get(caller)
    target_cfg = cfg["agents"].get(target)
    if not caller_cfg or not target_cfg:
        return False
    if caller_cfg.get("team_id") != target_cfg.get("team_id"):
        return False
    return target in caller_cfg.get("allowed_peers", [])


# Initial load
AGENTS = load_agents_config()
