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


# Web search backend for all agents. "ddgs" = DuckDuckGo via the `ddgs` Python
# package — zero API key, near-zero RAM (SearXNG was tried but its multi-engine
# aggregator OOM-crashed the host). The ddgs Hermes plugin auto-registers when
# the package is importable; we ALSO pin it explicitly in each agent's
# config.yaml (web.search_backend) so resolution never silently falls through to
# an unconfigured paid backend.
WEB_SEARCH_BACKEND = "ddgs"


def _ensure_full_path() -> None:
    """Guarantee a complete PATH for tool subprocesses spawned by Hermes.

    Hermes' file/search tools shell out to `rg`/`grep`/`find` and locate them via
    `shutil.which`, which reads os.environ["PATH"]. When the server is launched
    from a context with a stripped PATH (e.g. a bare GUI/launchd invocation), that
    lookup fails and `search_files` reports "requires ripgrep (rg) or grep" even
    though the binaries are installed — observed in the agent transcripts, where
    every search_files call died and agents wasted turns falling back to a login
    shell. We prepend the standard bin dirs (idempotently) so spawned subprocesses
    inherit a usable PATH regardless of how the server was started.
    """
    standard_dirs = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    seen = set(parts)
    missing = [d for d in standard_dirs if d not in seen and os.path.isdir(d)]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + parts) if parts else os.pathsep.join(missing)
        log.info("PATH augmented with %s", ", ".join(missing))

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
AGENT_CONTEXT_WINDOW = 256000
COMPRESSION_THRESHOLD = 0.75          # compact at ~75% of the window
COMPRESSION_TARGET_RATIO = 0.20      # summary budget as a fraction of compacted content
COMPRESSION_PROTECT_FIRST_N = 3      # head turns kept verbatim (besides system prompt)
COMPRESSION_PROTECT_LAST_N = 12      # recent tail turns kept verbatim
# Hard backstop: force compaction once a session exceeds this many messages,
# even if the token estimate hasn't crossed the threshold. Stops a session from
# accumulating hundreds of small turns that never trip the token check.
COMPRESSION_HYGIENE_HARD_MESSAGE_LIMIT = 400

# ---------------------------------------------------------------------------
# Autonomous 24/7 operation
# ---------------------------------------------------------------------------
# The swarm is task-queue-driven: an agent wakes on a task, processes it, then
# goes idle. That means a team goes DORMANT after finishing a brief. To run
# 24/7, mark a coordinator agent with cfg["autonomous"]=True: its daemon then
# self-injects a "continue the mission" task whenever it has been idle (empty
# queue, not busy) for AUTONOMOUS_HEARTBEAT_SECONDS. The coordinator reviews the
# mission + what's already done and delegates the next high-value increment.
# Specialists stay reactive (autonomous=False) so the team doesn't spin N
# independent loops — one driver pulls the whole team.
#
# COST WARNING: each heartbeat cycle consumes tokens (the last full GTM cycle
# was ~4.8M tokens). Tune the interval to your budget. Pause by setting the
# agent's autonomous flag False, stopping its daemon, or stopping the server.
# Override the interval at launch with SWARM_HEARTBEAT_SECONDS.
AUTONOMOUS_HEARTBEAT_SECONDS = int(os.environ.get("SWARM_HEARTBEAT_SECONDS", "1800"))

AUTONOMOUS_HEARTBEAT_PROMPT = (
    "[AUTONOMOUS HEARTBEAT — no human task is queued; you run 24/7]\n"
    "Your goal THIS cycle is to ship something LIVE — not to write another plan "
    "or refine an existing draft. Success = what went public this cycle.\n\n"
    "1. PUBLISH FIRST: what content/assets are already DRAFTED but not yet posted "
    "or sent? (skim the changelog ../agent_log.md and the specialists' outputs/). "
    "Unpublished work is your #1 priority — get it LIVE via the browser: have the "
    "right specialist post it to LinkedIn/Instagram/X/Facebook, send the email, or "
    "publish the blog post.\n"
    "2. IF BLOCKED ON ACCESS: if posting needs a login/account you don't have, "
    "call ask_human for that exact credential and stop — do not just re-draft. "
    "Once the human provides it, the session persists and you publish from then on.\n"
    "3. ONLY WHEN THE PIPELINE IS FLOWING (nothing valuable sits unpublished) do "
    "you create the next NEW asset — then publish that too.\n"
    "Delegate concrete PUBLISH-and-report tasks to specialists, and log_changes "
    "recording specifically WHAT WENT LIVE this cycle (URLs if available). Do not "
    "repeat already-published work."
)

# ---------------------------------------------------------------------------
# Tool output caps — bound a single tool result's size in the conversation.
# A browser DOM snapshot or a big file read can be hundreds of KB; with 12
# protected tail turns, a few of those alone can blow the post-compaction
# budget and trigger a compaction cascade (observed: the CMO compacted 8x with
# 4 zero-message cascade sessions). Capping tool output keeps the protected
# tail small enough that one compaction pass actually fits under the threshold.
# ---------------------------------------------------------------------------
TOOL_OUTPUT_MAX_BYTES = 16000
TOOL_OUTPUT_MAX_LINES = 400
TOOL_OUTPUT_MAX_LINE_LENGTH = 2000

# ---------------------------------------------------------------------------
# Disabled toolsets — removed from every agent's tool schema at init.
# delegate_task (the "delegation" toolset) is intentionally LEFT ENABLED: agents
# may spawn Hermes sub-agents for parallel subtasks. The old browser-collision
# problem (concurrent sub-agents driving one shared tab) is fixed instead by
# browser.fresh_tab_per_task (see write_agent_hermes_config) — each sub-agent's
# unique task_id now gets its own tab in the shared Chrome. Keep this list as
# the hook for disabling toolsets in the future.
# ---------------------------------------------------------------------------
DISABLED_TOOLSETS: List[str] = []

# ---------------------------------------------------------------------------
# Queue / sweep behavior
# ---------------------------------------------------------------------------
MAX_BATCH_SIZE = 10          # max tasks pulled into one LLM turn (backpressure)
MAX_TASK_RETRIES = 3         # batch failures requeue up to N times, then -> 'failed' (DLQ)
LLM_ERROR_EMIT_THROTTLE_SECONDS = 60  # min gap between "provider unreachable" UI errors per agent

# ---------------------------------------------------------------------------
# Browser tools
# ---------------------------------------------------------------------------
# Hermes gates the browser toolset (browser_* + web_search) behind
# check_browser_requirements(): it needs EITHER a configured cloud provider OR
# a local Chromium. The swarm server doesn't load ~/.hermes/.env, so the
# auto-selected cloud provider (Firecrawl) reports unconfigured and short-
# circuits the check to False — dropping every browser tool AND web_search.
# Pinning cloud_provider="local" forces _get_cloud_provider() to return None so
# the check uses the local Chromium instead (install once with
# `npx playwright install chromium`). Set to "" to leave provider auto-detect
# alone (cloud mode, needs creds in the server env).
BROWSER_CLOUD_PROVIDER = "local"

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
# COMMON_SOUL_TEMPLATE lives in swarm_server/soul_template.py so the (long)
# shared prompt prose can be edited independently of config logic.
from swarm_server.soul_template import COMMON_SOUL_TEMPLATE  # noqa: E402


# ---------------------------------------------------------------------------
# New config schema helpers
# ---------------------------------------------------------------------------
def _derive_workspace_path(team_id: str, agent_name: str) -> Path:
    """Return the on-disk workspace directory for an agent."""
    return WORKSPACE_ROOT / team_id / "workspace" / agent_name


def _get_team_workspace_path(team_id: str) -> Path:
    """Return the shared team workspace directory."""
    return WORKSPACE_ROOT / team_id / "workspace"


def write_agent_hermes_config(hermes_home: Path, cdp_url: Optional[str] = None) -> None:
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

    # Pin the model to the LiteLLM proxy. Beyond context_length, we set
    # default/provider/base_url/api_key so Hermes' auxiliary client can resolve
    # "the main model" (auxiliary_client._read_main_model + _resolve_custom_
    # runtime) to this proxy instead of falling back to "gpt-4o-mini" (which the
    # proxy doesn't serve) or to unauthenticated openrouter/nous.
    model_section = existing.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
    model_section["context_length"] = AGENT_CONTEXT_WINDOW
    model_section["default"] = "litellm-model"
    model_section["provider"] = "custom"
    model_section["base_url"] = LITELLM_API_BASE
    model_section["api_key"] = "sk-1234"
    existing["model"] = model_section

    existing["compression"] = {
        "enabled": COMPRESSION_ENABLED,
        "threshold": COMPRESSION_THRESHOLD,
        "target_ratio": COMPRESSION_TARGET_RATIO,
        "protect_first_n": COMPRESSION_PROTECT_FIRST_N,
        "protect_last_n": COMPRESSION_PROTECT_LAST_N,
        "hygiene_hard_message_limit": COMPRESSION_HYGIENE_HARD_MESSAGE_LIMIT,
        "abort_on_summary_failure": False,
    }

    # Route ALL auxiliary tasks (compaction summarizer, title-gen, web-extract,
    # session search, etc.) through the LiteLLM proxy. Without this, provider
    # "auto" tries openrouter then nous — both unauthenticated here — so the
    # compaction SUMMARY call fails and Hermes re-attempts compaction in a tight
    # loop (the 4 zero-message cascade sessions seen in the CMO's history).
    # Setting base_url forces provider=custom and uses our model directly
    # (auxiliary_client._resolve_aux_provider_and_model: base_url present =>
    # provider forced to "custom"; per-task config wins over "auto").
    aux_endpoint = {
        "provider": "custom",
        "model": "litellm-model",
        "base_url": LITELLM_API_BASE,
        "api_key": "sk-1234",
    }
    aux_section = existing.get("auxiliary")
    if not isinstance(aux_section, dict):
        aux_section = {}
    for _task in (
        "compression", "title_generation", "web_extract",
        "session_search", "triage_specifier", "curator", "approval",
    ):
        task_cfg = aux_section.get(_task)
        if not isinstance(task_cfg, dict):
            task_cfg = {}
        task_cfg.update(aux_endpoint)
        aux_section[_task] = task_cfg
    # The summarizer shares the main window; tell its feasibility check.
    aux_section["compression"]["context_length"] = AGENT_CONTEXT_WINDOW
    existing["auxiliary"] = aux_section

    # Pin ddgs (DuckDuckGo, no API key) as the web search backend so web_search
    # is always available and never silently falls through to an unconfigured
    # paid backend. Requires the `ddgs` package installed in the server's venv.
    web_section = existing.get("web")
    if not isinstance(web_section, dict):
        web_section = {}
    web_section["search_backend"] = WEB_SEARCH_BACKEND
    existing["web"] = web_section

    # Apply the disabled-toolsets list (currently empty — delegate_task stays
    # enabled; see DISABLED_TOOLSETS). Merge-safe with Hermes-seeded agent.* keys.
    agent_section = existing.get("agent")
    if not isinstance(agent_section, dict):
        agent_section = {}
    agent_section["disabled_toolsets"] = list(DISABLED_TOOLSETS)
    existing["agent"] = agent_section

    # Cap individual tool-result size so a giant browser snapshot / file read
    # can't bloat the protected tail and trigger a compaction cascade.
    existing["tool_output"] = {
        "max_bytes": TOOL_OUTPUT_MAX_BYTES,
        "max_lines": TOOL_OUTPUT_MAX_LINES,
        "max_line_length": TOOL_OUTPUT_MAX_LINE_LENGTH,
    }

    # Force local browser mode so the browser toolset (+ web_search) is enabled
    # via the locally-installed Chromium instead of an unconfigured cloud
    # provider. Merge-safe: other browser keys (timeouts, engine) are preserved.
    if BROWSER_CLOUD_PROVIDER or cdp_url:
        browser_section = existing.get("browser")
        if not isinstance(browser_section, dict):
            browser_section = {}
        if BROWSER_CLOUD_PROVIDER:
            browser_section["cloud_provider"] = BROWSER_CLOUD_PROVIDER
        # A per-team CDP endpoint makes every agent in the team share ONE
        # persistent Chrome (same cookies/logins, durable across restarts).
        # cdp_url takes precedence over cloud_provider/local in Hermes, so the
        # local setting above just stays as a graceful fallback. Pass "" to
        # clear a previously-written endpoint.
        if cdp_url is not None:
            browser_section["cdp_url"] = cdp_url
        # Each task_id (the top-level agent AND every delegate_task sub-agent)
        # gets its OWN tab in the shared Chrome instead of all adopting the
        # first existing page. This is what lets sub-agents browse concurrently
        # without hijacking each other's navigation, while still sharing the
        # team's cookies/logins (one browser, many tabs). Honored by the patched
        # browser_supervisor._attach_initial_page.
        browser_section["fresh_tab_per_task"] = True
        existing["browser"] = browser_section

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
def compose_soul_identity(agent_cfg: Dict[str, Any]) -> str:
    """Build the SOUL.md identity block for an agent.

    This becomes the agent's PRIMARY identity, written to {HERMES_HOME}/SOUL.md
    so Hermes loads it as the lead block of the (cached) system prompt — instead
    of the generic auto-seeded "You are Hermes Agent…" template. The richer
    operational framing (swarm rules, org chart, peers) stays in the ephemeral
    prompt via compose_agent_soul(..., include_role=False); the role lives here
    so it is the first thing the model reads and is cache-stable across turns.
    """
    agent_name = agent_cfg.get("name", "Agent")
    agent_id = agent_cfg.get("agent_id", "unknown")
    team_id = agent_cfg.get("team_id", "default")
    role = agent_cfg.get("role_soul", f"You are the {agent_name}.")
    return (
        f"{role}\n\n"
        f'You are "{agent_id}" ({agent_name}), one agent on the "{team_id}" team in an '
        f"autonomous multi-agent swarm. Operate strictly in the role described above."
    )


def _read_workspace_brief(team_id: str, max_chars: int = 8000) -> str:
    """Return the team's workspace.md text for inlining into the prompt.

    Truncated defensively so an oversized brief can't blow up every turn's
    token budget. Returns a friendly placeholder if the file is absent."""
    try:
        p = _get_team_workspace_path(team_id) / "workspace.md"
        if not p.exists():
            return "(no workspace.md yet — the team brief has not been written.)"
        text = p.read_text(encoding="utf-8").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…(brief truncated — read workspace.md on disk for the rest.)"
        return text or "(workspace.md is empty.)"
    except Exception as e:
        return f"(could not read workspace.md: {e})"


def _build_workspace_tree(team_id: str, max_entries: int = 160) -> str:
    """A compact directory listing of the team workspace, surfacing every
    agent's outputs/ files so each agent can SEE what exists team-wide without
    guessing paths. Skips noise (.git, .hermes, caches, browser profile) and
    caps total lines so a large repo copy can't dominate the prompt."""
    import os
    root = _get_team_workspace_path(team_id)
    if not root.exists():
        return "(team workspace not created yet.)"
    SKIP = {".git", ".hermes", "__pycache__", "node_modules", ".browser-profile",
            "context", ".DS_Store", "dist", "build", ".venv"}
    lines: List[str] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 4:
            dirnames[:] = []
            continue
        indent = "  " * depth
        label = "." if rel == "." else os.path.basename(dirpath)
        lines.append(f"{indent}{label}/")
        for fn in sorted(filenames):
            if fn.endswith((".pyc", ".db", ".db-shm", ".db-wal")) or fn == ".DS_Store":
                continue
            lines.append(f"{indent}  {fn}")
            if len(lines) >= max_entries:
                truncated = True
                break
        if truncated:
            break
    if truncated:
        lines.append("…(tree truncated)")
    return "\n".join(lines)


def _recent_peer_messages(team_id: str, full_config: Optional[Dict[str, Any]], limit: int = 10) -> str:
    """Render the last N send_peer_message events across the team, oldest→newest,
    so every agent has shared awareness of recent team chatter."""
    try:
        from swarm_server.monitoring import monitor_db
        # message_sent events aren't team-tagged, so scope by the team's agent set.
        team_agents = set()
        if full_config:
            team_agents = {
                aid for aid, a in (full_config.get("agents") or {}).items()
                if a.get("team_id") == team_id
            }
        events = monitor_db.get_events(limit=200)  # newest first
        msgs = []
        for e in events:
            if e.get("event_type") != "message_sent":
                continue
            frm, to = e.get("from_agent"), e.get("to_agent")
            if team_agents and frm not in team_agents and to not in team_agents:
                continue
            preview = ""
            try:
                preview = (json.loads(e.get("data") or "{}") or {}).get("message_preview", "")
            except Exception:
                pass
            msgs.append((e.get("timestamp", 0), frm, to, preview))
            if len(msgs) >= limit:
                break
        if not msgs:
            return "(no peer messages yet.)"
        msgs.reverse()  # oldest → newest reads naturally
        out = []
        for ts, frm, to, preview in msgs:
            stamp = ""
            try:
                from datetime import datetime
                stamp = datetime.fromtimestamp(ts).strftime("%H:%M")
            except Exception:
                pass
            out.append(f"  [{stamp}] {frm} → {to}: {preview}")
        return "\n".join(out)
    except Exception as e:
        return f"(could not load peer messages: {e})"


def compose_live_context(
    team_id: str,
    agent_id: str,
    full_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Dynamic, per-turn context appended to the ephemeral system prompt:
    the live project directory tree and recent team messages. Rebuilt each
    turn (cheap) and injected at API-call time, so it never pollutes the
    cached/stored system prompt."""
    tree = _build_workspace_tree(team_id)
    recent = _recent_peer_messages(team_id, full_config, limit=10)
    try:
        from datetime import datetime
        now_line = datetime.now().astimezone().strftime("%A, %B %d, %Y %H:%M %Z")
    except Exception:
        now_line = "(unavailable)"
    return (
        "--- LIVE TEAM CONTEXT (auto-refreshed each turn) ---\n"
        f"Current time: {now_line}\n\n"
        "Project directory structure (team workspace — every agent's outputs/ "
        "are visible here, so you can see what already exists):\n"
        f"{tree}\n\n"
        "Last 10 messages between teammates (send_peer_message), oldest first:\n"
        f"{recent}\n"
    )


def compose_agent_soul(
    agent_cfg: Dict[str, Any],
    full_config: Optional[Dict[str, Any]] = None,
    include_role: bool = True,
) -> str:
    """Build the full ephemeral system prompt for an agent.

    When include_role is False, the trailing "YOUR ROLE" block is omitted —
    used when the role identity is instead written to SOUL.md (so it is not
    duplicated in both the cached prompt and the ephemeral prompt).
    """
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

    # Inline the project brief (workspace.md) so it lives in the prompt itself
    # rather than something the agent must remember to open. Read at compose time.
    workspace_brief = _read_workspace_brief(team_id)

    common = COMMON_SOUL_TEMPLATE.format(
        agent_name=agent_name,
        team_id=team_id,
        allowed_peers_list=peers_str,
        sweep_interval=SWEEP_INTERVAL_SECONDS,
        workspace_path=workspace_path,
        workspace_brief=workspace_brief,
    )

    body = (
        f"{common}\n"
        f"--- TEAM ORGANIZATION ---\n"
        f"Your team: {team_id}\n\n"
        f"Agent connections and roles:\n"
        f"{org_diagram}\n\n"
        f"All team members:\n"
        f"{all_members}\n"
    )

    if include_role:
        role = agent_cfg.get("role_soul", f"You are the {agent_name}.")
        body += (
            f"\n{'=' * 60}\n"
            f"YOUR ROLE\n"
            f"{'=' * 60}\n"
            f"{role}\n"
        )

    return body


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


# Ensure tool subprocesses (rg/grep/find) always have a usable PATH, no matter
# how the server process was launched. Must run before any agent spawns tools.
_ensure_full_path()

# Initial load
AGENTS = load_agents_config()
