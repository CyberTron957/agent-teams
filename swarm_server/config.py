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


def ensure_hermes_importable() -> None:
    """Make the Hermes agent package importable, however it was obtained.

    Resolution order (first that works wins), so the same code runs whether
    Hermes is pip-installed, on PYTHONPATH, or a source checkout:
      1. Already importable (pip ``hermes-agent`` or PYTHONPATH) — do nothing.
      2. ``HERMES_AGENT_PATH`` env var pointing at a source checkout.
      3. The conventional ``~/.hermes/hermes-agent`` source location.
    If none resolve, the later ``import run_agent`` raises with a clear hint.
    Idempotent and cheap (the success path is a single import attempt).
    """
    import sys as _sys
    try:
        import run_agent  # noqa: F401  (probe: installed or already on path)
        return
    except Exception:
        pass
    candidates = []
    env_path = os.environ.get("HERMES_AGENT_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.home() / ".hermes" / "hermes-agent")
    for cand in candidates:
        try:
            if cand and (cand / "run_agent.py").exists():
                if str(cand) not in _sys.path:
                    _sys.path.insert(0, str(cand))
                return
        except OSError:
            continue
    log.warning(
        "Hermes agent not found. Install it (`pip install hermes-agent`) or set "
        "HERMES_AGENT_PATH to a hermes-agent checkout."
    )


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
# Paths — resolved so the same code works as a source checkout, a pip install,
# and inside Docker (where data lives on a mounted volume).
# ---------------------------------------------------------------------------
import sys as _sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_source_checkout() -> bool:
    """True when running from the repo (vs installed into site-packages)."""
    return any((PROJECT_ROOT / m).exists() for m in ("pyproject.toml", ".git", "dashboard"))


def _resolve_data_root() -> Path:
    """Writable state dir (configs, queues, workspaces, monitoring db).

    SWARM_DATA_DIR wins (Docker sets it to a mounted volume). Otherwise the repo's
    ./data in a source checkout, else ~/.hermes-swarm/data for a pip install (never
    write inside site-packages)."""
    env = os.environ.get("SWARM_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if _is_source_checkout():
        return PROJECT_ROOT / "data"
    return Path.home() / ".hermes-swarm" / "data"


def _resolve_dashboard_dir() -> Path:
    """Locate the static dashboard across source / in-package / pip data-files."""
    env = os.environ.get("SWARM_DASHBOARD_DIR")
    if env:
        return Path(env).expanduser()
    for cand in (
        PROJECT_ROOT / "dashboard",                                  # source / Docker
        Path(__file__).resolve().parent / "dashboard",              # bundled in-package
        Path(_sys.prefix) / "share" / "hermes-swarm" / "dashboard",  # pip data-files
    ):
        if (cand / "index.html").exists():
            return cand
    return PROJECT_ROOT / "dashboard"


DATA_ROOT = _resolve_data_root()
AGENTS_CONFIG_PATH = DATA_ROOT / "agents_config.json"
MONITORING_DB = DATA_ROOT / "monitoring.db"
DASHBOARD_DIR = _resolve_dashboard_dir()

WORKSPACE_ROOT = DATA_ROOT / "teams"

# ---------------------------------------------------------------------------
# Network / Runtime  (all env-overridable so the same code runs locally, in a
# pip install, and in Docker without edits)
# ---------------------------------------------------------------------------
# Dashboard bind address. Stays 127.0.0.1 locally for safety; Docker sets
# SWARM_HOST=0.0.0.0. Bind a public interface only with SWARM_API_KEY set.
SERVER_HOST = os.environ.get("SWARM_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SWARM_PORT", "8000"))

# LLM backend (OpenAI-compatible). Defaults to the local LiteLLM proxy for
# back-compat; new users point SWARM_LLM_BASE_URL at any OpenAI-compatible
# endpoint (their own proxy, OpenRouter, etc.) and supply the matching key.
LITELLM_API_BASE = os.environ.get("SWARM_LLM_BASE_URL", "http://127.0.0.1:4000/v1")
LLM_API_KEY = os.environ.get("SWARM_LLM_API_KEY", "sk-1234")
SWEEP_INTERVAL_SECONDS = int(os.environ.get("SWARM_SWEEP_INTERVAL", "10"))

# Model the backend serves by default. Per-agent overrides (cfg["model"]) fall
# back to this. The fallback list is what the model dropdown shows if the
# backend can't be queried.
DEFAULT_MODEL = os.environ.get("SWARM_DEFAULT_MODEL", "litellm-model")
AVAILABLE_MODELS_FALLBACK = [m.strip() for m in os.environ.get(
    "SWARM_FALLBACK_MODELS", "litellm-model,kimi").split(",") if m.strip()]

# Model used ONLY for browser_vision (screenshot reading). The main agent model
# may be text-only (DeepSeek/Kimi), so vision is pinned to a multimodal model the
# proxy serves. gpt-5.4-nano is verified to accept image_url input via the proxy;
# gpt-5.4-mini is not deployed on the backing Azure resource. Override with
# SWARM_VISION_MODEL if the proxy's vision model changes.
VISION_MODEL = os.environ.get("SWARM_VISION_MODEL", "gpt-5.4-nano")


def list_proxy_models(base_url: Optional[str] = None, api_key: Optional[str] = None) -> List[str]:
    """Return the model ids an OpenAI-compatible backend serves (for the dropdown).

    Queries {base_url}/models with the key. Defaults to the legacy proxy; callers
    pass the resolved default backend so the dropdown reflects what's actually
    configured. Falls back to AVAILABLE_MODELS_FALLBACK if unreachable so the UI
    is never empty. DEFAULT_MODEL is always present and first.
    """
    base = (base_url or LITELLM_API_BASE).rstrip("/")
    key = api_key or LLM_API_KEY
    models: List[str] = []
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base}/models", headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception as e:
        log.debug("list_proxy_models: query %s failed (%s) — using fallback", base, e)
    if not models:
        models = list(AVAILABLE_MODELS_FALLBACK)
    if DEFAULT_MODEL in models:
        models.remove(DEFAULT_MODEL)
    return [DEFAULT_MODEL] + sorted(models)


def list_toolsets() -> List[Dict[str, str]]:
    """Return Hermes' available toolsets as [{name, description}] for the UI.

    Reads them from Hermes' own registry (get_all_toolsets) so the list always
    matches what the installed Hermes actually supports. Best-effort: returns a
    minimal fallback if Hermes can't be imported.
    """
    try:
        ensure_hermes_importable()
        from toolsets import get_all_toolsets

        out = []
        for name, defn in sorted(get_all_toolsets().items()):
            desc = ""
            if isinstance(defn, dict):
                desc = str(defn.get("description") or "")
            out.append({"name": name, "description": desc[:160]})
        return out
    except Exception as e:
        log.debug("list_toolsets: Hermes registry unavailable (%s)", e)
        return [{"name": n, "description": ""} for n in
                ("web", "terminal", "file", "browser", "memory", "code_execution")]

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

# Prompt/soul construction (AUTONOMOUS_HEARTBEAT_PROMPT, CRON_WAKEUP_PROMPT,
# SUPERVISOR_SWEEP_PROMPT, SUPERVISOR_DEFAULT_SOUL, compose_* / _build_* helpers)
# lives in swarm_server/prompts.py — the single home for prompt text + assembly.

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
# Tool-result aging — stub out OLD, LARGE tool results at history replay.
# Measured: tool results are 50-60% of a working agent's history, and every
# LLM iteration (10-25 per turn) re-bills the full history — yet a stale
# browser snapshot or file dump is almost never read again, and is recoverable
# by re-running the tool. Replacing them with one-line stubs cuts a large slice
# of every request without summarizing away anything the agent said or decided
# (see prompts.age_stale_tool_results). KEEP_MESSAGES is the protected working
# set at the history tail; QUANTUM makes the cutoff advance in steps so the
# request prefix stays byte-stable between steps (a provider prompt cache
# survives ~QUANTUM messages instead of busting every turn).
# ---------------------------------------------------------------------------
TOOL_RESULT_AGE_ENABLED = os.environ.get(
    "SWARM_TOOL_RESULT_AGE_ENABLED", "1").lower() not in ("0", "false", "no")
TOOL_RESULT_AGE_KEEP_MESSAGES = int(os.environ.get("SWARM_TOOL_RESULT_AGE_KEEP_MESSAGES", "40"))
TOOL_RESULT_AGE_MIN_CHARS = int(os.environ.get("SWARM_TOOL_RESULT_AGE_MIN_CHARS", "600"))
TOOL_RESULT_AGE_QUANTUM = int(os.environ.get("SWARM_TOOL_RESULT_AGE_QUANTUM", "20"))

# Supervisor counterpart: a supervisor's history is dominated by its own past
# sweep PAYLOADS (user-side team feeds — measured 96% of one overseer's session
# chars), which tool-result aging never touches. At replay, all but the last
# SWEEP_PAYLOAD_KEEP_RECENT payloads are stubbed from the marker onward; the
# supervisor's own verdicts stay verbatim.
SWEEP_PAYLOAD_AGE_ENABLED = os.environ.get(
    "SWARM_SWEEP_PAYLOAD_AGE_ENABLED", "1").lower() not in ("0", "false", "no")
SWEEP_PAYLOAD_KEEP_RECENT = int(os.environ.get("SWARM_SWEEP_PAYLOAD_KEEP_RECENT", "2"))
SWEEP_PAYLOAD_MIN_CHARS = int(os.environ.get("SWARM_SWEEP_PAYLOAD_MIN_CHARS", "500"))

# ---------------------------------------------------------------------------
# Live-context size caps. The live block is rebuilt fresh each turn AND rides
# in history for every iteration of that turn, so each item here is paid
# dozens of times per turn. Keep it a digest, not a dump — anything an agent
# needs in full it can read from the shared store.
# ---------------------------------------------------------------------------
LIVE_CTX_TREE_MAX_ENTRIES = int(os.environ.get("SWARM_LIVE_CTX_TREE_MAX_ENTRIES", "60"))
LIVE_CTX_PEER_MESSAGES = int(os.environ.get("SWARM_LIVE_CTX_PEER_MESSAGES", "6"))
LIVE_CTX_DECISIONS = int(os.environ.get("SWARM_LIVE_CTX_DECISIONS", "12"))
LIVE_CTX_COMPLETED = int(os.environ.get("SWARM_LIVE_CTX_COMPLETED", "5"))

# ---------------------------------------------------------------------------
# Disabled toolsets — removed from every agent's tool schema at init.
# ---------------------------------------------------------------------------
# Hermes ships ~71 tools across many toolsets; every enabled tool's JSON schema
# is re-sent on EVERY turn (measured ~11k tokens for the default selection) and
# several toolsets also inject a multi-hundred-token guidance block into the
# system prompt. This swarm only ever uses browser + terminal/code + file ops +
# web search + memory/todo + its own custom peer/escalation tools, so we disable
# the rest. This trims per-turn tool-schema tokens and removes the skills /
# session_search guidance blocks.
#
# delegate_task (the "delegation" toolset) is ENABLED for workers: a grindy
# multi-step job (scraping, form-filling, bulk edits) run by a fresh-context
# sub-agent costs ~10x less than running it in the parent's full history, since
# every parent iteration re-bills the whole context. The old browser-collision
# problem (concurrent sub-agents driving one shared tab) is fixed by
# browser.fresh_tab_per_task (see write_agent_hermes_config) — each sub-agent's
# unique task_id gets its own tab in the shared Chrome. Supervisors still have
# it disabled (added per-daemon in agent._ensure_agent): they review and steer,
# never execute.
DISABLED_TOOLSETS: List[str] = [
    "session_search",  # cross-session search — unused; also drops its guidance block
    "tts",             # text_to_speech
    "image_gen", "video_gen", "video",  # media generation — not used
    "spotify", "homeassistant", "discord", "discord_admin",
    "feishu_doc", "feishu_drive", "hermes-yuanbao",
    "moa", "messaging",  # third-party integrations the swarm never calls
]

# ---------------------------------------------------------------------------
# Queue / sweep behavior
# ---------------------------------------------------------------------------
MAX_BATCH_SIZE = 10          # max tasks pulled into one LLM turn (backpressure)
MAX_TASK_RETRIES = 3         # batch failures requeue up to N times, then -> 'failed' (DLQ)
LLM_ERROR_EMIT_THROTTLE_SECONDS = 60  # min gap between "provider unreachable" UI errors per agent

# Hard ceiling on tool-loop iterations within ONE turn, applied when the agent
# config doesn't set its own max_iterations. Observed without it: a single agent
# ran a 49-tool-call turn, monopolizing its thread and evading per-turn review
# (supervisors/digests see activity between turns). 40 is generous for real
# work; a task that genuinely needs more should be split across turns anyway.
DEFAULT_MAX_ITERATIONS = int(os.environ.get("SWARM_MAX_ITERATIONS", "40"))

# ---------------------------------------------------------------------------
# Adaptive idle heartbeat (24/7 autonomy without idle burn)
# ---------------------------------------------------------------------------
# When consecutive heartbeat-driven turns produce ZERO concrete external action
# (the agent had nothing real to do), the effective heartbeat interval doubles
# per miss up to base * 2**HEARTBEAT_BACKOFF_MAX_DOUBLINGS. Any turn that takes
# a concrete action resets it. This resolves the old two-failure-modes-on-one-
# knob problem: heartbeat off => the org goes dormant; fixed-interval heartbeat
# on => idle token burn + invented busywork all night.
HEARTBEAT_BACKOFF_MAX_DOUBLINGS = int(os.environ.get("SWARM_HEARTBEAT_BACKOFF_MAX", "4"))

# ---------------------------------------------------------------------------
# Per-agent cross-turn repetition guard (self-loop detector)
# ---------------------------------------------------------------------------
# The team-level loop detector catches A<->B ping-pong and team stalls, but a
# SINGLE agent repeating the same tool call with the same args across separate
# turns (re-verifying, re-reading, re-sending) is invisible to it unless a
# supervisor happens to review. The daemon tracks normalized per-turn tool-call
# signatures; when one signature repeats in SELF_LOOP_REPEATS of the last
# SELF_LOOP_WINDOW turns, ONE corrective task is injected (cooldown-limited).
SELF_LOOP_REPEATS = int(os.environ.get("SWARM_SELF_LOOP_REPEATS", "3"))
SELF_LOOP_WINDOW = int(os.environ.get("SWARM_SELF_LOOP_WINDOW", "5"))
SELF_LOOP_COOLDOWN_SECONDS = int(os.environ.get("SWARM_SELF_LOOP_COOLDOWN", "1800"))

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
# Also bound the relational tables, which otherwise grow unbounded (prune() used
# to trim only events/messages/digests). Open delegations are preserved.
MONITORING_MAX_DIGESTS = 20000
MONITORING_MAX_DELEGATIONS = 20000
MONITORING_MAX_ACTIONS = 20000
MONITORING_MAX_DECISIONS = 20000
MONITORING_MAX_MILESTONES = 5000
MONITORING_PRUNE_INTERVAL_SECONDS = 300

# ---------------------------------------------------------------------------
# Agent activity digests (Layer 3 observability) — an out-of-band cheap model
# reads each agent's recorded transcript from monitoring.db and writes a short
# rolling status summary, so an operator monitoring 10+ agents reads digests
# instead of millions of tokens of chat. The summarizer is a plain LLM call in
# a background loop (NOT a full Hermes agent) and never touches an agent's own
# context or run path.
#
# Hybrid trigger: an agent is digested when it has accrued enough NEW transcript
# (volume) OR enough time has passed since its last digest WITH new activity
# (time). Idle agents (no new messages since the last digest) are skipped
# entirely, so a quiet team costs ~nothing.
# ---------------------------------------------------------------------------
# Default cheap model for digests. Empty => fall back to the swarm default model
# (DEFAULT_MODEL). The effective value is overridable live from the UI and
# stored in the global "settings" block (see get_global_settings()).
SUMMARY_MODEL = os.environ.get("SWARM_SUMMARY_MODEL", "").strip()
# How often the background loop sweeps agents for digest-eligibility.
DIGEST_SWEEP_INTERVAL_SECONDS = int(os.environ.get("SWARM_DIGEST_INTERVAL_SECONDS", "120"))
# Volume trigger: digest once an agent accrues this many new estimated tokens.
DIGEST_MIN_NEW_TOKENS = int(os.environ.get("SWARM_DIGEST_MIN_NEW_TOKENS", "4000"))
# Time trigger: digest if at least this long has passed since the last digest
# AND there is any new activity (even below the volume threshold).
DIGEST_MAX_AGE_SECONDS = int(os.environ.get("SWARM_DIGEST_MAX_AGE_SECONDS", "900"))
# Hard cap on transcript characters fed to the summarizer in one pass; on a
# burst we keep the most-recent slice and note the truncation (no silent loss).
DIGEST_INPUT_CHAR_CAP = int(os.environ.get("SWARM_DIGEST_INPUT_CHAR_CAP", "24000"))
# Master on/off; overridable live from the UI (global "settings" block).
DIGEST_ENABLED_DEFAULT = os.environ.get("SWARM_DIGEST_ENABLED", "1") not in ("0", "false", "False", "")

# ---------------------------------------------------------------------------
# Global loop detector (Layer 5 — emergent, swarm-wide). Per-agent supervision
# can't see a loop that spans agents (A↔B↔C ping-pong); this watches the whole
# team's message graph and nudges to break a no-progress storm.
# ---------------------------------------------------------------------------
LOOP_DETECT_ENABLED = os.environ.get("SWARM_LOOP_DETECT", "1") not in ("0", "false", "False", "")
LOOP_SWEEP_INTERVAL_SECONDS = int(os.environ.get("SWARM_LOOP_SWEEP_SECONDS", "120"))
# Window of recent message history each scan considers.
LOOP_WINDOW_SECONDS = int(os.environ.get("SWARM_LOOP_WINDOW_SECONDS", "600"))
# A single pair (A↔B) trading at least this many WAKING messages in the window,
# with mostly repeated content, is a ping-pong loop.
LOOP_PAIR_THRESHOLD = int(os.environ.get("SWARM_LOOP_PAIR_THRESHOLD", "6"))
# The team sending at least this many messages in the window while logging ZERO
# decisions AND ZERO actions is a team-wide stall ("everyone talks, nobody ships").
LOOP_TEAM_MSG_THRESHOLD = int(os.environ.get("SWARM_LOOP_TEAM_MSG_THRESHOLD", "14"))
# Don't re-alert the same signature within this cooldown.
LOOP_ALERT_COOLDOWN_SECONDS = int(os.environ.get("SWARM_LOOP_COOLDOWN_SECONDS", "900"))

# ---------------------------------------------------------------------------
# Decision-log rollup. The prompt injects the last DECISION_LIVE_WINDOW decisions;
# once unrolled decisions exceed DECISION_ROLLUP_TRIGGER, the oldest beyond the
# live window are summarized into a milestone so long-term memory isn't lost.
# ---------------------------------------------------------------------------
DECISION_LIVE_WINDOW = int(os.environ.get("SWARM_DECISION_LIVE_WINDOW", "20"))
DECISION_ROLLUP_TRIGGER = int(os.environ.get("SWARM_DECISION_ROLLUP_TRIGGER", "40"))

# ---------------------------------------------------------------------------
# Supervisor agents (Layer 4 observability)
# ---------------------------------------------------------------------------
# A supervisor is an ordinary Hermes agent flagged is_supervisor and LINKED to
# the agents it watches (its allowed_peers — a team can run several supervisors
# over different subsets). Its one extra behavior is push, not pull: every
# sweep interval the daemon assembles EVERYTHING each linked peer did since the
# previous sweep — straight from the live monitoring DB, so an agent mid-turn
# contributes its partial turn up to that moment — into ONE task in the
# supervisor's own queue: a section per peer (silent peers get an explicit
# "no activity" section) plus a ledger of states, open delegations with ages,
# and pending human questions. The supervisor then runs a normal turn and
# steers with send_peer_message / pause_agent. There is NO tool to call — the
# sweep is delivered automatically, exactly like any queued message.
#
# Interval: per-agent `supervisor_interval_minutes` (agent settings), else
# this default. The old token-threshold trigger (one peer at a time once it
# accrued SUPERVISOR_TOKEN_THRESHOLD new tokens) is RETIRED — it was volume-
# gated, single-peer, and silence-blind: an idle agent sitting on an open
# delegation never generated tokens, so it was never reviewed.
SUPERVISOR_SWEEP_INTERVAL_MINUTES = float(
    os.environ.get("SWARM_SUPERVISOR_SWEEP_MINUTES", "20"))
# Total char budget for one sweep prompt, split across the peers that were
# active this window (each peer keeps its most-recent slice, truncation noted).
SUPERVISOR_SWEEP_CHAR_CAP = int(os.environ.get("SWARM_SUPERVISOR_SWEEP_CHAR_CAP", "32000"))
# Floor for one active peer's slice so a noisy teammate can't starve the rest
# below readability.
SUPERVISOR_SWEEP_PER_PEER_FLOOR = int(
    os.environ.get("SWARM_SUPERVISOR_SWEEP_PER_PEER_FLOOR", "2000"))
# Idle-skip: when NO watched peer logged a single message since the last sweep,
# skip queueing one (a skipped sweep costs ~6 SQL rows; a queued all-IDLE sweep
# was measured at a full ~85k-token LLM turn). Never go silence-blind though —
# the very failure the sweep design exists to catch is an idle agent OWING
# work — so every MAX_IDLE_SKIPS-th consecutive idle interval forces a full
# review anyway, and an open delegation older than FORCE_OPEN_AGE forces one
# early (rate-limited to once per age window, so a permanently parked item
# can't degenerate back to sweeping every interval).
SUPERVISOR_SWEEP_MAX_IDLE_SKIPS = int(
    os.environ.get("SWARM_SUPERVISOR_SWEEP_MAX_IDLE_SKIPS", "3"))
SUPERVISOR_SWEEP_FORCE_OPEN_AGE_SECONDS = int(
    os.environ.get("SWARM_SUPERVISOR_SWEEP_FORCE_OPEN_AGE_SECONDS", "7200"))
# DEPRECATED — no longer read by the daemon; kept so old .env files don't error.
SUPERVISOR_TOKEN_THRESHOLD = int(os.environ.get("SWARM_SUPERVISOR_TOKEN_THRESHOLD", "6000"))
# Legacy single-review cap; still the default char_cap of _render_feed_transcript.
SUPERVISOR_FEED_CHAR_CAP = int(os.environ.get("SWARM_SUPERVISOR_FEED_CHAR_CAP", "24000"))
# SUPERVISOR_SWEEP_PROMPT and SUPERVISOR_DEFAULT_SOUL live in swarm_server/prompts.py.

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
# New config schema helpers
# ---------------------------------------------------------------------------
def _derive_workspace_path(team_id: str, agent_name: str) -> Path:
    """Return the on-disk workspace directory for an agent."""
    return WORKSPACE_ROOT / team_id / "workspace" / agent_name


def _get_team_workspace_path(team_id: str) -> Path:
    """Return the shared team workspace directory.

    Holds team-shared metadata — the project brief (workspace.md) and each
    agent's per-agent runtime home (.hermes + queue db). The team changelog is no
    longer a file (agent_log.md is sunset); decisions live in the decision log
    (monitoring.db). This is NOT where code/deliverables go; that is the shared
    project dir below.
    """
    return WORKSPACE_ROOT / team_id / "workspace"


def _get_project_dir(team_id: str, full_config: Optional[Dict[str, Any]] = None) -> Path:
    """Return the single SHARED work surface for a team.

    Every agent on the team reads and writes the SAME files here — there are no
    private per-agent copies. This is the real project/repo the team builds. Both
    the file tools and (via TERMINAL_CWD) the terminal operate here by default, so
    ``search_files`` sees the whole project including teammates' work.

    Configurable per team via ``teams.<team>.project_dir`` (an absolute path —
    e.g. point it at a repo you already use); otherwise defaults to a managed
    directory at ``<team>/project``.
    """
    cfg = full_config
    if cfg is None:
        try:
            cfg = load_agents_config()
        except Exception:
            cfg = None
    custom = None
    if cfg:
        team_cfg = (cfg.get("teams", {}) or {}).get(team_id) or {}
        custom = team_cfg.get("project_dir")
    if custom:
        return Path(str(custom)).expanduser()
    return WORKSPACE_ROOT / team_id / "project"


def _ensure_project_dir(team_id: str, full_config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve the shared project dir, creating it (and a git repo) if missing.

    Best-effort: a fresh team gets an empty git repo so commits work from the
    first task. If the path already exists (e.g. it points at a repo you already
    use) nothing is reinitialized.
    """
    project_dir = _get_project_dir(team_id, full_config)
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        if not (project_dir / ".git").exists():
            import subprocess

            subprocess.run(
                ["git", "init", "-q"], cwd=str(project_dir), check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass
    return project_dir


def write_agent_hermes_config(
    hermes_home: Path,
    cdp_url: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    compression_threshold: Optional[float] = None,
    provider: str = "custom",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    is_supervisor: bool = False,
) -> None:
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
    # Effective backend: resolved by the caller (per-agent override → swarm
    # default → legacy proxy). base_url present => OpenAI-compatible "custom"
    # path; absent => a native provider (e.g. anthropic) Hermes resolves itself.
    eff_base = base_url if base_url is not None else LITELLM_API_BASE
    eff_key = api_key if api_key is not None else LLM_API_KEY
    eff_provider = provider or "custom"
    model_section = existing.get("model")
    if not isinstance(model_section, dict):
        model_section = {}
    model_section["context_length"] = AGENT_CONTEXT_WINDOW
    model_section["default"] = model
    model_section["provider"] = eff_provider
    if eff_base:
        model_section["base_url"] = eff_base
    elif "base_url" in model_section:
        del model_section["base_url"]
    if eff_key:
        model_section["api_key"] = eff_key
    existing["model"] = model_section

    _threshold = compression_threshold if compression_threshold is not None else COMPRESSION_THRESHOLD
    existing["compression"] = {
        "enabled": COMPRESSION_ENABLED,
        "threshold": _threshold,
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
        "provider": eff_provider,
        "model": model,
    }
    if eff_base:
        aux_endpoint["base_url"] = eff_base
    if eff_key:
        aux_endpoint["api_key"] = eff_key
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
    # Vision (browser_vision screenshot reading) is special: it MUST use a
    # multimodal model. Prefer the agent's MAIN model when a one-time probe
    # shows it actually reads images (one model, no extra hop); otherwise pin
    # the dedicated vision model on the same proxy. Supplying both base_url AND
    # api_key forces the deterministic "custom endpoint" branch in
    # auxiliary_client.resolve_vision_provider_client, bypassing the capability
    # heuristic that would otherwise drop to unauthenticated aggregators (Gemini/
    # OpenRouter) and fail. Without this entry the "vision" task resolves to
    # provider="auto" and browser_vision raises "No LLM provider configured".
    # The probe needs an OpenAI-compatible endpoint; native providers (no
    # base_url) skip it and keep the dedicated vision model.
    vision_cfg = dict(aux_endpoint)
    vision_cfg["model"] = (
        resolve_screenshot_model(model, eff_base, eff_key)
        if eff_base else get_vision_model()
    )
    aux_section["vision"] = {**aux_section.get("vision", {}), **vision_cfg}
    existing["auxiliary"] = aux_section

    # Pin ddgs (DuckDuckGo, no API key) as the web search backend so web_search
    # is always available and never silently falls through to an unconfigured
    # paid backend. Requires the `ddgs` package installed in the server's venv.
    web_section = existing.get("web")
    if not isinstance(web_section, dict):
        web_section = {}
    web_section["search_backend"] = WEB_SEARCH_BACKEND
    existing["web"] = web_section

    # Apply the disabled-toolsets list (see DISABLED_TOOLSETS for what and
    # why). Merge-safe with Hermes-seeded agent.* keys.
    agent_section = existing.get("agent")
    if not isinstance(agent_section, dict):
        agent_section = {}
    _disabled = list(DISABLED_TOOLSETS)
    if is_supervisor:
        # Supervisors must OBSERVE, not do project work — "do no project tasks" is
        # otherwise just a request the model ignores (the saas overseer ran 57
        # terminal commands and joined the very loop it was meant to police).
        # Physically remove the action toolsets so a supervisor structurally CANNOT
        # run commands, browse, execute code, web-search, or spawn sub-agents. It
        # keeps file-read, memory, and the swarm tools (send_peer_message,
        # pause_agent, log_decision, ask_human) — everything oversight needs and
        # nothing project work needs.
        _disabled += [t for t in
                      ("terminal", "browser", "code_execution", "web", "delegation")
                      if t not in _disabled]
    agent_section["disabled_toolsets"] = _disabled
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
            # The file EXISTS but could not be read/parsed. Returning an empty
            # default here is catastrophic: the next load-modify-SAVE caller would
            # persist the empty roster over the real one (a full wipe from a
            # transient EMFILE/EACCES or a momentary corruption). Prefer the
            # last-known-good cache, then the newest backup, and only RAISE as a
            # last resort — never silently serve an empty config.
            log.error("Failed to load agents config: %s", e)
            if _config_cache is not None:
                log.warning("Serving last-known-good agents config from cache")
                return _deep_copy_config(_config_cache)
            restored = _load_newest_backup()
            if restored is not None:
                log.warning("Recovered agents config from newest backup")
                _config_cache = _deep_copy_config(restored)
                return _deep_copy_config(restored)
            raise RuntimeError(
                "agents_config.json exists but is unreadable and no cache/backup "
                "is available; refusing to serve an empty config that would wipe "
                f"the roster on the next save: {e}"
            ) from e

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


_CONFIG_BACKUP_KEEP = 10


def _load_newest_backup() -> Optional[Dict[str, Any]]:
    """Return the newest parseable backup of agents_config.json, or None.

    Used to recover from a corrupt/unreadable live file instead of wiping the
    roster. Best-effort: tries newest-first and skips any unparseable backup.
    """
    try:
        backup_dir = DATA_ROOT / "config_backups"
        backups = sorted(backup_dir.glob("agents_config.*.json"))
        for path in reversed(backups):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    return raw
            except Exception:
                continue
    except Exception:
        pass
    return None


def _backup_config_file() -> None:
    """Copy the current agents_config.json to a rotating backup before overwrite.

    Best-effort and never raises — a backup failure must not block the real save.
    Backups live in data/config_backups/ and the oldest beyond _CONFIG_BACKUP_KEEP
    are pruned. Recovery: copy the newest good backup over agents_config.json.
    """
    try:
        if not AGENTS_CONFIG_PATH.exists():
            return
        backup_dir = DATA_ROOT / "config_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Monotonic-ish name from the file's mtime (avoids needing a clock here).
        st = AGENTS_CONFIG_PATH.stat()
        dest = backup_dir / f"agents_config.{int(st.st_mtime)}.{st.st_size}.json"
        if not dest.exists():
            shutil.copy2(AGENTS_CONFIG_PATH, dest)
        backups = sorted(backup_dir.glob("agents_config.*.json"))
        for old in backups[:-_CONFIG_BACKUP_KEEP]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception as e:
        log.debug("config backup skipped: %s", e)


def _save_full_config(cfg: Dict[str, Any]) -> None:
    """Atomically persist the full config.

    Writes to a temp file in the same directory and os.replace()s it into
    place so a concurrent reader (or a crash mid-write) can never observe a
    half-written / truncated agents_config.json. The lock serializes writers
    so two concurrent saves cannot interleave.
    """
    with _config_lock:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        # Safety net: keep a small rotating set of timestamped backups of the
        # CURRENT file before overwriting it, so an accidental delete or a bad
        # write is always recoverable (the file is gitignored — no other history).
        _backup_config_file()
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
    # Atomic read-modify-write: hold the (reentrant) config lock across the load
    # AND the save so a concurrent writer (e.g. another agent's add_agent_cron, or
    # this agent's session-id rotation persist) can't read a stale full config and
    # clobber the other's change. load_agents_config/_save_full_config both
    # re-enter _config_lock, so wrapping them is safe.
    with _config_lock:
        full = load_agents_config()
        full["agents"][agent_name] = cfg
        _save_full_config(full)


def save_all_config(cfg: Dict[str, Any]) -> None:
    _save_full_config(cfg)


# ---------------------------------------------------------------------------
# Team CRUD
# ---------------------------------------------------------------------------
_SAFE_ID_RE = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_id(value: str, kind: str = "id") -> str:
    """Return a filesystem-safe id or raise ValueError.

    team_id / agent_name flow into ``WORKSPACE_ROOT/<id>/...`` paths (workspaces,
    credentials.json). Without this a value like ``../../etc`` would let an API
    caller create dirs / write files OUTSIDE DATA_ROOT (path traversal). Restrict
    to a slug charset — no separators, no dots, no leading separator.
    """
    v = (value or "").strip()
    if not _SAFE_ID_RE.match(v):
        raise ValueError(
            f"invalid {kind} {value!r}: must be 1–64 chars of letters/digits/_/- "
            f"and cannot contain path separators or '..'"
        )
    return v


def create_team(cfg: Dict[str, Any], team_id: str, name: str) -> Dict[str, Any]:
    team_id = validate_id(team_id, "team_id")
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
    # NOTE: agent_log.md is sunset — the team activity trail is now the
    # decision log (monitoring.db `decisions` table, written via the
    # log_decision tool and auto-injected into prompts). No file to seed.
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
    is_supervisor: bool = False,
) -> Dict[str, Any]:
    if name in cfg["agents"]:
        raise ValueError(f"Agent '{name}' already exists")
    if team_id not in cfg["teams"]:
        raise ValueError(f"Team '{team_id}' does not exist")

    # A supervisor with no custom soul gets the default supervisor identity.
    # Local import: prompt text lives in swarm_server.prompts, which imports
    # path/config helpers from this module — a function-local import here keeps
    # that one-directional (no config -> prompts top-level cycle).
    from swarm_server.prompts import SUPERVISOR_DEFAULT_SOUL
    default_soul = SUPERVISOR_DEFAULT_SOUL if is_supervisor else f"You are the {display_name}."
    agent_cfg = {
        "team_id": team_id,
        "name": display_name,
        "session_id": f"{name}-master-session-v1",
        "allowed_peers": list(allowed_peers or []),
        "role_soul": role_soul or default_soul,
    }
    if is_supervisor:
        agent_cfg["is_supervisor"] = True
    cfg["agents"][name] = agent_cfg
    # Prepare workspace dirs
    ws = _derive_workspace_path(team_id, name)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "context").mkdir(exist_ok=True)
    # agent_log.md is sunset — see make_team(); decisions now live in the
    # shared decision log (log_decision tool -> monitoring.db).
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
    """Set agent ``name``'s peer list to EXACTLY ``peers``, BIDIRECTIONALLY.

    All swarm connections are bidirectional: a link is stored on both agents'
    ``allowed_peers`` and authorizes messaging either way. This setter therefore
    syncs the reverse side too — it adds ``name`` to every new peer and removes
    ``name`` from any agent that is no longer a peer — so there is never a
    one-directional ("A can message B but not vice-versa") link.
    """
    if name not in cfg["agents"]:
        raise ValueError(f"Agent '{name}' not found")
    name_team = cfg["agents"][name].get("team_id")
    desired: List[str] = []
    for peer in peers:
        peer = str(peer)
        if peer == name:
            raise ValueError("An agent cannot be linked to itself.")
        if peer not in cfg["agents"]:
            raise ValueError(f"Peer agent '{peer}' not found")
        peer_team = cfg["agents"][peer].get("team_id")
        if name_team and peer_team and name_team != peer_team:
            raise ValueError(f"Cross-team peer links are blocked: '{name}' (team={name_team}) → '{peer}' (team={peer_team})")
        if peer not in desired:
            desired.append(peer)
    cfg["agents"][name]["allowed_peers"] = list(desired)
    # Sync the reverse side on every other agent so the relation stays symmetric.
    for other_name, other in cfg["agents"].items():
        if other_name == name:
            continue
        plist = list(other.get("allowed_peers", []))
        if other_name in desired and name not in plist:
            other["allowed_peers"] = plist + [name]
        elif other_name not in desired and name in plist:
            other["allowed_peers"] = [p for p in plist if p != name]
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
    """Return True if caller and target are linked AND on the same team.

    Connections are bidirectional, so EITHER side listing the other authorizes
    messaging in both directions. Checking both sides also makes the permission
    robust to any legacy data where only one side recorded the link.
    """
    caller_cfg = cfg["agents"].get(caller)
    target_cfg = cfg["agents"].get(target)
    if not caller_cfg or not target_cfg:
        return False
    if caller_cfg.get("team_id") != target_cfg.get("team_id"):
        return False
    return (target in caller_cfg.get("allowed_peers", [])
            or caller in target_cfg.get("allowed_peers", []))


# ---------------------------------------------------------------------------
# Per-agent cron wake-ups
# ---------------------------------------------------------------------------
# Each agent may carry a ``crons`` list in its config; every entry is a
# self-contained schedule that injects ``instruction`` as a task when it fires.
# Both the human (dashboard / REST) and the agent itself (schedule_wakeup tool)
# create these; the AgentDaemon evaluates them in its sweep loop. The schedule
# string is validated with cron.cron_validate before it is ever stored.
import uuid as _uuid

MAX_CRONS_PER_AGENT = int(os.environ.get("SWARM_MAX_CRONS_PER_AGENT", "25"))

# A cron instruction must stay GOAL-level. Long step-by-step scripts freeze a
# moment in time and the agent re-executes the stale script verbatim forever
# (observed: a 1.7k-char daily-checkin script pinned founder to June-10 state).
# Agents get a tight cap; humans a looser one (dashboard paste). Detailed
# procedures belong in a workspace runbook file the instruction references.
MAX_CRON_INSTRUCTION_CHARS_AGENT = int(os.environ.get("SWARM_MAX_CRON_INSTRUCTION_CHARS_AGENT", "600"))
MAX_CRON_INSTRUCTION_CHARS_HUMAN = int(os.environ.get("SWARM_MAX_CRON_INSTRUCTION_CHARS_HUMAN", "4000"))

_CRON_INSTRUCTION_TOO_LONG = (
    "Instruction too long ({n} chars > {cap}). Keep cron instructions GOAL-level: "
    "what to achieve, where the data/runbook lives, when to escalate. Write the "
    "detailed step-by-step procedure to a file in your workspace (e.g. "
    "docs/<name>-runbook.md) and reference its path — you'll re-read the LATEST "
    "version each time it fires instead of replaying a frozen script."
)


def _validate_cron_instruction(instruction: str, created_by: str) -> str:
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("A cron wake-up needs an instruction to run when it fires.")
    cap = (
        MAX_CRON_INSTRUCTION_CHARS_HUMAN
        if created_by == "human"
        else MAX_CRON_INSTRUCTION_CHARS_AGENT
    )
    if len(instruction) > cap:
        raise ValueError(_CRON_INSTRUCTION_TOO_LONG.format(n=len(instruction), cap=cap))
    return instruction


# ---------------------------------------------------------------------------
# Global swarm settings (top-level "settings" block in agents_config.json).
# Survives restarts; deep-copied with the rest of the config. Used for the
# UI-configurable digest summary model + on/off, etc. Unknown/legacy configs
# simply have no "settings" key and fall back to the env/code defaults.
# ---------------------------------------------------------------------------
_GLOBAL_SETTINGS_DEFAULTS = {
    # "" => use the swarm default model (DEFAULT_MODEL) for digests.
    "summary_model": SUMMARY_MODEL,
    "digest_enabled": DIGEST_ENABLED_DEFAULT,
    # Multimodal model for screenshot reading + GUI grounding (browser_vision,
    # browser_locate). "" => the VISION_MODEL env/code default. UI-settable;
    # written into every agent's auxiliary.vision config on re-init.
    "vision_model": "",
    # "" => the Architect (master team-builder) uses the swarm default model.
    "master_model": "",
}


def get_global_settings() -> Dict[str, Any]:
    """Effective global settings: stored values layered over code/env defaults."""
    cfg = load_agents_config()
    stored = cfg.get("settings") or {}
    out = dict(_GLOBAL_SETTINGS_DEFAULTS)
    for k in _GLOBAL_SETTINGS_DEFAULTS:
        if k in stored and stored[k] is not None:
            out[k] = stored[k]
    return out


def update_global_settings(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Patch known global settings keys and persist. Returns effective settings."""
    # Atomic RMW under the reentrant lock so a concurrent agent/config write isn't
    # clobbered by saving a stale full config (see save_agent_config).
    with _config_lock:
        cfg = load_agents_config()
        settings = dict(cfg.get("settings") or {})
        if "summary_model" in fields:
            settings["summary_model"] = (fields["summary_model"] or "").strip()
        if "digest_enabled" in fields:
            settings["digest_enabled"] = bool(fields["digest_enabled"])
        if "vision_model" in fields:
            settings["vision_model"] = (fields["vision_model"] or "").strip()
        if "master_model" in fields:
            settings["master_model"] = (fields["master_model"] or "").strip()
        cfg["settings"] = settings
        _save_full_config(cfg)
    out = dict(_GLOBAL_SETTINGS_DEFAULTS)
    for k in _GLOBAL_SETTINGS_DEFAULTS:
        if k in settings and settings[k] is not None:
            out[k] = settings[k]
    return out


def get_vision_model() -> str:
    """Effective multimodal model for screenshot reading / GUI grounding:
    the UI-set global setting, else the VISION_MODEL env/code default."""
    try:
        configured = (get_global_settings().get("vision_model") or "").strip()
        if configured:
            return configured
    except Exception:
        pass
    return VISION_MODEL


# Vision-capability probe. Proxy metadata can't be trusted here (LiteLLM's
# /model/info carries no supports_vision for generic aliases like
# "litellm-model"), and a gateway may silently ACCEPT image parts a text-only
# model never sees — so capability is established empirically: send a solid
# red PNG and require the model to name the color. Verdicts are cached for
# the process lifetime (capability doesn't change under a fixed model name;
# a restart re-probes after a proxy remap).
_VISION_PROBE_CACHE: Dict[tuple, bool] = {}
_VISION_PROBE_LOCK = threading.Lock()
VISION_PROBE_TIMEOUT_SECONDS = float(os.environ.get("SWARM_VISION_PROBE_TIMEOUT", "15"))


def _probe_png() -> bytes:
    """64x64 solid-red RGB PNG, stdlib only."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    w = h = 64
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _vision_probe(model: str, base_url: str, api_key: str) -> bool:
    """One uncached probe call. True only if the model demonstrably SAW the image.

    Uses max_completion_tokens (a tight max_tokens starves reasoning models —
    they burn the whole budget thinking and return nothing) and no temperature
    (the o-series/gpt-5 family rejects non-default values). Falls back to
    max_tokens for older backends that reject max_completion_tokens.
    """
    import base64
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    base_payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "What is the dominant color of this image? Reply with one word."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,"
                           + base64.b64encode(_probe_png()).decode("ascii")}},
            ],
        }],
    }

    def attempt(limit_key: str) -> str:
        payload = dict(base_payload)
        payload[limit_key] = 200
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=VISION_PROBE_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return ((body.get("choices") or [{}])[0]
                .get("message", {}).get("content", "") or "")

    try:
        reply = attempt("max_completion_tokens")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 400 and "max_completion_tokens" in detail:
            reply = attempt("max_tokens")
        else:
            raise RuntimeError(f"HTTP {e.code}: {detail[:200]}") from None
    return "red" in reply.lower()


def model_supports_vision(model: str, base_url: str, api_key: str) -> bool:
    """Whether `model` at `base_url` accepts AND understands image input.

    Probes once per (base_url, model) and caches the verdict for the process
    lifetime. Any failure (HTTP 4xx for image parts, timeout, proxy down,
    blind-but-200 answer) verdicts False — the safe direction, since callers
    fall back to the dedicated vision model.
    """
    model = (model or "").strip()
    base_url = (base_url or "").strip()
    if not model or not base_url:
        return False
    # Key includes the api_key: a probe that failed under a BAD key must not pin a
    # False verdict that survives the operator fixing the credential (same
    # base_url/model). The key is in-process memory only.
    key = (base_url, model, api_key or "")
    with _VISION_PROBE_LOCK:
        if key in _VISION_PROBE_CACHE:
            return _VISION_PROBE_CACHE[key]
    try:
        verdict = _vision_probe(model, base_url, api_key)
    except Exception as e:
        # Probe ERRORED (auth/network/proxy/timeout) — this is NOT a definitive
        # "text-only" answer. Return False for this call but do NOT cache it, so a
        # later call (e.g. after the key/proxy is fixed) re-probes instead of being
        # pinned to the fallback vision model for the whole process lifetime.
        log.info("vision probe failed for %s @ %s (%s) — treating as text-only "
                 "for now (not cached; will retry)", model, base_url, e)
        return False
    # Only a probe that actually completed (image seen or definitively not) is
    # cached — capability is stable under a fixed model name.
    with _VISION_PROBE_LOCK:
        _VISION_PROBE_CACHE[key] = verdict
    log.info("vision capability: %s @ %s -> %s", model, base_url, verdict)
    return verdict


def resolve_screenshot_model(model: str, base_url: str, api_key: str) -> str:
    """Model to use for screenshot reading / GUI grounding: the agent's MAIN
    model when it can read images (one model, no extra hop), else the
    configured vision model."""
    if model_supports_vision(model, base_url, api_key):
        return model
    return get_vision_model()


def list_agent_crons(cfg: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    a = cfg["agents"].get(name)
    if a is None:
        raise ValueError(f"Agent '{name}' not found")
    return list(a.get("crons") or [])


def add_agent_cron(
    cfg: Dict[str, Any],
    name: str,
    schedule: str,
    instruction: str,
    enabled: bool = True,
    created_by: str = "human",
) -> Dict[str, Any]:
    """Validate + append a cron wake-up to an agent. Returns the new entry."""
    from swarm_server.cron import cron_validate

    instruction = _validate_cron_instruction(instruction, created_by)
    ok, norm = cron_validate(schedule or "")
    if not ok:
        raise ValueError(f"Invalid schedule: {norm}")
    entry = {
        "id": str(_uuid.uuid4()),
        "schedule": norm,
        "instruction": instruction,
        "enabled": bool(enabled),
        "created_at": int(time.time()),
        "created_by": created_by,
    }
    # Atomic RMW against a FRESH load under the lock, so appending a cron can't
    # clobber a concurrent writer (e.g. another agent's session-id persist) by
    # saving a stale full config.
    with _config_lock:
        full = load_agents_config()
        a = full["agents"].get(name)
        if a is None:
            raise ValueError(f"Agent '{name}' not found")
        crons = a.setdefault("crons", [])
        if len(crons) >= MAX_CRONS_PER_AGENT:
            raise ValueError(f"Cron limit reached ({MAX_CRONS_PER_AGENT}). Delete one first.")
        crons.append(entry)
        _save_full_config(full)
        # Reflect into the caller's cfg view (best-effort) so it stays consistent.
        caller_agent = cfg.get("agents", {}).get(name)
        if isinstance(caller_agent, dict):
            caller_agent.setdefault("crons", []).append(entry)
    return entry


def update_agent_cron(
    cfg: Dict[str, Any], name: str, cron_id: str, fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Patch an existing cron (schedule / instruction / enabled). Returns the entry."""
    from swarm_server.cron import cron_validate

    a = cfg["agents"].get(name)
    if a is None:
        raise ValueError(f"Agent '{name}' not found")
    for c in a.get("crons", []):
        if c.get("id") != cron_id:
            continue
        if "schedule" in fields:
            ok, norm = cron_validate(fields["schedule"] or "")
            if not ok:
                raise ValueError(f"Invalid schedule: {norm}")
            c["schedule"] = norm
        if "instruction" in fields:
            # Updates come from the dashboard/REST only — human cap applies.
            c["instruction"] = _validate_cron_instruction(fields["instruction"], "human")
        if "enabled" in fields:
            c["enabled"] = bool(fields["enabled"])
        _save_full_config(cfg)
        return c
    raise ValueError(f"Cron '{cron_id}' not found on agent '{name}'")


def remove_agent_cron(cfg: Dict[str, Any], name: str, cron_id: str) -> bool:
    a = cfg["agents"].get(name)
    if a is None:
        raise ValueError(f"Agent '{name}' not found")
    crons = a.get("crons") or []
    remaining = [c for c in crons if c.get("id") != cron_id]
    if len(remaining) == len(crons):
        return False
    a["crons"] = remaining
    _save_full_config(cfg)
    return True


# Ensure tool subprocesses (rg/grep/find) always have a usable PATH, no matter
# how the server process was launched. Must run before any agent spawns tools.
_ensure_full_path()

# Initial load
AGENTS = load_agents_config()
