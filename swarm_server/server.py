"""FastAPI application, REST routes, WebSocket endpoint, and lifecycle management."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from swarm_server.agent import AgentDaemon, AGENT_STATE_ASKING_HUMAN
from swarm_server.config import (
    AGENTS,
    CORS_ALLOWED_ORIGINS,
    DASHBOARD_DIR,
    DEFAULT_MODEL,
    DIGEST_SWEEP_INTERVAL_SECONDS,
    LITELLM_API_BASE,
    list_proxy_models,
    list_toolsets,
    MONITORING_DB,
    MONITORING_MAX_EVENTS,
    MONITORING_MAX_MESSAGES,
    MONITORING_PRUNE_INTERVAL_SECONDS,
    get_global_settings,
    update_global_settings,
    SERVER_HOST,
    SERVER_PORT,
    SWARM_API_KEY,
    add_agent_cron,
    add_agent_peer,
    create_agent,
    create_team,
    delete_agent,
    delete_team,
    get_agent_team,
    get_team_agents,
    list_agent_crons,
    list_teams,
    load_agents_config,
    remove_agent_cron,
    remove_agent_peer,
    save_agent_config,
    save_all_config,
    set_agent_peers,
    update_agent_cron,
    _derive_workspace_path,
)
from swarm_server.monitoring import monitor_db
from swarm_server.tools import _daemon_registry
from swarm_server.websocket import ws_broadcaster
import swarm_server.websocket as _ws_mod

log = logging.getLogger("swarm.server")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
daemons: Dict[str, AgentDaemon] = {}


async def _periodic_monitoring_prune():
    """Keep monitoring.db bounded over long 24/7 runs."""
    while True:
        await asyncio.sleep(MONITORING_PRUNE_INTERVAL_SECONDS)
        try:
            monitor_db.prune(MONITORING_MAX_EVENTS, MONITORING_MAX_MESSAGES)
        except Exception as e:
            log.warning("[Prune] %s", e)


async def _periodic_digest():
    """Layer-3 observability: sweep agents and write a rolling status digest for
    any with enough new activity (hybrid volume/time trigger lives in
    maybe_digest; idle agents cost one COUNT(*) and are skipped).

    Each digest is a blocking LLM call, so it runs in a worker thread — the
    event loop is never blocked. A small semaphore bounds concurrent summary
    calls so a big team can't fan out into a thundering herd on the cheap model.
    """
    from swarm_server.summarizer import maybe_digest, maybe_rollup_decisions

    sem = asyncio.Semaphore(3)

    async def _one(name: str, team_id):
        async with sem:
            try:
                await asyncio.to_thread(maybe_digest, name, team_id)
            except Exception as e:
                log.warning("[Digest] sweep failed for %s: %s", name, e)

    while True:
        await asyncio.sleep(DIGEST_SWEEP_INTERVAL_SECONDS)
        try:
            jobs = [
                _one(name, (d.cfg or {}).get("team_id"))
                for name, d in list(daemons.items())
            ]
            if jobs:
                await asyncio.gather(*jobs, return_exceptions=True)
            # Decision-log rollup (long-term memory): once per team per sweep,
            # summarize decisions that have scrolled past the live window.
            teams = {(d.cfg or {}).get("team_id") for d in daemons.values()}
            for tid in teams:
                if tid:
                    try:
                        await asyncio.to_thread(maybe_rollup_decisions, tid)
                    except Exception as e:
                        log.warning("[Rollup] %s failed: %s", tid, e)
        except Exception as e:
            log.warning("[Digest] sweep error: %s", e)


async def _periodic_loop_detector():
    """Layer-5: scan each team's message graph for cross-agent loops / stalls and
    nudge to break them (see loop_detector.scan_team). Runs in a worker thread —
    the scan is pure SQLite reads + cheap string work, no LLM call."""
    from swarm_server.config import LOOP_DETECT_ENABLED, LOOP_SWEEP_INTERVAL_SECONDS
    if not LOOP_DETECT_ENABLED:
        log.info("[LoopDetector] disabled")
        return
    from swarm_server.loop_detector import scan_team

    def _ingest(agent_name: str, from_agent: str, payload: str) -> None:
        d = daemons.get(agent_name)
        if d is not None:
            d.ingest_task(from_agent=from_agent, payload=payload)

    while True:
        await asyncio.sleep(LOOP_SWEEP_INTERVAL_SECONDS)
        try:
            teams: Dict[str, list] = {}
            for name, d in list(daemons.items()):
                tid = (d.cfg or {}).get("team_id")
                if tid:
                    teams.setdefault(tid, []).append(name)
            for tid, members in teams.items():
                if len(members) < 2:
                    continue
                await asyncio.to_thread(scan_team, tid, members, _ingest)
        except Exception as e:
            log.warning("[LoopDetector] sweep error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    _ws_mod._main_event_loop = asyncio.get_running_loop()
    log.info("[Startup] Main event loop captured")

    # NOTE: queue DBs are intentionally NOT deleted here anymore. Each
    # AgentDaemon recovers its own in-flight ('processing') tasks on construction
    # (queue.recover_processing()), so a restart resumes work instead of losing it.
    cfg = load_agents_config()
    loop = asyncio.get_running_loop()
    for agent_name, agent_cfg in cfg["agents"].items():
        register_agent_daemon(agent_name, agent_cfg, loop)

    prune_task = loop.create_task(_periodic_monitoring_prune())
    digest_task = loop.create_task(_periodic_digest())
    loopdet_task = loop.create_task(_periodic_loop_detector())
    log.info("[Startup] All agents running. LiteLLM at %s", LITELLM_API_BASE)
    log.info("[Startup] Dashboard at http://%s:%s/", SERVER_HOST, SERVER_PORT)

    try:
        yield
    finally:
        # ---- shutdown ----
        prune_task.cancel()
        digest_task.cancel()
        loopdet_task.cancel()
        for name, daemon in list(daemons.items()):
            _stop_daemon(daemon)
        # Stop per-team browsers (their on-disk profiles persist for next run).
        try:
            from swarm_server.browser_pool import team_browser_manager
            team_browser_manager.shutdown_all()
        except Exception as e:
            log.warning("[Shutdown] team browser shutdown failed: %s", e)
        log.info("[Shutdown] All sweep tasks cancelled")


app = FastAPI(title="Hermes Swarm Server", version="0.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _auth_guard(request: Request, call_next):
    """Optional bearer/API-key auth on mutating endpoints.

    Disabled unless SWARM_API_KEY is set (the server binds localhost). When set,
    POST/PUT/PATCH/DELETE require either 'Authorization: Bearer <key>' or
    'X-API-Key: <key>'. Read-only GETs and the dashboard stay open.
    """
    if SWARM_API_KEY and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        provided = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if not provided and auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        if provided != SWARM_API_KEY:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    import time

    await ws_broadcaster.connect(ws)
    try:
        state_snapshot = {
            "type": "state_snapshot",
            "payload": {
                "agents": {
                    name: {
                        "state": d.state,
                        "pending_count": d.queue.get_pending_count(),
                        "config": d.cfg,
                        "next_sweep_at": d.next_sweep_at,
                        "telemetry": dict(getattr(d, "_telemetry", {}) or {}),
                    }
                    for name, d in daemons.items()
                },
                "timestamp": time.time(),
            },
        }
        await ws.send_text(json.dumps(state_snapshot))

        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "payload": {}}))
            except Exception:
                pass
    except WebSocketDisconnect:
        await ws_broadcaster.disconnect(ws)
    except Exception as e:
        log.warning("[WS] Error: %s", e)
        await ws_broadcaster.disconnect(ws)


# ---------------------------------------------------------------------------
# Core Agent Routes
# ---------------------------------------------------------------------------
@app.post("/agent/{agent_name}/task")
async def agent_ingest(agent_name: str, request: Request):
    body = await request.json()
    from_agent = body.get("from_agent", "unknown")
    payload = body.get("payload", "")
    if not payload:
        return JSONResponse({"error": "empty payload"}, status_code=400)
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    task_id = daemon.ingest_task(from_agent, payload)
    return JSONResponse({"task_id": task_id, "status": "queued"})


@app.get("/agent/{agent_name}/status")
async def agent_status(agent_name: str):
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({
        "agent": agent_name,
        "state": daemon.state,
        "pending_count": daemon.queue.get_pending_count(),
        "session_id": daemon.cfg.get("session_id"),
    })


def _shape_digest(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stored digest row: parse the summary JSON for the client."""
    if not row:
        return None
    out = dict(row)
    try:
        out["summary"] = json.loads(row.get("summary") or "{}")
    except Exception:
        out["summary"] = {"headline": str(row.get("summary") or ""),
                          "did": [], "blocked_on": None, "next": None,
                          "risk_level": "watch"}
    return out


@app.get("/agent/{agent_name}/digest")
async def agent_digest(agent_name: str):
    """Latest rolling activity digest for an agent (Layer-3 observability).

    Queried by agent_name only — names are globally unique here, and the
    transcript these digests summarize is logged with a NULL team_id."""
    row = monitor_db.get_last_digest(agent_name)
    return JSONResponse({"agent": agent_name, "digest": _shape_digest(row)})


@app.get("/agent/{agent_name}/digests")
async def agent_digests(agent_name: str, limit: int = 50):
    """Digest history (newest first) for an agent."""
    rows = monitor_db.get_digests(agent_name, limit=max(1, min(limit, 200)))
    return JSONResponse({"agent": agent_name,
                         "digests": [_shape_digest(r) for r in rows]})


@app.get("/settings")
async def get_settings():
    """Global swarm settings (e.g. the digest summary model + on/off)."""
    from swarm_server.config import VISION_MODEL

    out = dict(get_global_settings())
    # What "" falls back to — lets the UI show the real default as placeholder.
    out["vision_model_default"] = VISION_MODEL
    return JSONResponse(out)


@app.post("/settings")
async def post_settings(request: Request):
    """Patch global swarm settings. Recognized keys: summary_model,
    digest_enabled, vision_model."""
    body = await request.json()
    fields = {}
    if "summary_model" in body:
        fields["summary_model"] = body.get("summary_model")
    if "digest_enabled" in body:
        fields["digest_enabled"] = body.get("digest_enabled")
    if "vision_model" in body:
        fields["vision_model"] = body.get("vision_model")
    if not fields:
        return JSONResponse({"error": "no recognized settings keys"}, status_code=400)
    before_vision = (get_global_settings().get("vision_model") or "").strip()
    settings = update_global_settings(fields)
    # The vision model is baked into each agent's auxiliary.vision config —
    # re-init agents so browser_vision/browser_locate pick it up next turn.
    if "vision_model" in fields and (settings.get("vision_model") or "").strip() != before_vision:
        for name in list(daemons.keys()):
            try:
                _update_daemon_cfg(name, load_agents_config()["agents"].get(name, daemons[name].cfg))
            except Exception as e:
                log.warning("vision_model re-init failed for %s: %s", name, e)
    from swarm_server.websocket import _broadcast

    _broadcast("settings_updated", settings)
    return JSONResponse({"status": "ok", "settings": settings})


@app.post("/agent/{agent_name}/human_response")
async def human_response(agent_name: str, request: Request):
    body = await request.json()
    response_text = body.get("response", "")
    if not response_text:
        return JSONResponse({"error": "empty response"}, status_code=400)
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    if daemon.state != AGENT_STATE_ASKING_HUMAN:
        return JSONResponse(
            {"error": f"Agent is not asking human (state: {daemon.state})"},
            status_code=400,
        )
    daemon.human_response = response_text
    daemon.human_event.set()
    return {"status": "ok", "message": "Response sent to agent."}


# ---------------------------------------------------------------------------
# Team Routes
# ---------------------------------------------------------------------------
@app.get("/teams")
async def get_teams():
    cfg = load_agents_config()
    return JSONResponse({"teams": list_teams(cfg)})


@app.post("/teams")
async def post_team(request: Request):
    body = await request.json()
    team_id = body.get("team_id", "").strip()
    name = body.get("name", "").strip()
    if not team_id or not name:
        return JSONResponse({"error": "team_id and name required"}, status_code=400)
    cfg = load_agents_config()
    try:
        team = create_team(cfg, team_id, name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"status": "created", "team": team})


@app.delete("/teams/{team_id}")
async def del_team(team_id: str):
    cfg = load_agents_config()
    # Stop all daemons in this team before deleting
    agents_in_team = [n for n, a in cfg["agents"].items() if a.get("team_id") == team_id]
    for name in agents_in_team:
        _stop_and_unregister_daemon(name)
    if delete_team(cfg, team_id):
        return JSONResponse({"status": "deleted", "team_id": team_id})
    return JSONResponse({"error": "team not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Team credentials (per-team secrets registry; agents read via get_credential)
# ---------------------------------------------------------------------------
@app.get("/teams/{team_id}/credentials")
async def get_team_credentials(team_id: str):
    from swarm_server.credentials import list_credentials_public

    return JSONResponse({"team_id": team_id,
                         "credentials": list_credentials_public(team_id)})


@app.post("/teams/{team_id}/credentials")
async def post_team_credential(team_id: str, request: Request):
    from swarm_server.credentials import save_credential

    body = await request.json()
    try:
        save_credential(
            team_id,
            site=body.get("site", ""),
            username=body.get("username", ""),
            secret=body.get("secret", ""),
            purpose=body.get("purpose", ""),
            notes=body.get("notes", ""),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"status": "saved", "site": body.get("site", "").strip().lower()})


@app.delete("/teams/{team_id}/credentials/{site}")
async def delete_team_credential(team_id: str, site: str):
    from swarm_server.credentials import delete_credential

    if delete_credential(team_id, site):
        return JSONResponse({"status": "deleted", "site": site})
    return JSONResponse({"error": "credential not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Agent Management Routes
# ---------------------------------------------------------------------------
@app.get("/agents")
async def get_agents(team_id: str = None):
    cfg = load_agents_config()
    agents = cfg["agents"]
    if team_id:
        agents = {n: a for n, a in agents.items() if a.get("team_id") == team_id}
    return JSONResponse({"agents": agents, "teams": cfg["teams"]})


@app.post("/agent")
async def add_agent(request: Request):
    body = await request.json()
    agent_name = body.get("agent_name", "").strip()
    display_name = body.get("name", "").strip()
    team_id = body.get("team_id", "").strip()
    allowed_peers = body.get("allowed_peers", [])
    role_soul = body.get("role_soul") or body.get("soul")  # accept both for backward compat
    is_supervisor = bool(body.get("is_supervisor"))

    if not agent_name or not display_name or not team_id:
        return JSONResponse(
            {"error": "agent_name, name (display), and team_id are required"},
            status_code=400,
        )

    cfg = load_agents_config()
    if team_id not in cfg["teams"]:
        return JSONResponse({"error": f"Team '{team_id}' not found"}, status_code=404)

    try:
        agent_cfg = create_agent(
            cfg,
            name=agent_name,
            team_id=team_id,
            display_name=display_name,
            allowed_peers=allowed_peers,
            role_soul=role_soul,
            is_supervisor=is_supervisor,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    if agent_name not in daemons:
        loop = asyncio.get_running_loop()
        register_agent_daemon(agent_name, agent_cfg, loop)

    return JSONResponse({
        "status": "created",
        "agent_name": agent_name,
        "team_id": team_id,
    })


@app.delete("/agent/{agent_name}")
async def remove_agent(agent_name: str):
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)

    _stop_and_unregister_daemon(agent_name)
    delete_agent(cfg, agent_name)
    return JSONResponse({"status": "deleted", "agent_name": agent_name})


def _update_daemon_cfg(agent_name: str, new_cfg: Dict[str, Any]):
    daemon = daemons.get(agent_name)
    if daemon is not None:
        with daemon._lock:
            daemon.cfg = new_cfg
            # Force a re-init so model/sampling/soul changes take effect next turn.
            daemon._ai_agent = None
            # Refresh runtime knobs read outside _ensure_agent (heartbeat + sweep).
            daemon._autonomous = bool(new_cfg.get("autonomous", False))
            daemon._sweep_interval = daemon._resolve_sweep_interval(new_cfg)
            daemon._heartbeat_seconds = daemon._resolve_heartbeat_interval(new_cfg)
            daemon._load_crons(new_cfg)


@app.get("/agent/{agent_name}/peers")
async def get_agent_peers(agent_name: str):
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    return JSONResponse({
        "agent_name": agent_name,
        "allowed_peers": cfg["agents"][agent_name].get("allowed_peers", []),
    })


@app.post("/agent/{agent_name}/peers")
async def add_peers(agent_name: str, request: Request):
    body = await request.json()
    peers = body.get("peers", [])
    if not isinstance(peers, list):
        return JSONResponse({"error": "peers must be a list"}, status_code=400)

    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)

    try:
        set_agent_peers(cfg, agent_name, peers)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _update_daemon_cfg(agent_name, cfg["agents"][agent_name])
    return JSONResponse({
        "status": "updated",
        "agent_name": agent_name,
        "allowed_peers": peers,
    })


@app.delete("/agent/{agent_name}/peers/{peer_name}")
async def del_peer(agent_name: str, peer_name: str):
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    remove_agent_peer(cfg, agent_name, peer_name)

    # Also remove the reverse link if it exists
    if peer_name in cfg["agents"] and agent_name in cfg["agents"][peer_name].get("allowed_peers", []):
        remove_agent_peer(cfg, peer_name, agent_name)
        if peer_name in daemons:
            _update_daemon_cfg(peer_name, cfg["agents"][peer_name])

    _update_daemon_cfg(agent_name, cfg["agents"][agent_name])
    return JSONResponse({
        "status": "removed",
        "agent_name": agent_name,
        "removed_peer": peer_name,
    })


@app.post("/agent/{agent_name}/soul")
async def update_agent_soul(agent_name: str, request: Request):
    body = await request.json()
    role_soul = body.get("role_soul") or body.get("soul")
    if not role_soul:
        return JSONResponse({"error": "Missing 'role_soul' field"}, status_code=400)

    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "Agent not found"}, status_code=404)

    cfg["agents"][agent_name]["role_soul"] = role_soul
    save_agent_config(agent_name, cfg["agents"][agent_name])

    if agent_name in daemons:
        daemon = daemons[agent_name]
        with daemon._lock:
            daemon.cfg = cfg["agents"][agent_name]
            daemon._ai_agent = None
        log.info("[Dynamic Registry] Role soul updated for agent '%s'", agent_name)

    from swarm_server.websocket import _broadcast

    _broadcast("soul_updated", {"agent_name": agent_name, "timestamp": __import__("time").time()})
    return JSONResponse({"status": "success", "message": f"Role soul for '{agent_name}' updated."})


# ---------------------------------------------------------------------------
# Per-Agent Configuration (model + sampling + runtime knobs)
# ---------------------------------------------------------------------------
# Fields the UI may edit via the config endpoint. Everything else in the stored
# cfg (team_id, session_id, allowed_peers, …) is left untouched.
_EDITABLE_AGENT_FIELDS = {
    "name", "model", "provider", "autonomous", "sweep_interval", "heartbeat_seconds",
    "temperature", "max_tokens", "reasoning_effort", "max_iterations",
    "enabled_toolsets", "disabled_toolsets", "compression_threshold", "role_soul",
    "is_supervisor", "supervisor_interval_minutes", "context_isolated",
}
# Numeric fields where an empty value means "clear → use the default".
_NUMERIC_CLEARABLE = (
    "sweep_interval", "heartbeat_seconds", "temperature", "max_tokens",
    "max_iterations", "compression_threshold", "supervisor_interval_minutes",
)


def _apply_config_fields(base: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    """Merge UI/agent-supplied editable fields into a copy of the stored cfg.

    Empty values clear optional overrides (revert to default). Toolset fields
    accept a comma-string or a list. Shared by the PATCH endpoint and the
    self-config proposal-approval path so both normalize identically.
    """
    updated = dict(base)
    for key in _EDITABLE_AGENT_FIELDS:
        if key not in body:
            continue
        val = body[key]
        if key in _NUMERIC_CLEARABLE and val in ("", None):
            updated.pop(key, None)
            continue
        if key in ("reasoning_effort", "provider", "model") and val in ("", None, "default"):
            # Empty model/provider clears the per-agent override → inherit the
            # swarm default. (provider "default" sentinel handled the same way.)
            updated.pop(key, None)
            continue
        if key in ("enabled_toolsets", "disabled_toolsets"):
            if isinstance(val, str):
                val = [s.strip() for s in val.split(",") if s.strip()]
            if not val:
                updated.pop(key, None)
                continue
        if key in ("autonomous", "is_supervisor"):
            val = bool(val)
        updated[key] = val
    return updated


@app.get("/models")
async def get_models():
    """Model ids the CONFIGURED default backend serves (for the config dropdown)."""
    from swarm_server.model_config import resolve_model, get_default_model, _preset

    eff = resolve_model({})  # the default backend (no per-agent override)
    served = list_proxy_models(eff.get("base_url") or None, eff.get("api_key") or None)
    # Also offer the chosen provider's known models (helps native providers like
    # Anthropic where there's no /models listing).
    preset = _preset(get_default_model().get("provider") or "")
    extra = [m for m in preset.get("models", []) if m not in served]
    return JSONResponse({"models": served + extra, "default": eff.get("model") or DEFAULT_MODEL})


@app.get("/providers")
async def get_providers():
    """Provider catalogue for the setup screen (Hermes registry + OpenRouter/Custom)."""
    from swarm_server.model_config import build_provider_presets

    return JSONResponse({"providers": build_provider_presets()})


@app.get("/setup/status")
async def setup_status():
    """Whether a model is configured (drives the first-run setup screen)."""
    from swarm_server.model_config import (
        is_model_configured, get_default_model, detect_global_hermes_model,
    )

    default = get_default_model()
    detected = detect_global_hermes_model()
    return JSONResponse({
        "configured": is_model_configured(),
        "default": {
            "provider": default.get("provider"), "model": default.get("model"),
            "base_url": default.get("base_url"), "has_key": bool(default.get("api_key")),
        },
        "detected_hermes": {
            "provider": detected.get("provider"), "model": detected.get("model"),
        } if detected.get("model") else None,
    })


@app.post("/setup/model")
async def setup_model(request: Request):
    """Set the swarm-wide DEFAULT model and re-init all agents to pick it up."""
    from swarm_server.model_config import set_default_model, detect_global_hermes_model

    body = await request.json()
    # Convenience: "adopt" the model detected in the user's ~/.hermes.
    if body.get("adopt_detected"):
        det = detect_global_hermes_model()
        if not det.get("model"):
            return JSONResponse({"error": "No Hermes model detected to adopt."}, status_code=400)
        body = {
            "provider": det.get("provider") or "custom", "model": det["model"],
            "base_url": det.get("base_url", ""), "api_key": det.get("api_key", ""),
        }

    model = (body.get("model") or "").strip()
    provider = (body.get("provider") or "custom").strip()
    base_url = (body.get("base_url") or "").strip()
    api_key = (body.get("api_key") or "").strip()
    if not model:
        return JSONResponse({"error": "model is required"}, status_code=400)

    set_default_model(provider, model, base_url, api_key)
    # Re-init every agent so the new default takes effect on its next turn.
    for name in list(daemons.keys()):
        _update_daemon_cfg(name, load_agents_config()["agents"].get(name, daemons[name].cfg))

    from swarm_server.websocket import _broadcast

    _broadcast("model_default_updated", {
        "provider": provider, "model": model, "timestamp": __import__("time").time(),
    })
    return JSONResponse({"status": "ok", "provider": provider, "model": model})


@app.get("/agent/{agent_name}/config")
async def get_agent_config(agent_name: str):
    cfg = load_agents_config()
    a = cfg["agents"].get(agent_name)
    if a is None:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    daemon = daemons.get(agent_name)
    telemetry = dict(getattr(daemon, "_telemetry", {}) or {}) if daemon else {}
    return JSONResponse({
        "agent_name": agent_name,
        "name": a.get("name", agent_name),
        "team_id": a.get("team_id"),
        "model": a.get("model"),  # raw per-agent override (None = inherit default)
        "effective_model": __import__("swarm_server.model_config", fromlist=["resolve_model"]).resolve_model(a).get("model"),
        "provider": a.get("provider"),
        "autonomous": bool(a.get("autonomous", False)),
        "sweep_interval": a.get("sweep_interval"),
        "heartbeat_seconds": a.get("heartbeat_seconds"),
        "temperature": a.get("temperature"),
        "max_tokens": a.get("max_tokens"),
        "reasoning_effort": a.get("reasoning_effort"),
        "max_iterations": a.get("max_iterations"),
        "enabled_toolsets": a.get("enabled_toolsets") or [],
        "disabled_toolsets": a.get("disabled_toolsets") or [],
        "compression_threshold": a.get("compression_threshold"),
        "role_soul": a.get("role_soul") or a.get("soul") or "",
        "allowed_peers": a.get("allowed_peers", []),
        "is_supervisor": bool(a.get("is_supervisor", False)),
        "supervisor_interval_minutes": a.get("supervisor_interval_minutes"),
        "hermes_home": str(getattr(daemon, "_hermes_home", "")) if daemon else "",
        "telemetry": telemetry,
    })


@app.patch("/agent/{agent_name}/config")
async def patch_agent_config(agent_name: str, request: Request):
    """Merge UI-editable fields into an agent's stored config and hot-apply them.

    The model / sampling / soul changes take effect on the agent's NEXT turn
    (forced re-init via _update_daemon_cfg, which nulls the cached AIAgent);
    autonomous + sweep_interval are refreshed immediately. Empty values for the
    optional overrides clear them, reverting to the Hermes/global default.
    """
    body = await request.json()
    cfg = load_agents_config()
    a = cfg["agents"].get(agent_name)
    if a is None:
        return JSONResponse({"error": "agent not found"}, status_code=404)

    updated = _apply_config_fields(a, body)

    save_agent_config(agent_name, updated)
    _update_daemon_cfg(agent_name, updated)

    from swarm_server.websocket import _broadcast

    _broadcast("agent_config_updated", {
        "agent_name": agent_name,
        "timestamp": __import__("time").time(),
    })
    return JSONResponse({"status": "updated", "agent_name": agent_name, "config": updated})


# ---------------------------------------------------------------------------
# Per-agent cron wake-ups (CRUD). Schedules are validated in config; the running
# daemon is refreshed so changes take effect on its next sweep tick.
# ---------------------------------------------------------------------------
def _crons_with_runtime(agent_name: str, stored: list) -> list:
    """Merge stored crons with the daemon's live next/last-fire timestamps and a
    human-readable schedule description for the dashboard."""
    from swarm_server.cron import cron_describe

    daemon = daemons.get(agent_name)
    runtime = {c.get("id"): c for c in daemon.crons_runtime()} if daemon else {}
    out = []
    for c in stored:
        rt = runtime.get(c.get("id"), {})
        out.append({
            **c,
            "describe": cron_describe(c.get("schedule", "")),
            "next_fire_at": rt.get("next_fire_at"),
            "last_fired_at": rt.get("last_fired_at"),
        })
    return out


@app.get("/agent/{agent_name}/crons")
async def get_agent_crons(agent_name: str):
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    stored = list_agent_crons(cfg, agent_name)
    return JSONResponse({"agent_name": agent_name, "crons": _crons_with_runtime(agent_name, stored)})


@app.post("/agent/{agent_name}/crons")
async def post_agent_cron(agent_name: str, request: Request):
    body = await request.json()
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    try:
        entry = add_agent_cron(
            cfg, agent_name,
            schedule=body.get("schedule", ""),
            instruction=body.get("instruction", ""),
            enabled=body.get("enabled", True),
            created_by="human",
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _refresh_daemon_crons(agent_name)
    return JSONResponse({"status": "created", "cron": entry})


@app.patch("/agent/{agent_name}/crons/{cron_id}")
async def patch_agent_cron(agent_name: str, cron_id: str, request: Request):
    body = await request.json()
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    fields = {k: body[k] for k in ("schedule", "instruction", "enabled") if k in body}
    try:
        entry = update_agent_cron(cfg, agent_name, cron_id, fields)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _refresh_daemon_crons(agent_name)
    return JSONResponse({"status": "updated", "cron": entry})


@app.delete("/agent/{agent_name}/crons/{cron_id}")
async def del_agent_cron(agent_name: str, cron_id: str):
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    removed = remove_agent_cron(cfg, agent_name, cron_id)
    if not removed:
        return JSONResponse({"error": "cron not found"}, status_code=404)
    _refresh_daemon_crons(agent_name)
    return JSONResponse({"status": "deleted", "cron_id": cron_id})


def _refresh_daemon_crons(agent_name: str):
    """Reload the agent's crons into its running daemon and notify the dashboard."""
    daemon = daemons.get(agent_name)
    if daemon is not None:
        entry = load_agents_config()["agents"].get(agent_name, daemon.cfg)
        with daemon._lock:
            daemon.cfg = entry
            daemon._load_crons(entry)
    from swarm_server.websocket import _broadcast

    _broadcast("cron_updated", {"agent_name": agent_name, "timestamp": __import__("time").time()})


@app.get("/toolsets")
async def get_toolsets():
    """Hermes toolsets + swarm-native tools (for the allowed/disabled-tools picker)."""
    from swarm_server.tools import list_swarm_tools
    return JSONResponse({"toolsets": list_toolsets(), "swarm_tools": list_swarm_tools()})


@app.get("/agent/{agent_name}/cli")
async def get_agent_cli(agent_name: str):
    """Ready-to-paste Hermes CLI commands for configuring THIS agent in a terminal.

    Each command exports the agent's isolated HERMES_HOME so `hermes` operates on
    that specific agent's config — letting the operator run the interactive setup
    wizard (model, gateway/Telegram, tools, …) without the dashboard.
    """
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    home = str(daemon._hermes_home)
    prefix = f"HERMES_HOME='{home}'"
    commands = [
        {"label": "Full interactive setup wizard", "cmd": f"{prefix} hermes setup"},
        {"label": "Model / provider", "cmd": f"{prefix} hermes setup model"},
        {"label": "Messaging gateways (Telegram, Discord, Slack…)", "cmd": f"{prefix} hermes setup gateway"},
        {"label": "Tools", "cmd": f"{prefix} hermes setup tools"},
        {"label": "Agent settings (iterations, compression)", "cmd": f"{prefix} hermes setup agent"},
        {"label": "Set a single config key", "cmd": f"{prefix} hermes config set model.provider custom"},
        {"label": "Open a chat REPL as this agent", "cmd": f"{prefix} hermes"},
    ]
    return JSONResponse({"agent_name": agent_name, "hermes_home": home, "commands": commands})


# ---------------------------------------------------------------------------
# Self-config Proposals (agent proposes → human approves in the UI)
# ---------------------------------------------------------------------------
@app.get("/proposals")
async def get_proposals():
    from swarm_server.tools import get_config_proposals

    proposals = get_config_proposals()
    proposals.sort(key=lambda p: p["timestamp"], reverse=True)
    return JSONResponse({
        "proposals": proposals[:200],
        "pending_count": sum(1 for p in proposals if p["status"] == "pending"),
    })


@app.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: str):
    from swarm_server.tools import get_config_proposals, resolve_config_proposal
    from swarm_server.websocket import _broadcast

    target = next((p for p in get_config_proposals() if p["id"] == proposal_id), None)
    if target is None:
        return JSONResponse({"error": "proposal not found"}, status_code=404)
    if target["status"] != "pending":
        return JSONResponse({"error": f"proposal already {target['status']}"}, status_code=409)

    agent_name = target["agent_name"]
    cfg = load_agents_config()
    a = cfg["agents"].get(agent_name)
    if a is None:
        resolve_config_proposal(proposal_id, "rejected")
        return JSONResponse({"error": "agent no longer exists"}, status_code=404)

    updated = _apply_config_fields(a, target["changes"])
    save_agent_config(agent_name, updated)
    _update_daemon_cfg(agent_name, updated)
    resolve_config_proposal(proposal_id, "approved")

    _broadcast("proposal_resolved", {
        "agent_name": agent_name,
        "proposal_id": proposal_id,
        "status": "approved",
        "timestamp": __import__("time").time(),
    })
    # Tell the agent its proposal landed so it can proceed accordingly.
    daemon = daemons.get(agent_name)
    if daemon is not None:
        daemon.ingest_task("human", (
            f"✅ Your config-change proposal was APPROVED and applied: {target['changes']}. "
            "It takes effect on this turn (you were re-initialised). Continue your work."
        ))
    return JSONResponse({"status": "approved", "agent_name": agent_name, "config": updated})


@app.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: str):
    from swarm_server.tools import resolve_config_proposal
    from swarm_server.websocket import _broadcast

    resolved = resolve_config_proposal(proposal_id, "rejected")
    if resolved is None:
        return JSONResponse({"error": "proposal not found"}, status_code=404)
    _broadcast("proposal_resolved", {
        "agent_name": resolved["agent_name"],
        "proposal_id": proposal_id,
        "status": "rejected",
        "timestamp": __import__("time").time(),
    })
    return JSONResponse({"status": "rejected", "proposal_id": proposal_id})


# ---------------------------------------------------------------------------
# Monitoring Routes
# ---------------------------------------------------------------------------
@app.get("/monitoring/agents")
async def monitoring_agents(team_id: str = None):
    import time

    cfg = load_agents_config()
    result = {}
    for name, d in daemons.items():
        if team_id and d.cfg.get("team_id") != team_id:
            continue
        result[name] = {
            "state": d.state,
            "pending_count": d.queue.get_pending_count(),
            "next_sweep_at": d.next_sweep_at,
            "config": d.cfg,
            "allowed_peers": d.cfg.get("allowed_peers", []),
            "telemetry": dict(getattr(d, "_telemetry", {}) or {}),
            "digest": _shape_digest(monitor_db.get_last_digest(name)),
        }
    return JSONResponse({
        "agents": result,
        "timestamp": time.time(),
    })


@app.get("/monitoring/agents/{agent_name}/events")
async def monitoring_events(agent_name: str, limit: int = 50):
    events = monitor_db.get_events(agent_name=agent_name, limit=limit)
    return JSONResponse({"agent": agent_name, "events": events})


@app.get("/monitoring/agents/{agent_name}/messages")
async def monitoring_messages(agent_name: str, limit: int = 200, offset: int = 0):
    messages = monitor_db.get_messages(agent_name=agent_name, limit=limit, offset=offset)
    messages.reverse()
    return JSONResponse({"agent": agent_name, "messages": messages})


@app.get("/monitoring/agents/{agent_name}/queue")
async def monitoring_queue(agent_name: str):
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    tasks = daemon.queue.get_all_tasks(limit=100)
    return JSONResponse({
        "agent": agent_name,
        "pending_count": daemon.queue.get_pending_count(),
        "tasks": tasks,
    })


@app.get("/monitoring/stats")
async def monitoring_stats(team_id: str = None):
    import time

    stats = monitor_db.get_agent_stats(team_id=team_id)
    for name, daemon in daemons.items():
        if team_id and daemon.cfg.get("team_id") != team_id:
            continue
        if name not in stats:
            stats[name] = {"events": {}, "total_messages": 0}
        stats[name]["current_state"] = daemon.state
        stats[name]["pending_count"] = daemon.queue.get_pending_count()
    return JSONResponse({"stats": stats, "timestamp": time.time()})


@app.get("/monitoring/recent_events")
async def monitoring_recent(limit: int = 100):
    events = monitor_db.get_events(limit=limit)
    return JSONResponse({"events": events})


# ---------------------------------------------------------------------------
# Stop Execution
# ---------------------------------------------------------------------------
@app.post("/agent/{agent_name}/stop")
async def stop_agent_execution(agent_name: str):
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    if daemon.state == "idle":
        return JSONResponse({"success": True, "message": f"Agent '{agent_name}' is already idle."})
    await daemon.stop_execution()
    return JSONResponse({"success": True, "message": f"Execution stopped for '{agent_name}'."})


@app.post("/agent/{agent_name}/pause")
async def pause_agent_execution(agent_name: str, request: Request):
    """Emergency brake (human/dashboard). Freezes the agent mid-turn but keeps
    its queue — resume with /resume. Distinct from /stop, which drains the queue."""
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    reason = ""
    try:
        body = await request.json()
        reason = (body or {}).get("reason", "")
    except Exception:
        pass
    daemon.pause_execution(reason=reason or "Paused by operator", by="human")
    return JSONResponse({"success": True, "message": f"Agent '{agent_name}' paused."})


@app.post("/agent/{agent_name}/resume")
async def resume_agent_execution(agent_name: str):
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    if not getattr(daemon, "_paused", False):
        return JSONResponse({"success": True, "message": f"Agent '{agent_name}' is not paused."})
    daemon.resume_execution(by="human")
    return JSONResponse({"success": True, "message": f"Agent '{agent_name}' resumed."})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    cfg = load_agents_config()
    return {
        "status": "ok",
        "agents": list(daemons.keys()),
        "teams": list(cfg["teams"].keys()),
    }


# ---------------------------------------------------------------------------
# Dashboard Root
# ---------------------------------------------------------------------------
@app.get("/")
async def root_dashboard():
    dashboard_file = DASHBOARD_DIR / "index.html"
    if dashboard_file.exists():
        return FileResponse(str(dashboard_file))
    return HTMLResponse(
        "<h1>Dashboard not found</h1><p>Run from project root.</p>",
        status_code=404,
    )


# ---------------------------------------------------------------------------
# Human Inbox Endpoints
# ---------------------------------------------------------------------------
@app.get("/inbox")
async def get_human_inbox():
    from swarm_server.tools import get_pending_questions

    questions = get_pending_questions()
    # Sort by timestamp desc, keep only last 50 to avoid bloat
    questions.sort(key=lambda q: q["timestamp"], reverse=True)
    return JSONResponse({
        "questions": questions[:200],
        "pending_count": sum(1 for q in questions if q["status"] == "pending"),
    })


@app.post("/inbox/{agent_name}/respond")
async def respond_to_human_question(agent_name: str, request: Request):
    from swarm_server.tools import _pending_human_questions, _pending_lock, answer_question

    body = await request.json()
    response_text = body.get("response", "")
    question_id = body.get("question_id")

    if not response_text.strip():
        return JSONResponse({"error": "Empty response"}, status_code=400)

    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)

    # If a question_id was passed, validate it matches current daemon question
    if question_id:
        current_qid = getattr(daemon, "human_question_id", None)
        if current_qid != question_id:
            # Still update registry for historical record, but warn
            log.warning(
                "[Inbox] QID mismatch for %s: daemon has %s, request sent %s",
                agent_name, current_qid, question_id,
            )

    # Find the most recent pending question for this agent
    found_qid = question_id
    if not found_qid:
        with _pending_lock:
            for qid, q in sorted(
                _pending_human_questions.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True,
            ):
                if q["agent_name"] == agent_name and q["status"] == "pending":
                    found_qid = qid
                    break

    if not found_qid:
        return JSONResponse(
            {"error": f"No pending question found for agent '{agent_name}'."},
            status_code=404,
        )

    # Grab the question text (for the resume-task path) before marking answered.
    from swarm_server.tools import _pending_human_questions, _pending_lock as _pl
    with _pl:
        q_obj = _pending_human_questions.get(found_qid) or {}
        question_text = q_obj.get("question", "")

    # Update registry
    answer_question(found_qid, response_text)

    # Deliver the answer. Two cases:
    if daemon.state == AGENT_STATE_ASKING_HUMAN:
        # (a) Agent is still blocking inside ask_human → resume it in-turn.
        daemon.human_response = response_text
        daemon.human_event.set()
        log.info("[Inbox] Response delivered in-turn to %s (qid=%s)", agent_name, found_qid)
    else:
        # (b) Agent already ended its turn (the in-turn wait elapsed). Re-deliver
        # the answer as a NEW TASK so it wakes up and resumes the blocked action
        # (e.g. log in and publish). This is what lets the swarm survive a human
        # who answers hours later — the request never gets lost.
        daemon.human_event.set()  # harmless if nothing is waiting
        # If this was a browser takeover the human did out-of-turn, send the
        # browser back to its hidden display now (no-op for ordinary questions).
        try:
            from swarm_server.browser_pool import team_browser_manager
            team_browser_manager.end_takeover((daemon.cfg or {}).get("team_id", "default"))
        except Exception as _e:
            log.warning("[Inbox] end_takeover after delayed answer failed: %s", _e)
        resume_msg = (
            "✅ The human answered a question you were blocked on.\n\n"
            f"Your question: {question_text}\n"
            f"Human's answer: {response_text}\n\n"
            "Resume the action you were blocked on and COMPLETE it now "
            "(e.g. use the credentials to log in via the browser and publish). "
            "Do not just acknowledge — finish the task and report what went live."
        )
        daemon.ingest_task("human", resume_msg)
        log.info("[Inbox] Response re-delivered as resume-task to %s (qid=%s)", agent_name, found_qid)

    from swarm_server.websocket import _broadcast

    _broadcast("human_responded", {
        "agent_name": agent_name,
        "question_id": found_qid,
        "response_preview": response_text[:120],
        "timestamp": __import__("time").time(),
    })
    return JSONResponse({
        "success": True,
        "question_id": found_qid,
        "message": "Response delivered to agent." if daemon.state == AGENT_STATE_ASKING_HUMAN else "Response stored (agent already moved on).",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def register_agent_daemon(agent_name: str, cfg: Dict[str, Any], loop: asyncio.AbstractEventLoop):
    daemon = AgentDaemon(agent_name, cfg)
    daemons[agent_name] = daemon
    _daemon_registry[agent_name] = daemon
    daemon.start_sweep(loop)
    log.info("[Dynamic Registry] Registered agent '%s' daemon", agent_name)


def _stop_daemon(daemon: AgentDaemon):
    if daemon._sweep_task and not daemon._sweep_task.done():
        daemon._sweep_task.cancel()
        log.info("[Daemon] Cancelled sweep for '%s'", daemon.name)
    # Release the agent's dedicated worker thread. wait=False so shutdown never
    # blocks the event loop on an in-flight (possibly ask_human-blocked) run.
    daemon._executor.shutdown(wait=False, cancel_futures=True)


def _stop_and_unregister_daemon(agent_name: str):
    daemon = daemons.get(agent_name)
    if daemon is None:
        return
    _stop_daemon(daemon)
    daemons.pop(agent_name, None)
    _daemon_registry.pop(agent_name, None)
    log.info("[Daemon] Unregistered '%s'", agent_name)
