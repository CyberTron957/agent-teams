"""FastAPI application, REST routes, WebSocket endpoint, and lifecycle management."""

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from swarm_server import __version__
from swarm_server.agent import AgentDaemon
from swarm_server.config import (
    AGENTS,
    CORS_ALLOWED_ORIGINS,
    DASHBOARD_DIR,
    DEFAULT_MODEL,
    DIGEST_SWEEP_INTERVAL_SECONDS,
    list_backend_models,
    list_toolsets,
    MONITORING_DB,
    MONITORING_MAX_EVENTS,
    MONITORING_MAX_MESSAGES,
    MONITORING_MAX_DIGESTS,
    MONITORING_MAX_DELEGATIONS,
    MONITORING_MAX_ACTIONS,
    MONITORING_MAX_DECISIONS,
    MONITORING_MAX_MILESTONES,
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
            monitor_db.prune(
                MONITORING_MAX_EVENTS, MONITORING_MAX_MESSAGES,
                max_digests=MONITORING_MAX_DIGESTS,
                max_delegations=MONITORING_MAX_DELEGATIONS,
                max_actions=MONITORING_MAX_ACTIONS,
                max_decisions=MONITORING_MAX_DECISIONS,
                max_milestones=MONITORING_MAX_MILESTONES,
            )
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

    # Verify the fragile Hermes seams the swarm depends on still hold in the
    # installed Hermes. Warn-only: a Hermes update that moved an internal API
    # should be LOUD here, never a silent feature regression nor a boot blocker.
    try:
        from swarm_server.hermes_compat import log_self_check
        log_self_check()
    except Exception as e:
        log.warning("[Startup] Hermes compat self-check failed to run: %s", e)

    # Inherit the user's existing Hermes secrets (~/.hermes/.env) — provider AND
    # tool API keys (Firecrawl/Tavily/Exa/…) — into this process so every agent
    # can use them. Non-overriding, so explicit server/deployment env still wins.
    try:
        from swarm_server.model_config import import_hermes_secrets
        import_hermes_secrets()
    except Exception as e:
        log.warning("[Startup] importing ~/.hermes secrets failed: %s", e)

    # Rebuild today's per-team spend meter from monitoring.db BEFORE daemons
    # start sweeping, so a mid-day restart resumes with the correct budget state
    # (a team already over its cap stays paused instead of getting a free reset).
    try:
        from swarm_server.budget import budget_tracker
        budget_tracker.rebuild(monitor_db, cfg)
    except Exception as e:
        log.warning("[Startup] budget rebuild failed: %s", e)

    for agent_name, agent_cfg in cfg["agents"].items():
        register_agent_daemon(agent_name, agent_cfg, loop)

    # Wire the Architect (master team-builder): register its toolset (so team
    # agents' swarm_master deny-guard has something to deny) and inject the
    # thread-safe daemon lifecycle hooks it uses to spawn/despawn/update agents.
    _wire_master(loop)

    prune_task = loop.create_task(_periodic_monitoring_prune())
    digest_task = loop.create_task(_periodic_digest())
    loopdet_task = loop.create_task(_periodic_loop_detector())
    try:
        from swarm_server.model_config import resolve_model

        _eff = resolve_model({})
        log.info("[Startup] All agents running. Model: %s (source: %s)",
                 _eff.get("model") or "none — run `hermes setup`", _eff.get("source"))
    except Exception:
        log.info("[Startup] All agents running.")
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


app = FastAPI(title="Agent Teams Server", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths reachable WITHOUT the API key even when auth is on: the dashboard shell
# itself (so the login UI can render), the load-balancer health probe, and the
# self-reporting auth probe the dashboard hits at boot. None of these has a
# mutating handler, so exempting them by exact path (any method) is safe.
AUTH_EXEMPT_PATHS = {"/", "/health", "/auth/check", "/version/check"}

# Application-defined WebSocket close code for an unauthenticated socket
# (4000-4999 is the private-use range). The dashboard re-opens the login modal
# when it sees this.
WS_CLOSE_UNAUTHORIZED = 4401
WS_AUTH_TIMEOUT_SECONDS = 10.0


def _request_key_ok(request: Request) -> bool:
    """True iff the request carries the correct API key in either
    'X-API-Key: <key>' or 'Authorization: Bearer <key>'. Constant-time compare."""
    provided = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    if not provided and auth.lower().startswith("bearer "):
        provided = auth[7:].strip()
    return bool(provided) and secrets.compare_digest(provided, SWARM_API_KEY)


@app.middleware("http")
async def _auth_guard(request: Request, call_next):
    """Optional single-key auth on ALL endpoints.

    Disabled unless SWARM_API_KEY is set (the server then assumes a trusted
    localhost bind). When set, every request — reads included, since agent
    conversations and masked credentials are sensitive — needs the key, except
    the small AUTH_EXEMPT_PATHS allow-list. WebSockets are guarded separately by
    _ws_authenticate (HTTP middleware never sees WS scopes).
    """
    if SWARM_API_KEY and request.url.path not in AUTH_EXEMPT_PATHS:
        if not _request_key_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


async def _ws_authenticate(ws: WebSocket) -> bool:
    """Accept the socket, then (if auth is on) require a first message
    ``{"action":"auth","api_key":"..."}`` within the timeout before any data is
    sent. Returns True iff the connection may proceed. Reused by every WS
    endpoint, including the browser-control stream. Sends nothing pre-auth."""
    await ws.accept()
    if not SWARM_API_KEY:
        return True
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=WS_AUTH_TIMEOUT_SECONDS)
        msg = json.loads(raw)
        key = str(msg.get("api_key", ""))
        if msg.get("action") == "auth" and key and secrets.compare_digest(key, SWARM_API_KEY):
            await ws.send_text(json.dumps({"type": "auth_ok", "payload": {}}))
            return True
    except Exception:
        pass
    try:
        await ws.close(code=WS_CLOSE_UNAUTHORIZED)
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    import time

    if not await _ws_authenticate(ws):
        return
    await ws_broadcaster.register(ws)
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


@app.websocket("/teams/{team_id}/browser/ws")
async def browser_stream_endpoint(ws: WebSocket, team_id: str):
    """Live view + control of a team's headless browser (embedded handover).

    This socket grants FULL control of the team browser session, so it is
    authenticated exactly like /ws (same first-message scheme). It does not join
    the broadcast pool — it's a private 1:1 relay to that team's Chrome.
    """
    if not await _ws_authenticate(ws):
        return
    cfg = load_agents_config()
    if team_id not in cfg.get("teams", {}):
        await ws.send_text(json.dumps({"type": "error",
            "payload": {"message": f"Unknown team '{team_id}'."}}))
        await ws.close()
        return
    from swarm_server import browser_stream
    try:
        await browser_stream.relay(ws, team_id)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("[browser-stream] %s: %s", team_id, e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


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
    resp = {"task_id": task_id, "status": "queued"}
    # Optional time-boxed directive: until it expires, the agent's idle
    # heartbeats re-present this payload (base cadence, no backoff) so a
    # multi-hour push survives the queue going empty instead of being
    # treated as a one-shot task.
    duration = body.get("duration_minutes")
    if duration:
        import time

        try:
            daemon.set_directive(payload, float(duration), from_agent=from_agent)
            resp["directive_until"] = time.time() + float(duration) * 60.0
        except (TypeError, ValueError):
            pass
    return JSONResponse(resp)


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
    if "master_model" in body:
        fields["master_model"] = body.get("master_model")
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
    # The Architect picks up a new master_model on its next turn.
    if "master_model" in fields:
        try:
            from swarm_server.master import get_master

            get_master().reload()
        except Exception as e:  # noqa: BLE001
            log.warning("master reload failed: %s", e)
    from swarm_server.websocket import _broadcast

    _broadcast("settings_updated", settings)
    return JSONResponse({"status": "ok", "settings": settings})


async def _end_takeover_async(team_id: str) -> None:
    """Hand the team browser back to its hidden display WITHOUT blocking the event
    loop. end_takeover does sync urllib / time.sleep polling / a subprocess
    relaunch (multiple seconds), so it must run off the loop or it freezes every
    agent's sweep, all WS broadcasts, and every other request."""
    try:
        from swarm_server.browser_pool import team_browser_manager
        await asyncio.get_running_loop().run_in_executor(
            None, team_browser_manager.end_takeover, team_id
        )
    except Exception as e:
        log.warning("[human] end_takeover failed: %s", e)


async def _finalize_human_answer(daemon, qid: str, response_text: str) -> Dict[str, Any]:
    """Route a human's answer through the single atomic delivery point and run any
    follow-up (resume-task enqueue, browser hand-back). Returns a JSON-able dict."""
    from swarm_server.tools import deliver_human_answer
    from swarm_server.websocket import _broadcast

    result = deliver_human_answer(qid, response_text)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "delivery failed")}

    _broadcast("human_responded", {
        "agent_name": result["agent"],
        "question_id": qid,
        "response_preview": response_text[:120],
        "timestamp": __import__("time").time(),
    })

    if result["delivery"] == "task":
        # The agent already ended its turn (or this answers a stale/old question).
        # Only a real browser TAKEOVER hands the browser back here — an ordinary
        # question must NOT call end_takeover (it would yank a DIFFERENT agent's
        # live takeover window, or spawn an unused browser). Run it off the loop.
        if result.get("kind") == "takeover":
            await _end_takeover_async((daemon.cfg or {}).get("team_id", "default"))
        resume_msg = (
            "✅ The human answered a question you were blocked on.\n\n"
            f"Your question: {result.get('question', '')}\n"
            f"Human's answer: {response_text}\n\n"
            "Resume the action you were blocked on and COMPLETE it now "
            "(e.g. use the credentials to log in via the browser and publish). "
            "Do not just acknowledge — finish the task and report what went live."
        )
        daemon.ingest_task("human", resume_msg)
    return {"ok": True, "delivery": result["delivery"], "question_id": qid}


def _resolve_pending_qid(agent_name: str, daemon) -> Optional[str]:
    """The question this agent is (or was last) blocking on: its current
    human_question_id if still pending, else the newest pending question."""
    from swarm_server.tools import _pending_human_questions, _pending_lock
    with _pending_lock:
        cur = getattr(daemon, "human_question_id", None)
        q = _pending_human_questions.get(cur) if cur else None
        if q and q["status"] == "pending":
            return cur
        newest = None
        for qid, q in _pending_human_questions.items():
            if q["agent_name"] == agent_name and q["status"] == "pending":
                if newest is None or q["timestamp"] > newest[1]:
                    newest = (qid, q["timestamp"])
        return newest[0] if newest else None


@app.post("/agent/{agent_name}/human_response")
async def human_response(agent_name: str, request: Request):
    body = await request.json()
    response_text = body.get("response", "")
    if not response_text:
        return JSONResponse({"error": "empty response"}, status_code=400)
    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    qid = _resolve_pending_qid(agent_name, daemon)
    if not qid:
        return JSONResponse(
            {"error": f"Agent has no pending question (state: {daemon.state})"},
            status_code=400,
        )
    # Route through the atomic deliverer: it marks the question answered (so it
    # doesn't linger 'pending' forever and inflate counts / re-deliver later) and
    # either wakes the in-turn waiter or enqueues a resume task.
    result = await _finalize_human_answer(daemon, qid, response_text)
    if not result.get("ok"):
        return JSONResponse({"error": result.get("error")}, status_code=404)
    return {"status": "ok", "message": "Response delivered to agent.",
            "delivery": result["delivery"]}


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

    try:
        creds = list_credentials_public(team_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"team_id": team_id, "credentials": creds})


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

    try:
        deleted = delete_credential(team_id, site)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if deleted:
        return JSONResponse({"status": "deleted", "site": site})
    return JSONResponse({"error": "credential not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Team workspace — read-only file browser of the team's SHARED project dir (the
# real work surface every agent reads/writes). Lets the operator see what the
# team is actually producing without shelling into the host.
# ---------------------------------------------------------------------------
_WORKSPACE_MAX_FILES = 2000          # cap the tree so a huge repo can't hang the UI
_WORKSPACE_MAX_FILE_BYTES = 512 * 1024   # 512 KB per file view
# Directories that are noise (deps / VCS / caches) — skip them in the tree.
_WORKSPACE_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".next", "dist", "build", ".cache",
}


def _resolve_project_dir(team_id: str):
    """Return (project_dir Path, error_response_or_None). 404s an unknown team."""
    cfg = load_agents_config()
    if team_id not in (cfg.get("teams", {}) or {}):
        return None, JSONResponse({"error": f"Unknown team '{team_id}'."}, status_code=404)
    from swarm_server.config import _get_project_dir
    return _get_project_dir(team_id, cfg).resolve(), None


@app.get("/teams/{team_id}/workspace")
async def get_team_workspace(team_id: str):
    """List the team's shared project directory as a flat, sorted file tree.

    Paths are relative to the project root; directories are skipped if they're
    dependency/cache noise. Capped at ``_WORKSPACE_MAX_FILES`` entries."""
    import os

    root, err = _resolve_project_dir(team_id)
    if err:
        return err
    if not root.exists():
        return JSONResponse({"team_id": team_id, "root": str(root),
                             "exists": False, "files": []})

    files = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noise dirs in place so os.walk doesn't descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in _WORKSPACE_SKIP_DIRS
                             and not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, root)
            files.append({"path": rel, "size": st.st_size, "mtime": st.st_mtime})
            if len(files) >= _WORKSPACE_MAX_FILES:
                truncated = True
                break
        if truncated:
            break

    files.sort(key=lambda f: f["path"])
    return JSONResponse({"team_id": team_id, "root": str(root), "exists": True,
                         "truncated": truncated, "files": files})


@app.get("/teams/{team_id}/workspace/file")
async def get_team_workspace_file(team_id: str, path: str):
    """Return the text content of one file inside the team's project dir.

    Path-traversal safe: the resolved target must stay within the project root.
    Binary files and oversized files are reported rather than dumped."""
    import os

    root, err = _resolve_project_dir(team_id)
    if err:
        return err

    # Resolve and confine: the realpath must be inside the project root.
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return JSONResponse({"error": "Path escapes the workspace."}, status_code=400)
    if not target.is_file():
        return JSONResponse({"error": "File not found."}, status_code=404)

    size = target.stat().st_size
    if size > _WORKSPACE_MAX_FILE_BYTES:
        return JSONResponse({"path": path, "size": size, "too_large": True,
                             "content": "", "max_bytes": _WORKSPACE_MAX_FILE_BYTES})
    raw = target.read_bytes()
    if b"\x00" in raw:
        return JSONResponse({"path": path, "size": size, "binary": True, "content": ""})
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", "replace")
    return JSONResponse({"path": path, "size": size, "content": content})


@app.post("/teams/{team_id}/workspace/open")
async def open_team_workspace_file(team_id: str, request: Request):
    """Open a workspace file (or reveal its folder) in the SERVER host's desktop.

    Only meaningful when the dashboard and server share a machine (the common
    local single-operator setup) — on a headless VPS there is no desktop and this
    returns a clear error. Path-traversal safe like the file reader; spawns the
    platform opener detached so the request never blocks."""
    import os
    import platform
    import shutil
    import subprocess

    try:
        body = await request.json()
    except Exception:
        body = {}
    path = (body.get("path") or "").strip()
    reveal = bool(body.get("reveal"))
    if not path:
        return JSONResponse({"error": "path is required."}, status_code=400)

    root, err = _resolve_project_dir(team_id)
    if err:
        return err
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return JSONResponse({"error": "Path escapes the workspace."}, status_code=400)
    if not target.exists():
        return JSONResponse({"error": "File not found."}, status_code=404)

    # When revealing, open the containing folder; otherwise open the file itself.
    what = target.parent if reveal else target
    system = platform.system()
    try:
        if system == "Darwin":
            opener = ["open"]
        elif system == "Windows":
            opener = ["cmd", "/c", "start", ""]
        else:  # Linux / *nix
            if not shutil.which("xdg-open"):
                return JSONResponse(
                    {"error": "No desktop opener available (xdg-open not found). "
                              "This action only works when the server runs on a "
                              "machine with a desktop."}, status_code=501)
            opener = ["xdg-open"]
        subprocess.Popen(
            opener + [str(what)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Could not open: {e}"}, status_code=500)
    return JSONResponse({"ok": True, "opened": str(what), "reveal": reveal})


# ---------------------------------------------------------------------------
# Team costs — REAL provider token counts (token_usage events), priced with
# the swarm-side map. LiteLLM's Postgres SpendLogs stay the billing ground
# truth; this is the operational view: who spends, cache hit rates, sweep
# skip ratio.
# ---------------------------------------------------------------------------
@app.get("/teams/{team_id}/costs")
async def get_team_costs(team_id: str, hours: float = 24):
    import time

    from swarm_server.model_config import estimate_cost_usd

    hours = max(0.1, min(float(hours), 24 * 14))
    since_ts = time.time() - hours * 3600
    team_agents = get_team_agents(load_agents_config(), team_id) or {}
    events = monitor_db.get_token_usage_events(
        since_ts, agent_names=list(team_agents), team_id=team_id)
    cfg_models = {name: (a.get("model") or "")
                  for name, a in team_agents.items()}
    per_agent: Dict[str, Dict[str, Any]] = {}
    prev_cum: Dict[str, Dict[str, int]] = {}  # legacy-row differencing state

    for ev in events:
        agent = ev.get("agent_name") or "?"
        try:
            data = json.loads(ev.get("data") or "{}")
        except Exception:
            continue
        model = (data.get("model") or cfg_models.get(agent) or "").strip()
        provider = (data.get("provider") or "").strip() or None
        if "turn_input_tokens" in data:
            t_in = int(data.get("turn_input_tokens") or 0)
            t_out = int(data.get("turn_output_tokens") or 0)
            t_cache = int(data.get("turn_cache_read_tokens") or 0)
        else:
            # Legacy row: cumulative counters only. Difference against the
            # previous legacy row for this agent; the first row in the window
            # is the baseline (its turn happened mostly before the window).
            cum = {"in": int(data.get("input_tokens") or 0),
                   "out": int(data.get("output_tokens") or 0),
                   "cache": int(data.get("cache_read_tokens") or 0),
                   "total": int(data.get("total_tokens") or 0)}
            prev = prev_cum.get(agent)
            prev_cum[agent] = cum
            if prev is None:
                continue
            if cum["total"] < prev["total"]:  # session rotated → counter reset
                prev = {"in": 0, "out": 0, "cache": 0, "total": 0}
            t_in = max(0, cum["in"] - prev["in"])
            t_out = max(0, cum["out"] - prev["out"])
            t_cache = max(0, cum["cache"] - prev["cache"])

        a = per_agent.setdefault(agent, {
            "turns": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "est_cost_usd": 0.0, "model": model,
            "unpriced": False})
        a["turns"] += 1
        a["input_tokens"] += t_in
        a["output_tokens"] += t_out
        a["cache_read_tokens"] += t_cache
        if model:
            a["model"] = model
        cost = estimate_cost_usd(model, t_in, t_out, t_cache, provider=provider)
        if cost is None:
            a["unpriced"] = True
        else:
            a["est_cost_usd"] += cost

    totals = {"turns": 0, "input_tokens": 0, "output_tokens": 0,
              "cache_read_tokens": 0, "est_cost_usd": 0.0}
    for name, a in per_agent.items():
        seen = a["input_tokens"] + a["cache_read_tokens"]
        a["cache_hit_pct"] = round(100.0 * a["cache_read_tokens"] / seen, 1) if seen else 0.0
        a["est_cost_usd"] = round(a["est_cost_usd"], 4)
        for k in totals:
            totals[k] += a[k]
    totals["est_cost_usd"] = round(totals["est_cost_usd"], 4)
    seen = totals["input_tokens"] + totals["cache_read_tokens"]
    totals["cache_hit_pct"] = round(100.0 * totals["cache_read_tokens"] / seen, 1) if seen else 0.0

    sweeps = {"queued": 0, "skipped": 0}
    for name, a in team_agents.items():
        if a.get("is_supervisor"):
            sweeps["queued"] += monitor_db.count_events_since(
                name, "supervisor_sweep", since_ts)
            sweeps["skipped"] += monitor_db.count_events_since(
                name, "supervisor_sweep_skipped", since_ts)

    from swarm_server.budget import budget_tracker
    return JSONResponse({"team_id": team_id, "hours": hours,
                         "agents": per_agent, "totals": totals,
                         "sweeps": sweeps,
                         "budget": budget_tracker.status(team_id)})


# ---------------------------------------------------------------------------
# Team budget — daily spend cap with auto-pause (see swarm_server/budget.py)
# ---------------------------------------------------------------------------
@app.get("/teams/{team_id}/budget")
async def get_team_budget_status(team_id: str):
    from swarm_server.budget import budget_tracker
    return JSONResponse(budget_tracker.status(team_id))


@app.put("/teams/{team_id}/budget")
async def put_team_budget(team_id: str, request: Request):
    from swarm_server.config import set_team_budget
    from swarm_server.budget import budget_tracker

    body = await request.json()
    try:
        daily_usd = float(body.get("daily_usd", 0) or 0)
        daily_tokens = int(body.get("daily_tokens", 0) or 0)
    except (TypeError, ValueError):
        return JSONResponse({"error": "daily_usd / daily_tokens must be numbers"},
                            status_code=400)
    try:
        set_team_budget(team_id, daily_usd, daily_tokens)
    except ValueError as exc:
        msg = str(exc)
        code = 404 if "unknown team" in msg else 400
        return JSONResponse({"error": msg}, status_code=code)
    budget_tracker.set_limits(team_id, daily_usd, daily_tokens)
    return JSONResponse(budget_tracker.status(team_id))


@app.post("/teams/{team_id}/budget/override")
async def post_team_budget_override(team_id: str):
    """Resume a budget-paused team for the rest of the UTC day."""
    from swarm_server.budget import budget_tracker
    budget_tracker.override_today(team_id)
    # Wake the team's daemons so held work resumes immediately, not next tick.
    cfg = load_agents_config()
    for name, a in (cfg.get("agents") or {}).items():
        if a.get("team_id") == team_id:
            d = daemons.get(name)
            if d is not None:
                try:
                    d._signal_wake()
                except Exception:
                    pass
    return JSONResponse(budget_tracker.status(team_id))


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


async def _interrupt_and_drain(daemon, timeout: float = 30.0) -> None:
    """Interrupt any in-flight turn and WAIT (bounded, off the event loop) for the
    worker thread to unwind. Used before deleting an agent's workspace so rmtree
    doesn't pull state.db / the queue DB out from under a live turn (sqlite I/O
    errors, or a ghost thread recreating files in the just-deleted dir)."""
    try:
        agent = daemon._ai_agent
        if agent is not None and hasattr(agent, "interrupt"):
            agent.interrupt("Agent is being deleted.")
        daemon.human_event.set()
    except Exception as e:
        log.debug("[Daemon] interrupt before delete failed: %s", e)
    if daemon._sweep_task and not daemon._sweep_task.done():
        daemon._sweep_task.cancel()
        try:
            await daemon._sweep_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    try:
        await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, daemon._executor.shutdown, True
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        log.warning("[Daemon] '%s' turn still unwinding at delete — proceeding", daemon.name)
    except Exception as e:
        log.debug("[Daemon] drain before delete failed: %s", e)


@app.delete("/agent/{agent_name}")
async def remove_agent(agent_name: str):
    cfg = load_agents_config()
    if agent_name not in cfg["agents"]:
        return JSONResponse({"error": "agent not found"}, status_code=404)

    # Stop the in-flight turn and wait for its worker thread to finish BEFORE the
    # workspace is removed (delete_agent rmtree's it) — see _interrupt_and_drain.
    daemon = daemons.get(agent_name)
    if daemon is not None:
        await _interrupt_and_drain(daemon)
    _stop_and_unregister_daemon(agent_name)
    delete_agent(cfg, agent_name)
    return JSONResponse({"status": "deleted", "agent_name": agent_name})


def _update_daemon_cfg(agent_name: str, new_cfg: Dict[str, Any]):
    daemon = daemons.get(agent_name)
    if daemon is not None:
        with daemon._lock:
            daemon.cfg = new_cfg
            # Request a re-init at the START of the next turn rather than nulling
            # _ai_agent now: a BUSY agent's worker thread could be mid-turn about to
            # call self._ai_agent.run_conversation, which a null would crash (and a
            # rotated compaction session would be lost). The worker re-inits cleanly.
            daemon._reinit_requested = True
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
    # remove_agent_peer is ALWAYS bidirectional and mutates cfg in place, so it
    # already dropped peer_name → agent_name. Refresh BOTH running daemons' cfg so
    # the now-removed link is enforced live. (The old "also remove the reverse
    # link" branch was dead — the reverse link was already gone here — which left
    # the peer daemon running with a stale allowed_peers until an unrelated reload.)
    remove_agent_peer(cfg, agent_name, peer_name)
    if peer_name in cfg["agents"] and peer_name in daemons:
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
            # Re-init at the next turn (not a mid-turn null — see _update_daemon_cfg).
            daemon._reinit_requested = True
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
    served = list_backend_models(eff.get("base_url") or None, eff.get("api_key") or None)
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
            # Live fire count (in-memory for unbounded crons, persisted for bounded).
            "runs": rt.get("runs") or c.get("runs", 0),
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
        mr = body.get("max_runs")
        entry = add_agent_cron(
            cfg, agent_name,
            schedule=body.get("schedule", ""),
            instruction=body.get("instruction", ""),
            enabled=body.get("enabled", True),
            created_by="human",
            max_runs=(None if mr in ("", None) else mr),
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


# ---------------------------------------------------------------------------
# Architect / master team-builder. A teamless Hermes agent the human chats with
# to design and build teams. Its tools mutate config + spawn/despawn daemons; the
# daemon lifecycle must happen on THIS event loop, so we inject thread-safe hooks
# (the master turn runs on its own worker thread). See swarm_server/master.py.
# ---------------------------------------------------------------------------
def _master_spawn(agent_name: str, loop: asyncio.AbstractEventLoop) -> None:
    def _do():
        cfg = load_agents_config()
        ac = cfg["agents"].get(agent_name)
        if ac is not None and agent_name not in daemons:
            register_agent_daemon(agent_name, ac, loop)
    loop.call_soon_threadsafe(_do)


def _master_update(agent_name: str, loop: asyncio.AbstractEventLoop) -> None:
    def _do():
        cfg = load_agents_config()
        ac = cfg["agents"].get(agent_name)
        if ac is not None:
            _update_daemon_cfg(agent_name, ac)
    loop.call_soon_threadsafe(_do)


async def _master_despawn_coro(token: str) -> None:
    """Drain + delete an agent, or (for a '__team__:<id>' token) every agent in a
    team and then the team itself. Runs on the event loop."""
    if token.startswith("__team__:"):
        team_id = token.split(":", 1)[1]
        members = [
            n for n, a in load_agents_config()["agents"].items()
            if a.get("team_id") == team_id
        ]
        for n in members:
            d = daemons.get(n)
            if d is not None:
                await _interrupt_and_drain(d)
            _stop_and_unregister_daemon(n)
        delete_team(load_agents_config(), team_id)
        return
    d = daemons.get(token)
    if d is not None:
        await _interrupt_and_drain(d)
    _stop_and_unregister_daemon(token)
    delete_agent(load_agents_config(), token)


def _master_despawn(token: str, loop: asyncio.AbstractEventLoop) -> None:
    # Block the master worker thread until the drain+delete finishes, so a team
    # teardown deletes members before dropping the team record.
    fut = asyncio.run_coroutine_threadsafe(_master_despawn_coro(token), loop)
    try:
        fut.result(timeout=90)
    except Exception as e:  # noqa: BLE001
        log.warning("[master] despawn '%s' failed: %s", token, e)


def _wire_master(loop: asyncio.AbstractEventLoop) -> None:
    try:
        from swarm_server import master as _master

        _master.set_master_hooks(
            spawn=lambda n: _master_spawn(n, loop),
            despawn=lambda n: _master_despawn(n, loop),
            update=lambda n: _master_update(n, loop),
        )
        try:
            _master.register_master_tools()
        except Exception as e:  # noqa: BLE001 — registry may be unavailable in tests
            log.warning("[master] tool registration deferred: %s", e)
        log.info("[master] Architect wired")
    except Exception as e:  # noqa: BLE001
        log.warning("[master] wiring failed: %s", e)


@app.get("/master/status")
async def master_status():
    from swarm_server.master import get_master

    m = get_master()
    return JSONResponse({"configured": m.is_configured(), "busy": m.is_busy(), "model": m.model()})


@app.get("/master/history")
async def master_history(limit: int = 120):
    from swarm_server.master import get_master

    return JSONResponse({"messages": get_master().history(max(1, min(int(limit or 120), 500)))})


@app.post("/master/chat")
async def master_chat(request: Request):
    from swarm_server.master import get_master

    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    m = get_master()
    if not m.is_configured():
        return JSONResponse({"error": "no model configured — set one in Model settings first"}, status_code=409)
    if not m.submit(message):
        return JSONResponse({"error": "the Architect is still working on the previous message"}, status_code=409)
    return JSONResponse({"status": "accepted"})


@app.post("/master/reset")
async def master_reset():
    from swarm_server.master import get_master

    get_master().reset()
    return JSONResponse({"status": "ok"})


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


def _team_daemons(team_id: str):
    """Live daemons whose agent belongs to ``team_id``."""
    return [d for n, d in daemons.items()
            if (d.cfg or {}).get("team_id") == team_id]


@app.post("/teams/{team_id}/pause")
async def pause_team_execution(team_id: str, request: Request):
    """Freeze every agent on the team at once (each keeps its queue). The
    counterpart to the per-agent brake — one click to halt a whole team."""
    members = _team_daemons(team_id)
    if not members:
        return JSONResponse({"error": f"No running agents for team '{team_id}'."},
                            status_code=404)
    reason = ""
    try:
        reason = ((await request.json()) or {}).get("reason", "")
    except Exception:
        pass
    paused = []
    for d in members:
        if not getattr(d, "_paused", False):
            d.pause_execution(reason=reason or "Team paused by operator", by="human")
            paused.append(d.name)
    return JSONResponse({"success": True, "team_id": team_id,
                         "paused": paused, "total": len(members)})


@app.post("/teams/{team_id}/resume")
async def resume_team_execution(team_id: str):
    """Lift the freeze on every paused agent in the team."""
    members = _team_daemons(team_id)
    if not members:
        return JSONResponse({"error": f"No running agents for team '{team_id}'."},
                            status_code=404)
    resumed = []
    for d in members:
        if getattr(d, "_paused", False):
            d.resume_execution(by="human")
            resumed.append(d.name)
    return JSONResponse({"success": True, "team_id": team_id,
                         "resumed": resumed, "total": len(members)})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
_PROCESS_START_TS = None  # set on first /health call (import time is fine too)
_BACKEND_CHECK = {"ts": 0.0, "ok": None}


def _backend_reachable_cached() -> Optional[bool]:
    """Is the configured LLM backend reachable? Cached 60s so /health polling
    can't hammer it. Only custom / OpenAI-compatible endpoints expose a /models
    list to probe; for native providers this returns None (not probed)."""
    import time as _t
    import urllib.request
    if _t.time() - _BACKEND_CHECK["ts"] < 60:
        return _BACKEND_CHECK["ok"]
    _BACKEND_CHECK["ts"] = _t.time()
    try:
        from swarm_server.model_config import resolve_model
        eff = resolve_model({})
        base_url = (eff.get("base_url") or "").strip()
        if not base_url:
            _BACKEND_CHECK["ok"] = None  # native provider — nothing to GET /models
            return None
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {eff.get('api_key') or ''}"})
        with urllib.request.urlopen(req, timeout=3) as r:
            _BACKEND_CHECK["ok"] = (r.status == 200)
    except Exception:
        _BACKEND_CHECK["ok"] = False
    return _BACKEND_CHECK["ok"]


@app.get("/health")
async def health(request: Request):
    import time as _t
    global _PROCESS_START_TS
    if _PROCESS_START_TS is None:
        _PROCESS_START_TS = _t.time()
    # When auth is on, an UNauthenticated probe gets only liveness — the agent /
    # team names are sensitive. An authenticated probe (or no-auth localhost)
    # gets the full operational picture.
    if SWARM_API_KEY and not _request_key_ok(request):
        return {"status": "ok"}
    cfg = load_agents_config()
    queue_depth = 0
    for d in daemons.values():
        try:
            queue_depth += d.queue.get_pending_count()
        except Exception:
            pass
    return {
        "status": "ok",
        "version": app.version,
        "uptime_s": round(_t.time() - _PROCESS_START_TS, 1),
        "agents": list(daemons.keys()),
        "teams": list(cfg["teams"].keys()),
        "queue_depth": queue_depth,
        "llm_backend_ok": _backend_reachable_cached(),
    }


@app.get("/version/check")
async def version_check(force: bool = False):
    """Report whether a newer version is on `main` and how to upgrade.

    Auth-exempt (like /health) so the dashboard can call it pre-login. The check
    is cached (~6h) and reads GitHub off the event loop, so this never blocks.
    """
    from swarm_server.update_check import check_for_update

    return await asyncio.to_thread(check_for_update, force)


@app.get("/auth/check")
async def auth_check(request: Request):
    """Self-reporting auth probe the dashboard hits at boot. Always 200 so a
    reverse proxy's own 401s can't be mistaken for 'wrong key'. Leaks nothing
    beyond whether auth is enabled."""
    if not SWARM_API_KEY:
        return {"auth_required": False, "authorized": True}
    return {"auth_required": True, "authorized": _request_key_ok(request)}


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
    from swarm_server.tools import _pending_human_questions, _pending_lock

    body = await request.json()
    response_text = body.get("response", "")
    question_id = body.get("question_id")

    if not response_text.strip():
        return JSONResponse({"error": "Empty response"}, status_code=400)

    daemon = daemons.get(agent_name)
    if daemon is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)

    # Resolve the target question. An explicit question_id is honored ONLY if it is
    # genuinely a pending question for THIS agent — never blindly delivered as the
    # answer to whatever the agent is currently blocked on (answering an OLD
    # question must not feed its text to a NEW one). Otherwise pick the newest
    # pending question for this agent.
    found_qid = None
    with _pending_lock:
        if question_id:
            q = _pending_human_questions.get(question_id)
            if q and q["agent_name"] == agent_name and q["status"] == "pending":
                found_qid = question_id
            else:
                log.warning("[Inbox] Ignoring question_id %s for %s (not a pending "
                            "question for this agent)", question_id, agent_name)
        if not found_qid:
            newest = None
            for qid, q in _pending_human_questions.items():
                if q["agent_name"] == agent_name and q["status"] == "pending":
                    if newest is None or q["timestamp"] > newest[1]:
                        newest = (qid, q["timestamp"])
            found_qid = newest[0] if newest else None

    if not found_qid:
        return JSONResponse(
            {"error": f"No pending question found for agent '{agent_name}'."},
            status_code=404,
        )

    # Single atomic delivery path (marks answered + wakes in-turn XOR enqueues a
    # resume task; only a real takeover hands the browser back, off the loop).
    result = await _finalize_human_answer(daemon, found_qid, response_text)
    if not result.get("ok"):
        return JSONResponse({"error": result.get("error")}, status_code=404)
    return JSONResponse({
        "success": True,
        "question_id": found_qid,
        "delivery": result["delivery"],
        "message": ("Response delivered to agent." if result["delivery"] == "in_turn"
                    else "Response delivered as a resume task (agent had moved on)."),
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
