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
    LITELLM_API_BASE,
    MONITORING_DB,
    MONITORING_MAX_EVENTS,
    MONITORING_MAX_MESSAGES,
    MONITORING_PRUNE_INTERVAL_SECONDS,
    SERVER_HOST,
    SERVER_PORT,
    SWARM_API_KEY,
    add_agent_peer,
    create_agent,
    create_team,
    delete_agent,
    delete_team,
    get_agent_team,
    get_team_agents,
    list_teams,
    load_agents_config,
    remove_agent_peer,
    save_agent_config,
    save_all_config,
    set_agent_peers,
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
    log.info("[Startup] All agents running. LiteLLM at %s", LITELLM_API_BASE)
    log.info("[Startup] Dashboard at http://%s:%s/", SERVER_HOST, SERVER_PORT)

    try:
        yield
    finally:
        # ---- shutdown ----
        prune_task.cancel()
        for name, daemon in list(daemons.items()):
            _stop_daemon(daemon)
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
            daemon._ai_agent = None


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

    # Update registry
    answer_question(found_qid, response_text)

    # Wake the daemon if it's still waiting
    if daemon.state == AGENT_STATE_ASKING_HUMAN:
        daemon.human_response = response_text
        daemon.human_event.set()
        log.info("[Inbox] Response delivered to %s (qid=%s)", agent_name, found_qid)
    else:
        log.warning(
            "[Inbox] Agent %s not in asking_human state (state=%s). Response stored but not delivered.",
            agent_name, daemon.state,
        )

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
