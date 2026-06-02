"""``hermes-swarm`` command-line entry point.

Subcommands:
  hermes-swarm up        Run the swarm server + dashboard (default)
  hermes-swarm init      Scaffold a starter team + coordinator agent
  hermes-swarm doctor    Check Hermes, the model backend, and Chromium

Installed as a console script via pyproject (``hermes-swarm = swarm_server.cli:main``).
"""

import argparse
import logging
import os
import sys


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_up(args) -> int:
    """Launch uvicorn serving the FastAPI app (host/port from env)."""
    import uvicorn

    from swarm_server.config import SERVER_HOST, SERVER_PORT, LITELLM_API_BASE

    log = logging.getLogger("swarm")
    log.info("=" * 60)
    log.info("  Hermes Swarm Server")
    log.info("  Dashboard:    http://%s:%s/", SERVER_HOST, SERVER_PORT)
    log.info("  LLM backend:  %s", LITELLM_API_BASE)
    log.info("=" * 60)
    uvicorn.run(
        "swarm_server.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False,
    )
    return 0


def cmd_doctor(args) -> int:
    """Preflight: verify the three things a fresh install needs."""
    ok = True

    # 1) Hermes importable
    from swarm_server.config import ensure_hermes_importable, LITELLM_API_BASE, LLM_API_KEY, DEFAULT_MODEL

    ensure_hermes_importable()
    try:
        import run_agent  # noqa: F401
        ver = getattr(__import__("hermes_constants", fromlist=["__version__"]), "__version__", "?")
        print(f"✓ Hermes agent importable (hermes_constants {ver})")
    except Exception as e:
        ok = False
        print(f"✗ Hermes agent NOT importable: {e}")
        print("   → pip install hermes-agent   (or set HERMES_AGENT_PATH)")

    # 2) Model backend reachable
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{LITELLM_API_BASE}/models", headers={"Authorization": f"Bearer {LLM_API_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json as _json

            data = _json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        print(f"✓ LLM backend reachable at {LITELLM_API_BASE} — models: {ids or '(none listed)'}")
        if DEFAULT_MODEL not in ids:
            print(f"   ⚠ default model '{DEFAULT_MODEL}' not in the served list — set SWARM_DEFAULT_MODEL")
    except Exception as e:
        ok = False
        print(f"✗ LLM backend NOT reachable at {LITELLM_API_BASE}: {e}")
        print("   → set SWARM_LLM_BASE_URL + SWARM_LLM_API_KEY to an OpenAI-compatible endpoint")

    # 3) Chromium for the browser tools (optional but recommended)
    try:
        from swarm_server.browser_pool import _find_chromium

        chromium = _find_chromium()
        if chromium:
            print(f"✓ Chromium found: {chromium}")
        else:
            print("⚠ Chromium not found — browser publishing tools will be unavailable.")
            print("   → playwright install chromium")
    except Exception as e:
        print(f"⚠ Could not probe Chromium: {e}")

    print("\nResult:", "ready ✅" if ok else "issues above ⚠️")
    return 0 if ok else 1


def cmd_init(args) -> int:
    """Scaffold a starter team + one autonomous coordinator agent.

    No-op-safe: skips anything that already exists so it can be re-run.
    """
    from swarm_server.config import (
        load_agents_config,
        create_team,
        create_agent,
        save_agent_config,
    )

    team_id = args.team
    cfg = load_agents_config()
    if team_id not in cfg["teams"]:
        create_team(cfg, team_id, args.team_name or team_id.title())
        print(f"✓ Created team '{team_id}'")
        cfg = load_agents_config()
    else:
        print(f"• Team '{team_id}' already exists")

    agent_id = args.agent
    if agent_id in cfg["agents"]:
        print(f"• Agent '{agent_id}' already exists — nothing to do")
        return 0

    role = (
        "You are the COORDINATOR of this team. Break incoming goals into concrete, "
        "finished, shippable deliverables, delegate to teammates when present, and "
        "drive work to completion. Never stop at a draft."
    )
    create_agent(
        cfg, name=agent_id, team_id=team_id,
        display_name=args.agent_name or "Coordinator",
        allowed_peers=[], role_soul=role,
    )
    # Make the coordinator self-driving so a fresh install does something.
    cfg = load_agents_config()
    entry = cfg["agents"][agent_id]
    entry["autonomous"] = True
    save_agent_config(agent_id, entry)
    print(f"✓ Created autonomous agent '{agent_id}' on team '{team_id}'")
    print("\nNext: `hermes-swarm up` and open the dashboard.")
    return 0


def main(argv=None) -> int:
    _setup_logging()
    p = argparse.ArgumentParser(
        prog="hermes-swarm",
        description="P2P multi-agent swarm server + real-time dashboard, powered by Hermes.",
    )
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("up", help="Run the swarm server + dashboard")
    up.set_defaults(func=cmd_up)

    doc = sub.add_parser("doctor", help="Check Hermes, model backend, and Chromium")
    doc.set_defaults(func=cmd_doctor)

    init = sub.add_parser("init", help="Scaffold a starter team + coordinator agent")
    init.add_argument("--team", default="default", help="team id (slug)")
    init.add_argument("--team-name", default=None, help="team display name")
    init.add_argument("--agent", default="coordinator", help="agent id (slug)")
    init.add_argument("--agent-name", default=None, help="agent display name")
    init.set_defaults(func=cmd_init)

    args = p.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:  # bare `hermes-swarm` → run the server
        return cmd_up(args)
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
