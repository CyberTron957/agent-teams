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
    from swarm_server.config import configure_logging
    configure_logging()


def cmd_up(args) -> int:
    """Launch uvicorn serving the FastAPI app (host/port from env)."""
    import uvicorn

    from swarm_server.config import SERVER_HOST, SERVER_PORT, LITELLM_API_BASE

    log = logging.getLogger("swarm")
    log.info("=" * 60)
    log.info("  Hermes Swarm Server")
    log.info("  Dashboard:    http://%s:%s/", SERVER_HOST, SERVER_PORT)
    try:
        from swarm_server.model_config import resolve_model, is_model_configured

        if is_model_configured():
            eff = resolve_model()
            log.info("  Model:        %s  (provider %s)", eff.get("model"),
                     eff.get("display_provider") or eff.get("provider"))
        else:
            log.warning("  Model:        none configured — run `hermes setup` (or set SWARM_LLM_*)")
    except Exception as e:  # never let a config probe block server start
        log.info("  LLM backend:  %s", LITELLM_API_BASE)
        log.debug("startup model resolve failed: %s", e)
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
    from swarm_server.config import ensure_hermes_importable, LLM_API_KEY

    ensure_hermes_importable()
    try:
        import run_agent  # noqa: F401
        try:
            from importlib.metadata import version as _pkg_version
            ver = _pkg_version("hermes-agent")
        except Exception:
            ver = "?"
        print(f"✓ Hermes agent importable (hermes-agent {ver})")
    except Exception as e:
        ok = False
        print(f"✗ Hermes agent NOT importable: {e}")
        print("   → pip install hermes-agent   (or set HERMES_AGENT_PATH)")

    # 2) Provider configured (via Hermes) + backend reachable
    from swarm_server.model_config import resolve_model, is_model_configured

    if not is_model_configured():
        ok = False
        print("✗ No model configured.")
        print("   → run `hermes setup`   (pick a provider + key + model — Hermes saves it in ~/.hermes)")
        print("   → or set SWARM_LLM_BASE_URL + SWARM_LLM_API_KEY for an OpenAI-compatible / LiteLLM proxy")
    else:
        eff = resolve_model()
        prov = eff.get("display_provider") or eff.get("provider") or "?"
        srclabel = {
            "default": "swarm default",
            "hermes": "hermes setup (~/.hermes)",
            "proxy": "LiteLLM proxy (SWARM_LLM_*)",
        }.get(eff.get("source"), str(eff.get("source")))
        print(f"✓ Model: {eff.get('model')}  (provider {prov}, source: {srclabel})")

        base = eff.get("base_url")
        if base:
            # OpenAI-compatible / proxy: we know the endpoint, so probe it.
            try:
                import urllib.request, json as _json

                req = urllib.request.Request(
                    f"{base}/models",
                    headers={"Authorization": f"Bearer {eff.get('api_key') or LLM_API_KEY}"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
                ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
                print(f"✓ Backend reachable at {base} — models: {ids or '(none listed)'}")
                if ids and eff.get("model") not in ids:
                    print(f"   ⚠ '{eff.get('model')}' not in the served list")
            except Exception as e:
                ok = False
                print(f"✗ Backend NOT reachable at {base}: {e}")
        else:
            # Native Hermes provider: Hermes resolves the endpoint itself; we can
            # only confirm a key is present for it.
            has_key = bool(eff.get("api_key"))
            mark = "✓" if has_key else "⚠"
            print(f"  {mark} Native provider — Hermes resolves the endpoint; "
                  f"API key {'present' if has_key else 'NOT found (run `hermes setup`)'}.")
            if not has_key:
                ok = False

    # 3) Chromium for the browser tools (optional but recommended)
    try:
        from swarm_server.browser_pool import _find_browser

        chromium = _find_browser()
        if chromium:
            print(f"✓ Chromium found: {chromium}")
        else:
            print("⚠ Chromium not found — browser publishing tools will be unavailable.")
            print("   → playwright install chromium")
    except Exception as e:
        print(f"⚠ Could not probe Chromium: {e}")

    # 4) Hermes compat seams — the internal APIs the swarm builds over. Drift here
    # (after a Hermes update) silently disables features, so surface it explicitly.
    try:
        from swarm_server.hermes_compat import run_self_check

        report = run_self_check()
        if report.ok:
            print(f"✓ Hermes compat: {len(report.probes)}/{len(report.probes)} seams verified")
        else:
            for p in report.failures:
                mark = "✗" if p.critical else "⚠"
                print(f"{mark} Hermes seam '{p.name}': {p.detail}")
            if report.critical_failures:
                ok = False
                print("   → a Hermes update likely moved an internal API; see "
                      "swarm_server/hermes_compat.py")
    except Exception as e:
        print(f"⚠ Could not run Hermes compat self-check: {e}")

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
