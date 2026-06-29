"""``agent-teams`` command-line entry point.

Subcommands:
  agent-teams up        Run the teams server + dashboard (default)
  agent-teams down      Stop a server started with `up` (incl. detached)
  agent-teams status    Is the server running? show URL + health
  agent-teams setup     Full interactive provider/tool wizard (`hermes setup`)
  agent-teams set-model Set provider/model non-interactively (scriptable)
  agent-teams init      Scaffold a starter team + coordinator agent
  agent-teams doctor    Check Hermes, the model backend, and Chromium

Installed as a console script via pyproject (``agent-teams = teams_server.cli:main``).
"""

import argparse
import logging
import os
import sys


# ---------------------------------------------------------------------------
# Running-server tracking (pidfile) so `down`/`status` work for a detached `up`.
# ---------------------------------------------------------------------------
def _pidfile_path():
    from teams_server.config import DATA_ROOT
    return DATA_ROOT / "teams.pid"


def _write_pidfile() -> None:
    try:
        p = _pidfile_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(os.getpid()))
    except Exception:
        pass


def _clear_pidfile() -> None:
    try:
        _pidfile_path().unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _running_pid():
    """PID of the server per the pidfile if that process is alive, else None."""
    try:
        pid = int(_pidfile_path().read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)            # signal 0 = liveness probe
    except OSError:
        return None                # stale pidfile (process gone)
    return pid


def _probe_health(host: str, port: int, timeout: float = 1.5) -> bool:
    """True if GET /health succeeds. 0.0.0.0 means 'all interfaces' — dial loopback."""
    import urllib.request

    h = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    try:
        with urllib.request.urlopen(f"http://{h}:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _setup_logging() -> None:
    from teams_server.config import configure_logging
    configure_logging()


BANNER = r"""
  _    _                                 _____                                 
 | |  | |                               / ____|                                
 | |__| | ___ _ __ _ __ ___   ___  ___ | (___ __      ____ _ _ __ _ __ ___     
 |  __  |/ _ \ '__| '_ ` _ \ / _ \/ __| \___ \\ \ /\ / / _` | '__| '_ ` _ \    
 | |  | |  __/ |  | | | | | |  __/\__ \ ____) |\ V  V / (_| | |  | | | | | |   
 |_|  |_|\___|_|  |_| |_| |_|\___||___/|_____/  \_/\_/ \__,_|_|  |_| |_| |_|   

    Commands:
      • Start Teams (FG):  agent-teams up
      • Start (Detached):  agent-teams up --detach
      • Stop Teams:        agent-teams down
      • Check Status:      agent-teams status
      • Configuration:     agent-teams setup
      • Teams Doctor:      agent-teams doctor
"""

def print_banner() -> None:
    if sys.stdout.isatty():
        print(f"\033[1;32m{BANNER}\033[0m")
    else:
        print(BANNER)


def _git_head(project_root) -> "str | None":
    """Current HEAD commit SHA, or None if it can't be read."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _maybe_auto_update() -> None:
    """Startup auto-update: fast-forward `main` + reinstall, then re-exec.

    Commit-driven (not version-driven): any new commit on `main` — a hotfix with
    no version bump, or a tagged release — is applied. Runs before agents start, so
    it never interrupts in-flight work; on by default (TEAMS_AUTO_UPDATE=0 opts out).
    No-op outside a git checkout (Docker rebuilds the image instead). Guarded by
    TEAMS_DID_AUTOUPDATE so the post-update re-exec can't loop. Any failure (offline,
    dirty/diverged tree, timeout) is non-fatal — we just start the current version.
    """
    import subprocess

    from teams_server.config import AUTO_UPDATE_ENABLED, PROJECT_ROOT
    from teams_server.update_check import get_install_method

    if not AUTO_UPDATE_ENABLED or os.environ.get("TEAMS_DID_AUTOUPDATE"):
        return
    if get_install_method() != "git":
        return  # Docker / non-checkout: no in-place pull
    try:
        before = _git_head(PROJECT_ROOT)
        rc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"],
            capture_output=True, text=True, timeout=15,
        ).returncode
        if rc != 0:
            return  # offline or can't fast-forward — keep running current code
        after = _git_head(PROJECT_ROOT)
        if not after or before == after:
            return  # already up to date — fast path, no reinstall/re-exec

        print(f"● auto-update: new commits on main ({(before or '?')[:7]} → {after[:7]}); "
              "reinstalling…")
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
            capture_output=True, text=True, timeout=300,
        ).returncode
        if rc != 0:
            print("auto-update: reinstall failed; starting the current version.",
                  file=sys.stderr)
            return
        # Re-exec the upgraded code in place. The marker prevents a re-exec loop.
        print("● restarting with the updated version…")
        os.environ["TEAMS_DID_AUTOUPDATE"] = "1"
        os.execv(sys.executable, [sys.executable, "-m", "teams_server.cli", *sys.argv[1:]])
    except Exception as e:  # never let auto-update block a normal start
        print(f"auto-update skipped ({e}); starting the current version.", file=sys.stderr)


def cmd_up(args) -> int:
    """Run the server — in the foreground, or detached with `--detach`."""
    _maybe_auto_update()
    print_banner()
    if getattr(args, "detach", False):
        return _start_detached(args)
    return _serve(args)


def _start_detached(args) -> int:
    """Daemonize and return immediately, leaving the server running independently.

    Why this exists: `nohup agent-teams up &` does NOT reliably survive an AI
    coding agent starting it — the agent's bash tool kills its whole process
    group when the command returns, taking the backgrounded server with it (the
    server answers once, then vanishes, and `status` reports it down). A real
    double-fork + setsid detaches into its own session so it outlives the
    launching shell/agent.
    """
    import time

    from teams_server.config import DATA_ROOT, SERVER_HOST, SERVER_PORT

    if not hasattr(os, "fork"):
        print("--detach needs POSIX fork (Linux/macOS). On Windows, run "
              "`agent-teams up` under a service manager, or use Docker.",
              file=sys.stderr)
        return 2

    if _running_pid() and _probe_health(SERVER_HOST, SERVER_PORT):
        print(f"● already running (pid {_running_pid()}) → "
              f"http://{SERVER_HOST}:{SERVER_PORT}/")
        return 0

    log_path = getattr(args, "log", None) or str(DATA_ROOT / "teams.log")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    pid = os.fork()
    if pid > 0:
        # Parent: wait (≤20s) for the daemon to answer /health, then report + return.
        for _ in range(40):
            time.sleep(0.5)
            if _probe_health(SERVER_HOST, SERVER_PORT):
                print(f"● started (pid {_running_pid() or '?'}) → "
                      f"http://{SERVER_HOST}:{SERVER_PORT}/")
                print(f"  logs:    {log_path}")
                print(f"  status:  agent-teams status   ·   stop:  agent-teams down")
                return 0
        print(f"Started, but it hasn't answered /health yet — check the log: {log_path}",
              file=sys.stderr)
        return 0

    # First child: new session leader, then fork again so the daemon can never
    # reacquire a controlling terminal.
    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    # Grandchild = the daemon. Detach stdio to the log file and serve.
    try:
        with open(os.devnull, "rb") as devnull:
            os.dup2(devnull.fileno(), sys.stdin.fileno())
        logf = open(log_path, "ab", buffering=0)
        os.dup2(logf.fileno(), sys.stdout.fileno())
        os.dup2(logf.fileno(), sys.stderr.fileno())
    except Exception:
        pass
    os._exit(_serve(args))


def _serve(args) -> int:
    """Launch uvicorn serving the FastAPI app (host/port from env)."""
    import uvicorn

    from teams_server.config import SERVER_HOST, SERVER_PORT

    log = logging.getLogger("teams")
    log.info("=" * 60)
    log.info("  Agent Teams Server")
    log.info("  Dashboard:    http://%s:%s/", SERVER_HOST, SERVER_PORT)
    try:
        from teams_server.model_config import resolve_model, is_model_configured

        if is_model_configured():
            eff = resolve_model()
            log.info("  Model:        %s  (provider %s)", eff.get("model"),
                     eff.get("display_provider") or eff.get("provider"))
        else:
            log.warning("  Model:        none configured — run `hermes setup`")
    except Exception as e:  # never let a config probe block server start
        log.debug("startup model resolve failed: %s", e)
    log.info("  Stop it with:  agent-teams down   (status: agent-teams status)")
    log.info("=" * 60)
    _write_pidfile()              # so `down`/`status` find a detached server
    try:
        uvicorn.run(
            "teams_server.server:app",
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="info",
            reload=False,
        )
    finally:
        _clear_pidfile()
    return 0


def cmd_down(args) -> int:
    """Stop a server started with `up` (foreground or detached)."""
    import signal
    import time

    from teams_server.config import SERVER_HOST, SERVER_PORT

    pid = _running_pid()
    if not pid:
        if _probe_health(SERVER_HOST, SERVER_PORT):
            print("A server is responding but no pidfile was found "
                  "(started outside this data dir).")
            print("   → stop it where it runs (Ctrl-C), or:  pkill -f 'agent-teams up'")
            return 1
        print("○ Not running — nothing to stop.")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"Couldn't signal pid {pid}: {e}")
        _clear_pidfile()
        return 1
    # Give uvicorn a few seconds for a graceful shutdown, then force it.
    for _ in range(50):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    _clear_pidfile()
    print(f"■ Stopped the teams server (pid {pid}).")
    return 0


def cmd_status(args) -> int:
    """Report whether the server is up, its URL, and health."""
    from teams_server.config import SERVER_HOST, SERVER_PORT

    pid = _running_pid()
    healthy = _probe_health(SERVER_HOST, SERVER_PORT)
    url = f"http://{SERVER_HOST}:{SERVER_PORT}"
    if pid or healthy:
        where = f"pid {pid}" if pid else "detected on port (no pidfile)"
        print(f"● running ({where})")
        print(f"   Dashboard:  {url}/")
        print(f"   Health:     {'ok' if healthy else 'starting… (not responding yet)'}")
        print(f"   Stop it:    agent-teams down")
        return 0
    print("○ not running")
    print("   Start it:   agent-teams up")
    return 1


def cmd_setup(args) -> int:
    """Launch the FULL interactive Hermes wizard against the teams's shared config.

    A superset of ``set-model``: besides the provider + model, ``hermes setup``
    configures web-search / vision / browser tool providers, memory, reasoning
    effort, credential rotation, and more. It writes to the same shared home
    (``data/.hermes-shared``) that the teams reads as its default, so settings
    apply to every agent. Use this when you want more than just the model.
    """
    import subprocess

    from teams_server.model_config import SHARED_HERMES_HOME

    SHARED_HERMES_HOME.mkdir(parents=True, exist_ok=True)
    hermes = os.path.join(os.path.dirname(sys.executable), "hermes")
    if not os.path.exists(hermes):
        hermes = "hermes"                      # fall back to PATH
    # Hermes reads HERMES_HOME from the environment (hermes_constants.get_hermes_home).
    env = dict(os.environ, HERMES_HOME=str(SHARED_HERMES_HOME))
    print(f"Launching `hermes setup` against the teams config "
          f"({SHARED_HERMES_HOME}) — providers, web/vision/browser tools, memory…\n")
    try:
        return subprocess.call([hermes, "setup", *getattr(args, "rest", [])], env=env)
    except FileNotFoundError:
        print("error: the `hermes` CLI isn't on PATH — install hermes-agent.",
              file=sys.stderr)
        return 1


def cmd_doctor(args) -> int:
    """Preflight: verify the three things a fresh install needs."""
    ok = True

    # 1) Hermes importable
    from teams_server.config import ensure_hermes_importable

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
    from teams_server.model_config import resolve_model, is_model_configured

    if not is_model_configured():
        ok = False
        print("✗ No model configured.")
        print("   → run `hermes setup`   (pick a provider + key + model — Hermes saves it in ~/.hermes)")
        print("     For a custom / OpenAI-compatible endpoint (e.g. a LiteLLM proxy), choose the")
        print("     'custom' provider in `hermes setup` and enter its base URL + key.")
    else:
        eff = resolve_model()
        prov = eff.get("display_provider") or eff.get("provider") or "?"
        srclabel = {
            "default": "teams default",
            "hermes": "hermes setup (~/.hermes)",
        }.get(eff.get("source"), str(eff.get("source")))
        print(f"✓ Model: {eff.get('model')}  (provider {prov}, source: {srclabel})")

        base = eff.get("base_url")
        if base:
            # Custom / OpenAI-compatible endpoint: we know the URL, so probe it.
            try:
                import urllib.request, json as _json

                req = urllib.request.Request(
                    f"{base}/models",
                    headers={"Authorization": f"Bearer {eff.get('api_key') or ''}"},
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
        from teams_server.browser_pool import _find_browser

        chromium = _find_browser()
        if chromium:
            print(f"✓ Chromium found: {chromium}")
        else:
            print("⚠ Chromium not found — browser publishing tools will be unavailable.")
            print("   → playwright install chromium")
    except Exception as e:
        print(f"⚠ Could not probe Chromium: {e}")

    # 4) Hermes compat seams — the internal APIs the teams builds over. Drift here
    # (after a Hermes update) silently disables features, so surface it explicitly.
    try:
        from teams_server.hermes_compat import run_self_check

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
                      "teams_server/hermes_compat.py")
    except Exception as e:
        print(f"⚠ Could not run Hermes compat self-check: {e}")

    print("\nResult:", "ready ✅" if ok else "issues above ⚠️")
    return 0 if ok else 1


def cmd_init(args) -> int:
    """Scaffold a starter team. Optionally creates a coordinator agent if --agent is specified.

    No-op-safe: skips anything that already exists so it can be re-run.
    """
    from teams_server.config import (
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
    if not agent_id:
        print("\nNext: `agent-teams up` and open the dashboard.")
        return 0

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
    print("\nNext: `agent-teams up` and open the dashboard.")
    return 0


def cmd_set_model(args) -> int:
    """Set the teams's default provider/model non-interactively.

    A scriptable alternative to the interactive ``hermes setup`` wizard — for
    AI-agent installs, CI, and headless servers where no TTY is available. Writes
    to the teams's shared config (``data/.hermes-shared``), which every agent
    reads as its default. Example:

      agent-teams set-model --provider custom --model deepseek-chat \\
        --base-url http://localhost:4000/v1 --api-key sk-...
    """
    import re
    from teams_server.model_config import set_default_model, get_default_model

    model = (args.model or "").strip()
    if not model:
        print("error: --model is required", file=sys.stderr)
        return 2
    provider = (args.provider or "").strip()
    base_url = (args.base_url or "").strip()
    api_key = args.api_key or ""
    if not provider:
        provider = "custom" if base_url else "openai"
    if base_url:
        if "://" not in base_url:
            base_url = "http://" + base_url           # tolerate a bare host:port
        if not re.search(r"/v\d+/?$", base_url.rstrip("/") + "/"):
            print(f"note: base-url '{base_url}' doesn't end in a version path (e.g. /v1) — "
                  "most OpenAI-compatible endpoints need one. Writing it as given.")
    set_default_model(provider=provider, model=model, base_url=base_url, api_key=api_key)
    cur = get_default_model()
    shown = f"provider={cur.get('provider') or provider} model={cur.get('model') or model}"
    if base_url:
        shown += f" base_url={base_url}"
    print(f"✓ Default model set ({shown}).")
    print("  Written to the teams's shared config. Verify reachability with: agent-teams doctor")
    return 0


def _run_upgrade() -> int:
    """Pull `main` and reinstall in place. Returns 0 on success, non-zero on failure.

    Used by both `agent-teams update` and the opt-in startup auto-update. Runs
    subprocesses with arg lists (no shell-string interpolation) targeting THIS
    interpreter's environment via `sys.executable -m pip`.
    """
    import subprocess

    from teams_server.config import PROJECT_ROOT

    print(f"→ git pull --ff-only  ({PROJECT_ROOT})")
    rc = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"]
    ).returncode
    if rc != 0:
        print("✗ git pull failed (uncommitted changes or diverged branch?). "
              "Resolve manually, then retry.", file=sys.stderr)
        return rc

    print(f"→ pip install -e .  ({sys.executable})")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)]
    ).returncode
    if rc != 0:
        print("✗ pip reinstall failed.", file=sys.stderr)
        return rc
    return 0


def cmd_update(args) -> int:
    """Upgrade this install to the version on `main` (git pull + reinstall)."""
    from teams_server.update_check import check_for_update

    info = check_for_update(force=True)
    current, latest = info["current"], info.get("latest")
    method = info["install_method"]

    print(f"  current: {current}")
    print(f"  latest:  {latest or '(could not reach GitHub)'}")

    if method == "docker":
        print("\nThis is a Docker install — the running container can't update itself.")
        print("Rebuild the image instead:")
        print(f"  {info['upgrade_hint']}")
        return 0

    if latest is None:
        print("\nCouldn't determine the latest version (network?). Try again later.",
              file=sys.stderr)
        return 1

    if not info["update_available"]:
        print("\n✓ Already up to date.")
        return 0

    if method == "unknown":
        print("\n⚠ Couldn't confirm this is a git checkout — the upgrade path "
              "(git pull + pip install -e .) may not apply to your install.")

    if getattr(args, "check", False):
        print(f"\nUpdate available → {latest}. Run `agent-teams update` to install.")
        return 0

    if not getattr(args, "yes", False):
        try:
            ans = input(f"\nUpdate {current} → {latest}? [Y/n] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("", "y", "yes"):
            print("Aborted.")
            return 0

    rc = _run_upgrade()
    if rc == 0:
        print(f"\n✓ Updated to {latest}. Verify with: agent-teams doctor")
        print("  Restart a running server for the new version to take effect.")
    return rc


def main_compat(argv=None) -> int:
    """DEPRECATED ``hermes-swarm`` console alias → ``agent-teams``.

    Kept so existing users' muscle memory and systemd ``ExecStart=hermes-swarm …``
    units keep working after the rename. Prints a one-line notice to stderr (so
    only people invoking the OLD command ever see it) and then runs normally.
    Remove this — and the ``hermes-swarm`` entry in pyproject ``[project.scripts]``
    — in a future major version.
    """
    print(
        "⚠ 'hermes-swarm' is deprecated and will be removed in a future release — "
        "use 'agent-teams' instead (same arguments).",
        file=sys.stderr,
    )
    return main(argv)


def main(argv=None) -> int:
    _setup_logging()
    p = argparse.ArgumentParser(
        prog="agent-teams",
        description="P2P multi-agent teams server + real-time dashboard, powered by Hermes.",
    )
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("up", help="Run the teams server + dashboard")
    up.add_argument("-d", "--detach", action="store_true",
        help="daemonize and return immediately (survives the launching shell/agent)")
    up.add_argument("--log", default=None,
        help="log file for --detach (default: <data>/teams.log)")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="Stop a server started with `up` (incl. detached)")
    down.set_defaults(func=cmd_down)

    st = sub.add_parser("status", help="Is the server running? show URL + health")
    st.set_defaults(func=cmd_status)

    setup = sub.add_parser("setup",
        help="Full interactive provider/tool wizard (web search, vision, browser, memory…)")
    setup.add_argument("rest", nargs=argparse.REMAINDER,
        help="extra args passed through to `hermes setup`")
    setup.set_defaults(func=cmd_setup)

    doc = sub.add_parser("doctor", help="Check Hermes, model backend, and Chromium")
    doc.set_defaults(func=cmd_doctor)

    upd = sub.add_parser("update", help="Update to the latest version on `main` (git pull + reinstall)")
    upd.add_argument("--check", action="store_true",
        help="only report whether an update is available; don't install")
    upd.add_argument("-y", "--yes", action="store_true",
        help="install without the confirmation prompt")
    upd.set_defaults(func=cmd_update)

    init = sub.add_parser("init", help="Scaffold a starter team")
    init.add_argument("--team", default="default", help="team id (slug)")
    init.add_argument("--team-name", default=None, help="team display name")
    init.add_argument("--agent", default=None, help="agent id (slug) to create (optional)")
    init.add_argument("--agent-name", default=None, help="agent display name (optional)")
    init.set_defaults(func=cmd_init)

    sm = sub.add_parser("set-model",
        help="Set the default provider/model non-interactively (scriptable alt to `hermes setup`)")
    sm.add_argument("--model", required=True, help="model name (e.g. deepseek-chat, gpt-4o)")
    sm.add_argument("--provider", default=None,
        help="provider id (e.g. custom, openai, anthropic); defaults to 'custom' when --base-url is set")
    sm.add_argument("--base-url", default=None,
        help="OpenAI-compatible endpoint, e.g. http://localhost:4000/v1 (for custom/proxy)")
    sm.add_argument("--api-key", default=None, help="API key for the provider (stored in the home .env)")
    sm.set_defaults(func=cmd_set_model)

    args = p.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:  # bare `agent-teams` → run the server
        return cmd_up(args)
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
