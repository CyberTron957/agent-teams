"""Per-team persistent, shared browser pool.

Each team gets ONE long-lived Chrome (DevTools/CDP) bound to a stable
``--user-data-dir`` under ``data/teams/<team>/.browser-profile``. Every agent in
that team is pointed at the same ``browser.cdp_url`` (written into its
config.yaml), so they share one browser — same cookies, logins, and storage.

Durability: because the profile directory is a fixed path on disk, the browser's
state survives a server restart. On restart we relaunch Chrome against the same
profile dir; cookies/logins are still there. (Chrome itself only allows one
process per user-data-dir, which is exactly the one-browser-per-team invariant.)

Isolation: one profile per team => teams never see each other's cookies/sessions.

This connects via Hermes' CDP-override path (``browser.cdp_url``), which takes
precedence over both the cloud provider and the local launcher — so it works
without cloud credentials and reuses the Playwright Chromium we install locally.
"""

import asyncio
import glob
import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from swarm_server.config import WORKSPACE_ROOT

log = logging.getLogger("swarm.browser")

# Base port for per-team CDP endpoints; each team gets the next free port up.
_BASE_CDP_PORT = 9333


def _find_chromium() -> Optional[str]:
    """Locate a Chromium executable from the Playwright browser cache.

    Prefers the full "Chrome for Testing" build (best site compatibility +
    persistent profile support); falls back to the lighter headless-shell.
    """
    roots = []
    pbp = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if pbp:
        roots.append(Path(pbp))
    roots += [
        Path.home() / "Library" / "Caches" / "ms-playwright",   # macOS
        Path.home() / ".cache" / "ms-playwright",                # Linux
    ]
    patterns = [
        "chromium-*/chrome-mac*/*.app/Contents/MacOS/*",         # mac full chromium
        "chromium-*/chrome-linux*/chrome",                       # linux full chromium
        "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
    ]
    for root in roots:
        for pat in patterns:
            for hit in sorted(glob.glob(str(root / pat)), reverse=True):
                if os.path.isfile(hit) and os.access(hit, os.X_OK):
                    return hit
    return None


class TeamBrowserManager:
    """Launches and tracks one persistent Chrome per team."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # team_id -> {"proc": Popen, "port": int, "profile": str}
        self._browsers: Dict[str, dict] = {}
        self._ports: Dict[str, int] = {}
        # Xvfb is only a FALLBACK display for headless hosts with no real X server.
        self._xvfb_proc: Optional[subprocess.Popen] = None
        self._xvfb_display: Optional[str] = None
        # team_id -> the ONE display its browser is pinned to for life (never moved).
        self._team_disp: Dict[str, str] = {}
        # teams whose browser is currently handed to a human (window restored/raised).
        self._takeover_active: set = set()
        self._chromium = _find_chromium()
        if self._chromium:
            log.info("Team browser pool using chromium: %s", self._chromium)
        else:
            log.warning(
                "No Chromium found for team browser pool "
                "(install with: npx playwright install chromium). "
                "Team browsers disabled."
            )

    # -- port helpers -------------------------------------------------------
    @staticmethod
    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    def _assign_port(self, team_id: str) -> int:
        if team_id in self._ports:
            return self._ports[team_id]
        used = set(self._ports.values())
        port = _BASE_CDP_PORT
        while port in used or not self._port_free(port):
            port += 1
        self._ports[team_id] = port
        return port

    def _healthy(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=2
            ) as r:
                return r.status == 200
        except Exception:
            return False

    # -- display selection --------------------------------------------------
    @staticmethod
    def _display_socket_exists(disp: str) -> bool:
        """True if an X server is listening on a DISPLAY like ':0' (socket present)."""
        try:
            num = (disp or "").strip().lstrip(":").split(".")[0]
            return bool(num) and os.path.exists(f"/tmp/.X11-unix/X{num}")
        except Exception:
            return False

    def _pick_display(self, team_id: str) -> str:
        """Choose, ONCE per team, the display the browser lives on for its whole
        life. Robust on any host:
          1. SWARM_BROWSER_DISPLAY override (explicit escape hatch),
          2. the display the server was launched in (if live) — what a local
             user actually sees,
          3. the first conventional real seat (:0, :1),
          4. our own hidden Xvfb (headless hosts; human view then needs VNC).
        The browser is NEVER moved off this display — visibility is handled in
        place via CDP, so there is no relaunch and the agent's CDP session is
        never broken.
        """
        if team_id in self._team_disp:
            return self._team_disp[team_id]
        disp = None
        override = os.environ.get("SWARM_BROWSER_DISPLAY", "").strip()
        if override:
            disp = override
        elif self._display_socket_exists(os.environ.get("DISPLAY", "")):
            disp = os.environ.get("DISPLAY", "").strip()
        else:
            for cand in (":0", ":1"):
                if self._display_socket_exists(cand):
                    disp = cand
                    break
        if not disp:
            disp = self._ensure_xvfb()  # headless host fallback
        self._team_disp[team_id] = disp
        log.info("[%s] Browser display pinned to %s", team_id, disp)
        return disp

    def _ensure_xvfb(self) -> str:
        """Start ONE hidden Xvfb as a FALLBACK display for headless hosts with no
        real X server. Returns the DISPLAY string; falls back to ':0' if Xvfb
        itself is unavailable so callers always get something usable."""
        if self._xvfb_display and self._xvfb_proc and self._xvfb_proc.poll() is None:
            return self._xvfb_display
        xvfb = shutil.which("Xvfb")
        if not xvfb:
            self._xvfb_display = os.environ.get("DISPLAY") or ":0"
            return self._xvfb_display
        for num in range(99, 130):
            if os.path.exists(f"/tmp/.X11-unix/X{num}"):
                continue
            disp = f":{num}"
            try:
                proc = subprocess.Popen(
                    [xvfb, disp, "-screen", "0", "1280x800x24", "-nolisten", "tcp"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                log.error("Failed to start Xvfb: %s", e)
                break
            for _ in range(30):
                if os.path.exists(f"/tmp/.X11-unix/X{num}"):
                    self._xvfb_proc = proc
                    self._xvfb_display = disp
                    log.info("Fallback Xvfb display ready: %s", disp)
                    return disp
                time.sleep(0.1)
            self._terminate(proc)
        self._xvfb_display = os.environ.get("DISPLAY") or ":0"
        return self._xvfb_display

    # -- human takeover (in-place window control over CDP; NO relaunch) ------
    @staticmethod
    def _run_coro_in_thread(coro_factory, timeout: float = 30.0) -> bool:
        """Run an async coroutine to completion in a throwaway thread+loop.

        Safe whether the caller is on a plain worker thread (agent tool) OR on
        the server's asyncio event loop (inbox endpoint) — we never touch the
        caller's loop, so 'asyncio.run() inside a running loop' can't happen.
        """
        box = {"ok": False, "err": None}

        def runner():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro_factory())
                box["ok"] = True
            except Exception as e:  # noqa: BLE001
                box["err"] = e
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout)
        if box["err"] is not None:
            log.warning("CDP window op failed: %s", box["err"])
        return box["ok"]

    def _cdp_set_window(self, team_id: str, show: bool) -> bool:
        """Show (restore + raise) or hide (minimize) the team browser window
        PURELY over CDP — no window manager, no xdotool, no relaunch. Best-effort:
        returns False (logged) on any failure rather than raising, so a takeover
        never crashes the agent."""
        with self._lock:
            info = self._browsers.get(team_id)
        if not info:
            return False
        cdp = f"http://127.0.0.1:{info['port']}"

        async def work():
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                # connect_over_cdp().close() only DISconnects our client; it never
                # kills the browser the agent is also using.
                b = await p.chromium.connect_over_cdp(cdp)
                try:
                    ctx = b.contexts[0] if b.contexts else await b.new_context()
                    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
                    s = await ctx.new_cdp_session(page)
                    wid = (await s.send("Browser.getWindowForTarget"))["windowId"]
                    if show:
                        # Un-minimize FIRST, then position on-screen (safe order).
                        await s.send("Browser.setWindowBounds",
                                     {"windowId": wid, "bounds": {"windowState": "normal"}})
                        await s.send("Browser.setWindowBounds",
                                     {"windowId": wid, "bounds":
                                      {"left": 60, "top": 60, "width": 1280, "height": 820}})
                        try:
                            await page.bring_to_front()
                        except Exception:
                            pass
                    else:
                        await s.send("Browser.setWindowBounds",
                                     {"windowId": wid, "bounds": {"windowState": "minimized"}})
                finally:
                    await b.close()

        return self._run_coro_in_thread(work)

    def begin_takeover(self, team_id: str, display: Optional[str] = None) -> Optional[str]:
        """Hand the live browser to a human: restore + raise its window IN PLACE
        (no relaunch, so the agent's CDP session survives). Returns the unchanged
        CDP URL. `display` is accepted for backward-compat but ignored — the
        browser is pinned to one display for life."""
        cdp = self.ensure_team_browser(team_id)
        with self._lock:
            self._takeover_active.add(team_id)
        self._cdp_set_window(team_id, show=True)
        return cdp

    def end_takeover(self, team_id: str) -> Optional[str]:
        """Hand control back: minimize the window IN PLACE (no relaunch)."""
        with self._lock:
            self._takeover_active.discard(team_id)
            info = self._browsers.get(team_id)
        self._cdp_set_window(team_id, show=False)
        return f"http://127.0.0.1:{info['port']}" if info else None

    # -- lifecycle ----------------------------------------------------------
    def ensure_team_browser(self, team_id: str) -> Optional[str]:
        """Return the team's CDP URL, launching/healing the browser as needed.

        Idempotent and cheap on the happy path (one HTTP health probe). Returns
        None when no Chromium is available so callers fall back gracefully.
        """
        if not self._chromium:
            return None
        with self._lock:
            info = self._browsers.get(team_id)
            if info and info["proc"].poll() is None and self._healthy(info["port"]):
                return f"http://127.0.0.1:{info['port']}"

            display = self._pick_display(team_id)

            # Reuse the team's port across relaunches so the cdp_url written into
            # agent configs stays valid within a server run.
            port = info["port"] if info else self._assign_port(team_id)

            # Reap a dead/stale process holding this slot.
            if info and info["proc"].poll() is None:
                self._terminate(info["proc"])

            profile = WORKSPACE_ROOT / team_id / ".browser-profile"
            profile.mkdir(parents=True, exist_ok=True)
            # A stale lock from an unclean shutdown blocks relaunch; clear it.
            for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
                try:
                    (profile / lock_name).unlink()
                except OSError:
                    pass

            # Headful (NO --headless) so the browser passes headless fingerprinting
            # (webdriver=false via AutomationControlled off). It is pinned to ONE
            # display for life and kept MINIMIZED during normal work; the
            # anti-throttle flags keep it painting — so the agent can still
            # screenshot/drive it even while minimized and invisible to the user.
            args = [
                self._chromium,
                f"--remote-debugging-port={port}",
                "--remote-debugging-address=127.0.0.1",
                f"--user-data-dir={profile}",
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-background-timer-throttling",
                "--window-size=1280,800",
                "about:blank",
            ]
            try:
                # start_new_session=True puts Chrome (and its renderer/gpu
                # helper children) in its own process group so we can reap the
                # WHOLE tree on shutdown instead of orphaning helpers.
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env={**os.environ, "DISPLAY": display},
                )
            except Exception as e:
                log.error("[%s] Failed to launch team browser: %s", team_id, e)
                return None

            self._browsers[team_id] = {
                "proc": proc, "port": port, "profile": str(profile),
                "display": display,
            }
            self._ports[team_id] = port

            # Wait (≤10s) for the CDP endpoint to come up.
            ready = False
            for _ in range(50):
                if proc.poll() is not None:
                    log.error("[%s] Team browser exited during startup", team_id)
                    return None
                if self._healthy(port):
                    log.info("[%s] Team browser ready on port %d display=%s (profile=%s)",
                             team_id, port, display, profile)
                    ready = True
                    break
                time.sleep(0.2)
            if not ready:
                log.error("[%s] Team browser did not become healthy on port %d", team_id, port)
                return None

        # Outside the lock: tuck the window away (minimized) unless this team is
        # mid-takeover. Best-effort — the agent can drive a minimized window fine.
        if team_id not in self._takeover_active:
            self._cdp_set_window(team_id, show=False)
        return f"http://127.0.0.1:{port}"

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """Terminate Chrome and its whole process group (renderers/gpu helpers)."""
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        pass
                else:
                    proc.kill()
        except Exception:
            pass

    def shutdown_all(self) -> None:
        """Terminate all team browsers (profiles persist on disk for next run)."""
        with self._lock:
            for team_id, info in self._browsers.items():
                log.info("[%s] Stopping team browser (port %d)", team_id, info["port"])
                self._terminate(info["proc"])
            self._browsers.clear()
            if self._xvfb_proc is not None:
                self._terminate(self._xvfb_proc)
                self._xvfb_proc = None
                self._xvfb_display = None


# Process-wide singleton.
team_browser_manager = TeamBrowserManager()
