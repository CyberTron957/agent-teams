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
        # ONE hidden Xvfb display shared by all team browsers (headful but offscreen).
        self._xvfb_proc: Optional[subprocess.Popen] = None
        self._xvfb_display: Optional[str] = None
        # team_id -> desired DISPLAY (hidden Xvfb normally; real screen on takeover).
        self._team_display: Dict[str, str] = {}
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

    # -- virtual display / human takeover -----------------------------------
    def _ensure_xvfb(self) -> str:
        """Lazily start ONE hidden Xvfb display shared by all team browsers.

        Headful Chrome on this display is invisible to the user but defeats
        headless fingerprinting. Falls back to the real $DISPLAY (or ':2') when
        Xvfb is unavailable, so the pool still works on a plain desktop.
        """
        if self._xvfb_display and self._xvfb_proc and self._xvfb_proc.poll() is None:
            return self._xvfb_display
        xvfb = shutil.which("Xvfb")
        if not xvfb:
            self._xvfb_display = os.environ.get("DISPLAY") or ":2"
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
                    log.info("Hidden Xvfb display ready: %s", disp)
                    return disp
                time.sleep(0.1)
            self._terminate(proc)
        self._xvfb_display = os.environ.get("DISPLAY") or ":2"
        return self._xvfb_display

    def begin_takeover(self, team_id: str, display: Optional[str] = None) -> Optional[str]:
        """Bring the team browser onto a VISIBLE display for a human takeover.

        Relaunches the SAME persistent profile on the real desktop so the human
        can log in / solve a challenge; cookies persist via the profile. The CDP
        port is reused, so the returned CDP URL is unchanged.
        """
        target = (display or os.environ.get("HANDOFF_DISPLAY")
                  or os.environ.get("DISPLAY") or ":2")
        with self._lock:
            self._team_display[team_id] = target
        return self.ensure_team_browser(team_id)

    def end_takeover(self, team_id: str) -> Optional[str]:
        """Send the team browser back to the hidden display after a takeover."""
        with self._lock:
            self._team_display[team_id] = self._ensure_xvfb()
        return self.ensure_team_browser(team_id)

    # -- lifecycle ----------------------------------------------------------
    def ensure_team_browser(self, team_id: str) -> Optional[str]:
        """Return the team's CDP URL, launching/healing the browser as needed.

        Idempotent and cheap on the happy path (one HTTP health probe). Returns
        None when no Chromium is available so callers fall back gracefully.
        """
        if not self._chromium:
            return None
        with self._lock:
            desired_display = self._team_display.get(team_id) or self._ensure_xvfb()
            info = self._browsers.get(team_id)
            if (info and info["proc"].poll() is None and self._healthy(info["port"])
                    and info.get("display") == desired_display):
                return f"http://127.0.0.1:{info['port']}"

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

            # Headful (NO --headless) so the browser passes headless fingerprinting;
            # it lives on a hidden Xvfb display by default and is moved to the real
            # screen only during a human takeover. Turning AutomationControlled off
            # makes navigator.webdriver report false.
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
                    env={**os.environ, "DISPLAY": desired_display},
                )
            except Exception as e:
                log.error("[%s] Failed to launch team browser: %s", team_id, e)
                return None

            self._browsers[team_id] = {
                "proc": proc, "port": port, "profile": str(profile),
                "display": desired_display,
            }
            self._ports[team_id] = port

            # Wait (≤10s) for the CDP endpoint to come up.
            for _ in range(50):
                if proc.poll() is not None:
                    log.error("[%s] Team browser exited during startup", team_id)
                    return None
                if self._healthy(port):
                    log.info(
                        "[%s] Team browser ready on port %d (profile=%s)",
                        team_id, port, profile,
                    )
                    return f"http://127.0.0.1:{port}"
                time.sleep(0.2)
            log.error("[%s] Team browser did not become healthy on port %d", team_id, port)
            return None

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
