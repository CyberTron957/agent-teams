"""Version-update detection for self-hosted Agent Teams installs.

The supported installs both track the GitHub ``main`` branch — the ``curl|bash``
installer git-clones + ``pip install -e .``, and Docker builds from the same
checkout. So the truthful "what you'd get if you update" signal is the version
on ``main``, not a package registry (PyPI/pip is being sunset).

This module is the single source of truth for "is there a newer version, and how
would you get it." It is deliberately network-failure tolerant: every lookup can
return ``None``/cached data so the dashboard and CLI never hang or crash when
GitHub is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from packaging.version import InvalidVersion, parse as _parse_version

from swarm_server import __version__
from swarm_server.config import PROJECT_ROOT, _is_source_checkout, _resolve_data_root

log = logging.getLogger(__name__)

# Raw view of pyproject.toml on the branch a `git pull` would fast-forward to.
_LATEST_PYPROJECT_URL = (
    "https://raw.githubusercontent.com/"
    "CyberTron957/hermes-mission-control/main/pyproject.toml"
)

# Re-hit GitHub at most this often; on-load checks read the cache in between so
# the dashboard never blocks on the network.
_CACHE_TTL_SECONDS = 6 * 60 * 60

_VERSION_RE = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _cache_path() -> Path:
    return _resolve_data_root() / "update_check.json"


def get_install_method() -> str:
    """How this install would be upgraded: ``"docker"``, ``"git"``, or ``"unknown"``.

    ``unknown`` is treated like ``git`` by callers, but they warn that the upgrade
    path is unverified.
    """
    if os.path.exists("/.dockerenv") or os.environ.get("SWARM_DATA_DIR") == "/data":
        return "docker"
    if _is_source_checkout() and (PROJECT_ROOT / ".git").exists():
        return "git"
    return "unknown"


def _upgrade_hint(method: str) -> str:
    if method == "docker":
        return "git pull && docker compose build --pull && docker compose up -d"
    # git / unknown
    return "agent-teams update   (git pull --ff-only + pip install -e .)"


def fetch_latest_version(timeout: float = 4.0) -> Optional[str]:
    """Read the version string from ``main``'s pyproject.toml.

    Returns the version (e.g. ``"0.5.0"``) or ``None`` on any network/parse error —
    this is the exact value a ``git pull`` would bring down.
    """
    try:
        import httpx

        resp = httpx.get(_LATEST_PYPROJECT_URL, timeout=timeout)
        resp.raise_for_status()
        m = _VERSION_RE.search(resp.text)
        if not m:
            log.warning("[update_check] no version field in remote pyproject.toml")
            return None
        return m.group(1).strip()
    except Exception as e:  # network down, DNS, timeout, parse — all non-fatal
        log.debug("[update_check] latest-version lookup failed: %s", e)
        return None


def _read_cache() -> Optional[dict]:
    try:
        return json.loads(_cache_path().read_text())
    except Exception:
        return None


def _write_cache(data: dict) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception as e:
        log.debug("[update_check] could not write cache: %s", e)


def _is_newer(latest: str, current: str) -> bool:
    """True if ``latest`` is a strictly greater release than ``current``.

    Falls back to a string inequality if either side isn't PEP 440 parseable
    (e.g. ``current`` is the ``0.0.0+source`` placeholder)."""
    try:
        return _parse_version(latest) > _parse_version(current)
    except InvalidVersion:
        return latest != current


def check_for_update(force: bool = False, timeout: float = 4.0) -> dict:
    """Return update status, hitting GitHub at most once per TTL.

    Shape::

        {current, latest, update_available, install_method, upgrade_hint, checked_at}

    ``latest`` is ``None`` if the network lookup failed and no cache exists.
    """
    current = __version__
    method = get_install_method()

    cache = _read_cache()
    now = time.time()
    if (
        not force
        and cache
        and isinstance(cache.get("checked_at"), (int, float))
        and (now - cache["checked_at"]) < _CACHE_TTL_SECONDS
    ):
        latest = cache.get("latest")
    else:
        latest = fetch_latest_version(timeout=timeout)
        if latest is None and cache:
            latest = cache.get("latest")  # fall back to last known good
        else:
            _write_cache({"latest": latest, "checked_at": now})

    return {
        "current": current,
        "latest": latest,
        "update_available": bool(latest) and _is_newer(latest, current),
        "install_method": method,
        "upgrade_hint": _upgrade_hint(method),
        "checked_at": now,
    }
