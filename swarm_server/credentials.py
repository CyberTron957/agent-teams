"""Per-team credentials registry.

Why this exists: secrets used to live inline in workspace.md, which (a) rides
in EVERY LLM prompt of every agent on the team, and (b) carries no
what-is-this-for metadata — which is how an SMTP app password ended up typed
into LinkedIn and X login forms. Here each credential is stored once, outside
the prompt stream, under a stable site key with an explicit ``purpose``;
agents fetch one on demand via the get_credential tool and see the purpose
alongside the secret.

Storage: ``data/teams/<team>/credentials.json`` (0600), shape::

    {"gmail-smtp": {"username": "...", "secret": "...",
                    "purpose": "SMTP email sending ONLY — not a login password",
                    "notes": "host smtp.gmail.com:587"}}

Plain JSON on local disk — the threat model is prompt leakage and wrong-purpose
use, not a hostile host (the host already holds every agent's workspace).
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from swarm_server.config import WORKSPACE_ROOT

log = logging.getLogger("swarm.credentials")

_LOCK = threading.Lock()


def _creds_path(team_id: str) -> Path:
    return WORKSPACE_ROOT / team_id / "credentials.json"


def load_credentials(team_id: str) -> Dict[str, Dict[str, Any]]:
    path = _creds_path(team_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("could not read %s (%s)", path, e)
        return {}


def save_credential(team_id: str, site: str, username: str, secret: str,
                    purpose: str, notes: str = "") -> Dict[str, Any]:
    """Add or replace one credential. Returns the stored entry (with secret)."""
    site = (site or "").strip().lower()
    if not site:
        raise ValueError("site key is required")
    if not purpose or not purpose.strip():
        raise ValueError("purpose is required — it is what prevents wrong-purpose use")
    entry = {"username": (username or "").strip(), "secret": secret or "",
             "purpose": purpose.strip(), "notes": (notes or "").strip()}
    with _LOCK:
        creds = load_credentials(team_id)
        creds[site] = entry
        path = _creds_path(team_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(creds, f, indent=2, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    return entry


def delete_credential(team_id: str, site: str) -> bool:
    site = (site or "").strip().lower()
    with _LOCK:
        creds = load_credentials(team_id)
        if site not in creds:
            return False
        del creds[site]
        path = _creds_path(team_id)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(creds, f, indent=2, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    return True


def get_credential(team_id: str, site: str) -> Optional[Dict[str, Any]]:
    return load_credentials(team_id).get((site or "").strip().lower())


def list_credentials_public(team_id: str) -> Dict[str, Dict[str, str]]:
    """Site -> {username, purpose, notes} — NO secrets. Safe for prompts/UI."""
    return {site: {"username": e.get("username", ""),
                   "purpose": e.get("purpose", ""),
                   "notes": e.get("notes", "")}
            for site, e in load_credentials(team_id).items()}


# ---------------------------------------------------------------------------
# Tool schemas + handlers (registered by tools._register_custom_tools)
# ---------------------------------------------------------------------------

GET_CREDENTIAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_credential",
        "description": (
            "Fetch ONE stored team credential (username + secret) by its site "
            "key — e.g. get_credential(site='gmail-smtp'). ALWAYS use this "
            "instead of credentials found in documents. Each credential states "
            "its PURPOSE: use it ONLY for that purpose — an SMTP app password "
            "is not a website login, an API key is not an email password. If "
            "no credential exists for the account you need, do NOT guess or "
            "reuse another one: call list_credentials to see what exists, then "
            "request_human_takeover (interactive login) or ask_human."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "site": {"type": "string",
                         "description": "The credential's site key (see list_credentials)."},
            },
            "required": ["site"],
        },
    },
}

LIST_CREDENTIALS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_credentials",
        "description": (
            "List the team's stored credentials: site key, username, and "
            "purpose — never the secrets. Check this BEFORE attempting any "
            "login or authenticated call, so you use the right credential for "
            "the right purpose (or learn that none exists and escalate to "
            "request_human_takeover / ask_human instead of guessing)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def _caller_team(kwargs: dict) -> "tuple[str, str]":
    """(caller, team_id) from a tool call's task_id, via the live registry."""
    task_id = kwargs.get("task_id", "") or ""
    caller = task_id.split(":", 1)[1] if task_id.startswith("agent_name:") else task_id
    try:
        from swarm_server.tools import _daemon_registry

        daemon = _daemon_registry.get(caller)
        team = (getattr(daemon, "cfg", None) or {}).get("team_id", "")
        return caller, team or "default"
    except Exception:
        return caller, "default"


def get_credential_handler(args: dict, **kwargs) -> str:
    site = (args.get("site") or "").strip().lower()
    if not site:
        return json.dumps({"success": False, "error": "'site' is required."})
    caller, team = _caller_team(kwargs)
    entry = get_credential(team, site)
    if entry is None:
        available = sorted(load_credentials(team))
        return json.dumps({
            "success": False,
            "error": f"no credential stored for '{site}'",
            "available_sites": available,
            "hint": "Do NOT reuse a credential meant for something else. If the "
                    "human must log in interactively, use request_human_takeover.",
        })
    log.info("[credentials] %s fetched '%s' (team %s)", caller, site, team)
    return json.dumps({
        "success": True, "site": site,
        "username": entry.get("username", ""),
        "secret": entry.get("secret", ""),
        "purpose": entry.get("purpose", ""),
        "notes": entry.get("notes", ""),
        "warning": "Use ONLY for the stated purpose. Never type this secret "
                   "into a different site's login form.",
    }, ensure_ascii=False)


def list_credentials_handler(args: dict, **kwargs) -> str:
    _, team = _caller_team(kwargs)
    return json.dumps({"success": True,
                       "credentials": list_credentials_public(team),
                       "note": "Secrets are returned by get_credential(site=...) only."},
                      ensure_ascii=False)
