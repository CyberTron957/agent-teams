"""Messaging-gateway configuration: a teams-wide gateway the swarm uses to push
human-inbox questions out to a chat platform (Telegram/Discord/…) and accept the
human's reply back.

Hermes-native, and a deliberate mirror of ``model_config.py``: the gateway is
stored in Hermes' OWN config — platform credentials as env vars in a Hermes home's
``.env`` (``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_HOME_CHANNEL``, etc.) — written via
Hermes' ``save_env_value`` wrapped in a HERMES_HOME override so we target the
teams-managed shared home (``data/.hermes-shared``). The platform catalogue and
per-platform "configured?" status come straight from Hermes
(``hermes_cli.gateway._all_platforms`` / ``_platform_status``), exactly as the
model setup reuses ``PROVIDER_REGISTRY``.

The active gateway is read exclusively from the shared home (``data/.hermes-shared``).
NO fallback to the operator's ``~/.hermes`` is performed at runtime — a Telegram/Discord
bot token can only have one long-polling listener, so using the same bot as the operator's
main ``hermes gateway`` would cause the two to steal each other's replies. Agent-teams
must be configured with its OWN dedicated bot (set via the dashboard → Gateway).

The operator's ``~/.hermes`` bot IS still detected and shown in the setup UI (detect_global_gateway),
but only so the operator can see the clash risk and choose to configure a separate one.

NOTE: gateway creds live in ``.env`` (env vars), NOT ``config.yaml`` — so the
read/write path here mirrors ``model_config``'s ``.env`` helpers, not its
``cfg["model"]`` path.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from teams_server.config import ensure_hermes_importable
# Reuse the model-setup plumbing verbatim — same homes, same HERMES_HOME override,
# same .env parsing. Keeps the two "reuse Hermes config" flows in lockstep.
from teams_server.model_config import (
    GLOBAL_HERMES_HOME,
    SHARED_HERMES_HOME,
    _home,
    _parse_env_file,
    _read_env_value,
)

log = logging.getLogger("teams.gateway")

# Adapters that aren't chat surfaces for human notifications: the local CLI, the
# cron scheduler, the OpenAI-compatible API server, and inbound-only webhook
# callbacks. Everything else with a bot token is offerable.
_NON_MESSAGING = {"cli", "cron", "api_server", "webhook", "wecom_callback"}


# ---------------------------------------------------------------------------
# Env-var naming (the two keys a platform needs to send + receive)
# ---------------------------------------------------------------------------
def _home_channel_var(key: str) -> str:
    """The .env var holding a platform's home-channel id (where we deliver to and
    listen on), e.g. telegram -> TELEGRAM_HOME_CHANNEL."""
    return f"{key.upper()}_HOME_CHANNEL"


def _token_var(key: str) -> str:
    """The .env var holding a platform's bot token. Prefer the catalogue's declared
    token_var; fall back to <KEY>_BOT_TOKEN."""
    try:
        for p in _messaging_platforms():
            if p.get("key") == key:
                return p.get("token_var") or f"{key.upper()}_BOT_TOKEN"
    except Exception:
        pass
    return f"{key.upper()}_BOT_TOKEN"


# ---------------------------------------------------------------------------
# Catalogue + per-platform status (straight from Hermes)
# ---------------------------------------------------------------------------
def _messaging_platforms() -> List[Dict[str, Any]]:
    """Hermes' platform catalogue, filtered to chat surfaces that carry a bot token
    (the ones a human-inbox notification can be delivered to)."""
    ensure_hermes_importable()
    from hermes_cli.gateway import _all_platforms

    out: List[Dict[str, Any]] = []
    for p in _all_platforms():
        key = p.get("key", "")
        if key in _NON_MESSAGING or not p.get("token_var"):
            continue
        out.append(p)
    return out


def list_gateways() -> List[Dict[str, Any]]:
    """Platform catalogue for the setup UI: ``[{key, label, token_var, status}]``
    with each platform's status read against the shared home. Mirrors
    ``build_provider_presets()`` + ``/providers``."""
    out: List[Dict[str, Any]] = []
    try:
        platforms = _messaging_platforms()
    except Exception as e:
        log.warning("gateway catalogue unavailable (%s) — empty list", e)
        return out
    with _home(SHARED_HERMES_HOME):
        from hermes_cli.gateway import _platform_status

        for p in platforms:
            try:
                status = _platform_status(p)
            except Exception:
                status = "not configured"
            out.append({
                "key": p.get("key"),
                "label": p.get("label") or p.get("key"),
                "token_var": p.get("token_var", ""),
                # Telegram is the only platform we can also RECEIVE replies on today
                # (the inbound listener). Outbound works for all; the UI uses this to
                # mark which give two-way inbox.
                "reply_supported": p.get("key") == "telegram",
                "status": status,
            })
    return out


# ---------------------------------------------------------------------------
# Read a configured gateway from a Hermes home
# ---------------------------------------------------------------------------
def _read_gateway_from_home(home: Path) -> Optional[Dict[str, Any]]:
    """First messaging platform that has a bot token set in ``home/.env``, or None.
    Returns ``{key, label, token, home_chat_id}`` (token included for server-side
    adopt — callers that face the client must strip it)."""
    try:
        platforms = _messaging_platforms()
    except Exception:
        return None
    parsed = _parse_env_file(home / ".env")
    for p in platforms:
        tv = p.get("token_var", "")
        if tv and parsed.get(tv):
            key = p.get("key")
            return {
                "key": key,
                "label": p.get("label") or key,
                "token": parsed.get(tv, ""),
                "home_chat_id": parsed.get(_home_channel_var(key), ""),
            }
    return None


def get_active_gateway() -> Optional[Dict[str, Any]]:
    """The swarm's configured gateway (from the shared home only), or None.

    Reads ONLY from ``data/.hermes-shared`` — NOT from ``~/.hermes``.
    A Telegram/Discord bot token supports only one long-polling listener; sharing
    the same bot as the operator's running ``hermes gateway`` would cause the two
    listeners to steal each other's replies. Agent-teams must be configured with
    its own dedicated bot (set via the dashboard Gateway button)."""
    return _read_gateway_from_home(SHARED_HERMES_HOME)


def detect_global_gateway() -> Optional[Dict[str, Any]]:
    """A gateway already configured in the operator's ``~/.hermes`` (to offer
    "adopt"), or None. Mirrors ``detect_global_hermes_model()``."""
    return _read_gateway_from_home(GLOBAL_HERMES_HOME)


def is_gateway_configured() -> bool:
    return get_active_gateway() is not None


# ---------------------------------------------------------------------------
# Write a gateway into the shared home
# ---------------------------------------------------------------------------
def set_gateway(key: str, token: str, home_chat_id: str = "") -> None:
    """Persist a platform's bot token (+ optional home channel) into the shared
    home's ``.env`` via Hermes' ``save_env_value``. Mirrors ``set_default_model``."""
    tv = _token_var(key)
    with _home(SHARED_HERMES_HOME):
        from hermes_cli.config import save_env_value

        if token:
            save_env_value(tv, token)
        if home_chat_id:
            save_env_value(_home_channel_var(key), home_chat_id)
    log.info("Gateway written to %s: platform=%s home_chat=%s",
             SHARED_HERMES_HOME.name, key, "set" if home_chat_id else "unset")


def adopt_global_gateway() -> Optional[Dict[str, Any]]:
    """Copy the gateway detected in ``~/.hermes`` into the shared home. Returns
    ``{key, label}`` on success, None if nothing detected. Mirrors the model's
    ``adopt_detected`` branch."""
    det = detect_global_gateway()
    if not det or not det.get("token"):
        return None
    set_gateway(det["key"], det["token"], det.get("home_chat_id", ""))
    return {"key": det["key"], "label": det.get("label", det["key"])}
