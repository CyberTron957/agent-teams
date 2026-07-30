"""Messaging-gateway configuration: a teams-wide gateway the swarm uses to push
human-inbox questions out to a chat platform (Telegram/Discord/…) and accept the
human's reply back.

WHERE THE CREDENTIALS LIVE (and why it is deliberately NOT a Hermes home)
------------------------------------------------------------------------
The swarm's bot token lives in ``data/gateway.json`` (mode 0600), owned by
agent-teams — **not** in the shared Hermes home's ``.env``.

That is a structural safety property, not a style choice. A Telegram/Discord bot
token permits exactly ONE long-polling listener. Any ``hermes gateway run`` whose
``HERMES_HOME`` points at a home is enabled for a platform purely by finding that
platform's token in the home's ``.env`` (``gateway.config.load_gateway_config()``
reads ``os.getenv("TELEGRAM_BOT_TOKEN")``). So while the swarm's token sat in
``data/.hermes-shared/.env``, a stray or service-managed ``hermes gateway run``
against that home would silently start a SECOND listener on the same bot. The two
then split updates — a human's reply could be consumed by the other process and
never reach the agent blocked waiting for it.

Keeping the token out of every Hermes ``.env`` makes that impossible rather than
merely detected: an external gateway pointed at the swarm home finds no token and
cannot enable the platform at all.

Outbound sending is unaffected. ``apply_gateway_env()`` injects the credentials
into THIS process's ``os.environ`` at startup, so ``send_message_tool`` and any
agent's own messaging tools keep working exactly as before. The difference is the
credential is only ever in agent-teams' own memory and its own 0600 file — never
on disk in a location an external gateway process reads.

The platform catalogue still comes straight from Hermes
(``hermes_cli.gateway._all_platforms``), exactly as the model setup reuses
``PROVIDER_REGISTRY``.

The operator's ``~/.hermes`` bot IS still detected and shown in the setup UI
(``detect_global_gateway``), but only so the operator can see the clash risk and
configure a separate bot. Reusing it is refused by the setup route.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from teams_server.config import DATA_ROOT, ensure_hermes_importable
# Reuse the model-setup plumbing verbatim — same homes, same HERMES_HOME override,
# same .env parsing. Keeps the two "reuse Hermes config" flows in lockstep.
from teams_server.model_config import (
    GLOBAL_HERMES_HOME,
    SHARED_HERMES_HOME,
    _home,
    _parse_env_file,
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
    """Platform catalogue for the setup UI: ``[{key, label, token_var, status}]``.
    Mirrors ``build_provider_presets()`` + ``/providers``.

    Status comes from OUR store, not from the Hermes home's ``.env`` — the swarm's
    token deliberately no longer lives there (see the module docstring), so asking
    Hermes would report every platform as unconfigured."""
    out: List[Dict[str, Any]] = []
    try:
        platforms = _messaging_platforms()
    except Exception as e:
        log.warning("gateway catalogue unavailable (%s) — empty list", e)
        return out
    active = _read_store()
    active_key = active.get("key", "") if active.get("token") else ""
    if not active_key:
        legacy = _read_gateway_from_home(SHARED_HERMES_HOME)   # pre-migration
        active_key = legacy["key"] if legacy else ""
    for p in platforms:
        key = p.get("key")
        out.append({
            "key": key,
            "label": p.get("label") or key,
            "token_var": p.get("token_var", ""),
            # Telegram is the only platform we can also RECEIVE replies on today
            # (the inbound listener). Outbound works for all; the UI uses this to
            # mark which give two-way inbox.
            "reply_supported": key == "telegram",
            "status": "configured" if key == active_key else "not configured",
        })
    return out


# ---------------------------------------------------------------------------
# The swarm's own credential store (NOT a Hermes home — see module docstring)
# ---------------------------------------------------------------------------
GATEWAY_STORE = DATA_ROOT / "gateway.json"

_store_lock = threading.Lock()


def _read_store() -> Dict[str, str]:
    """The configured gateway as ``{key, token, home_chat_id}``, or ``{}``."""
    try:
        with _store_lock:
            if not GATEWAY_STORE.exists():
                return {}
            data = json.loads(GATEWAY_STORE.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("Could not read gateway store (%s) — treating as unconfigured", e)
        return {}


def _write_store(key: str, token: str, home_chat_id: str = "",
                 bot_username: str = "", home_chat_name: str = "") -> None:
    """Persist the gateway atomically with owner-only permissions. The file holds a
    live bot token, so it must never be group/world readable.

    ``bot_username``/``home_chat_name`` are display-only (never used for auth or
    matching) — they're what let the dashboard say "Connected to @my_bot, replying
    in chat with Ada" instead of a bare "Connected", without exposing the token or
    numeric chat id."""
    payload = {
        "key": key, "token": token, "home_chat_id": home_chat_id or "",
        "bot_username": bot_username or "", "home_chat_name": home_chat_name or "",
    }
    with _store_lock:
        GATEWAY_STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = GATEWAY_STORE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)          # before the rename, so it is never 0644
        except OSError:
            pass
        os.replace(tmp, GATEWAY_STORE)
        try:
            os.chmod(GATEWAY_STORE, 0o600)
        except OSError:
            pass


def apply_gateway_env() -> bool:
    """Load the stored credentials into THIS process's environment.

    This is what keeps outbound sending working after the token stopped living in
    a Hermes ``.env``: ``gateway.config.load_gateway_config()`` reads the platform
    token from ``os.getenv``, so an in-process ``send_message_tool`` call (ours or
    an agent's own tool) behaves exactly as before. An EXTERNAL ``hermes gateway
    run`` gets a fresh environment and reads the home's ``.env``, where the token
    no longer is — so it cannot enable the platform or start a rival listener."""
    data = _read_store()
    key, token = data.get("key", ""), data.get("token", "")
    if not key or not token:
        return False
    os.environ[_token_var(key)] = token
    chat = data.get("home_chat_id", "")
    if chat:
        os.environ[_home_channel_var(key)] = chat
    return True


def migrate_gateway_out_of_hermes_home() -> bool:
    """One-time move of gateway creds from the shared Hermes ``.env`` into our own
    store, deleting them from the ``.env``.

    Deleting is the whole point: as long as the token is readable in a Hermes
    home, any ``hermes gateway run`` pointed at that home starts a competing
    listener for the same bot. Existing installs are silently vulnerable until
    this runs. Returns True when something was migrated."""
    legacy = _read_gateway_from_home(SHARED_HERMES_HOME)
    if not legacy or not legacy.get("token"):
        return False

    key = legacy["key"]
    # Keep whatever the store already has if it is configured; otherwise adopt the
    # legacy values so the user's existing gateway keeps working untouched.
    if not _read_store().get("token"):
        _write_store(key, legacy["token"], legacy.get("home_chat_id", ""))

    removed: List[str] = []
    try:
        with _home(SHARED_HERMES_HOME):
            from hermes_cli.config import remove_env_value

            for var in (_token_var(key), _home_channel_var(key)):
                try:
                    if remove_env_value(var):
                        removed.append(var)
                except Exception as e:
                    log.warning("Could not remove %s from the shared Hermes .env: %s", var, e)
    except Exception as e:
        log.warning("Gateway migration could not open the shared Hermes home: %s", e)

    # remove_env_value() also clears os.environ, which would break in-process
    # sending — put the credentials back into the environment from our store.
    apply_gateway_env()

    log.info("Gateway credentials moved out of the Hermes home into %s "
             "(removed from .env: %s). An external 'hermes gateway run' against "
             "%s can no longer poll this bot.",
             GATEWAY_STORE.name, ", ".join(removed) or "nothing",
             SHARED_HERMES_HOME.name)
    return True


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
            key = str(p.get("key") or "")
            return {
                "key": key,
                "label": p.get("label") or key,
                "token": parsed.get(tv, ""),
                "home_chat_id": parsed.get(_home_channel_var(key), ""),
            }
    return None


def get_active_gateway() -> Optional[Dict[str, Any]]:
    """The swarm's configured gateway, or None.

    Reads agent-teams' own store first. Falls back to READING (never writing) the
    shared Hermes home's ``.env`` so an install that hasn't migrated yet keeps
    working; ``migrate_gateway_out_of_hermes_home()`` moves it on next startup.

    Never falls back to the operator's ``~/.hermes``: a bot token supports only one
    listener, so reusing the operator's bot would make the two steal each other's
    replies. The swarm needs its own dedicated bot."""
    data = _read_store()
    if data.get("key") and data.get("token"):
        key = data["key"]
        label = key
        try:
            for p in _messaging_platforms():
                if p.get("key") == key:
                    label = p.get("label") or key
                    break
        except Exception:
            pass
        return {
            "key": key,
            "label": label,
            "token": data["token"],
            "home_chat_id": data.get("home_chat_id", ""),
            "bot_username": data.get("bot_username", ""),
            "home_chat_name": data.get("home_chat_name", ""),
        }
    return _read_gateway_from_home(SHARED_HERMES_HOME)


def detect_global_gateway() -> Optional[Dict[str, Any]]:
    """A gateway already configured in the operator's ``~/.hermes`` (to offer
    "adopt"), or None. Mirrors ``detect_global_hermes_model()``."""
    return _read_gateway_from_home(GLOBAL_HERMES_HOME)


def is_gateway_configured() -> bool:
    return get_active_gateway() is not None


# ---------------------------------------------------------------------------
# Write a gateway
# ---------------------------------------------------------------------------
def set_gateway(key: str, token: str, home_chat_id: str = "",
                bot_username: str = "", home_chat_name: str = "") -> None:
    """Persist a platform's bot token (+ optional home channel) into agent-teams'
    own store, load it into this process's environment, and make sure no copy is
    left behind in the shared Hermes ``.env``.

    Writing to our own store rather than the Hermes home is what prevents a second
    listener: see the module docstring."""
    _write_store(key, token, home_chat_id, bot_username, home_chat_name)
    apply_gateway_env()

    # Purge any credential a previous version left in the Hermes home, otherwise
    # an external `hermes gateway run` against that home would still find a token.
    try:
        with _home(SHARED_HERMES_HOME):
            from hermes_cli.config import remove_env_value

            for var in (_token_var(key), _home_channel_var(key)):
                try:
                    remove_env_value(var)
                except Exception:
                    pass
    except Exception as e:
        log.warning("Could not purge legacy gateway creds from the Hermes home: %s", e)
    apply_gateway_env()      # remove_env_value() also clears os.environ

    log.info("Gateway saved to %s: platform=%s home_chat=%s",
             GATEWAY_STORE.name, key, "set" if home_chat_id else "unset")


def clear_gateway() -> None:
    """Remove the configured gateway entirely (the "Disconnect" action).

    Deletes agent-teams' own store and drops the credential from this process's
    environment so outbound sending stops immediately too. The caller is
    responsible for stopping the inbound listener — this function only owns the
    stored config."""
    key = ""
    try:
        data = _read_store()
        key = data.get("key", "")
    except Exception:
        pass
    with _store_lock:
        try:
            if GATEWAY_STORE.exists():
                GATEWAY_STORE.unlink()
        except Exception as e:
            log.warning("Could not remove gateway store: %s", e)
    if key:
        for var in (_token_var(key), _home_channel_var(key)):
            os.environ.pop(var, None)
    log.info("Gateway disconnected — %s removed", GATEWAY_STORE.name)
