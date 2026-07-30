"""Runtime bridge between the human inbox and a configured messaging gateway.

Two directions, both reusing Hermes' own gateway:

  OUTBOUND  ``forward_question()`` — when an agent blocks on ``ask_human`` /
    ``request_human_takeover`` (the single ``_block_on_human`` choke point in
    ``tools.py``), push the question to the swarm gateway's home channel via
    Hermes' one-shot ``send_message_tool`` (no running ``hermes gateway`` needed).
    Runs on a throwaway thread so a slow network send never extends the agent's
    block. Remembers the sent ``message_id`` so a reply can be matched back to the
    exact question.

  INBOUND   ``start_inbox_gateway_listener()`` — a single asyncio task long-polls
    Telegram ``getUpdates`` for replies in the home channel and routes them to the
    SAME delivery path the dashboard uses (``_finalize_human_answer``), so an
    answer from chat is indistinguishable from one typed in the dashboard. Started
    from the FastAPI lifespan hook and re-started by ``POST /setup/gateway``;
    a no-op unless a reply-capable gateway (Telegram today) is configured with a
    home channel.

Telegram is the first inbound adapter (simplest API, most common). Outbound is
already platform-agnostic via ``send_message_tool``; ``_poll_telegram`` is the
clear per-platform seam for adding Discord/Slack later without touching the
forward path.
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("teams.gateway.bridge")

# message_id (as str) -> (qid, agent_name) for replies we can match exactly.
# Bounded like the inbox registry; resets on restart (a reply after restart falls
# back to newest-pending, same as the dashboard respond route).
_sent_index: Dict[str, Tuple[str, str]] = {}
_sent_lock = threading.Lock()
_SENT_INDEX_CAP = 500


def _remember_sent(message_id: str, qid: str, agent_name: str) -> None:
    if not message_id:
        return
    with _sent_lock:
        _sent_index[str(message_id)] = (qid, agent_name)
        if len(_sent_index) > _SENT_INDEX_CAP:
            # Drop oldest insertions (dict preserves insertion order).
            for k in list(_sent_index.keys())[: len(_sent_index) - _SENT_INDEX_CAP]:
                _sent_index.pop(k, None)


def _lookup_sent(message_id: str) -> Optional[Tuple[str, str]]:
    with _sent_lock:
        return _sent_index.get(str(message_id))


# ---------------------------------------------------------------------------
# OUTBOUND: forward a blocking question to the gateway
# ---------------------------------------------------------------------------
def _do_forward(qid: str, agent_name: str, question: str, kind: str) -> None:
    """Send the question and remember the resulting message_id. Best-effort: any
    failure is logged once and swallowed — the question still lives in the
    dashboard inbox, so the human can always answer there."""
    try:
        from teams_server.gateway_config import get_active_gateway, SHARED_HERMES_HOME
        from teams_server.model_config import _home

        gw = get_active_gateway()
        if not gw:
            return  # no gateway configured — nothing to do
        platform = gw["key"]
        team_hint = ""
        try:
            from teams_server.config import load_agents_config
            team_hint = load_agents_config()["agents"].get(agent_name, {}).get("team_id", "") or ""
        except Exception:
            pass
        who = f"{agent_name}" + (f" ({team_hint})" if team_hint else "")
        label = "needs you to take over" if kind == "takeover" else "needs your input"
        text = (
            f"🔔 *{who}* {label}\n\n"
            f"{question}\n\n"
            "↩️ Reply to this message to answer (or open the dashboard inbox)."
        )

        import json as _json
        with _home(SHARED_HERMES_HOME):
            from tools.send_message_tool import send_message_tool

            raw = send_message_tool({"action": "send", "target": platform, "message": text})
        result = _json.loads(raw) if isinstance(raw, str) else (raw or {})
        if result.get("success"):
            mid = result.get("message_id")
            if mid is not None:
                _remember_sent(str(mid), qid, agent_name)
            log.info("[gateway] forwarded question %s from %s to %s", qid, agent_name, platform)
        else:
            log.warning("[gateway] forward failed (%s): %s",
                        platform, result.get("error") or result)
    except Exception as e:
        log.warning("[gateway] forward_question errored (non-fatal): %s", e)


def forward_question(qid: str, agent_name: str, question: str, kind: str = "question") -> None:
    """Fire-and-forget: push a blocking human question to the gateway on a separate
    thread so the agent's worker thread never waits on the network. Safe to call
    even when no gateway is configured (cheap no-op inside the thread)."""
    try:
        threading.Thread(
            target=_do_forward, args=(qid, agent_name, question, kind),
            name=f"gw-forward-{qid[:8]}", daemon=True,
        ).start()
    except Exception as e:
        log.warning("[gateway] could not spawn forward thread: %s", e)


# ---------------------------------------------------------------------------
# INBOUND: listen for replies on the gateway and deliver them to the agent
# ---------------------------------------------------------------------------
# A bot token permits exactly ONE long-polling consumer. When something else is
# already polling (a stray `hermes gateway run`, a second agent-teams instance,
# another machine, or a registered webhook) Telegram answers every getUpdates
# with 409 Conflict and inbound replies silently stop working. That failure is
# invisible from the dashboard unless we track it, so the listener records its
# own health here and `/setup/gateway/status` surfaces it.
_health: Dict[str, Any] = {
    "state": "idle",          # idle | starting | live | conflict | error | unsupported
    "detail": "Inbound listener is not running.",
    "remedy": "",
    "since": 0.0,
    "conflict_count": 0,
    "last_update_at": 0.0,
}
_health_lock = threading.Lock()


def _set_health(state: str, detail: str, remedy: str = "", **extra: Any) -> None:
    """Record listener state. Resets ``since`` only on a real state transition so
    the dashboard can show how long a problem has persisted."""
    with _health_lock:
        if _health.get("state") != state:
            _health["since"] = time.time()
        _health["state"] = state
        _health["detail"] = detail
        _health["remedy"] = remedy
        for k, v in extra.items():
            _health[k] = v


def get_listener_health() -> Dict[str, Any]:
    """Snapshot of inbound-listener health for the status route / dashboard.

    Two audiences, two vocabularies. ``detail``/``remedy``/``fix_command`` are
    internal — accurate, technical, useful in logs and to us when debugging.
    ``user_message`` is the ONLY field the dashboard should render: one calm
    sentence from a fixed, small vocabulary. A non-technical user should never
    see a pid, a launchctl command, or the word "conflict" — they should see
    "Connected" or "Not connected yet — replies go to your dashboard inbox"."""
    with _health_lock:
        h = dict(_health)
    state = h.get("state", "idle")
    # 'blocked' == replies definitely are NOT reaching agents right now.
    h["blocked"] = state in ("conflict", "error")

    if state == "live":
        h["user_message"] = "Connected — Telegram replies are working."
        h["user_action"] = ""
    elif state == "starting":
        h["user_message"] = "Connecting…"
        h["user_action"] = ""
    elif state == "unsupported":
        h["user_message"] = "Questions are sent to chat; replies happen in the dashboard."
        h["user_action"] = ""
    elif state == "idle":
        h["user_message"] = "Not connected."
        h["user_action"] = "Connect a Telegram bot to get questions in chat."
    else:  # conflict | error — never expose why; the dashboard inbox always works
        h["user_message"] = "Telegram isn't connected right now — use the dashboard inbox for now."
        h["user_action"] = "Reconnecting automatically. If this continues, create a new bot below."
    return h


def _supervisor_for_pid(pid: int) -> Optional[Tuple[str, str]]:
    """Return (manager, label) when ``pid`` is run by a service supervisor.

    This matters for the remedy, not just the diagnosis: a supervised gateway has
    KeepAlive/Restart semantics, so "kill the pid" is useless advice — it comes
    straight back. Detection is label-agnostic (we match on the pid) so it works
    for any user's install, whatever their service is named."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["launchctl", "list"], capture_output=True,
                                 text=True, timeout=5)
            for line in (out.stdout or "").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and parts[0].strip() == str(pid):
                    return "launchd", parts[2].strip()
        elif sys.platform.startswith("linux"):
            cgroup = Path(f"/proc/{pid}/cgroup")
            if cgroup.exists():
                m = re.search(r"([A-Za-z0-9@:_.\-]+\.service)", cgroup.read_text())
                if m:
                    return "systemd", m.group(1)
    except Exception:
        pass
    return None


def _stop_command_for(pid: int, supervisor: Optional[Tuple[str, str]]) -> str:
    """The command that actually makes a competing gateway stay stopped."""
    if supervisor and supervisor[0] == "launchd":
        label = supervisor[1]
        return (f"launchctl bootout gui/{os.getuid()}/{label}"
                if hasattr(os, "getuid") else f"launchctl bootout {label}")
    if supervisor and supervisor[0] == "systemd":
        unit = supervisor[1]
        return f"systemctl --user stop {unit} && systemctl --user disable {unit}"
    return f"kill {pid}"


def _describe_local_conflict() -> Optional[Dict[str, Any]]:
    """Identify a process on THIS machine that is already polling the same bot.

    Hermes writes a structured ``gateway.pid`` into its home whenever
    ``hermes gateway run`` is active. If that process is alive it is by far the
    most common cause of a 409, so naming its pid + argv turns an opaque error
    into a one-line fix. Also reports whether it is supervised (launchd/systemd),
    because that changes the fix from "kill it" to "unload the service".
    Returns None when no local culprit can be identified."""
    try:
        import json as _json
        from teams_server.gateway_config import SHARED_HERMES_HOME

        pid_file = Path(SHARED_HERMES_HOME) / "gateway.pid"
        if not pid_file.exists():
            return None
        info = _json.loads(pid_file.read_text() or "{}") or {}
        pid = int(info.get("pid") or 0)
        if pid <= 0 or pid == os.getpid():
            return None
        alive = True
        try:
            os.kill(pid, 0)          # signal 0 = liveness probe, does not kill
        except ProcessLookupError:
            alive = False            # stale pid file from a crashed run
        except PermissionError:
            alive = True             # running, owned by another user
        except OSError:
            return None
        supervisor = _supervisor_for_pid(pid) if alive else None
        return {
            "pid": pid,
            "alive": alive,
            "kind": info.get("kind") or "hermes-gateway",
            "argv": [str(a) for a in (info.get("argv") or [])],
            "supervisor": supervisor[0] if supervisor else "",
            "service_label": supervisor[1] if supervisor else "",
            "stop_command": _stop_command_for(pid, supervisor) if alive else "",
        }
    except Exception:
        return None


async def _describe_local_conflict_async() -> Optional[Dict[str, Any]]:
    """Thread-offloaded ``_describe_local_conflict``.

    Supervisor detection shells out to launchctl/reads /proc, which blocks. This
    runs on the same event loop that serves the dashboard and agent websockets,
    so it must never block it."""
    try:
        return await asyncio.to_thread(_describe_local_conflict)
    except Exception:
        return None


def _hermes_reported_gateway_pids() -> list:
    """Ask Hermes itself which gateway processes own this Hermes home.

    Hermes' own detector combines the pid file, service-managed pids AND a
    process-table scan, so it still finds a competitor when the pid file is
    stale or missing — cases our own file parse would miss."""
    try:
        from teams_server.gateway_config import SHARED_HERMES_HOME
        from teams_server.model_config import _home

        with _home(SHARED_HERMES_HOME):
            from hermes_cli.gateway import find_gateway_pids

            return [int(p) for p in (find_gateway_pids() or [])
                    if int(p) > 0 and int(p) != os.getpid()]
    except Exception:
        return []


def _foreign_gateway_owner() -> Optional[Dict[str, Any]]:
    """A process OTHER than us that already owns this bot's update stream, or None.

    Prefers Hermes' detector and falls back to the pid file, then enriches the
    result with supervisor info so the caller can give a remedy that sticks."""
    pid_info = _describe_local_conflict()
    for pid in _hermes_reported_gateway_pids():
        if pid_info and pid_info.get("pid") == pid:
            return pid_info                       # already fully described
        sup = _supervisor_for_pid(pid)
        return {
            "pid": pid,
            "alive": True,
            "kind": "hermes-gateway",
            "argv": [],
            "supervisor": sup[0] if sup else "",
            "service_label": sup[1] if sup else "",
            "stop_command": _stop_command_for(pid, sup),
        }
    return pid_info if (pid_info and pid_info.get("alive")) else None


def detect_conflicting_gateway_service() -> Optional[Dict[str, str]]:
    """Find an INSTALLED (not necessarily running) gateway service pointed at the
    swarm's Hermes home.

    This is the preventive check: a launchd/systemd service with RunAtLoad or
    KeepAlive will come back at next login even if the user kills it once, so
    warning only while it happens to be running is too late."""
    try:
        from teams_server.gateway_config import SHARED_HERMES_HOME

        home = str(Path(SHARED_HERMES_HOME).resolve())
        candidates: list[Path] = []
        if sys.platform == "darwin":
            candidates = sorted(Path.home().glob("Library/LaunchAgents/*.plist"))
        elif sys.platform.startswith("linux"):
            candidates = sorted(Path.home().glob(".config/systemd/user/*.service"))
        for path in candidates:
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            if home not in text:
                continue
            if "gateway" not in text.lower():
                continue
            return {
                "manager": "launchd" if sys.platform == "darwin" else "systemd",
                "label": path.stem,
                "path": str(path),
            }
    except Exception:
        pass
    return None


def auto_disable_own_gateway_service() -> Optional[Dict[str, str]]:
    """Silently retire a launchd/systemd service that points at agent-teams' OWN
    managed Hermes home, if one is installed.

    This is what actually eliminates the most common conflict a non-technical
    user hits: a leftover ``hermes gateway`` service was, at some point, pointed
    at agent-teams' data directory (usually from an earlier setup attempt or a
    copy-pasted command). That service and agent-teams' own inbound listener then
    fight over the same bot token forever — the service restarts itself
    (RunAtLoad/KeepAlive), so nothing short of disabling it fixes this for good.

    Scope is intentionally narrow and safe: ``detect_conflicting_gateway_service``
    only matches files whose contents mention agent-teams' OWN Hermes home path.
    A service pointed at the user's main Hermes profile (``~/.hermes``) does not
    contain that path and is never touched — that is a different app the user
    runs on purpose, not a leftover.

    The unit file is archived (moved, not deleted) into
    ``data/.hermes-shared/_retired_services/`` so nothing is unrecoverable.
    Best-effort throughout: any failure is logged and swallowed — worst case the
    user sees the (still accurate) conflict banner, no worse than before."""
    svc = detect_conflicting_gateway_service()
    if not svc:
        return None

    path = Path(svc["path"])
    label = svc["label"]
    manager = svc["manager"]

    try:
        if manager == "launchd":
            uid = os.getuid() if hasattr(os, "getuid") else None
            target = f"gui/{uid}/{label}" if uid is not None else label
            subprocess.run(["launchctl", "bootout", target],
                           capture_output=True, text=True, timeout=10)
        elif manager == "systemd":
            subprocess.run(["systemctl", "--user", "stop", label],
                           capture_output=True, text=True, timeout=10)
            subprocess.run(["systemctl", "--user", "disable", label],
                           capture_output=True, text=True, timeout=10)
    except Exception as e:
        log.warning("[gateway] could not stop leftover service %s (%s): %s",
                    label, manager, e)
        # Still try to archive the file below — even a disabled-but-not-yet-
        # stopped unit won't restart after the file is moved out of the
        # directory launchd/systemd scans on next login.

    try:
        from teams_server.gateway_config import SHARED_HERMES_HOME

        archive_dir = Path(SHARED_HERMES_HOME) / "_retired_services"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / path.name
        if path.exists():
            path.replace(dest)
    except Exception as e:
        log.warning("[gateway] could not archive leftover service file %s: %s",
                    path, e)
        # Stopped (best-effort) even if the file move failed; report what we
        # attempted rather than claiming full success.

    log.info("[gateway] retired a leftover %s service (%s) that was pointed at "
             "agent-teams' own data directory and competing for the same bot — "
             "this should not recur.", manager, label)
    return svc


async def _diagnose_conflict(client, base: str) -> Tuple[str, str]:
    """Explain a 409 Conflict as (detail, remedy) in user-actionable terms.

    Two distinct causes, with different fixes: a registered webhook (long polling
    can NEVER work until it is removed) versus a competing poller."""
    webhook_url = ""
    try:
        wr = await client.get(f"{base}/getWebhookInfo", timeout=10.0)
        webhook_url = ((wr.json() or {}).get("result") or {}).get("url", "").strip()
    except Exception:
        pass

    if webhook_url:
        return (
            f"This bot has a webhook registered ({webhook_url}), which prevents "
            "agent-teams from receiving replies by long polling.",
            "Re-save the bot in Gateway settings with “Remove webhook” checked, or "
            "use a bot that has no webhook configured.",
        )

    local = await _describe_local_conflict_async()
    if local and local.get("alive"):
        argv = " ".join(local.get("argv") or [])
        where = f"pid {local['pid']}" + (f" — {argv}" if argv else "")
        stop = local.get("stop_command") or f"kill {local['pid']}"
        if local.get("supervisor"):
            # Supervised: KeepAlive/Restart brings it straight back after a kill,
            # so the only remedy that sticks is unloading the service.
            return (
                f"A {local['supervisor']}-managed Hermes gateway ("
                f"{local.get('service_label') or 'service'}, {where}) is already "
                "receiving updates for this bot, and it is configured to restart "
                "automatically — so killing it will not help.",
                f"Stop and disable the service: {stop}. It points at the same Hermes "
                "home as agent-teams, so both are polling one bot token — only one "
                "listener is allowed. Either disable the service and let agent-teams "
                "own replies, or give agent-teams its own dedicated bot token.",
            )
        return (
            f"Another process on this machine is already receiving updates for this "
            f"bot ({where}).",
            f"Stop that process ({stop}), then replies will start working. "
            "Alternatively give agent-teams its own dedicated bot token — a bot token "
            "allows only one listener at a time.",
        )

    return (
        "Another client is already receiving updates for this bot token.",
        "Check for another app, container, or machine polling this same bot (including "
        "a second copy of agent-teams). A bot token allows only one listener at a time, "
        "so give agent-teams a dedicated bot if you need both.",
    )


async def delete_telegram_webhook(token: str) -> Tuple[bool, str]:
    """Remove a webhook so long polling can work. Requires explicit user consent
    (the Gateway settings checkbox) because the bot may be shared with another app."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{token}/deleteWebhook",
                params={"drop_pending_updates": "false"},
            )
        data = r.json() or {}
        if data.get("ok"):
            return True, ""
        return False, str(data.get("description") or "deleteWebhook failed")
    except Exception as e:
        return False, str(e)


async def discover_home_chat(token: str) -> Optional[Dict[str, Any]]:
    """If someone has already messaged this bot, return the sender's chat id
    (+ a friendly name) so the user never has to find or type a numeric chat ID.

    This is what lets setup be "paste the token, message the bot, done" instead
    of a scavenger hunt through @userinfobot. Looks at whatever Telegram still
    has buffered for this token; returns None if nobody has messaged it yet."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"https://api.telegram.org/bot{token}/getUpdates",
                                 params={"timeout": 0, "limit": 100})
        data = r.json() or {}
        if not data.get("ok"):
            return None
        updates = data.get("result") or []
        for upd in reversed(updates):          # most recent sender wins
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue
            name = (chat.get("first_name") or chat.get("title")
                    or chat.get("username") or "your chat")
            return {"chat_id": str(chat_id), "name": str(name)}
    except Exception:
        pass
    return None


async def preflight_telegram(token: str) -> Dict[str, Any]:
    """Validate a bot token BEFORE saving it, so a typo or a conflicting bot is
    caught at setup time instead of silently breaking replies later.

    Returns {ok, bot_username, webhook_url, error}."""
    import httpx

    out: Dict[str, Any] = {"ok": False, "bot_username": "", "webhook_url": "", "error": ""}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            me = await client.get(f"https://api.telegram.org/bot{token}/getMe")
            data = me.json() or {}
            if not data.get("ok"):
                desc = str(data.get("description") or "")
                out["error"] = (
                    "Telegram rejected this bot token"
                    + (f": {desc}" if desc else ". Check for a typo or get a fresh token from @BotFather.")
                )
                return out
            out["bot_username"] = (data.get("result") or {}).get("username", "")
            try:
                wr = await client.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
                out["webhook_url"] = ((wr.json() or {}).get("result") or {}).get("url", "").strip()
            except Exception:
                pass
        out["ok"] = True
        return out
    except Exception as e:
        # Network failure shouldn't hard-block setup (user may be offline while
        # configuring); report it and let the caller decide.
        out["error"] = f"Could not reach Telegram to verify the token: {e}"
        return out


def _newest_pending_global() -> Optional[Tuple[str, str]]:
    """The newest pending question across ALL agents → (qid, agent_name), or None.
    The fallback when a chat message isn't a reply to a tracked send (mirrors the
    dashboard respond route's newest-pending behavior, but team-wide)."""
    from teams_server.tools import _pending_human_questions, _pending_lock
    with _pending_lock:
        newest = None
        for qid, q in _pending_human_questions.items():
            if q["status"] == "pending":
                if newest is None or q["timestamp"] > newest[2]:
                    newest = (qid, q["agent_name"], q["timestamp"])
        return (newest[0], newest[1]) if newest else None


async def _deliver_reply(text: str, reply_to_message_id: Optional[str]) -> bool:
    """Route a chat reply to the matching pending question via the SAME path the
    dashboard uses. Returns True if it was delivered to an agent."""
    from teams_server.tools import _daemon_registry

    match = _lookup_sent(reply_to_message_id) if reply_to_message_id else None
    if match is None:
        match = _newest_pending_global()
    if match is None:
        log.info("[gateway] reply with no pending question to match — ignored")
        return False
    qid, agent_name = match
    daemon = _daemon_registry.get(agent_name)
    if daemon is None:
        log.warning("[gateway] reply for %s but agent not registered", agent_name)
        return False
    # _finalize_human_answer lives in server.py (atomic deliver + resume-task +
    # browser hand-back). Import locally to avoid a circular import at module load.
    from teams_server.server import _finalize_human_answer

    result = await _finalize_human_answer(daemon, qid, text)
    if result.get("ok"):
        log.info("[gateway] delivered chat reply to %s (qid=%s, %s)",
                 agent_name, qid, result.get("delivery"))
        return True
    log.warning("[gateway] chat reply delivery failed: %s", result.get("error"))
    return False


async def _poll_telegram(token: str, home_chat_id: str,
                         max_conflicts: int = 3) -> Optional[str]:
    """Long-poll Telegram getUpdates and deliver home-channel replies. Strictly
    drops messages from any chat other than ``home_chat_id`` so a stranger can't
    answer an agent. Runs until cancelled.

    Returns a reason string when it stops on purpose: ``"token"`` (credential is
    rejected, retrying is pointless) or ``"conflict"`` (another process owns the
    update stream, so the supervisor should park rather than keep competing).

    Failure handling is deliberate: a 409 Conflict means someone else owns this
    bot's update stream, which is a configuration problem that will not fix
    itself. We diagnose it, report it once (not once per retry), and hand back to
    the supervisor instead of racing the other listener for updates."""
    import httpx

    base = f"https://api.telegram.org/bot{token}"
    offset: Optional[int] = None
    backoff = 5.0
    BACKOFF_MAX = 60.0
    conflicts = 0
    # Competing pollers can each intermittently win an update, which means a
    # human's reply may be consumed by the other process and never reach the
    # waiting agent. After a few conflicts we stop and let the supervisor park.
    MAX_CONFLICTS_BEFORE_PARK = max(1, max_conflicts)

    _set_health("starting", "Connecting to Telegram…")

    async with httpx.AsyncClient(timeout=40.0) as client:
        # Drain any backlog WITHOUT processing it, so a restart doesn't replay old
        # chat messages as answers to (possibly stale) questions.
        try:
            r = await client.get(f"{base}/getUpdates", params={"timeout": 0})
            drained = r.json() or {}
            updates = drained.get("result") or []
            if updates:
                offset = updates[-1]["update_id"] + 1
            log.info("[gateway] telegram listener started (drained %d backlog update(s))",
                     len(updates))
        except Exception as e:
            log.warning("[gateway] telegram drain failed (%s); starting fresh", e)

        while True:
            try:
                params = {"timeout": 30}
                if offset is not None:
                    params["offset"] = offset
                r = await client.get(f"{base}/getUpdates", params=params)
                data = r.json() or {}

                if not data.get("ok"):
                    code = data.get("error_code")
                    desc = str(data.get("description") or "")

                    if code == 409:
                        conflicts += 1
                        detail, remedy = await _diagnose_conflict(client, base)
                        loc = await _describe_local_conflict_async() or {}
                        _set_health("conflict", detail, remedy,
                                    conflict_count=conflicts,
                                    fix_command=loc.get("stop_command", ""))
                        # Report fully the first time and then only occasionally:
                        # the condition persists, so per-retry logging is spam.
                        if conflicts == 1 or conflicts % 20 == 0:
                            log.warning(
                                "[gateway] telegram replies are NOT being received — %s "
                                "FIX: %s (retrying every %ds; attempt %d)",
                                detail, remedy, int(backoff), conflicts)
                        else:
                            log.debug("[gateway] telegram still conflicted (attempt %d)", conflicts)
                        if conflicts >= MAX_CONFLICTS_BEFORE_PARK:
                            return "conflict"

                    elif code in (401, 404):
                        # Revoked or malformed token: retrying cannot help.
                        _set_health(
                            "error",
                            f"Telegram rejected the bot token ({code}"
                            + (f": {desc}" if desc else "") + ").",
                            "Open Gateway settings and paste a valid token from @BotFather. "
                            "If you revoked the old token, generate a new one.",
                        )
                        log.error("[gateway] telegram token rejected (%s) — inbound replies "
                                  "disabled until a valid token is saved in Gateway settings", code)
                        return "token"

                    else:
                        _set_health("error", f"Telegram error {code}"
                                    + (f": {desc}" if desc else "") + ".",
                                    "Usually transient; the listener keeps retrying.")
                        log.warning("[gateway] telegram getUpdates not ok (code=%s): %s", code, desc)

                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX)
                    continue

                # Success: announce recovery if we were previously broken.
                if conflicts or get_listener_health().get("state") != "live":
                    if conflicts:
                        log.info("[gateway] telegram conflict cleared — replies are being "
                                 "received again (after %d failed attempt(s))", conflicts)
                    _set_health("live", "Receiving replies from your chat.", "",
                                conflict_count=0)
                conflicts = 0
                backoff = 5.0

                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue
                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue
                    chat_id = str((msg.get("chat") or {}).get("id", ""))
                    # Security: only the configured home channel may answer.
                    if chat_id != str(home_chat_id):
                        log.info("[gateway] dropping telegram msg from non-home chat %s", chat_id)
                        continue
                    with _health_lock:
                        _health["last_update_at"] = time.time()
                    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
                    await _deliver_reply(text, str(reply_to) if reply_to is not None else None)
            except asyncio.CancelledError:
                _set_health("idle", "Inbound listener stopped.")
                raise
            except Exception as e:
                _set_health("error", f"Connection problem: {e}",
                            "Usually transient; the listener keeps retrying.")
                log.warning("[gateway] telegram poll error (retrying in %ds): %s",
                            int(backoff), e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)


# How often to re-check whether a competing gateway has gone away. Keeps the
# fix self-service: stop the other gateway and replies resume without a restart.
_OWNERSHIP_RECHECK_SECONDS = 30.0


def _same_bot_in_other_hermes_profile() -> Optional[str]:
    """Label of another Hermes profile configured with the SAME bot token, or None.

    Attribution for the cross-profile case: the operator's own ``~/.hermes``
    gateway is a different profile, so the profile-scoped pid lookup can't see it.
    Comparing tokens is precise — a gateway on another profile using a *different*
    bot is not a conflict at all, so we must not blame it."""
    try:
        from teams_server.gateway_config import detect_global_gateway, get_active_gateway

        ours = get_active_gateway() or {}
        theirs = detect_global_gateway() or {}
        if ours.get("token") and ours["token"] == theirs.get("token"):
            return str(theirs.get("label") or theirs.get("key") or "your main Hermes")
    except Exception:
        pass
    return None


async def _supervise_inbound(token: str, home_chat_id: str) -> None:
    """Own the inbound listener, but never poll while another process owns this
    bot's update stream.

    This is the prevention layer. Telegram hands each update to whichever poller
    asks first, so two listeners on one token don't just produce 409s — they can
    silently split the updates, and a human's reply may be consumed by the other
    process and never reach the agent that is blocked waiting for it. Refusing to
    poll is strictly safer than competing: the dashboard inbox still works, and
    the state is reported instead of losing answers."""
    parked_pid: Optional[int] = None
    unattributed = 0            # consecutive conflicts we could not attribute
    while True:
        owner = await asyncio.to_thread(_foreign_gateway_owner)
        if owner and owner.get("alive"):
            stop = owner.get("stop_command") or f"kill {owner['pid']}"
            sup = owner.get("supervisor")
            detail = (
                f"Another Hermes gateway (pid {owner['pid']}"
                + (f", {sup} service {owner.get('service_label')}" if sup else "")
                + ") is already receiving updates for this bot, so agent-teams is "
                "not polling — running both would let it consume replies meant "
                "for your agents."
            )
            remedy = (
                f"Stop it with: {stop}"
                + (" (it restarts automatically otherwise)" if sup else "")
                + ". Replies resume within a minute — no restart needed. Or give "
                "agent-teams its own dedicated bot token."
            )
            _set_health("conflict", detail, remedy, fix_command=stop)
            if parked_pid != owner["pid"]:
                parked_pid = owner["pid"]
                log.warning("[gateway] NOT starting the reply listener — %s FIX: %s",
                            detail, remedy)
            unattributed = 0
            await asyncio.sleep(_OWNERSHIP_RECHECK_SECONDS)
            continue

        if parked_pid is not None:
            log.info("[gateway] competing gateway is gone — starting the reply listener")
            parked_pid = None

        # A 409 is definitive proof that something owns this bot's update stream,
        # even when we cannot point at a local process (another Hermes profile,
        # another machine, a container, a second agent-teams). Park anyway rather
        # than keep competing — otherwise replies can be silently swallowed.
        if unattributed:
            shared_with = _same_bot_in_other_hermes_profile()
            if shared_with:
                detail = (f"This bot token is also configured in {shared_with}, whose "
                          "gateway is receiving the updates — so agent-teams is not polling.")
                remedy = ("Give agent-teams its own bot from @BotFather (one bot per "
                          "listener; separate chats/channels do NOT help — the limit is "
                          "per bot token).")
            else:
                detail = ("Something else is receiving updates for this bot token, so "
                          "agent-teams is not polling — competing would let it swallow "
                          "replies meant for your agents.")
                remedy = ("Find the other listener (another machine, container, Hermes "
                          "profile, or a second copy of agent-teams) and stop it, or give "
                          "agent-teams its own dedicated bot token.")
            _set_health("conflict", detail, remedy, fix_command="")
            if unattributed == 1:
                log.warning("[gateway] NOT polling — %s FIX: %s", detail, remedy)
            await asyncio.sleep(_OWNERSHIP_RECHECK_SECONDS)

        # While parked, probe with a single request instead of a full retry cycle,
        # so re-checking costs one call rather than sustained competition.
        reason = await _poll_telegram(token, home_chat_id,
                                     max_conflicts=1 if unattributed else 3)
        if reason == "token":
            return                      # credential problem; health already set
        if reason == "conflict":
            unattributed += 1
            continue                    # re-evaluate ownership, then park
        return


# Holds the running listener task so setup can cancel + restart it.
_listener_task: Optional[asyncio.Task] = None


def start_inbox_gateway_listener(app=None) -> None:
    """Start (or restart) the inbound listener for the configured gateway. No-op
    unless a reply-capable gateway (Telegram today) is configured WITH a home
    channel. Safe to call repeatedly — cancels any previous task first.

    Called from the FastAPI lifespan startup and from POST /setup/gateway."""
    global _listener_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.warning("[gateway] no running loop — listener not started")
        return

    # Cancel a previous listener (e.g. the operator just changed the bot).
    if _listener_task is not None and not _listener_task.done():
        _listener_task.cancel()
        _listener_task = None

    try:
        from teams_server.gateway_config import get_active_gateway
        gw = get_active_gateway()
    except Exception as e:
        log.warning("[gateway] could not read active gateway: %s", e)
        return

    if not gw:
        log.info("[gateway] no gateway configured — inbound listener idle")
        _set_health("idle", "No messaging gateway configured.",
                    "Add a bot in Gateway settings to get questions in chat.")
        return
    if gw["key"] != "telegram":
        # Outbound still works for this platform; we just can't receive replies yet.
        log.info("[gateway] inbound replies not supported for '%s' yet "
                 "(outbound notifications still active)", gw["key"])
        _set_health("unsupported",
                    f"Replies from chat aren't supported for {gw.get('label') or gw['key']} yet.",
                    "Questions are still sent to chat — answer them in the dashboard inbox.")
        return
    if not gw.get("home_chat_id"):
        log.warning("[gateway] telegram configured without a home channel — "
                    "set one to receive replies (outbound still works)")
        _set_health("error", "No home channel is set, so replies can't be matched.",
                    "Open Gateway settings and set the home channel (chat) ID.")
        return

    # A leftover service pointed at agent-teams' OWN data directory is the most
    # common cause of the conflict a user would otherwise have to debug by hand
    # (RunAtLoad/KeepAlive means it always comes back). Retire it automatically —
    # this never touches a service pointed at the user's own ~/.hermes profile.
    auto_disable_own_gateway_service()

    # The supervisor parks instead of competing when another gateway owns the bot.
    _listener_task = loop.create_task(
        _supervise_inbound(gw["token"], gw["home_chat_id"])
    )
    log.info("[gateway] inbound telegram reply listener started")


def stop_inbox_gateway_listener() -> None:
    """Cancel the inbound listener (used on shutdown)."""
    global _listener_task
    if _listener_task is not None and not _listener_task.done():
        _listener_task.cancel()
    _listener_task = None
    _set_health("idle", "Inbound listener stopped.")
