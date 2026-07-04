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
import threading
from typing import Dict, Optional, Tuple

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


async def _poll_telegram(token: str, home_chat_id: str) -> None:
    """Long-poll Telegram getUpdates and deliver home-channel replies. Strictly
    drops messages from any chat other than ``home_chat_id`` so a stranger can't
    answer an agent. Runs until cancelled."""
    import httpx

    base = f"https://api.telegram.org/bot{token}"
    offset: Optional[int] = None

    async with httpx.AsyncClient(timeout=40.0) as client:
        # Drain any backlog WITHOUT processing it, so a restart doesn't replay old
        # chat messages as answers to (possibly stale) questions.
        try:
            r = await client.get(f"{base}/getUpdates", params={"timeout": 0})
            updates = (r.json() or {}).get("result", [])
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
                    log.warning("[gateway] telegram getUpdates not ok: %s", data)
                    await asyncio.sleep(5)
                    continue
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
                    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
                    await _deliver_reply(text, str(reply_to) if reply_to is not None else None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("[gateway] telegram poll error (retrying): %s", e)
                await asyncio.sleep(5)


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
        return
    if gw["key"] != "telegram":
        # Outbound still works for this platform; we just can't receive replies yet.
        log.info("[gateway] inbound replies not supported for '%s' yet "
                 "(outbound notifications still active)", gw["key"])
        return
    if not gw.get("home_chat_id"):
        log.warning("[gateway] telegram configured without a home channel — "
                    "set one to receive replies (outbound still works)")
        return

    _listener_task = loop.create_task(
        _poll_telegram(gw["token"], gw["home_chat_id"])
    )
    log.info("[gateway] inbound telegram reply listener started")


def stop_inbox_gateway_listener() -> None:
    """Cancel the inbound listener (used on shutdown)."""
    global _listener_task
    if _listener_task is not None and not _listener_task.done():
        _listener_task.cancel()
    _listener_task = None
