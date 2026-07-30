"""Tests for the human-inbox → messaging-gateway bridge.

Covers the config-reuse layer (teams_server.gateway_config — set/get/detect/adopt
against temp Hermes homes, the same SHARED/GLOBAL_HERMES_HOME pattern the model
setup uses), the setup routes, and the runtime bridge's reply-matching
(teams_server.gateway_bridge — exact reply-to match + newest-pending fallback).

Uses temp Hermes homes (no real ~/.hermes) and stubs the one-shot send + the
server-side answer finalizer; no network, no live agents.

Run:  pytest tests/test_gateway.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import teams_server.gateway_config as gc        # noqa: E402
import teams_server.gateway_bridge as gb         # noqa: E402


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    """Point both Hermes homes AND the gateway store at temp paths so writes never
    touch a real config or the operator's live bot credentials."""
    shared = tmp_path / "shared"
    glob = tmp_path / "global"
    shared.mkdir()
    glob.mkdir()
    monkeypatch.setattr(gc, "SHARED_HERMES_HOME", shared)
    monkeypatch.setattr(gc, "GLOBAL_HERMES_HOME", glob)
    # The swarm's creds now live in agent-teams' own store, not a Hermes .env.
    monkeypatch.setattr(gc, "GATEWAY_STORE", tmp_path / "gateway.json")
    # Keep the process environment clean between tests (apply_gateway_env writes it).
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL",
                "DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"):
        monkeypatch.delenv(var, raising=False)
    return shared, glob


# --------------------------------------------------------------------------- #
# Config reuse: set / get / detect / adopt
# --------------------------------------------------------------------------- #
def test_unconfigured_by_default(homes):
    assert gc.get_active_gateway() is None
    assert gc.is_gateway_configured() is False


def test_set_and_get_roundtrip(homes):
    shared, _ = homes
    gc.set_gateway("telegram", "123:TOKEN", "987654321")

    active = gc.get_active_gateway()
    assert active is not None
    assert active["key"] == "telegram"
    assert active["home_chat_id"] == "987654321"
    assert active["token"] == "123:TOKEN"


def test_creds_never_written_into_the_hermes_home(homes):
    """The core prevention property: a bot token inside a Hermes home means any
    `hermes gateway run` pointed at that home starts a rival listener for the same
    bot. So the token must NOT be there."""
    shared, _ = homes
    gc.set_gateway("telegram", "123:SECRET", "555")

    env_text = (shared / ".env").read_text() if (shared / ".env").exists() else ""
    assert "123:SECRET" not in env_text
    assert "TELEGRAM_BOT_TOKEN" not in env_text
    # It lives in agent-teams' own store instead.
    assert "123:SECRET" in gc.GATEWAY_STORE.read_text()


def test_store_is_owner_only_readable(homes):
    """The file holds a live bot token; group/world readability would leak it."""
    import stat
    gc.set_gateway("telegram", "123:TOKEN", "5")
    mode = stat.S_IMODE(gc.GATEWAY_STORE.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_set_gateway_populates_process_env_for_outbound(homes):
    """Outbound sending must keep working: gateway.config reads the token from
    os.getenv, so the store has to be applied to this process's environment."""
    import os
    gc.set_gateway("telegram", "123:TOKEN", "555")
    assert os.environ.get("TELEGRAM_BOT_TOKEN") == "123:TOKEN"
    assert os.environ.get("TELEGRAM_HOME_CHANNEL") == "555"


def test_migration_moves_creds_out_of_hermes_env(homes):
    """Existing installs are silently vulnerable until the token is removed from
    the Hermes .env — migration must both copy it out AND delete it."""
    import os
    shared, _ = homes
    (shared / ".env").write_text(
        'TELEGRAM_BOT_TOKEN="legacy-tok"\nTELEGRAM_HOME_CHANNEL="42"\nOTHER=keepme\n')

    assert gc.migrate_gateway_out_of_hermes_home() is True

    # Moved into our store, with the gateway still fully working.
    active = gc.get_active_gateway()
    assert active["token"] == "legacy-tok" and active["home_chat_id"] == "42"
    # Gone from the Hermes .env, so an external gateway can't find it...
    env_text = (shared / ".env").read_text()
    assert "legacy-tok" not in env_text
    assert "TELEGRAM_BOT_TOKEN" not in env_text
    # ...but unrelated secrets in that .env are preserved.
    assert "OTHER=keepme" in env_text
    # ...and outbound sending still has what it needs.
    assert os.environ.get("TELEGRAM_BOT_TOKEN") == "legacy-tok"


def test_migration_is_a_noop_when_nothing_to_move(homes):
    assert gc.migrate_gateway_out_of_hermes_home() is False


def test_migration_does_not_clobber_an_existing_store(homes):
    shared, _ = homes
    gc.set_gateway("telegram", "current-tok", "9")
    (shared / ".env").write_text('TELEGRAM_BOT_TOKEN="stale-tok"\n')

    gc.migrate_gateway_out_of_hermes_home()
    assert gc.get_active_gateway()["token"] == "current-tok"
    assert "stale-tok" not in (shared / ".env").read_text()


def test_legacy_env_still_readable_before_migration(homes):
    """An install that hasn't restarted yet must keep working."""
    shared, _ = homes
    (shared / ".env").write_text('TELEGRAM_BOT_TOKEN="not-yet-migrated"\n')
    active = gc.get_active_gateway()
    assert active is not None and active["token"] == "not-yet-migrated"


def test_apply_gateway_env_noop_when_unconfigured(homes):
    import os
    assert gc.apply_gateway_env() is False
    assert "TELEGRAM_BOT_TOKEN" not in os.environ


def test_set_gateway_stores_display_names(homes):
    """bot_username/home_chat_name are display-only context, never used for auth
    or reply matching — but persisted so the UI can show what it's connected to."""
    gc.set_gateway("telegram", "tok", "555", bot_username="my_bot", home_chat_name="Ada")
    active = gc.get_active_gateway()
    assert active["bot_username"] == "my_bot"
    assert active["home_chat_name"] == "Ada"


def test_clear_gateway_removes_store_and_env(homes):
    import os
    gc.set_gateway("telegram", "tok", "555")
    assert gc.get_active_gateway() is not None
    assert os.environ.get("TELEGRAM_BOT_TOKEN") == "tok"

    gc.clear_gateway()
    assert gc.get_active_gateway() is None
    assert not gc.GATEWAY_STORE.exists()
    assert "TELEGRAM_BOT_TOKEN" not in os.environ


def test_clear_gateway_noop_when_already_unconfigured(homes):
    gc.clear_gateway()  # must not raise
    assert gc.get_active_gateway() is None


def test_detect_global_gateway_is_read_only(homes):
    """The operator's bot is detected only to warn them; it must never be copied
    into agent-teams because one bot token cannot serve two listeners."""
    _, glob = homes
    (glob / ".env").write_text('DISCORD_BOT_TOKEN="d-tok"\nDISCORD_HOME_CHANNEL="42"\n')

    det = gc.detect_global_gateway()
    assert det is not None and det["key"] == "discord" and det["home_chat_id"] == "42"
    assert gc.get_active_gateway() is None
    assert not hasattr(gc, "adopt_global_gateway")


def test_list_gateways_includes_telegram_marked_reply(homes):
    cat = gc.list_gateways()
    by_key = {g["key"]: g for g in cat}
    assert "telegram" in by_key
    assert by_key["telegram"]["reply_supported"] is True
    assert by_key["telegram"]["token_var"] == "TELEGRAM_BOT_TOKEN"
    # A send-only platform is present but not reply-capable.
    assert by_key.get("slack", {}).get("reply_supported") is False


# --------------------------------------------------------------------------- #
# Setup routes (mirror /setup/model)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(homes, monkeypatch):
    from fastapi.testclient import TestClient
    import teams_server.server as server_mod
    # The POST route (re)starts the listener; make that a no-op in tests.
    monkeypatch.setattr(gb, "start_inbox_gateway_listener", lambda app=None: None)

    # The POST route now verifies the token with Telegram before saving. Stub it so
    # the suite stays offline and deterministic (a real call would make these tests
    # depend on network availability).
    async def ok_preflight(token):
        return {"ok": True, "bot_username": "swarm_bot", "webhook_url": "", "error": ""}

    monkeypatch.setattr(gb, "preflight_telegram", ok_preflight)

    # The route auto-discovers the home chat when the caller doesn't supply one
    # (the "message the bot, we'll find you" flow) — pretend someone already
    # messaged the bot so tests that omit home_chat_id still succeed by default.
    async def ok_discover(token):
        return {"chat_id": "999", "name": "Test User"}

    monkeypatch.setattr(gb, "discover_home_chat", ok_discover)
    return TestClient(server_mod.app)


def test_route_status_reflects_config(client, homes):
    r = client.get("/setup/gateway/status")
    assert r.status_code == 200 and r.json()["configured"] is False

    client.post("/setup/gateway", json={"key": "telegram", "token": "t", "home_chat_id": "55"})
    d = client.get("/setup/gateway/status").json()
    assert d["configured"] is True
    assert d["active"]["key"] == "telegram"
    assert d["active"]["home_chat_id"] == "55"
    assert d["active"]["reply_supported"] is True
    # The status payload must never leak the bot token.
    assert "token" not in d["active"]


def test_route_requires_token(client):
    r = client.post("/setup/gateway", json={"key": "telegram", "token": ""})
    assert r.status_code == 400


def test_route_status_and_response_expose_bot_and_chat_names(client, homes):
    """A bare 'Connected' isn't enough — the user should see WHICH bot and
    WHICH chat, without exposing the token or a raw numeric chat id."""
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "t"})
    assert r.status_code == 200
    body = r.json()
    assert body["bot_username"] == "swarm_bot"
    assert body["home_chat_name"] == "Test User"

    d = client.get("/setup/gateway/status").json()
    assert d["active"]["bot_username"] == "swarm_bot"
    assert d["active"]["home_chat_name"] == "Test User"
    assert "token" not in d["active"]


def test_route_status_flags_main_hermes_only_when_present(client, homes):
    _, glob = homes
    d = client.get("/setup/gateway/status").json()
    assert d["has_main_hermes"] is False

    (glob / ".env").write_text('TELEGRAM_BOT_TOKEN="their-tok"\nTELEGRAM_HOME_CHANNEL="1"\n')
    d = client.get("/setup/gateway/status").json()
    assert d["has_main_hermes"] is True
    # Still never exposes the other app's token or any detail beyond the flag.
    assert "their-tok" not in str(d)


def test_route_disconnect_clears_config_and_stops_listener(client, homes, monkeypatch):
    client.post("/setup/gateway", json={"key": "telegram", "token": "t"})
    assert gc.get_active_gateway() is not None

    stopped = {"n": 0}
    monkeypatch.setattr(gb, "stop_inbox_gateway_listener", lambda: stopped.__setitem__("n", stopped["n"] + 1))

    r = client.delete("/setup/gateway")
    assert r.status_code == 200
    assert stopped["n"] == 1
    assert gc.get_active_gateway() is None

    d = client.get("/setup/gateway/status").json()
    assert d["configured"] is False


def test_route_rejects_removed_adoption_path(client, homes):
    """Older dashboards and hand-crafted requests cannot re-enable the unsafe
    main-Hermes-token adoption path after the UI button is removed."""
    r = client.post("/setup/gateway", json={"adopt_detected": True})
    assert r.status_code == 400
    assert r.json()["reason"] == "gateway_adoption_removed"
    assert gc.get_active_gateway() is None


def test_route_lists_catalogue(client):
    gws = client.get("/gateways").json()["gateways"]
    assert any(g["key"] == "telegram" for g in gws)


# --------------------------------------------------------------------------- #
# Setup-time validation: catch a bad/conflicting bot when the token is pasted,
# instead of letting replies fail silently at runtime.
# --------------------------------------------------------------------------- #
def test_route_rejects_token_telegram_refuses(client, monkeypatch):
    async def bad_preflight(token):
        return {"ok": False, "bot_username": "", "webhook_url": "",
                "error": "Telegram rejected this bot token: Unauthorized"}

    monkeypatch.setattr(gb, "preflight_telegram", bad_preflight)
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "typo"})
    assert r.status_code == 400
    # Plain-language error — no technical detail, safe for a non-technical user.
    assert "valid bot token" in r.json()["error"]
    # Nothing should have been persisted.
    assert gc.get_active_gateway() is None


def test_route_saves_despite_unreachable_telegram(client, monkeypatch):
    """Being offline while configuring must not block setup — warn and save."""
    async def offline_preflight(token):
        return {"ok": False, "bot_username": "", "webhook_url": "",
                "error": "Could not reach Telegram to verify the token: timeout"}

    monkeypatch.setattr(gb, "preflight_telegram", offline_preflight)
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "tok", "home_chat_id": "9"})
    assert r.status_code == 200
    assert "warning" in r.json()
    assert gc.get_active_gateway()["key"] == "telegram"


def test_route_blocks_webhook_bot_without_consent(client, monkeypatch):
    """A registered webhook makes long polling impossible, so replies would never
    arrive. Refuse with an actionable 409 rather than saving a broken config."""
    async def webhook_preflight(token):
        return {"ok": True, "bot_username": "b", "webhook_url": "https://ex.com/hook", "error": ""}

    monkeypatch.setattr(gb, "preflight_telegram", webhook_preflight)
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "tok"})
    assert r.status_code == 409
    body = r.json()
    assert body["needs_webhook_removal"] is True
    assert body["webhook_url"] == "https://ex.com/hook"
    assert gc.get_active_gateway() is None


def test_route_removes_webhook_with_consent(client, monkeypatch):
    async def webhook_preflight(token):
        return {"ok": True, "bot_username": "b", "webhook_url": "https://ex.com/hook", "error": ""}

    deleted = {}

    async def fake_delete(token):
        deleted["token"] = token
        return True, ""

    monkeypatch.setattr(gb, "preflight_telegram", webhook_preflight)
    monkeypatch.setattr(gb, "delete_telegram_webhook", fake_delete)
    r = client.post("/setup/gateway", json={
        "key": "telegram", "token": "tok", "home_chat_id": "9", "delete_webhook": True})
    assert r.status_code == 200
    assert deleted["token"] == "tok"
    assert gc.get_active_gateway()["key"] == "telegram"


def test_status_exposes_inbound_health(client):
    d = client.get("/setup/gateway/status").json()
    assert "inbound" in d
    assert "state" in d["inbound"] and "blocked" in d["inbound"]


# --------------------------------------------------------------------------- #
# 409 Conflict handling: a bot token allows only one long-polling listener, so
# the listener must explain the clash instead of retrying silently forever.
# --------------------------------------------------------------------------- #
def test_health_reports_conflict_state():
    gb._set_health("conflict", "Someone else is polling.", "Stop it.", conflict_count=3)
    h = gb.get_listener_health()
    assert h["state"] == "conflict"
    assert h["blocked"] is True
    assert h["conflict_count"] == 3
    gb._set_health("live", "ok")
    assert gb.get_listener_health()["blocked"] is False


def test_describe_local_conflict_finds_live_hermes_gateway(homes, monkeypatch):
    """Hermes writes gateway.pid into its home; a live pid there is the usual
    culprit, and naming it turns an opaque 409 into a one-line fix."""
    import json
    import os
    shared, _ = homes
    (shared / "gateway.pid").write_text(json.dumps({
        "pid": os.getpid() + 1, "kind": "hermes-gateway",
        "argv": ["hermes_cli/main.py", "gateway", "run", "--replace"],
    }))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)   # pretend it's alive

    info = gb._describe_local_conflict()
    assert info is not None
    assert info["alive"] is True
    assert "gateway" in " ".join(info["argv"])


def test_describe_local_conflict_ignores_dead_pid(homes, monkeypatch):
    import json
    import os
    shared, _ = homes
    (shared / "gateway.pid").write_text(json.dumps({"pid": os.getpid() + 1}))

    def dead(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "kill", dead)
    info = gb._describe_local_conflict()
    assert info is not None and info["alive"] is False


def test_describe_local_conflict_none_without_pidfile(homes):
    assert gb._describe_local_conflict() is None


# --------------------------------------------------------------------------- #
# Supervised competitors: a launchd/systemd-managed gateway restarts itself, so
# "kill the pid" is wrong advice — the remedy has to unload the service.
# --------------------------------------------------------------------------- #
def test_stop_command_for_launchd_unloads_service():
    cmd = gb._stop_command_for(4242, ("launchd", "ai.hermes.gateway"))
    assert "launchctl bootout" in cmd
    assert "ai.hermes.gateway" in cmd
    assert "kill" not in cmd


def test_stop_command_for_systemd_disables_unit():
    cmd = gb._stop_command_for(4242, ("systemd", "hermes-gateway.service"))
    assert "systemctl --user stop hermes-gateway.service" in cmd
    assert "disable" in cmd


def test_stop_command_for_plain_process_is_kill():
    assert gb._stop_command_for(4242, None) == "kill 4242"


def test_diagnose_conflict_supervised_says_kill_wont_help(homes, monkeypatch):
    """The failure mode that made the original bug so confusing: the process is
    restarted automatically, so the obvious fix silently doesn't work."""
    class FakeResp:
        @staticmethod
        def json():
            return {"ok": True, "result": {"url": ""}}

    class FakeClient:
        async def get(self, url, **kw):
            return FakeResp()

    monkeypatch.setattr(gb, "_describe_local_conflict", lambda: {
        "pid": 3616, "alive": True, "kind": "hermes-gateway",
        "argv": ["hermes_cli/main.py", "gateway", "run", "--replace"],
        "supervisor": "launchd", "service_label": "ai.hermes.gateway",
        "stop_command": "launchctl bootout gui/501/ai.hermes.gateway",
    })

    detail, remedy = asyncio.run(gb._diagnose_conflict(FakeClient(), "https://api.telegram.org/botX"))
    assert "launchd" in detail
    assert "restart automatically" in detail        # explains why kill fails
    assert "launchctl bootout" in remedy
    assert "kill" not in remedy


def test_describe_local_conflict_reports_supervisor(homes, monkeypatch):
    import json
    import os
    shared, _ = homes
    (shared / "gateway.pid").write_text(json.dumps({
        "pid": os.getpid() + 1, "kind": "hermes-gateway", "argv": ["gateway", "run"],
    }))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(gb, "_supervisor_for_pid",
                        lambda pid: ("launchd", "ai.hermes.gateway"))

    info = gb._describe_local_conflict()
    assert info["supervisor"] == "launchd"
    assert info["service_label"] == "ai.hermes.gateway"
    assert "bootout" in info["stop_command"]


def test_supervisor_parks_on_unattributed_conflict(homes, monkeypatch):
    """A 409 proves someone owns the stream even when no local process can be
    named (another Hermes profile, another machine, a container). Parking is still
    required — otherwise we keep competing and replies can be swallowed."""
    calls = {"polls": 0, "sleeps": 0}

    async def conflicting_poll(token, chat, max_conflicts=3):
        calls["polls"] += 1
        return "conflict"

    async def fake_sleep(_s):
        calls["sleeps"] += 1
        if calls["sleeps"] >= 3:
            raise asyncio.CancelledError()

    monkeypatch.setattr(gb, "_poll_telegram", conflicting_poll)
    monkeypatch.setattr(gb, "_foreign_gateway_owner", lambda: None)   # unattributable
    monkeypatch.setattr(gb, "_same_bot_in_other_hermes_profile", lambda: None)
    monkeypatch.setattr(gb.asyncio, "sleep", fake_sleep)

    async def run():
        try:
            await gb._supervise_inbound("tok", "55")
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    # It backed off between probes rather than looping straight into polling.
    assert calls["sleeps"] >= 1
    h = gb.get_listener_health()
    assert h["state"] == "conflict" and h["blocked"] is True
    assert "not polling" in h["detail"]


def test_supervisor_probes_with_single_request_while_parked(homes, monkeypatch):
    """Re-checking a parked conflict must cost one request, not a full retry cycle."""
    seen = {"limits": []}

    async def poll(token, chat, max_conflicts=3):
        seen["limits"].append(max_conflicts)
        return "conflict"

    async def fake_sleep(_s):
        if len(seen["limits"]) >= 3:
            raise asyncio.CancelledError()

    monkeypatch.setattr(gb, "_poll_telegram", poll)
    monkeypatch.setattr(gb, "_foreign_gateway_owner", lambda: None)
    monkeypatch.setattr(gb, "_same_bot_in_other_hermes_profile", lambda: None)
    monkeypatch.setattr(gb.asyncio, "sleep", fake_sleep)

    async def run():
        try:
            await gb._supervise_inbound("tok", "55")
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert seen["limits"][0] == 3        # normal first attempt
    assert seen["limits"][1] == 1        # single probe once parked


def test_unattributed_conflict_names_shared_profile(homes, monkeypatch):
    """Case: the operator reused their main Hermes bot. That gateway is a different
    profile so the pid lookup can't see it, but the token match names it exactly."""
    async def poll(token, chat, max_conflicts=3):
        return "conflict"

    async def fake_sleep(_s):
        raise asyncio.CancelledError()

    monkeypatch.setattr(gb, "_poll_telegram", poll)
    monkeypatch.setattr(gb, "_foreign_gateway_owner", lambda: None)
    monkeypatch.setattr(gb, "_same_bot_in_other_hermes_profile", lambda: "Telegram")
    monkeypatch.setattr(gb.asyncio, "sleep", fake_sleep)

    async def run():
        try:
            await gb._supervise_inbound("tok", "55")
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    h = gb.get_listener_health()
    assert "also configured in Telegram" in h["detail"]
    # Must correct the common misconception that separate chats are enough.
    assert "per bot token" in h["remedy"]


def test_same_bot_detection_requires_matching_token(homes):
    """A different bot on another profile is NOT a conflict — never blame it."""
    shared, glob = homes
    gc.set_gateway("telegram", "ours-tok", "1")
    (glob / ".env").write_text('TELEGRAM_BOT_TOKEN="different-tok"\n')
    assert gb._same_bot_in_other_hermes_profile() is None

    (glob / ".env").write_text('TELEGRAM_BOT_TOKEN="ours-tok"\n')
    assert gb._same_bot_in_other_hermes_profile() == "Telegram"


def test_supervisor_detection_parses_launchctl(monkeypatch):
    """Matches on pid, not on a hardcoded label, so it works for any install."""
    import subprocess as sp

    class R:
        stdout = "PID\tStatus\tLabel\n-\t0\tcom.other\n3616\t0\tai.hermes.gateway\n"

    monkeypatch.setattr(gb.sys, "platform", "darwin")
    monkeypatch.setattr(sp, "run", lambda *a, **kw: R())
    assert gb._supervisor_for_pid(3616) == ("launchd", "ai.hermes.gateway")
    assert gb._supervisor_for_pid(9999) is None


# --------------------------------------------------------------------------- #
# PREVENTION: refuse to poll while another process owns the update stream.
# Two listeners on one token is worse than none — Telegram gives each update to
# whichever poller asks first, so a reply can be consumed by the other process
# and never reach the agent waiting on it.
# --------------------------------------------------------------------------- #
def test_supervisor_parks_instead_of_competing(homes, monkeypatch):
    polled = {"count": 0}

    async def never_called(token, chat, max_conflicts=3):
        polled["count"] += 1
        return None

    monkeypatch.setattr(gb, "_poll_telegram", never_called)
    monkeypatch.setattr(gb, "_foreign_gateway_owner", lambda: {
        "pid": 3616, "alive": True, "supervisor": "launchd",
        "service_label": "ai.hermes.gateway",
        "stop_command": "launchctl bootout gui/501/ai.hermes.gateway",
    })

    async def run():
        t = asyncio.create_task(gb._supervise_inbound("tok", "55"))
        await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert polled["count"] == 0                      # never competed
    h = gb.get_listener_health()
    assert h["state"] == "conflict" and h["blocked"] is True
    assert "not polling" in h["detail"]
    assert "launchctl bootout" in h["fix_command"]


def test_supervisor_starts_polling_when_no_owner(homes, monkeypatch):
    started = {"count": 0}

    async def poll(token, chat, max_conflicts=3):
        started["count"] += 1
        return None                                   # clean stop

    monkeypatch.setattr(gb, "_poll_telegram", poll)
    monkeypatch.setattr(gb, "_foreign_gateway_owner", lambda: None)

    asyncio.run(asyncio.wait_for(gb._supervise_inbound("tok", "55"), timeout=5))
    assert started["count"] == 1


def test_supervisor_resumes_after_owner_disappears(homes, monkeypatch):
    """Stopping the other gateway must restore replies without a server restart."""
    owners = [{"pid": 99, "alive": True, "stop_command": "kill 99"}, None]
    started = {"count": 0}

    def next_owner():
        return owners.pop(0) if owners else None

    async def poll(token, chat, max_conflicts=3):
        started["count"] += 1
        return None

    monkeypatch.setattr(gb, "_foreign_gateway_owner", next_owner)
    monkeypatch.setattr(gb, "_poll_telegram", poll)
    monkeypatch.setattr(gb, "_OWNERSHIP_RECHECK_SECONDS", 0.01)

    asyncio.run(asyncio.wait_for(gb._supervise_inbound("tok", "55"), timeout=5))
    assert started["count"] == 1                      # polled once freed


def test_poll_parks_after_repeated_conflicts(homes, monkeypatch):
    """Sustained 409s hand control back so the supervisor can park us."""
    import httpx

    class FakeResp:
        def __init__(self, p):
            self._p = p

        def json(self):
            return self._p

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            if "getWebhookInfo" in url:
                return FakeResp({"ok": True, "result": {"url": ""}})
            if (kw.get("params") or {}).get("timeout") == 0:
                return FakeResp({"ok": True, "result": []})
            return FakeResp({"ok": False, "error_code": 409, "description": "Conflict"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(gb.asyncio, "sleep", no_sleep)
    reason = asyncio.run(asyncio.wait_for(gb._poll_telegram("tok", "55"), timeout=5))
    assert reason == "conflict"


def test_foreign_owner_prefers_hermes_detector(homes, monkeypatch):
    """Hermes' detector also finds gateways with a stale/missing pid file."""
    monkeypatch.setattr(gb, "_describe_local_conflict", lambda: None)
    monkeypatch.setattr(gb, "_hermes_reported_gateway_pids", lambda: [4321])
    monkeypatch.setattr(gb, "_supervisor_for_pid", lambda pid: None)

    owner = gb._foreign_gateway_owner()
    assert owner["pid"] == 4321 and owner["alive"] is True
    assert owner["stop_command"] == "kill 4321"


def test_foreign_owner_none_when_clear(homes, monkeypatch):
    monkeypatch.setattr(gb, "_describe_local_conflict", lambda: None)
    monkeypatch.setattr(gb, "_hermes_reported_gateway_pids", lambda: [])
    assert gb._foreign_gateway_owner() is None


def test_detect_installed_service_before_it_runs(homes, monkeypatch, tmp_path):
    """RunAtLoad/KeepAlive means an installed service returns at next login, so
    detection must not depend on it currently running."""
    shared, _ = homes
    fake_home = tmp_path / "home"
    (fake_home / "Library" / "LaunchAgents").mkdir(parents=True)
    plist = fake_home / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"
    plist.write_text(f"<plist><dict><key>gateway</key><string>{shared}</string>"
                     "</dict></plist>")

    monkeypatch.setattr(gb.sys, "platform", "darwin")
    monkeypatch.setattr(gb.Path, "home", staticmethod(lambda: fake_home))

    svc = gb.detect_conflicting_gateway_service()
    assert svc is not None
    assert svc["manager"] == "launchd"
    assert svc["label"] == "ai.hermes.gateway"


def test_detect_installed_service_ignores_unrelated(homes, monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    (fake_home / "Library" / "LaunchAgents").mkdir(parents=True)
    (fake_home / "Library" / "LaunchAgents" / "com.other.plist").write_text(
        "<plist><dict><key>Program</key><string>/usr/bin/true</string></dict></plist>")
    monkeypatch.setattr(gb.sys, "platform", "darwin")
    monkeypatch.setattr(gb.Path, "home", staticmethod(lambda: fake_home))
    assert gb.detect_conflicting_gateway_service() is None


def test_route_refuses_token_shared_with_main_hermes(client, homes):
    """The most common way users create this bug: pasting their existing bot."""
    shared, glob = homes
    (glob / ".env").write_text('TELEGRAM_BOT_TOKEN="reused-tok"\nTELEGRAM_HOME_CHANNEL="1"\n')

    r = client.post("/setup/gateway", json={"key": "telegram", "token": "reused-tok"})
    assert r.status_code == 409
    assert r.json()["reason"] == "token_shared_with_hermes"
    assert gc.get_active_gateway() is None       # not persisted

    # A different bot for the swarm is accepted.
    ok = client.post("/setup/gateway", json={
        "key": "telegram", "token": "dedicated-tok", "home_chat_id": "5"})
    assert ok.status_code == 200


def test_status_no_longer_exposes_technical_fields(client, monkeypatch):
    """Non-technical users must never see service/process details — those are
    resolved automatically (see auto_disable_own_gateway_service) rather than
    surfaced for the user to act on."""
    monkeypatch.setattr(gb, "detect_conflicting_gateway_service", lambda: {
        "manager": "launchd", "label": "ai.hermes.gateway", "path": "/tmp/x.plist"})
    d = client.get("/setup/gateway/status").json()
    assert "service_conflict" not in d
    assert "detected_hermes" not in d


# --------------------------------------------------------------------------- #
# Home-chat auto-discovery: the user should never have to find or type a
# numeric Telegram chat ID. If they've already messaged the bot, the server
# finds that chat on its own; if not, it asks them to send one message.
# --------------------------------------------------------------------------- #
def test_route_saves_without_home_chat_id_via_discovery(client, homes, monkeypatch):
    async def discover(token):
        return {"chat_id": "42424242", "name": "Ada"}

    monkeypatch.setattr(gb, "discover_home_chat", discover)
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "tok"})
    assert r.status_code == 200
    assert r.json()["home_chat_id"] == "42424242"
    assert gc.get_active_gateway()["home_chat_id"] == "42424242"


def test_route_asks_for_first_message_when_nothing_to_discover(client, homes, monkeypatch):
    async def discover(token):
        return None

    monkeypatch.setattr(gb, "discover_home_chat", discover)
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "tok"})
    assert r.status_code == 409
    assert r.json()["needs_first_message"] is True
    # Nothing saved — the user has to message the bot first.
    assert gc.get_active_gateway() is None


def test_route_skips_discovery_when_home_chat_id_supplied(client, homes, monkeypatch):
    """An explicit chat id (e.g. from an older client) is trusted as-is; discovery
    is a convenience for setup, not a requirement on every save."""
    called = {"n": 0}

    async def discover(token):
        called["n"] += 1
        return {"chat_id": "should-not-be-used", "name": "x"}

    monkeypatch.setattr(gb, "discover_home_chat", discover)
    r = client.post("/setup/gateway", json={"key": "telegram", "token": "tok", "home_chat_id": "explicit"})
    assert r.status_code == 200
    assert called["n"] == 0
    assert gc.get_active_gateway()["home_chat_id"] == "explicit"


def test_discover_home_chat_reads_latest_sender(monkeypatch):
    import httpx

    class FakeResp:
        @staticmethod
        def json():
            return {"ok": True, "result": [
                {"update_id": 1, "message": {"chat": {"id": 111, "first_name": "Old"}}},
                {"update_id": 2, "message": {"chat": {"id": 222, "first_name": "Newest"}}},
            ]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
    found = asyncio.run(gb.discover_home_chat("tok"))
    assert found == {"chat_id": "222", "name": "Newest"}


def test_discover_home_chat_none_when_nobody_messaged(monkeypatch):
    import httpx

    class FakeResp:
        @staticmethod
        def json():
            return {"ok": True, "result": []}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())
    assert asyncio.run(gb.discover_home_chat("tok")) is None


# --------------------------------------------------------------------------- #
# Plain-English health for non-technical users: the dashboard only ever reads
# ``user_message`` — never pid/command/service detail, regardless of cause.
# --------------------------------------------------------------------------- #
def test_health_user_message_hides_technical_detail_on_conflict():
    gb._set_health("conflict",
                   "A launchd-managed Hermes gateway (pid 3616) is already receiving "
                   "updates for this bot.",
                   "Stop it with: launchctl bootout gui/501/ai.hermes.gateway.",
                   fix_command="launchctl bootout gui/501/ai.hermes.gateway")
    h = gb.get_listener_health()
    assert "3616" not in h["user_message"]
    assert "launchctl" not in h["user_message"]
    assert h["blocked"] is True
    gb._set_health("idle", "reset for other tests")


def test_health_user_message_calm_on_error_too():
    """Conflict and error causes are both internal — the user sees the same calm
    sentence either way, since the dashboard inbox works regardless of cause."""
    gb._set_health("error", "Telegram rejected the bot token (401).", "Get a new token.")
    conflict_msg = gb.get_listener_health()["user_message"]
    gb._set_health("conflict", "Something else owns the stream.", "Stop it.")
    conflict_msg2 = gb.get_listener_health()["user_message"]
    assert conflict_msg == conflict_msg2
    gb._set_health("idle", "reset for other tests")


def test_health_user_message_for_live_and_idle():
    gb._set_health("live", "Receiving replies from your chat.")
    assert "Connected" in gb.get_listener_health()["user_message"]
    gb._set_health("idle", "No messaging gateway configured.")
    h = gb.get_listener_health()
    assert h["user_message"] == "Not connected."
    assert "Connect" in h["user_action"]


# --------------------------------------------------------------------------- #
# Auto-cleanup: a leftover gateway service pointed at agent-teams' OWN Hermes
# home is retired automatically — no terminal command, no user decision. A
# service pointed at the user's OWN ~/.hermes profile is never touched.
# --------------------------------------------------------------------------- #
def test_auto_disable_noop_when_no_service_installed(homes, monkeypatch):
    monkeypatch.setattr(gb, "detect_conflicting_gateway_service", lambda: None)
    assert gb.auto_disable_own_gateway_service() is None


def test_auto_disable_stops_and_archives_launchd_service(homes, monkeypatch, tmp_path):
    shared, _ = homes
    plist_dir = tmp_path / "LaunchAgents"
    plist_dir.mkdir()
    plist = plist_dir / "ai.hermes.gateway.plist"
    plist.write_text("<plist>fake</plist>")

    monkeypatch.setattr(gb, "detect_conflicting_gateway_service", lambda: {
        "manager": "launchd", "label": "ai.hermes.gateway", "path": str(plist)})

    calls = []
    monkeypatch.setattr(gb.subprocess, "run",
                        lambda *a, **kw: calls.append(a[0]) or type("R", (), {
                            "returncode": 0, "stdout": "", "stderr": ""})())

    result = gb.auto_disable_own_gateway_service()
    assert result["label"] == "ai.hermes.gateway"
    # It actually issued a stop command...
    assert any("launchctl" in c[0] and "bootout" in c for c in calls)
    # ...and archived (moved, not deleted) the unit file rather than losing it.
    assert not plist.exists()
    archived = shared / "_retired_services" / "ai.hermes.gateway.plist"
    assert archived.exists()


def test_auto_disable_never_touches_service_for_other_home(homes, monkeypatch, tmp_path):
    """The scope guard: detect_conflicting_gateway_service only ever matches a
    file that mentions agent-teams' OWN Hermes home — a service for the user's
    main ~/.hermes profile does not contain that path and is left alone."""
    monkeypatch.setattr(gb, "detect_conflicting_gateway_service", lambda: None)
    called = {"n": 0}
    monkeypatch.setattr(gb.subprocess, "run", lambda *a, **kw: called.__setitem__("n", called["n"] + 1))
    assert gb.auto_disable_own_gateway_service() is None
    assert called["n"] == 0


def test_auto_disable_survives_stop_command_failure(homes, monkeypatch, tmp_path):
    """Best-effort: if the stop command errors, still archive the file so the
    service can't restart on next login, and never raise out to the caller."""
    shared, _ = homes
    plist_dir = tmp_path / "LaunchAgents"
    plist_dir.mkdir()
    plist = plist_dir / "ai.hermes.gateway.plist"
    plist.write_text("<plist>fake</plist>")

    monkeypatch.setattr(gb, "detect_conflicting_gateway_service", lambda: {
        "manager": "launchd", "label": "ai.hermes.gateway", "path": str(plist)})

    def boom(*a, **kw):
        raise OSError("launchctl not found")

    monkeypatch.setattr(gb.subprocess, "run", boom)
    result = gb.auto_disable_own_gateway_service()  # must not raise
    assert result is not None
    assert not plist.exists()
    assert (shared / "_retired_services" / "ai.hermes.gateway.plist").exists()


def test_start_listener_calls_auto_disable(homes, monkeypatch):
    """Cleanup runs as part of starting the listener, not as a separate step the
    user (or an admin route) has to trigger."""
    gc.set_gateway("telegram", "tok", "1")
    called = {"n": 0}
    monkeypatch.setattr(gb, "auto_disable_own_gateway_service", lambda: called.__setitem__("n", called["n"] + 1))

    class FakeTask:
        def done(self):
            return True

    class FakeLoop:
        def create_task(self, coro):
            coro.close()
            return FakeTask()

    monkeypatch.setattr(gb.asyncio, "get_running_loop", lambda: FakeLoop())
    gb.start_inbox_gateway_listener()
    assert called["n"] == 1


def test_diagnose_conflict_prefers_webhook_cause():
    """A webhook and a competing poller both yield 409 but need different fixes."""
    class FakeResp:
        @staticmethod
        def json():
            return {"ok": True, "result": {"url": "https://ex.com/hook"}}

    class FakeClient:
        async def get(self, url, **kw):
            return FakeResp()

    detail, remedy = asyncio.run(gb._diagnose_conflict(FakeClient(), "https://api.telegram.org/botX"))
    assert "webhook" in detail.lower()
    assert "Remove webhook" in remedy or "webhook" in remedy.lower()


def test_diagnose_conflict_reports_competing_listener(homes):
    class FakeResp:
        @staticmethod
        def json():
            return {"ok": True, "result": {"url": ""}}       # no webhook

    class FakeClient:
        async def get(self, url, **kw):
            return FakeResp()

    detail, remedy = asyncio.run(gb._diagnose_conflict(FakeClient(), "https://api.telegram.org/botX"))
    assert "already receiving updates" in detail
    assert "only one listener" in remedy


def test_listener_recovers_when_conflict_clears(homes, monkeypatch):
    """The conflict is usually transient operator error (a stray gateway process).
    Once it clears, the listener must go back to 'live' on its own — no restart."""
    import httpx

    seen = {"conflicted": False, "recovered": False}

    class FakeResp:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            if "getWebhookInfo" in url:
                return FakeResp({"ok": True, "result": {"url": ""}})
            params = kw.get("params") or {}
            if params.get("timeout") == 0:                       # initial drain
                return FakeResp({"ok": True, "result": []})
            if not seen["conflicted"]:                           # first real poll → 409
                seen["conflicted"] = True
                return FakeResp({"ok": False, "error_code": 409,
                                 "description": "Conflict: terminated by other getUpdates request"})
            if gb.get_listener_health()["state"] == "live":      # recovered → stop
                seen["recovered"] = True
                raise asyncio.CancelledError()
            return FakeResp({"ok": True, "result": []})          # conflict cleared

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(gb.asyncio, "sleep", no_sleep)

    async def run():
        try:
            await asyncio.wait_for(gb._poll_telegram("tok", "55"), timeout=5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    asyncio.run(run())
    assert seen["conflicted"] is True        # it really did hit the 409 path
    assert seen["recovered"] is True         # and returned to 'live' unaided
    h = gb.get_listener_health()
    assert h["conflict_count"] == 0          # reset after recovery


def test_listener_gives_up_on_revoked_token(homes, monkeypatch):
    """A 401 can't be fixed by retrying, so the listener must stop and say so
    instead of hammering Telegram forever."""
    import httpx

    class FakeResp:
        def json(self):
            return {"ok": False, "error_code": 401, "description": "Unauthorized"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            params = kw.get("params") or {}
            if params.get("timeout") == 0:
                return FakeResp()
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(gb.asyncio, "sleep", no_sleep)
    # Returns (does not loop) on an unauthorized token.
    asyncio.run(asyncio.wait_for(gb._poll_telegram("bad", "55"), timeout=5))
    h = gb.get_listener_health()
    assert h["state"] == "error"
    assert "token" in h["detail"].lower()


# --------------------------------------------------------------------------- #
# Credential hygiene: tokens ride inside request URLs, and logs get pasted into
# bug reports — so they must never reach a log sink.
# --------------------------------------------------------------------------- #
def test_log_redaction_strips_bot_token():
    from teams_server.config import redact_secrets

    tok = "8954351324:AAHiA2zPoH8N-tkz2mBZNXRd99DhONgx-4A"
    line = f"HTTP Request: GET https://api.telegram.org/bot{tok}/getUpdates?timeout=30"
    out = redact_secrets(line)
    assert tok not in out
    assert "<redacted" in out


def test_log_redaction_filter_applied_to_records():
    import logging
    from teams_server.config import _RedactSecretsFilter

    tok = "123456789:AAEEmaskedTOKENvalue0123456789abcdefg"
    rec = logging.LogRecord("x", logging.INFO, __file__, 1,
                            "calling /bot%s/getUpdates", (tok,), None)
    _RedactSecretsFilter().filter(rec)
    assert tok not in rec.getMessage()


# --------------------------------------------------------------------------- #
# Runtime bridge: outbound forward + inbound reply matching
# --------------------------------------------------------------------------- #
def test_forward_noop_when_unconfigured(homes):
    # No gateway set → _do_forward returns quietly and remembers nothing.
    gb._sent_index.clear()
    gb._do_forward("qid-x", "agent1", "question?", "question")
    assert gb._lookup_sent("anything") is None


def test_forward_sends_and_remembers(homes, monkeypatch):
    import json
    import tools.send_message_tool as smt

    gc.set_gateway("telegram", "tok", "555")
    captured = {}
    monkeypatch.setattr(
        smt, "send_message_tool",
        lambda args, **kw: (captured.update(args) or json.dumps({"success": True, "message_id": 7})),
    )
    gb._sent_index.clear()
    gb._do_forward("qid-A", "engineer", "Deploy now?", "question")

    assert captured["target"] == "telegram"
    assert "engineer" in captured["message"]
    assert gb._lookup_sent("7") == ("qid-A", "engineer")


def _seed_pending(agent_name, question):
    import teams_server.tools as T
    return T.add_pending_question(agent_name, question, waiting_in_turn=True)


def test_deliver_reply_exact_match(monkeypatch):
    import teams_server.tools as T
    import teams_server.server as S

    class FakeDaemon:
        cfg = {}
    T._daemon_registry["eng"] = FakeDaemon()
    qid = _seed_pending("eng", "Ship it?")
    gb._sent_index.clear()
    gb._remember_sent("100", qid, "eng")

    got = {}

    async def fake_final(daemon, q, text):
        got.update(qid=q, text=text)
        return {"ok": True, "delivery": "in_turn"}

    monkeypatch.setattr(S, "_finalize_human_answer", fake_final)

    ok = asyncio.run(gb._deliver_reply("yes", "100"))
    assert ok is True
    assert got["qid"] == qid and got["text"] == "yes"


def test_deliver_reply_falls_back_to_newest_pending(monkeypatch):
    import teams_server.tools as T
    import teams_server.server as S

    class FakeDaemon:
        cfg = {}
    T._daemon_registry["eng"] = FakeDaemon()
    qid = _seed_pending("eng", "Latest question?")
    gb._sent_index.clear()  # no reply-to mapping → fallback

    got = {}

    async def fake_final(daemon, q, text):
        got.update(qid=q, text=text)
        return {"ok": True, "delivery": "task"}

    monkeypatch.setattr(S, "_finalize_human_answer", fake_final)

    ok = asyncio.run(gb._deliver_reply("fallback answer", "does-not-exist"))
    assert ok is True
    assert got["text"] == "fallback answer"


def test_deliver_reply_no_pending_is_ignored(monkeypatch):
    import teams_server.tools as T
    # Clear any pending questions left by earlier tests in this module.
    with T._pending_lock:
        T._pending_human_questions.clear()
    gb._sent_index.clear()
    ok = asyncio.run(gb._deliver_reply("orphan reply", None))
    assert ok is False
