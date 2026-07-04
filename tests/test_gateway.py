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
    """Point both Hermes homes at temp dirs so writes never touch a real config."""
    shared = tmp_path / "shared"
    glob = tmp_path / "global"
    shared.mkdir()
    glob.mkdir()
    monkeypatch.setattr(gc, "SHARED_HERMES_HOME", shared)
    monkeypatch.setattr(gc, "GLOBAL_HERMES_HOME", glob)
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
    # Credentials live as env vars in the home's .env (NOT config.yaml).
    env = (shared / ".env").read_text()
    assert "TELEGRAM_BOT_TOKEN=" in env
    assert "TELEGRAM_HOME_CHANNEL=" in env


def test_detect_and_adopt_global(homes, monkeypatch):
    shared, glob = homes
    # Simulate a bot already configured in the operator's ~/.hermes.
    (glob / ".env").write_text('DISCORD_BOT_TOKEN="d-tok"\nDISCORD_HOME_CHANNEL="42"\n')

    det = gc.detect_global_gateway()
    assert det is not None and det["key"] == "discord" and det["home_chat_id"] == "42"

    # Adopt copies it into the shared home; afterwards it's the active gateway.
    adopted = gc.adopt_global_gateway()
    assert adopted == {"key": "discord", "label": "Discord"}
    assert gc.get_active_gateway()["key"] == "discord"


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


def test_route_adopt_without_detection_fails(client, homes):
    r = client.post("/setup/gateway", json={"adopt_detected": True})
    assert r.status_code == 400


def test_route_lists_catalogue(client):
    gws = client.get("/gateways").json()["gateways"]
    assert any(g["key"] == "telegram" for g in gws)


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
