#!/usr/bin/env python3
"""Unit tests for the Architect (master team-builder) — swarm_server/master.py.

No LLM and no Hermes AIAgent: we exercise the tool handlers directly against a
real (but temp-rooted) config + workspace, plus the chat-log persistence and the
caller/confirmation guards. The AIAgent construction is covered only by an
import smoke (it builds lazily, so importing the module needs no model).

Run:  pytest tests/test_master_agent.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import swarm_server.config as cfg_mod  # noqa: E402
import swarm_server.master as master  # noqa: E402

MASTER_KW = {"task_id": "master.id:master"}  # parsed by _caller → "master"
# The real caller convention is task_id="agent_name:master".
MK = {"task_id": "agent_name:master"}
NOT_MASTER = {"task_id": "agent_name:founder2"}


@pytest.fixture()
def swarm(tmp_path, monkeypatch):
    """Point the config module at a throwaway data root and seed an empty swarm."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(cfg_mod, "DATA_ROOT", data)
    monkeypatch.setattr(cfg_mod, "WORKSPACE_ROOT", data / "teams")
    monkeypatch.setattr(cfg_mod, "AGENTS_CONFIG_PATH", data / "agents_config.json")
    monkeypatch.setattr(cfg_mod, "_config_cache", None, raising=False)
    monkeypatch.setattr(cfg_mod, "_config_cache_key", None, raising=False)
    (data / "agents_config.json").write_text(
        json.dumps({"teams": {}, "agents": {}, "settings": {}}), encoding="utf-8"
    )
    return data


def _call(handler, args, kw=None):
    return json.loads(handler(args, **(kw or MK)))


# ---------------------------------------------------------------------------
# Caller guard
# ---------------------------------------------------------------------------
def test_non_master_caller_is_rejected(swarm):
    for handler in (
        master._master_overview_handler,
        master._master_create_team_handler,
        master._master_delete_team_handler,
    ):
        r = json.loads(handler({}, **NOT_MASTER))
        assert r["success"] is False
        assert "Architect" in r["error"]


def test_unknown_caller_rejected(swarm):
    r = json.loads(master._master_overview_handler({}, task_id="nope"))
    assert r["success"] is False


# ---------------------------------------------------------------------------
# Build flow: team → agents → links → files
# ---------------------------------------------------------------------------
def test_full_build_flow(swarm):
    # Create a team.
    r = _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme Co"})
    assert r["success"] and r["team_id"] == "acme"

    # Lead (autonomous) + worker, then a supervisor over both.
    r = _call(master._master_create_agent_handler, {
        "team_id": "acme", "agent_name": "lead", "display_name": "Lead",
        "role_soul": "You are the Lead.", "autonomous": True,
    })
    assert r["success"]
    r = _call(master._master_create_agent_handler, {
        "team_id": "acme", "agent_name": "worker", "display_name": "Worker",
        "role_soul": "You are the Worker.", "allowed_peers": ["lead"],
    })
    assert r["success"]
    r = _call(master._master_create_agent_handler, {
        "team_id": "acme", "agent_name": "boss", "display_name": "Overseer",
        "role_soul": "You watch the team.", "is_supervisor": True,
        "allowed_peers": ["lead", "worker"],
    })
    assert r["success"]

    cfg = cfg_mod.load_agents_config()
    assert cfg["agents"]["lead"]["autonomous"] is True
    assert cfg["agents"]["boss"]["is_supervisor"] is True
    # Links are bidirectional: worker↔lead set during worker creation.
    assert "worker" in cfg["agents"]["lead"]["allowed_peers"]
    assert "lead" in cfg["agents"]["worker"]["allowed_peers"]
    # Supervisor linked to both.
    assert set(cfg["agents"]["boss"]["allowed_peers"]) == {"lead", "worker"}

    # set_links replaces the list (and stays bidirectional).
    r = _call(master._master_set_links_handler, {"agent_name": "worker", "peers": ["boss"]})
    assert r["success"]
    cfg = cfg_mod.load_agents_config()
    assert cfg["agents"]["worker"]["allowed_peers"] == ["boss"]
    assert "worker" not in cfg["agents"]["lead"]["allowed_peers"]  # old link dropped

    # update_agent respects the editable-field whitelist.
    r = _call(master._master_update_agent_handler,
              {"agent_name": "lead", "fields": {"autonomous": False, "bogus_key": "x"}})
    assert r["success"]
    assert r["config"]["autonomous"] is False
    assert "bogus_key" not in r["config"]


def test_overview_and_get_team(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    _call(master._master_create_agent_handler, {
        "team_id": "acme", "agent_name": "lead", "display_name": "Lead",
        "role_soul": "You are the Lead.",
    })
    ov = _call(master._master_overview_handler, {})
    assert ov["team_count"] == 1
    assert ov["teams"][0]["agents"][0]["agent_name"] == "lead"
    # workspace.md seeded by create_team is visible.
    assert "Project: Acme" in ov["teams"][0]["workspace_md"]

    gt = _call(master._master_get_team_handler, {"team_id": "acme"})
    assert gt["success"]
    assert gt["agents"][0]["role_soul"] == "You are the Lead."

    miss = _call(master._master_get_team_handler, {"team_id": "ghost"})
    assert miss["success"] is False


# ---------------------------------------------------------------------------
# Files: write / read / list + path safety
# ---------------------------------------------------------------------------
def test_file_write_read_list(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    r = _call(master._master_write_file_handler, {
        "team_id": "acme", "area": "workspace", "path": "workspace.md",
        "content": "# Acme\nNorth star: 10 users.",
    })
    assert r["success"] and r["path"] == "workspace.md"

    r = _call(master._master_read_file_handler,
              {"team_id": "acme", "area": "workspace", "path": "workspace.md"})
    assert r["success"] and "North star" in r["content"]

    # A project seed file.
    r = _call(master._master_write_file_handler, {
        "team_id": "acme", "area": "project", "path": "README.md", "content": "hello",
    })
    assert r["success"]
    r = _call(master._master_list_files_handler, {"team_id": "acme", "area": "workspace"})
    assert "workspace.md" in r["files"]


def test_path_traversal_rejected(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    r = _call(master._master_write_file_handler, {
        "team_id": "acme", "area": "workspace", "path": "../escape.txt", "content": "x",
    })
    assert r["success"] is False and ".." in r["error"]


def test_bad_area_and_missing_team_rejected(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    r = _call(master._master_write_file_handler, {
        "team_id": "acme", "area": "secrets", "path": "x", "content": "x",
    })
    assert r["success"] is False
    r = _call(master._master_write_file_handler, {
        "team_id": "ghost", "area": "workspace", "path": "x", "content": "x",
    })
    assert r["success"] is False


def test_write_file_size_cap(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    r = _call(master._master_write_file_handler, {
        "team_id": "acme", "area": "workspace", "path": "big.txt",
        "content": "z" * (master._MAX_FILE_CHARS + 1),
    })
    assert r["success"] is False and "too long" in r["error"]


# ---------------------------------------------------------------------------
# Destructive ops require explicit confirmation
# ---------------------------------------------------------------------------
def test_delete_requires_confirm(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    _call(master._master_create_agent_handler, {
        "team_id": "acme", "agent_name": "lead", "display_name": "Lead",
        "role_soul": "x",
    })
    # Without confirm → refused, nothing removed.
    r = _call(master._master_delete_agent_handler, {"agent_name": "lead", "confirm": False})
    assert r["success"] is False and "confirm" in r["error"]
    assert "lead" in cfg_mod.load_agents_config()["agents"]

    r = _call(master._master_delete_team_handler, {"team_id": "acme", "confirm": False})
    assert r["success"] is False
    assert "acme" in cfg_mod.load_agents_config()["teams"]


def test_delete_with_confirm_succeeds_without_hooks(swarm):
    # Hooks are unset in tests, so despawn is a no-op — but config-level deletion
    # for delete_team still happens inside the (unwired) hook on the server side.
    # Here we only assert the handler accepts confirm=true and reports success.
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    r = _call(master._master_delete_team_handler, {"team_id": "acme", "confirm": True})
    assert r["success"] and r["deleted"] is True


# ---------------------------------------------------------------------------
# send_task: no daemon → graceful error
# ---------------------------------------------------------------------------
def test_send_task_without_daemon(swarm):
    _call(master._master_create_team_handler, {"team_id": "acme", "name": "Acme"})
    _call(master._master_create_agent_handler, {
        "team_id": "acme", "agent_name": "lead", "display_name": "Lead", "role_soul": "x",
    })
    r = _call(master._master_send_task_handler, {"agent_name": "lead", "message": "go"})
    assert r["success"] is False and "daemon" in r["error"]


# ---------------------------------------------------------------------------
# Chat-log persistence + soul content
# ---------------------------------------------------------------------------
def test_chat_log_roundtrip(swarm):
    m = master.MasterAgent()
    m._append_log("user", "hi")
    m._append_log("assistant", "hello, what do you want to build?")
    hist = m.history()
    assert [h["role"] for h in hist] == ["user", "assistant"]
    assert hist[1]["content"].startswith("hello")


def test_reset_starts_new_session(swarm):
    m = master.MasterAgent()
    first = m._session_id
    m.reset()
    assert m._session_id != first
    # The reset marker is logged.
    assert any("reset" in h.get("content", "") for h in m.history())


def test_soul_has_key_sections():
    soul = master.MASTER_SOUL.lower()
    for needle in ("role_soul", "allowed_peers", "is_supervisor", "autonomous",
                   "workspace.md", "interview", "credentials",
                   # environment-awareness section + web research
                   "environment", "dashboard", "web_search"):
        assert needle in soul, f"MASTER_SOUL missing: {needle}"
    # All 12 tools have a matching handler.
    names = {s["function"]["name"] for s in master._MASTER_TOOL_SCHEMAS}
    assert names == set(master._MASTER_HANDLERS)


def test_architect_gets_web_toolset_only():
    # The Architect is wired with exactly swarm_master + web (no terminal/browser).
    import inspect
    src = inspect.getsource(master.MasterAgent._ensure_agent)
    assert '"enabled_toolsets": ["swarm_master", "web"]' in src


# ---------------------------------------------------------------------------
# Bidirectional connections (framework-wide)
# ---------------------------------------------------------------------------
def _mk_team(team_id="t", members=("a", "b", "c")):
    cfg = cfg_mod.load_agents_config()
    cfg_mod.create_team(cfg, team_id, team_id.upper())
    for m in members:
        cfg_mod.create_agent(cfg_mod.load_agents_config(), name=m, team_id=team_id,
                             display_name=m.upper(), role_soul="x")


def test_set_agent_peers_is_bidirectional(swarm):
    _mk_team()
    # Linking a→[b] must also put a into b's list.
    cfg_mod.set_agent_peers(cfg_mod.load_agents_config(), "a", ["b"])
    cfg = cfg_mod.load_agents_config()
    assert cfg["agents"]["a"]["allowed_peers"] == ["b"]
    assert "a" in cfg["agents"]["b"]["allowed_peers"]

    # Re-setting a→[c] drops the a↔b link from BOTH sides and adds a↔c.
    cfg_mod.set_agent_peers(cfg_mod.load_agents_config(), "a", ["c"])
    cfg = cfg_mod.load_agents_config()
    assert cfg["agents"]["a"]["allowed_peers"] == ["c"]
    assert "a" in cfg["agents"]["c"]["allowed_peers"]
    assert "a" not in cfg["agents"]["b"]["allowed_peers"]


def test_set_agent_peers_rejects_self_link(swarm):
    _mk_team()
    with pytest.raises(ValueError):
        cfg_mod.set_agent_peers(cfg_mod.load_agents_config(), "a", ["a"])


def test_peer_allowed_is_symmetric(swarm):
    _mk_team()
    # Manually store a one-directional link (legacy data): only a lists b.
    cfg = cfg_mod.load_agents_config()
    cfg["agents"]["a"]["allowed_peers"] = ["b"]
    cfg["agents"]["b"]["allowed_peers"] = []
    cfg_mod._save_full_config(cfg)
    cfg = cfg_mod.load_agents_config()
    # Both directions are authorized despite only one side recording the link.
    assert cfg_mod.peer_allowed(cfg, "a", "b") is True
    assert cfg_mod.peer_allowed(cfg, "b", "a") is True
    # Cross-team still blocked.
    assert cfg_mod.peer_allowed(cfg, "a", "missing") is False
