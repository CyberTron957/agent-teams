#!/usr/bin/env python3
"""Tests for two strategic-autonomy fixes (2026-06-15):

1. ONE-SHOT / BOUNDED CRONS — agents kept scheduling one-time work as recurring
   wake-ups that fired forever and monopolised them. `max_runs` + record_cron_fire
   make a wake-up auto-stop after N fires (max_runs=1 == run once).

2. recall_decisions TOOL + search_decisions — agents can search the team's past
   decision log on demand (beyond the 20 auto-injected) to review a stalling
   strategy and pivot.

Run:  pytest tests/test_cron_oneshot_and_recall.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import swarm_server.config as cfg_mod  # noqa: E402
from swarm_server.config import add_agent_cron, record_cron_fire  # noqa: E402
from swarm_server.monitoring import MonitoringDB  # noqa: E402


# ---------------------------------------------------------------------------
# Shared in-memory config (no file writes)
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_config(monkeypatch):
    full = {"agents": {"founder": {"team_id": "t1", "crons": []}}}
    monkeypatch.setattr(cfg_mod, "load_agents_config", lambda: full)
    monkeypatch.setattr(cfg_mod, "_save_full_config", lambda c: None)
    return full


# ---------------------------------------------------------------------------
# 1. Bounded / one-shot crons
# ---------------------------------------------------------------------------
def test_add_cron_without_max_runs_is_unbounded(fake_config):
    entry = add_agent_cron(fake_config, "founder", "0 9 * * *", "daily check")
    assert "max_runs" not in entry
    assert "runs" not in entry


def test_add_cron_with_max_runs_one(fake_config):
    entry = add_agent_cron(fake_config, "founder", "0 9 * * *", "one-time", max_runs=1)
    assert entry["max_runs"] == 1
    assert entry["runs"] == 0


def test_add_cron_rejects_bad_max_runs(fake_config):
    with pytest.raises(ValueError):
        add_agent_cron(fake_config, "founder", "0 9 * * *", "x", max_runs=0)
    with pytest.raises(ValueError):
        add_agent_cron(fake_config, "founder", "0 9 * * *", "x", max_runs=-3)


def test_record_cron_fire_oneshot_completes_and_disables(fake_config):
    entry = add_agent_cron(fake_config, "founder", "0 9 * * *", "one-time", max_runs=1)
    res = record_cron_fire("founder", entry["id"])
    assert res == {"runs": 1, "max_runs": 1, "completed": True}
    stored = fake_config["agents"]["founder"]["crons"][0]
    assert stored["runs"] == 1
    assert stored["enabled"] is False           # auto-stopped — won't fire again


def test_record_cron_fire_bounded_three(fake_config):
    entry = add_agent_cron(fake_config, "founder", "@hourly", "thrice", max_runs=3)
    r1 = record_cron_fire("founder", entry["id"])
    r2 = record_cron_fire("founder", entry["id"])
    r3 = record_cron_fire("founder", entry["id"])
    assert (r1["completed"], r2["completed"], r3["completed"]) == (False, False, True)
    assert r3["runs"] == 3
    assert fake_config["agents"]["founder"]["crons"][0]["enabled"] is False


def test_record_cron_fire_unbounded_never_completes(fake_config):
    entry = add_agent_cron(fake_config, "founder", "@hourly", "forever")
    for _ in range(5):
        res = record_cron_fire("founder", entry["id"])
        assert res["completed"] is False
    # runs still tracked, cron stays enabled
    assert fake_config["agents"]["founder"]["crons"][0].get("enabled", True) is True
    assert fake_config["agents"]["founder"]["crons"][0]["runs"] == 5


def test_record_cron_fire_unknown_cron_is_safe(fake_config):
    res = record_cron_fire("founder", "does-not-exist")
    assert res["completed"] is False


# ---------------------------------------------------------------------------
# 2. Decision search / recall
# ---------------------------------------------------------------------------
@pytest.fixture()
def db(tmp_path):
    return MonitoringDB(tmp_path / "mon.db")


def test_search_decisions_team_scoped(db):
    db.log_decision("a1", "Switched checkout to live_mode", team_id="t1")
    db.log_decision("a2", "Picked Stripe for billing", team_id="t1")
    db.log_decision("x9", "Other team thing", team_id="t2")

    rows = db.search_decisions(team_id="t1")
    decisions = {r["decision"] for r in rows}
    assert "Switched checkout to live_mode" in decisions
    assert "Picked Stripe for billing" in decisions
    assert "Other team thing" not in decisions          # scoped out


def test_search_decisions_keyword_filter(db):
    db.log_decision("a1", "Switched checkout to live_mode", team_id="t1")
    db.log_decision("a2", "Picked Stripe for billing", team_id="t1")
    db.log_decision("a3", "Onboarding email v2 shipped", team_id="t1")

    rows = db.search_decisions(team_id="t1", query="checkout")
    assert len(rows) == 1
    assert rows[0]["decision"] == "Switched checkout to live_mode"


def test_search_decisions_newest_first_and_limit(db):
    for i in range(10):
        db.log_decision("a1", f"decision {i}", team_id="t1")
    rows = db.search_decisions(team_id="t1", limit=3)
    assert len(rows) == 3
    # newest first -> decision 9, 8, 7
    assert rows[0]["decision"] == "decision 9"


def test_search_decisions_like_wildcards_are_literal(db):
    db.log_decision("a1", "100% done with pricing", team_id="t1")
    db.log_decision("a2", "started pricing work", team_id="t1")
    # '%' in the query must match a literal percent, not act as a wildcard
    rows = db.search_decisions(team_id="t1", query="100%")
    assert len(rows) == 1
    assert rows[0]["decision"] == "100% done with pricing"
