"""Unit tests for the server-lifecycle CLI commands: `status`, `down`, `setup`,
and the pidfile helpers that make them work for a detached `agent-teams up`.

These never start a real server — they patch the pidfile path to a tmp dir and
fake liveness/health probes, so they stay fast and hermetic.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import swarm_server.cli as cli          # noqa: E402


def _point_pidfile(monkeypatch, tmp_path):
    pf = tmp_path / "swarm.pid"
    monkeypatch.setattr(cli, "_pidfile_path", lambda: pf)
    return pf


# ---- pidfile lifecycle -------------------------------------------------------
def test_write_and_clear_pidfile(monkeypatch, tmp_path):
    pf = _point_pidfile(monkeypatch, tmp_path)
    cli._write_pidfile()
    assert pf.read_text().strip() == str(os.getpid())
    # our own pid is alive → _running_pid finds it
    assert cli._running_pid() == os.getpid()
    cli._clear_pidfile()
    assert not pf.exists()
    assert cli._running_pid() is None


def test_running_pid_stale_file(monkeypatch, tmp_path):
    pf = _point_pidfile(monkeypatch, tmp_path)
    pf.write_text("999999")              # a pid that (almost certainly) isn't alive
    assert cli._running_pid() is None


def test_clear_missing_pidfile_is_silent(monkeypatch, tmp_path):
    _point_pidfile(monkeypatch, tmp_path)
    cli._clear_pidfile()                 # no file → no raise


# ---- status ------------------------------------------------------------------
def test_status_not_running(monkeypatch, tmp_path, capsys):
    _point_pidfile(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_probe_health", lambda *a, **k: False)
    assert cli.cmd_status(argparse.Namespace()) == 1
    assert "not running" in capsys.readouterr().out


def test_status_running_via_pidfile(monkeypatch, tmp_path, capsys):
    pf = _point_pidfile(monkeypatch, tmp_path)
    pf.write_text(str(os.getpid()))
    monkeypatch.setattr(cli, "_probe_health", lambda *a, **k: True)
    assert cli.cmd_status(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "running" in out and "ok" in out


def test_status_running_no_pidfile_but_port_up(monkeypatch, tmp_path, capsys):
    _point_pidfile(monkeypatch, tmp_path)          # empty (no pidfile)
    monkeypatch.setattr(cli, "_probe_health", lambda *a, **k: True)
    assert cli.cmd_status(argparse.Namespace()) == 0
    assert "no pidfile" in capsys.readouterr().out


# ---- down --------------------------------------------------------------------
def test_down_when_not_running(monkeypatch, tmp_path, capsys):
    _point_pidfile(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_probe_health", lambda *a, **k: False)
    assert cli.cmd_down(argparse.Namespace()) == 0
    assert "nothing to stop" in capsys.readouterr().out


def test_down_no_pidfile_but_responding(monkeypatch, tmp_path, capsys):
    _point_pidfile(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_probe_health", lambda *a, **k: True)
    # A server answers but we have no pid to signal → guide the user, don't crash.
    assert cli.cmd_down(argparse.Namespace()) == 1
    assert "no pidfile" in capsys.readouterr().out


def test_down_signals_and_clears(monkeypatch, tmp_path, capsys):
    pf = _point_pidfile(monkeypatch, tmp_path)
    pf.write_text("4242")
    monkeypatch.setattr(cli, "_running_pid", lambda: 4242)
    signalled = {}

    def fake_kill(pid, sig):
        signalled.setdefault("calls", []).append((pid, sig))
        if sig == 0:                     # liveness probe → report dead so the loop exits
            raise OSError("gone")

    monkeypatch.setattr(cli.os, "kill", fake_kill)
    assert cli.cmd_down(argparse.Namespace()) == 0
    assert signalled["calls"][0][0] == 4242        # SIGTERM sent to our pid
    assert not pf.exists()                          # pidfile cleared
    assert "Stopped" in capsys.readouterr().out


# ---- up / detach dispatch ----------------------------------------------------
def test_up_runs_foreground_when_not_detached(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "_serve", lambda a: called.setdefault("serve", True) or 0)
    monkeypatch.setattr(cli, "_start_detached", lambda a: called.setdefault("detach", True) or 0)
    cli.cmd_up(argparse.Namespace(detach=False))
    assert called == {"serve": True}


def test_up_daemonizes_when_detached(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "_serve", lambda a: called.setdefault("serve", True) or 0)
    monkeypatch.setattr(cli, "_start_detached", lambda a: called.setdefault("detach", True) or 0)
    cli.cmd_up(argparse.Namespace(detach=True))
    assert called == {"detach": True}


def test_detach_is_noop_when_already_running(monkeypatch, capsys):
    # Already up → report and return WITHOUT forking (guard runs before os.fork).
    monkeypatch.setattr(cli, "_running_pid", lambda: 123)
    monkeypatch.setattr(cli, "_probe_health", lambda *a, **k: True)
    rc = cli._start_detached(argparse.Namespace(detach=True, log=None))
    assert rc == 0
    assert "already running" in capsys.readouterr().out


# ---- setup -------------------------------------------------------------------
def test_setup_invokes_hermes_against_shared_home(monkeypatch, tmp_path, capsys):
    import swarm_server.model_config as mc
    monkeypatch.setattr(mc, "SHARED_HERMES_HOME", tmp_path / ".hermes-shared")
    calls = {}

    def fake_call(argv, env=None):
        calls["argv"] = argv
        calls["home"] = env.get("HERMES_HOME") if env else None
        return 0

    monkeypatch.setattr(cli.subprocess if hasattr(cli, "subprocess") else __import__("subprocess"),
                        "call", fake_call, raising=False)
    # cmd_setup imports subprocess locally; patch the module object it will import.
    import subprocess
    monkeypatch.setattr(subprocess, "call", fake_call)

    rc = cli.cmd_setup(argparse.Namespace(rest=[]))
    assert rc == 0
    assert calls["argv"][1] == "setup"
    assert calls["home"] == str(tmp_path / ".hermes-shared")
