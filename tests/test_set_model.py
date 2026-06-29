"""Unit tests for the `agent-teams set-model` CLI command.

These patch out the actual config write (set_default_model) so they stay fast
and never touch a real ~/.hermes — we only assert the command's normalization
and provider-defaulting logic.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import swarm_server.model_config as mc          # noqa: E402
from swarm_server.cli import cmd_set_model       # noqa: E402


def _args(**kw):
    d = dict(model=None, provider=None, base_url=None, api_key=None)
    d.update(kw)
    return argparse.Namespace(**d)


def _patch(monkeypatch):
    captured = {}
    monkeypatch.setattr(mc, "set_default_model", lambda **k: captured.update(k))
    monkeypatch.setattr(mc, "get_default_model", lambda: {})
    return captured


def test_requires_model():
    assert cmd_set_model(_args(model="")) == 2


def test_provider_defaults_to_custom_with_base_url(monkeypatch):
    cap = _patch(monkeypatch)
    assert cmd_set_model(_args(model="deepseek-chat", base_url="localhost:4000/v1")) == 0
    assert cap["provider"] == "custom"
    assert cap["base_url"] == "http://localhost:4000/v1"   # bare host:port gets a scheme


def test_provider_defaults_to_openai_without_base_url(monkeypatch):
    cap = _patch(monkeypatch)
    assert cmd_set_model(_args(model="gpt-4o", api_key="sk-x")) == 0
    assert cap["provider"] == "openai"
    assert cap["base_url"] == ""
    assert cap["api_key"] == "sk-x"


def test_explicit_provider_is_respected(monkeypatch):
    cap = _patch(monkeypatch)
    cmd_set_model(_args(model="claude-x", provider="anthropic", api_key="k"))
    assert cap["provider"] == "anthropic"


def test_typo_base_url_warns(monkeypatch, capsys):
    _patch(monkeypatch)
    cmd_set_model(_args(model="m", base_url="http://localhost:4000/vi"))   # /vi not /v1
    assert "doesn't end in a version path" in capsys.readouterr().out


def test_good_base_url_no_warn(monkeypatch, capsys):
    _patch(monkeypatch)
    cmd_set_model(_args(model="m", base_url="http://localhost:4000/v1"))
    assert "doesn't end in a version path" not in capsys.readouterr().out
