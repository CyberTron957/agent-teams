"""Hermes compatibility self-check.

The teams is built OVER Hermes and *wants* to ride Hermes' constant updates —
so we deliberately do NOT pin an upper version bound. The price of that freedom
is that a few load-bearing features reach into Hermes' internal/underscore APIs,
and several teams config lists mirror names Hermes owns. If a Hermes update moves
one of those, today it degrades SILENTLY (every reach-in is wrapped in
``try/except``).

This module turns silent drift into a LOUD signal: it probes each fragile seam
against the *installed* Hermes and reports which ones still hold. Run at server
startup (warn-only — never blocks boot) and surfaced by ``agent-teams doctor``.

Add a probe here whenever the teams starts depending on a new Hermes internal,
so "did this Hermes release break us?" is answerable in one place.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger("teams.compat")


# --------------------------------------------------------------------------- #
# Probe result + registry
# --------------------------------------------------------------------------- #
@dataclass
class Probe:
    name: str
    ok: bool
    critical: bool          # True => failure breaks a user-visible feature
    detail: str = ""


@dataclass
class CompatReport:
    hermes_version: str = "?"
    probes: List[Probe] = field(default_factory=list)

    @property
    def failures(self) -> List[Probe]:
        return [p for p in self.probes if not p.ok]

    @property
    def critical_failures(self) -> List[Probe]:
        return [p for p in self.probes if not p.ok and p.critical]

    @property
    def ok(self) -> bool:
        return not self.failures


def _hermes_version() -> str:
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:
        try:
            import hermes_constants  # noqa: F401
            return getattr(hermes_constants, "__version__", "?")
        except Exception:
            return "?"


# --------------------------------------------------------------------------- #
# Individual probes. Each returns (ok, detail). Keep them cheap + import-only;
# this runs on the hot boot path and inside `doctor`.
# --------------------------------------------------------------------------- #
def _probe_symbols(module: str, names: List[str]):
    """All `names` importable from `module`?"""
    try:
        mod = importlib.import_module(module)
    except Exception as e:
        return False, f"cannot import {module}: {e}"
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        return False, f"{module} missing {missing}"
    return True, f"{module}: {', '.join(names)}"


def _probe_browser_internals():
    # browser_gui_tools.py rides these three private fns for ALL GUI browser tools.
    return _probe_symbols("tools.browser_tool",
                          ["_run_browser_command", "_last_session_key", "_browser_eval"])


def _probe_provider_model_seed():
    # model_config.py imports this private dict to seed the setup dropdown.
    return _probe_symbols("hermes_cli.setup", ["_DEFAULT_PROVIDER_MODELS"])


def _probe_web_registry():
    # web_crawl4ai.py asks Hermes' resolver before overriding web tools.
    return _probe_symbols("agent.web_search_registry",
                          ["get_active_search_provider", "get_active_extract_provider"])


def _probe_provider_registry():
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        n = len(PROVIDER_REGISTRY)
        return (n > 0), f"PROVIDER_REGISTRY: {n} providers"
    except Exception as e:
        return False, f"PROVIDER_REGISTRY unavailable: {e}"


def _probe_gateway_bridge():
    # gateway_config.py + gateway_bridge.py ride these to reuse Hermes' messaging
    # gateway: the platform catalogue (_all_platforms / _platform_status), the
    # static PLATFORMS map, and the one-shot send used to forward inbox questions.
    # A rename here disables the human-inbox→chat bridge (outbound + reply-back),
    # not core agent operation — hence non-critical.
    ok1, d1 = _probe_symbols("hermes_cli.gateway", ["_all_platforms", "_platform_status"])
    ok2, d2 = _probe_symbols("hermes_cli.platforms", ["PLATFORMS"])
    ok3, d3 = _probe_symbols("tools.send_message_tool", ["send_message_tool"])
    if ok1 and ok2 and ok3:
        return True, "gateway send + platform catalogue present"
    return False, "; ".join(d for ok, d in ((ok1, d1), (ok2, d2), (ok3, d3)) if not ok)


def _probe_aiagent_tool_injection():
    # agent.py injects teams tools by mutating an AIAgent's `.tools` /
    # `.valid_tool_names` / `._tool_use_enforcement` after construction. Hermes
    # sets these during agent init (agent/agent_init.py), not in AIAgent.__init__,
    # so source-scan that module for the assignments — a rename surfaces loudly.
    try:
        mod = importlib.import_module("agent.agent_init")
    except Exception as e:
        return False, f"agent.agent_init unimportable: {e}"
    try:
        src = inspect.getsource(mod)
    except (OSError, TypeError):
        return True, "agent_init imported (source unavailable — attrs unverified)"
    needed = ["valid_tool_names", "_tool_use_enforcement", ".tools"]
    missing = [n for n in needed if f"{n} =" not in src and f"{n}=" not in src]
    if missing:
        return False, f"agent_init no longer assigns {missing} (tool injection at risk)"
    return True, "tool-injection attrs assigned by agent_init"


def _probe_aiagent_callbacks():
    # Phase-2 reuse target: pass live callbacks to the ctor instead of setattr.
    try:
        from run_agent import AIAgent
        params = inspect.signature(AIAgent.__init__).parameters
        cbs = [p for p in params if p.endswith("_callback")]
        return (len(cbs) > 0), f"{len(cbs)} *_callback ctor params"
    except Exception as e:
        return False, f"AIAgent signature unavailable: {e}"


def _probe_disabled_toolsets():
    # DISABLED_TOOLSETS is a hand-maintained denylist of Hermes toolset NAMES.
    # Validate every entry still exists in the live registry — a renamed toolset
    # becomes a dead no-op (and a brand-new heavyweight toolset ships enabled).
    try:
        from toolsets import get_all_toolsets
        from teams_server.config import DISABLED_TOOLSETS
        known = set(get_all_toolsets().keys())
    except Exception as e:
        return False, f"toolset registry unavailable: {e}"
    stale = [t for t in DISABLED_TOOLSETS if t not in known]
    new_since = sorted(known - known.intersection(_KNOWN_TOOLSETS_BASELINE)) if _KNOWN_TOOLSETS_BASELINE else []
    detail = f"{len(known)} toolsets"
    if stale:
        detail += f"; DISABLED_TOOLSETS has unknown names {stale}"
    if new_since:
        detail += f"; new since baseline {new_since}"
    return (not stale), detail


# Toolset names known at the time this baseline was last reviewed (hermes-agent
# 0.15.2). New names that appear in a future Hermes are surfaced (they ship
# ENABLED unless added to DISABLED_TOOLSETS). Update intentionally when you
# reconcile the denylist against a new Hermes release.
_KNOWN_TOOLSETS_BASELINE: set = {
    "browser", "clarify", "code_execution", "computer_use", "context_engine",
    "cronjob", "debugging", "delegation", "discord", "discord_admin", "feishu_doc",
    "feishu_drive", "file", "hermes-acp", "hermes-api-server", "hermes-bluebubbles",
    "hermes-cli", "hermes-cron", "hermes-dingtalk", "hermes-discord", "hermes-email",
    "hermes-feishu", "hermes-gateway", "hermes-homeassistant", "hermes-matrix",
    "hermes-mattermost", "hermes-qqbot", "hermes-signal", "hermes-slack", "hermes-sms",
    "hermes-telegram", "hermes-webhook", "hermes-wecom", "hermes-wecom-callback",
    "hermes-weixin", "hermes-whatsapp", "hermes-yuanbao", "homeassistant", "image_gen",
    "kanban", "memory", "messaging", "moa", "safe", "search", "session_search",
    "skills", "spotify", "terminal", "todo", "tts", "video", "video_gen", "vision",
    "web", "x_search", "yuanbao", "browser-cdp",
}


_PROBES: List[tuple] = [
    # (name, fn, critical)
    ("browser_gui_internals", _probe_browser_internals, True),
    ("aiagent_tool_injection", _probe_aiagent_tool_injection, True),
    ("provider_registry", _probe_provider_registry, True),
    ("gateway_bridge", _probe_gateway_bridge, False),
    ("web_search_registry", _probe_web_registry, False),
    ("provider_model_seed", _probe_provider_model_seed, False),
    ("aiagent_callbacks", _probe_aiagent_callbacks, False),
    ("disabled_toolsets", _probe_disabled_toolsets, False),
]


def run_self_check() -> CompatReport:
    """Probe every fragile Hermes seam. Never raises."""
    report = CompatReport(hermes_version=_hermes_version())
    for name, fn, critical in _PROBES:
        try:
            ok, detail = fn()
        except Exception as e:  # a probe must never break boot
            ok, detail = False, f"probe error: {e}"
        report.probes.append(Probe(name=name, ok=bool(ok), critical=critical, detail=detail))
    return report


def log_self_check(report: Optional[CompatReport] = None) -> CompatReport:
    """Run (or take) a report and log it — LOUD on failure, quiet on success."""
    report = report or run_self_check()
    if report.ok:
        log.info("Hermes compat OK (hermes-agent %s) — %d/%d seams verified",
                 report.hermes_version, len(report.probes), len(report.probes))
        return report
    crit = report.critical_failures
    bar = "!" * 60
    log.warning("\n%s\nHERMES COMPAT DRIFT — hermes-agent %s\n%s", bar, report.hermes_version, bar)
    for p in report.failures:
        tag = "CRITICAL" if p.critical else "warning "
        log.warning("  [%s] %s: %s", tag, p.name, p.detail)
    if crit:
        log.warning("  → %d CRITICAL seam(s) broke: a Hermes update likely moved an "
                    "internal API the teams depends on. Features above may silently "
                    "fail until reconciled. See teams_server/hermes_compat.py.", len(crit))
    log.warning("%s", bar)
    return report
