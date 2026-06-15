"""Hermes compatibility self-check tests.

These run against the REAL installed Hermes (the same venv the server uses), so a
Hermes upgrade that moves an internal API the swarm depends on fails CI here —
the whole point of the self-check is to make that drift loud, not silent.
"""

import pytest

from swarm_server.hermes_compat import (
    run_self_check,
    log_self_check,
    CompatReport,
    Probe,
)


def test_self_check_runs_and_reports_version():
    report = run_self_check()
    assert isinstance(report, CompatReport)
    assert report.probes, "self-check produced no probes"
    # version resolvable from importlib.metadata in a real install
    assert report.hermes_version and report.hermes_version != "?"


def test_no_critical_seam_drift_against_installed_hermes():
    """The internal Hermes APIs the swarm builds over must still hold.

    If this fails after a `pip install -U hermes-agent`, a load-bearing seam
    moved — reconcile swarm_server/hermes_compat.py (and the dependent feature)
    before shipping, rather than letting the feature silently no-op.
    """
    report = run_self_check()
    critical = report.critical_failures
    assert not critical, "Hermes critical seam drift: " + "; ".join(
        f"{p.name}: {p.detail}" for p in critical
    )


def test_all_seams_currently_green():
    # Pinned-version expectation: on the tested Hermes, every seam (incl.
    # non-critical) verifies. A non-critical failure is a heads-up, not a block.
    report = run_self_check()
    failing = [p.name for p in report.failures]
    assert not failing, f"unexpected seam drift (non-critical incl.): {failing}"


def test_log_self_check_never_raises():
    # Boot path calls this; it must be exception-proof even on a broken report.
    broken = CompatReport(hermes_version="9.9.9", probes=[
        Probe(name="fake", ok=False, critical=True, detail="synthetic"),
    ])
    # Should log loudly and return the same report, not raise.
    assert log_self_check(broken) is broken


def test_disabled_toolsets_have_no_stale_entries():
    # Every name in DISABLED_TOOLSETS must exist in Hermes' live registry, else
    # it's a dead no-op that silently stops disabling anything.
    from toolsets import get_all_toolsets
    from swarm_server.config import DISABLED_TOOLSETS

    known = set(get_all_toolsets().keys())
    stale = [t for t in DISABLED_TOOLSETS if t not in known]
    assert not stale, f"DISABLED_TOOLSETS references unknown toolsets: {stale}"
