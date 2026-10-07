#!/usr/bin/env python3
"""Regression tests for dashboard chat classification and rendering."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = PROJECT_ROOT / "dashboard" / "advanced.html"


def _source() -> str:
    # The dashboard intentionally contains a raw NUL separator elsewhere; UTF-8
    # decoding keeps it intact and avoids grep's binary-file behavior.
    return DASHBOARD.read_bytes().decode("utf-8")


def _helper_block(source: str) -> str:
    start = source.index("function parseScheduledWakeup(content) {")
    end = source.index("/* ===== LIVE EXECUTION TRACE", start)
    return source[start:end]


def _markdown_function(source: str) -> str:
    start = source.index("function mdToHtml(raw) {")
    end = source.index("\nfunction ", start + 1)
    return source[start:end]


def _run_node(script: str) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for inline dashboard JavaScript tests")
    result = subprocess.run(
        [node], input=script, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_inline_javascript_is_syntactically_valid():
    source = _source()
    scripts = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", source, re.I)
    assert scripts
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for inline dashboard JavaScript tests")
    result = subprocess.run(
        [node, "--check"],
        input="\n".join(scripts),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_message_classifier_and_renderers_use_production_helpers():
    source = _source()
    cron = """You have 1 new message(s) to process:

--[1] from cron --
[SCHEDULED WAKE-UP — cron '/3*' fired at 2026-08-26 13:46; scheduled 11h52m ago; you run 24/7 and nobody may be watching]
This is an automated wake-up you or your operator scheduled.
Carry out the instruction, then end your turn (do not loop):

Search for **new leads** and write `inbox/web-leads.xlsx`."""
    compact_tool = (
        '🛠️ execute_code({"code":"print(1)"}) → '
        '{"status":"success","output":"ok"}'
    )
    prelude = r"""
function esc(value) {
    return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escAttr(value) { return esc(value).replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }
function icon() { return ''; }
"""
    script = (
        prelude
        + _markdown_function(source)
        + "\n"
        + _helper_block(source)
        + f"""
const cron = {json.dumps(cron)};
const compactText = {json.dumps(compact_tool)};
const wakeup = parseScheduledWakeup(cron);
const compact = parseCompactToolRecord(compactText);
const automationHtml = formatMessageContent({{role: 'user', content: cron}});
const toolHtml = formatMessageContent({{role: 'assistant', content: compactText}});
const escapedHtml = mdToHtml('<img src=x onerror=alert(1)> **safe**');
console.log(JSON.stringify({{
  wakeup,
  cronKind: classifyMessage({{role: 'user', content: cron}}),
  cronFalsePositive: classifyMessage({{role: 'user', content: 'Please discuss cron scheduling'}}),
  assistantKind: classifyMessage({{role: 'assistant', content: 'Done'}}),
  nativeToolKind: classifyMessage({{role: 'tool', content: 'result'}}),
  compact,
  automationHtml,
  toolHtml,
  escapedHtml
}}));
"""
    )
    data = _run_node(script)

    assert data["cronKind"] == "automation"
    assert data["cronFalsePositive"] == "user"
    assert data["assistantKind"] == "assistant"
    assert data["nativeToolKind"] == "tool-result"
    assert data["wakeup"]["schedule"] == "/3*"
    assert data["wakeup"]["instruction"].startswith("Search for")
    assert data["compact"]["name"] == "execute_code"
    assert "Scheduled wake-up" not in data["automationHtml"]  # label belongs to header
    assert "automation-instruction" in data["automationHtml"]
    assert "Automation context" in data["automationHtml"]
    assert "tool-disclosure" in data["toolHtml"]
    assert "execute_code" in data["toolHtml"]
    assert "complete" in data["toolHtml"]
    assert "&lt;img" in data["escapedHtml"]
    assert "<img" not in data["escapedHtml"]


def test_mixed_cron_batch_preserves_other_work():
    source = _source()
    mixed = """You have 2 new message(s) to process:

--[1] from supervisor --
Review the latest report.

--[2] from cron --
[SCHEDULED WAKE-UP — cron '0 9 * * *' fired at 2026-08-26 09:00; scheduled 1d ago; you run 24/7 and nobody may be watching]
Automation context.
Carry out the instruction, then end your turn (do not loop):

Run the daily sweep."""
    script = (
        "function esc(s){return String(s ?? '')} function escAttr(s){return esc(s)} "
        "function icon(){return ''} function mdToHtml(s){return esc(s)}\n"
        + _helper_block(source)
        + f"\nconsole.log(JSON.stringify(parseScheduledWakeup({json.dumps(mixed)})));"
    )
    data = _run_node(script)
    assert "from supervisor" in data["leading"]
    assert "Review the latest report" in data["leading"]
    assert data["instruction"] == "Run the daily sweep."


def test_unified_stream_order_and_reconciliation_invariants():
    source = _source()
    assert "const msgs = data.messages || [];" in source
    assert "(data.messages || []).slice().reverse()" not in source
    assert "const scrollContainer = document.getElementById('unified-chat-panel') || container;" in source
    assert "container.insertAdjacentHTML('beforeend', messageBlockHTML(msg));" in source
    assert "msgCount++;" in source
    assert "setTimeout(() => { _execRenderTimer = null; renderExecTrace(); }, 80)" in source
    assert "if (ch.dataset.k) existing.set(ch.dataset.k, ch);" in source
    assert "if (node._inner !== item.inner)" in source


def test_accessible_disclosures_and_distinct_message_styles_exist():
    source = _source()
    for kind in ("assistant", "user", "system", "tool-call", "tool-result", "automation"):
        assert f".msg-block.{kind}" in source
    assert 'class="bare tool-disclosure-toggle" aria-expanded="false"' in source
    assert 'class="bare automation-context-toggle" aria-expanded="false"' in source
    assert 'role="status" aria-live="polite"' in source
    assert ".tool-disclosure-toggle:focus-visible" in source
    assert '@media (max-width: 640px)' in source
