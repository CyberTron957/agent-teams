#!/usr/bin/env python3
"""Unit tests for wake-payload aging at history replay
(prompts.age_stale_wake_payloads). Pure function — no LLM, no Hermes.

A worker's user-role history is dominated by already-processed batch
envelopes ("You have N new message(s)" — cron instructions, peer
deliveries). The pass trims envelopes past the quantized cutoff to
sender headers + a short payload preview, keeping the recent working
set verbatim and the request prefix byte-stable between boundary moves.

Run:  pytest tests/test_wake_payload_aging.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server.prompts import (  # noqa: E402
    STALE_CONTEXT_NOTE,
    WAKE_PAYLOAD_ELIDED_MARK,
    _age_one_batch_envelope,
    age_stale_sweep_payloads,
    age_stale_wake_payloads,
)


def _wake(sender="cron", payload="PAYLOAD " + "x" * 900, n=1, again=False):
    """One stored batch envelope as it looks after strip_stale_live_context."""
    note = " · SEEN BEFORE — redelivery" if again else ""
    return (
        f"{STALE_CONTEXT_NOTE}\n\n"
        f"You have {n} new message(s) to process:\n\n"
        f"--- [1] from {sender}{note} ---\n{payload}\n\n"
    )


def _multi_wake():
    return (
        f"{STALE_CONTEXT_NOTE}\n\n"
        "You have 2 new message(s) to process:\n\n"
        "--- [1] from orchestrator ---\n[TASK · id=ab12 · from orchestrator]\n"
        + "do the thing " + "y" * 800 + "\n\n"
        "--- [2] from peer-2 ---\nshort note\n\n"
    )


def _hist(n_old=30, tail=10):
    """n_old wake envelopes, then `tail` small recent messages."""
    msgs = []
    for i in range(n_old):
        msgs.append({"role": "user", "content": _wake(payload=f"instr {i} " + "x" * 900)})
        msgs.append({"role": "assistant", "content": f"worked on {i}"})
    for i in range(tail):
        msgs.append({"role": "user", "content": f"recent {i}"})
    return msgs


def test_old_envelopes_stubbed_headers_and_preview_survive():
    msgs = _hist()
    out = age_stale_wake_payloads(msgs, keep_recent=10, min_chars=400,
                                quantum=20, preview_chars=60)
    aged = [m for m in out if WAKE_PAYLOAD_ELIDED_MARK in str(m.get("content", ""))]
    assert aged, "expected at least one aged envelope"
    m = aged[0]
    assert m["role"] == "user"
    assert "from cron" in m["content"]            # sender still visible
    assert "You have 1 new message(s)" in m["content"]
    assert "instr" in m["content"]               # preview keeps the payload head
    assert "x" * 900 not in m["content"]         # bulk is gone
    assert len(m["content"]) < 800


def test_protected_tail_untouched():
    msgs = _hist(n_old=10, tail=0)
    msgs.append({"role": "user", "content": _wake(payload="fresh " + "z" * 900)})
    out = age_stale_wake_payloads(msgs, keep_recent=5, min_chars=400,
                                quantum=4, preview_chars=60)
    assert WAKE_PAYLOAD_ELIDED_MARK not in out[-1]["content"]
    assert "z" * 900 in out[-1]["content"]


def test_non_batch_user_messages_untouched():
    """Compaction summaries, plain human text, and RESULT data payloads are
    NOT wake envelopes — they keep their full body."""
    msgs = [
        {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] " + "c" * 3000},
        {"role": "user", "content": "operator: please redo the report " + "d" * 800},
    ] * 30 + [{"role": "user", "content": "tail"}]
    out = age_stale_wake_payloads(msgs, keep_recent=4, min_chars=400,
                                quantum=4, preview_chars=60)
    assert out is msgs  # nothing eligible -> original object, cheap no-op


def test_multi_section_envelope_each_section_aged():
    content = _multi_wake()
    aged = _age_one_batch_envelope(content, preview_chars=60)
    assert "--- [1] from orchestrator ---" in aged
    assert "--- [2] from peer-2 ---" in aged
    assert "[TASK · id=ab12" in aged              # bracket tag survives preview
    assert "y" * 800 not in aged
    assert WAKE_PAYLOAD_ELIDED_MARK in aged
    assert "short note" in aged                  # short payloads kept whole
    assert aged.count(WAKE_PAYLOAD_ELIDED_MARK) == 1


def test_idempotent():
    msgs = _hist()
    once = age_stale_wake_payloads(msgs, keep_recent=10, min_chars=400,
                                 quantum=20, preview_chars=60)
    twice = age_stale_wake_payloads(once, keep_recent=10, min_chars=400,
                                  quantum=20, preview_chars=60)
    assert twice is once


def test_quantized_boundary_stable_across_growth():
    """Appending fewer than `quantum` messages must not move the cutoff —
    the aged prefix stays byte-identical (prompt-cache friendly)."""
    base = _hist(n_old=30, tail=10)  # 70 messages
    kw = dict(keep_recent=10, min_chars=400, quantum=20, preview_chars=60)
    out1 = age_stale_wake_payloads(list(base), **kw)
    grown = list(base) + [{"role": "assistant", "content": f"new {i}"} for i in range(6)]
    out2 = age_stale_wake_payloads(grown, **kw)
    # cutoff = (70-10)//20*20 = 60 both ways
    assert [m["content"] for m in out2[:60]] == [m["content"] for m in out1[:60]]


def test_short_history_returned_unchanged():
    msgs = _hist(n_old=3, tail=2)
    out = age_stale_wake_payloads(msgs, keep_recent=40, min_chars=400,
                                quantum=20, preview_chars=60)
    assert out is msgs


def test_small_envelopes_under_min_chars_untouched():
    msgs = [
        {"role": "user", "content": _wake(payload="hi", n=1)}
        for _ in range(30)
    ] + [{"role": "user", "content": "tail"}]
    out = age_stale_wake_payloads(msgs, keep_recent=4, min_chars=400,
                                quantum=4, preview_chars=60)
    assert out is msgs


# ── sweep-payload quantization ─────────────────────────────────────────────

def _sweep(n):
    from teams_server.prompts import SWEEP_PAYLOAD_MARKER
    return {"role": "user", "content": f"{SWEEP_PAYLOAD_MARKER} #{n}] " + "s" * 2000}


def test_sweep_quantum_batches_boundary_moves():
    """With quantum>1, stale payloads are stubbed only in whole quantum-sized
    blocks — the stub boundary holds steady instead of creeping one message
    per sweep."""
    msgs = []
    for i in range(7):
        msgs.append(_sweep(i))
        msgs.append({"role": "assistant", "content": f"verdict {i}"})
    # 7 payloads, keep_recent=2 -> 5 stale; quantum=3 -> only first 3 stubbed
    out = age_stale_sweep_payloads(msgs, keep_recent=2, min_chars=500, quantum=3)
    elided = [m for m in out if "elided" in str(m.get("content", ""))]
    assert len(elided) == 3
    # payloads 3 and 4 (the 4th/5th stale) still intact — boundary waits for
    # a full quantum before moving
    assert "s" * 2000 in out[6]["content"]
    assert "s" * 2000 in out[8]["content"]


def test_sweep_quantum_default_is_legacy_behavior():
    msgs = [_sweep(i) for i in range(5)]
    out = age_stale_sweep_payloads(msgs, keep_recent=2, min_chars=500)
    elided = [m for m in out if "elided" in str(m.get("content", ""))]
    assert len(elided) == 3  # all but last 2 — same as before the knob
