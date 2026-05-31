#!/usr/bin/env python3
"""
Compaction validation
======================
Proves the auto-compaction wiring actually fires and that the swarm-side
rotation handling is correct. Earlier this feature was untested: real runs
never grew a session past the threshold, so we never observed a rotation.

Three layers, fast → slow:
  1. CONFIG    — write_agent_hermes_config emits valid, in-range compression
                 settings (>= Hermes MINIMUM_CONTEXT_LENGTH).
  2. TRIGGER   — drive Hermes' own ContextCompressor with our exact config
                 values and a synthetic conversation; assert it compacts above
                 the threshold and stays quiet below it. No LLM needed.
  3. ROTATION  — drive AgentDaemon._persist_session_id_if_rotated with a faked
                 rotated session_id; assert it persists the new id so a restart
                 resumes from the compacted child session.

Run:
    PYTHONPATH=/Users/pradhyun/.hermes/hermes-agent \
      python3 tests/test_compaction.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Make the swarm package + Hermes importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
HERMES_PATH = "/Users/pradhyun/.hermes/hermes-agent"
if HERMES_PATH not in sys.path:
    sys.path.insert(0, HERMES_PATH)

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. CONFIG — the written config.yaml is valid and enables compression
# ---------------------------------------------------------------------------
def test_config():
    print("\n[1] CONFIG: write_agent_hermes_config")
    import yaml
    from swarm_server.config import (
        write_agent_hermes_config,
        AGENT_CONTEXT_WINDOW,
        COMPRESSION_THRESHOLD,
        COMPRESSION_ENABLED,
    )

    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        write_agent_hermes_config(home)
        cfg = yaml.safe_load((home / "config.yaml").read_text())

        check("config.yaml written", (home / "config.yaml").exists())
        check("compression enabled",
              cfg.get("compression", {}).get("enabled") == COMPRESSION_ENABLED)
        check("threshold matches constant",
              cfg["compression"]["threshold"] == COMPRESSION_THRESHOLD)
        check("context_length set on model",
              cfg["model"]["context_length"] == AGENT_CONTEXT_WINDOW)

        # Hermes refuses to init below MINIMUM_CONTEXT_LENGTH (64000).
        try:
            from agent.context_compressor import MINIMUM_CONTEXT_LENGTH
        except Exception:
            MINIMUM_CONTEXT_LENGTH = 64000
        check("context_length >= Hermes minimum",
              AGENT_CONTEXT_WINDOW >= MINIMUM_CONTEXT_LENGTH,
              f"{AGENT_CONTEXT_WINDOW} < {MINIMUM_CONTEXT_LENGTH}")

        # A merge-safe second write must not duplicate/clobber the section.
        write_agent_hermes_config(home)
        cfg2 = yaml.safe_load((home / "config.yaml").read_text())
        check("re-write is idempotent", cfg2["compression"] == cfg["compression"])


# ---------------------------------------------------------------------------
# 2. TRIGGER — Hermes' compressor fires at our threshold (deterministic, no LLM)
# ---------------------------------------------------------------------------
def test_trigger():
    print("\n[2] TRIGGER: should_compress() at the configured threshold")
    from agent.context_compressor import ContextCompressor
    from agent.model_metadata import estimate_messages_tokens_rough
    from swarm_server.config import (
        AGENT_CONTEXT_WINDOW,
        COMPRESSION_THRESHOLD,
        COMPRESSION_PROTECT_FIRST_N,
        COMPRESSION_PROTECT_LAST_N,
        COMPRESSION_TARGET_RATIO,
    )

    # config_context_length pins the window so the compressor doesn't try to
    # probe the (proxy-hidden) model — same as our written config.yaml does.
    comp = ContextCompressor(
        model="litellm-model",
        threshold_percent=COMPRESSION_THRESHOLD,
        protect_first_n=COMPRESSION_PROTECT_FIRST_N,
        protect_last_n=COMPRESSION_PROTECT_LAST_N,
        summary_target_ratio=COMPRESSION_TARGET_RATIO,
        quiet_mode=True,
        config_context_length=AGENT_CONTEXT_WINDOW,
    )
    threshold_tokens = comp.threshold_tokens
    check("threshold_tokens ~= window*threshold",
          abs(threshold_tokens - AGENT_CONTEXT_WINDOW * COMPRESSION_THRESHOLD) < 2000,
          f"got {threshold_tokens}")

    # Below threshold: a short conversation must NOT compact.
    small = [{"role": "user", "content": "hello there"}]
    small_tokens = estimate_messages_tokens_rough(small)
    check("small conversation under threshold",
          small_tokens < threshold_tokens)
    check("should_compress() False below threshold",
          comp.should_compress(small_tokens) is False)

    # Above threshold: build a conversation whose rough estimate clears it.
    # ~4 chars/token, so we need > threshold_tokens*4 chars of content.
    chars_needed = int(threshold_tokens * 4 * 1.2)
    big_blob = "x " * (chars_needed // 2)
    big = [{"role": "user", "content": big_blob}]
    big_tokens = estimate_messages_tokens_rough(big)
    check("large conversation over threshold",
          big_tokens >= threshold_tokens,
          f"{big_tokens} < {threshold_tokens}")
    check("should_compress() True above threshold",
          comp.should_compress(big_tokens) is True)


# ---------------------------------------------------------------------------
# 3. ROTATION — the swarm persists a rotated session_id (compaction aftermath)
# ---------------------------------------------------------------------------
def test_rotation_persist():
    print("\n[3] ROTATION: _persist_session_id_if_rotated")
    # Heavy imports (FastAPI/Hermes) live inside agent.py; importing it is fine.
    from swarm_server import agent as agent_mod
    from swarm_server.agent import AgentDaemon

    # Capture the config that would be persisted instead of touching disk.
    saved = {}
    orig_save = agent_mod.save_agent_config
    agent_mod.save_agent_config = lambda name, cfg: saved.update({name: dict(cfg)})

    # Avoid emitting real monitoring/broadcast side effects.
    orig_log = agent_mod.monitor_db.log_event
    agent_mod.monitor_db.log_event = lambda *a, **k: None
    orig_bcast = agent_mod._broadcast
    agent_mod._broadcast = lambda *a, **k: None

    try:
        with tempfile.TemporaryDirectory() as d:
            # Point the workspace root at a temp dir so queue/home init is isolated.
            orig_ws_root = agent_mod._derive_workspace_path
            agent_mod._derive_workspace_path = lambda team, name: Path(d) / team / name

            cfg = {"team_id": "t", "session_id": "agent-master-session-v1",
                   "allowed_peers": [], "role_soul": "x"}
            daemon = AgentDaemon("agent", cfg)

            # Simulate a Hermes auto-compaction that rotated the live session id.
            class _FakeAI:
                session_id = "agent-master-session-v1-child-0001"
            daemon._ai_agent = _FakeAI()

            daemon._persist_session_id_if_rotated()

            check("rotated id mirrored into cfg",
                  daemon.cfg["session_id"] == "agent-master-session-v1-child-0001")
            check("rotation persisted via save_agent_config",
                  saved.get("agent", {}).get("session_id")
                  == "agent-master-session-v1-child-0001")

            # No rotation → no-op (no extra save, id unchanged).
            saved.clear()
            daemon._persist_session_id_if_rotated()
            check("no-op when id unchanged", saved == {})

            agent_mod._derive_workspace_path = orig_ws_root
    finally:
        agent_mod.save_agent_config = orig_save
        agent_mod.monitor_db.log_event = orig_log
        agent_mod._broadcast = orig_bcast


if __name__ == "__main__":
    print("=" * 60)
    print("  Compaction validation")
    print("=" * 60)
    test_config()
    try:
        test_trigger()
    except Exception as e:
        failed += 1
        print(f"  FAIL  trigger layer raised: {e}")
    try:
        test_rotation_persist()
    except Exception as e:
        failed += 1
        print(f"  FAIL  rotation layer raised: {e}")

    print("\n" + "=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
