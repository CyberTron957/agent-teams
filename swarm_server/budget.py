"""Per-team daily spend guardrails with auto-pause.

A 24/7 swarm on a paid API can quietly burn money overnight. This module meters
each team's spend for the current UTC day and, when a team crosses its cap, makes
``is_blocked(team_id)`` return True — the AgentDaemon sweep loop then HOLDS that
team's work (exactly like the existing infra-outage hold: tasks are kept, not
failed) and skips crons/heartbeats until the cap is raised, an override is set,
or the day rolls over at 00:00 UTC.

Design notes:
  * One process-wide singleton, one lock. ``record_turn`` is called from agent
    worker threads; ``is_blocked`` from the asyncio sweep loop; the endpoints
    from request threads — all serialized by ``self._lock``.
  * ``is_blocked`` is O(1) dict lookups (no DB, no JSON parse) so it's cheap to
    call before every sweep. Team limits are read through a ~15s TTL cache.
  * Pricing uses model_config.estimate_cost_usd, which returns None for unknown
    models. Those turns can't grow the USD meter, so a token cap (daily_tokens)
    is the enforceable fallback; ``unpriced_turns`` surfaces the gap in the UI.
  * The Architect (master) agent does not log token_usage, so it is outside the
    meter by design.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Set

log = logging.getLogger("swarm.budget")

_LIMITS_TTL_SECONDS = 15.0


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _next_utc_midnight(now: float) -> float:
    # Epoch 0 is itself a UTC midnight, so day boundaries align to 86400s.
    return (int(now // 86400) + 1) * 86400


class TeamBudgetTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = _utc_day()
        # team_id -> {"usd": float, "tokens": int, "unpriced_turns": int, "turns": int}
        self._spend: Dict[str, Dict[str, Any]] = {}
        self._exceeded: Set[str] = set()   # one-shot broadcast latch
        self._override: Set[str] = set()    # "resume anyway" until rollover
        self._limits_cache: Dict[str, Dict[str, float]] = {}
        self._limits_cache_at = 0.0

    # -- internal helpers (call under the lock) -----------------------------
    def _bucket(self, team_id: str) -> Dict[str, Any]:
        return self._spend.setdefault(
            team_id, {"usd": 0.0, "tokens": 0, "unpriced_turns": 0, "turns": 0})

    def _limits(self, team_id: str) -> Dict[str, float]:
        now = time.time()
        if now - self._limits_cache_at > _LIMITS_TTL_SECONDS:
            self._limits_cache = {}
            self._limits_cache_at = now
        if team_id not in self._limits_cache:
            try:
                from swarm_server.config import get_team_budget, load_agents_config
                self._limits_cache[team_id] = get_team_budget(load_agents_config(), team_id)
            except Exception:
                self._limits_cache[team_id] = {"daily_usd": 0.0, "daily_tokens": 0}
        return self._limits_cache[team_id]

    def _over(self, team_id: str) -> bool:
        lim = self._limits(team_id)
        b = self._spend.get(team_id)
        if not b:
            return False
        return ((lim["daily_usd"] > 0 and b["usd"] >= lim["daily_usd"])
                or (lim["daily_tokens"] > 0 and b["tokens"] >= lim["daily_tokens"]))

    def _maybe_rollover(self) -> list:
        """If the UTC day changed, reset all counters. Returns a list of
        (event, team_id) to broadcast AFTER the lock is released."""
        today = _utc_day()
        if today == self._day:
            return []
        resumed = [("budget_resumed", t) for t in self._exceeded]
        self._day = today
        self._spend.clear()
        self._exceeded.clear()
        self._override.clear()
        self._limits_cache.clear()
        self._limits_cache_at = 0.0
        return resumed

    # -- public API ---------------------------------------------------------
    def record_turn(self, team_id: str, model: str,
                    t_in: int, t_out: int, t_cache: int,
                    provider: Optional[str] = None,
                    base_url: Optional[str] = None) -> None:
        """Meter one completed turn. Latches + broadcasts once when a team first
        crosses its cap. provider/base_url select native (Hermes) vs proxy (table)
        pricing."""
        from swarm_server.model_config import estimate_cost_usd
        events = []
        newly_exceeded = False
        with self._lock:
            events += self._maybe_rollover()
            b = self._bucket(team_id)
            b["turns"] += 1
            b["tokens"] += int(t_in) + int(t_out) + int(t_cache)
            cost = estimate_cost_usd(model, t_in, t_out, t_cache,
                                     provider=provider, base_url=base_url)
            if cost is None:
                b["unpriced_turns"] += 1
            else:
                b["usd"] += cost
            if (team_id not in self._exceeded and team_id not in self._override
                    and self._over(team_id)):
                self._exceeded.add(team_id)
                newly_exceeded = True
                events.append(("budget_exceeded", team_id))
                snap = self._status_locked(team_id)
        for evt, tid in events:
            self._emit(evt, tid, reason="rollover" if evt == "budget_resumed" else None)
        if newly_exceeded:
            log.warning("[budget] team '%s' hit its daily cap (%.4f USD / %d tokens)",
                        team_id, snap["spent_usd"], snap["spent_tokens"])

    def is_blocked(self, team_id: str) -> bool:
        """Cheap gate for the sweep loop: True iff the team is over its cap and
        not overridden for today."""
        with self._lock:
            resumed = self._maybe_rollover()
            blocked = team_id not in self._override and self._over(team_id)
        for evt, tid in resumed:
            self._emit(evt, tid, reason="rollover")
        return blocked

    def override_today(self, team_id: str) -> None:
        """Resume a paused team for the rest of the UTC day (manual 'resume
        anyway')."""
        with self._lock:
            self._maybe_rollover()
            self._override.add(team_id)
            was = team_id in self._exceeded
            self._exceeded.discard(team_id)
        if was:
            self._emit("budget_resumed", team_id, reason="override")

    def set_limits(self, team_id: str, daily_usd: float, daily_tokens: int) -> None:
        """Refresh the cached caps immediately (so the gate reacts without
        waiting for the TTL) and un-latch if the team is no longer over."""
        resumed = False
        with self._lock:
            self._limits_cache[team_id] = {
                "daily_usd": max(0.0, float(daily_usd or 0)),
                "daily_tokens": max(0, int(daily_tokens or 0)),
            }
            if team_id in self._exceeded and not self._over(team_id):
                self._exceeded.discard(team_id)
                resumed = True
        if resumed:
            self._emit("budget_resumed", team_id, reason="limit_raised")

    def _status_locked(self, team_id: str) -> Dict[str, Any]:
        lim = self._limits(team_id)
        b = self._spend.get(team_id) or {"usd": 0.0, "tokens": 0,
                                          "unpriced_turns": 0, "turns": 0}
        over = self._over(team_id)
        override = team_id in self._override
        rem_usd = (max(0.0, lim["daily_usd"] - b["usd"]) if lim["daily_usd"] > 0 else None)
        return {
            "team_id": team_id,
            "day": self._day,
            "limit_usd": round(lim["daily_usd"], 4),
            "limit_tokens": int(lim["daily_tokens"]),
            "spent_usd": round(b["usd"], 4),
            "spent_tokens": int(b["tokens"]),
            "remaining_usd": (round(rem_usd, 4) if rem_usd is not None else None),
            "unpriced_turns": int(b["unpriced_turns"]),
            "turns": int(b["turns"]),
            "over_budget": over,
            "blocked": over and not override,
            "override_active": override,
            "resets_at_ts": _next_utc_midnight(time.time()),
        }

    def status(self, team_id: str) -> Dict[str, Any]:
        with self._lock:
            self._maybe_rollover()
            return self._status_locked(team_id)

    def rebuild(self, db, cfg: Dict[str, Any]) -> None:
        """Re-sum today's spend from monitoring.db so a mid-day restart keeps
        the meter. Only modern per-turn rows (those carrying turn_input_tokens)
        are counted; legacy cumulative rows are pre-deploy history."""
        import json

        from swarm_server.model_config import estimate_cost_usd
        agent_team = {name: a.get("team_id", "default")
                      for name, a in (cfg.get("agents") or {}).items()}
        cfg_models = {name: (a.get("model") or "")
                      for name, a in (cfg.get("agents") or {}).items()}
        since = _next_utc_midnight(time.time()) - 86400  # today's UTC midnight
        try:
            events = db.get_token_usage_events(since, limit=100000)
        except Exception as e:
            log.warning("[budget] rebuild skipped (%s)", e)
            return
        with self._lock:
            self._day = _utc_day()
            self._spend.clear()
            for ev in events:
                try:
                    data = json.loads(ev.get("data") or "{}")
                except Exception:
                    continue
                if "turn_input_tokens" not in data:
                    continue  # legacy cumulative row — skip
                agent = ev.get("agent_name") or "?"
                team_id = agent_team.get(agent) or ev.get("team_id") or "default"
                model = (data.get("model") or cfg_models.get(agent) or "").strip()
                provider = (data.get("provider") or "").strip() or None
                t_in = int(data.get("turn_input_tokens") or 0)
                t_out = int(data.get("turn_output_tokens") or 0)
                t_cache = int(data.get("turn_cache_read_tokens") or 0)
                b = self._bucket(team_id)
                b["turns"] += 1
                b["tokens"] += t_in + t_out + t_cache
                cost = estimate_cost_usd(model, t_in, t_out, t_cache, provider=provider)
                if cost is None:
                    b["unpriced_turns"] += 1
                else:
                    b["usd"] += cost
            # Latch any team already over its cap so the first sweep holds it.
            for team_id in list(self._spend):
                if self._over(team_id):
                    self._exceeded.add(team_id)
        log.info("[budget] rebuilt today's meter for %d team(s)", len(self._spend))

    # -- broadcasting -------------------------------------------------------
    def _emit(self, event: str, team_id: str, reason: str = None) -> None:
        payload = self.status(team_id)
        if reason:
            payload["reason"] = reason
        payload["timestamp"] = time.time()
        try:
            from swarm_server.websocket import _broadcast
            _broadcast(event, payload)
        except Exception as e:
            log.debug("[budget] broadcast %s failed: %s", event, e)
        try:
            from swarm_server.monitoring import monitor_db
            monitor_db.log_event(team_id, event, data={
                "spent_usd": payload["spent_usd"], "limit_usd": payload["limit_usd"],
                "spent_tokens": payload["spent_tokens"], "reason": reason or "",
            })
        except Exception:
            pass


budget_tracker = TeamBudgetTracker()
