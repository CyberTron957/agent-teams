"""Central monitoring database for events and messages."""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("teams.monitoring")


def _estimate_tokens(content: str) -> int:
    """Rough token estimate (~4 chars/token) — matches the order of magnitude of
    the char-based heuristic Hermes' ContextCompressor uses to decide when to
    compact. Good enough for dashboard usage tracking; not a billing figure."""
    if not content:
        return 0
    return max(1, len(content) // 4)


class MonitoringDB:
    """Central SQLite database for all monitoring events and message history."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   REAL    NOT NULL,
        agent_name  TEXT    NOT NULL,
        team_id     TEXT,
        event_type  TEXT    NOT NULL,
        from_agent  TEXT,
        to_agent    TEXT,
        task_id     TEXT,
        data        TEXT
    );

    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   REAL    NOT NULL,
        agent_name  TEXT    NOT NULL,
        team_id     TEXT,
        role        TEXT    NOT NULL,
        content     TEXT    NOT NULL,
        task_id     TEXT,
        tokens      INTEGER
    );

    CREATE TABLE IF NOT EXISTS digests (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        REAL    NOT NULL,
        agent_name       TEXT    NOT NULL,
        team_id          TEXT,
        summary          TEXT    NOT NULL,
        covers_to_msg_id INTEGER,
        msg_count        INTEGER,
        tokens_in        INTEGER,
        model            TEXT
    );

    CREATE TABLE IF NOT EXISTS decisions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   REAL    NOT NULL,
        agent_name  TEXT    NOT NULL,
        team_id     TEXT,
        decision    TEXT    NOT NULL
    );

    -- Correlation ledger for typed messages: every TASK/QUESTION opens a row,
    -- the matching RESULT (reply_to) answers it. Lets the system see what work
    -- is outstanding without anyone polling, and feeds the loop detector.
    CREATE TABLE IF NOT EXISTS delegations (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id       TEXT    NOT NULL,
        timestamp    REAL    NOT NULL,
        from_agent   TEXT    NOT NULL,
        to_agent     TEXT    NOT NULL,
        team_id      TEXT,
        kind         TEXT    NOT NULL,
        summary      TEXT,
        status       TEXT    NOT NULL DEFAULT 'open',
        answered_at  REAL,
        reply_to     TEXT
    );

    -- Audit trail of side-effecting actions (deploy, email, publish, payment …)
    -- keyed by an idempotency_key so two agents cannot double-do the same thing.
    CREATE TABLE IF NOT EXISTS actions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       REAL    NOT NULL,
        agent_name      TEXT    NOT NULL,
        team_id         TEXT,
        action_type     TEXT    NOT NULL,
        target          TEXT,
        idempotency_key TEXT    NOT NULL,
        outcome         TEXT,
        verified        INTEGER NOT NULL DEFAULT 0
    );

    -- Rolled-up summaries of old decisions so long-term team memory isn't lost
    -- when entries scroll past the injected window.
    CREATE TABLE IF NOT EXISTS milestones (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           REAL    NOT NULL,
        team_id             TEXT,
        summary             TEXT    NOT NULL,
        covers_to_decision  INTEGER
    );

    -- Results: outcome cards an agent deliberately publishes for the owner
    -- (via publish_result). Not file listings — business-language outcomes.
    CREATE TABLE IF NOT EXISTS results (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           REAL    NOT NULL,
        agent_name          TEXT    NOT NULL,
        team_id             TEXT,
        title               TEXT    NOT NULL,
        summary             TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_results_team ON results(team_id);

    -- Full LLM API call traces: one row per agent turn, capturing inputs sent
    -- to the model and outputs received, for the Observability dashboard.
    CREATE TABLE IF NOT EXISTS llm_traces (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         REAL    NOT NULL,
        agent_name        TEXT    NOT NULL,
        team_id           TEXT    NOT NULL,
        turn_id           TEXT    NOT NULL,
        task_ids          TEXT,
        trigger_type      TEXT    NOT NULL,
        model             TEXT,
        provider          TEXT,
        system_prompt     TEXT,
        live_context      TEXT,
        user_prompt       TEXT,
        history_json      TEXT,
        history_len       INTEGER,
        tools_count       INTEGER,
        steps_json        TEXT,
        final_response    TEXT,
        duration_seconds  REAL,
        tokens_in         INTEGER,
        tokens_out        INTEGER,
        cost_usd          REAL,
        status            TEXT    NOT NULL DEFAULT 'running'
    );

    CREATE INDEX IF NOT EXISTS idx_events_agent     ON events(agent_name);
    CREATE INDEX IF NOT EXISTS idx_events_time      ON events(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_messages_agent   ON messages(agent_name);
    CREATE INDEX IF NOT EXISTS idx_messages_time    ON messages(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_digests_agent    ON digests(agent_name, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_decisions_team   ON decisions(team_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_deleg_team       ON delegations(team_id, status, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_deleg_msg        ON delegations(msg_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_key ON actions(team_id, idempotency_key);
    CREATE INDEX IF NOT EXISTS idx_milestones_team  ON milestones(team_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_traces_agent     ON llm_traces(agent_name, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_traces_team      ON llm_traces(team_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_traces_turn      ON llm_traces(turn_id);
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()
        self._migrate_add_team_id()
        self._migrate_add_llm_traces()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        # All agent threads write this one DB concurrently; WAL + busy_timeout
        # prevent "database is locked" under load.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _migrate_add_team_id(self):
        """Add team_id column if the DB was created before the schema update."""
        with self._conn() as conn:
            cursor = conn.execute("PRAGMA table_info(events)")
            cols = [c[1] for c in cursor.fetchall()]
            if "team_id" not in cols:
                log.info("[MonitoringDB] Migrating: adding team_id to events table")
                conn.execute("ALTER TABLE events ADD COLUMN team_id TEXT")
                conn.execute("UPDATE events SET team_id = 'default'")
                conn.commit()
            # Ensure team index exists
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_team ON events(team_id)")
            conn.commit()

            cursor = conn.execute("PRAGMA table_info(messages)")
            cols = [c[1] for c in cursor.fetchall()]
            if "team_id" not in cols:
                log.info("[MonitoringDB] Migrating: adding team_id to messages table")
                conn.execute("ALTER TABLE messages ADD COLUMN team_id TEXT")
                conn.execute("UPDATE messages SET team_id = 'default'")
                conn.commit()
            if "tokens" not in cols:
                # Hermes stores NULL token_count in its own state.db, so the
                # dashboard had no usage signal. We record a rough char-based
                # estimate per message (same heuristic Hermes' compressor uses
                # for its trigger) and backfill existing rows.
                log.info("[MonitoringDB] Migrating: adding tokens to messages table")
                conn.execute("ALTER TABLE messages ADD COLUMN tokens INTEGER")
                conn.execute("UPDATE messages SET tokens = MAX(1, LENGTH(content) / 4)")
                conn.commit()
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_team ON messages(team_id)")
            conn.commit()

    def _migrate_add_llm_traces(self) -> None:
        """Create llm_traces table if the DB was created before the observability schema,
        and ensure all columns exist if the table was created by an older version."""
        with self._conn() as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "llm_traces" not in tables:
                log.info("[MonitoringDB] Migrating: creating llm_traces table")
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS llm_traces (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp         REAL    NOT NULL,
                        agent_name        TEXT    NOT NULL,
                        team_id           TEXT    NOT NULL,
                        turn_id           TEXT    NOT NULL,
                        task_ids          TEXT,
                        trigger_type      TEXT    NOT NULL,
                        model             TEXT,
                        provider          TEXT,
                        system_prompt     TEXT,
                        live_context      TEXT,
                        user_prompt       TEXT,
                        history_json      TEXT,
                        history_len       INTEGER,
                        tools_count       INTEGER,
                        steps_json        TEXT,
                        final_response    TEXT,
                        duration_seconds  REAL,
                        tokens_in         INTEGER,
                        tokens_out        INTEGER,
                        cost_usd          REAL,
                        status            TEXT    NOT NULL DEFAULT 'running'
                    );
                    CREATE INDEX IF NOT EXISTS idx_traces_agent ON llm_traces(agent_name, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_traces_team  ON llm_traces(team_id, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_traces_turn  ON llm_traces(turn_id);
                """)
                conn.commit()
            else:
                cursor = conn.execute("PRAGMA table_info(llm_traces)")
                cols = {c[1] for c in cursor.fetchall()}
                col_defs = {
                    "history_len": "INTEGER",
                    "tools_count": "INTEGER",
                    "live_context": "TEXT",
                    "user_prompt": "TEXT",
                    "history_json": "TEXT",
                    "steps_json": "TEXT",
                    "final_response": "TEXT",
                    "duration_seconds": "REAL",
                    "tokens_in": "INTEGER",
                    "tokens_out": "INTEGER",
                    "cost_usd": "REAL",
                    "status": "TEXT DEFAULT 'running'",
                    "trigger_type": "TEXT DEFAULT 'task'",
                    "model": "TEXT",
                    "provider": "TEXT",
                    "system_prompt": "TEXT",
                    "task_ids": "TEXT",
                    "turn_id": "TEXT DEFAULT ''",
                    "team_id": "TEXT DEFAULT 'default'",
                }
                for col_name, col_type in col_defs.items():
                    if col_name not in cols:
                        log.info("[MonitoringDB] Migrating: adding %s to llm_traces", col_name)
                        conn.execute(f"ALTER TABLE llm_traces ADD COLUMN {col_name} {col_type}")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_agent ON llm_traces(agent_name, timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_team  ON llm_traces(team_id, timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_turn  ON llm_traces(turn_id)")
                conn.commit()

    def log_event(
        self,
        agent_name: str,
        event_type: str,
        from_agent: str = None,
        to_agent: str = None,
        task_id: str = None,
        data: dict = None,
        team_id: str = None,
    ):
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO events
                       (timestamp, agent_name, team_id, event_type, from_agent, to_agent, task_id, data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        time.time(),
                        agent_name,
                        team_id,
                        event_type,
                        from_agent,
                        to_agent,
                        task_id,
                        json.dumps(data) if data else None,
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] Failed to log event: %s", e)

    def log_message(
        self,
        agent_name: str,
        role: str,
        content: str,
        task_id: str = None,
        team_id: str = None,
    ):
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO messages
                       (timestamp, agent_name, team_id, role, content, task_id, tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (time.time(), agent_name, team_id, role, content, task_id,
                     _estimate_tokens(content)),
                )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] Failed to log message: %s", e)

    def log_decision(self, agent_name: str, decision: str, team_id: str = None) -> None:
        """Append a one-line decision to the shared, team-scoped decision log.

        This is the durable team memory that replaces the old agent_log.md /
        log_changes file trail: the last N entries are injected into every
        agent's prompt each turn (see prompts.compose_live_context), so a
        decision recorded here is visible team-wide without anyone reading a
        file. Entries are append-only and never edited."""
        # Collapse to a single line — the whole point of the decision log is a
        # scannable one-liner per entry; newlines would break the injected view.
        one_line = " ".join((decision or "").split()).strip()
        if not one_line:
            return
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO decisions (timestamp, agent_name, team_id, decision) "
                    "VALUES (?, ?, ?, ?)",
                    (time.time(), agent_name, team_id, one_line[:500]),
                )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] Failed to log decision: %s", e)

    def get_recent_decisions(self, team_id: str = None, limit: int = 20) -> List[dict]:
        """Return the most recent decisions (newest first) for prompt injection.

        Scoped by team_id when given. Each row: {timestamp, agent_name, decision}."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                if team_id:
                    rows = conn.execute(
                        "SELECT timestamp, agent_name, decision FROM decisions "
                        "WHERE team_id = ? ORDER BY timestamp DESC LIMIT ?",
                        (team_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT timestamp, agent_name, decision FROM decisions "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] Failed to read decisions: %s", e)
            return []

    def search_decisions(self, team_id: str = None, query: str = None,
                         limit: int = 40) -> List[dict]:
        """Search the decision log on demand (for the recall_decisions tool).

        Newest first. Optionally scoped by ``team_id`` and filtered to decisions
        whose text contains ``query`` (case-insensitive substring). Unlike
        get_recent_decisions (fixed 20, prompt injection), this lets an agent
        reach further back and grep its team's strategy history."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = ("SELECT timestamp, agent_name, decision FROM decisions "
                       "WHERE 1=1")
                params: list = []
                if team_id:
                    sql += " AND team_id = ?"
                    params.append(team_id)
                if query:
                    sql += " AND decision LIKE ? ESCAPE '\\'"
                    esc = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    params.append(f"%{esc}%")
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(max(1, min(int(limit), 200)))
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] search_decisions failed: %s", e)
            return []

    def get_events(
        self,
        agent_name: str = None,
        team_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM events WHERE 1=1"
                params = []
                if agent_name:
                    sql += " AND agent_name = ?"
                    params.append(agent_name)
                if team_id:
                    # events rows have team_id NULL (writers don't set it); recover
                    # the team's rows via the agent-name set too. See _team_filter.
                    tclause, tparams = self._team_filter(conn, team_id)
                    if tclause:
                        sql += " AND " + tclause
                        params.extend(tparams)
                sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] Failed to get events: %s", e)
            return []

    def get_messages(
        self,
        agent_name: str,
        team_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM messages WHERE agent_name = ?"
                params = [agent_name]
                if team_id:
                    sql += " AND team_id = ?"
                    params.append(team_id)
                sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] Failed to get messages: %s", e)
            return []

    def get_owner_messages(self, agent_name: str, limit: int = 200) -> List[dict]:
        """The owner↔agent chat channel only: roles 'owner_in' (owner → agent)
        and 'owner_out' (agent → owner via message_owner). Excludes the internal
        execution transcript the ops dashboard shows."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM messages WHERE agent_name = ? "
                    "AND role IN ('owner_in', 'owner_out') "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (agent_name, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] Failed to get owner messages: %s", e)
            return []

    # ---- digests (Layer 3 observability) -----------------------------------

    def get_new_activity(
        self, agent_name: str, after_id: int = 0, team_id: str = None
    ) -> Dict[str, int]:
        """Cheap aggregate of un-digested transcript: how many new messages,
        their estimated tokens, and the latest message id. Used by the digest
        trigger WITHOUT pulling any content, so an idle-agent check is one row."""
        out = {"count": 0, "tokens": 0, "max_id": after_id}
        try:
            with self._conn() as conn:
                sql = ("SELECT COUNT(*) c, COALESCE(SUM(tokens),0) t, "
                       "COALESCE(MAX(id), ?) m FROM messages "
                       "WHERE agent_name = ? AND id > ?")
                params = [after_id, agent_name, after_id]
                if team_id:
                    sql += " AND team_id = ?"
                    params.append(team_id)
                row = conn.execute(sql, tuple(params)).fetchone()
                if row:
                    out = {"count": row[0] or 0, "tokens": row[1] or 0,
                           "max_id": row[2] or after_id}
        except Exception as e:
            log.warning("[MonitorDB] get_new_activity failed: %s", e)
        return out

    def get_messages_since(
        self, agent_name: str, after_id: int = 0, team_id: str = None,
        limit: int = 400,
    ) -> List[dict]:
        """New transcript messages (id > after_id), oldest-first, for summarizing."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = ("SELECT id, timestamp, role, content, tokens FROM messages "
                       "WHERE agent_name = ? AND id > ?")
                params = [agent_name, after_id]
                if team_id:
                    sql += " AND team_id = ?"
                    params.append(team_id)
                sql += " ORDER BY id ASC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] get_messages_since failed: %s", e)
            return []

    def get_last_digest(self, agent_name: str, team_id: str = None) -> Optional[dict]:
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM digests WHERE agent_name = ?"
                params = [agent_name]
                if team_id:
                    sql += " AND team_id = ?"
                    params.append(team_id)
                sql += " ORDER BY id DESC LIMIT 1"
                row = conn.execute(sql, tuple(params)).fetchone()
                return dict(row) if row else None
        except Exception as e:
            log.warning("[MonitorDB] get_last_digest failed: %s", e)
            return None

    def get_digests(
        self, agent_name: str, team_id: str = None, limit: int = 50,
    ) -> List[dict]:
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM digests WHERE agent_name = ?"
                params = [agent_name]
                if team_id:
                    sql += " AND team_id = ?"
                    params.append(team_id)
                sql += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] get_digests failed: %s", e)
            return []

    def save_digest(
        self, agent_name: str, summary: str, covers_to_msg_id: int,
        msg_count: int, tokens_in: int, model: str, team_id: str = None,
    ) -> Optional[int]:
        """Persist one rolling digest. summary is a JSON string. Returns row id."""
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    """INSERT INTO digests
                       (timestamp, agent_name, team_id, summary, covers_to_msg_id,
                        msg_count, tokens_in, model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (time.time(), agent_name, team_id, summary, covers_to_msg_id,
                     msg_count, tokens_in, model),
                )
                conn.commit()
                return cur.lastrowid
        except Exception as e:
            log.warning("[MonitorDB] save_digest failed: %s", e)
            return None

    # Event types that make up the owner-facing audit feed.
    AUDIT_EVENT_TYPES = (
        "human_responded", "proposal_resolved", "agent_paused",
        "agent_resumed", "execution_stopped",
    )

    def get_audit_feed(self, team_id: str = None, limit: int = 60) -> List[dict]:
        """Merged audit trail for the consumer History tab: the human's decisions
        (answered questions, approved/rejected proposals, pauses/stops) plus the
        side-effecting actions agents recorded via log_action. Newest first."""
        feed: List[dict] = []
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                ph = ",".join("?" * len(self.AUDIT_EVENT_TYPES))
                sql = f"SELECT * FROM events WHERE event_type IN ({ph})"
                params: list = list(self.AUDIT_EVENT_TYPES)
                if team_id:
                    tclause, tparams = self._team_filter(conn, team_id)
                    if tclause:
                        sql += " AND " + tclause
                        params.extend(tparams)
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                for r in conn.execute(sql, tuple(params)).fetchall():
                    d = dict(r)
                    try:
                        d["data"] = json.loads(d.get("data") or "{}")
                    except Exception:
                        d["data"] = {}
                    feed.append(d)
        except Exception as e:
            log.warning("[MonitorDB] get_audit_feed events failed: %s", e)
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM actions"
                params = []
                if team_id:
                    sql += " WHERE team_id = ?"
                    params.append(team_id)
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                for r in conn.execute(sql, tuple(params)).fetchall():
                    feed.append({"event_type": "agent_action", **dict(r)})
        except Exception as e:
            log.warning("[MonitorDB] get_audit_feed actions failed: %s", e)
        feed.sort(key=lambda r: r.get("timestamp") or 0, reverse=True)
        return feed[:limit]

    def log_result(self, agent_name: str, title: str, summary: str,
                   team_id: str = None) -> Optional[int]:
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO results (timestamp, agent_name, team_id, title, summary) "
                    "VALUES (?,?,?,?,?)",
                    (time.time(), agent_name, team_id, title, summary))
                conn.commit()
                return cur.lastrowid
        except Exception as e:
            log.warning("[MonitorDB] log_result failed: %s", e)
            return None

    def get_results(self, team_id: str = None, limit: int = 50) -> List[dict]:
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                if team_id:
                    rows = conn.execute(
                        "SELECT * FROM results WHERE team_id = ? ORDER BY id DESC LIMIT ?",
                        (team_id, limit)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM results ORDER BY id DESC LIMIT ?",
                        (limit,)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] get_results failed: %s", e)
            return []

    def prune(self, max_events: int, max_messages: int,
              max_digests: int = 5000, max_delegations: int = 20000,
              max_actions: int = 20000, max_decisions: int = 20000,
              max_milestones: int = 5000) -> Dict[str, int]:
        """Trim each table to its most recent N rows (rolling retention).

        Without this the DB grows unbounded on a 24/7 run. Every table that
        runtime code appends to is bounded here — not just events/messages/digests
        but the relational tables (delegations/actions/decisions/milestones), which
        previously had NO retention at all. Still-OPEN delegations are NEVER pruned
        (they represent outstanding work the supervisor still tracks). Returns rows
        deleted per table.
        """
        deleted = {"events": 0, "messages": 0, "digests": 0,
                   "delegations": 0, "actions": 0, "decisions": 0, "milestones": 0}
        # (table, keep-N, extra WHERE that protects rows from pruning)
        plans = [
            ("events", max_events, ""),
            ("messages", max_messages, ""),
            ("digests", max_digests, ""),
            # Keep open delegations regardless of age/count.
            ("delegations", max_delegations, "status != 'open'"),
            ("actions", max_actions, ""),
            ("decisions", max_decisions, ""),
            ("milestones", max_milestones, ""),
        ]
        try:
            with self._conn() as conn:
                for table, keep, guard in plans:
                    keep_clause = (
                        f"id NOT IN (SELECT id FROM {table} ORDER BY id DESC LIMIT ?)"
                    )
                    where = keep_clause + (f" AND ({guard})" if guard else "")
                    cur = conn.execute(f"DELETE FROM {table} WHERE {where}", (keep,))
                    deleted[table] = cur.rowcount or 0
                conn.commit()
            if any(deleted.values()):
                log.info(
                    "[MonitorDB] Pruned %s",
                    ", ".join(f"{n} {t}" for t, n in deleted.items() if n),
                )
        except Exception as e:
            log.warning("[MonitorDB] Prune failed: %s", e)
        return deleted

    def delete_team_records(self, team_id: str, member_names: Optional[list] = None) -> Dict[str, int]:
        """Purge all monitoring.db entries for a team and its member agents."""
        names = list(member_names or [])
        deleted: Dict[str, int] = {}
        tables_with_team_and_agent = ["events", "messages", "digests", "decisions", "actions"]
        try:
            with self._conn() as conn:
                for table in tables_with_team_and_agent:
                    if names:
                        placeholders = ",".join("?" for _ in names)
                        sql = f"DELETE FROM {table} WHERE team_id=? OR agent_name IN ({placeholders})"
                        params = [team_id] + names
                    else:
                        sql = f"DELETE FROM {table} WHERE team_id=?"
                        params = [team_id]
                    cur = conn.execute(sql, params)
                    deleted[table] = cur.rowcount or 0

                # milestones has team_id (no agent_name column)
                cur = conn.execute("DELETE FROM milestones WHERE team_id=?", (team_id,))
                deleted["milestones"] = cur.rowcount or 0

                # delegations table has from_agent and to_agent
                if names:
                    placeholders = ",".join("?" for _ in names)
                    sql = f"DELETE FROM delegations WHERE team_id=? OR from_agent IN ({placeholders}) OR to_agent IN ({placeholders})"
                    params = [team_id] + names + names
                else:
                    sql = f"DELETE FROM delegations WHERE team_id=?"
                    params = [team_id]
                cur = conn.execute(sql, params)
                deleted["delegations"] = cur.rowcount or 0

                # llm_traces is keyed by both team_id and agent_name
                if names:
                    placeholders = ",".join("?" for _ in names)
                    sql = f"DELETE FROM llm_traces WHERE team_id=? OR agent_name IN ({placeholders})"
                    params = [team_id] + names
                else:
                    sql = "DELETE FROM llm_traces WHERE team_id=?"
                    params = [team_id]
                cur = conn.execute(sql, params)
                deleted["llm_traces"] = cur.rowcount or 0

                conn.commit()
            if any(deleted.values()):
                log.info("[MonitorDB] Purged team '%s' records: %s", team_id, deleted)
        except Exception as e:
            log.warning("[MonitorDB] Purge team '%s' failed: %s", team_id, e)
        return deleted

    def delete_agent_records(self, agent_name: str) -> Dict[str, int]:
        """Purge all monitoring.db entries for a single agent."""
        deleted: Dict[str, int] = {}
        tables_with_agent = ["events", "messages", "digests", "decisions", "actions"]
        try:
            with self._conn() as conn:
                for table in tables_with_agent:
                    cur = conn.execute(f"DELETE FROM {table} WHERE agent_name=?", (agent_name,))
                    deleted[table] = cur.rowcount or 0
                cur = conn.execute("DELETE FROM delegations WHERE from_agent=? OR to_agent=?", (agent_name, agent_name))
                deleted["delegations"] = cur.rowcount or 0
                cur = conn.execute("DELETE FROM llm_traces WHERE agent_name=?", (agent_name,))
                deleted["llm_traces"] = cur.rowcount or 0
                conn.commit()
            if any(deleted.values()):
                log.info("[MonitorDB] Purged agent '%s' records: %s", agent_name, deleted)
        except Exception as e:
            log.warning("[MonitorDB] Purge agent '%s' failed: %s", agent_name, e)
        return deleted

    # ---- Delegation correlation (typed messages: TASK/QUESTION -> RESULT) ----
    def open_delegation(self, msg_id: str, from_agent: str, to_agent: str,
                        kind: str, summary: str = "", team_id: str = None) -> None:
        try:
            with self._conn() as conn:
                # Idempotent on msg_id: a deduped re-send (same deterministic msg_id)
                # must not open a SECOND delegation that no RESULT will ever close
                # (a phantom "overdue" the supervisor chases forever).
                existing = conn.execute(
                    "SELECT 1 FROM delegations WHERE msg_id=? AND status='open' LIMIT 1",
                    (msg_id,),
                ).fetchone()
                if existing:
                    return
                conn.execute(
                    "INSERT INTO delegations (msg_id, timestamp, from_agent, to_agent, "
                    "team_id, kind, summary, status) VALUES (?,?,?,?,?,?,?, 'open')",
                    (msg_id, time.time(), from_agent, to_agent, team_id, kind,
                     (summary or "")[:200]),
                )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] open_delegation failed: %s", e)

    def answer_delegation(self, reply_to_msg_id: str, by_agent: str = None) -> bool:
        """Mark the open delegation with msg_id == reply_to as answered. Returns
        True if a matching open row was closed."""
        if not reply_to_msg_id:
            return False
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE delegations SET status='answered', answered_at=? "
                    "WHERE msg_id=? AND status='open'",
                    (time.time(), reply_to_msg_id),
                )
                conn.commit()
                return (cur.rowcount or 0) > 0
        except Exception as e:
            log.warning("[MonitorDB] answer_delegation failed: %s", e)
            return False

    def close_delegation_manual(self, msg_id_prefix: str, by_agent: str = None,
                                reason: str = "", team_id: str = None,
                                require_participant: bool = False) -> dict:
        """Close an open delegation by msg_id (full id or unambiguous prefix).

        For orphaned ledger rows whose work completed via another task chain:
        no RESULT carrying their msg_id will ever arrive, so without an explicit
        dismiss they sit OPEN forever — re-analyzed on every heartbeat and
        periodically forcing stale-delegation supervisor sweeps."""
        prefix = (msg_id_prefix or "").strip()
        if len(prefix) < 6:
            return {"success": False,
                    "error": "msg_id must be at least 6 characters."}
        try:
            with self._conn() as conn:
                sql = ("SELECT msg_id, from_agent, to_agent FROM delegations "
                       "WHERE status='open' AND msg_id LIKE ?")
                params: list = [prefix + "%"]
                if team_id:
                    sql += " AND team_id = ?"; params.append(team_id)
                rows = conn.execute(sql + " LIMIT 3", tuple(params)).fetchall()
                if not rows:
                    return {"success": False,
                            "error": f"No OPEN delegation matches '{prefix}' — "
                                     "it may already be closed."}
                if len(rows) > 1:
                    return {"success": False,
                            "error": f"'{prefix}' is ambiguous "
                                     f"({len(rows)} open matches) — use more characters."}
                msg_id, d_from, d_to = rows[0]
                if require_participant and by_agent not in (d_from, d_to):
                    return {"success": False,
                            "error": f"Only {d_from} or {d_to} (or a supervisor) "
                                     "may close this entry."}
                conn.execute(
                    "UPDATE delegations SET status='closed', answered_at=? "
                    "WHERE msg_id=? AND status='open'",
                    (time.time(), msg_id),
                )
                conn.commit()
            self.log_event(by_agent or "unknown", "ledger_entry_closed",
                           data={"msg_id": msg_id, "from_agent": d_from,
                                 "to_agent": d_to, "reason": (reason or "")[:200]})
            return {"success": True, "msg_id": msg_id,
                    "from_agent": d_from, "to_agent": d_to}
        except Exception as e:
            log.warning("[MonitorDB] close_delegation_manual failed: %s", e)
            return {"success": False, "error": str(e)}

    def get_open_delegations(self, to_agent: str = None, from_agent: str = None,
                             team_id: str = None, limit: int = 50) -> List[dict]:
        """Outstanding TASK/QUESTION delegations (status='open'), newest first."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM delegations WHERE status='open'"
                params: list = []
                if team_id:
                    sql += " AND team_id = ?"; params.append(team_id)
                if to_agent:
                    sql += " AND to_agent = ?"; params.append(to_agent)
                if from_agent:
                    sql += " AND from_agent = ?"; params.append(from_agent)
                sql += " ORDER BY timestamp DESC LIMIT ?"; params.append(limit)
                return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        except Exception as e:
            log.warning("[MonitorDB] get_open_delegations failed: %s", e)
            return []

    def get_recent_closed_delegations(self, team_id: str = None,
                                      limit: int = 8) -> List[dict]:
        """Recently ANSWERED delegations, newest first.

        Injected into the live context as RECENTLY COMPLETED so a delegator can
        SEE that a task was already delivered before re-asking it. (Observed
        failure: a coordinator re-delegated the same prospect-list task three
        times because nothing in its prompt showed the closed item.)"""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = "SELECT * FROM delegations WHERE status='answered'"
                params: list = []
                if team_id:
                    sql += " AND team_id = ?"; params.append(team_id)
                sql += " ORDER BY answered_at DESC LIMIT ?"; params.append(limit)
                return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        except Exception as e:
            log.warning("[MonitorDB] get_recent_closed_delegations failed: %s", e)
            return []

    def get_latest_event_id(self) -> int:
        """Current MAX(id) of the events table (watermark seeding)."""
        try:
            with self._conn() as conn:
                row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()
                return int(row[0] or 0)
        except Exception as e:
            log.warning("[MonitorDB] get_latest_event_id failed: %s", e)
            return 0

    def get_passive_messages_for(self, to_agent: str, after_event_id: int = 0,
                                 limit: int = 20) -> Dict[str, Any]:
        """Non-waking peer messages (FYI) addressed to one agent, with
        event id > after_event_id, oldest first.

        Passive messages deliberately create no task — but before this query
        they were visible only in the rolling team feed, where they scroll away
        on a busy team. Senders who noticed their STATUS was never seen escalated
        to waking TASKs just to be heard (observed in transcripts). The daemon
        now drains this per-recipient backlog into the agent's next normal turn.
        Returns {"messages": [...], "max_id": highest event id seen}."""
        out: Dict[str, Any] = {"messages": [], "max_id": after_event_id}
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, timestamp, from_agent, data FROM events "
                    "WHERE event_type = 'message_sent' AND to_agent = ? AND id > ? "
                    "ORDER BY id ASC LIMIT ?",
                    (to_agent, after_event_id, limit),
                ).fetchall()
                msgs = []
                max_id = after_event_id
                for r in rows:
                    max_id = max(max_id, int(r["id"]))
                    try:
                        d = json.loads(r["data"] or "{}") or {}
                    except Exception:
                        d = {}
                    if d.get("waking", True):
                        continue  # waking kinds were delivered via the queue
                    msgs.append({
                        "id": int(r["id"]),
                        "timestamp": r["timestamp"],
                        "from_agent": r["from_agent"],
                        "kind": d.get("kind") or "STATUS",
                        "text": d.get("message_full") or d.get("message_preview") or "",
                    })
                out = {"messages": msgs, "max_id": max_id}
        except Exception as e:
            log.warning("[MonitorDB] get_passive_messages_for failed: %s", e)
        return out

    # ---- Action audit log + idempotency ----
    def record_action(self, agent_name: str, action_type: str, idempotency_key: str,
                      target: str = "", outcome: str = "", verified: bool = False,
                      team_id: str = None) -> Dict[str, Any]:
        """Record a side-effecting action. Idempotent on (team_id, key): if the
        key already exists this records NOTHING and returns the prior row so the
        caller can skip the duplicate. Returns {duplicate: bool, existing|recorded}."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                prior = conn.execute(
                    "SELECT * FROM actions WHERE team_id IS ? AND idempotency_key = ?",
                    (team_id, idempotency_key),
                ).fetchone()
                if prior:
                    return {"duplicate": True, "existing": dict(prior)}
                try:
                    conn.execute(
                        "INSERT INTO actions (timestamp, agent_name, team_id, action_type, "
                        "target, idempotency_key, outcome, verified) VALUES (?,?,?,?,?,?,?,?)",
                        (time.time(), agent_name, team_id, action_type, (target or "")[:300],
                         idempotency_key, (outcome or "")[:500], 1 if verified else 0),
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
                    # Lost a concurrent race on the UNIQUE (team_id, idempotency_key)
                    # index — another agent inserted between our SELECT and INSERT.
                    # This is the exact double-action the guard exists for: report it
                    # as a DUPLICATE so the loser skips the side effect, instead of
                    # the old `except Exception` path that returned duplicate=False
                    # and told BOTH agents to proceed.
                    conn.rollback()
                    row = conn.execute(
                        "SELECT * FROM actions WHERE team_id IS ? AND idempotency_key = ?",
                        (team_id, idempotency_key),
                    ).fetchone()
                    return {"duplicate": True, "existing": dict(row) if row else {}}
                return {"duplicate": False, "recorded": True}
        except Exception as e:
            log.warning("[MonitorDB] record_action failed: %s", e)
            return {"duplicate": False, "recorded": False, "error": str(e)}

    def get_events_since(self, event_type: str, since_ts: float,
                         limit: int = 1000) -> List[dict]:
        """Events of one type newer than since_ts, newest first. Used by the loop
        detector so a flood of other event types can't push message_sent rows out
        of a fixed-size get_events() page."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM events WHERE event_type = ? AND timestamp >= ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (event_type, since_ts, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] get_events_since failed: %s", e)
            return []

    def get_token_usage_events(self, since_ts: float,
                               agent_names: Optional[List[str]] = None,
                               team_id: Optional[str] = None,
                               limit: int = 5000) -> List[dict]:
        """token_usage events newer than since_ts, OLDEST first (id ASC) so
        the costs endpoint can difference legacy cumulative rows per agent in
        order. Scope with agent_names (the config is the membership authority —
        events are written with team_id NULL); team_id is OR-ed in as a
        belt-and-braces match, or used via _team_filter when no names given."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = ("SELECT * FROM events WHERE event_type = 'token_usage' "
                       "AND timestamp >= ?")
                params: list = [since_ts]
                if agent_names:
                    ph = ",".join("?" * len(agent_names))
                    if team_id:
                        sql += f" AND (agent_name IN ({ph}) OR team_id = ?)"
                        params += list(agent_names) + [team_id]
                    else:
                        sql += f" AND agent_name IN ({ph})"
                        params += list(agent_names)
                elif team_id:
                    tclause, tparams = self._team_filter(conn, team_id)
                    if tclause:
                        sql += " AND " + tclause
                        params += tparams
                sql += " ORDER BY id ASC LIMIT ?"
                params.append(limit)
                return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        except Exception as e:
            log.warning("[MonitorDB] get_token_usage_events failed: %s", e)
            return []

    def count_events_since(self, agent_name: str, event_type: str,
                           since_ts: float) -> int:
        """Cheap COUNT of one agent's events of one type newer than since_ts."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE agent_name = ? "
                    "AND event_type = ? AND timestamp >= ?",
                    (agent_name, event_type, since_ts),
                ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def count_actions_since(self, team_id: str, since_ts: float) -> int:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM actions WHERE team_id IS ? AND timestamp >= ?",
                    (team_id, since_ts),
                ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def count_decisions_since(self, team_id: str, since_ts: float) -> int:
        try:
            with self._conn() as conn:
                # Exclude the loop detector's OWN alert rows: it writes a decision
                # note when it fires, and counting that as team productivity would
                # suppress the very team-stall detection that reads this count.
                row = conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE team_id IS ? AND timestamp >= ? "
                    "AND agent_name != 'loop_detector'",
                    (team_id, since_ts),
                ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    # ---- Decision-log rollup (long-term memory) ----
    def get_decisions_after(self, team_id: str, after_id: int = 0,
                            limit: int = 1000) -> List[dict]:
        """Decisions with id > after_id, OLDEST first (for rollup batching)."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, timestamp, agent_name, decision FROM decisions "
                    "WHERE team_id IS ? AND id > ? ORDER BY id ASC LIMIT ?",
                    (team_id, after_id, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] get_decisions_after failed: %s", e)
            return []

    def save_milestone(self, team_id: str, summary: str, covers_to_decision: int) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO milestones (timestamp, team_id, summary, covers_to_decision) "
                    "VALUES (?,?,?,?)",
                    (time.time(), team_id, summary, covers_to_decision),
                )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] save_milestone failed: %s", e)

    def get_latest_milestone(self, team_id: str) -> Optional[dict]:
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                r = conn.execute(
                    "SELECT * FROM milestones WHERE team_id IS ? "
                    "ORDER BY timestamp DESC LIMIT 1", (team_id,),
                ).fetchone()
                return dict(r) if r else None
        except Exception as e:
            log.warning("[MonitorDB] get_latest_milestone failed: %s", e)
            return None

    def _team_agent_names(self, conn, team_id: str) -> set:
        """Agent names known to belong to ``team_id``, resolved from the tables
        that reliably carry team_id (decisions/actions/delegations).

        The events/messages tables are written with team_id NULL by the runtime,
        so a plain ``team_id = ?`` filter on them matches nothing — callers OR this
        set in to recover the team's rows.
        """
        names = set()
        for sql in (
            "SELECT DISTINCT agent_name FROM decisions WHERE team_id IS ?",
            "SELECT DISTINCT agent_name FROM actions WHERE team_id IS ?",
            "SELECT DISTINCT from_agent FROM delegations WHERE team_id IS ?",
            "SELECT DISTINCT to_agent FROM delegations WHERE team_id IS ?",
        ):
            try:
                for r in conn.execute(sql, (team_id,)).fetchall():
                    if r[0]:
                        names.add(r[0])
            except Exception:
                pass
        return names

    def _team_filter(self, conn, team_id, col: str = "agent_name"):
        """Return (sql_fragment, params) selecting a team's rows even when the
        row's own team_id is NULL (events/messages). Empty fragment if no team."""
        if not team_id:
            return "", []
        names = sorted(self._team_agent_names(conn, team_id))
        if names:
            ph = ",".join("?" * len(names))
            return f"(team_id = ? OR {col} IN ({ph}))", [team_id, *names]
        return "team_id = ?", [team_id]

    def get_agent_stats(self, team_id: Optional[str] = None) -> Dict[str, dict]:
        stats = {}
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                tclause, tparams = self._team_filter(conn, team_id)
                where = (" WHERE " + tclause) if tclause else ""

                sql = ("SELECT agent_name, event_type, COUNT(*) as count FROM events"
                       + where + " GROUP BY agent_name, event_type")
                rows = conn.execute(sql, tuple(tparams)).fetchall()
                for r in rows:
                    aname = r["agent_name"]
                    if aname not in stats:
                        stats[aname] = {
                            "events": {},
                            "last_active": None,
                            "total_messages": 0,
                            "total_tokens": 0,
                        }
                    stats[aname]["events"][r["event_type"]] = r["count"]

                sql_last = ("SELECT agent_name, MAX(timestamp) as last_ts FROM events"
                            + where + " GROUP BY agent_name")
                rows = conn.execute(sql_last, tuple(tparams)).fetchall()
                for r in rows:
                    if r["agent_name"] in stats:
                        stats[r["agent_name"]]["last_active"] = r["last_ts"]

                sql_msg = ("SELECT agent_name, COUNT(*) as count, "
                           "COALESCE(SUM(tokens), 0) as tokens FROM messages"
                           + where + " GROUP BY agent_name")
                rows = conn.execute(sql_msg, tuple(tparams)).fetchall()
                for r in rows:
                    if r["agent_name"] in stats:
                        stats[r["agent_name"]]["total_messages"] = r["count"]
                        stats[r["agent_name"]]["total_tokens"] = r["tokens"]
        except Exception as e:
            log.warning("[MonitorDB] Failed to get stats: %s", e)
        return stats

    # ---- LLM Traces (Observability) ----------------------------------------

    def start_llm_trace(
        self,
        agent_name: str,
        team_id: str,
        turn_id: str,
        trigger_type: str,
        task_ids: str = "",
        model: str = "",
        provider: str = "",
        system_prompt: str = "",
        live_context: str = "",
        user_prompt: str = "",
        history_json: str = "[]",
        history_len: int = 0,
        tools_count: int = 0,
    ) -> int:
        """Create a 'running' trace row and return its id for later completion."""
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO llm_traces (
                        timestamp, agent_name, team_id, turn_id, task_ids,
                        trigger_type, model, provider, system_prompt, live_context,
                        user_prompt, history_json, history_len, tools_count,
                        steps_json, final_response, duration_seconds,
                        tokens_in, tokens_out, cost_usd, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(), agent_name, team_id, turn_id, task_ids,
                        trigger_type, model, provider, system_prompt, live_context,
                        user_prompt, history_json, history_len, tools_count,
                        "[]", "", 0.0, 0, 0, 0.0, "running",
                    ),
                )
                conn.commit()
                return cur.lastrowid
        except Exception as e:
            log.warning("[MonitorDB] Failed to start llm_trace for %s: %s", agent_name, e)
            return 0

    def complete_llm_trace(
        self,
        trace_id: int,
        final_response: str = "",
        duration_seconds: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        status: str = "completed",
        steps_json: str = "[]",
        history_json: Optional[str] = None,
    ) -> None:
        """Finalize a trace with output, token counts, cost, duration, and steps.

        ``history_json`` replaces the start-time ref with the full snapshot —
        callers pass it only for failed turns (or TEAMS_TRACE_FULL_HISTORY),
        since storing ~O(history) per row is quadratic growth."""
        if not trace_id:
            return
        try:
            with self._conn() as conn:
                if history_json:
                    conn.execute(
                        """
                        UPDATE llm_traces SET
                            final_response=?, duration_seconds=?, tokens_in=?,
                            tokens_out=?, cost_usd=?, status=?, steps_json=?,
                            history_json=?
                        WHERE id=?
                        """,
                        (final_response[:8000], duration_seconds, tokens_in,
                         tokens_out, cost_usd, status, steps_json, history_json,
                         trace_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE llm_traces SET
                            final_response=?, duration_seconds=?, tokens_in=?,
                            tokens_out=?, cost_usd=?, status=?, steps_json=?
                        WHERE id=?
                        """,
                        (final_response[:8000], duration_seconds, tokens_in,
                         tokens_out, cost_usd, status, steps_json, trace_id),
                    )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] Failed to complete llm_trace %s: %s", trace_id, e)

    def get_llm_traces(
        self,
        agent_name: Optional[str] = None,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
        trigger_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return trace metadata list — no steps_json or history_json to keep payloads small."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                clauses, params = [], []
                if agent_name:
                    clauses.append("agent_name = ?")
                    params.append(agent_name)
                if team_id:
                    clauses.append("team_id = ?")
                    params.append(team_id)
                if status:
                    clauses.append("status = ?")
                    params.append(status)
                if trigger_type:
                    clauses.append("trigger_type = ?")
                    params.append(trigger_type)
                where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
                sql = f"""
                    SELECT id, timestamp, agent_name, team_id, turn_id, task_ids,
                           trigger_type, model, provider, duration_seconds,
                           history_len, tools_count,
                           tokens_in, tokens_out, cost_usd, status,
                           length(system_prompt)  AS system_prompt_len,
                           length(live_context)   AS live_context_len,
                           length(user_prompt)    AS user_prompt_len,
                           substr(final_response, 1, 300) AS response_preview,
                           json_array_length(CASE WHEN steps_json IS NULL OR steps_json = '' THEN '[]'
                                             ELSE steps_json END) AS step_count
                    FROM llm_traces{where}
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([max(1, min(limit, 500)), max(0, offset)])
                rows = conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("[MonitorDB] Failed to get llm_traces: %s", e)
            return []

    def get_llm_trace(self, trace_id: Any) -> Optional[Dict[str, Any]]:
        """Fetch the full, untruncated details of a single trace including history and steps."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                if isinstance(trace_id, int) or (
                    isinstance(trace_id, str) and trace_id.isdigit()
                ):
                    row = conn.execute(
                        "SELECT * FROM llm_traces WHERE id = ?", (int(trace_id),)
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM llm_traces WHERE turn_id = ? ORDER BY id DESC LIMIT 1",
                        (str(trace_id),),
                    ).fetchone()
                if not row:
                    return None
                d = dict(row)
                # Parse JSON fields for API consumers
                for key, default in (("history_json", []), ("steps_json", [])):
                    try:
                        d[key] = json.loads(d.get(key) or json.dumps(default))
                    except Exception:
                        d[key] = default
                return d
        except Exception as e:
            log.warning("[MonitorDB] Failed to get llm_trace %s: %s", trace_id, e)
            return None

    def get_observability_stats(
        self, team_id: Optional[str] = None, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate stats over llm_traces (total turns, tokens, costs, trigger distribution)."""
        stats: Dict[str, Any] = {
            "total_turns": 0, "total_tokens_in": 0, "total_tokens_out": 0,
            "total_cost_usd": 0.0, "avg_duration_seconds": 0.0,
            "status_counts": {}, "trigger_counts": {}, "model_counts": {},
            "recent_error_rate": 0.0,
        }
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                clauses, params = [], []
                if team_id:
                    clauses.append("team_id = ?")
                    params.append(team_id)
                if agent_name:
                    clauses.append("agent_name = ?")
                    params.append(agent_name)
                where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
                agg = conn.execute(
                    f"SELECT COUNT(*) as total, COALESCE(SUM(tokens_in),0) as tot_in,"
                    f" COALESCE(SUM(tokens_out),0) as tot_out,"
                    f" COALESCE(SUM(cost_usd),0.0) as tot_cost,"
                    f" COALESCE(AVG(duration_seconds),0.0) as avg_dur"
                    f" FROM llm_traces{where}",
                    tuple(params),
                ).fetchone()
                if agg:
                    stats["total_turns"] = agg["total"]
                    stats["total_tokens_in"] = agg["tot_in"]
                    stats["total_tokens_out"] = agg["tot_out"]
                    stats["total_cost_usd"] = round(agg["tot_cost"], 4)
                    stats["avg_duration_seconds"] = round(agg["avg_dur"], 2)
                for r in conn.execute(
                    f"SELECT status, COUNT(*) c FROM llm_traces{where} GROUP BY status",
                    tuple(params),
                ).fetchall():
                    stats["status_counts"][r["status"]] = r["c"]
                for r in conn.execute(
                    f"SELECT trigger_type, COUNT(*) c FROM llm_traces{where} GROUP BY trigger_type",
                    tuple(params),
                ).fetchall():
                    stats["trigger_counts"][r["trigger_type"]] = r["c"]
                for r in conn.execute(
                    f"SELECT COALESCE(model,'?') mdl, COUNT(*) c"
                    f" FROM llm_traces{where} GROUP BY model",
                    tuple(params),
                ).fetchall():
                    stats["model_counts"][r["mdl"]] = r["c"]
                if stats["total_turns"] > 0:
                    errors = stats["status_counts"].get("error", 0)
                    stats["recent_error_rate"] = round(100.0 * errors / stats["total_turns"], 1)
        except Exception as e:
            log.warning("[MonitorDB] Failed to get observability stats: %s", e)
        return stats


# Global singleton instance
from teams_server.config import MONITORING_DB  # noqa: E402

monitor_db = MonitoringDB(MONITORING_DB)
