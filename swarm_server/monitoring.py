"""Central monitoring database for events and messages."""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("swarm.monitoring")


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
        task_id     TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_events_agent     ON events(agent_name);
    CREATE INDEX IF NOT EXISTS idx_events_time      ON events(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_messages_agent   ON messages(agent_name);
    CREATE INDEX IF NOT EXISTS idx_messages_time    ON messages(timestamp DESC);
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()
        self._migrate_add_team_id()

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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_team ON messages(team_id)")
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
                       (timestamp, agent_name, team_id, role, content, task_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (time.time(), agent_name, team_id, role, content, task_id),
                )
                conn.commit()
        except Exception as e:
            log.warning("[MonitorDB] Failed to log message: %s", e)

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
                    sql += " AND team_id = ?"
                    params.append(team_id)
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

    def prune(self, max_events: int, max_messages: int) -> Dict[str, int]:
        """Trim events/messages to the most recent N rows each (rolling retention).

        Without this the DB grows unbounded on a 24/7 run (state-change and
        message volume dominate). Returns rows deleted per table.
        """
        deleted = {"events": 0, "messages": 0}
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "DELETE FROM events WHERE id NOT IN "
                    "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
                    (max_events,),
                )
                deleted["events"] = cur.rowcount or 0
                cur = conn.execute(
                    "DELETE FROM messages WHERE id NOT IN "
                    "(SELECT id FROM messages ORDER BY id DESC LIMIT ?)",
                    (max_messages,),
                )
                deleted["messages"] = cur.rowcount or 0
                conn.commit()
            if deleted["events"] or deleted["messages"]:
                log.info(
                    "[MonitorDB] Pruned %d events, %d messages",
                    deleted["events"], deleted["messages"],
                )
        except Exception as e:
            log.warning("[MonitorDB] Prune failed: %s", e)
        return deleted

    def get_agent_stats(self, team_id: Optional[str] = None) -> Dict[str, dict]:
        stats = {}
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                sql = (
                    "SELECT agent_name, event_type, COUNT(*) as count FROM events"
                )
                params = []
                if team_id:
                    sql += " WHERE team_id = ?"
                    params.append(team_id)
                sql += " GROUP BY agent_name, event_type"
                rows = conn.execute(sql, tuple(params)).fetchall()
                for r in rows:
                    aname = r["agent_name"]
                    if aname not in stats:
                        stats[aname] = {
                            "events": {},
                            "last_active": None,
                            "total_messages": 0,
                        }
                    stats[aname]["events"][r["event_type"]] = r["count"]

                sql_last = "SELECT agent_name, MAX(timestamp) as last_ts FROM events"
                params_last = []
                if team_id:
                    sql_last += " WHERE team_id = ?"
                    params_last.append(team_id)
                sql_last += " GROUP BY agent_name"
                rows = conn.execute(sql_last, tuple(params_last)).fetchall()
                for r in rows:
                    if r["agent_name"] in stats:
                        stats[r["agent_name"]]["last_active"] = r["last_ts"]

                sql_msg = "SELECT agent_name, COUNT(*) as count FROM messages"
                params_msg = []
                if team_id:
                    sql_msg += " WHERE team_id = ?"
                    params_msg.append(team_id)
                sql_msg += " GROUP BY agent_name"
                rows = conn.execute(sql_msg, tuple(params_msg)).fetchall()
                for r in rows:
                    if r["agent_name"] in stats:
                        stats[r["agent_name"]]["total_messages"] = r["count"]
        except Exception as e:
            log.warning("[MonitorDB] Failed to get stats: %s", e)
        return stats


# Global singleton instance
from swarm_server.config import MONITORING_DB  # noqa: E402

monitor_db = MonitoringDB(MONITORING_DB)
