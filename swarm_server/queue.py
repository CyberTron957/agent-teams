"""SQLite-backed task queue per agent."""

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("swarm.queue")


class TaskQueue:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS tasks (
        id           TEXT PRIMARY KEY,
        from_agent   TEXT NOT NULL,
        payload      TEXT NOT NULL,
        status       TEXT NOT NULL DEFAULT 'pending',
        created_at   REAL NOT NULL,
        processed_at REAL,
        retries      INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at);
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        # WAL lets readers and a writer coexist without "database is locked";
        # busy_timeout makes brief write contention wait instead of erroring.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
            # Migrate older DBs that predate the retries column.
            cols = [c[1] for c in conn.execute("PRAGMA table_info(tasks)").fetchall()]
            if "retries" not in cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN retries INTEGER NOT NULL DEFAULT 0")
            conn.commit()

    def enqueue(self, from_agent: str, payload: str) -> str:
        task_id = str(uuid.uuid4())
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks (id, from_agent, payload, status, created_at) VALUES (?,?,?,?,?)",
                (task_id, from_agent, payload, "pending", time.time()),
            )
            conn.commit()
        log.info("[Queue] Enqueued task %s from '%s'", task_id[:8], from_agent)
        return task_id

    def drain_pending(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Atomically claim up to ``limit`` pending tasks (0 = no cap).

        The cap is backpressure: it bounds how many messages get concatenated
        into a single LLM turn so a flood can't blow the context window.
        """
        with self._lock, self._conn() as conn:
            sql = "SELECT id, from_agent, payload, retries FROM tasks WHERE status='pending' ORDER BY created_at"
            if limit and limit > 0:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql).fetchall()
            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE tasks SET status='processing', processed_at=? WHERE id IN ({placeholders})",
                    [time.time()] + ids,
                )
                conn.commit()
        return [{"id": r[0], "from_agent": r[1], "payload": r[2], "retries": r[3]} for r in rows]

    def mark_done(self, task_id: str):
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
            conn.commit()

    def requeue(self, task_ids: List[str]):
        """Return tasks to 'pending' and bump their retry counter (after a failure)."""
        if not task_ids:
            return
        with self._lock, self._conn() as conn:
            placeholders = ",".join("?" * len(task_ids))
            conn.execute(
                f"UPDATE tasks SET status='pending', processed_at=NULL, retries=retries+1 "
                f"WHERE id IN ({placeholders})",
                task_ids,
            )
            conn.commit()

    def requeue_no_penalty(self, task_ids: List[str]):
        """Return tasks to 'pending' WITHOUT bumping the retry counter.

        Used for infrastructure failures (LLM proxy down, billing exhausted)
        that are not the task's fault — the work should wait for recovery and
        resume, not burn its retry budget and dead-letter during an outage."""
        if not task_ids:
            return
        with self._lock, self._conn() as conn:
            placeholders = ",".join("?" * len(task_ids))
            conn.execute(
                f"UPDATE tasks SET status='pending', processed_at=NULL "
                f"WHERE id IN ({placeholders})",
                task_ids,
            )
            conn.commit()

    def mark_failed(self, task_ids: List[str]):
        """Dead-letter tasks that exhausted their retries."""
        if not task_ids:
            return
        with self._lock, self._conn() as conn:
            placeholders = ",".join("?" * len(task_ids))
            conn.execute(
                f"UPDATE tasks SET status='failed', processed_at=? WHERE id IN ({placeholders})",
                [time.time()] + task_ids,
            )
            conn.commit()

    def recover_processing(self) -> int:
        """Requeue tasks left 'processing' by a previous run that crashed/restarted.

        Without this, a restart would either lose in-flight tasks (old behavior
        deleted the DB) or strand them forever in 'processing'. Returns the count
        recovered.
        """
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status='pending', processed_at=NULL WHERE status='processing'"
            )
            conn.commit()
            return cur.rowcount or 0

    def get_pending_count(self) -> int:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='pending'").fetchone()
            return row[0] if row else 0

    def get_all_tasks(self, limit: int = 50) -> List[dict]:
        with self._lock, self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
