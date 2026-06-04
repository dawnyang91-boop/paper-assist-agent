import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class SQLiteCheckpointStore:
    """Small SQLite checkpoint store for AgentState snapshots."""

    def __init__(self, db_path: str = "data/checkpoints/agent_checkpoints.sqlite3", enabled: bool = False):
        self.db_path = Path(db_path)
        self.enabled = enabled
        if self.enabled:
            self._init_db()

    def save(self, session_id: str, node: str, snapshot: Dict[str, Any]) -> Optional[int]:
        if not self.enabled:
            return None
        self._init_db()
        payload = json.dumps(snapshot, ensure_ascii=False, default=str)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO checkpoints (session_id, node, ts, payload)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, node, time.time(), payload),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, session_id, node, ts, payload
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def list(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, node, ts, payload
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    ts REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id, id)")
            conn.commit()

    def _row_to_checkpoint(self, row) -> Dict[str, Any]:
        checkpoint_id, session_id, node, ts, payload = row
        return {
            "id": checkpoint_id,
            "session_id": session_id,
            "node": node,
            "ts": ts,
            "payload": json.loads(payload),
        }
