from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Optional


class AuthStore:
    """Pader-compatible auth store backed by the shared paper-tinder SQLite database."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path.as_posix(),
            check_same_thread=False,
            isolation_level=None,
            timeout=30,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT, role TEXT, institution TEXT, avatar_url TEXT, interests TEXT, research_fields TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS auth_users (user_id TEXT PRIMARY KEY, email TEXT UNIQUE, phone TEXT, password_hash TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS folders (id TEXT PRIMARY KEY, name TEXT, creator_id TEXT, creator_handle TEXT, is_public INTEGER DEFAULT 1, likes INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS folder_papers (folder_id TEXT, paper_id TEXT, added_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS folder_likes (folder_id TEXT, user_id TEXT, status INTEGER DEFAULT 1, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_cache (paper_id TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER, citation_count INTEGER, venue TEXT, url TEXT)"
        )
        self._ensure_column("users", "research_fields", "TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_folders_creator ON folders (creator_id)")
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def create_user_with_auth(
        self,
        name: str,
        email: str,
        phone: Optional[str],
        password: str,
        avatar_url: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        if self.get_user_by_email(normalized_email):
            raise ValueError("email_exists")

        user_id = f"u_{uuid.uuid4().hex[:12]}"
        user = {
            "id": user_id,
            "name": name.strip(),
            "role": "Researcher",
            "institution": "",
            "avatar_url": avatar_url or "",
            "interests": [],
            "research_fields": [],
        }
        with self._lock:
            self.upsert_user(user)
            self._conn.execute(
                "INSERT INTO auth_users (user_id, email, phone, password_hash) VALUES (?, ?, ?, ?)",
                (user_id, normalized_email, phone or "", hash_password(password)),
            )
            self._conn.commit()
        return self.get_user_profile(user_id) or self._profile(user, normalized_email)

    def register(
        self,
        name: str,
        email: str,
        phone: Optional[str],
        password: str,
        avatar_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """PaderAuthClient-compatible local development entrypoint."""
        return self.create_user_with_auth(
            name=name,
            email=email,
            phone=phone,
            password=password,
            avatar_url=avatar_url,
        )

    def login(self, email: str, password: str) -> Optional[dict[str, Any]]:
        record = self.get_user_by_email(email)
        if not record:
            return None
        if record.get("password_hash") != hash_password(password):
            return None
        return record["profile"]

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM auth_users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if not row:
            return None
        profile = self.get_user_profile(row["user_id"])
        if not profile:
            return None
        return {
            "profile": profile,
            "email": row["email"],
            "phone": row["phone"],
            "password_hash": row["password_hash"],
        }

    def get_user_profile(self, user_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        auth = self._conn.execute("SELECT email FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
        user = {
            "id": row["id"],
            "name": row["name"] or "",
            "role": row["role"] or "",
            "institution": row["institution"] or "",
            "avatar_url": row["avatar_url"] or "",
            "interests": _loads_list(row["interests"]),
            "research_fields": _loads_list(row["research_fields"]),
        }
        return self._profile(user, auth["email"] if auth else None)

    def upsert_user(self, user: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO users (id, name, role, institution, avatar_url, interests, research_fields) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, role=excluded.role, institution=excluded.institution, avatar_url=excluded.avatar_url, interests=excluded.interests, research_fields=excluded.research_fields",
            (
                user["id"],
                user.get("name", ""),
                user.get("role", ""),
                user.get("institution", ""),
                user.get("avatar_url", ""),
                json.dumps(user.get("interests", []), ensure_ascii=False),
                json.dumps(user.get("research_fields", []), ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def _profile(self, user: dict[str, Any], email: Optional[str]) -> dict[str, Any]:
        return {
            "id": user["id"],
            "email": email,
            "name": user.get("name", ""),
            "role": user.get("role", ""),
            "institution": user.get("institution", ""),
            "avatarUrl": user.get("avatar_url", ""),
            "stats": {
                "liked": self._liked_count(user["id"]),
                "streak": 0,
                "folders": self._folder_count(user["id"]),
            },
            "interests": user.get("interests", []),
            "researchFields": user.get("research_fields", []),
        }

    def _liked_count(self, user_id: str) -> int:
        if not self._table_exists("feedback"):
            return 0
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM feedback WHERE user_id = ? AND action = 'like'",
                (user_id,),
            ).fetchone()
            return int(row["count"] if row else 0)
        except sqlite3.Error:
            return 0

    def _folder_count(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM folders WHERE creator_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["count"] if row else 0)

    def _table_exists(self, table: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _loads_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [str(item) for item in data]
    return []
