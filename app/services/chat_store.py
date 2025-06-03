"""Simple SQLite-backed chat storage service."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import structlog


class ChatStore:
    """Manage persistent chat history using SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.logger = structlog.get_logger(__name__)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    folder TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def create_chat(self, title: str = "New Chat", folder: str = "") -> int:
        ts = datetime.utcnow().isoformat()
        with self._get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO chats (title, folder, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (title, folder, ts, ts),
            )
            conn.commit()
            return cur.lastrowid

    def delete_chat(self, chat_id: int) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
            conn.commit()

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        ts = datetime.utcnow().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, ts),
            )
            conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (ts, chat_id))
            conn.commit()

    def list_chats(self, folder: str = "") -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if folder:
                rows = conn.execute(
                    "SELECT id, title, folder, updated_at FROM chats WHERE folder=? ORDER BY updated_at DESC",
                    (folder,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, title, folder, updated_at FROM chats ORDER BY updated_at DESC"
                ).fetchall()

            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "folder": r["folder"],
                    "updated_at": r["updated_at"],
                    "message_count": self.count_messages(r["id"]),
                }
                for r in rows
            ]

    def list_folders(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT folder, COUNT(*) as cnt FROM chats WHERE folder != '' GROUP BY folder"
            ).fetchall()
            return [
                {"name": r["folder"], "count": r["cnt"]}
                for r in rows
            ]

    def search_chats(self, query: str, folder: str = "") -> List[Dict[str, Any]]:
        like = f"%{query}%"
        with self._get_conn() as conn:
            if folder:
                rows = conn.execute(
                    """
                    SELECT DISTINCT c.id, c.title, c.folder, c.updated_at
                    FROM chats c
                    LEFT JOIN messages m ON c.id = m.chat_id
                    WHERE c.folder=? AND (c.title LIKE ? OR m.content LIKE ?)
                    ORDER BY c.updated_at DESC
                    """,
                    (folder, like, like),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT DISTINCT c.id, c.title, c.folder, c.updated_at
                    FROM chats c
                    LEFT JOIN messages m ON c.id = m.chat_id
                    WHERE c.title LIKE ? OR m.content LIKE ?
                    ORDER BY c.updated_at DESC
                    """,
                    (like, like),
                ).fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "folder": r["folder"],
                    "updated_at": r["updated_at"],
                    "message_count": self.count_messages(r["id"]),
                }
                for r in rows
            ]

    def count_messages(self, chat_id: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,)
            ).fetchone()
            return int(row[0]) if row else 0

    def get_chat(self, chat_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            chat_row = conn.execute(
                "SELECT id, title, folder FROM chats WHERE id=?", (chat_id,)
            ).fetchone()
            if not chat_row:
                return None
            msg_rows = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE chat_id=? ORDER BY id",
                (chat_id,),
            ).fetchall()
            messages = [
                {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
                for r in msg_rows
            ]
            return {
                "id": chat_row["id"],
                "title": chat_row["title"],
                "folder": chat_row["folder"],
                "messages": messages,
                "rag_files": [],
            }

