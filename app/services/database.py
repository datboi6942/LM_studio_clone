import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from app.config import Config


class Database:
    """Simple SQLite wrapper for chat persistence."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or Config.DATABASE_URL.replace("sqlite:///", "")
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if not self._conn:
            db_path = Path(self.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self.connect()

    def init_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS folders (
            name TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            folder TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(folder) REFERENCES folders(name) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        """
        self.conn.executescript(sql)
        self.conn.commit()

    # Folder operations
    def create_folder(self, name: str) -> None:
        """Create a folder if it doesn't already exist."""
        self.conn.execute(
            "INSERT OR IGNORE INTO folders(name) VALUES (?)",
            (name,)
        )
        self.conn.commit()

    def list_folders(self) -> List[Dict[str, Any]]:
        """Return folders with chat counts."""
        rows = self.conn.execute(
            "SELECT f.name, COUNT(c.id) AS count "
            "FROM folders f LEFT JOIN chats c ON c.folder = f.name "
            "GROUP BY f.name ORDER BY f.name"
        ).fetchall()
        return [dict(row) for row in rows]

    # Chat operations
    def create_chat(self, title: str = "New Chat", folder: str | None = None) -> int:
        now = datetime.utcnow().isoformat()
        if folder:
            self.create_folder(folder)
        cur = self.conn.execute(
            "INSERT INTO chats(title, folder, created_at, updated_at) VALUES (?,?,?,?)",
            (title, folder, now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def delete_chat(self, chat_id: int) -> None:
        self.conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        self.conn.commit()

    def rename_chat(self, chat_id: int, title: str) -> None:
        self.conn.execute(
            "UPDATE chats SET title=?, updated_at=? WHERE id=?",
            (title, datetime.utcnow().isoformat(), chat_id),
        )
        self.conn.commit()

    def list_chats(
        self, folder: str | None = None, search: str | None = None
    ) -> List[Dict[str, Any]]:
        query = (
            "SELECT c.id, c.title, c.folder, c.created_at, c.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.chat_id=c.id) AS message_count "
            "FROM chats c"
        )
        conditions: list[str] = []
        params: list[Any] = []
        if folder:
            conditions.append("c.folder = ?")
            params.append(folder)
        if search:
            conditions.append(
                "(c.title LIKE ? OR EXISTS(SELECT 1 FROM messages m WHERE m.chat_id=c.id AND m.content LIKE ?))"
            )
            params.extend([f"%{search}%", f"%{search}%"])
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.updated_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: int) -> Dict[str, Any]:
        chat = self.conn.execute(
            "SELECT id, title, folder, created_at, updated_at FROM chats WHERE id=?",
            (chat_id,),
        ).fetchone()
        if not chat:
            raise KeyError(chat_id)
        messages = self.conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE chat_id=? ORDER BY id",
            (chat_id,),
        ).fetchall()
        return {
            "id": chat["id"],
            "title": chat["title"],
            "folder": chat["folder"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"],
            "messages": [dict(m) for m in messages],
            "rag_files": [],
        }

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO messages(chat_id, role, content, timestamp) VALUES (?,?,?,?)",
            (chat_id, role, content, now),
        )
        self.conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
        self.conn.commit()
