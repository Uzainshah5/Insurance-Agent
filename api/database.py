"""
database.py — SQLite storage for conversation sessions and messages.

Uses Python's built-in sqlite3 module (no external dependencies needed).
Stores data in 'chat_history.db' in the project root.
"""

import sqlite3
import uuid
from pathlib import Path

# Path to SQLite database file
DB_PATH = Path(__file__).resolve().parent.parent / "chat_history.db"


def get_connection():
    """Return a database connection with dictionary-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the sessions and messages tables if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            );
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Sessions Operations
# ---------------------------------------------------------------------------

def get_all_sessions():
    """Return all conversation sessions ordered by newest first."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, created_at 
            FROM sessions 
            ORDER BY created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def create_session(session_id: str = None, title: str = "New Chat"):
    """Create a new chat session."""
    if not session_id:
        session_id = str(uuid.uuid4())

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title) VALUES (?, ?)",
            (session_id, title),
        )
        conn.commit()

    return {"id": session_id, "title": title}


def get_session(session_id: str):
    """Retrieve a single session by its ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_session_title(session_id: str, title: str):
    """Update a session's title (e.g. from 'New Chat' to the user question)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )
        conn.commit()


def delete_session(session_id: str):
    """Delete a session and all its associated messages."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Messages Operations
# ---------------------------------------------------------------------------

def get_session_messages(session_id: str):
    """Retrieve all messages for a given session in chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content 
            FROM messages 
            WHERE session_id = ? 
            ORDER BY id ASC
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]


def add_message(session_id: str, role: str, content: str):
    """Save a single message turn (user or assistant) into SQLite."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()

