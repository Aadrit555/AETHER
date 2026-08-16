from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from aether.core.config import settings

DB_PATH = settings.data_dir / "aether.db"
_lock = Lock()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _lock, connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                source TEXT NOT NULL,
                rows INTEGER NOT NULL DEFAULT 0,
                valid_rows INTEGER NOT NULL DEFAULT 0,
                invalid_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                result_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, status);
            """
        )
