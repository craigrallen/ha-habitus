"""Recorder DB path and managed SQLite read helpers.

Centralizes Home Assistant recorder path discovery plus read-only SQLite access
patterns shared by analytics modules.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any


class _ReadPool:
    """Small process-local pool for read-only recorder SQLite connections."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: dict[str, sqlite3.Connection] = {}

    def get(self, db_path: str) -> sqlite3.Connection:
        with self._lock:
            existing = self._connections.get(db_path)
            if existing is not None:
                try:
                    existing.execute("SELECT 1")
                    return existing
                except sqlite3.Error:
                    try:
                        existing.close()
                    except sqlite3.Error:
                        pass
                    self._connections.pop(db_path, None)

            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                timeout=10,
                check_same_thread=False,
            )
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA journal_mode=OFF")
            self._connections[db_path] = conn
            return conn

    def close_all(self) -> None:
        with self._lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._connections.clear()


_READ_POOL = _ReadPool()


def resolve_ha_db_path() -> str | None:
    """Return first existing recorder DB path.

    Resolution order:
    1. HABITUS_HA_DB (explicit override)
    2. HA_DB_PATH (legacy env override)
    3. Common HA add-on/container mount points
    """
    configured = os.environ.get("HABITUS_HA_DB", "").strip()
    legacy = os.environ.get("HA_DB_PATH", "").strip()

    candidates = (
        configured,
        legacy,
        "/homeassistant/home-assistant_v2.db",
        "/config/home-assistant_v2.db",
        "/mnt/data/supervisor/homeassistant/home-assistant_v2.db",
    )
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def get_pooled_read_connection(db_path: str | None = None) -> sqlite3.Connection | None:
    """Get a pooled read-only SQLite connection for the recorder DB."""
    path = db_path or resolve_ha_db_path()
    if not path:
        return None
    return _READ_POOL.get(path)


@contextmanager
def managed_read_connection(db_path: str | None = None) -> Iterator[sqlite3.Connection | None]:
    """Yield a pooled recorder read connection, or None when DB is unavailable."""
    conn = get_pooled_read_connection(db_path)
    yield conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True when table exists in the SQLite schema."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def fetch_rows(
    conn: sqlite3.Connection,
    query: str,
    params: Sequence[Any] | None = None,
) -> list[tuple[Any, ...]]:
    """Execute a read query and return all rows."""
    if params is None:
        return conn.execute(query).fetchall()
    return conn.execute(query, tuple(params)).fetchall()


def close_pooled_connections() -> None:
    """Close all pooled DB connections (mainly for tests/process shutdown)."""
    _READ_POOL.close_all()
