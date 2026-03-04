"""Recorder DB path resolution helpers.

Centralizes Home Assistant SQLite recorder path discovery across runtime modules.
"""

from __future__ import annotations

import os


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
