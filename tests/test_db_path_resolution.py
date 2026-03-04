from __future__ import annotations

import sqlite3

from habitus.habitus.appliance_fingerprint import detect_power_steps
from habitus.habitus.ha_db import resolve_ha_db_path
from habitus.habitus.scene_detector import _get_state_changes


def test_resolve_ha_db_path_prefers_explicit_env(monkeypatch, tmp_path):
    db1 = tmp_path / "explicit.db"
    db2 = tmp_path / "legacy.db"
    db1.touch()
    db2.touch()

    monkeypatch.setenv("HABITUS_HA_DB", str(db1))
    monkeypatch.setenv("HA_DB_PATH", str(db2))

    assert resolve_ha_db_path() == str(db1)


def test_scene_detector_reads_from_habitus_ha_db_env(monkeypatch, tmp_path):
    db = tmp_path / "ha.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT)")
    conn.execute(
        "CREATE TABLE states (metadata_id INTEGER, state TEXT, last_changed_ts REAL)"
    )
    conn.execute(
        "INSERT INTO states_meta (metadata_id, entity_id) VALUES (1, 'light.kitchen')"
    )
    conn.execute(
        "INSERT INTO states (metadata_id, state, last_changed_ts) VALUES (1, 'on', strftime('%s','now'))"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))

    rows = _get_state_changes(days=1)
    assert rows
    assert rows[0]["entity_id"] == "light.kitchen"


def test_appliance_fingerprint_reads_from_habitus_ha_db_env(monkeypatch, tmp_path):
    db = tmp_path / "ha_steps.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT)")
    conn.execute(
        "CREATE TABLE states (metadata_id INTEGER, state TEXT, last_changed_ts REAL)"
    )
    conn.execute(
        "INSERT INTO states_meta (metadata_id, entity_id) VALUES (1, 'sensor.main_power_w')"
    )

    base = 1_700_000_000
    readings = [100, 100, 300, 300, 90, 90, 320, 320, 100, 100, 310, 310]
    for idx, val in enumerate(readings):
        conn.execute(
            "INSERT INTO states (metadata_id, state, last_changed_ts) VALUES (1, ?, ?)",
            (str(val), base + idx * 60),
        )

    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))

    steps = detect_power_steps("sensor.main_power_w", days=3650)
    assert steps
    assert any(s["direction"] == "up" for s in steps)
    assert any(abs(s["delta_w"]) >= 150 for s in steps)
