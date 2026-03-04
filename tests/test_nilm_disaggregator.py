from __future__ import annotations

import sqlite3

from habitus.habitus.nilm_disaggregator import _auto_detect_power_entity


def test_auto_detect_power_entity_with_states_meta(tmp_path):
    db = tmp_path / "ha_meta.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT)")
    conn.execute("INSERT INTO states_meta (metadata_id, entity_id) VALUES (1, 'sensor.house_power_w')")
    conn.execute("INSERT INTO states_meta (metadata_id, entity_id) VALUES (2, 'sensor.temp')")
    conn.commit()
    conn.close()

    assert _auto_detect_power_entity(str(db)) == "sensor.house_power_w"


def test_auto_detect_power_entity_with_legacy_states_table(tmp_path):
    db = tmp_path / "ha_legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_changed_ts REAL)")
    conn.execute(
        "INSERT INTO states (entity_id, state, last_changed_ts) VALUES ('sensor.main_electric_w', '100', 1)"
    )
    conn.execute(
        "INSERT INTO states (entity_id, state, last_changed_ts) VALUES ('sensor.temperature', '22', 1)"
    )
    conn.commit()
    conn.close()

    assert _auto_detect_power_entity(str(db)) == "sensor.main_electric_w"
