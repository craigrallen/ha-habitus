from __future__ import annotations

import sqlite3
import time

from habitus.habitus.correlation_engine import _get_state_change_events
from habitus.habitus.ha_db import close_pooled_connections, get_pooled_read_connection
from habitus.habitus.room_predictor import _get_actions_after_entry, _get_room_entry_events
from habitus.habitus.routine_predictor import _find_humidity_sensors, _get_humidity_history
from habitus.habitus.scene_detector import _get_state_changes


def _create_meta_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT)")
    conn.execute("CREATE TABLE states (metadata_id INTEGER, state TEXT, last_changed_ts REAL)")
    conn.commit()
    conn.close()


def _create_legacy_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE states (entity_id TEXT, state TEXT, last_changed_ts REAL)")
    conn.commit()
    conn.close()


def test_pooled_read_connection_reuses_connection(monkeypatch, tmp_path):
    db = tmp_path / "ha.db"
    _create_legacy_db(db)
    monkeypatch.setenv("HABITUS_HA_DB", str(db))

    conn1 = get_pooled_read_connection()
    conn2 = get_pooled_read_connection()
    assert conn1 is not None
    assert conn1 is conn2

    close_pooled_connections()


def test_scene_detector_get_state_changes_meta_schema(monkeypatch, tmp_path):
    db = tmp_path / "ha_meta_scene.db"
    _create_meta_db(db)
    now = time.time()

    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO states_meta (metadata_id, entity_id) VALUES (1, 'light.kitchen_main')")
    conn.execute("INSERT INTO states_meta (metadata_id, entity_id) VALUES (2, 'switch.tv_power')")
    conn.execute("INSERT INTO states (metadata_id, state, last_changed_ts) VALUES (1, 'on', ?)", (now - 60,))
    conn.execute("INSERT INTO states (metadata_id, state, last_changed_ts) VALUES (2, 'unavailable', ?)", (now - 50,))
    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))
    changes = _get_state_changes(days=1)

    assert len(changes) == 1
    assert changes[0]["entity_id"] == "light.kitchen_main"
    assert changes[0]["state"] == "on"

    close_pooled_connections()


def test_routine_predictor_humidity_sensors_and_history(monkeypatch, tmp_path):
    db = tmp_path / "ha_meta_routine.db"
    _create_meta_db(db)
    now = time.time()

    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO states_meta (metadata_id, entity_id) VALUES (1, 'sensor.bathroom_humidity')")
    conn.execute("INSERT INTO states_meta (metadata_id, entity_id) VALUES (2, 'sensor.kitchen_hum')")
    conn.execute("INSERT INTO states (metadata_id, state, last_changed_ts) VALUES (1, '45.5', ?)", (now - 300,))
    conn.execute("INSERT INTO states (metadata_id, state, last_changed_ts) VALUES (1, 'bad', ?)", (now - 100,))
    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))

    sensors = _find_humidity_sensors()
    assert sensors["sensor.bathroom_humidity"] == "bathroom"
    assert sensors["sensor.kitchen_hum"] == "kitchen"

    readings = _get_humidity_history("sensor.bathroom_humidity", days=1)
    assert readings == [(now - 300, 45.5)]

    close_pooled_connections()


def test_room_predictor_reads_entry_and_actions_legacy(monkeypatch, tmp_path):
    db = tmp_path / "ha_legacy_room.db"
    _create_legacy_db(db)
    now = time.time()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO states (entity_id, state, last_changed_ts) VALUES ('binary_sensor.kitchen_motion', 'on', ?)",
        (now - 120,),
    )
    conn.execute(
        "INSERT INTO states (entity_id, state, last_changed_ts) VALUES ('light.kitchen_main', 'on', ?)",
        (now - 90,),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))
    entity_to_area = {
        "binary_sensor.kitchen_motion": "Kitchen",
        "light.kitchen_main": "Kitchen",
    }

    entries = _get_room_entry_events(entity_to_area, days=1)
    assert len(entries) == 1

    actions = _get_actions_after_entry(entries[0]["timestamp"], "Kitchen", entity_to_area)
    assert len(actions) == 1
    assert actions[0]["entity_id"] == "light.kitchen_main"

    close_pooled_connections()


def test_correlation_engine_reads_legacy_state_events(monkeypatch, tmp_path):
    db = tmp_path / "ha_legacy_corr.db"
    _create_legacy_db(db)
    now = time.time()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO states (entity_id, state, last_changed_ts) VALUES ('binary_sensor.hall_motion', 'on', ?)",
        (now - 120,),
    )
    conn.execute(
        "INSERT INTO states (entity_id, state, last_changed_ts) VALUES ('light.hall_main', 'on', ?)",
        (now - 60,),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))
    events = _get_state_change_events(days=1)

    assert "binary_sensor.hall_motion" in events
    assert "light.hall_main" in events
    assert len(events["light.hall_main"]) == 1

    close_pooled_connections()
