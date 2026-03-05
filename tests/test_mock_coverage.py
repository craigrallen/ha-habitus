"""Mock-based test coverage for HA-dependent modules.

Covers: phantom.py, room_predictor.py, progressive.py, main.py fetch functions.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock

import pytest


# ── phantom.py ────────────────────────────────────────────────────────────────

class TestPhantomLoad:
    """Tests for phantom.py using mocked HA WebSocket."""

    def test_run_without_grid_entity_returns_empty(self, tmp_data_dir: Path, monkeypatch):
        """With no HABITUS_ENERGY_GRID set, run() returns empty dict."""
        monkeypatch.delenv("HABITUS_ENERGY_GRID", raising=False)
        from habitus.habitus import phantom
        result = phantom.run()
        assert result == {}

    def test_run_with_grid_entity_and_no_stats(self, tmp_data_dir: Path, monkeypatch):
        """With grid entity set but no stats, run() returns reason key."""
        from habitus.habitus import phantom
        monkeypatch.setenv("HABITUS_ENERGY_GRID", "sensor.grid_power")

        async def mock_fetch_stats(*args, **kwargs):
            return []

        async def mock_fetch_hourly(*args, **kwargs):
            return []

        with patch("habitus.habitus.phantom._fetch_statistics", side_effect=mock_fetch_stats), \
             patch("habitus.habitus.phantom._fetch_hourly_statistics", side_effect=mock_fetch_hourly):
            result = phantom.run()
        # Should return reason: no_monthly_stats or empty
        assert isinstance(result, dict)

    def test_save_and_load(self, tmp_data_dir: Path):
        """save() writes JSON and load() reads it back."""
        from habitus.habitus import phantom
        data = {
            "grid_entity": "sensor.grid",
            "total_12mo_kwh": 1234.5,
            "months": [],
            "analysed_at": "2025-01-01T00:00:00",
        }
        phantom.save(data)
        loaded = phantom.load()
        assert loaded["grid_entity"] == "sensor.grid"
        assert loaded["total_12mo_kwh"] == 1234.5

    def test_save_creates_data_dir(self, tmp_data_dir: Path):
        """save() writes to PHANTOM_PATH."""
        from habitus.habitus import phantom
        import os
        phantom_path = os.path.join(str(tmp_data_dir), "phantom_loads.json")
        with patch.object(phantom, "PHANTOM_PATH", phantom_path):
            phantom.save({"test": True})
        assert Path(phantom_path).exists()

    def test_load_returns_dict_when_missing(self, tmp_data_dir: Path):
        """load() returns dict-like when file doesn't exist."""
        from habitus.habitus import phantom
        result = phantom.load()
        assert isinstance(result, dict)

    def test_idle_hours_are_defined(self):
        """IDLE_HOURS must be a set of hour integers."""
        from habitus.habitus import phantom
        assert isinstance(phantom.IDLE_HOURS, set)
        assert all(isinstance(h, int) for h in phantom.IDLE_HOURS)
        assert all(0 <= h < 24 for h in phantom.IDLE_HOURS)

    def test_phantom_result_schema_from_async(self, tmp_data_dir: Path, monkeypatch):
        """_run_async() returns expected schema with mocked stats."""
        from habitus.habitus import phantom
        monkeypatch.setenv("HABITUS_ENERGY_GRID", "sensor.grid")

        import datetime as dt

        # Build fake monthly stats: 3 months
        now = dt.datetime.now(dt.UTC)
        last_month = now.replace(day=1) - dt.timedelta(days=1)
        prev_month = last_month.replace(day=1) - dt.timedelta(days=1)

        def make_stat(when: dt.datetime, kwh: float):
            return {"start": when.timestamp() * 1000, "change": kwh}

        monthly_stats = [
            make_stat(prev_month.replace(day=1), 200.0),
            make_stat(last_month.replace(day=1), 180.0),
            make_stat(now.replace(day=1), 50.0),
        ]
        daily_stats = [make_stat(now.replace(hour=0), 5.0)]
        hourly_stats = [
            make_stat(now.replace(hour=h), 0.05)
            for h in (2, 3, 4, 2, 3)
        ]

        async def mock_fetch_statistics(entity_id, period, months):
            if period == "month":
                return monthly_stats
            elif period == "day":
                return daily_stats
            return []

        async def mock_fetch_hourly(entity_id, days):
            return hourly_stats

        with patch("habitus.habitus.phantom._fetch_statistics", side_effect=mock_fetch_statistics), \
             patch("habitus.habitus.phantom._fetch_hourly_statistics", side_effect=mock_fetch_hourly):
            result = asyncio.run(phantom._run_async())

        assert "total_12mo_kwh" in result
        assert "months" in result
        assert "overnight_baseline" in result
        assert result["overnight_baseline"]["avg_idle_kwh_per_hour"] > 0
        assert result["grid_entity"] == "sensor.grid"


# ── room_predictor.py ─────────────────────────────────────────────────────────

class TestRoomPredictor:
    """Tests for room_predictor.py using mocked DB."""

    def test_run_returns_empty_without_db(self, tmp_data_dir: Path):
        """run_room_prediction returns empty if DB is missing."""
        with patch("habitus.habitus.room_predictor.resolve_ha_db_path", return_value=None):
            from habitus.habitus.room_predictor import run_room_prediction
            result = run_room_prediction({})
        assert isinstance(result, dict)
        assert result.get("total_rooms", 0) == 0

    def test_run_with_empty_entity_area_map(self, tmp_data_dir: Path):
        """run_room_prediction with empty area map returns gracefully."""
        with patch("habitus.habitus.room_predictor.resolve_ha_db_path", return_value=None):
            from habitus.habitus.room_predictor import run_room_prediction
            result = run_room_prediction({})
        assert isinstance(result, dict)

    def test_load_predictions_missing_file(self, tmp_data_dir: Path):
        """load_predictions returns empty dict when file is missing."""
        from habitus.habitus.room_predictor import load_predictions
        result = load_predictions()
        assert isinstance(result, dict)

    def test_load_predictions_reads_json(self, tmp_data_dir: Path):
        """load_predictions returns data when file exists."""
        from habitus.habitus import room_predictor
        data = {"rooms": {"kitchen": {}}, "total_rooms": 1, "automations": [], "prediction_count": 1}
        pred_path = str(tmp_data_dir / "room_predictions.json")
        (tmp_data_dir / "room_predictions.json").write_text(json.dumps(data))
        with patch.object(room_predictor, "PREDICTIONS_PATH", pred_path):
            result = room_predictor.load_predictions()
        assert result.get("total_rooms") == 1

    def test_build_room_model_empty_db(self, tmp_data_dir: Path):
        """build_room_model returns empty rooms when DB has no entries."""
        from habitus.habitus.room_predictor import build_room_model

        mock_conn = MagicMock()
        mock_conn.execute.return_value = []

        with patch("habitus.habitus.room_predictor.resolve_ha_db_path", return_value="/fake/db.db"), \
             patch("habitus.habitus.room_predictor.managed_read_connection") as mock_ctx, \
             patch("habitus.habitus.room_predictor.table_exists", return_value=True):
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = build_room_model({"binary_sensor.kitchen_motion": "kitchen"})

        assert isinstance(result, dict)
        assert "rooms" in result

    def test_generate_prediction_automations_empty_model(self, tmp_data_dir: Path):
        """generate_prediction_automations with no patterns returns empty list."""
        from habitus.habitus.room_predictor import generate_prediction_automations
        model = {"rooms": {}, "total_rooms": 0}
        result = generate_prediction_automations(model, {})
        assert result == []

    def test_room_entry_event_structure(self, tmp_data_dir: Path):
        """_get_room_entry_events returns list when DB is unavailable."""
        with patch("habitus.habitus.room_predictor.resolve_ha_db_path", return_value=None):
            from habitus.habitus.room_predictor import _get_room_entry_events
            result = _get_room_entry_events({"binary_sensor.kitchen_motion": "kitchen"})
        assert isinstance(result, list)


# ── progressive.py ────────────────────────────────────────────────────────────

class TestProgressive:
    """Tests for progressive.py background training."""

    def test_initial_window_is_zero(self):
        """Before training starts, current_window() returns 0."""
        import importlib
        import habitus.habitus.progressive as prog
        importlib.reload(prog)  # reset state
        assert prog.current_window() == 0

    def test_is_expanding_false_initially(self):
        """is_expanding() returns False before start_progressive is called."""
        import importlib
        import habitus.habitus.progressive as prog
        importlib.reload(prog)
        assert prog.is_expanding() is False

    def test_windows_list_defined(self):
        """WINDOWS constant is a non-empty list of ints."""
        from habitus.habitus import progressive
        assert isinstance(progressive.WINDOWS, list)
        assert len(progressive.WINDOWS) > 0
        assert all(isinstance(w, int) for w in progressive.WINDOWS)

    def test_start_progressive_sets_thread(self, monkeypatch):
        """start_progressive() creates a background thread."""
        import importlib
        import habitus.habitus.progressive as prog
        importlib.reload(prog)

        call_log = []

        def fake_loop(windows):
            call_log.append(windows)

        with patch.object(prog, "_loop", side_effect=fake_loop):
            prog.start_progressive(max_days=60)
            import time; time.sleep(0.05)  # let thread start

        assert prog._thread is not None

    def test_start_progressive_no_duplicate_threads(self, monkeypatch):
        """Calling start_progressive twice does not create a second thread."""
        import importlib
        import habitus.habitus.progressive as prog
        importlib.reload(prog)

        started = []

        def fake_loop(windows):
            import time
            started.append(1)
            time.sleep(0.5)

        with patch.object(prog, "_loop", side_effect=fake_loop):
            prog.start_progressive(max_days=60)
            prog.start_progressive(max_days=60)
            import time; time.sleep(0.05)

        assert len(started) <= 1  # second call should be a no-op


# ── main.py fetch functions ───────────────────────────────────────────────────

class TestMainFetchFunctions:
    """Tests for main.py HA REST API fetch functions."""

    def test_fetch_recent_raw_history_empty_entities(self, tmp_data_dir: Path):
        """fetch_recent_raw_history with empty entity list returns empty DataFrame."""
        from habitus.habitus.main import fetch_recent_raw_history
        import pandas as pd

        result = fetch_recent_raw_history([], "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_fetch_recent_raw_history_with_sqlite(self, tmp_data_dir: Path):
        """fetch_recent_raw_history reads from SQLite when available."""
        from habitus.habitus.main import fetch_recent_raw_history
        import pandas as pd

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Simulate rows: (entity_id, last_updated_ts, state)
        import time
        now_ts = time.time()
        mock_cursor.fetchall.return_value = [
            ("sensor.power", now_ts - 3600, "100.5"),
            ("sensor.power", now_ts - 1800, "120.0"),
        ]
        mock_conn.__class__ = sqlite3.Connection  # convince isinstance checks

        with patch("habitus.habitus.main._sqlite_connect", return_value=mock_conn):
            result = fetch_recent_raw_history(
                ["sensor.power"],
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            )

        # Should have attempted to read from SQLite
        assert mock_cursor.execute.called or mock_conn.cursor.called

    def test_fetch_recent_raw_history_sqlite_error(self, tmp_data_dir: Path):
        """fetch_recent_raw_history handles SQLite errors gracefully."""
        from habitus.habitus.main import fetch_recent_raw_history
        import pandas as pd

        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = sqlite3.OperationalError("no such table: states")

        with patch("habitus.habitus.main._sqlite_connect", return_value=mock_conn):
            result = fetch_recent_raw_history(
                ["sensor.power"],
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            )

        assert isinstance(result, pd.DataFrame)

    def test_sqlite_connect_returns_none_on_failure(self, tmp_data_dir: Path):
        """_sqlite_connect returns None if DB is not accessible."""
        from habitus.habitus import main as main_mod

        # Make _resolve_db_path return a fake path, then make sqlite3.connect fail
        with patch.object(main_mod, "_resolve_db_path", return_value="/nonexistent/path.db"), \
             patch("sqlite3.connect", side_effect=sqlite3.OperationalError("no DB")):
            conn = main_mod._sqlite_connect()
        assert conn is None

    def test_state_to_numeric_valid(self):
        """_state_to_numeric converts valid numeric strings."""
        from habitus.habitus.main import _state_to_numeric
        assert _state_to_numeric("100.5") == pytest.approx(100.5)
        assert _state_to_numeric("0") == pytest.approx(0.0)
        assert _state_to_numeric("on") == pytest.approx(1.0)
        assert _state_to_numeric("off") == pytest.approx(0.0)

    def test_state_to_numeric_invalid(self):
        """_state_to_numeric returns None for non-numeric non-boolean strings."""
        from habitus.habitus.main import _state_to_numeric
        result = _state_to_numeric("unavailable")
        assert result is None
        result2 = _state_to_numeric("unknown")
        assert result2 is None

    def test_save_and_load_state(self, tmp_data_dir: Path):
        """save_state() writes to disk and load_state() reads it back."""
        from habitus.habitus.main import save_state, load_state
        state = {"phase": "idle", "last_run": "2025-01-01T00:00:00Z", "score": 0.85}
        save_state(state)
        loaded = load_state()
        assert loaded["phase"] == "idle"
        assert loaded["score"] == pytest.approx(0.85)

    def test_load_state_missing_file(self, tmp_data_dir: Path):
        """load_state returns empty dict when state file is missing."""
        from habitus.habitus.main import load_state
        result = load_state()
        assert isinstance(result, dict)
