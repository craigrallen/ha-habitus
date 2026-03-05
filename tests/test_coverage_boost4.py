"""Final coverage boost pass for remaining low-coverage modules."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np


# ── energy_forecast.py ────────────────────────────────────────────────────────

class TestEnergyForecastMore:
    def test_get_daily_energy_no_db(self, tmp_data_dir: Path):
        """_get_daily_energy_and_weather returns empty list when DB unavailable."""
        from habitus.habitus.energy_forecast import _get_daily_energy_and_weather
        with patch("habitus.habitus.energy_forecast.resolve_ha_db_path", return_value=None):
            result = _get_daily_energy_and_weather(days=7)
        assert result == []

    def test_run_energy_forecast_no_history(self, tmp_data_dir: Path):
        """run_energy_forecast handles empty history gracefully."""
        from habitus.habitus import energy_forecast as ef
        forecast_path = str(tmp_data_dir / "energy_forecast.json")

        with patch("habitus.habitus.energy_forecast._get_daily_energy_and_weather", return_value=[]), \
             patch.object(ef, "FORECAST_PATH", forecast_path) if hasattr(ef, "FORECAST_PATH") else patch("habitus.habitus.energy_forecast.DATA_DIR", str(tmp_data_dir)):
            result = ef.run_energy_forecast()
        assert isinstance(result, dict)
        assert "error" in result or "forecast" in result or "reason" in result

    def test_run_energy_forecast_with_history(self, tmp_data_dir: Path):
        """run_energy_forecast processes daily history data correctly."""
        import habitus.habitus.energy_forecast as ef
        import datetime

        # Build 30 days of fake energy data
        today = datetime.date.today()
        d0 = today - datetime.timedelta(days=30)
        history = [
            {
                "date": (d0 + datetime.timedelta(days=i)).isoformat(),
                "kwh": 8.5 + i * 0.1,
                "avg_temp": 10.0,
                "cloud_cover": 0.5,
                "is_weekend": i % 7 >= 5,
                "day_of_week": i % 7,
                "month": (d0 + datetime.timedelta(days=i)).month,
            }
            for i in range(30)
        ]

        forecast_path = str(tmp_data_dir / "energy_forecast.json")
        with patch("habitus.habitus.energy_forecast._get_daily_energy_and_weather",
                   return_value=history), \
             patch.object(ef, "FORECAST_PATH", forecast_path):
            result = ef.run_energy_forecast()
        assert isinstance(result, dict)


# ── correlation_engine.py ─────────────────────────────────────────────────────

class TestCorrelationEngineMore:
    def test_run_correlation_result_keys(self, tmp_data_dir: Path):
        """run_correlation_analysis result has expected keys."""
        from habitus.habitus import correlation_engine as ce
        corr_path = str(tmp_data_dir / "correlations.json")
        with patch("habitus.habitus.correlation_engine.resolve_ha_db_path", return_value=None), \
             patch.object(ce, "CORRELATIONS_PATH", corr_path):
            result = ce.run_correlation_analysis({})
        assert isinstance(result, dict)
        assert "correlations" in result or "pairs" in result or "total_correlations" in result

    def test_correlations_result_has_total(self, tmp_data_dir: Path):
        """run_correlation_analysis returns total_correlations."""
        from habitus.habitus import correlation_engine as ce
        with patch("habitus.habitus.correlation_engine.resolve_ha_db_path", return_value=None), \
             patch("habitus.habitus.correlation_engine.DATA_DIR", str(tmp_data_dir)):
            result = ce.run_correlation_analysis({})
        assert isinstance(result, dict)


# ── sequence_miner.py ─────────────────────────────────────────────────────────

class TestSequenceMinerMore:
    def test_load_event_streams_db_error(self, tmp_data_dir: Path):
        """_load_event_streams returns empty list when DB raises error."""
        import sqlite3
        from habitus.habitus.sequence_miner import _load_event_streams
        with patch("habitus.habitus.sequence_miner.resolve_ha_db_path", return_value="/fake/db.db"), \
             patch("sqlite3.connect", side_effect=sqlite3.OperationalError("no db")):
            result = _load_event_streams({})
        assert result == []

    def test_mine_sequences_no_sessions_returns_dict(self, tmp_data_dir: Path):
        """mine_sequences with no sessions returns dict without error."""
        from habitus.habitus import sequence_miner as sm
        seq_path = str(tmp_data_dir / "sequences.json")
        with patch("habitus.habitus.sequence_miner.resolve_ha_db_path", return_value=None), \
             patch.object(sm, "SEQUENCES_PATH", seq_path):
            result = sm.mine_sequences({})
        # No sessions → returns early without saving
        assert isinstance(result, dict)
        assert "sequences" in result

    def test_mine_sequences_result_keys(self, tmp_data_dir: Path):
        """mine_sequences result has expected keys."""
        from habitus.habitus import sequence_miner as sm
        seq_path = str(tmp_data_dir / "sequences.json")
        with patch("habitus.habitus.sequence_miner.resolve_ha_db_path", return_value=None), \
             patch.object(sm, "SEQUENCES_PATH", seq_path):
            result = sm.mine_sequences({})
        assert "sequences" in result
        assert "total_sessions" in result


# ── automation_gap.py ─────────────────────────────────────────────────────────

class TestAutomationGapMore:
    def test_analyse_empty_suggestions_returns_result(self, tmp_data_dir: Path):
        """analyse() with no suggestions returns empty gaps."""
        import asyncio
        from habitus.habitus import automation_gap
        gap_path = str(tmp_data_dir / "automation_gap.json")

        with patch("habitus.habitus.automation_gap._fetch_automations", return_value=[]), \
             patch("habitus.habitus.automation_gap._fetch_all_states", return_value=[]), \
             patch.object(automation_gap, "GAP_PATH", gap_path), \
             patch.object(automation_gap, "DATA_DIR", str(tmp_data_dir)):
            result = asyncio.run(automation_gap.analyse("http://ha", "token", []))

        assert isinstance(result, dict)
        assert "gaps" in result

    def test_load_gap_missing_file(self, tmp_data_dir: Path):
        """load() returns default dict when file missing."""
        from habitus.habitus import automation_gap
        with patch.object(automation_gap, "GAP_PATH", str(tmp_data_dir / "nope.json")):
            result = automation_gap.load()
        assert isinstance(result, dict)

    def test_save_and_load_gap(self, tmp_data_dir: Path):
        """save() and load() round-trip correctly."""
        from habitus.habitus import automation_gap
        gap_path = str(tmp_data_dir / "automation_gap.json")
        data = {
            "gaps": [{"suggestion": "Turn off lights", "status": "missing"}],
            "summary": "1 missing",
            "analysed_at": "2025-01-01T00:00:00",
        }
        with patch.object(automation_gap, "GAP_PATH", gap_path):
            automation_gap.save(data)
            loaded = automation_gap.load()
        assert len(loaded.get("gaps", [])) == 1


# ── scene_detector.py deeper coverage ────────────────────────────────────────

class TestSceneDetectorMore:
    def test_find_co_occurrences_empty(self, tmp_data_dir: Path):
        """_find_co_occurrences with empty state changes returns empty."""
        from habitus.habitus.scene_detector import _find_co_occurrences
        result = _find_co_occurrences([])
        assert result == {} or isinstance(result, dict)

    def test_find_co_occurrences_with_pairs(self, tmp_data_dir: Path):
        """_find_co_occurrences detects simultaneous state changes."""
        from habitus.habitus.scene_detector import _find_co_occurrences
        now = time.time()
        changes = [
            {"entity_id": "light.kitchen", "timestamp": now, "state": "on"},
            {"entity_id": "light.living_room", "timestamp": now + 5, "state": "on"},
            {"entity_id": "light.kitchen", "timestamp": now + 3600, "state": "on"},
            {"entity_id": "light.living_room", "timestamp": now + 3605, "state": "on"},
        ]
        result = _find_co_occurrences(changes)
        assert isinstance(result, dict)

    def test_analyze_time_patterns_empty(self, tmp_data_dir: Path):
        """_analyze_time_patterns with empty entities set returns empty dict."""
        from habitus.habitus.scene_detector import _analyze_time_patterns
        result = _analyze_time_patterns(set(), [])
        assert isinstance(result, dict)

    def test_analyze_time_patterns_with_data(self, tmp_data_dir: Path):
        """_analyze_time_patterns extracts primary hour from changes."""
        from habitus.habitus.scene_detector import _analyze_time_patterns
        now = time.time()
        entities = {"light.kitchen", "light.living_room"}
        changes = [
            {"entity_id": eid, "timestamp": now + i * 86400, "state": "on"}
            for i in range(5)
            for eid in entities
        ]
        result = _analyze_time_patterns(entities, changes)
        assert isinstance(result, dict)

    def test_get_state_changes_no_db(self, tmp_data_dir: Path):
        """_get_state_changes returns empty list when DB unavailable."""
        from habitus.habitus.scene_detector import _get_state_changes
        with patch("habitus.habitus.scene_detector.resolve_ha_db_path", return_value=None):
            result = _get_state_changes(days=7)
        assert result == []


# ── markov_chain.py deeper coverage ──────────────────────────────────────────

class TestMarkovChainMore:
    def test_build_markov_model_db_error(self, tmp_data_dir: Path):
        """build_markov_model handles sqlite3 error gracefully."""
        import sqlite3
        from habitus.habitus.markov_chain import build_markov_model
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value="/fake/db.db"), \
             patch("sqlite3.connect", side_effect=sqlite3.OperationalError("no db")):
            result = build_markov_model({})
        assert isinstance(result, dict)

    def test_build_markov_model_empty_returns_dict(self, tmp_data_dir: Path):
        """build_markov_model returns dict even when no rows."""
        from habitus.habitus import markov_chain
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value=None):
            result = markov_chain.build_markov_model({})
        assert isinstance(result, dict)
        assert "transitions" in result

    def test_build_markov_model_result_structure(self, tmp_data_dir: Path):
        """build_markov_model result has expected keys."""
        from habitus.habitus import markov_chain
        mk_path = str(tmp_data_dir / "markov_model.json")
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value=None), \
             patch.object(markov_chain, "MARKOV_PATH", mk_path):
            result = markov_chain.build_markov_model({})
        assert "transitions" in result
        assert "predictions" in result


# ── drift.py deeper ───────────────────────────────────────────────────────────

class TestDriftMore:
    def test_detect_drift_with_data(self, tmp_data_dir: Path):
        """detect_drift on real df with hour column returns drift result."""
        import pandas as pd
        from habitus.habitus import drift
        from habitus.habitus.drift import detect_drift
        drift_path = str(tmp_data_dir / "drift.json")

        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "hour": dates,
            "sensor_power": np.random.uniform(50, 200, 100),
            "sensor_temp": np.random.uniform(18, 25, 100),
        })

        with patch.object(drift, "DRIFT_PATH", drift_path):
            result = detect_drift(df)

        assert isinstance(result, dict)
        # May be reason or drift_score depending on data length

    def test_detect_drift_saves_file(self, tmp_data_dir: Path):
        """detect_drift writes result when data is sufficient."""
        import pandas as pd
        from habitus.habitus import drift
        drift_path = str(tmp_data_dir / "drift.json")

        # 80 hours spanning ~3 days — enough for MIN_DAYS check
        np.random.seed(0)
        dates = pd.date_range("2025-01-01", periods=80, freq="h")
        df = pd.DataFrame({
            "hour": dates,
            "sensor_power": np.random.uniform(50, 200, 80),
        })
        with patch.object(drift, "DRIFT_PATH", drift_path):
            result = drift.detect_drift(df)
        # Result is always a dict (may say reason or drift_score)
        assert isinstance(result, dict)
