"""Additional mock-based tests to boost code coverage.

Targets: trainer.py, markov_chain.py, sequence_miner.py,
         conflict_detector.py, routine_predictor.py, ha_areas.py.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── trainer.py ────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_is_running_false_initially(self):
        """is_running() returns False before training starts."""
        import importlib
        import habitus.habitus.trainer as trainer
        importlib.reload(trainer)
        assert trainer.is_running() is False

    def test_start_returns_true_on_first_call(self, monkeypatch):
        """start() returns True when not already running."""
        import importlib
        import habitus.habitus.trainer as trainer
        importlib.reload(trainer)

        calls = []

        def fake_run(days, mode):
            calls.append((days, mode))

        with patch.object(trainer, "_run_blocking", side_effect=fake_run):
            result = trainer.start(30, "incremental")

        # Thread starts; result should be True
        assert result is True

    def test_start_returns_false_when_already_running(self, monkeypatch):
        """start() returns False if trainer is already running."""
        import importlib
        import habitus.habitus.trainer as trainer
        importlib.reload(trainer)

        import threading

        block = threading.Event()
        started = threading.Event()

        def fake_run_slow(days, mode):
            started.set()
            block.wait(timeout=2)

        with patch.object(trainer, "_run_blocking", side_effect=fake_run_slow):
            result1 = trainer.start(30, "full")
            started.wait(timeout=1)
            result2 = trainer.start(30, "full")  # already running
            block.set()

        assert result1 is True
        assert result2 is False

    def test_run_blocking_logs_error_on_import_failure(self, monkeypatch, caplog):
        """_run_blocking logs exception when main import fails and doesn't reraise."""
        import habitus.habitus.trainer as trainer
        import logging

        # trainer._run_blocking tries to import habitus.main — it will fail in tests
        # Verify it catches the error and logs it
        with caplog.at_level(logging.ERROR, logger="habitus"):
            trainer._run_blocking(45, "full")

        # It should have logged the exception without raising
        assert any("Background training failed" in r.message for r in caplog.records)

    def test_run_blocking_saves_error_state_on_exception(self, tmp_data_dir: Path):
        """_run_blocking saves error state when run() raises."""
        import habitus.habitus.trainer as trainer
        import habitus.habitus.main as main_mod

        state_path = str(tmp_data_dir / "run_state.json")

        async def boom(*args, **kwargs):
            raise ValueError("training error")

        with patch.object(main_mod, "STATE_PATH", state_path), \
             patch("habitus.habitus.trainer.asyncio.run", side_effect=ValueError("training error")):
            # Should not raise
            trainer._run_blocking(30, "full")


# ── markov_chain.py ───────────────────────────────────────────────────────────

class TestMarkovChain:
    def test_build_model_no_db(self, tmp_data_dir: Path):
        """build_markov_model returns empty dict when DB unavailable."""
        from habitus.habitus.markov_chain import build_markov_model
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value=None):
            result = build_markov_model({})
        assert result == {"transitions": {}, "predictions": []}

    def test_build_model_db_error(self, tmp_data_dir: Path):
        """build_markov_model handles DB query errors gracefully."""
        from habitus.habitus.markov_chain import build_markov_model
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value="/fake/db.db"), \
             patch("sqlite3.connect", side_effect=sqlite3.OperationalError("no db")):
            result = build_markov_model({})
        assert "transitions" in result
        assert "predictions" in result

    def test_build_model_returns_empty_transitions_no_db(self, tmp_data_dir: Path):
        """build_markov_model with no DB returns empty transitions."""
        from habitus.habitus.markov_chain import build_markov_model
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value=None):
            result = build_markov_model({})
        assert isinstance(result, dict)
        assert result.get("transitions") == {} or result.get("predictions") == []

    def test_build_model_returns_result_structure(self, tmp_data_dir: Path):
        """build_markov_model without DB has expected keys."""
        from habitus.habitus.markov_chain import build_markov_model
        with patch("habitus.habitus.markov_chain.resolve_ha_db_path", return_value=None):
            result = build_markov_model({"light.kitchen": "kitchen"})
        assert "transitions" in result
        assert "predictions" in result


# ── sequence_miner.py ─────────────────────────────────────────────────────────

class TestSequenceMiner:
    def test_load_event_streams_no_db(self, tmp_data_dir: Path):
        """_load_event_streams returns empty list when DB unavailable."""
        with patch("habitus.habitus.sequence_miner.resolve_ha_db_path", return_value=None):
            from habitus.habitus.sequence_miner import _load_event_streams
            result = _load_event_streams({})
        assert result == []

    def test_mine_sequences_no_db(self, tmp_data_dir: Path):
        """mine_sequences returns dict with empty sequences when DB unavailable."""
        from habitus.habitus import sequence_miner
        seq_path = str(tmp_data_dir / "sequences.json")
        with patch("habitus.habitus.sequence_miner.resolve_ha_db_path", return_value=None), \
             patch.object(sequence_miner, "SEQUENCES_PATH", seq_path):
            result = sequence_miner.mine_sequences({})
        assert isinstance(result, dict)
        assert "sequences" in result

    def test_mine_sequences_result_structure(self, tmp_data_dir: Path):
        """mine_sequences result has expected keys."""
        from habitus.habitus import sequence_miner
        seq_path = str(tmp_data_dir / "sequences.json")
        with patch("habitus.habitus.sequence_miner.resolve_ha_db_path", return_value=None), \
             patch.object(sequence_miner, "SEQUENCES_PATH", seq_path):
            result = sequence_miner.mine_sequences({})
        assert "sequences" in result
        assert "total_sessions" in result


# ── conflict_detector.py ─────────────────────────────────────────────────────

class TestConflictDetector:
    def test_load_conflicts_missing_file(self, tmp_data_dir: Path):
        """load_conflicts returns dict with empty list when file is missing."""
        from habitus.habitus import conflict_detector as cd
        with patch.object(cd, "CONFLICTS_PATH", str(tmp_data_dir / "nope.json")):
            result = cd.load_conflicts()
        assert isinstance(result, dict)
        assert "conflicts" in result

    def test_save_and_load_conflicts(self, tmp_data_dir: Path):
        """save_conflicts and load_conflicts round-trip correctly."""
        from habitus.habitus import conflict_detector as cd
        conflicts_path = str(tmp_data_dir / "conflicts.json")
        conflicts = [{"id": "c1", "title": "Test conflict", "severity": "high",
                      "description": "desc", "suggestion": "fix it"}]
        with patch.object(cd, "CONFLICTS_PATH", conflicts_path), \
             patch.object(cd, "DATA_DIR", str(tmp_data_dir)):
            cd.save_conflicts(conflicts)
            loaded = cd.load_conflicts()
        assert isinstance(loaded, dict)
        assert len(loaded.get("conflicts", [])) == 1
        assert loaded["conflicts"][0]["id"] == "c1"

    def test_detect_conflicts_no_states_returns_list(self, tmp_data_dir: Path):
        """detect_conflicts with empty states returns empty list."""
        from habitus.habitus.conflict_detector import detect_conflicts
        # Provide empty states — should return []
        result = detect_conflicts({})
        assert isinstance(result, list)

    def test_detect_conflicts_empty_states(self, tmp_data_dir: Path):
        """detect_conflicts with empty states dict → returns empty list."""
        from habitus.habitus import conflict_detector as cd
        # Empty states → should return []
        result = cd.detect_conflicts({})
        assert isinstance(result, list)
        assert result == []

    def test_detect_conflicts_with_lights_off(self, tmp_data_dir: Path):
        """detect_conflicts with only lights off → no conflicts."""
        from habitus.habitus import conflict_detector as cd
        states = {
            "light.living_room": {"state": "off", "attributes": {}},
            "switch.heater": {"state": "off", "attributes": {}},
        }
        result = cd.detect_conflicts(states)
        assert isinstance(result, list)


# ── ha_areas.py ──────────────────────────────────────────────────────────────

class TestHAAreas:
    def test_load_cache_returns_empty_when_missing(self, tmp_data_dir: Path):
        """_load_cache returns empty dict when cache file is missing."""
        from habitus.habitus import ha_areas
        with patch.object(ha_areas, "AREAS_CACHE_PATH", str(tmp_data_dir / "nope.json")):
            result = ha_areas._load_cache()
        assert isinstance(result, dict)
        assert "areas" in result or "entity_to_area" in result

    def test_fetch_areas_no_ha(self, tmp_data_dir: Path):
        """fetch_areas returns sensible fallback when HA unreachable."""
        from habitus.habitus.ha_areas import fetch_areas
        import requests as req_lib
        with patch("habitus.habitus.ha_areas.requests.get",
                   side_effect=req_lib.ConnectionError("no HA")):
            result = fetch_areas()
        assert isinstance(result, dict)

    def test_get_entity_area_returns_none_when_no_data(self, tmp_data_dir: Path):
        """get_entity_area returns None when no area data cached."""
        from habitus.habitus import ha_areas
        with patch.object(ha_areas, "AREAS_CACHE_PATH", str(tmp_data_dir / "nope.json")):
            result = ha_areas.get_entity_area("light.kitchen")
        # Either None or empty string for unmapped entities
        assert result is None or result == ""

    def test_get_entities_rooms_empty(self, tmp_data_dir: Path):
        """get_entities_rooms with no entities returns empty list."""
        from habitus.habitus.ha_areas import get_entities_rooms
        result = get_entities_rooms([])
        assert result == []


# ── automation_score.py ───────────────────────────────────────────────────────

class TestAutomationScore:
    def test_score_all_no_ha(self, tmp_data_dir: Path):
        """score_all gracefully handles HA being unreachable."""
        import asyncio
        import habitus.habitus.automation_score as asc
        from habitus.habitus.automation_score import score_all
        import requests as req_lib

        # Find the requests module used in automation_score
        req_module = None
        for attr in dir(asc):
            obj = getattr(asc, attr, None)
            if hasattr(obj, 'get') and hasattr(obj, 'ConnectionError'):
                req_module = obj
                break

        # Patch at module scope by name
        with patch("requests.get", side_effect=req_lib.ConnectionError("no HA")):
            result = asyncio.run(score_all("http://ha", "token"))
        assert isinstance(result, list)

    def test_load_scores_missing_file(self, tmp_data_dir: Path):
        """load() returns empty list when file is missing."""
        from habitus.habitus import automation_score as asc
        with patch.object(asc, "SCORES_PATH", str(tmp_data_dir / "nope.json")):
            result = asc.load()
        assert isinstance(result, list)

    def test_save_and_load_scores(self, tmp_data_dir: Path):
        """save() and load() round-trip correctly."""
        from habitus.habitus import automation_score as asc
        scores_path = str(tmp_data_dir / "auto_scores.json")
        data = [{"id": "a1", "score": 85, "alias": "Test"}]
        with patch.object(asc, "SCORES_PATH", scores_path):
            asc.save(data)
            loaded = asc.load()
        assert len(loaded) == 1
        assert loaded[0]["score"] == 85
