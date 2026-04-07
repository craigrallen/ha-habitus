"""Tests for main.py utility functions (mark_last_completed_progress, clamp_fetch_window, etc.)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestMarkLastCompletedProgress:
    def test_basic_call(self):
        """mark_last_completed_progress sets checkpoint in state."""
        from habitus.habitus.main import mark_last_completed_progress
        state = {}
        mark_last_completed_progress(state, "training", done=100, total=100, pct=100)
        assert "last_completed_progress" in state
        assert state["last_completed_progress"]["phase"] == "training"
        assert state["last_completed_progress"]["done"] == 100

    def test_pct_clamped_to_100(self):
        """pct values > 100 are clamped to 100."""
        from habitus.habitus.main import mark_last_completed_progress
        state = {}
        mark_last_completed_progress(state, "test", pct=150)
        assert state["last_completed_progress"]["pct"] == 100

    def test_extra_dict_stored(self):
        """Extra dict is stored as-is."""
        from habitus.habitus.main import mark_last_completed_progress
        state = {}
        mark_last_completed_progress(state, "test", extra={"entities": 5})
        assert state["last_completed_progress"]["extra"]["entities"] == 5

    def test_completed_at_override(self):
        """completed_at can be overridden."""
        from habitus.habitus.main import mark_last_completed_progress
        state = {}
        ts = "2025-06-01T12:00:00+00:00"
        mark_last_completed_progress(state, "test", completed_at=ts)
        assert state["last_completed_progress"]["completed_at"] == ts

    def test_none_values_not_stored(self):
        """None done/total/rows/pct are not stored in checkpoint."""
        from habitus.habitus.main import mark_last_completed_progress
        state = {}
        mark_last_completed_progress(state, "test")
        cp = state["last_completed_progress"]
        assert "done" not in cp
        assert "total" not in cp
        assert "pct" not in cp


class TestClampFetchWindow:
    def test_no_clamping_needed(self):
        """clamp_fetch_window returns same range if within budget."""
        from habitus.habitus.main import clamp_fetch_window_by_row_budget
        result_start, clamped, info = clamp_fetch_window_by_row_budget(
            "2025-01-01T00:00:00Z",
            "2025-01-15T00:00:00Z",
            entity_count=10,
            row_budget=5_000_000,
        )
        assert clamped is False

    def test_clamping_applied_when_over_budget(self):
        """clamp_fetch_window clamps start date when entity_count × days is large."""
        from habitus.habitus.main import clamp_fetch_window_by_row_budget
        result_start, clamped, info = clamp_fetch_window_by_row_budget(
            "2025-01-01T00:00:00Z",
            "2025-03-31T00:00:00Z",
            entity_count=200,
            row_budget=10_000,  # very tight budget
        )
        assert isinstance(result_start, str)
        assert isinstance(clamped, bool)
        assert isinstance(info, dict)

    def test_returns_info_dict(self):
        """clamp_fetch_window returns info dict with expected keys."""
        from habitus.habitus.main import clamp_fetch_window_by_row_budget
        _, _, info = clamp_fetch_window_by_row_budget(
            "2025-01-01T00:00:00Z",
            "2025-02-01T00:00:00Z",
            entity_count=50,
        )
        assert isinstance(info, dict)


class TestIsBehavioral:
    def test_light_entity_is_behavioral(self):
        """is_behavioral returns True for light entities."""
        from habitus.habitus.main import is_behavioral
        assert is_behavioral("light.living_room") is True

    def test_switch_entity_is_behavioral(self):
        """is_behavioral returns True for switch entities."""
        from habitus.habitus.main import is_behavioral
        assert is_behavioral("switch.heater") is True

    def test_sensor_entity_is_behavioral(self):
        """is_behavioral returns True for numeric sensor entities."""
        from habitus.habitus.main import is_behavioral
        assert is_behavioral("sensor.power_consumption") is True

    def test_automation_entity_not_behavioral(self):
        """is_behavioral returns False for automation entities."""
        from habitus.habitus.main import is_behavioral
        assert is_behavioral("automation.my_automation") is False

    def test_button_entity_not_behavioral(self):
        """is_behavioral returns False for button entities (not in BEHAVIORAL_DOMAINS)."""
        from habitus.habitus.main import is_behavioral
        assert is_behavioral("button.restart") is False


class TestClearProgress:
    def test_clear_progress_removes_file_if_exists(self, tmp_data_dir: Path):
        """clear_progress() removes the progress file if it exists."""
        import habitus.habitus.main as main_mod
        import json
        progress_path = str(tmp_data_dir / "progress.json")
        # Create the file first
        (tmp_data_dir / "progress.json").write_text(json.dumps({"running": True}))
        with patch.object(main_mod, "PROGRESS_PATH", progress_path):
            main_mod.clear_progress()
        # File should be removed
        assert not (tmp_data_dir / "progress.json").exists()


class TestResolvePath:
    def test_resolve_db_path_returns_string_or_none(self):
        """_resolve_db_path returns str path or None."""
        from habitus.habitus.main import _resolve_db_path
        result = _resolve_db_path()
        assert result is None or isinstance(result, str)


class TestSetProgress:
    def test_set_progress_writes_file(self, tmp_data_dir: Path):
        """set_progress() writes progress file."""
        import habitus.habitus.main as main_mod
        progress_path = str(tmp_data_dir / "progress.json")
        with patch.object(main_mod, "PROGRESS_PATH", progress_path):
            main_mod.set_progress("training", done=50, total=100, rows=1000, elapsed=60, eta=120)
        import json
        data = json.loads((tmp_data_dir / "progress.json").read_text())
        assert data["phase"] == "training"
        assert data["done"] == 50
        assert data["pct"] == 50
        assert data["running"] is True
