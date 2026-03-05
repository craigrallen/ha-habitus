"""Tests verifying that all feature modules are wired into the post-train pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestPostAnalysisPipeline:
    """Verify _run_post_analysis calls every module and writes cache files."""

    def test_post_analysis_calls_all_modules(self, tmp_data_dir: Path):
        """After _run_post_analysis, all feature module functions are called."""
        import habitus.habitus.main as main_mod

        state = {"phase": "training"}
        stat_ids: list[str] = []

        mock_routine = MagicMock()
        mock_routine.run = MagicMock(return_value={"routines": [], "total": 0})

        mock_battery = MagicMock()
        mock_battery.run_battery_check = MagicMock(return_value={"total": 0, "summary": {}, "batteries": []})
        mock_battery.save_battery_status = MagicMock()

        mock_int_health = MagicMock()
        mock_int_health.run_integration_health_check = MagicMock(return_value={"total_entities": 0})
        mock_int_health.save_integration_health = MagicMock()

        mock_changelog_mod = MagicMock()
        mock_changelog_mod.run_diff_and_log = MagicMock(return_value=[])

        mock_conflict = MagicMock()
        mock_conflict.detect_conflicts = MagicMock(return_value=[])
        mock_conflict.save_conflicts = MagicMock()

        mock_auto_health = MagicMock()
        mock_auto_health.run_health_check = MagicMock(return_value={"total": 0, "automations": []})
        mock_auto_health.save_health = MagicMock()

        mock_guest = MagicMock()
        mock_guest.run = MagicMock(return_value={"guest_probability": 0.0})

        mock_seasonal = MagicMock()
        mock_seasonal.run = MagicMock(return_value={"season": "winter", "suggestions": [], "total": 0})

        with patch.multiple(
            "habitus.habitus.main",
            _routine_builder=mock_routine,
            _battery_watchdog=mock_battery,
            _integration_health=mock_int_health,
            _changelog=mock_changelog_mod,
            conflict_detector=mock_conflict,
            _automation_health=mock_auto_health,
            _guest_mode=mock_guest,
            _seasonal_adapter=mock_seasonal,
            set_progress=MagicMock(),
            save_state=MagicMock(),
            mark_last_completed_progress=MagicMock(),
        ):
            main_mod._run_post_analysis(state, stat_ids)

        mock_routine.run.assert_called_once()
        mock_battery.run_battery_check.assert_called_once()
        mock_battery.save_battery_status.assert_called_once()
        mock_int_health.run_integration_health_check.assert_called_once()
        mock_int_health.save_integration_health.assert_called_once()
        mock_changelog_mod.run_diff_and_log.assert_called_once()
        mock_conflict.detect_conflicts.assert_called_once()
        mock_conflict.save_conflicts.assert_called_once()
        mock_auto_health.run_health_check.assert_called_once()
        mock_auto_health.save_health.assert_called_once()
        mock_guest.run.assert_called_once()
        mock_seasonal.run.assert_called_once()

    def test_post_analysis_failure_is_non_fatal(self, tmp_data_dir: Path, caplog):
        """A failure in one module does not abort the pipeline."""
        import habitus.habitus.main as main_mod

        state = {}
        stat_ids: list[str] = []

        with patch.multiple(
            "habitus.habitus.main",
            _routine_builder=MagicMock(run=MagicMock(side_effect=RuntimeError("boom"))),
            _battery_watchdog=MagicMock(
                run_battery_check=MagicMock(return_value={"total": 0, "summary": {}, "batteries": []}),
                save_battery_status=MagicMock(),
            ),
            _integration_health=MagicMock(
                run_integration_health_check=MagicMock(return_value={}),
                save_integration_health=MagicMock(),
            ),
            _changelog=MagicMock(run_diff_and_log=MagicMock(return_value=[])),
            conflict_detector=MagicMock(
                detect_conflicts=MagicMock(return_value=[]),
                save_conflicts=MagicMock(),
            ),
            _automation_health=MagicMock(
                run_health_check=MagicMock(return_value={}),
                save_health=MagicMock(),
            ),
            _guest_mode=MagicMock(run=MagicMock(return_value={})),
            _seasonal_adapter=MagicMock(run=MagicMock(return_value={})),
            set_progress=MagicMock(),
            save_state=MagicMock(),
            mark_last_completed_progress=MagicMock(),
        ):
            # Should NOT raise
            main_mod._run_post_analysis(state, stat_ids)

    def test_post_analysis_updates_progress_phases(self, tmp_data_dir: Path):
        """_run_post_analysis calls set_progress with post_analysis phase."""
        import habitus.habitus.main as main_mod

        progress_phases = []

        def capture_progress(phase, *args, **kwargs):
            progress_phases.append(phase)

        with patch.multiple(
            "habitus.habitus.main",
            _routine_builder=MagicMock(run=MagicMock(return_value={})),
            _battery_watchdog=MagicMock(
                run_battery_check=MagicMock(return_value={"total": 0, "summary": {}, "batteries": []}),
                save_battery_status=MagicMock(),
            ),
            _integration_health=MagicMock(
                run_integration_health_check=MagicMock(return_value={}),
                save_integration_health=MagicMock(),
            ),
            _changelog=MagicMock(run_diff_and_log=MagicMock(return_value=[])),
            conflict_detector=MagicMock(
                detect_conflicts=MagicMock(return_value=[]),
                save_conflicts=MagicMock(),
            ),
            _automation_health=MagicMock(
                run_health_check=MagicMock(return_value={}),
                save_health=MagicMock(),
            ),
            _guest_mode=MagicMock(run=MagicMock(return_value={})),
            _seasonal_adapter=MagicMock(run=MagicMock(return_value={})),
            set_progress=capture_progress,
            save_state=MagicMock(),
            mark_last_completed_progress=MagicMock(),
        ):
            main_mod._run_post_analysis({}, [])

        assert "post_analysis" in progress_phases
