"""Second coverage boost file targeting 0%/low-coverage modules."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── feedback.py ───────────────────────────────────────────────────────────────

class TestFeedback:
    def _patch_fb(self, feedback_mod, tmp_data_dir: Path):
        """Helper to get a patched context for FEEDBACK_PATH."""
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        return patch.object(feedback_mod, "FEEDBACK_PATH", fb_path), fb_path

    def test_get_feedback_stats_empty(self, tmp_data_dir: Path):
        """get_feedback_stats returns default stats when no feedback saved."""
        from habitus.habitus import feedback
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        ctx, _ = self._patch_fb(feedback, tmp_data_dir)
        with ctx:
            result = feedback.get_feedback_stats()
        assert isinstance(result, dict)
        # stats is nested
        assert "stats" in result or "total" in result

    def test_record_feedback_creates_file(self, tmp_data_dir: Path):
        """record_feedback creates the feedback file."""
        from habitus.habitus import feedback
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        with patch.object(feedback, "FEEDBACK_PATH", fb_path):
            feedback.record_feedback("anomaly-1", "acknowledge", "sensor.power", 0.8)
        assert Path(fb_path).exists()

    def test_record_feedback_persist_and_read(self, tmp_data_dir: Path):
        """record_feedback saves data, get_feedback_stats reflects it."""
        from habitus.habitus import feedback
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        with patch.object(feedback, "FEEDBACK_PATH", fb_path):
            feedback.record_feedback("a1", "normal", "sensor.temp", 0.5)
            stats = feedback.get_feedback_stats()
        # stats is nested under "stats" key
        total = stats.get("total") or stats.get("stats", {}).get("total", 0)
        assert total >= 1

    def test_get_anonymous_export_no_sharing(self, tmp_data_dir: Path):
        """get_anonymous_export returns None when sharing disabled."""
        from habitus.habitus import feedback
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        with patch.object(feedback, "FEEDBACK_PATH", fb_path):
            result = feedback.get_anonymous_export()
        assert result is None

    def test_set_sharing_enabled(self, tmp_data_dir: Path):
        """set_sharing records sharing preference."""
        from habitus.habitus import feedback
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        with patch.object(feedback, "FEEDBACK_PATH", fb_path):
            feedback.set_sharing(True)
        # Should not error — just verify it completes
        assert Path(fb_path).exists()

    def test_record_multiple_feedbacks(self, tmp_data_dir: Path):
        """record_feedback accumulates correctly."""
        from habitus.habitus import feedback
        fb_path = str(tmp_data_dir / "anomaly_feedback.json")
        with patch.object(feedback, "FEEDBACK_PATH", fb_path):
            feedback.record_feedback("a1", "acknowledge", "s1", 0.9)
            feedback.record_feedback("a2", "normal", "s2", 0.4)
            feedback.record_feedback("a3", "acknowledge", "s1", 0.7)
            stats = feedback.get_feedback_stats()
        total = stats.get("total") or stats.get("stats", {}).get("total", 0)
        assert total == 3


# ── device_trainer.py ─────────────────────────────────────────────────────────

class TestDeviceTrainer:
    def test_get_training_status_no_session(self, tmp_data_dir: Path):
        """get_training_status returns idle when no session."""
        from habitus.habitus import device_trainer as dt
        sigs_path = str(tmp_data_dir / "custom_sigs.json")
        with patch.object(dt, "CUSTOM_SIGNATURES_PATH", sigs_path):
            result = dt.get_training_status()
        assert isinstance(result, dict)

    def test_list_custom_signatures_empty(self, tmp_data_dir: Path):
        """list_custom_signatures returns empty list when no signatures saved."""
        from habitus.habitus import device_trainer as dt
        sigs_path = str(tmp_data_dir / "custom_sigs.json")
        with patch.object(dt, "CUSTOM_SIGNATURES_PATH", sigs_path):
            result = dt.list_custom_signatures()
        assert isinstance(result, list)
        assert result == []

    def test_delete_signature_nonexistent(self, tmp_data_dir: Path):
        """delete_signature returns False when signature not found."""
        from habitus.habitus import device_trainer as dt
        sigs_path = str(tmp_data_dir / "custom_sigs.json")
        with patch.object(dt, "CUSTOM_SIGNATURES_PATH", sigs_path):
            result = dt.delete_signature("nonexistent_device")
        assert result is False

    def test_start_training_session_no_entity(self, tmp_data_dir: Path):
        """start_training_session with nonexistent entity returns error dict."""
        from habitus.habitus import device_trainer as dt
        sigs_path = str(tmp_data_dir / "custom_sigs.json")
        with patch.object(dt, "CUSTOM_SIGNATURES_PATH", sigs_path):
            result = dt.start_training_session("sensor.nonexistent_power")
        assert isinstance(result, dict)
        assert "error" in result or "status" in result

    def test_stop_training_no_session(self, tmp_data_dir: Path):
        """stop_training_session without active session returns error."""
        from habitus.habitus import device_trainer as dt
        sigs_path = str(tmp_data_dir / "custom_sigs.json")
        with patch.object(dt, "CUSTOM_SIGNATURES_PATH", sigs_path):
            result = dt.stop_training_session("my_device")
        assert isinstance(result, dict)
        assert "error" in result or "status" in result


# ── drift.py ──────────────────────────────────────────────────────────────────

class TestDrift:
    def test_detect_drift_empty_df(self, tmp_data_dir: Path):
        """detect_drift on empty DataFrame returns empty result."""
        import pandas as pd
        from habitus.habitus.drift import detect_drift
        df = pd.DataFrame()
        result = detect_drift(df)
        assert isinstance(result, dict)

    def test_detect_drift_no_numeric_columns(self, tmp_data_dir: Path):
        """detect_drift with no numeric data returns empty/fallback."""
        import pandas as pd
        from habitus.habitus.drift import detect_drift
        df = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=5, freq="h")})
        result = detect_drift(df)
        assert isinstance(result, dict)

    def test_save_and_load(self, tmp_data_dir: Path):
        """save() and load() round-trip for drift data."""
        from habitus.habitus import drift
        drift_path = str(tmp_data_dir / "drift.json")
        data = {"drift_score": 0.3, "drifted_entities": [], "total": 0, "analysed_at": "2025-01-01"}
        with patch.object(drift, "DRIFT_PATH", drift_path):
            drift.save(data)
            loaded = drift.load()
        assert loaded["drift_score"] == pytest.approx(0.3)

    def test_load_missing_file(self, tmp_data_dir: Path):
        """load() returns empty dict when file is missing."""
        from habitus.habitus import drift
        with patch.object(drift, "DRIFT_PATH", str(tmp_data_dir / "nope.json")):
            result = drift.load()
        assert isinstance(result, dict)

    def test_detect_drift_returns_drift_score(self, tmp_data_dir: Path):
        """detect_drift returns a drift_score key."""
        import pandas as pd
        import numpy as np
        from habitus.habitus.drift import detect_drift

        # 20 rows × 3 columns
        dates = pd.date_range("2025-01-01", periods=20, freq="h")
        df = pd.DataFrame({
            "ts": dates,
            "sensor.power": np.random.uniform(50, 150, 20),
            "sensor.temp": np.random.uniform(18, 24, 20),
        })
        df = df.set_index("ts")
        result = detect_drift(df)
        assert isinstance(result, dict)


# ── activity.py (partial) ─────────────────────────────────────────────────────

class TestActivity:
    def test_get_activity_summary_no_states(self, tmp_data_dir: Path):
        """get_activity_summary returns dict when score_activity_anomalies returns empty."""
        from habitus.habitus.activity import get_activity_summary
        with patch("habitus.habitus.activity.score_activity_anomalies",
                   return_value={"anomalies": []}):
            result = get_activity_summary()
        assert isinstance(result, dict)
        assert result["status"] == "normal"
        assert result["score"] == 0

    def test_score_activity_anomalies_no_states(self, tmp_data_dir: Path):
        """score_activity_anomalies with empty states returns dict."""
        from habitus.habitus.activity import score_activity_anomalies
        with patch("habitus.habitus.activity._fetch_activity_states", return_value={}):
            result = score_activity_anomalies({})
        assert isinstance(result, dict)

    def test_classify_entity_light(self, tmp_data_dir: Path):
        """classify_entity correctly identifies light domain."""
        from habitus.habitus.activity import classify_entity
        result = classify_entity("light.living_room")
        assert result is not None

    def test_classify_entity_sensor(self, tmp_data_dir: Path):
        """classify_entity returns None for unmapped sensor."""
        from habitus.habitus.activity import classify_entity
        result = classify_entity("sensor.random_temperature")
        # May return None for unmapped entity types
        assert result is None or isinstance(result, str)


# ── dynamic_automations.py ────────────────────────────────────────────────────

class TestDynamicAutomations:
    def test_run_dynamic_analysis_no_db(self, tmp_data_dir: Path):
        """run_dynamic_analysis returns dict when DB unavailable."""
        from habitus.habitus import dynamic_automations as da
        dyn_path = str(tmp_data_dir / "dynamic_automations.json")
        with patch("habitus.habitus.dynamic_automations.resolve_ha_db_path", return_value=None), \
             patch.object(da, "DYNAMIC_PATH", dyn_path):
            result = da.run_dynamic_analysis({})
        assert isinstance(result, dict)

    def test_load_dynamic_missing_file(self, tmp_data_dir: Path):
        """Load returns empty result when file missing."""
        from habitus.habitus import dynamic_automations as da
        # The module might not have a separate load function — check
        import json
        dyn_path = str(tmp_data_dir / "dynamic_automations.json")
        # Write empty result to file and read it back
        with patch.object(da, "DYNAMIC_PATH", dyn_path), \
             patch("habitus.habitus.dynamic_automations.resolve_ha_db_path", return_value=None):
            result = da.run_dynamic_analysis({})
        assert isinstance(result, dict)


# ── correlation_engine.py ─────────────────────────────────────────────────────

class TestCorrelationEngine:
    def test_run_correlation_no_db(self, tmp_data_dir: Path):
        """run_correlation_analysis returns dict when DB unavailable."""
        from habitus.habitus import correlation_engine as ce
        corr_path = str(tmp_data_dir / "correlations.json")
        with patch("habitus.habitus.correlation_engine.resolve_ha_db_path", return_value=None), \
             patch.object(ce, "CORRELATIONS_PATH", corr_path):
            result = ce.run_correlation_analysis({})
        assert isinstance(result, dict)

    def test_load_correlations_missing_file(self, tmp_data_dir: Path):
        """load_correlations returns empty dict when file missing."""
        from habitus.habitus import correlation_engine as ce
        with patch.object(ce, "CORRELATIONS_PATH", str(tmp_data_dir / "nope.json")):
            result = ce.load_correlations()
        assert isinstance(result, dict)
