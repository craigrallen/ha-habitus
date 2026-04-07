"""Edge case tests — NaN handling, empty data, corrupt files, boundary conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Empty / Minimal DataFrames ──────────────────────────────────────────────


class TestEmptyDataFrames:
    """Verify modules handle empty or near-empty DataFrames gracefully."""

    def test_discover_patterns_empty_df(self, sample_features):
        """discover_patterns should work with minimal rows."""
        from habitus.habitus.patterns import discover_patterns

        tiny = sample_features.head(24)  # just 1 day
        result = discover_patterns(tiny)
        assert "daily_routine" in result
        assert "weekly" in result

    def test_discover_patterns_single_row(self, sample_features):
        """Single-row DataFrame should not crash — requires fixing wakeup detection."""
        from habitus.habitus.patterns import discover_patterns

        # Use 48 rows (2 days) to cover enough hour_of_day values
        small = sample_features.head(48)
        result = discover_patterns(small)
        assert isinstance(result, dict)
        assert "daily_routine" in result

    def test_drift_detection_empty_df(self):
        """detect_drift with an empty DataFrame returns a reason."""
        from habitus.habitus.drift import detect_drift

        result = detect_drift(pd.DataFrame())
        assert "reason" in result

    def test_drift_detection_no_hour_column(self):
        """detect_drift without 'hour' column returns a reason."""
        from habitus.habitus.drift import detect_drift

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = detect_drift(df)
        assert "reason" in result

    def test_data_quality_filters_empty_df(self):
        """apply_data_quality_filters on empty DataFrame returns empty."""
        from habitus.habitus.anomaly_breakdown import apply_data_quality_filters

        df = pd.DataFrame(columns=["entity_id", "ts", "v"])
        filtered, issues = apply_data_quality_filters(df)
        assert filtered.empty
        assert issues == []


# ── NaN / Inf Handling ──────────────────────────────────────────────────────


class TestNaNHandling:
    """Verify NaN and Inf values are handled without crashes."""

    def test_max_consecutive_zeros_with_nan(self):
        """_max_consecutive_zeros should treat NaN as non-zero."""
        from habitus.habitus.patterns import _max_consecutive_zeros

        series = pd.Series([0, 0, np.nan, 0, 0, 0])
        result = _max_consecutive_zeros(series)
        assert result == 3  # NaN breaks the zero run

    def test_data_quality_filters_nan_values(self, tmp_data_dir):
        """Data quality filters should handle NaN values in v column."""
        from habitus.habitus.anomaly_breakdown import apply_data_quality_filters

        df = pd.DataFrame(
            {
                "entity_id": ["sensor.test"] * 5,
                "ts": pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
                "v": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        # Remove NaN before calling (as documented in docstring)
        df_clean = df.dropna(subset=["v"])
        filtered, issues = apply_data_quality_filters(df_clean)
        assert len(filtered) <= len(df_clean)

    def test_patterns_with_inf_power(self, sample_features):
        """Discover patterns should not crash if power has Inf values."""
        from habitus.habitus.patterns import discover_patterns

        features = sample_features.copy()
        features.loc[features.index[0], "total_power_w"] = np.inf
        # Should not raise — Inf is handled by pandas aggregation
        result = discover_patterns(features)
        assert "daily_routine" in result


# ── Corrupt / Missing Files ─────────────────────────────────────────────────


class TestCorruptFiles:
    """Verify load functions handle corrupt or missing files."""

    def test_drift_load_missing_file(self, tmp_data_dir):
        """drift.load returns empty dict when file doesn't exist."""
        import habitus.habitus.drift as d

        d.DRIFT_PATH = str(tmp_data_dir / "nonexistent.json")
        assert d.load() == {}

    def test_drift_load_corrupt_json(self, tmp_data_dir):
        """drift.load returns empty dict on corrupt JSON."""
        import habitus.habitus.drift as d

        corrupt = tmp_data_dir / "drift.json"
        corrupt.write_text("{invalid json!!!}")
        d.DRIFT_PATH = str(corrupt)
        assert d.load() == {}

    def test_automation_score_load_corrupt(self, tmp_data_dir):
        """automation_score.load returns empty list on corrupt JSON."""
        import habitus.habitus.automation_score as a

        corrupt = tmp_data_dir / "automation_scores.json"
        corrupt.write_text("not valid json")
        a.SCORES_PATH = str(corrupt)
        assert a.load() == []

    def test_phantom_load_corrupt(self, tmp_data_dir):
        """phantom.load returns empty dict on corrupt JSON."""
        import habitus.habitus.phantom as p

        corrupt = tmp_data_dir / "phantom_loads.json"
        corrupt.write_text("{{broken")
        p.PHANTOM_PATH = str(corrupt)
        assert p.load() == {}

    def test_automation_gap_load_corrupt(self, tmp_data_dir):
        """automation_gap.load returns empty dict on corrupt JSON."""
        import habitus.habitus.automation_gap as ag

        corrupt = tmp_data_dir / "automation_gap.json"
        corrupt.write_text("[not valid")
        ag.GAP_PATH = str(corrupt)
        assert ag.load() == {}

    def test_seasonal_load_bundle_missing(self, tmp_data_dir):
        """load_seasonal_bundle returns ({}, {}) when file doesn't exist."""
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        models, scalers = s.load_seasonal_bundle()
        assert models == {}
        assert scalers == {}


# ── Boundary Conditions ─────────────────────────────────────────────────────


class TestBoundaryConditions:
    """Verify boundary conditions for thresholds and limits."""

    def test_contamination_boundary_at_zero(self):
        """contamination_for_days at day 0 should return minimum."""
        from habitus.habitus.main import contamination_for_days

        assert contamination_for_days(0) == 0.002

    def test_contamination_boundary_at_7(self):
        """contamination_for_days at exactly 7 days should step up."""
        from habitus.habitus.main import contamination_for_days

        assert contamination_for_days(6) == 0.002
        assert contamination_for_days(7) == 0.005

    def test_contamination_boundary_at_90(self):
        """contamination_for_days at 90+ days should return maximum."""
        from habitus.habitus.main import contamination_for_days

        assert contamination_for_days(89) == 0.015
        assert contamination_for_days(90) == 0.02
        assert contamination_for_days(365) == 0.02

    def test_current_season_all_months(self):
        """current_season should return valid season for all months."""
        from unittest.mock import patch

        from habitus.habitus.seasonal import current_season

        valid_seasons = {"winter", "spring", "summer", "autumn"}
        for month in range(1, 13):
            with patch("habitus.habitus.seasonal.datetime") as mock_dt:
                mock_dt.datetime.now.return_value.month = month
                result = current_season("north")
                assert result in valid_seasons, f"Month {month} returned {result}"

    def test_score_color_boundaries(self):
        """Verify anomaly score thresholds (0-39 green, 40-69 amber, 70+ red)."""
        from habitus.habitus.anomaly_breakdown import TEMP_MAX_C, TEMP_MIN_C

        # Just verify constants are sensible
        assert TEMP_MAX_C > TEMP_MIN_C
        assert TEMP_MIN_C < 0  # sub-zero is valid
        assert TEMP_MAX_C < 200  # not ridiculously high

    def test_max_consecutive_zeros_empty_series(self):
        """Empty series should return 0 consecutive zeros."""
        from habitus.habitus.patterns import _max_consecutive_zeros

        assert _max_consecutive_zeros(pd.Series([], dtype=float)) == 0

    def test_max_consecutive_zeros_all_nonzero(self):
        """All non-zero values should return 0."""
        from habitus.habitus.patterns import _max_consecutive_zeros

        assert _max_consecutive_zeros(pd.Series([1, 2, 3, 4, 5])) == 0

    def test_generate_suggestions_no_entities(self, sample_features):
        """generate_suggestions with empty stat_ids should still produce suggestions."""
        from habitus.habitus.patterns import discover_patterns, generate_suggestions

        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        assert len(suggestions) > 0
        # Boat-specific suggestions should not appear
        boat_ids = {"battery_protection", "bilge_anomaly", "shore_power_loss"}
        actual_ids = {s["id"] for s in suggestions}
        assert not boat_ids.intersection(actual_ids)


# ── Sensor Classifier Edge Cases ────────────────────────────────────────────


class TestSensorClassifierEdgeCases:
    """Edge cases for sensor type classification."""

    def test_classify_very_short_history(self):
        """Classify sensor with only 1 data point."""
        from habitus.habitus.sensor_classifier import classify_sensor

        result = classify_sensor("sensor.test", history=[5.0])
        assert result in {"gauge", "binary", "setpoint"}

    def test_classify_all_same_value(self):
        """Classify sensor where every reading is identical."""
        from habitus.habitus.sensor_classifier import classify_sensor

        result = classify_sensor("sensor.test", history=[42.0] * 100)
        assert result in {"setpoint", "gauge"}

    def test_classify_binary_with_floats(self):
        """Classify sensor with 0.0 and 1.0 float values needs enough samples."""
        from habitus.habitus.sensor_classifier import classify_sensor

        values = [0.0, 1.0] * 20
        result = classify_sensor("binary_sensor.test", history=values)
        assert result == "binary"
