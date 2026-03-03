"""Tests for the activity baseline engine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import responses as resp_lib

from habitus.habitus.activity import (
    classify_entity,
    extract_activity_features,
    build_activity_baseline,
    score_activity_anomalies,
    _derive_current_features,
    _feature_label,
    _fmt_feature,
    ACTIVITY_BASELINE_PATH,
    ACTIVITY_ANOMALIES_PATH,
)


class TestClassifyEntity:
    """classify_entity correctly categorises entity IDs."""

    def test_motion_sensor(self):
        assert classify_entity("binary_sensor.hallway_motion") == "motion"

    def test_pir_sensor(self):
        assert classify_entity("binary_sensor.bedroom_pir") == "motion"

    def test_light_domain(self):
        assert classify_entity("light.kitchen_ceiling") == "light"

    def test_person_domain(self):
        assert classify_entity("person.craig") == "presence"

    def test_device_tracker(self):
        assert classify_entity("device_tracker.craigs_phone") == "presence"

    def test_media_player(self):
        assert classify_entity("media_player.living_room_tv") == "media"

    def test_door_sensor(self):
        assert classify_entity("binary_sensor.front_door") == "door"

    def test_window_sensor(self):
        assert classify_entity("binary_sensor.bedroom_window") == "door"

    def test_weather_sensor(self):
        assert classify_entity("sensor.outdoor_temperature") == "weather"

    def test_openweather(self):
        assert classify_entity("sensor.openweather_temperature") == "weather"

    def test_power_sensor_not_categorised(self):
        assert classify_entity("sensor.kitchen_electric_consumed_w") is None

    def test_unknown_returns_none(self):
        assert classify_entity("sensor.unknown_thing") is None

    def test_case_insensitive(self):
        assert classify_entity("binary_sensor.HALLWAY_MOTION") == "motion"


class TestExtractActivityFeatures:
    """extract_activity_features produces correct feature columns."""

    def test_returns_dataframe(self, sample_df):
        result = extract_activity_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        result = extract_activity_features(sample_df)
        expected = [
            "lights_on", "motion_events", "presence_count",
            "people_home_pct", "media_active", "door_events",
            "outdoor_temp_c", "activity_diversity",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_values_non_negative(self, sample_df):
        result = extract_activity_features(sample_df)
        # Exclude temperature columns — outdoor_temp_c is legitimately negative in winter
        non_temp_cols = [
            c for c in result.select_dtypes(include=np.number).columns if "temp" not in c
        ]
        assert (result[non_temp_cols] >= 0).all().all()

    def test_empty_df_returns_empty(self):
        result = extract_activity_features(pd.DataFrame())
        assert result.empty

    def test_lights_on_bounded(self, sample_df):
        result = extract_activity_features(sample_df)
        # Lights on should be a reasonable count, not absurdly high
        assert result["lights_on"].max() <= 100

    def test_presence_pct_bounded(self, sample_df):
        result = extract_activity_features(sample_df)
        assert result["people_home_pct"].between(0, 1).all()

    def test_media_active_binary(self, sample_df):
        result = extract_activity_features(sample_df)
        assert result["media_active"].isin([0, 0.5, 1.0]).all() or result["media_active"].max() <= 1


class TestBuildActivityBaseline:
    """build_activity_baseline saves correct structure."""

    def test_saves_json(self, sample_df, tmp_data_dir):
        import habitus.habitus.activity as act
        act.ACTIVITY_BASELINE_PATH = str(tmp_data_dir / "activity_baseline.json")
        features = extract_activity_features(sample_df)
        baseline = build_activity_baseline(features)
        assert (tmp_data_dir / "activity_baseline.json").exists()

    def test_has_time_slots(self, sample_df, tmp_data_dir):
        import habitus.habitus.activity as act
        act.ACTIVITY_BASELINE_PATH = str(tmp_data_dir / "activity_baseline.json")
        features = extract_activity_features(sample_df)
        baseline = build_activity_baseline(features)
        # Should have many time slots (24h × 7 days = up to 168)
        assert len(baseline) > 50

    def test_slot_has_feature_stats(self, sample_df, tmp_data_dir):
        import habitus.habitus.activity as act
        act.ACTIVITY_BASELINE_PATH = str(tmp_data_dir / "activity_baseline.json")
        features = extract_activity_features(sample_df)
        baseline = build_activity_baseline(features)
        # Pick any slot and check structure
        slot = next(iter(baseline.values()))
        feat = next(iter(slot.values()))
        assert "mean" in feat and "std" in feat and "n" in feat

    def test_empty_features_returns_empty(self, tmp_data_dir):
        import habitus.habitus.activity as act
        act.ACTIVITY_BASELINE_PATH = str(tmp_data_dir / "activity_baseline.json")
        result = build_activity_baseline(pd.DataFrame())
        assert result == {}


class TestDeriveCurrentFeatures:
    """_derive_current_features aggregates live states correctly."""

    def test_lights_counted(self):
        states = {
            "light.kitchen": 1.0,
            "light.living_room": 1.0,
            "light.bedroom": 0.0,
        }
        result = _derive_current_features(states)
        assert result["lights_on"] == 2.0

    def test_presence_fraction(self):
        states = {
            "person.alice": 1.0,
            "person.bob": 0.0,
        }
        result = _derive_current_features(states)
        assert result["presence_count"] == 1.0
        assert result["people_home_pct"] == pytest.approx(0.5)

    def test_media_active(self):
        states = {"media_player.tv": 1.0}
        result = _derive_current_features(states)
        assert result["media_active"] == 1.0

    def test_empty_states(self):
        result = _derive_current_features({})
        assert result["lights_on"] == 0.0
        assert result["media_active"] == 0.0


class TestFeatureHelpers:
    """Label and format helpers."""

    def test_feature_label_known(self):
        assert _feature_label("lights_on") == "Lights on"
        assert _feature_label("media_active") == "Media playing"

    def test_feature_label_unknown(self):
        label = _feature_label("weird_thing")
        assert label == "Weird Thing"

    def test_fmt_feature_temp(self):
        assert "°C" in _fmt_feature("outdoor_temp_c", 5.5)

    def test_fmt_feature_pct(self):
        assert "%" in _fmt_feature("people_home_pct", 0.5)
