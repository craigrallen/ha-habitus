"""Tests for pattern discovery and automation suggestion generation."""
from __future__ import annotations

import json
import os

import pytest

from habitus.habitus.patterns import (
    _has,
    _max_consecutive_zeros,
    discover_patterns,
    generate_suggestions,
    run,
)


class TestDiscoverPatterns:
    def test_returns_daily_routine(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "daily_routine" in patterns
        assert "peak_usage_hour" in patterns["daily_routine"]

    def test_returns_weekly(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "weekly" in patterns
        assert "Mon" in patterns["weekly"]

    def test_returns_seasonal(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "seasonal" in patterns

    def test_peak_hour_in_range(self, sample_features):
        patterns = discover_patterns(sample_features)
        peak = patterns["daily_routine"]["peak_usage_hour"]
        assert 0 <= peak <= 23

    def test_night_baseline_positive(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert patterns["daily_routine"]["night_baseline_watts"] >= 0


class TestHasHelper:
    def test_detects_keyword(self):
        assert _has(["sensor.bilge_pump"], "bilge")

    def test_case_insensitive(self):
        assert _has(["sensor.BATTERY_SOC"], "battery")

    def test_returns_false_when_absent(self):
        assert not _has(["sensor.temperature"], "bilge")

    def test_multiple_keywords_any_match(self):
        assert _has(["sensor.shore_power"], "bilge", "shore")


class TestGenerateSuggestions:
    def test_returns_list(self, sample_features):
        patterns = discover_patterns(sample_features)
        stat_ids = [
            "sensor.mastervolt_total_load",
            "sensor.house_battery_soc",
            "binary_sensor.bilge_pump",
            "sensor.shore_power_w",
        ]
        suggestions = generate_suggestions(patterns, sample_features, stat_ids)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_each_has_required_fields(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        required = {"id", "title", "description", "confidence", "category", "yaml"}
        for s in suggestions:
            assert required.issubset(s.keys()), f"Missing fields in {s['id']}"

    def test_confidence_in_range(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        for s in suggestions:
            assert 0 <= s["confidence"] <= 100

    def test_boat_suggestions_when_entities_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        stat_ids = ["sensor.bilge_sensor_air_temperature", "sensor.house_battery_energy_watts"]
        suggestions = generate_suggestions(patterns, sample_features, stat_ids)
        categories = {s["category"] for s in suggestions}
        assert "boat" in categories

    def test_yaml_parseable(self, sample_features):
        import yaml

        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        for s in suggestions:
            if s.get("yaml"):
                # Should parse without error
                parsed = yaml.safe_load(s["yaml"])
                assert parsed is not None

    def test_personalization_explanation_fields_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, ["person.craig"])
        assert suggestions
        for s in suggestions[:5]:
            assert "why_suggested" in s and s["why_suggested"]
            assert "confidence_rationale" in s and "Confidence" in s["confidence_rationale"]
            assert "expected_benefit" in s and s["expected_benefit"]
            assert "status_badges" in s and isinstance(s["status_badges"], list)


class TestMaxConsecutiveZeros:
    def test_all_zeros(self):
        import pandas as pd

        assert _max_consecutive_zeros(pd.Series([0, 0, 0])) == 3

    def test_no_zeros(self):
        import pandas as pd

        assert _max_consecutive_zeros(pd.Series([1, 2, 3])) == 0

    def test_mixed(self):
        import pandas as pd

        assert _max_consecutive_zeros(pd.Series([1, 0, 0, 1, 0])) == 2

    def test_empty(self):
        import pandas as pd

        assert _max_consecutive_zeros(pd.Series([], dtype=float)) == 0


class TestNewPatterns:
    def test_morning_lights_pattern_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "morning_lights_pattern" in patterns

    def test_morning_lights_ratio_in_range(self, sample_features):
        patterns = discover_patterns(sample_features)
        mlp = patterns["morning_lights_pattern"]
        assert 0.0 <= mlp["lights_on_ratio"] <= 1.0
        assert 0 <= mlp["confidence"] <= 95

    def test_peak_tariff_pattern_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "peak_tariff_pattern" in patterns

    def test_peak_tariff_fields(self, sample_features):
        patterns = discover_patterns(sample_features)
        ptp = patterns["peak_tariff_pattern"]
        assert "high_power_ratio" in ptp
        assert "mean_power_w" in ptp
        assert ptp["threshold_w"] == 800
        assert 0.0 <= ptp["high_power_ratio"] <= 1.0

    def test_vacancy_pattern_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "vacancy_pattern" in patterns

    def test_vacancy_pattern_fields(self, sample_features):
        patterns = discover_patterns(sample_features)
        vp = patterns["vacancy_pattern"]
        assert "max_no_motion_hours" in vp
        assert "extended_vacancy_detected" in vp
        assert vp["max_no_motion_hours"] >= 0

    def test_bilge_temp_baseline_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        assert "bilge_temp_baseline" in patterns

    def test_bilge_temp_baseline_fields(self, sample_features):
        patterns = discover_patterns(sample_features)
        btp = patterns["bilge_temp_baseline"]
        assert "mean_c" in btp
        assert "std_c" in btp
        assert "alert_threshold_c" in btp
        # alert threshold should be 3°C above mean
        assert abs(btp["alert_threshold_c"] - (btp["mean_c"] + 3.0)) < 0.01


class TestNewSuggestions:
    def test_morning_lights_suggestion_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        ids = {s["id"] for s in suggestions}
        assert "morning_lights" in ids

    def test_peak_tariff_alert_suggestion_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        ids = {s["id"] for s in suggestions}
        assert "peak_tariff_alert" in ids

    def test_vacancy_security_suggestion_present(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        ids = {s["id"] for s in suggestions}
        assert "vacancy_security" in ids

    def test_bilge_temp_anomaly_suggestion_when_bilge_entity(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(
            patterns, sample_features, ["sensor.bilge_sensor_air_temperature"]
        )
        ids = {s["id"] for s in suggestions}
        assert "bilge_temp_anomaly" in ids

    def test_bilge_temp_anomaly_absent_without_entity(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(patterns, sample_features, [])
        ids = {s["id"] for s in suggestions}
        assert "bilge_temp_anomaly" not in ids

    def test_shore_power_battery_when_both_entities(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(
            patterns,
            sample_features,
            ["sensor.shore_power_smart_meter_electric_consumption_w", "sensor.house_battery_soc"],
        )
        ids = {s["id"] for s in suggestions}
        assert "shore_power_battery" in ids

    def test_shore_power_battery_absent_without_battery(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(
            patterns, sample_features, ["sensor.shore_power_smart_meter_electric_consumption_w"]
        )
        ids = {s["id"] for s in suggestions}
        assert "shore_power_battery" not in ids

    def test_new_suggestions_have_required_fields(self, sample_features):
        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(
            patterns,
            sample_features,
            [
                "sensor.bilge_sensor_air_temperature",
                "sensor.shore_power_smart_meter_electric_consumption_w",
                "sensor.house_battery_soc",
            ],
        )
        required = {"id", "title", "description", "confidence", "category", "yaml"}
        new_ids = {
            "morning_lights",
            "peak_tariff_alert",
            "vacancy_security",
            "bilge_temp_anomaly",
            "shore_power_battery",
        }
        for s in suggestions:
            if s["id"] in new_ids:
                assert required.issubset(s.keys()), f"Missing fields in {s['id']}"
                assert 0 <= s["confidence"] <= 100

    def test_new_suggestions_yaml_parseable(self, sample_features):
        import yaml

        patterns = discover_patterns(sample_features)
        suggestions = generate_suggestions(
            patterns,
            sample_features,
            [
                "sensor.bilge_sensor_air_temperature",
                "sensor.shore_power_smart_meter_electric_consumption_w",
                "sensor.house_battery_soc",
            ],
        )
        new_ids = {
            "morning_lights",
            "peak_tariff_alert",
            "vacancy_security",
            "bilge_temp_anomaly",
            "shore_power_battery",
        }
        for s in suggestions:
            if s["id"] in new_ids and s.get("yaml"):
                parsed = yaml.safe_load(s["yaml"])
                assert parsed is not None, f"YAML parse failed for {s['id']}"


class TestHomeProfilePrioritization:
    def test_home_profile_ranks_home_suggestions_above_boat(self, sample_features):
        patterns = discover_patterns(sample_features)
        stat_ids = [
            "sensor.total_home_power_w",
            "binary_sensor.hallway_motion",
            "light.living_room_ceiling",
            "person.craig",
            "climate.living_room_thermostat",
        ]
        suggestions = generate_suggestions(patterns, sample_features, stat_ids)
        assert suggestions

        top_ids = [s["id"] for s in suggestions[:10]]
        assert "occupancy_lights" in top_ids
        assert "climate_preheat" in top_ids

        away_idx = next(i for i, s in enumerate(suggestions) if s["id"] == "away_mode")
        boat_idx = [i for i, s in enumerate(suggestions) if s.get("category") == "boat"]
        if boat_idx:
            assert away_idx < min(boat_idx)

        boat = [s for s in suggestions if s.get("category") == "boat"]
        assert all(s.get("applicable") is False for s in boat)

    def test_home_core_routines_present_when_entities_available(self, sample_features):
        patterns = discover_patterns(sample_features)
        stat_ids = [
            "sensor.total_home_power_w",
            "binary_sensor.hallway_motion",
            "light.living_room_ceiling",
            "person.craig",
            "climate.living_room_thermostat",
        ]
        suggestions = generate_suggestions(patterns, sample_features, stat_ids)
        ids = {s["id"] for s in suggestions}
        assert "occupancy_lights" in ids
        assert "away_mode" in ids
        assert "overnight_standby" in ids
        assert "climate_preheat" in ids
        assert "peak_tariff_alert" in ids
        assert "sensor_watchdog" in ids
        assert "anomaly_alert" in ids

        first = suggestions[0]
        assert "household_rhythm" in first
        assert set(first["household_rhythm"].keys()) == {
            "weekday_window",
            "weekend_window",
            "homecoming_hour",
        }


class TestRunFunction:
    def test_run_writes_patterns_and_suggestions(self, sample_features, tmp_data_dir):
        import habitus.habitus.patterns as pat

        pat.PATTERNS_PATH = str(tmp_data_dir / "patterns.json")
        pat.SUGGESTIONS_PATH = str(tmp_data_dir / "suggestions.json")
        run(sample_features, [])
        assert (tmp_data_dir / "patterns.json").exists()
        assert (tmp_data_dir / "suggestions.json").exists()

    def test_run_suggestions_json_valid(self, sample_features, tmp_data_dir):
        import habitus.habitus.patterns as pat

        pat.PATTERNS_PATH = str(tmp_data_dir / "patterns.json")
        pat.SUGGESTIONS_PATH = str(tmp_data_dir / "suggestions.json")
        run(sample_features, ["sensor.house_battery_soc"])
        with open(os.path.join(str(tmp_data_dir), "suggestions.json")) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_run_returns_tuple(self, sample_features, tmp_data_dir):
        import habitus.habitus.patterns as pat

        pat.PATTERNS_PATH = str(tmp_data_dir / "patterns.json")
        pat.SUGGESTIONS_PATH = str(tmp_data_dir / "suggestions.json")
        result = run(sample_features, [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        patterns, suggestions = result
        assert isinstance(patterns, dict)
        assert isinstance(suggestions, list)
