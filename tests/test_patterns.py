"""Tests for pattern discovery and automation suggestion generation."""
from __future__ import annotations

import pytest

from habitus.habitus.patterns import (
    discover_patterns,
    generate_suggestions,
    _has,
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
