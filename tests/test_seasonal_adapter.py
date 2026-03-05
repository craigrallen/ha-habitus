"""Tests for seasonal adaptation suggestions."""
from __future__ import annotations

import datetime
import os

import pytest

from habitus.habitus.seasonal_adapter import (
    get_current_season,
    get_seasonal_suggestions,
    run,
    load_seasonal_suggestions,
)


class TestGetCurrentSeason:
    def test_northern_winter(self):
        for month in [12, 1, 2]:
            d = datetime.date(2025, month, 15)
            assert get_current_season(d, "north") == "winter"

    def test_northern_spring(self):
        for month in [3, 4, 5]:
            d = datetime.date(2025, month, 15)
            assert get_current_season(d, "north") == "spring"

    def test_northern_summer(self):
        for month in [6, 7, 8]:
            d = datetime.date(2025, month, 15)
            assert get_current_season(d, "north") == "summer"

    def test_northern_autumn(self):
        for month in [9, 10, 11]:
            d = datetime.date(2025, month, 15)
            assert get_current_season(d, "north") == "autumn"

    def test_southern_winter(self):
        for month in [6, 7, 8]:
            d = datetime.date(2025, month, 15)
            assert get_current_season(d, "south") == "winter"

    def test_southern_summer(self):
        for month in [12, 1, 2]:
            d = datetime.date(2025, month, 15)
            assert get_current_season(d, "south") == "summer"

    def test_defaults_to_today(self):
        """Calling without date doesn't raise."""
        season = get_current_season()
        assert season in ("winter", "spring", "summer", "autumn")


class TestGetSeasonalSuggestions:
    def test_winter_suggestions_generated(self):
        d = datetime.date(2025, 1, 15)
        result = get_seasonal_suggestions(d, hemisphere="north")
        assert result["season"] == "winter"
        assert result["total"] > 0
        assert len(result["suggestions"]) > 0

    def test_summer_suggestions_generated(self):
        d = datetime.date(2025, 7, 15)
        result = get_seasonal_suggestions(d, hemisphere="north")
        assert result["season"] == "summer"
        assert len(result["suggestions"]) > 0

    def test_spring_suggestions_generated(self):
        d = datetime.date(2025, 4, 15)
        result = get_seasonal_suggestions(d, hemisphere="north")
        assert result["season"] == "spring"
        assert len(result["suggestions"]) > 0

    def test_autumn_suggestions_generated(self):
        d = datetime.date(2025, 10, 15)
        result = get_seasonal_suggestions(d, hemisphere="north")
        assert result["season"] == "autumn"
        assert len(result["suggestions"]) > 0

    def test_suggestion_has_required_fields(self):
        d = datetime.date(2025, 1, 15)
        result = get_seasonal_suggestions(d)
        for sug in result["suggestions"]:
            assert "title" in sug
            assert "seasonal_reason" in sug
            assert "confidence" in sug
            assert "generated_yaml" in sug

    def test_yaml_is_non_empty(self):
        d = datetime.date(2025, 7, 15)
        result = get_seasonal_suggestions(d, hemisphere="north")
        for sug in result["suggestions"]:
            assert len(sug["generated_yaml"]) > 10

    def test_confidence_between_0_and_1(self):
        d = datetime.date(2025, 1, 15)
        result = get_seasonal_suggestions(d)
        for sug in result["suggestions"]:
            assert 0.0 <= sug["confidence"] <= 1.0

    def test_result_has_season_field(self):
        result = get_seasonal_suggestions(datetime.date(2025, 3, 15))
        assert "season" in result
        assert "hemisphere" in result
        assert "date" in result


class TestRun:
    def test_run_saves_file(self, tmp_data_dir):
        run(datetime.date(2025, 1, 15))
        path = os.path.join(str(tmp_data_dir), "seasonal_suggestions.json")
        assert os.path.exists(path)

    def test_load_missing_returns_default(self, tmp_data_dir):
        result = load_seasonal_suggestions()
        assert result["season"] == "unknown"
        assert result["suggestions"] == []
