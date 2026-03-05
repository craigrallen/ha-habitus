"""Tests for guest mode detection."""
from __future__ import annotations

import datetime
import json
import os

import pytest

from habitus.habitus.guest_mode import (
    compute_guest_probability,
    run,
    load_guest_mode,
    GUEST_PROBABILITY_THRESHOLD,
    _hour_spread,
    _night_activity_ratio,
)


def _make_event(entity_id: str, state: str, hour: int, minute: int = 0, day: int = 1) -> dict:
    dt = datetime.datetime(2025, 1, day, hour, minute, tzinfo=datetime.timezone.utc)
    return {"entity_id": entity_id, "state": state, "timestamp": dt.isoformat()}


class TestGuestProbability:
    def test_low_probability_baseline_activity(self):
        """Normal baseline activity → low guest probability."""
        events = []
        # Regular daytime activity only
        for day in range(7):
            for h in range(8, 18):
                events.append(_make_event("light.living_room", "on", h, day=day + 1))

        baseline = {
            "typical_hour_spread": 0.5,
            "typical_night_ratio": 0.0,
            "typical_concurrent": 1.0,
            "typical_daily_events": 50,
        }
        result = compute_guest_probability(events, baseline)
        assert result["guest_probability"] < GUEST_PROBABILITY_THRESHOLD

    def test_high_probability_with_unusual_activity(self):
        """Wide time spread + high night activity → high guest probability."""
        events = []
        # Activity across all hours including night
        for day in range(7):
            for h in range(0, 24):
                events.append(_make_event("light.living_room", "on", h, day=day + 1))
            # Guest room activity
            events.append(_make_event("light.guest_room", "on", 20, day=day + 1))
            events.append(_make_event("light.spare_bedroom", "on", 21, day=day + 1))

        baseline = {
            "typical_hour_spread": 0.3,
            "typical_night_ratio": 0.01,
            "typical_concurrent": 1.0,
            "typical_daily_events": 20,
        }
        result = compute_guest_probability(events, baseline)
        assert result["guest_probability"] > GUEST_PROBABILITY_THRESHOLD

    def test_suggestions_generated_above_threshold(self):
        """Suggestions generated when probability > threshold."""
        events = []
        for day in range(7):
            for h in range(0, 24):
                events.append(_make_event("light.guest_room", "on", h, day=day + 1))
            events.append(_make_event("light.spare_bedroom", "on", 23, day=day + 1))

        baseline = {
            "typical_hour_spread": 0.2,
            "typical_night_ratio": 0.01,
            "typical_concurrent": 1.0,
            "typical_daily_events": 10,
        }
        result = compute_guest_probability(events, baseline)
        if result["guest_probability"] > GUEST_PROBABILITY_THRESHOLD:
            assert "suggestions" in result
            assert len(result["suggestions"]) > 0

    def test_empty_events_low_probability(self):
        """Empty events → zero probability."""
        result = compute_guest_probability([])
        assert result["guest_probability"] == 0.0
        assert result["factors"] == {}

    def test_result_has_required_fields(self):
        """Result has required fields."""
        result = compute_guest_probability([])
        assert "guest_probability" in result
        assert "factors" in result
        assert 0.0 <= result["guest_probability"] <= 1.0

    def test_probability_bounded_0_to_1(self):
        """Probability is always between 0 and 1."""
        events = []
        for day in range(7):
            for h in range(0, 24):
                for i in range(5):
                    events.append(_make_event(f"light.room_{i}", "on", h, day=day + 1))
        result = compute_guest_probability(events, baseline={
            "typical_hour_spread": 0.0,
            "typical_night_ratio": 0.0,
            "typical_concurrent": 0.5,
            "typical_daily_events": 1,
        })
        assert 0.0 <= result["guest_probability"] <= 1.0


class TestHourSpread:
    def test_all_same_hour(self):
        spread = _hour_spread([8, 8, 8, 8])
        assert spread == pytest.approx(1 / 24)

    def test_all_different_hours(self):
        spread = _hour_spread(list(range(24)))
        assert spread == pytest.approx(1.0)

    def test_empty(self):
        spread = _hour_spread([])
        assert spread == 0.0


class TestNightActivityRatio:
    def test_all_daytime(self):
        ratio = _night_activity_ratio([9, 10, 11, 12, 13, 14])
        assert ratio == 0.0

    def test_all_night(self):
        ratio = _night_activity_ratio([22, 23, 0, 1, 2])
        assert ratio == 1.0

    def test_mixed(self):
        ratio = _night_activity_ratio([22, 10, 10, 10])
        assert ratio == pytest.approx(0.25)


class TestRun:
    def test_run_saves_file(self, tmp_data_dir):
        """run() saves guest_mode.json."""
        run([])
        assert os.path.exists(os.path.join(str(tmp_data_dir), "guest_mode.json"))

    def test_load_missing_returns_default(self, tmp_data_dir):
        """load_guest_mode() returns default when no file."""
        result = load_guest_mode()
        assert result["guest_probability"] == 0.0
