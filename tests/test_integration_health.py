"""Tests for integration health monitoring."""
from __future__ import annotations

import datetime
import os

import pytest

from habitus.habitus.integration_health import (
    check_entity_health,
    run_integration_health_check,
    save_integration_health,
    load_integration_health,
    _get_stale_threshold,
    _infer_integration,
    STALE_THRESHOLDS,
)


def _make_state(entity_id: str, state: str, last_updated: str | None = None, hours_ago: float | None = None) -> dict:
    """Helper to make HA state dict."""
    if hours_ago is not None:
        dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours_ago)
        last_updated = dt.isoformat()
    elif last_updated is None:
        last_updated = datetime.datetime.now(datetime.UTC).isoformat()

    return {
        "entity_id": entity_id,
        "state": state,
        "last_updated": last_updated,
        "attributes": {"friendly_name": entity_id.split(".")[-1].replace("_", " ").title()},
    }


class TestCheckEntityHealth:
    def test_healthy_entity(self):
        """Recently updated entity is healthy."""
        state = _make_state("sensor.temperature", "22.0", hours_ago=1)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "healthy"

    def test_unavailable_entity(self):
        """Unavailable state → unavailable status."""
        state = _make_state("sensor.broken", "unavailable", hours_ago=1)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "unavailable"

    def test_unknown_entity(self):
        """Unknown state → unavailable status."""
        state = _make_state("sensor.broken", "unknown", hours_ago=1)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "unavailable"

    def test_stale_sensor_over_24h(self):
        """Sensor not updated in >24h is stale."""
        state = _make_state("sensor.temperature", "22.0", hours_ago=25)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "stale"

    def test_fresh_sensor_under_24h(self):
        """Sensor updated <24h ago is healthy."""
        state = _make_state("sensor.temperature", "22.0", hours_ago=23)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "healthy"

    def test_stale_switch_over_7days(self):
        """Switch not updated in >7 days is stale."""
        state = _make_state("switch.test", "off", hours_ago=7 * 24 + 1)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "stale"

    def test_motion_sensor_stale_over_1h(self):
        """Motion sensor not triggered in >1h is stale."""
        state = _make_state("binary_sensor.motion_hallway", "off", hours_ago=2)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        assert result["status"] == "stale"

    def test_result_has_required_fields(self):
        state = _make_state("sensor.temp", "20.0", hours_ago=1)
        now = datetime.datetime.now(datetime.UTC)
        result = check_entity_health(state, now)
        required = {"entity_id", "domain", "state", "status", "last_updated", "integration"}
        for field in required:
            assert field in result


class TestRunIntegrationHealthCheck:
    def test_empty_states(self):
        """Empty states returns zero counts."""
        report = run_integration_health_check([])
        assert report["total_entities"] == 0
        assert report["stale_count"] == 0

    def test_computes_overall_score(self):
        """Overall score computed correctly."""
        states = [
            _make_state("sensor.a", "22.0", hours_ago=1),  # healthy
            _make_state("sensor.b", "unavailable", hours_ago=1),  # unavailable
        ]
        report = run_integration_health_check(states)
        assert 0.0 <= report["overall_score"] <= 100.0
        assert report["unavailable_count"] == 1

    def test_integration_scores_per_domain(self):
        """Integration scores computed per domain."""
        states = [
            _make_state("sensor.a", "22.0", hours_ago=1),
            _make_state("sensor.b", "20.0", hours_ago=1),
            _make_state("light.c", "on", hours_ago=1),
        ]
        report = run_integration_health_check(states)
        assert "sensor" in report["integration_scores"]
        assert "light" in report["integration_scores"]

    def test_score_calculation(self):
        """Score is % of healthy entities in domain."""
        states = [
            _make_state("sensor.a", "22.0", hours_ago=1),   # healthy
            _make_state("sensor.b", "22.0", hours_ago=25),  # stale
        ]
        report = run_integration_health_check(states)
        sensor_score = report["integration_scores"]["sensor"]["score"]
        assert sensor_score == pytest.approx(50.0)

    def test_stale_entities_list(self):
        """Stale entities appear in stale list."""
        states = [_make_state("sensor.old", "22.0", hours_ago=25)]
        report = run_integration_health_check(states)
        assert report["stale_count"] == 1
        assert any(e["entity_id"] == "sensor.old" for e in report["stale_entities"])


class TestStalenessThresholds:
    def test_sensor_threshold_24h(self):
        threshold = _get_stale_threshold("sensor.temperature", "sensor")
        assert threshold == 24 * 3600

    def test_motion_threshold_1h(self):
        threshold = _get_stale_threshold("binary_sensor.hallway_motion", "binary_sensor")
        assert threshold == 3600

    def test_switch_threshold_7days(self):
        threshold = _get_stale_threshold("switch.lamp", "switch")
        assert threshold == 7 * 24 * 3600


class TestInferIntegration:
    def test_hue_integration(self):
        assert _infer_integration("light.hue_bulb_1") == "Philips Hue"

    def test_sonos_integration(self):
        assert _infer_integration("media_player.sonos_living_room") == "Sonos"

    def test_falls_back_to_domain(self):
        result = _infer_integration("unknown_domain.entity")
        assert result  # non-empty string


class TestSaveLoad:
    def test_save_and_load(self, tmp_data_dir):
        states = [_make_state("sensor.test", "22.0", hours_ago=1)]
        report = run_integration_health_check(states)
        save_integration_health(report)
        loaded = load_integration_health()
        assert loaded["total_entities"] == 1

    def test_load_missing_returns_default(self, tmp_data_dir):
        result = load_integration_health()
        assert result["total_entities"] == 0
        assert result["stale_entities"] == []
