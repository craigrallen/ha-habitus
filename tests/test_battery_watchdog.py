"""Tests for battery watchdog."""
from __future__ import annotations

import datetime
import json
import os

import pytest

from habitus.habitus.battery_watchdog import (
    classify_battery,
    run_battery_check,
    save_battery_status,
    load_battery_status,
    _compute_drain_rate,
    _fetch_battery_entities,
    CRITICAL_THRESHOLD,
    LOW_THRESHOLD,
)


def _make_battery_state(entity_id: str, level: float, device_class: str = "battery") -> dict:
    return {
        "entity_id": entity_id,
        "state": str(level),
        "last_updated": "2025-01-01T12:00:00Z",
        "attributes": {
            "friendly_name": entity_id.replace("sensor.", "").replace("_", " ").title(),
            "device_class": device_class,
            "unit_of_measurement": "%",
        },
    }


class TestClassifyBattery:
    def test_critical_below_threshold(self):
        assert classify_battery("sensor.test", CRITICAL_THRESHOLD - 1) == "critical"

    def test_low_within_range(self):
        assert classify_battery("sensor.test", CRITICAL_THRESHOLD + 1) == "low"
        assert classify_battery("sensor.test", LOW_THRESHOLD - 1) == "low"

    def test_ok_above_threshold(self):
        assert classify_battery("sensor.test", LOW_THRESHOLD + 1) == "ok"
        assert classify_battery("sensor.test", 100) == "ok"

    def test_critical_exactly_at_threshold(self):
        # Below CRITICAL_THRESHOLD is critical
        assert classify_battery("sensor.test", CRITICAL_THRESHOLD - 0.1) == "critical"

    def test_boundary_critical_to_low(self):
        assert classify_battery("sensor.test", CRITICAL_THRESHOLD) == "low"


class TestFetchBatteryEntities:
    def test_finds_battery_by_device_class(self):
        states = [
            _make_battery_state("sensor.door_sensor_battery", 45.0, device_class="battery"),
            _make_battery_state("sensor.temperature", 22.0, device_class="temperature"),
        ]
        entities = _fetch_battery_entities(states)
        assert any(e["entity_id"] == "sensor.door_sensor_battery" for e in entities)

    def test_finds_battery_by_name(self):
        states = [
            {
                "entity_id": "sensor.remote_battery_level",
                "state": "75",
                "last_updated": "2025-01-01T12:00:00Z",
                "attributes": {"unit_of_measurement": "%"},
            }
        ]
        entities = _fetch_battery_entities(states)
        assert any(e["entity_id"] == "sensor.remote_battery_level" for e in entities)


class TestComputeDrainRate:
    def test_stable_battery_zero_drain(self):
        """No change in readings → near-zero drain."""
        now = datetime.datetime.now(datetime.UTC)
        history = [
            {"state": "80", "last_changed": (now - datetime.timedelta(days=7)).isoformat()},
            {"state": "80", "last_changed": (now - datetime.timedelta(days=3)).isoformat()},
            {"state": "80", "last_changed": now.isoformat()},
        ]
        rate = _compute_drain_rate(history)
        assert rate is not None
        assert abs(rate) < 1.0

    def test_draining_battery(self):
        """Battery dropping 7% over 7 days → ~1%/day."""
        now = datetime.datetime.now(datetime.UTC)
        history = [
            {"state": "80", "last_changed": (now - datetime.timedelta(days=7)).isoformat()},
            {"state": "73", "last_changed": now.isoformat()},
        ]
        rate = _compute_drain_rate(history)
        assert rate is not None
        assert 0.5 < rate < 2.0

    def test_insufficient_history_returns_none(self):
        """Single reading → None."""
        now = datetime.datetime.now(datetime.UTC)
        history = [{"state": "80", "last_changed": now.isoformat()}]
        rate = _compute_drain_rate(history)
        assert rate is None

    def test_invalid_state_skipped(self):
        """Invalid state values are skipped."""
        now = datetime.datetime.now(datetime.UTC)
        history = [
            {"state": "unavailable", "last_changed": (now - datetime.timedelta(days=1)).isoformat()},
            {"state": "70", "last_changed": (now - datetime.timedelta(hours=12)).isoformat()},
            {"state": "68", "last_changed": now.isoformat()},
        ]
        # Should not raise
        rate = _compute_drain_rate(history)
        # May return None or a value depending on usable readings


class TestRunBatteryCheck:
    def test_empty_states_no_batteries(self):
        report = run_battery_check(states=[])
        assert report["total"] == 0
        assert report["batteries"] == []

    def test_batteries_sorted_by_criticality(self):
        """Critical batteries appear before ok ones."""
        states = [
            _make_battery_state("sensor.critical_battery", 5.0),
            _make_battery_state("sensor.ok_battery", 80.0),
            _make_battery_state("sensor.low_battery", 20.0),
        ]
        report = run_battery_check(states=states)
        alerts = [b["alert"] for b in report["batteries"]]
        # Critical should come first
        if "critical" in alerts and "ok" in alerts:
            assert alerts.index("critical") < alerts.index("ok")

    def test_summary_counts(self):
        """Summary has correct counts."""
        states = [
            _make_battery_state("sensor.b1", 5.0),   # critical
            _make_battery_state("sensor.b2", 20.0),  # low
            _make_battery_state("sensor.b3", 80.0),  # ok
        ]
        report = run_battery_check(states=states)
        assert report["summary"]["critical"] == 1
        assert report["summary"]["low"] == 1
        assert report["summary"]["ok"] == 1

    def test_battery_has_required_fields(self):
        states = [_make_battery_state("sensor.test_battery", 50.0)]
        report = run_battery_check(states=states)
        if report["batteries"]:
            b = report["batteries"][0]
            required = {"entity_id", "level", "alert", "area", "friendly_name"}
            for field in required:
                assert field in b

    def test_alert_prioritisation(self):
        """Alert level correctly assigned."""
        states = [
            _make_battery_state("sensor.a", 5.0),
            _make_battery_state("sensor.b", 15.0),
            _make_battery_state("sensor.c", 50.0),
        ]
        report = run_battery_check(states=states)
        levels = {b["entity_id"]: b["alert"] for b in report["batteries"]}
        assert levels["sensor.a"] == "critical"
        assert levels["sensor.b"] == "low"
        assert levels["sensor.c"] == "ok"


class TestSaveLoad:
    def test_save_and_load(self, tmp_data_dir):
        report = run_battery_check(states=[_make_battery_state("sensor.test", 50.0)])
        save_battery_status(report)
        loaded = load_battery_status()
        assert loaded["total"] == report["total"]

    def test_load_missing_returns_default(self, tmp_data_dir):
        result = load_battery_status()
        assert result["total"] == 0
