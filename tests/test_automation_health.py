"""Tests for automation health detection."""
from __future__ import annotations

import datetime
import json
import os

import pytest

from habitus.habitus.automation_health import (
    classify_automation,
    run_health_check,
    save_health,
    load_health,
    DEAD_DAYS,
    STALE_DAYS,
    OVER_TRIGGER_COUNT,
)


def _make_automation(alias: str = "Test Automation") -> dict:
    return {
        "alias": alias,
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion"}],
        "action": [{"service": "light.turn_on", "target": {"entity_id": "light.hall"}}],
    }


def _ts(dt: datetime.datetime) -> str:
    """Format datetime as ISO string with Z suffix (naive UTC)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _make_state(alias: str, last_triggered: str | None = None) -> dict:
    attrs = {}
    if last_triggered:
        attrs["last_triggered"] = last_triggered
    return {
        "entity_id": f"automation.{alias.lower().replace(' ', '_')}",
        "state": "on",
        "attributes": attrs,
    }


class TestClassifyAutomation:
    def test_healthy_automation(self):
        """Recently triggered automation is healthy."""
        recent = _ts(datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1))
        state = _make_state("test", last_triggered=recent)
        result = classify_automation(_make_automation("test"), state, [], {"binary_sensor.motion"})
        assert result["status"] == "healthy"

    def test_dead_never_triggered(self):
        """Never triggered automation is dead."""
        state = _make_state("test", last_triggered=None)
        result = classify_automation(_make_automation("test"), state, [], {"binary_sensor.motion"})
        assert result["status"] == "dead"

    def test_dead_old_trigger(self):
        """Automation not triggered in >30 days is dead."""
        old = _ts(datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=DEAD_DAYS + 5))
        state = _make_state("test", last_triggered=old)
        result = classify_automation(_make_automation("test"), state, [], {"binary_sensor.motion"})
        assert result["status"] == "dead"

    def test_stale_detection(self):
        """Automation not triggered in 8-30 days is stale."""
        stale_days = STALE_DAYS + 1
        old = _ts(datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=stale_days))
        state = _make_state("test", last_triggered=old)
        result = classify_automation(_make_automation("test"), state, [], {"binary_sensor.motion"})
        assert result["status"] == "stale"

    def test_dead_missing_trigger_entity(self):
        """Automation with missing trigger entity is dead."""
        recent = _ts(datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1))
        state = _make_state("test", last_triggered=recent)
        # Pass empty set of known entities → trigger entity doesn't exist
        result = classify_automation(_make_automation("test"), state, [], set())
        assert result["status"] == "dead"
        assert "no longer exists" in result["recommendation"].lower()

    def test_over_triggering(self):
        """Automation triggered >50x in 7 days is over_triggering."""
        now = datetime.datetime.now(datetime.UTC)
        recent = _ts(now - datetime.timedelta(days=1))
        state = _make_state("test", last_triggered=recent)

        # Build fake history with 60 triggers in last 7 days
        history = []
        for i in range(60):
            dt = now - datetime.timedelta(hours=i * 2)
            history.append({"state": "on", "last_changed": dt.isoformat()})

        result = classify_automation(
            _make_automation("test"),
            state,
            history,
            {"binary_sensor.motion"},
        )
        assert result["status"] == "over_triggering"
        assert result["trigger_count_7d"] >= OVER_TRIGGER_COUNT

    def test_result_has_required_fields(self):
        """Result has all required fields."""
        state = _make_state("test")
        result = classify_automation(_make_automation("test"), state, [], {"binary_sensor.motion"})
        required = {"alias", "entity_id", "status", "last_triggered", "trigger_count_7d", "recommendation"}
        for field in required:
            assert field in result, f"Missing field: {field}"


class TestRunHealthCheck:
    def test_empty_automations(self):
        """Empty automation list returns empty report."""
        report = run_health_check(automations=[], states={})
        assert report["total"] == 0
        assert report["automations"] == []

    def test_runs_with_automations(self):
        """Health check runs with provided automations."""
        automations = [_make_automation("Test Auto")]
        states = {}
        report = run_health_check(automations=automations, states=states)
        assert report["total"] == 1
        assert len(report["automations"]) == 1
        assert "summary" in report

    def test_summary_counts(self):
        """Summary has correct status counts."""
        recent = _ts(datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1))
        automations = [
            _make_automation("Active Auto"),
            _make_automation("Dead Auto"),
        ]
        states = {
            "automation.active_auto": {
                "entity_id": "automation.active_auto",
                "state": "on",
                "attributes": {"last_triggered": recent},
            }
        }
        report = run_health_check(automations=automations, states=states)
        assert report["total"] == 2
        assert "summary" in report


class TestSaveLoadHealth:
    def test_save_and_load(self, tmp_data_dir):
        """Save and load health report (uses tmp DATA_DIR via env)."""
        report = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "total": 2,
            "summary": {"healthy": 1, "dead": 1},
            "automations": [],
        }
        save_health(report)
        loaded = load_health()
        assert loaded["total"] == 2
        assert loaded["summary"]["healthy"] == 1

    def test_load_missing_file(self, tmp_data_dir):
        """Load returns default when file missing."""
        result = load_health()
        assert result["total"] == 0
        assert result["automations"] == []
