"""Tests for automation conflict detection (automation-to-automation conflicts)."""
from __future__ import annotations

import pytest

from habitus.habitus.conflict_detector import detect_conflicts, save_conflicts, load_conflicts


class TestDetectConflicts:
    """Tests for cross-domain entity conflict detection."""

    def test_window_heating_conflict(self):
        """Window open + heating active → conflict detected."""
        states = {
            "binary_sensor.living_room_window": {"state": "on", "last_changed": "2025-01-01T10:00:00Z"},
            "climate.living_room": {"state": "heat"},
        }
        conflicts = detect_conflicts(states)
        ids = [c["id"] for c in conflicts]
        assert "window_heating" in ids

    def test_nobody_home_lights_conflict(self):
        """Nobody home + lights on → conflict."""
        states = {
            "person.craig": {"state": "not_home"},
            "light.living_room": {"state": "on"},
        }
        conflicts = detect_conflicts(states)
        ids = [c["id"] for c in conflicts]
        assert "nobody_home_lights" in ids

    def test_no_false_positive_when_window_closed(self):
        """Window closed + heating on → no conflict."""
        states = {
            "binary_sensor.living_room_window": {"state": "off"},
            "climate.living_room": {"state": "heat"},
        }
        conflicts = detect_conflicts(states)
        ids = [c["id"] for c in conflicts]
        assert "window_heating" not in ids

    def test_no_false_positive_when_someone_home(self):
        """Someone home + lights on → no nobody_home conflict."""
        states = {
            "person.craig": {"state": "home"},
            "light.living_room": {"state": "on"},
        }
        conflicts = detect_conflicts(states)
        ids = [c["id"] for c in conflicts]
        assert "nobody_home_lights" not in ids

    def test_heat_cool_fight_critical(self):
        """Heating + cooling simultaneously → critical severity."""
        states = {
            "climate.living_room": {"state": "heat"},
            "climate.bedroom": {"state": "cool"},
        }
        conflicts = detect_conflicts(states)
        ids = [c["id"] for c in conflicts]
        assert "heat_cool_fight" in ids
        heat_cool = next(c for c in conflicts if c["id"] == "heat_cool_fight")
        assert heat_cool["severity"] == "critical"

    def test_empty_states_no_conflicts(self):
        """Empty states → no conflicts."""
        conflicts = detect_conflicts({})
        assert conflicts == []

    def test_severity_scoring(self):
        """Conflicts have valid severity values."""
        states = {
            "binary_sensor.living_room_window": {"state": "on", "last_changed": "2025-01-01T10:00:00Z"},
            "climate.living_room": {"state": "heat"},
            "person.craig": {"state": "not_home"},
            "light.kitchen": {"state": "on"},
        }
        conflicts = detect_conflicts(states)
        valid_severities = {"critical", "high", "medium", "low"}
        for c in conflicts:
            assert c["severity"] in valid_severities, f"Invalid severity: {c['severity']}"

    def test_conflict_has_required_fields(self):
        """Each conflict has required fields."""
        states = {
            "binary_sensor.living_room_window": {"state": "on", "last_changed": "2025-01-01T10:00:00Z"},
            "climate.living_room": {"state": "heat"},
        }
        conflicts = detect_conflicts(states)
        assert len(conflicts) > 0
        for c in conflicts:
            assert "id" in c
            assert "severity" in c
            assert "description" in c
            assert "suggestion" in c

    def test_save_and_load_conflicts(self, tmp_data_dir, monkeypatch):
        """save_conflicts + load_conflicts round-trips."""
        import habitus.habitus.conflict_detector as cd
        monkeypatch.setattr(cd, "DATA_DIR", str(tmp_data_dir))
        monkeypatch.setattr(cd, "CONFLICTS_PATH", str(tmp_data_dir / "conflicts.json"))

        conflicts = [{"id": "test", "severity": "medium", "description": "test", "suggestion": "fix it"}]
        save_conflicts(conflicts)

        loaded = load_conflicts()
        assert loaded["count"] == 1
        assert loaded["conflicts"][0]["id"] == "test"

    def test_warm_outside_heating(self):
        """Warm outdoor temp + heating → conflict."""
        states = {
            "sensor.outdoor_temperature": {"state": "22.0"},
            "climate.living_room": {"state": "heat"},
        }
        conflicts = detect_conflicts(states)
        ids = [c["id"] for c in conflicts]
        assert "warm_outside_heating" in ids
