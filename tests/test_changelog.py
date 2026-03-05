"""Tests for automation changelog."""
from __future__ import annotations

import datetime
import json
import os

import pytest

from habitus.habitus.changelog import (
    diff_automations,
    append_entry,
    append_entries,
    load_changelog,
    log_habitus_add,
    log_habitus_remove,
    log_triggered,
    run_diff_and_log,
    MAX_ENTRIES,
    CHANGE_ADDED,
    CHANGE_REMOVED,
    CHANGE_MODIFIED,
    CHANGE_HABITUS_ADD,
)


def _make_automation(alias: str, action_service: str = "light.turn_on") -> dict:
    return {
        "alias": alias,
        "trigger": [{"platform": "state", "entity_id": "binary_sensor.motion"}],
        "action": [{"service": action_service}],
    }


class TestDiffAutomations:
    def test_detect_added(self):
        """New automation detected as added."""
        prev = []
        current = [_make_automation("New Auto")]
        changes = diff_automations(prev, current)
        assert any(c["type"] == CHANGE_ADDED for c in changes)
        assert any(c["alias"] == "New Auto" for c in changes)

    def test_detect_removed(self):
        """Missing automation detected as removed."""
        prev = [_make_automation("Old Auto")]
        current = []
        changes = diff_automations(prev, current)
        assert any(c["type"] == CHANGE_REMOVED for c in changes)
        assert any(c["alias"] == "Old Auto" for c in changes)

    def test_detect_modified(self):
        """Modified automation detected."""
        prev = [_make_automation("Changed Auto", "light.turn_on")]
        current = [_make_automation("Changed Auto", "light.turn_off")]
        changes = diff_automations(prev, current)
        types = [c["type"] for c in changes]
        assert "modified" in types

    def test_no_changes_no_entries(self):
        """Identical lists → no changes."""
        auto = [_make_automation("Static Auto")]
        changes = diff_automations(auto, auto)
        assert changes == []

    def test_empty_to_empty(self):
        """Both empty → no changes."""
        assert diff_automations([], []) == []

    def test_change_has_required_fields(self):
        """Each change has required fields."""
        changes = diff_automations([], [_make_automation("Test")])
        for c in changes:
            assert "type" in c
            assert "alias" in c
            assert "timestamp" in c
            assert "description" in c


class TestAppendEntry:
    def test_append_single_entry(self, tmp_data_dir):
        """Appending an entry saves it."""
        append_entry({"type": "added", "alias": "Test", "description": "test"})
        entries = load_changelog()
        assert len(entries) == 1
        assert entries[0]["type"] == "added"

    def test_append_sets_timestamp(self, tmp_data_dir):
        """Append sets timestamp if not provided."""
        append_entry({"type": "added", "alias": "Test", "description": "test"})
        entries = load_changelog()
        assert "timestamp" in entries[0]

    def test_append_multiple(self, tmp_data_dir):
        """Multiple appends accumulate."""
        for i in range(5):
            append_entry({"type": "added", "alias": f"Auto {i}", "description": f"test {i}"})
        entries = load_changelog()
        assert len(entries) == 5


class TestMaxSizeTrim:
    def test_trims_to_max_entries(self, tmp_data_dir):
        """Changelog is trimmed to MAX_ENTRIES."""
        entries = [{"type": "added", "alias": f"a{i}", "description": f"d{i}"} for i in range(MAX_ENTRIES + 50)]
        append_entries(entries)
        loaded = load_changelog()
        assert len(loaded) <= MAX_ENTRIES


class TestLoadChangelog:
    def test_load_missing_returns_empty(self, tmp_data_dir):
        """Missing file → empty list."""
        result = load_changelog()
        assert result == []

    def test_load_with_limit(self, tmp_data_dir):
        """limit parameter returns at most N entries."""
        for i in range(10):
            append_entry({"type": "added", "alias": f"a{i}", "description": f"d{i}"})
        entries = load_changelog(limit=3)
        assert len(entries) == 3

    def test_sorted_newest_first(self, tmp_data_dir):
        """Entries returned newest first."""
        for i in range(5):
            dt = datetime.datetime(2025, 1, i + 1, tzinfo=datetime.timezone.utc)
            append_entry({"type": "added", "alias": f"a{i}", "description": f"d{i}",
                         "timestamp": dt.isoformat()})
        entries = load_changelog()
        timestamps = [e["timestamp"] for e in entries]
        assert timestamps == sorted(timestamps, reverse=True)


class TestHelperFunctions:
    def test_log_habitus_add(self, tmp_data_dir):
        log_habitus_add("My Auto")
        entries = load_changelog()
        assert any(e["type"] == CHANGE_HABITUS_ADD for e in entries)
        assert any(e["alias"] == "My Auto" for e in entries)

    def test_log_habitus_remove(self, tmp_data_dir):
        log_habitus_remove("Old Auto")
        entries = load_changelog()
        assert any(e["type"] == "habitus_remove" for e in entries)

    def test_log_triggered(self, tmp_data_dir):
        log_triggered("Active Auto")
        entries = load_changelog()
        assert any(e["type"] == "triggered" for e in entries)


class TestRunDiffAndLog:
    def test_run_with_no_files(self, tmp_data_dir):
        """run_diff_and_log with no files doesn't crash."""
        changes = run_diff_and_log()
        assert isinstance(changes, list)

    def test_run_detects_added(self, tmp_data_dir):
        """run_diff_and_log detects new automations."""
        automations = [_make_automation("New Auto")]
        ha_auto_path = os.path.join(str(tmp_data_dir), "ha_automations.json")
        with open(ha_auto_path, "w") as f:
            json.dump(automations, f)

        changes = run_diff_and_log()
        assert any(c["type"] == CHANGE_ADDED for c in changes)
