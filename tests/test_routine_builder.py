"""Tests for routine builder — sequence detection and YAML generation."""
from __future__ import annotations

import datetime
import json
import os

import pytest

from habitus.habitus.routine_builder import (
    mine_sequences,
    generate_routine_yaml,
    run,
    load_routines,
    _classify_time_cluster,
    MIN_FREQUENCY_DAYS,
    MIN_SEQUENCE_LENGTH,
)


def _make_event(entity_id: str, state: str, hour: int, day_offset: int = 0, minute: int = 0) -> dict:
    """Helper to build a test event."""
    dt = datetime.datetime(2025, 1, 1 + day_offset, hour, minute, tzinfo=datetime.timezone.utc)
    return {"entity_id": entity_id, "state": state, "timestamp": dt.isoformat()}


def _make_daily_sequence(day_offset: int, hour: int = 7) -> list[dict]:
    """Build a 3-step sequence at a given day/hour."""
    return [
        _make_event("light.bedroom", "on", hour, day_offset, 0),
        _make_event("light.kitchen", "on", hour, day_offset, 2),
        _make_event("media_player.kitchen", "on", hour, day_offset, 5),
    ]


class TestMineSequences:
    def test_detects_repeated_sequence(self):
        """Repeated sequence on 5+ days is detected."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=7))
        sequences = mine_sequences(events)
        assert len(sequences) > 0

    def test_minimum_frequency_gate(self):
        """Sequence on fewer than MIN_FREQUENCY_DAYS is not detected."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS - 1):
            events.extend(_make_daily_sequence(day, hour=7))
        sequences = mine_sequences(events)
        assert len(sequences) == 0

    def test_minimum_sequence_length(self):
        """Sequence of length >= MIN_SEQUENCE_LENGTH is detected."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=7))
        sequences = mine_sequences(events)
        for seq in sequences:
            assert len(seq["steps"]) >= MIN_SEQUENCE_LENGTH

    def test_time_cluster_morning(self):
        """Sequence at 7am is classified as morning."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=7))
        sequences = mine_sequences(events)
        assert len(sequences) > 0
        assert sequences[0]["time_cluster"] == "morning"

    def test_time_cluster_evening(self):
        """Sequence at 19:00 is classified as evening."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=19))
        sequences = mine_sequences(events)
        assert len(sequences) > 0
        assert sequences[0]["time_cluster"] == "evening"

    def test_confidence_increases_with_frequency(self):
        """Higher frequency → higher confidence."""
        events_low = []
        for day in range(MIN_FREQUENCY_DAYS):
            events_low.extend(_make_daily_sequence(day, hour=7))

        events_high = []
        for day in range(MIN_FREQUENCY_DAYS + 5):
            events_high.extend(_make_daily_sequence(day, hour=7))

        seqs_low = mine_sequences(events_low)
        seqs_high = mine_sequences(events_high)

        if seqs_low and seqs_high:
            assert seqs_high[0]["confidence"] >= seqs_low[0]["confidence"]

    def test_empty_events_returns_empty(self):
        """Empty events list returns no sequences."""
        result = mine_sequences([])
        assert result == []

    def test_result_has_required_fields(self):
        """Each sequence has required fields."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=7))
        sequences = mine_sequences(events)
        if sequences:
            seq = sequences[0]
            assert "steps" in seq
            assert "frequency_days" in seq
            assert "time_cluster" in seq
            assert "confidence" in seq


class TestGenerateRoutineYaml:
    def test_generates_valid_yaml(self):
        """generate_routine_yaml returns valid YAML string."""
        import yaml
        routine = {
            "steps": [
                {"entity_id": "light.bedroom", "state": "on"},
                {"entity_id": "light.kitchen", "state": "on"},
                {"entity_id": "media_player.radio", "state": "on"},
            ],
            "time_cluster": "morning",
            "avg_hour": 7.0,
            "frequency_days": 10,
            "confidence": 0.85,
        }
        yaml_str = generate_routine_yaml(routine)
        assert yaml_str
        assert isinstance(yaml_str, str)
        # Should be parseable YAML
        parsed = yaml.safe_load(yaml_str)
        assert parsed is not None

    def test_yaml_contains_entity_refs(self):
        """Generated YAML references the sequence entities."""
        routine = {
            "steps": [
                {"entity_id": "light.bedroom", "state": "on"},
                {"entity_id": "light.kitchen", "state": "on"},
                {"entity_id": "switch.coffee", "state": "on"},
            ],
            "time_cluster": "morning",
            "avg_hour": 7.5,
            "frequency_days": 7,
            "confidence": 0.75,
        }
        yaml_str = generate_routine_yaml(routine)
        assert "light.bedroom" in yaml_str
        assert "light.kitchen" in yaml_str


class TestClassifyTimeCluster:
    def test_morning_hours(self):
        for h in [5, 6, 7, 8]:
            assert _classify_time_cluster(h) == "morning"

    def test_evening_hours(self):
        for h in [17, 18, 19, 20, 21]:
            assert _classify_time_cluster(h) == "evening"

    def test_night_hours(self):
        for h in [22, 23, 0]:
            assert _classify_time_cluster(h) == "night"


class TestRun:
    def test_run_saves_routines(self, tmp_data_dir):
        """run() saves routines.json."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=7))
        result = run(events)
        assert "routines" in result
        assert os.path.exists(os.path.join(str(tmp_data_dir), "routines.json"))

    def test_run_empty_events(self, tmp_data_dir):
        """run() with empty events returns empty routines."""
        result = run([])
        assert result["total"] == 0
        assert result["routines"] == []

    def test_load_routines(self, tmp_data_dir):
        """load_routines returns saved data."""
        events = []
        for day in range(MIN_FREQUENCY_DAYS):
            events.extend(_make_daily_sequence(day, hour=7))
        run(events)
        loaded = load_routines()
        assert "routines" in loaded
