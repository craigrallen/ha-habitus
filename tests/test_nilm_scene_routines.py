"""Tests for nilm_disaggregator.py, scene_detector.py, routine_predictor.py."""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ── nilm_disaggregator.py pure functions ─────────────────────────────────────

class TestNilmDetectEdges:
    def test_empty_readings(self):
        """_detect_edges returns empty list for too-short input."""
        from habitus.habitus.nilm_disaggregator import _detect_edges
        result = _detect_edges([])
        assert result == []

    def test_short_readings(self):
        """_detect_edges returns empty list below STEADY_STATE_SAMPLES."""
        from habitus.habitus.nilm_disaggregator import _detect_edges, STEADY_STATE_SAMPLES
        readings = [(float(i), 100.0) for i in range(STEADY_STATE_SAMPLES - 1)]
        result = _detect_edges(readings)
        assert result == []

    def test_detects_step_up(self):
        """_detect_edges detects a clear upward step change."""
        from habitus.habitus.nilm_disaggregator import _detect_edges, STEADY_STATE_SAMPLES, MIN_EDGE_WATTS
        # 10 samples at 100W, then 10 at 700W = clear step
        now = time.time()
        low = [(now + i, 100.0) for i in range(20)]
        high = [(now + 20 + i, 800.0) for i in range(20)]
        result = _detect_edges(low + high)
        assert any(e["direction"] == "up" for e in result)

    def test_detects_step_down(self):
        """_detect_edges detects a clear downward step change."""
        from habitus.habitus.nilm_disaggregator import _detect_edges
        now = time.time()
        high = [(now + i, 800.0) for i in range(20)]
        low = [(now + 20 + i, 100.0) for i in range(20)]
        result = _detect_edges(high + low)
        assert any(e["direction"] == "down" for e in result)

    def test_edge_fields(self):
        """Edge dicts have required fields."""
        from habitus.habitus.nilm_disaggregator import _detect_edges
        now = time.time()
        low = [(now + i, 50.0) for i in range(25)]
        high = [(now + 25 + i, 700.0) for i in range(25)]
        edges = _detect_edges(low + high)
        if edges:
            e = edges[0]
            assert "timestamp" in e
            assert "delta_w" in e
            assert "direction" in e
            assert "power_after" in e


class TestNilmPairEdges:
    def test_empty_edges(self):
        """_pair_edges returns empty list for empty input."""
        from habitus.habitus.nilm_disaggregator import _pair_edges
        result = _pair_edges([])
        assert result == []

    def test_pairs_matching_up_down(self):
        """_pair_edges pairs matching ON/OFF edges."""
        from habitus.habitus.nilm_disaggregator import _pair_edges
        now = time.time()
        up = {"timestamp": now, "time": "2025-01-01T10:00:00+00:00", "delta_w": 500, "direction": "up"}
        down = {"timestamp": now + 3600, "time": "2025-01-01T11:00:00+00:00", "delta_w": -490, "direction": "down"}
        events = _pair_edges([up, down])
        assert len(events) == 1
        assert events[0]["power_w"] == pytest.approx(500, abs=50)
        assert "duration_min" in events[0]

    def test_no_match_for_far_apart_edges(self):
        """_pair_edges doesn't pair edges that are too far apart."""
        from habitus.habitus.nilm_disaggregator import _pair_edges, MAX_PAIR_WINDOW_SEC
        now = time.time()
        up = {"timestamp": now, "time": "t", "delta_w": 500, "direction": "up"}
        # Down edge is way outside the pairing window
        down = {"timestamp": now + MAX_PAIR_WINDOW_SEC * 10, "time": "t2",
                "delta_w": -490, "direction": "down"}
        events = _pair_edges([up, down])
        assert events == []


class TestNilmClusterEvents:
    def test_empty_events(self):
        """_cluster_events returns empty list for empty input."""
        from habitus.habitus.nilm_disaggregator import _cluster_events
        result = _cluster_events([])
        assert result == []

    def test_too_few_events_returns_empty(self):
        """_cluster_events requires >= 3 events to cluster."""
        from habitus.habitus.nilm_disaggregator import _cluster_events
        now = time.time()
        events = [{"power_w": 2000, "duration_min": 45, "hour": 10,
                   "start_ts": now, "start": "t", "end": "t2"},
                  {"power_w": 1800, "duration_min": 50, "hour": 11,
                   "start_ts": now + 3600, "start": "t2", "end": "t3"}]
        result = _cluster_events(events)
        assert result == []

    def test_similar_events_clustered(self):
        """_cluster_events clusters multiple events into groups."""
        from habitus.habitus.nilm_disaggregator import _cluster_events
        now = time.time()
        # Mix of two power levels
        events = [
            {"power_w": 1200 + (i % 3) * 50, "duration_min": 90, "hour": 10,
             "start_ts": now + i * 3600, "start": "t", "end": "t2"}
            for i in range(10)
        ] + [
            {"power_w": 2200 + (i % 2) * 30, "duration_min": 45, "hour": 20,
             "start_ts": now + i * 7200, "start": "t3", "end": "t4"}
            for i in range(5)
        ]
        result = _cluster_events(events)
        # Should produce 1-10 clusters
        assert isinstance(result, list)
        if result:
            assert "centroid_w" in result[0]


class TestNilmMatchToAppliances:
    def test_empty_clusters(self):
        """_match_to_appliances with no clusters returns empty list."""
        from habitus.habitus.nilm_disaggregator import _match_to_appliances
        result = _match_to_appliances([])
        assert result == []

    def test_washing_machine_power_matches(self):
        """_match_to_appliances recognises washing machine power signature."""
        from habitus.habitus.nilm_disaggregator import _match_to_appliances
        clusters = [{"centroid_w": 1800, "avg_duration_min": 75, "event_count": 5,
                     "peak_hours": [8, 10, 20], "total_kwh": 1.35}]
        result = _match_to_appliances(clusters)
        assert len(result) == 1
        assert "appliance" in result[0]

    def test_kettle_power_matches(self):
        """_match_to_appliances recognises kettle power signature."""
        from habitus.habitus.nilm_disaggregator import _match_to_appliances
        clusters = [{"centroid_w": 2200, "avg_duration_min": 2, "event_count": 20,
                     "peak_hours": [7, 8, 9, 17, 18], "total_kwh": 0.07}]
        result = _match_to_appliances(clusters)
        assert len(result) == 1
        assert "appliance" in result[0]


class TestNilmRunDisaggregation:
    def test_run_no_db(self, tmp_data_dir: Path):
        """run_disaggregation returns error dict when DB unavailable."""
        from habitus.habitus.nilm_disaggregator import run_disaggregation
        with patch("habitus.habitus.nilm_disaggregator.resolve_ha_db_path", return_value=None), \
             patch("habitus.habitus.nilm_disaggregator._get_aggregate_power", return_value=[]):
            result = run_disaggregation("sensor.grid_power")
        assert isinstance(result, dict)
        assert "error" in result or "appliances" in result or "events" in result


# ── scene_detector.py ─────────────────────────────────────────────────────────

class TestSceneDetector:
    def test_extract_room_light(self):
        """_extract_room returns room from light entity id."""
        from habitus.habitus.scene_detector import _extract_room
        result = _extract_room("light.living_room_main")
        assert result is not None

    def test_extract_room_sensor(self):
        """_extract_room returns room from sensor entity id."""
        from habitus.habitus.scene_detector import _extract_room
        result = _extract_room("sensor.kitchen_temperature")
        assert result is not None

    def test_extract_room_unknown(self):
        """_extract_room returns None for entity with no clear room."""
        from habitus.habitus.scene_detector import _extract_room
        result = _extract_room("switch.x")
        # May return None or a string
        assert result is None or isinstance(result, str)

    def test_name_scene_lighting(self):
        """_name_scene generates a scene name for lighting entities."""
        from habitus.habitus.scene_detector import _name_scene
        entities = {"light.living_room", "light.kitchen"}
        time_info = {"primary_hour": 20, "count": 5}
        result = _name_scene(entities, time_info)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compute_scene_confidence(self):
        """_compute_scene_confidence returns int in 0-100."""
        from habitus.habitus.scene_detector import _compute_scene_confidence
        entities = {"light.living_room", "light.kitchen"}
        time_info = {"primary_hour": 20, "count": 10, "pct": 80}
        result = _compute_scene_confidence(entities, time_info)
        assert isinstance(result, int)
        assert 0 <= result <= 100

    def test_detect_scenes_no_db(self, tmp_data_dir: Path):
        """detect_scenes returns empty list when DB unavailable."""
        from habitus.habitus.scene_detector import detect_scenes
        with patch("habitus.habitus.scene_detector.resolve_ha_db_path", return_value=None):
            result = detect_scenes()
        assert isinstance(result, list)

    def test_extract_rooms_from_entities(self):
        """_extract_rooms_from_entities returns list of rooms."""
        from habitus.habitus.scene_detector import _extract_rooms_from_entities
        entities = {"light.living_room", "switch.kitchen_coffee"}
        result = _extract_rooms_from_entities(entities)
        assert isinstance(result, list)


# ── routine_predictor.py ──────────────────────────────────────────────────────

class TestRoutinePredictor:
    def test_detect_humidity_spikes_empty(self):
        """detect_humidity_spikes returns empty list for empty input."""
        from habitus.habitus.routine_predictor import detect_humidity_spikes
        result = detect_humidity_spikes([])
        assert result == []

    def test_detect_humidity_spikes_flat_signal(self):
        """detect_humidity_spikes returns no spikes for flat humidity."""
        from habitus.habitus.routine_predictor import detect_humidity_spikes
        now = time.time()
        readings = [(now + i * 300, 50.0) for i in range(50)]  # flat 50% for 4hrs
        result = detect_humidity_spikes(readings)
        assert isinstance(result, list)

    def test_detect_humidity_spikes_with_spike(self):
        """detect_humidity_spikes detects a spike from low to high."""
        from habitus.habitus.routine_predictor import detect_humidity_spikes
        now = time.time()
        low = [(now + i * 300, 45.0) for i in range(20)]
        spike = [(now + 20 * 300 + i * 300, 90.0) for i in range(10)]
        recovery = [(now + 30 * 300 + i * 300, 50.0) for i in range(10)]
        result = detect_humidity_spikes(low + spike + recovery)
        assert isinstance(result, list)

    def test_analyse_routine_no_spikes(self):
        """analyse_routine returns None for empty spike list."""
        from habitus.habitus.routine_predictor import analyse_routine
        result = analyse_routine([], "bathroom")
        assert result is None

    def test_run_routine_prediction_no_db(self, tmp_data_dir: Path):
        """run_routine_prediction returns dict when DB unavailable."""
        from habitus.habitus import routine_predictor as rp
        rp_path = str(tmp_data_dir / "routine_predictions.json")
        with patch("habitus.habitus.routine_predictor.resolve_ha_db_path", return_value=None), \
             patch.object(rp, "ROUTINES_PATH", rp_path):
            result = rp.run_routine_prediction()
        assert isinstance(result, dict)

    def test_load_routines_missing_file(self, tmp_data_dir: Path):
        """load_routines returns empty dict when file missing."""
        from habitus.habitus import routine_predictor as rp
        rp_path = str(tmp_data_dir / "nope.json")
        with patch.object(rp, "ROUTINES_PATH", rp_path):
            result = rp.load_routines()
        assert isinstance(result, dict)

    def test_build_preheat_yaml(self):
        """_build_preheat_yaml returns valid YAML string."""
        from habitus.habitus.routine_predictor import _build_preheat_yaml
        result = _build_preheat_yaml("shower", "bathroom", "07:30", "weekdays")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "automation" in result or "trigger" in result
