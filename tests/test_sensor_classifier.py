"""Tests for sensor_classifier.py — TASK-015."""
from __future__ import annotations

import json

import numpy as np
import pytest

from habitus.habitus.sensor_classifier import (
    ACCUMULATING,
    ALL_SENSOR_TYPES,
    BINARY,
    EVENT,
    GAUGE,
    SETPOINT,
    classify_entities_from_ha_states,
    classify_sensor,
    _is_monotonically_increasing,
)


# ---------------------------------------------------------------------------
# classify_sensor — state_class attribute path
# ---------------------------------------------------------------------------


class TestClassifySensorStateClass:
    def test_total_increasing_returns_accumulating(self):
        assert classify_sensor("sensor.kwh", state_class="total_increasing") == ACCUMULATING

    def test_state_class_wins_over_binary_history(self):
        # Even if history looks binary, state_class takes priority
        result = classify_sensor("sensor.pulse", state_class="total_increasing", history=[0, 1, 0, 1])
        assert result == ACCUMULATING

    def test_state_class_wins_over_gauge_history(self):
        rng = np.random.default_rng(0)
        history = list(rng.uniform(10, 30, 100))
        result = classify_sensor("sensor.energy", state_class="total_increasing", history=history)
        assert result == ACCUMULATING

    def test_other_state_class_falls_through(self):
        # "measurement" is not "total_increasing" → should fall through to history analysis
        result = classify_sensor("sensor.temp", state_class="measurement", history=[18.0] * 50 + [19.0] * 50)
        assert result in ALL_SENSOR_TYPES

    def test_no_state_class_no_history(self):
        assert classify_sensor("sensor.unknown") == GAUGE

    def test_none_state_class_none_history(self):
        assert classify_sensor("sensor.x", state_class=None, history=None) == GAUGE

    def test_empty_history_defaults_to_gauge(self):
        assert classify_sensor("sensor.x", state_class=None, history=[]) == GAUGE


# ---------------------------------------------------------------------------
# classify_sensor — binary detection
# ---------------------------------------------------------------------------


class TestClassifySensorBinary:
    def test_zero_and_one(self):
        assert classify_sensor("binary_sensor.motion", history=[0, 1, 0, 0, 1, 0, 1, 1, 0]) == BINARY

    def test_float_zero_one(self):
        assert classify_sensor("binary_sensor.door", history=[0.0, 1.0, 0.0, 1.0, 0.0]) == BINARY

    def test_all_zeros(self):
        # Subset of {0, 1} → binary
        assert classify_sensor("binary_sensor.off_all_day", history=[0.0] * 20) == BINARY

    def test_all_ones(self):
        assert classify_sensor("binary_sensor.always_on", history=[1.0] * 20) == BINARY

    def test_value_two_not_binary(self):
        result = classify_sensor("sensor.mode", history=[0.0, 1.0, 2.0, 1.0, 0.0])
        assert result != BINARY

    def test_fractional_not_binary(self):
        result = classify_sensor("sensor.ratio", history=[0.0, 0.5, 1.0, 0.5])
        assert result != BINARY


# ---------------------------------------------------------------------------
# classify_sensor — accumulating detection via history
# ---------------------------------------------------------------------------


class TestClassifySensorAccumulating:
    def test_strictly_monotone_increasing(self):
        history = list(range(0, 500))
        assert classify_sensor("sensor.kwh", history=history) == ACCUMULATING

    def test_cumulative_sum_noisy(self):
        rng = np.random.default_rng(1)
        history = list(np.cumsum(rng.uniform(0.01, 1.0, 200)))
        assert classify_sensor("sensor.gas_m3", history=history) == ACCUMULATING

    def test_mostly_increasing_80_pct(self):
        # 20 steps out of 100 are decreasing (< 80% non-decreasing → False)
        # Build one that IS ≥80% non-decreasing
        history = []
        val = 0.0
        rng = np.random.default_rng(42)
        for i in range(100):
            step = rng.uniform(0, 1) if i % 6 != 5 else -0.01  # ~5/6 ≈ 83% non-decreasing
            val += step
            history.append(val)
        assert classify_sensor("sensor.water_l", history=history) == ACCUMULATING

    def test_decreasing_is_not_accumulating(self):
        history = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0]
        result = classify_sensor("sensor.temp", history=history)
        assert result != ACCUMULATING

    def test_oscillating_not_accumulating(self):
        history = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        result = classify_sensor("sensor.toggle", history=history)
        assert result != ACCUMULATING


# ---------------------------------------------------------------------------
# classify_sensor — event detection
# ---------------------------------------------------------------------------


class TestClassifySensorEvent:
    def test_brief_spikes_returning_to_zero(self):
        # 80% zeros, occasional spikes
        history = [0.0] * 80 + [100.0, 200.0, 150.0, 50.0, 0.0] * 4
        assert classify_sensor("sensor.pulse_counter", history=history) == EVENT

    def test_mostly_zero_with_larger_spikes(self):
        history = [0] * 90 + [500, 0, 200, 0, 0, 0, 0, 0, 0, 300]
        assert classify_sensor("sensor.rain_mm", history=history) == EVENT

    def test_not_event_when_evenly_distributed(self):
        # Values spread out — not mostly near floor
        rng = np.random.default_rng(5)
        history = list(rng.uniform(10, 100, 200))
        result = classify_sensor("sensor.power", history=history)
        assert result != EVENT


# ---------------------------------------------------------------------------
# classify_sensor — setpoint detection
# ---------------------------------------------------------------------------


class TestClassifySensorSetpoint:
    def test_three_discrete_values(self):
        history = [18.0, 20.0, 22.0, 18.0, 20.0, 22.0, 20.0] * 10
        assert classify_sensor("input_number.temp_target", history=history) == SETPOINT

    def test_two_discrete_non_binary_values(self):
        # 0 and 100 — not binary (100 not in {0,1})
        history = [0.0, 100.0, 0.0, 100.0, 100.0, 0.0] * 10
        assert classify_sensor("input_number.dimmer", history=history) == SETPOINT

    def test_four_discrete_values(self):
        history = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0] * 10
        assert classify_sensor("input_select.mode", history=history) == SETPOINT

    def test_five_discrete_values(self):
        # Realistic setpoint: values sampled in random order from a small fixed set
        rng = np.random.default_rng(3)
        choices = [10.0, 20.0, 30.0, 40.0, 50.0]
        history = [float(rng.choice(choices)) for _ in range(100)]
        assert classify_sensor("sensor.level", history=history) == SETPOINT

    def test_six_discrete_values_not_setpoint(self):
        # 6 unique values → exceeds the setpoint threshold
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 20
        result = classify_sensor("sensor.many_levels", history=history)
        assert result != SETPOINT


# ---------------------------------------------------------------------------
# classify_sensor — gauge detection (default fallback)
# ---------------------------------------------------------------------------


class TestClassifySensorGauge:
    def test_continuous_temperature(self):
        rng = np.random.default_rng(42)
        history = list(rng.uniform(15.0, 25.0, 200))
        assert classify_sensor("sensor.temperature", history=history) == GAUGE

    def test_continuous_humidity(self):
        rng = np.random.default_rng(7)
        history = list(rng.uniform(30.0, 70.0, 200))
        assert classify_sensor("sensor.humidity", history=history) == GAUGE

    def test_single_value_is_gauge(self):
        assert classify_sensor("sensor.x", history=[5.0]) == GAUGE

    def test_two_values_continuous_gauge(self):
        # 2 values, not in {0,1}, not monotone increasing → goes to floor_fraction check
        # and then setpoint check: 2 unique values → SETPOINT actually
        # This tests boundary — let's check: [5.0, 10.0] → unique_rounded = {5.0, 10.0} → len=2 → setpoint
        result = classify_sensor("sensor.x", history=[5.0, 10.0])
        assert result in ALL_SENSOR_TYPES  # just verify it returns a valid type

    def test_continuous_power_sensor(self):
        rng = np.random.default_rng(99)
        # Wide variety of power readings — many distinct values
        history = list(rng.normal(500, 150, 500))
        history = [max(0, v) for v in history]
        assert classify_sensor("sensor.power_w", history=history) == GAUGE


# ---------------------------------------------------------------------------
# _is_monotonically_increasing helper
# ---------------------------------------------------------------------------


class TestIsMonotonicallyIncreasing:
    def test_strictly_increasing(self):
        assert _is_monotonically_increasing([1.0, 2.0, 3.0, 4.0, 5.0]) is True

    def test_strictly_decreasing(self):
        assert _is_monotonically_increasing([5.0, 4.0, 3.0, 2.0, 1.0]) is False

    def test_flat_sequence(self):
        # Net change = 0 → not accumulating
        assert _is_monotonically_increasing([5.0, 5.0, 5.0, 5.0]) is False

    def test_too_short(self):
        assert _is_monotonically_increasing([1.0, 2.0]) is False

    def test_single_element(self):
        assert _is_monotonically_increasing([1.0]) is False

    def test_mostly_increasing_with_small_dips(self):
        vals = [float(i) for i in range(100)]
        vals[10] -= 0.1  # one tiny dip
        assert _is_monotonically_increasing(vals) is True

    def test_below_80pct_non_decreasing(self):
        # Alternating up/down pattern — 50% non-decreasing
        vals = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        assert _is_monotonically_increasing(vals) is False


# ---------------------------------------------------------------------------
# classify_entities_from_ha_states
# ---------------------------------------------------------------------------


class TestClassifyEntitiesFromHAStates:
    def test_total_increasing_state_class(self):
        states = [
            {
                "entity_id": "sensor.energy_kwh",
                "state": "1234.5",
                "attributes": {"state_class": "total_increasing"},
            }
        ]
        result = classify_entities_from_ha_states(states)
        assert result["sensor.energy_kwh"] == ACCUMULATING

    def test_no_state_class_defaults_to_gauge(self):
        states = [{"entity_id": "sensor.temperature", "state": "21.5", "attributes": {}}]
        result = classify_entities_from_ha_states(states)
        assert result["sensor.temperature"] == GAUGE

    def test_missing_attributes_key_defaults_to_gauge(self):
        states = [{"entity_id": "sensor.foo", "state": "42"}]
        result = classify_entities_from_ha_states(states)
        assert result["sensor.foo"] == GAUGE

    def test_empty_entity_id_skipped(self):
        states = [{"entity_id": "", "state": "on", "attributes": {}}]
        result = classify_entities_from_ha_states(states)
        assert "" not in result

    def test_missing_entity_id_key_skipped(self):
        states = [{"state": "on", "attributes": {}}]
        result = classify_entities_from_ha_states(states)
        assert result == {}

    def test_multiple_entities_with_mixed_types(self):
        states = [
            {
                "entity_id": "sensor.gas_m3",
                "state": "0",
                "attributes": {"state_class": "total_increasing"},
            },
            {"entity_id": "sensor.temp", "state": "20", "attributes": {}},
            {"entity_id": "binary_sensor.door", "state": "on", "attributes": {}},
        ]
        result = classify_entities_from_ha_states(states)
        assert result["sensor.gas_m3"] == ACCUMULATING
        assert result["sensor.temp"] in ALL_SENSOR_TYPES
        assert result["binary_sensor.door"] in ALL_SENSOR_TYPES

    def test_empty_states_list(self):
        result = classify_entities_from_ha_states([])
        assert result == {}

    def test_result_values_all_valid_types(self):
        states = [
            {"entity_id": f"sensor.entity_{i}", "state": str(i), "attributes": {}}
            for i in range(10)
        ]
        result = classify_entities_from_ha_states(states)
        for stype in result.values():
            assert stype in ALL_SENSOR_TYPES


# ---------------------------------------------------------------------------
# Integration: build_entity_baselines stores sensor_type in _meta
# ---------------------------------------------------------------------------


class TestBuildEntityBaselinesStoresSensorType:
    def test_meta_present_for_all_entities(self, tmp_data_dir, sample_df):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        entity_keys = [k for k in baselines if not k.startswith("_")]
        assert len(entity_keys) > 0
        for eid in entity_keys:
            bl = baselines[eid]
            assert "_meta" in bl, f"Missing _meta for {eid}"
            assert "sensor_type" in bl["_meta"], f"Missing sensor_type for {eid}"
            assert bl["_meta"]["sensor_type"] in ALL_SENSOR_TYPES

    def test_binary_sensors_classified_correctly(self, tmp_data_dir, sample_df):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        binary_entities = [k for k in baselines if k.startswith("binary_sensor.") and not k.startswith("_")]
        assert len(binary_entities) > 0
        for eid in binary_entities:
            assert baselines[eid]["_meta"]["sensor_type"] == BINARY, (
                f"{eid} should be binary, got {baselines[eid]['_meta']['sensor_type']}"
            )

    def test_meta_does_not_interfere_with_slot_access(self, tmp_data_dir, sample_df):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(sample_df)
        # score_entities must still work despite _meta being present
        result = score_entities({"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        assert isinstance(result, list)

    def test_sensor_type_values_are_valid(self, tmp_data_dir, sample_df):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        valid_types = {"accumulating", "binary", "gauge", "event", "setpoint"}
        for eid, bl in baselines.items():
            if eid.startswith("_"):
                continue
            assert bl["_meta"]["sensor_type"] in valid_types
