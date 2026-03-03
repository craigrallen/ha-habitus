"""Tests for per-entity anomaly scoring."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import responses

from habitus.habitus.anomaly_breakdown import (
    build_entity_baselines,
    compute_breakdown,
    compute_entity_confidence,
    compute_weighted_score,
    score_entities,
    _guess_unit,
    _fmt,
    _day,
    COLD_START_DAYS,
    LOW_SAMPLE_THRESHOLD,
    MIN_CONFIDENCE,
    SENSOR_TYPE_CERTAINTY,
    ENTITY_BASELINES_PATH,
    ENTITY_ANOMALIES_PATH,
    ENTITY_LIFECYCLE_PATH,
)


class TestBuildEntityBaselines:
    def test_saves_file(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        assert (tmp_data_dir / "entity_baselines.json").exists()

    def test_baseline_structure(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert len(data) > 0
        # Find an absolute-type slot and verify its structure
        for eid, entity in data.items():
            if eid.startswith("_"):
                continue
            for key, slot in entity.items():
                if key.startswith("_") or not isinstance(slot, dict):
                    continue
                if slot.get("baseline_type") == "absolute":
                    assert "mean" in slot and "std" in slot and "n" in slot
                    return
        pytest.skip("No absolute-type slots found in baselines")

    def test_empty_df_safe(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(pd.DataFrame())  # Should not raise

    def test_skips_slots_with_fewer_than_3_samples(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        # Two rows only — should produce no baseline slots
        df = pd.DataFrame(
            [
                {"entity_id": "sensor.foo", "ts": "2025-01-01T00:00:00Z", "mean": 1.0, "sum": None},
                {"entity_id": "sensor.foo", "ts": "2025-01-08T00:00:00Z", "mean": 2.0, "sum": None},
            ]
        )
        build_entity_baselines(df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        # Either entity missing or has no slots (all require ≥3 samples)
        if "sensor.foo" in data:
            assert len(data["sensor.foo"]) == 0

    def test_uses_sum_when_mean_is_null(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        hours = pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz="UTC")
        rows = [
            {"entity_id": "sensor.energy", "ts": h, "mean": None, "sum": float(i % 10 + 1)}
            for i, h in enumerate(hours)
        ]
        build_entity_baselines(pd.DataFrame(rows))
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "sensor.energy" in data


class TestGuessUnit:
    def test_watts(self):
        assert _guess_unit("sensor.bathroom_lights_electric_consumed_w") == "W"

    def test_temperature(self):
        assert _guess_unit("sensor.living_room_thermostat_air_temperature") == "°C"

    def test_humidity(self):
        assert _guess_unit("sensor.kitchen_humidity") == "%"

    def test_unknown(self):
        assert _guess_unit("sensor.mystery_value") == ""

    def test_energy_kwh(self):
        assert _guess_unit("sensor.solar_energy_kwh") == "kWh"

    def test_current_amps(self):
        # Entity ID must not contain "power", "watt", or "_w" so current/A branch is reached
        assert _guess_unit("sensor.battery_current_a") == "A"

    def test_voltage(self):
        assert _guess_unit("sensor.battery_voltage") == "V"

    def test_pressure(self):
        assert _guess_unit("sensor.barometric_pressure") == "hPa"

    def test_watt_keyword(self):
        assert _guess_unit("sensor.device_watt_usage") == "W"


class TestHelpers:
    def test_fmt_with_unit(self):
        result = _fmt(42.5, "W")
        assert "42.5" in result and "W" in result

    def test_fmt_no_unit(self):
        result = _fmt(3.14, "")
        assert "3.14" in result

    def test_day_names(self):
        assert _day(0) == "Mon"
        assert _day(6) == "Sun"

    def test_day_all_seven(self):
        expected = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, name in enumerate(expected):
            assert _day(i) == name


class TestScoreEntities:
    def test_no_baselines_returns_empty(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        result = score_entities({"sensor.foo": 42.0})
        assert result == []

    def test_scores_entity_with_extreme_value(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        current = {"sensor.bathroom_lights_electric_consumed_w": 99999.0}
        result = score_entities(current)
        assert len(result) > 0
        assert result[0]["z_score"] > 0
        assert "description" in result[0]

    def test_output_sorted_by_zscore_desc(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        current = {
            "sensor.bathroom_lights_electric_consumed_w": 99999.0,
            "sensor.kitchen_sensor_electric_consumed_w": 99999.0,
        }
        result = score_entities(current)
        scores = [r["z_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_saves_entity_anomalies_json(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        score_entities({"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        assert (tmp_data_dir / "entity_anomalies.json").exists()
        with open(tmp_data_dir / "entity_anomalies.json") as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "anomalies" in data

    def test_entity_not_in_current_states_skipped(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        build_entity_baselines(sample_df)
        # No matching entity in current_states
        result = score_entities({"sensor.nonexistent_entity": 100.0})
        # All baseline entities have no current value → empty result
        assert isinstance(result, list)

    def test_low_zscore_entity_excluded(self, sample_df, tmp_data_dir):
        import datetime
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)
        now = datetime.datetime.now()
        # Use the current time's slot so score_entities() looks up the same slot we read
        current_key = f"{now.hour}_{now.weekday()}"
        for eid, slots in baselines.items():
            if eid.startswith("_"):
                continue
            if current_key not in slots:
                continue
            slot = slots[current_key]
            if not isinstance(slot, dict) or slot.get("baseline_type") != "absolute":
                continue  # skip binary/rate sensors — they use on_fraction not mean
            mean_val = slot["mean"]
            # Pass value exactly at mean → z_score = 0, below 0.5 threshold → excluded
            result = score_entities({eid: mean_val})
            ids = [r["entity_id"] for r in result]
            assert eid not in ids
            break

    def test_anomaly_result_has_required_fields(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        result = score_entities({"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        if result:
            r = result[0]
            for field in ("entity_id", "name", "current_value", "baseline_mean", "baseline_std", "z_score", "unit", "description", "direction"):
                assert field in r, f"Missing field: {field}"

    def test_direction_high(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        result = score_entities({"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        if result and result[0]["entity_id"] == "sensor.bathroom_lights_electric_consumed_w":
            assert result[0]["direction"] == "high"

    def test_direction_low(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        result = score_entities({"sensor.mastervolt_total_load": -99999.0})
        if result and result[0]["entity_id"] == "sensor.mastervolt_total_load":
            assert result[0]["direction"] == "low"

    @responses.activate
    def test_fetches_from_ha_when_no_current_states(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        responses.add(
            responses.GET,
            "http://localhost:8123/api/states",
            json=[
                {
                    "entity_id": "sensor.bathroom_lights_electric_consumed_w",
                    "state": "99999.0",
                }
            ],
            status=200,
        )
        result = score_entities()  # None → calls _fetch_current_states
        assert isinstance(result, list)

    @responses.activate
    def test_ha_fetch_error_returns_empty_gracefully(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        responses.add(
            responses.GET,
            "http://localhost:8123/api/states",
            status=500,
        )
        # Should not raise even with HA error
        result = score_entities()
        assert isinstance(result, list)


class TestComputeBreakdown:
    def test_below_threshold_returns_empty(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        assert compute_breakdown(40.0) == []

    def test_at_threshold_returns_empty(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        assert compute_breakdown(40) == []

    def test_zero_score_returns_empty(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        assert compute_breakdown(0.0) == []

    def test_above_threshold_calls_score_entities(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        current = {"sensor.bathroom_lights_electric_consumed_w": 99999.0}
        result = compute_breakdown(50.0, current)
        assert isinstance(result, list)

    def test_limits_to_top5(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        # Drive multiple entities to extreme values
        current = {
            "sensor.bathroom_lights_electric_consumed_w": 99999.0,
            "sensor.kitchen_sensor_electric_consumed_w": 99999.0,
            "sensor.mastervolt_total_load": 99999.0,
            "sensor.living_room_thermostat_air_temperature": 999.0,
            "sensor.bilge_sensor_air_temperature": 999.0,
            "sensor.outdoor_temperature": 999.0,
        }
        result = compute_breakdown(80.0, current)
        assert len(result) <= 5

    def test_updates_entity_baselines_json(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        compute_breakdown(75.0, {"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "_z_score_run" in data
        run = data["_z_score_run"]
        assert "timestamp" in run
        assert "anomaly_score" in run
        assert run["anomaly_score"] == 75.0
        assert "top5" in run
        assert isinstance(run["top5"], list)

    def test_no_baselines_file_returns_empty_list(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        # No baselines built — score_entities() returns [] → top5 is []
        result = compute_breakdown(75.0, {"sensor.foo": 42.0})
        assert result == []
        # entity_baselines.json should NOT be created from scratch
        assert not (tmp_data_dir / "entity_baselines.json").exists()

    def test_breakdown_result_has_required_fields(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        result = compute_breakdown(85.0, {"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        if result:
            r = result[0]
            for field in (
                "entity_id",
                "name",
                "current_value",
                "baseline_mean",
                "baseline_std",
                "z_score",
                "description",
            ):
                assert field in r, f"Missing field: {field}"

    def test_breakdown_description_contains_baseline_info(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        result = compute_breakdown(90.0, {"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        if result:
            desc = result[0]["description"]
            assert "baseline" in desc.lower()

    def test_baselines_json_preserves_existing_entity_slots(self, sample_df, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        build_entity_baselines(sample_df)
        compute_breakdown(70.0, {"sensor.bathroom_lights_electric_consumed_w": 99999.0})
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        # Entity baselines should still be there alongside _z_score_run
        entity_keys = [k for k in data if not k.startswith("_")]
        assert len(entity_keys) > 0


class TestBinaryBaseline:
    def test_binary_baseline_has_on_fraction(self, sample_df, tmp_data_dir):
        """Binary sensors must have on_fraction in their baseline slots."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "binary_sensor.front_door" in data
        entity = data["binary_sensor.front_door"]
        found_slot = False
        for key, slot in entity.items():
            if key.startswith("_") or not isinstance(slot, dict):
                continue
            assert slot["baseline_type"] == "binary"
            assert "on_fraction" in slot
            assert 0.0 <= slot["on_fraction"] <= 1.0
            assert "avg_transitions" in slot
            found_slot = True
        assert found_slot, "No binary slot found for binary_sensor.front_door"

    def test_binary_baseline_has_binary_meta(self, sample_df, tmp_data_dir):
        """Binary entity baseline must include _binary_meta with avg_duration_on."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "binary_sensor.front_door" in data
        entity = data["binary_sensor.front_door"]
        assert "_binary_meta" in entity
        assert "avg_duration_on" in entity["_binary_meta"]
        assert entity["_binary_meta"]["avg_duration_on"] > 0

    def test_non_binary_entity_has_mean_not_on_fraction(self, sample_df, tmp_data_dir):
        """Non-binary entities should still have mean/std, not on_fraction."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        eid = "sensor.bathroom_lights_electric_consumed_w"
        assert eid in data
        for key, slot in data[eid].items():
            if key.startswith("_") or not isinstance(slot, dict):
                continue
            assert slot.get("baseline_type") == "absolute"
            assert "mean" in slot
            assert "on_fraction" not in slot


class TestBinaryScoring:
    def _write_binary_baselines(
        self,
        tmp_data_dir,
        on_fraction: float,
        avg_transitions: float = 0.0,
        avg_duration_on: float = 1.0,
    ) -> None:
        import datetime
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        bern_std = float(np.sqrt(max(on_fraction * (1.0 - on_fraction), 1e-4)))
        baselines = {
            "binary_sensor.test_motion": {
                f"{h}_{d}": {
                    "on_fraction": on_fraction,
                    "std": bern_std,
                    "avg_transitions": avg_transitions,
                    "n": 30,
                    "baseline_type": "binary",
                },
                "_binary_meta": {"avg_duration_on": avg_duration_on},
                "_meta": {"sensor_type": "binary"},
            }
        }
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

    def test_normally_off_sensor_on_is_anomalous(self, tmp_data_dir):
        """Sensor normally 5% on → being on should score high (> 1.0)."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(tmp_data_dir, on_fraction=0.05)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        result = score_entities({"binary_sensor.test_motion": 1.0})
        assert len(result) > 0
        assert result[0]["entity_id"] == "binary_sensor.test_motion"
        assert result[0]["z_score"] > 1.0

    def test_normally_on_sensor_on_is_not_anomalous(self, tmp_data_dir):
        """Sensor normally 90% on → z_expected < 0.5 → not flagged."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(tmp_data_dir, on_fraction=0.9)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        result = score_entities({"binary_sensor.test_motion": 1.0})
        ids = [r["entity_id"] for r in result]
        assert "binary_sensor.test_motion" not in ids

    def test_binary_sensor_off_never_anomalous(self, tmp_data_dir):
        """Binary sensor being off (value=0) must never be flagged."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(tmp_data_dir, on_fraction=0.05)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        result = score_entities({"binary_sensor.test_motion": 0.0})
        ids = [r["entity_id"] for r in result]
        assert "binary_sensor.test_motion" not in ids

    def test_frequency_anomaly_detected(self, tmp_data_dir):
        """transitions_this_hour > 3 × baseline → frequency anomaly flagged."""
        import datetime

        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(tmp_data_dir, on_fraction=0.5, avg_transitions=1.0)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")

        # Inject binary state: 5 transitions already recorded this hour (sensor currently on)
        now = datetime.datetime.now()
        with open(tmp_data_dir / "entity_baselines.json") as f:
            bl = json.load(f)
        bl["_binary_state"] = {
            "binary_sensor.test_motion": {
                "current_value": 1.0,
                "hour": now.hour,
                "transitions_this_hour": 5,
                "state_start_ts": now.isoformat(),
            }
        }
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(bl, f)

        # Transition: sensor goes to 0, making 6 total > 3 × 1.0
        result = score_entities({"binary_sensor.test_motion": 0.0})
        ids = [r["entity_id"] for r in result]
        assert "binary_sensor.test_motion" in ids
        entry = next(r for r in result if r["entity_id"] == "binary_sensor.test_motion")
        assert "transitions" in entry["description"]

    def test_duration_anomaly_detected(self, tmp_data_dir):
        """Sensor on for > 2 × avg_duration_on → duration anomaly flagged."""
        import datetime

        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(
            tmp_data_dir, on_fraction=0.3, avg_transitions=0.0, avg_duration_on=1.0
        )
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")

        now = datetime.datetime.now()
        five_hours_ago = (now - datetime.timedelta(hours=5)).isoformat()
        with open(tmp_data_dir / "entity_baselines.json") as f:
            bl = json.load(f)
        bl["_binary_state"] = {
            "binary_sensor.test_motion": {
                "current_value": 1.0,
                "hour": now.hour,
                "transitions_this_hour": 0,
                "state_start_ts": five_hours_ago,
            }
        }
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(bl, f)

        result = score_entities({"binary_sensor.test_motion": 1.0})
        ids = [r["entity_id"] for r in result]
        assert "binary_sensor.test_motion" in ids
        entry = next(r for r in result if r["entity_id"] == "binary_sensor.test_motion")
        assert "on for" in entry["description"]

    def test_anomaly_result_fields_present(self, tmp_data_dir):
        """Binary anomaly result must contain all required fields."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(tmp_data_dir, on_fraction=0.02)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        result = score_entities({"binary_sensor.test_motion": 1.0})
        assert result
        r = result[0]
        for field in (
            "entity_id",
            "name",
            "current_value",
            "baseline_mean",
            "baseline_std",
            "z_score",
            "unit",
            "description",
            "direction",
        ):
            assert field in r, f"Missing field: {field}"
        assert r["direction"] == "high"

    def test_binary_state_persisted_to_json(self, tmp_data_dir):
        """After scoring, _binary_state must be written to entity_baselines.json."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_binary_baselines(tmp_data_dir, on_fraction=0.5)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        score_entities({"binary_sensor.test_motion": 1.0})
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "_binary_state" in data
        assert "binary_sensor.test_motion" in data["_binary_state"]


class TestColdStartProtection:
    """Tests for TASK-018: Per-Entity Cold Start Protection."""

    def _write_baselines_with_age(
        self,
        tmp_data_dir,
        days_old: float,
        n_slot_samples: int = 30,
        on_fraction: float | None = None,
    ) -> None:
        """Write a minimal baselines file for sensor.test_entity with a given age."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")

        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        first_seen = (now - datetime.timedelta(days=days_old)).isoformat()

        if on_fraction is not None:
            # Binary sensor
            bern_std = float(np.sqrt(max(on_fraction * (1.0 - on_fraction), 1e-4)))
            slot = {
                "on_fraction": on_fraction,
                "std": bern_std,
                "avg_transitions": 0.0,
                "n": n_slot_samples,
                "baseline_type": "binary",
            }
            entity_data = {
                f"{h}_{d}": slot,
                "_binary_meta": {"avg_duration_on": 1.0},
                "_meta": {"sensor_type": "binary", "first_seen": first_seen, "n_samples": 100},
            }
        else:
            # Absolute sensor with a known mean/std
            slot = {
                "mean": 100.0,
                "std": 10.0,
                "n": n_slot_samples,
                "baseline_type": "absolute",
            }
            entity_data = {
                f"{h}_{d}": slot,
                "_meta": {
                    "sensor_type": "gauge",
                    "first_seen": first_seen,
                    "n_samples": 100,
                },
            }

        baselines = {"sensor.test_entity": entity_data}
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

    def test_first_seen_stored_in_meta(self, sample_df, tmp_data_dir):
        """build_entity_baselines must store first_seen in each entity's _meta."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        for eid, entity in data.items():
            if eid.startswith("_"):
                continue
            assert "_meta" in entity
            assert "first_seen" in entity["_meta"], f"No first_seen in _meta for {eid}"
            # Verify it parses as a valid ISO datetime
            datetime.datetime.fromisoformat(entity["_meta"]["first_seen"])

    def test_n_samples_stored_in_meta(self, sample_df, tmp_data_dir):
        """build_entity_baselines must store n_samples in each entity's _meta."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        for eid, entity in data.items():
            if eid.startswith("_"):
                continue
            assert "n_samples" in entity["_meta"], f"No n_samples in _meta for {eid}"
            assert isinstance(entity["_meta"]["n_samples"], int)
            assert entity["_meta"]["n_samples"] > 0

    def test_first_seen_preserved_across_retrain(self, sample_df, tmp_data_dir):
        """Re-running build_entity_baselines must not overwrite an earlier first_seen."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data1 = json.load(f)
        # Inject an earlier first_seen into one entity
        eid = next(k for k in data1 if not k.startswith("_"))
        early_ts = "2000-01-01T00:00:00"
        data1[eid]["_meta"]["first_seen"] = early_ts
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(data1, f)
        # Re-run — should preserve the earlier timestamp
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data2 = json.load(f)
        assert data2[eid]["_meta"]["first_seen"] == early_ts, (
            "first_seen was overwritten on retrain"
        )

    def test_entity_in_learning_not_scored(self, tmp_data_dir):
        """Entity younger than COLD_START_DAYS must not appear in score_entities result."""
        import habitus.habitus.anomaly_breakdown as ab

        # Entity is only 1 day old — should be in learning phase
        self._write_baselines_with_age(tmp_data_dir, days_old=1.0)
        result = score_entities({"sensor.test_entity": 999999.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.test_entity" not in ids

    def test_entity_past_learning_is_scored(self, tmp_data_dir):
        """Entity older than COLD_START_DAYS must be scored normally."""
        import habitus.habitus.anomaly_breakdown as ab

        # Entity is 10 days old — cold start period over
        self._write_baselines_with_age(tmp_data_dir, days_old=10.0)
        result = score_entities({"sensor.test_entity": 999999.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.test_entity" in ids

    def test_learning_entities_written_to_baselines(self, tmp_data_dir):
        """score_entities must write _learning_entities to entity_baselines.json."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_baselines_with_age(tmp_data_dir, days_old=1.0)
        score_entities({"sensor.test_entity": 50.0})
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "_learning_entities" in data
        entities = data["_learning_entities"]
        assert isinstance(entities, list)
        assert any(e["entity_id"] == "sensor.test_entity" for e in entities)

    def test_learning_entity_has_status_learning(self, tmp_data_dir):
        """Entities in cold start must appear with status='learning' in _learning_entities."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_baselines_with_age(tmp_data_dir, days_old=2.0)
        score_entities({"sensor.test_entity": 50.0})
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        learning = data.get("_learning_entities", [])
        test_entry = next(
            (e for e in learning if e["entity_id"] == "sensor.test_entity"), None
        )
        assert test_entry is not None
        assert test_entry["status"] == "learning"
        assert "first_seen" in test_entry
        assert "days_old" in test_entry

    def test_low_sample_slot_halves_zscore(self, tmp_data_dir):
        """Slot with n < LOW_SAMPLE_THRESHOLD must produce a halved z-score."""
        import habitus.habitus.anomaly_breakdown as ab

        # Create two identical baselines: one with n=5 (low), one with n=30 (normal)
        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        old_first_seen = (now - datetime.timedelta(days=30)).isoformat()

        def make_baselines(n_samples: int) -> dict:
            return {
                "sensor.test_entity": {
                    f"{h}_{d}": {
                        "mean": 100.0,
                        "std": 10.0,
                        "n": n_samples,
                        "baseline_type": "absolute",
                    },
                    "_meta": {
                        "sensor_type": "gauge",
                        "first_seen": old_first_seen,
                        "n_samples": n_samples * 7,
                    },
                }
            }

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")

        # Score with normal (high) sample count
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(make_baselines(30), f)
        result_normal = score_entities({"sensor.test_entity": 200.0})

        # Score with low sample count
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(make_baselines(5), f)
        result_low = score_entities({"sensor.test_entity": 200.0})

        assert result_normal, "Expected anomaly with normal sample count"
        assert result_low, "Expected anomaly with low sample count (just halved)"
        z_normal = result_normal[0]["z_score"]
        z_low = result_low[0]["z_score"]
        assert z_low == pytest.approx(z_normal * 0.5, rel=1e-3), (
            f"Low-sample z={z_low} should be half of normal z={z_normal}"
        )

    def test_new_entity_not_in_baselines_not_scored(self, tmp_data_dir):
        """Entity in current_states but absent from baselines must not be scored."""
        import habitus.habitus.anomaly_breakdown as ab

        # Write baselines for a different entity
        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        old_first_seen = (now - datetime.timedelta(days=30)).isoformat()
        baselines = {
            "sensor.known_entity": {
                f"{h}_{d}": {"mean": 50.0, "std": 5.0, "n": 30, "baseline_type": "absolute"},
                "_meta": {"sensor_type": "gauge", "first_seen": old_first_seen, "n_samples": 100},
            }
        }
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

        # Pass a brand-new entity not in baselines
        result = score_entities({"sensor.brand_new_entity": 9999.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.brand_new_entity" not in ids

    def test_new_entity_written_to_lifecycle(self, tmp_data_dir):
        """Entity in current_states but absent from baselines must be in entity_lifecycle.json."""
        import habitus.habitus.anomaly_breakdown as ab

        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        old_first_seen = (now - datetime.timedelta(days=30)).isoformat()
        baselines = {
            "sensor.known_entity": {
                f"{h}_{d}": {"mean": 50.0, "std": 5.0, "n": 30, "baseline_type": "absolute"},
                "_meta": {"sensor_type": "gauge", "first_seen": old_first_seen, "n_samples": 100},
            }
        }
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

        score_entities({"sensor.brand_new_entity": 9999.0})
        assert (tmp_data_dir / "entity_lifecycle.json").exists()
        with open(tmp_data_dir / "entity_lifecycle.json") as f:
            lifecycle = json.load(f)
        assert "sensor.brand_new_entity" in lifecycle
        entry = lifecycle["sensor.brand_new_entity"]
        assert entry["status"] == "new"
        assert "first_seen" in entry
        assert "last_seen" in entry

    def test_compute_breakdown_includes_learning_entities(self, tmp_data_dir):
        """compute_breakdown must include learning_entities in _z_score_run."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_baselines_with_age(tmp_data_dir, days_old=1.0)
        compute_breakdown(80.0, {"sensor.test_entity": 9999.0})
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        assert "_z_score_run" in data
        run = data["_z_score_run"]
        assert "learning_entities" in run
        assert isinstance(run["learning_entities"], list)

    def test_runtime_state_preserved_across_retrain(self, sample_df, tmp_data_dir):
        """build_entity_baselines must carry over _binary_state and _z_score_run."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(sample_df)
        # Inject fake state keys into the baselines file
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data = json.load(f)
        data["_binary_state"] = {"binary_sensor.front_door": {"current_value": 0.0}}
        data["_z_score_run"] = {"timestamp": "2026-01-01T00:00:00", "anomaly_score": 55, "top5": []}
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(data, f)
        # Re-run build — state keys should be preserved
        build_entity_baselines(sample_df)
        with open(tmp_data_dir / "entity_baselines.json") as f:
            data2 = json.load(f)
        assert "_binary_state" in data2
        assert "_z_score_run" in data2


class TestConfidenceWeighting:
    """Tests for TASK-019: Confidence-Weighted Anomaly Score Aggregation."""

    # ── compute_entity_confidence ───────────────────────────────────────────

    def test_full_confidence_at_30_days_20_samples_gauge(self):
        """30 days + 20 slot samples + gauge type → confidence = 1.0."""
        conf = compute_entity_confidence(30.0, 20, "gauge")
        assert conf == pytest.approx(1.0)

    def test_age_factor_caps_at_1(self):
        """More than 30 days of data still caps age factor at 1.0."""
        conf_30 = compute_entity_confidence(30.0, 20, "gauge")
        conf_90 = compute_entity_confidence(90.0, 20, "gauge")
        assert conf_30 == pytest.approx(conf_90)

    def test_sample_factor_caps_at_1(self):
        """More than 20 slot samples still caps sample factor at 1.0."""
        conf_20 = compute_entity_confidence(30.0, 20, "gauge")
        conf_100 = compute_entity_confidence(30.0, 100, "gauge")
        assert conf_20 == pytest.approx(conf_100)

    def test_low_days_reduces_confidence(self):
        """7 days of data (age_factor=7/30) reduces confidence proportionally."""
        conf = compute_entity_confidence(7.0, 20, "gauge")
        assert conf == pytest.approx(7.0 / 30.0 * 1.0 * 1.0, rel=1e-4)

    def test_low_samples_reduces_confidence(self):
        """5 slot samples (sample_factor=5/20) reduces confidence proportionally."""
        conf = compute_entity_confidence(30.0, 5, "gauge")
        assert conf == pytest.approx(1.0 * (5.0 / 20.0) * 1.0, rel=1e-4)

    def test_sensor_type_certainty_applied(self):
        """Binary sensor certainty (0.8) is applied to the confidence formula."""
        conf_gauge = compute_entity_confidence(30.0, 20, "gauge")
        conf_binary = compute_entity_confidence(30.0, 20, "binary")
        assert conf_gauge == pytest.approx(1.0)
        assert conf_binary == pytest.approx(0.8)

    def test_unknown_sensor_type_defaults_to_0_7(self):
        """Unknown sensor type defaults to 0.7 certainty."""
        conf = compute_entity_confidence(30.0, 20, "unknown_type")
        assert conf == pytest.approx(0.7)

    def test_zero_days_gives_zero_confidence(self):
        """Zero days of data → zero confidence regardless of other factors."""
        conf = compute_entity_confidence(0.0, 20, "gauge")
        assert conf == pytest.approx(0.0)

    def test_zero_samples_gives_zero_confidence(self):
        """Zero slot samples → zero confidence."""
        conf = compute_entity_confidence(30.0, 0, "gauge")
        assert conf == pytest.approx(0.0)

    def test_all_sensor_types_in_certainty_map(self):
        """All five sensor types must have a certainty entry."""
        for stype in ("gauge", "accumulating", "binary", "setpoint", "event"):
            assert stype in SENSOR_TYPE_CERTAINTY
            assert 0.0 < SENSOR_TYPE_CERTAINTY[stype] <= 1.0

    # ── compute_weighted_score ──────────────────────────────────────────────

    def test_empty_list_returns_zero(self):
        """compute_weighted_score on empty list must return 0.0."""
        assert compute_weighted_score([]) == pytest.approx(0.0)

    def test_all_below_min_confidence_returns_zero(self):
        """All entities below MIN_CONFIDENCE threshold → weighted score = 0.0."""
        anomalies = [
            {"z_score": 5.0, "confidence": 0.05},
            {"z_score": 3.0, "confidence": 0.0},
        ]
        assert compute_weighted_score(anomalies) == pytest.approx(0.0)

    def test_weighted_average_math(self):
        """Verify: Σ(z×conf) / Σ(conf) for two entities above threshold."""
        anomalies = [
            {"z_score": 4.0, "confidence": 0.5},
            {"z_score": 2.0, "confidence": 1.0},
        ]
        # Expected: (4.0*0.5 + 2.0*1.0) / (0.5 + 1.0) = (2.0+2.0)/1.5 = 2.667
        expected = (4.0 * 0.5 + 2.0 * 1.0) / (0.5 + 1.0)
        assert compute_weighted_score(anomalies) == pytest.approx(expected, rel=1e-3)

    def test_low_confidence_entity_excluded_from_numerator(self):
        """Entity below MIN_CONFIDENCE must not contribute to weighted average."""
        # One high-confidence entity at z=1.0, one below-threshold at z=99.0
        anomalies = [
            {"z_score": 1.0, "confidence": 1.0},
            {"z_score": 99.0, "confidence": MIN_CONFIDENCE / 2},
        ]
        result = compute_weighted_score(anomalies)
        # Only the high-confidence entity contributes → weighted score = 1.0
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_single_entity_returns_its_zscore(self):
        """Single eligible entity → weighted score equals its z_score."""
        anomalies = [{"z_score": 3.5, "confidence": 0.9}]
        assert compute_weighted_score(anomalies) == pytest.approx(3.5, rel=1e-3)

    def test_higher_confidence_entity_dominates(self):
        """Higher-confidence entity should pull weighted score toward its z_score."""
        # High-conf (0.9) at z=2.0 vs low-conf (0.2) at z=8.0
        anomalies = [
            {"z_score": 2.0, "confidence": 0.9},
            {"z_score": 8.0, "confidence": 0.2},
        ]
        result = compute_weighted_score(anomalies)
        # Weighted avg: (2.0*0.9 + 8.0*0.2) / (0.9+0.2) = (1.8+1.6)/1.1 ≈ 3.09
        expected = (2.0 * 0.9 + 8.0 * 0.2) / (0.9 + 0.2)
        assert result == pytest.approx(expected, rel=1e-3)
        # Must be closer to 2.0 than to 8.0
        assert abs(result - 2.0) < abs(result - 8.0)

    # ── Integration: confidence in score_entities() output ─────────────────

    def _write_scored_baselines(
        self,
        tmp_data_dir,
        days_old: float = 30.0,
        slot_n: int = 20,
        sensor_type: str = "gauge",
    ) -> None:
        """Write a minimal baselines file with a known entity old enough to score."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")

        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        first_seen = (now - datetime.timedelta(days=days_old)).isoformat()
        baselines = {
            "sensor.conf_test": {
                f"{h}_{d}": {
                    "mean": 100.0,
                    "std": 10.0,
                    "n": slot_n,
                    "baseline_type": "absolute",
                },
                "_meta": {
                    "sensor_type": sensor_type,
                    "first_seen": first_seen,
                    "n_samples": slot_n * 7,
                },
            }
        }
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

    def test_anomaly_result_has_confidence_field(self, tmp_data_dir):
        """score_entities() must add 'confidence' to each anomaly dict."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_scored_baselines(tmp_data_dir)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        result = score_entities({"sensor.conf_test": 999999.0})
        assert result, "Expected an anomaly for extreme value"
        assert "confidence" in result[0], "Missing 'confidence' field in anomaly"
        assert 0.0 <= result[0]["confidence"] <= 1.0

    def test_anomaly_result_has_confidence_label(self, tmp_data_dir):
        """score_entities() must add 'confidence_label' to each anomaly dict."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_scored_baselines(tmp_data_dir)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        result = score_entities({"sensor.conf_test": 999999.0})
        assert result
        label = result[0].get("confidence_label", "")
        assert "confident" in label
        assert "day" in label

    def test_confidence_label_contains_day_count(self, tmp_data_dir):
        """confidence_label must mention the number of days of data."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_scored_baselines(tmp_data_dir, days_old=15.0)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        result = score_entities({"sensor.conf_test": 999999.0})
        assert result
        label = result[0]["confidence_label"]
        # 15 days → label should start with "50% confident — 15 days of data"
        assert "15" in label

    def test_entity_anomalies_json_has_weighted_score(self, tmp_data_dir):
        """score_entities() must write 'weighted_score' to entity_anomalies.json."""
        import habitus.habitus.anomaly_breakdown as ab

        self._write_scored_baselines(tmp_data_dir)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        score_entities({"sensor.conf_test": 999999.0})
        with open(tmp_data_dir / "entity_anomalies.json") as f:
            data = json.load(f)
        assert "weighted_score" in data
        assert isinstance(data["weighted_score"], float)

    def test_low_confidence_entity_excluded_from_weighted_score(self, tmp_data_dir):
        """Entity with confidence < MIN_CONFIDENCE must not affect weighted_score."""
        import habitus.habitus.anomaly_breakdown as ab

        # Very few samples → confidence near zero
        self._write_scored_baselines(tmp_data_dir, days_old=30.0, slot_n=1)
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        result = score_entities({"sensor.conf_test": 999999.0})
        if result:
            conf = result[0]["confidence"]
            if conf < MIN_CONFIDENCE:
                with open(tmp_data_dir / "entity_anomalies.json") as f:
                    data = json.load(f)
                # Weighted score should be 0 since sole entity is below threshold
                assert data["weighted_score"] == pytest.approx(0.0)

    def test_higher_confidence_entity_has_more_weight(self, tmp_data_dir):
        """Entity with 30 days of data should have higher confidence than 7 days."""
        conf_30 = compute_entity_confidence(30.0, 20, "gauge")
        conf_7 = compute_entity_confidence(7.0, 20, "gauge")
        assert conf_30 > conf_7

    def test_binary_sensor_has_confidence_field(self, tmp_data_dir):
        """Binary sensor anomalies must also carry 'confidence' and 'confidence_label'."""
        import habitus.habitus.anomaly_breakdown as ab

        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        first_seen = (now - datetime.timedelta(days=30)).isoformat()
        bern_std = float(np.sqrt(0.05 * 0.95))
        baselines = {
            "binary_sensor.conf_test": {
                f"{h}_{d}": {
                    "on_fraction": 0.05,
                    "std": bern_std,
                    "avg_transitions": 0.0,
                    "n": 20,
                    "baseline_type": "binary",
                },
                "_binary_meta": {"avg_duration_on": 1.0},
                "_meta": {"sensor_type": "binary", "first_seen": first_seen, "n_samples": 140},
            }
        }
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        ab.ENTITY_LIFECYCLE_PATH = str(tmp_data_dir / "entity_lifecycle.json")
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)
        result = score_entities({"binary_sensor.conf_test": 1.0})
        assert result, "Expected binary anomaly for normally-off sensor being on"
        assert "confidence" in result[0]
        assert "confidence_label" in result[0]
        assert "confident" in result[0]["confidence_label"]
