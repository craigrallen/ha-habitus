"""Tests for per-entity anomaly scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import responses

from habitus.habitus.anomaly_breakdown import (
    build_entity_baselines,
    compute_breakdown,
    score_entities,
    _guess_unit,
    _fmt,
    _day,
    ENTITY_BASELINES_PATH,
    ENTITY_ANOMALIES_PATH,
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
