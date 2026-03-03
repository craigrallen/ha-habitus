"""Tests for TASK-021: Impossible Value & Data Quality Guard."""
from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habitus.habitus.anomaly_breakdown import (
    apply_data_quality_filters,
    _persist_data_quality,
    _is_power_entity,
    _is_temperature_entity,
    _is_humidity_entity,
    build_entity_baselines,
    score_entities,
    DATA_QUALITY_PATH,
)


def _make_df(entity_id: str, values: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    """Build a minimal DataFrame with entity_id, ts, v columns."""
    hours = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({"entity_id": entity_id, "ts": hours, "v": values})


# ── Entity type helpers ─────────────────────────────────────────────────────────


class TestEntityTypeHelpers:
    def test_power_entity_by_w_suffix(self):
        assert _is_power_entity("sensor.bathroom_lights_electric_consumed_w") is True

    def test_power_entity_by_watt(self):
        assert _is_power_entity("sensor.device_watt_usage") is True

    def test_power_entity_by_power(self):
        assert _is_power_entity("sensor.total_power") is True

    def test_non_power_entity(self):
        assert _is_power_entity("sensor.outdoor_temperature") is False

    def test_temperature_entity(self):
        assert _is_temperature_entity("sensor.living_room_temperature") is True

    def test_temperature_temp_suffix(self):
        assert _is_temperature_entity("sensor.bilge_temp") is True

    def test_non_temperature_entity(self):
        assert _is_temperature_entity("sensor.battery_voltage") is False

    def test_humidity_entity(self):
        assert _is_humidity_entity("sensor.bathroom_humidity") is True

    def test_non_humidity_entity(self):
        assert _is_humidity_entity("sensor.total_power") is False


# ── Negative power clamping ─────────────────────────────────────────────────────


class TestNegativePowerClamping:
    def test_negative_values_clamped_to_zero(self):
        """Negative power values must be clamped to 0, not discarded."""
        df = _make_df("sensor.device_power", [-50.0, -10.0, 100.0, -5.0, 200.0])
        result, issues = apply_data_quality_filters(df)
        power_vals = result.loc[result["entity_id"] == "sensor.device_power", "v"].values
        assert (power_vals >= 0).all(), "All power values must be ≥ 0 after clamping"

    def test_negative_power_issue_reported(self):
        """Clamping negative power must produce a 'negative_power' quality issue."""
        df = _make_df("sensor.device_power", [-50.0, 100.0])
        _, issues = apply_data_quality_filters(df)
        assert any(i["issue"] == "negative_power" for i in issues)

    def test_negative_power_issue_has_required_fields(self):
        df = _make_df("sensor.device_w", [-1.0, 5.0, 10.0])
        _, issues = apply_data_quality_filters(df)
        issue = next(i for i in issues if i["issue"] == "negative_power")
        for field in ("entity_id", "issue", "since", "last_valid"):
            assert field in issue

    def test_positive_power_not_modified(self):
        """Positive power values must not be altered."""
        df = _make_df("sensor.device_w", [100.0, 200.0, 150.0])
        result, issues = apply_data_quality_filters(df)
        assert list(result["v"]) == [100.0, 200.0, 150.0]
        assert not any(i["issue"] == "negative_power" for i in issues)

    def test_clamping_preserves_row_count(self):
        """Clamping should keep all rows — only values change."""
        df = _make_df("sensor.device_power", [-10.0, 50.0, -5.0, 100.0])
        result, _ = apply_data_quality_filters(df)
        assert len(result) == 4


# ── Temperature out-of-range discarding ────────────────────────────────────────


class TestTemperatureFilter:
    def test_temperature_above_85_discarded(self):
        """Temperature >85°C must be discarded (row removed)."""
        df = _make_df("sensor.room_temperature", [20.0, 90.0, 22.0])
        result, issues = apply_data_quality_filters(df)
        assert 90.0 not in result["v"].values
        assert len(result) == 2  # one row removed

    def test_temperature_below_minus_60_discarded(self):
        """Temperature <-60°C must be discarded."""
        df = _make_df("sensor.outdoor_temperature", [-70.0, 15.0, 18.0])
        result, _ = apply_data_quality_filters(df)
        assert -70.0 not in result["v"].values

    def test_valid_temperature_not_discarded(self):
        """Normal temperatures must not be removed."""
        df = _make_df("sensor.room_temperature", [-5.0, 20.0, 80.0])
        result, issues = apply_data_quality_filters(df)
        assert len(result) == 3
        assert not any(i["issue"] == "temperature_out_of_range" for i in issues)

    def test_temperature_issue_reported(self):
        """Out-of-range temperature must produce a 'temperature_out_of_range' issue."""
        df = _make_df("sensor.living_room_temperature", [200.0, 22.0])
        _, issues = apply_data_quality_filters(df)
        assert any(i["issue"] == "temperature_out_of_range" for i in issues)

    def test_temperature_issue_has_required_fields(self):
        df = _make_df("sensor.bilge_temp", [99.0, 22.0])
        _, issues = apply_data_quality_filters(df)
        issue = next(i for i in issues if i["issue"] == "temperature_out_of_range")
        for field in ("entity_id", "issue", "since", "last_valid"):
            assert field in issue


# ── Humidity clamping ───────────────────────────────────────────────────────────


class TestHumidityFilter:
    def test_humidity_below_0_clamped(self):
        """Humidity < 0% must be clamped to 0."""
        df = _make_df("sensor.bathroom_humidity", [-5.0, 50.0, 60.0])
        result, _ = apply_data_quality_filters(df)
        assert result["v"].min() >= 0.0

    def test_humidity_above_100_clamped(self):
        """Humidity > 100% must be clamped to 100."""
        df = _make_df("sensor.kitchen_humidity", [50.0, 110.0, 60.0])
        result, _ = apply_data_quality_filters(df)
        assert result["v"].max() <= 100.0

    def test_humidity_clamp_preserves_row_count(self):
        """Clamping humidity must not remove rows."""
        df = _make_df("sensor.basement_humidity", [-10.0, 50.0, 105.0])
        result, _ = apply_data_quality_filters(df)
        assert len(result) == 3

    def test_humidity_issue_reported(self):
        """Out-of-range humidity must produce a 'humidity_clamped' issue."""
        df = _make_df("sensor.bathroom_humidity", [150.0, 50.0])
        _, issues = apply_data_quality_filters(df)
        assert any(i["issue"] == "humidity_clamped" for i in issues)

    def test_valid_humidity_unmodified(self):
        """Humidity in [0, 100] must not be changed or flagged."""
        df = _make_df("sensor.room_humidity", [0.0, 50.0, 100.0])
        result, issues = apply_data_quality_filters(df)
        assert list(result["v"]) == [0.0, 50.0, 100.0]
        assert not any(i["issue"] == "humidity_clamped" for i in issues)


# ── Value jump >10× discarding ──────────────────────────────────────────────────


class TestValueJumpFilter:
    def test_10x_jump_discarded(self):
        """A single value that is >10× the previous reading must be removed."""
        # 5.0 → 1000.0 is a 200× jump; 1000.0 must be discarded
        vals = [5.0, 5.1, 5.0, 1000.0, 5.2, 5.0] * 5  # 30 readings to be gauge-like
        df = _make_df("sensor.living_room_temperature", vals)
        result, issues = apply_data_quality_filters(df)
        assert 1000.0 not in result["v"].values

    def test_10x_jump_issue_reported(self):
        """A value-jump discard must produce a 'value_jump' quality issue."""
        # Use varied values so classify_sensor sees >2 unique values (gauge heuristic).
        # 10.0–10.9 baseline with a 500.0 spike — 500/10.9 ≈ 45× > 10×
        varied = [10.0 + i * 0.1 for i in range(10)]  # 10 unique-ish values
        spike = [500.0]
        tail = [10.0 + i * 0.1 for i in range(20)]  # 20 more varied values
        vals = varied + spike + tail
        df = _make_df("sensor.living_room_gauge", vals)
        _, issues = apply_data_quality_filters(df)
        assert any(i["issue"] == "value_jump" for i in issues)

    def test_normal_variation_not_flagged(self):
        """Values varying by <10× must not trigger jump detection."""
        # 10.0 → 50.0 is a 5× increase — within the 10× threshold
        vals = [10.0, 50.0, 20.0, 30.0, 25.0] * 6  # 30 readings
        df = _make_df("sensor.room_temperature", vals)
        result, issues = apply_data_quality_filters(df)
        assert not any(i["issue"] == "value_jump" for i in issues)

    def test_jump_from_near_zero_not_flagged(self):
        """Jumps from values ≤0.5 must not be flagged (near-zero noise guard)."""
        # 0.0 → 100.0: previous is 0, so not significant — must not be flagged
        vals = [0.0, 100.0, 0.2, 99.0, 0.1, 98.0] * 5  # 30 readings
        df = _make_df("sensor.device_power", vals)
        _, issues = apply_data_quality_filters(df)
        assert not any(i["issue"] == "value_jump" for i in issues)


# ── Stuck sensor detection ─────────────────────────────────────────────────────


class TestStuckSensorDetection:
    def _make_stuck_df(self, entity_id: str, stuck_value: float = 42.0) -> pd.DataFrame:
        """25 readings of the same value (with variation before) for stuck sensor."""
        varying = list(np.linspace(10.0, 50.0, 5))  # 5 varied readings
        stuck = [stuck_value] * 25
        vals = varying + stuck
        return _make_df(entity_id, vals)

    def test_stuck_gauge_detected(self):
        """Gauge sensor with ≥24 identical readings must be flagged as stuck."""
        df = self._make_stuck_df("sensor.bathroom_temperature")
        _, issues = apply_data_quality_filters(df)
        assert any(i["issue"] == "stuck" for i in issues)

    def test_stuck_issue_entity_id_correct(self):
        eid = "sensor.living_room_temperature"
        df = self._make_stuck_df(eid)
        _, issues = apply_data_quality_filters(df)
        stuck = next(i for i in issues if i["issue"] == "stuck")
        assert stuck["entity_id"] == eid

    def test_stuck_issue_has_since_and_last_valid(self):
        df = self._make_stuck_df("sensor.room_temperature")
        _, issues = apply_data_quality_filters(df)
        stuck = next(i for i in issues if i["issue"] == "stuck")
        assert "since" in stuck
        assert "last_valid" in stuck

    def test_non_stuck_sensor_not_flagged(self):
        """Sensor with varying values must not be flagged as stuck."""
        varying = list(np.sin(np.linspace(0, 10, 30)) * 20 + 50)  # 30 varied readings
        df = _make_df("sensor.room_temperature", varying)
        _, issues = apply_data_quality_filters(df)
        assert not any(i["issue"] == "stuck" for i in issues)

    def test_fewer_than_24_readings_not_flagged(self):
        """Sensor with fewer than 24 readings must not trigger stuck detection."""
        vals = [42.0] * 10  # only 10 same values → not enough for 24h window
        df = _make_df("sensor.test_temperature", vals)
        _, issues = apply_data_quality_filters(df)
        assert not any(i["issue"] == "stuck" for i in issues)


# ── _persist_data_quality ──────────────────────────────────────────────────────


class TestPersistDataQuality:
    def test_writes_json_file(self, tmp_data_dir):
        """_persist_data_quality must create data_quality.json."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        issues = [
            {"entity_id": "sensor.foo", "issue": "stuck", "since": "2025-01-01", "last_valid": ""}
        ]
        _persist_data_quality(issues)
        assert (tmp_data_dir / "data_quality.json").exists()

    def test_written_content_is_list(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        issues = [
            {"entity_id": "sensor.bar", "issue": "negative_power", "since": "2025-01-01", "last_valid": "2024-12-31"}
        ]
        _persist_data_quality(issues)
        with open(tmp_data_dir / "data_quality.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["entity_id"] == "sensor.bar"

    def test_new_issue_overwrites_old_for_same_entity(self, tmp_data_dir):
        """Calling _persist_data_quality twice for the same entity must overwrite."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        _persist_data_quality(
            [{"entity_id": "sensor.x", "issue": "stuck", "since": "old", "last_valid": ""}]
        )
        _persist_data_quality(
            [{"entity_id": "sensor.x", "issue": "value_jump", "since": "new", "last_valid": ""}]
        )
        with open(tmp_data_dir / "data_quality.json") as f:
            data = json.load(f)
        entries = [d for d in data if d["entity_id"] == "sensor.x"]
        assert len(entries) == 1
        assert entries[0]["issue"] == "value_jump"

    def test_multiple_entities_all_written(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        issues = [
            {"entity_id": "sensor.a", "issue": "stuck", "since": "t1", "last_valid": ""},
            {"entity_id": "sensor.b", "issue": "negative_power", "since": "t2", "last_valid": ""},
        ]
        _persist_data_quality(issues)
        with open(tmp_data_dir / "data_quality.json") as f:
            data = json.load(f)
        ids = {d["entity_id"] for d in data}
        assert "sensor.a" in ids
        assert "sensor.b" in ids


# ── Integration: build_entity_baselines calls filters ─────────────────────────


class TestBuildBaselinesAppliesFilters:
    def test_negative_power_removed_from_baseline(self, tmp_data_dir):
        """build_entity_baselines must clamp negative power before building slots."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")

        hours = pd.date_range("2025-01-01", periods=90 * 24, freq="h", tz="UTC")
        rows = []
        for i, h in enumerate(hours):
            v = -100.0 if i % 10 == 0 else float(100 + i % 5)
            rows.append({"entity_id": "sensor.device_w", "ts": h, "mean": v, "sum": None})
        df = pd.DataFrame(rows)
        build_entity_baselines(df)

        with open(tmp_data_dir / "entity_baselines.json") as f:
            bl = json.load(f)
        # The baseline mean must be ≥ 0 (clamped negatives don't lower it below 0)
        for key, slot in bl.get("sensor.device_w", {}).items():
            if key.startswith("_") or not isinstance(slot, dict):
                continue
            assert slot.get("mean", 0.0) >= 0.0

    def test_data_quality_file_created_on_filter_hit(self, tmp_data_dir):
        """When filters find issues, data_quality.json must be created."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")

        hours = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
        rows = [
            {"entity_id": "sensor.room_temperature", "ts": h, "mean": 200.0, "sum": None}
            for h in hours
        ]
        build_entity_baselines(pd.DataFrame(rows))
        # data_quality.json should exist since 200°C is out of range
        assert (tmp_data_dir / "data_quality.json").exists()


# ── Integration: score_entities skips stuck sensors ───────────────────────────


class TestScoreEntitiesSkipsStuck:
    def _write_stuck_quality(self, tmp_data_dir: Path, entity_id: str) -> None:
        """Write a stuck issue for the given entity to data_quality.json."""
        issues = [{"entity_id": entity_id, "issue": "stuck", "since": "2025-01-01", "last_valid": "2024-12-31"}]
        with open(tmp_data_dir / "data_quality.json", "w") as f:
            json.dump(issues, f)

    def _write_baselines(self, tmp_data_dir: Path, entity_id: str) -> None:
        """Write a minimal absolute baseline for the given entity."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")

        now = datetime.datetime.now()
        h, d = now.hour, now.weekday()
        old_first_seen = (now - datetime.timedelta(days=30)).isoformat()
        baselines = {
            entity_id: {
                f"{h}_{d}": {
                    "mean": 100.0,
                    "std": 10.0,
                    "n": 30,
                    "baseline_type": "absolute",
                },
                "_meta": {"sensor_type": "gauge", "first_seen": old_first_seen, "n_samples": 200},
            }
        }
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

    def test_stuck_entity_excluded_from_scoring(self, tmp_data_dir):
        """Entity flagged as stuck must not appear in score_entities output."""
        eid = "sensor.test_gauge"
        self._write_baselines(tmp_data_dir, eid)
        self._write_stuck_quality(tmp_data_dir, eid)

        result = score_entities({eid: 999999.0})
        ids = [r["entity_id"] for r in result]
        assert eid not in ids, "Stuck entity must be excluded from scoring"

    def test_non_stuck_entity_still_scored(self, tmp_data_dir):
        """An entity NOT in data_quality.json as stuck must still be scored normally."""
        eid = "sensor.healthy_gauge"
        self._write_baselines(tmp_data_dir, eid)
        # Write data_quality.json but with a different entity flagged
        with open(tmp_data_dir / "data_quality.json", "w") as f:
            json.dump(
                [{"entity_id": "sensor.other_entity", "issue": "stuck", "since": "", "last_valid": ""}],
                f,
            )
        result = score_entities({eid: 999999.0})
        ids = [r["entity_id"] for r in result]
        assert eid in ids

    def test_no_data_quality_file_does_not_crash(self, tmp_data_dir):
        """score_entities must work even when data_quality.json doesn't exist."""
        eid = "sensor.test_entity"
        self._write_baselines(tmp_data_dir, eid)
        # Ensure data_quality.json doesn't exist
        dqp = tmp_data_dir / "data_quality.json"
        if dqp.exists():
            dqp.unlink()
        result = score_entities({eid: 999999.0})
        assert isinstance(result, list)
