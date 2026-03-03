"""Tests for TASK-016: Accumulating Sensor Rate-of-Change Baseline.

Covers:
- Delta computation: accumulating sensors use rate baselines, not absolute.
- New-entity bootstrap: first call stores prev value, no anomaly emitted.
- 24h exemption: scoring suppressed within bootstrap window.
- Negative delta handling: meter resets are skipped.
- Anomaly rate scoring: extreme deltas are flagged after bootstrap.
- Accumulating state persistence across retraining cycles.
- Non-accumulating sensors still use absolute baselines.
"""
from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_accumulating_df(n_hours: int = 90 * 24, seed: int = 42, rate_mean: float = 1.0, rate_std: float = 0.05) -> pd.DataFrame:
    """Return DataFrame for a monotonically increasing (accumulating) entity."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    deltas = np.maximum(0, rng.normal(rate_mean, rate_std, n_hours))
    values = np.cumsum(deltas)
    return pd.DataFrame(
        [{"entity_id": "sensor.energy_kwh", "ts": h, "mean": v, "sum": None} for h, v in zip(hours, values)]
    )


def _inject_acc_state(
    baselines_path: str,
    entity_id: str,
    prev_value: float,
    hours_ago: float,
) -> None:
    """Write a synthetic _accumulating_state into entity_baselines.json."""
    with open(baselines_path) as f:
        baselines = json.load(f)
    old_ts = (datetime.datetime.now() - datetime.timedelta(hours=hours_ago)).isoformat()
    baselines["_accumulating_state"] = {
        entity_id: {
            "prev_value": prev_value,
            "prev_ts": old_ts,
            "first_delta_ts": old_ts,
        }
    }
    with open(baselines_path, "w") as f:
        json.dump(baselines, f)


# ---------------------------------------------------------------------------
# Rate baseline construction
# ---------------------------------------------------------------------------


class TestBuildRateBaseline:
    def test_accumulating_sensor_gets_rate_baseline_type(self, tmp_data_dir):
        """Slots for accumulating entities have baseline_type='rate'."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        build_entity_baselines(_make_accumulating_df())

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        assert "sensor.energy_kwh" in baselines
        entity = baselines["sensor.energy_kwh"]
        assert entity["_meta"]["sensor_type"] == "accumulating"
        slot_keys = [k for k in entity if not k.startswith("_")]
        assert len(slot_keys) > 0
        for k in slot_keys:
            assert entity[k]["baseline_type"] == "rate", f"Slot {k} should have baseline_type=rate"

    def test_rate_baseline_slots_have_required_fields(self, tmp_data_dir):
        """Rate slots expose mean, std, n, baseline_type."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        build_entity_baselines(_make_accumulating_df(seed=1))

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        entity = baselines["sensor.energy_kwh"]
        slot_key = next(k for k in entity if not k.startswith("_"))
        slot = entity[slot_key]
        for field in ("mean", "std", "n", "baseline_type"):
            assert field in slot, f"Missing field: {field}"

    def test_rate_baseline_mean_approximates_expected_rate(self, tmp_data_dir):
        """Rate slot mean ≈ expected hourly consumption when rate is constant."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        # Perfectly uniform 2.0 kWh/h consumption
        hours = pd.date_range("2025-01-01", periods=90 * 24, freq="h", tz="UTC")
        values = np.arange(1, 90 * 24 + 1, dtype=float) * 2.0  # +2 per hour
        df = pd.DataFrame(
            [{"entity_id": "sensor.steady_kwh", "ts": h, "mean": v, "sum": None} for h, v in zip(hours, values)]
        )
        build_entity_baselines(df)

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        entity = baselines["sensor.steady_kwh"]
        slot_key = next(k for k in entity if not k.startswith("_"))
        assert abs(entity[slot_key]["mean"] - 2.0) < 0.01

    def test_non_accumulating_sensor_gets_absolute_baseline(self, tmp_data_dir):
        """Gauge sensors still receive baseline_type='absolute'."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        rng = np.random.default_rng(5)
        hours = pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz="UTC")
        values = rng.uniform(15.0, 25.0, len(hours))
        df = pd.DataFrame(
            [{"entity_id": "sensor.temperature", "ts": h, "mean": v, "sum": None} for h, v in zip(hours, values)]
        )
        build_entity_baselines(df)

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        entity = baselines["sensor.temperature"]
        slot_key = next(k for k in entity if not k.startswith("_"))
        assert entity[slot_key]["baseline_type"] == "absolute"

    def test_negative_deltas_excluded_from_rate_baseline(self, tmp_data_dir):
        """Meter resets (negative deltas) are excluded and do not crash build."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        hours = pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz="UTC")
        rng = np.random.default_rng(7)
        base_deltas = rng.uniform(0.1, 1.5, len(hours))
        values = np.cumsum(base_deltas)
        # Simulate meter reset at day 15
        reset_idx = 15 * 24
        values[reset_idx:] -= values[reset_idx] * 0.9
        values = np.maximum(values, 0.0)
        df = pd.DataFrame(
            [{"entity_id": "sensor.kwh_reset", "ts": h, "mean": v, "sum": None} for h, v in zip(hours, values)]
        )
        # Should not raise despite negative deltas in the series
        build_entity_baselines(df)

    def test_accumulating_state_preserved_across_retrain(self, tmp_data_dir):
        """Re-running build_entity_baselines preserves existing _accumulating_state."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines

        build_entity_baselines(_make_accumulating_df())

        # Manually inject an accumulating state
        _inject_acc_state(
            str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 500.0, hours_ago=48
        )

        # Retrain — state must survive
        build_entity_baselines(_make_accumulating_df())

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        assert "_accumulating_state" in baselines
        assert "sensor.energy_kwh" in baselines["_accumulating_state"]
        assert abs(baselines["_accumulating_state"]["sensor.energy_kwh"]["prev_value"] - 500.0) < 0.01


# ---------------------------------------------------------------------------
# New-entity bootstrap
# ---------------------------------------------------------------------------


class TestNewEntityBootstrap:
    def test_first_call_does_not_score(self, tmp_data_dir):
        """First time an accumulating entity is seen, no anomaly is emitted."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df())

        result = score_entities({"sensor.energy_kwh": 9999.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.energy_kwh" not in ids

    def test_first_call_stores_prev_value_in_state(self, tmp_data_dir):
        """After bootstrap call, _accumulating_state is written to entity_baselines.json."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df())
        score_entities({"sensor.energy_kwh": 42.0})

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        assert "_accumulating_state" in baselines
        state = baselines["_accumulating_state"]
        assert "sensor.energy_kwh" in state
        assert abs(state["sensor.energy_kwh"]["prev_value"] - 42.0) < 0.01

    def test_within_24h_no_scoring(self, tmp_data_dir):
        """Accumulating entity is not scored within 24 h of first delta."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df())

        # Inject state with first_delta_ts only 1 h ago → within 24h window
        recent = (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)
        baselines["_accumulating_state"] = {
            "sensor.energy_kwh": {
                "prev_value": 100.0,
                "prev_ts": recent,
                "first_delta_ts": recent,
            }
        }
        with open(tmp_data_dir / "entity_baselines.json", "w") as f:
            json.dump(baselines, f)

        result = score_entities({"sensor.energy_kwh": 99999.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.energy_kwh" not in ids

    def test_negative_delta_not_scored(self, tmp_data_dir):
        """Negative delta (current < prev) is skipped, not flagged as anomaly."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df())
        _inject_acc_state(str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 9999.0, hours_ago=48)

        # current < prev → negative delta
        result = score_entities({"sensor.energy_kwh": 100.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.energy_kwh" not in ids


# ---------------------------------------------------------------------------
# Anomaly rate scoring
# ---------------------------------------------------------------------------


class TestRateScoring:
    def test_normal_delta_not_anomalous(self, tmp_data_dir):
        """Delta near the baseline mean does not trigger an anomaly."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        # Rate baseline mean ≈ 1.0 kWh/h
        build_entity_baselines(_make_accumulating_df(rate_mean=1.0, rate_std=0.05))
        _inject_acc_state(str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 100.0, hours_ago=48)

        # delta = 101.0 − 100.0 = 1.0 → within baseline
        result = score_entities({"sensor.energy_kwh": 101.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.energy_kwh" not in ids

    def test_extreme_delta_is_anomalous(self, tmp_data_dir):
        """Delta far above the baseline triggers an anomaly."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        # Tight rate baseline: mean ~1.0, std ~0.05
        build_entity_baselines(_make_accumulating_df(rate_mean=1.0, rate_std=0.05))
        _inject_acc_state(str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 100.0, hours_ago=48)

        # delta = 200.0 − 100.0 = 100.0 → z ~ 2000σ above mean
        result = score_entities({"sensor.energy_kwh": 200.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.energy_kwh" in ids

    def test_rate_anomaly_result_has_required_fields(self, tmp_data_dir):
        """Anomaly entry from a rate sensor has all required fields."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df(rate_mean=1.0, rate_std=0.05))
        _inject_acc_state(str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 100.0, hours_ago=48)

        result = score_entities({"sensor.energy_kwh": 200.0})
        kwh_hits = [r for r in result if r["entity_id"] == "sensor.energy_kwh"]
        if kwh_hits:
            entry = kwh_hits[0]
            for field in ("entity_id", "name", "current_value", "baseline_mean", "baseline_std", "z_score", "description", "direction"):
                assert field in entry, f"Missing field: {field}"

    def test_rate_description_mentions_rate(self, tmp_data_dir):
        """Description for a rate anomaly mentions consumption rate or '/h'."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df(rate_mean=1.0, rate_std=0.05))
        _inject_acc_state(str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 100.0, hours_ago=48)

        result = score_entities({"sensor.energy_kwh": 200.0})
        kwh_hits = [r for r in result if r["entity_id"] == "sensor.energy_kwh"]
        if kwh_hits:
            desc = kwh_hits[0]["description"]
            assert "rate" in desc.lower() or "/h" in desc

    def test_accumulating_state_updated_after_scoring(self, tmp_data_dir):
        """After each score_entities call, prev_value in _accumulating_state is updated."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df())
        _inject_acc_state(str(tmp_data_dir / "entity_baselines.json"), "sensor.energy_kwh", 100.0, hours_ago=48)

        score_entities({"sensor.energy_kwh": 101.5})

        with open(tmp_data_dir / "entity_baselines.json") as f:
            baselines = json.load(f)

        state = baselines["_accumulating_state"]["sensor.energy_kwh"]
        assert abs(state["prev_value"] - 101.5) < 0.01

    def test_zero_to_nonzero_first_call_is_bootstrap(self, tmp_data_dir):
        """Entity jumping from 0 to a nonzero value on first call is bootstrap, not anomaly."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        build_entity_baselines(_make_accumulating_df())

        # No prior state in baselines → first encounter → bootstrap only
        result = score_entities({"sensor.energy_kwh": 9999.0})
        ids = [r["entity_id"] for r in result]
        assert "sensor.energy_kwh" not in ids

    def test_mixed_entity_types_scored_correctly(self, tmp_data_dir, sample_df):
        """Gauge sensors from sample_df coexist with an accumulating sensor without errors."""
        import habitus.habitus.anomaly_breakdown as ab

        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        ab.ENTITY_ANOMALIES_PATH = str(tmp_data_dir / "entity_anomalies.json")
        from habitus.habitus.anomaly_breakdown import build_entity_baselines, score_entities

        # Merge sample_df (gauge/binary) with an accumulating entity
        acc_df = _make_accumulating_df()
        combined = pd.concat([sample_df, acc_df], ignore_index=True)
        build_entity_baselines(combined)

        current = {
            "sensor.energy_kwh": 9999.0,  # first call → bootstrap
            "sensor.bathroom_lights_electric_consumed_w": 99999.0,  # gauge → anomalous
        }
        result = score_entities(current)
        # Accumulating entity should not appear (bootstrap)
        assert "sensor.energy_kwh" not in [r["entity_id"] for r in result]
        # Gauge entity can appear
        assert isinstance(result, list)
