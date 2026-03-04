from __future__ import annotations

import pandas as pd
import pytest

from habitus.habitus.main import build_features


def test_build_features_leak_sensor_uses_mean_when_state_absent() -> None:
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-01T01:00:00Z")

    df = pd.DataFrame(
        [
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts0, "mean": 500.0, "sum": None},
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts1, "mean": 450.0, "sum": None},
            {"entity_id": "binary_sensor.water_leak_engine_room", "ts": ts0, "mean": 1.0, "sum": None},
            {"entity_id": "binary_sensor.water_leak_engine_room", "ts": ts1, "mean": 0.0, "sum": None},
        ]
    )

    features = build_features(df)

    per_hour = dict(zip(features["hour"], features["water_leak"], strict=False))
    assert per_hour[ts0] == 1.0
    assert per_hour[ts1] == 0.0


def test_build_features_grid_energy_populates_grid_kwh_w(monkeypatch) -> None:
    monkeypatch.setenv("HABITUS_ENERGY_GRID", "sensor.grid_energy")

    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-01T01:00:00Z")

    df = pd.DataFrame(
        [
            {"entity_id": "sensor.grid_energy", "ts": ts0, "mean": 100.0, "sum": None},
            {"entity_id": "sensor.grid_energy", "ts": ts1, "mean": 101.5, "sum": None},
        ]
    )

    features = build_features(df)
    per_hour_grid = dict(zip(features["hour"], features["grid_kwh_w"], strict=False))
    per_hour_power = dict(zip(features["hour"], features["total_power_w"], strict=False))

    assert per_hour_grid[ts0] == 0.0
    assert per_hour_grid[ts1] == 1500.0
    assert per_hour_power[ts1] == 1500.0


def test_build_features_gas_entities_compute_hourly_delta(monkeypatch) -> None:
    monkeypatch.setenv("HABITUS_GAS_ENTITIES", "sensor.gas_meter_total")

    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-01T01:00:00Z")

    df = pd.DataFrame(
        [
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts0, "mean": 500.0, "sum": None},
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts1, "mean": 450.0, "sum": None},
            {"entity_id": "sensor.gas_meter_total", "ts": ts0, "mean": 100.0, "sum": None},
            {"entity_id": "sensor.gas_meter_total", "ts": ts1, "mean": 101.2, "sum": None},
        ]
    )

    features = build_features(df)

    per_hour = dict(zip(features["hour"], features["gas_m3_per_h"], strict=False))
    assert per_hour[ts0] == 0.0
    assert per_hour[ts1] == pytest.approx(1.2)


def test_build_features_invalid_max_power_env_uses_default(monkeypatch) -> None:
    monkeypatch.setenv("HABITUS_POWER_ENTITY", "sensor.house_power")
    monkeypatch.setenv("HABITUS_MAX_POWER_KW", "not-a-number")

    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-01T01:00:00Z")
    df = pd.DataFrame(
        [
            {"entity_id": "sensor.house_power", "ts": ts0, "mean": 1000.0, "sum": None},
            {"entity_id": "sensor.house_power", "ts": ts1, "mean": 50000.0, "sum": None},
        ]
    )

    features = build_features(df)
    per_hour = dict(zip(features["hour"], features["total_power_w"], strict=False))

    assert per_hour[ts0] == 1000.0
    assert per_hour[ts1] == 25000.0


def test_build_features_invalid_max_gas_env_uses_default(monkeypatch) -> None:
    monkeypatch.setenv("HABITUS_GAS_ENTITIES", "sensor.gas_meter_total")
    monkeypatch.setenv("HABITUS_MAX_GAS_M3_H", "oops")

    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-01T01:00:00Z")

    df = pd.DataFrame(
        [
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts0, "mean": 500.0, "sum": None},
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts1, "mean": 450.0, "sum": None},
            {"entity_id": "sensor.gas_meter_total", "ts": ts0, "mean": 100.0, "sum": None},
            {"entity_id": "sensor.gas_meter_total", "ts": ts1, "mean": 180.0, "sum": None},
        ]
    )

    features = build_features(df)
    per_hour = dict(zip(features["hour"], features["gas_m3_per_h"], strict=False))

    assert per_hour[ts0] == 0.0
    assert per_hour[ts1] == 20.0


def test_build_features_leak_sensor_prefers_state_when_available() -> None:
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = pd.Timestamp("2026-01-01T01:00:00Z")

    df = pd.DataFrame(
        [
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts0, "mean": 500.0, "sum": None},
            {"entity_id": "sensor.mastervolt_total_load", "ts": ts1, "mean": 450.0, "sum": None},
            {
                "entity_id": "binary_sensor.bilge_leak_detect",
                "ts": ts0,
                "mean": 0.0,
                "state": "detected",
                "sum": None,
            },
            {
                "entity_id": "binary_sensor.bilge_leak_detect",
                "ts": ts1,
                "mean": 1.0,
                "state": "off",
                "sum": None,
            },
        ]
    )

    features = build_features(df)

    per_hour = dict(zip(features["hour"], features["water_leak"], strict=False))
    assert per_hour[ts0] == 1.0
    assert per_hour[ts1] == 0.0
