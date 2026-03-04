from __future__ import annotations

import pandas as pd

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
