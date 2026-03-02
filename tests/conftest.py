"""Shared test fixtures for Habitus."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary DATA_DIR and patch environment."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HA_URL", "http://localhost:8123")
    monkeypatch.setenv("SUPERVISOR_TOKEN", "test-token")
    return tmp_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a realistic 90-day hourly stats DataFrame.

    Covers all behavioural entity categories so tests can exercise
    the full feature extraction pipeline.
    """
    rng = np.random.default_rng(42)
    hours = pd.date_range("2025-01-01", periods=90 * 24, freq="h", tz="UTC")

    rows = []
    entities = {
        # Power
        "sensor.bathroom_lights_electric_consumed_w": ("power", 0, 150),
        "sensor.kitchen_sensor_electric_consumed_w": ("power", 0, 800),
        "sensor.mastervolt_total_load": ("power", 200, 2000),
        # Temperature
        "sensor.living_room_thermostat_air_temperature": ("temp", 18, 23),
        "sensor.bilge_sensor_air_temperature": ("temp", 8, 15),
        # Motion
        "binary_sensor.hallway_motion": ("motion", 0, 1),
        "binary_sensor.kitchen_motion": ("motion", 0, 1),
        # Lights
        "light.living_room_ceiling": ("light", 0, 1),
        "light.kitchen_ceiling": ("light", 0, 1),
        # Presence
        "person.craig": ("presence", 0, 1),
        # Media
        "media_player.living_room_tv": ("media", 0, 1),
        # Door
        "binary_sensor.front_door": ("door", 0, 1),
        # Weather
        "sensor.outdoor_temperature": ("weather", -5, 25),
        # Solar
        "sensor.solar_panel_roof_fore_energy_watts": ("power", 0, 1500),
    }

    for eid, (cat, lo, hi) in entities.items():
        for h in hours:
            hour_of_day = h.hour
            # Simulate realistic daily patterns
            if cat == "power":
                base = lo + (hi - lo) * 0.3
                if 7 <= hour_of_day <= 22:
                    base = lo + (hi - lo) * 0.6
                val = max(0, rng.normal(base, (hi - lo) * 0.1))
            elif cat == "temp":
                val = rng.uniform(lo, hi)
            elif cat in ("motion", "light", "presence", "media", "door"):
                # Higher probability during daytime
                p = 0.6 if 7 <= hour_of_day <= 22 else 0.05
                val = float(rng.random() < p)
            else:
                val = rng.uniform(lo, hi)
            rows.append({"entity_id": eid, "ts": h, "mean": val, "sum": None})

    return pd.DataFrame(rows)


@pytest.fixture
def sample_features(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from sample_df (uses main.build_features)."""
    from habitus.habitus.main import build_features
    return build_features(sample_df)


@pytest.fixture
def mock_ha_states() -> list[dict]:
    """Realistic HA /api/states response."""
    return [
        {"entity_id": "binary_sensor.hallway_motion", "state": "on"},
        {"entity_id": "binary_sensor.kitchen_motion", "state": "off"},
        {"entity_id": "light.living_room_ceiling", "state": "on"},
        {"entity_id": "light.kitchen_ceiling", "state": "on"},
        {"entity_id": "person.craig", "state": "home"},
        {"entity_id": "media_player.living_room_tv", "state": "playing"},
        {"entity_id": "binary_sensor.front_door", "state": "off"},
        {"entity_id": "sensor.outdoor_temperature", "state": "4.5"},
        {"entity_id": "sensor.mastervolt_total_load", "state": "845.0"},
    ]
