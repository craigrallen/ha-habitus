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
    score_entities,
    _guess_unit,
    _fmt,
    _day,
    ENTITY_BASELINES_PATH,
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
        # Pick an entity and verify slot structure
        entity = next(iter(data.values()))
        slot = next(iter(entity.values()))
        assert "mean" in slot and "std" in slot and "n" in slot

    def test_empty_df_safe(self, tmp_data_dir):
        import habitus.habitus.anomaly_breakdown as ab
        ab.ENTITY_BASELINES_PATH = str(tmp_data_dir / "entity_baselines.json")
        build_entity_baselines(pd.DataFrame())  # Should not raise


class TestGuessUnit:
    def test_watts(self):
        assert _guess_unit("sensor.bathroom_lights_electric_consumed_w") == "W"

    def test_temperature(self):
        assert _guess_unit("sensor.living_room_thermostat_air_temperature") == "°C"

    def test_humidity(self):
        assert _guess_unit("sensor.kitchen_humidity") == "%"

    def test_unknown(self):
        assert _guess_unit("sensor.mystery_value") == ""


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
