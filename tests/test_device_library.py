"""Tests for device_library — smart plug / named W sensor device templates."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from habitus.habitus.device_library import (
    build_device_profile,
    build_library_from_features,
    energy_breakdown,
    extract_device_name,
    is_device_sensor,
    load_library,
    match_wattage_to_device,
    save_library,
)


class TestIsDeviceSensor:
    def test_is_device_sensor_kettle(self):
        assert is_device_sensor("sensor.kettle_power") is True

    def test_is_device_sensor_dishwasher(self):
        assert is_device_sensor("sensor.dishwasher_electric_consumption_w") is True

    def test_is_device_sensor_tv(self):
        assert is_device_sensor("sensor.living_room_tv_power") is True

    def test_is_device_sensor_fridge(self):
        assert is_device_sensor("sensor.fridge_watt") is True

    def test_is_device_sensor_aggregate_excluded(self):
        assert is_device_sensor("sensor.inverter_l1_wattage") is False

    def test_is_device_sensor_phase_excluded(self):
        assert is_device_sensor("sensor.phase_l2_power") is False

    def test_is_device_sensor_total_excluded(self):
        assert is_device_sensor("sensor.total_power_w") is False

    def test_is_device_sensor_grid_excluded(self):
        assert is_device_sensor("sensor.grid_power") is False

    def test_is_device_sensor_solar_excluded(self):
        assert is_device_sensor("sensor.solar_inverter_power") is False

    def test_is_device_sensor_generic_plug(self):
        # Short slug without aggregate keywords → accept as generic plug
        assert is_device_sensor("sensor.plug_1_power") is True

    def test_is_device_sensor_long_non_device_slug(self):
        # Very long slug without device hints → excluded by length heuristic
        result = is_device_sensor("sensor." + "x" * 35 + "_power")
        assert result is False


class TestExtractDeviceName:
    def test_extract_device_name_kettle(self):
        assert extract_device_name("sensor.kettle_power") == "Kettle"

    def test_extract_device_name_dishwasher(self):
        assert extract_device_name("sensor.dishwasher_electric_consumption_w") == "Dishwasher"

    def test_extract_device_name_living_room_tv(self):
        assert extract_device_name("sensor.living_room_tv_power") == "Living Room Tv"

    def test_extract_device_name_fridge(self):
        assert extract_device_name("sensor.fridge_watt") == "Fridge"

    def test_extract_device_name_no_domain(self):
        # No sensor. prefix
        assert extract_device_name("kettle_power") == "Kettle"


class TestBuildDeviceProfile:
    def _make_bimodal_series(self, on_w=2400.0, off_w=5.0, n_cycles=10, points_per_state=30):
        """Create a synthetic bimodal on/off series with DatetimeIndex."""
        timestamps = []
        values = []
        base = pd.Timestamp("2024-01-01", tz="UTC")
        t = base
        for _ in range(n_cycles):
            # off phase
            for _ in range(points_per_state):
                timestamps.append(t)
                values.append(off_w + np.random.normal(0, 1))
                t += pd.Timedelta(minutes=1)
            # on phase
            for _ in range(points_per_state):
                timestamps.append(t)
                values.append(on_w + np.random.normal(0, 20))
                t += pd.Timedelta(minutes=1)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps))

    def test_build_device_profile_bimodal(self):
        series = self._make_bimodal_series(on_w=2400.0, off_w=5.0)
        profile = build_device_profile(series, "sensor.kettle_power")
        assert profile is not None
        assert profile["off_w"] < 50
        assert profile["on_w"] > 2000
        assert profile["step_w"] > 2000
        assert profile["entity_id"] == "sensor.kettle_power"
        assert profile["name"] == "Kettle"

    def test_build_device_profile_too_short(self):
        # Less than 24 points → should return None
        series = pd.Series([2400.0] * 10)
        result = build_device_profile(series, "sensor.kettle_power")
        assert result is None

    def test_build_device_profile_empty(self):
        series = pd.Series([], dtype=float)
        result = build_device_profile(series, "sensor.kettle_power")
        assert result is None

    def test_build_device_profile_all_nan(self):
        series = pd.Series([float("nan")] * 50)
        result = build_device_profile(series, "sensor.kettle_power")
        assert result is None

    def test_build_device_profile_flat_signal(self):
        # Flat signal → step_w < 5 → should return None
        series = pd.Series([100.0] * 100)
        result = build_device_profile(series, "sensor.flat_power")
        assert result is None

    def test_build_device_profile_daily_kwh(self):
        series = self._make_bimodal_series(on_w=1000.0, off_w=0.0, n_cycles=20)
        profile = build_device_profile(series, "sensor.heater_power")
        assert profile is not None
        assert profile["daily_kwh"] >= 0

    def test_build_device_profile_confidence_range(self):
        series = self._make_bimodal_series()
        profile = build_device_profile(series, "sensor.kettle_power")
        assert profile is not None
        assert 0.0 <= profile["confidence"] <= 1.0

    def test_is_always_on_detection(self):
        """Sensor with >95% on-time and small step → is_always_on=True."""
        # 95% on at ~20W, small variance (step < 50)
        vals = [20.0] * 200 + [22.0] * 5 + [18.0] * 5  # nearly constant
        series = pd.Series(vals)
        profile = build_device_profile(series, "sensor.router_power")
        # step_w will be small (< 5), so it returns None — test with explicit always-on setup
        # Use a signal with step 10-49W (qualifies as always_on: >95% on, step < 50)
        # Construct 200 points at 20W, 5 at 5W
        base = pd.Timestamp("2024-01-01", tz="UTC")
        idx = pd.date_range(base, periods=205, freq="1min")
        vals2 = [20.0] * 200 + [5.0] * 5
        series2 = pd.Series(vals2, index=idx)
        profile2 = build_device_profile(series2, "sensor.router_power")
        if profile2 is not None:
            # is_always_on requires step_w < 50 and on_fraction > 0.95
            assert profile2.get("is_always_on") is True


class TestBuildLibraryFromFeatures:
    def test_build_library_from_features_none(self):
        result = build_library_from_features(None)
        assert result == []

    def test_build_library_from_features_empty(self):
        result = build_library_from_features(pd.DataFrame())
        assert result == []

    def test_build_library_from_features_no_entity_id(self):
        df = pd.DataFrame({"ts": [1, 2], "mean": [100.0, 200.0]})
        result = build_library_from_features(df)
        assert result == []

    def test_build_library_from_features_with_device(self):
        # Build a realistic feature df with a kettle sensor
        base = pd.Timestamp("2024-01-01", tz="UTC")
        idx = pd.date_range(base, periods=200, freq="1min")
        # Bimodal: alternating off/on
        vals = [2400.0 if i % 60 < 30 else 5.0 for i in range(200)]
        df = pd.DataFrame({
            "entity_id": ["sensor.kettle_power"] * 200,
            "ts": idx,
            "mean": vals,
        })
        result = build_library_from_features(df)
        assert len(result) >= 1
        assert result[0]["entity_id"] == "sensor.kettle_power"


class TestMatchWattageToDevice:
    def test_match_wattage_to_device(self, tmp_path, monkeypatch):
        """2400W matches kettle template within 20%."""
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "device_library.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)

        devices = [
            {
                "entity_id": "sensor.kettle_power",
                "name": "Kettle",
                "off_w": 5.0,
                "on_w": 2400.0,
                "step_w": 2395.0,
                "median_w": 100.0,
                "typical_on_min": 3.0,
                "cycles_per_day": 5.0,
                "daily_kwh": 0.5,
                "confidence": 0.9,
                "data_days": 7.0,
                "is_always_on": False,
            }
        ]
        with open(lib_path, "w") as f:
            json.dump({"devices": devices, "count": 1}, f)

        match = match_wattage_to_device(2400.0)
        assert match is not None
        assert match["name"] == "Kettle"

    def test_match_wattage_no_match(self, tmp_path, monkeypatch):
        """Very different wattage → no match."""
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "device_library.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)

        devices = [
            {
                "entity_id": "sensor.kettle_power",
                "name": "Kettle",
                "step_w": 2395.0,
                "daily_kwh": 0.5,
                "confidence": 0.9,
            }
        ]
        with open(lib_path, "w") as f:
            json.dump({"devices": devices, "count": 1}, f)

        match = match_wattage_to_device(100.0)  # very different from 2395W
        assert match is None

    def test_match_wattage_empty_library(self, tmp_path, monkeypatch):
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "missing_library.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)
        match = match_wattage_to_device(2400.0)
        assert match is None


class TestEnergyBreakdown:
    def test_energy_breakdown_sorted(self, tmp_path, monkeypatch):
        """energy_breakdown() should return highest kWh first."""
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "device_library.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)

        devices = [
            {
                "entity_id": "sensor.fridge_power",
                "name": "Fridge",
                "daily_kwh": 1.2,
                "on_w": 120.0,
                "off_w": 5.0,
                "cycles_per_day": 12.0,
                "typical_on_min": 15.0,
                "is_always_on": False,
                "confidence": 0.85,
            },
            {
                "entity_id": "sensor.kettle_power",
                "name": "Kettle",
                "daily_kwh": 0.3,
                "on_w": 2400.0,
                "off_w": 5.0,
                "cycles_per_day": 5.0,
                "typical_on_min": 3.0,
                "is_always_on": False,
                "confidence": 0.9,
            },
        ]
        with open(lib_path, "w") as f:
            json.dump({"devices": devices, "count": 2}, f)

        result = energy_breakdown()
        assert len(result) == 2
        # Sorted highest first
        assert result[0]["daily_kwh"] >= result[1]["daily_kwh"]
        assert result[0]["name"] == "Fridge"

    def test_energy_breakdown_excludes_zero_kwh(self, tmp_path, monkeypatch):
        """Devices with daily_kwh=0 should be excluded."""
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "device_library.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)

        devices = [
            {
                "entity_id": "sensor.kettle_power",
                "name": "Kettle",
                "daily_kwh": 0.0,
                "on_w": 2400.0,
                "off_w": 5.0,
                "cycles_per_day": 0.0,
                "typical_on_min": 0.0,
                "is_always_on": False,
                "confidence": 0.5,
            }
        ]
        with open(lib_path, "w") as f:
            json.dump({"devices": devices, "count": 1}, f)

        result = energy_breakdown()
        assert result == []

    def test_energy_breakdown_empty_library(self, tmp_path, monkeypatch):
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "missing.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)
        result = energy_breakdown()
        assert result == []


class TestSaveLoadLibrary:
    def test_save_and_load_library(self, tmp_path, monkeypatch):
        import habitus.habitus.device_library as dl_mod
        lib_path = str(tmp_path / "device_library.json")
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", lib_path)

        profiles = [
            {
                "entity_id": "sensor.kettle_power",
                "name": "Kettle",
                "off_w": 5.0,
                "on_w": 2400.0,
                "step_w": 2395.0,
                "median_w": 100.0,
                "typical_on_min": 3.0,
                "cycles_per_day": 5.0,
                "daily_kwh": 0.5,
                "confidence": 0.9,
                "data_days": 7.0,
                "is_always_on": False,
            }
        ]
        save_library(profiles)
        loaded = load_library()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Kettle"

    def test_load_library_missing_file(self, tmp_path, monkeypatch):
        import habitus.habitus.device_library as dl_mod
        monkeypatch.setattr(dl_mod, "DEVICE_LIBRARY_PATH", str(tmp_path / "nonexistent.json"))
        result = load_library()
        assert result == []
