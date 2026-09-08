"""Regression coverage for metadata-driven identification."""

import json

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from habitus.habitus import anomaly_breakdown as ab
from habitus.habitus.entity_metadata import SCHEMA_VERSION, identify, power_factor, temperature_c
from habitus.habitus.sensor_classifier import classify_sensor
from habitus.habitus.insights import compute_top_consumers


@pytest.mark.parametrize(
    "unit,kind",
    [
        ("Wh", "energy"),
        ("kW", "power"),
        ("°F", "temperature"),
        ("A", "current"),
        ("V", "voltage"),
        ("L", "volume"),
        ("bar", "pressure"),
    ],
)
def test_explicit_unit_beats_misleading_name(unit, kind):
    info = identify("sensor.energy_kwh", {"unit_of_measurement": unit})
    assert info["unit_of_measurement"] == unit
    assert info["quantity"] == kind


@pytest.mark.parametrize(
    "eid", ["sensor.water_level", "sensor.attic", "sensor.energy", "sensor.power_factor"]
)
def test_ambiguous_names_do_not_invent_units(eid):
    assert identify(eid)["unit_of_measurement"] == ""


def test_explicit_unitless_is_preserved():
    assert identify("sensor.energy_kwh", {"unit_of_measurement": None})["unit_of_measurement"] == ""


def test_measurement_wins_over_rising_history():
    assert classify_sensor("sensor.temperature", "measurement", np.arange(100)) == "gauge"


def test_domain_identifies_binary_without_history():
    assert classify_sensor("binary_sensor.door") == "binary"


def test_total_counter_and_nonfinite_history():
    assert classify_sensor("sensor.water", "total") == "accumulating"
    assert classify_sensor("sensor.unknown", history=[np.nan, np.inf]) == "gauge"


def test_conversions_and_conflicts():
    assert power_factor(identify("sensor.x", {"unit_of_measurement": "kW"})) == 1000
    assert (
        power_factor(identify("sensor.x", {"unit_of_measurement": "kW", "device_class": "energy"}))
        is None
    )
    assert temperature_c(212, "°F") == 100
    assert temperature_c(273.15, "K") == 0
    with pytest.raises(ValueError):
        temperature_c(1, "unknown")


def test_power_insights_exclude_energy_and_convert_kw():
    def slots(unit, mean):
        return {"_meta": identify("sensor.x", {"unit_of_measurement": unit}), "0_0": {"mean": mean}}

    results = compute_top_consumers(
        {"sensor.electric_energy": slots("kWh", 5000), "sensor.opaque": slots("kW", 2)}
    )
    assert len(results) == 1
    assert results[0]["mean_w"] == 2000


def test_metadata_persisted_and_temperature_not_differenced(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "ENTITY_BASELINES_PATH", str(tmp_path / "entity_baselines.json"))
    ts = pd.date_range("2026-01-01", periods=24 * 28, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "entity_id": "sensor.energy_kwh",
            "ts": ts,
            "mean": np.linspace(40, 80, len(ts)),
            "sum": np.nan,
        }
    )
    df.attrs["entity_metadata"] = {
        "sensor.energy_kwh": {
            "unit_of_measurement": "°F",
            "device_class": "temperature",
            "state_class": "measurement",
            "friendly_name": "Cabin",
        }
    }
    ab.build_entity_baselines(df)
    baseline = json.loads((tmp_path / "entity_baselines.json").read_text())["sensor.energy_kwh"]
    assert baseline["_meta"]["unit_of_measurement"] == "°F"
    assert baseline["_meta"]["sensor_type"] == "gauge"
    assert all(
        v["baseline_type"] == "absolute" for k, v in baseline.items() if not k.startswith("_")
    )


def test_rate_baseline_divides_by_elapsed_hours():
    ts = pd.date_range("2026-01-01", periods=24 * 28, freq="2h", tz="UTC")
    group = pd.DataFrame(
        {
            "ts": ts,
            "v": np.arange(len(ts)) * 4.0,
            "hour_of_day": ts.hour,
            "day_of_week": ts.dayofweek,
        }
    )
    baseline = ab._build_rate_baseline(group)
    assert baseline
    assert all(v["mean"] == 2 for v in baseline.values())


@freeze_time("2026-02-02 12:00:00")
def test_live_metadata_and_rate_units(tmp_path, monkeypatch):
    path = tmp_path / "entity_baselines.json"
    monkeypatch.setattr(ab, "ENTITY_BASELINES_PATH", str(path))
    monkeypatch.setattr(ab, "ENTITY_ANOMALIES_PATH", str(tmp_path / "anomalies.json"))
    eid = "sensor.energy_kwh"
    meta = {
        **identify(
            eid,
            {
                "unit_of_measurement": "L",
                "state_class": "total_increasing",
                "friendly_name": "Water",
            },
        ),
        "schema_version": SCHEMA_VERSION,
        "sensor_type": "accumulating",
        "first_seen": "2026-01-01T00:00:00",
    }
    path.write_text(
        json.dumps(
            {
                eid: {
                    "_meta": meta,
                    "12_0": {"mean": 1, "std": 0.1, "n": 30, "baseline_type": "rate"},
                },
                "_accumulating_state": {
                    eid: {
                        "prev_value": 100,
                        "prev_ts": "2026-02-02T10:00:00",
                        "first_delta_ts": "2026-01-01T00:00:00",
                    }
                },
            }
        )
    )
    result = ab.score_entities({eid: {"state": "110", "attributes": {"unit_of_measurement": "L"}}})
    assert result[0]["current_value"] == 5
    assert result[0]["unit"] == "L/h"
    assert result[0]["name"] == "Water"


@pytest.mark.asyncio
async def test_discovery_preserves_statistics_units_and_live_attributes(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, Mock
    from habitus.habitus import main

    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE statistics_meta (statistic_id TEXT, unit_of_measurement TEXT, has_sum INTEGER)"
    )
    conn.execute("INSERT INTO statistics_meta VALUES (?,?,?)", ("sensor.meter", "Wh", 1))
    monkeypatch.setattr(main, "_sqlite_connect", lambda: conn)
    ws = AsyncMock()
    ws.recv.return_value = json.dumps(
        {"result": [{"statistic_id": "sensor.meter", "unit_of_measurement": "Wh", "has_sum": True}]}
    )
    monkeypatch.setattr(main, "ws_connect", AsyncMock(return_value=ws))
    response = Mock()
    response.json.return_value = [
        {
            "entity_id": "sensor.meter",
            "attributes": {
                "unit_of_measurement": "kWh",
                "state_class": "total_increasing",
                "device_class": "energy",
            },
        }
    ]
    monkeypatch.setattr(main.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    await main._refresh_entity_metadata()
    assert main._ENTITY_METADATA["sensor.meter"]["unit_of_measurement"] == "Wh"
    assert main._ENTITY_METADATA["sensor.meter"]["state_class"] == "total_increasing"


@pytest.mark.parametrize("unit,value,expected", [("W", 2000, 2000), ("kW", 2, 2000)])
def test_features_normalize_power(tmp_path, monkeypatch, unit, value, expected):
    from habitus.habitus import main

    for key in ("HABITUS_POWER_ENTITY", "HABITUS_ENERGY_GRID", "HABITUS_ENERGY_RATES"):
        monkeypatch.delenv(key, raising=False)
    df = pd.DataFrame(
        {
            "entity_id": ["sensor.opaque"] * 4,
            "ts": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "mean": [value] * 4,
            "sum": [np.nan] * 4,
        }
    )
    df.attrs["entity_metadata"] = {
        "sensor.opaque": {
            "device_class": "power",
            "unit_of_measurement": unit,
            "state_class": "measurement",
        }
    }
    result = main.build_features(df)
    assert result["total_power_w"].tolist() == [expected] * 4


def test_overrides_roundtrip_and_sources(tmp_path):
    from habitus.habitus.entity_metadata import read_overrides, apply_override

    assert read_overrides(str(tmp_path)) == {}
    (tmp_path / "entity_overrides.json").write_text('{"sensor.x":{"unit_of_measurement":"A"}}')
    attrs = apply_override(
        {"unit_of_measurement": "kWh"}, read_overrides(str(tmp_path))["sensor.x"]
    )
    result = identify("sensor.x", attrs)
    assert result["unit_of_measurement"] == "A"
    assert identify("sensor.x", result)["identification_source"] == "override"
    (tmp_path / "entity_overrides.json").write_text("broken")
    assert read_overrides(str(tmp_path)) == {}


def test_sensor_page_escapes_metadata_and_persists_correction(tmp_path, monkeypatch):
    from habitus.habitus import web

    monkeypatch.setattr(web, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(web, "STATE_PATH", str(tmp_path / "run_state.json"))
    monkeypatch.setattr(web._trainer, "is_running", lambda: False)
    (tmp_path / "entity_metadata.json").write_text(
        json.dumps(
            {"sensor.x": {"friendly_name": "<script>bad</script>", "unit_of_measurement": "Wh"}}
        )
    )
    client = web.app.test_client()
    response = client.get("/sensors")
    assert response.status_code == 200
    assert b"&lt;script&gt;" in response.data
    assert client.post("/sensors", data={"entity_id": "sensor.nope"}).status_code == 400
    response = client.post(
        "/sensors", data={"entity_id": "sensor.x", "unit": "A", "action": "save"}
    )
    assert response.status_code == 200
    assert (
        json.loads((tmp_path / "entity_overrides.json").read_text())["sensor.x"][
            "unit_of_measurement"
        ]
        == "A"
    )
    client.post("/sensors", data={"entity_id": "sensor.x", "action": "reset"})
    assert json.loads((tmp_path / "entity_overrides.json").read_text()) == {}


@pytest.mark.asyncio
@freeze_time("2026-02-02 12:00:00")
async def test_phantom_converts_wh_and_accepts_seconds(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from habitus.habitus import phantom

    monkeypatch.setenv("HABITUS_ENERGY_GRID", "sensor.meter")
    monkeypatch.setattr(phantom, "DATA_DIR", str(tmp_path))
    (tmp_path / "entity_metadata.json").write_text(
        json.dumps({"sensor.meter": {"unit_of_measurement": "Wh", "device_class": "energy"}})
    )
    stamp = pd.Timestamp("2026-02-01", tz="UTC").timestamp()
    monkeypatch.setattr(
        phantom, "_fetch_statistics", AsyncMock(return_value=[{"start": stamp, "change": 2000}])
    )
    monkeypatch.setattr(
        phantom,
        "_fetch_hourly_statistics",
        AsyncMock(return_value=[{"start": stamp + 7200, "change": 500}]),
    )
    result = await phantom._run_async()
    assert result["total_12mo_kwh"] == 2
    assert result["same_days_comparison"]["this_month_first_n"] == 2
    assert result["overnight_baseline"]["avg_idle_kwh_per_hour"] == 0.5
    assert result["overnight_baseline"]["overnight_kwh_year"] == round(0.5 * 3 * 365)
    assert phantom._timestamp(stamp) == phantom._timestamp(stamp * 1000)


def test_activity_uses_device_class_before_name():
    from habitus.habitus.activity import classify_entity

    assert classify_entity("binary_sensor.opaque", {"device_class": "motion"}) == "motion"
    assert classify_entity("binary_sensor.motion", {"device_class": "battery"}) is None


def test_retrain_endpoint_is_nonblocking(monkeypatch):
    from unittest.mock import Mock
    from habitus.habitus import web

    start = Mock(return_value=True)
    monkeypatch.setattr(web._trainer, "start", start)
    client = web.app.test_client()
    assert client.post("/api/full_train").status_code == 200
    start.assert_called_once_with(365, mode="full")
    start.return_value = False
    assert client.post("/api/full_train").status_code == 409


@pytest.mark.parametrize(
    "value,source,target,expected",
    [
        (2, "kWh", "Wh", 2000),
        (1000, "mA", "A", 1),
        (32, "°F", "°C", 0),
        (0, "°C", "K", 273.15),
        (1, "bar", "hPa", 1000),
    ],
)
def test_live_unit_conversion(value, source, target, expected):
    from habitus.habitus.entity_metadata import convert_value

    assert convert_value(value, source, target) == pytest.approx(expected)


def test_incompatible_units_are_rejected():
    from habitus.habitus.entity_metadata import convert_value

    with pytest.raises(ValueError):
        convert_value(12, "A", "kWh")


@freeze_time("2026-02-02 12:00:00")
def test_live_kw_compared_to_watt_baseline(tmp_path, monkeypatch):
    path = tmp_path / "entity_baselines.json"
    monkeypatch.setattr(ab, "ENTITY_BASELINES_PATH", str(path))
    monkeypatch.setattr(ab, "ENTITY_ANOMALIES_PATH", str(tmp_path / "anomalies.json"))
    eid = "sensor.opaque"
    meta = {
        **identify(eid, {"unit_of_measurement": "W", "state_class": "measurement"}),
        "schema_version": SCHEMA_VERSION,
        "sensor_type": "gauge",
    }
    path.write_text(
        json.dumps(
            {
                eid: {
                    "_meta": meta,
                    "12_0": {"mean": 1000, "std": 100, "n": 30, "baseline_type": "absolute"},
                }
            }
        )
    )
    result = ab.score_entities({eid: {"state": "2", "attributes": {"unit_of_measurement": "kW"}}})
    assert result[0]["current_value"] == 2000
    assert result[0]["unit"] == "W"
