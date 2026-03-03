"""Tests for habitus.habitus.insights — energy insights computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from habitus.habitus import insights as ins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_baseline(hour_powers: dict[int, float]) -> dict[str, Any]:
    """Build a minimal baseline.json dict with a single day-of-week."""
    return {
        f"{h}_0": {"mean_power": w, "std_power": w * 0.1, "n_samples": 10}
        for h, w in hour_powers.items()
    }


def _make_entity_baselines(
    entities: dict[str, dict[int, float]],
) -> dict[str, Any]:
    """Build entity_baselines.json with one slot per hour (dow=0)."""
    result: dict[str, Any] = {}
    for eid, hour_means in entities.items():
        result[eid] = {
            f"{h}_0": {"mean": m, "std": m * 0.1, "n": 10}
            for h, m in hour_means.items()
        }
    return result


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------


def test_load_json_missing_file(tmp_path: Path) -> None:
    assert ins._load_json(str(tmp_path / "no.json"), "default") == "default"


def test_load_json_valid(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text('{"a": 1}')
    assert ins._load_json(str(p), {}) == {"a": 1}


def test_load_json_invalid(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert ins._load_json(str(p), []) == []


# ---------------------------------------------------------------------------
# _is_power_entity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "eid,expected",
    [
        ("sensor.kitchen_sensor_electric_consumed_w", True),
        ("sensor.solar_panel_roof_fore_energy_watts", True),
        ("sensor.mastervolt_total_load", True),
        ("sensor.bathroom_lights_electric_consumed_w", True),
        ("binary_sensor.hallway_motion", False),
        ("light.living_room_ceiling", False),
        ("sensor.outdoor_temperature", False),
        ("person.craig", False),
    ],
)
def test_is_power_entity(eid: str, expected: bool) -> None:
    assert ins._is_power_entity(eid) == expected


# ---------------------------------------------------------------------------
# _is_solar_entity
# ---------------------------------------------------------------------------


def test_is_solar_entity_match() -> None:
    assert ins._is_solar_entity("sensor.solar_panel_roof_fore_energy_watts") is True


def test_is_solar_entity_no_match() -> None:
    assert ins._is_solar_entity("sensor.kitchen_sensor_electric_consumed_w") is False


# ---------------------------------------------------------------------------
# _entity_hour_means
# ---------------------------------------------------------------------------


def test_entity_hour_means_basic() -> None:
    slots = {
        "8_0": {"mean": 100.0, "std": 10.0, "n": 5},
        "8_1": {"mean": 120.0, "std": 10.0, "n": 5},
        "9_0": {"mean": 200.0, "std": 20.0, "n": 5},
    }
    result = ins._entity_hour_means(slots)
    assert result[8] == pytest.approx(110.0)
    assert result[9] == pytest.approx(200.0)


def test_entity_hour_means_empty() -> None:
    assert ins._entity_hour_means({}) == {}


def test_entity_hour_means_skips_bad_keys() -> None:
    slots = {
        "bad_key": {"mean": 50.0},
        "x_0": {"mean": 50.0},
        "8_0": {"mean": 100.0},
    }
    result = ins._entity_hour_means(slots)
    assert 8 in result
    assert len(result) == 1  # only hour 8 parsed cleanly


# ---------------------------------------------------------------------------
# compute_peak_hours
# ---------------------------------------------------------------------------


def test_compute_peak_hours_empty() -> None:
    assert ins.compute_peak_hours({}) == []


def test_compute_peak_hours_returns_top3() -> None:
    baseline = _make_baseline({7: 500, 8: 1200, 12: 900, 18: 800, 0: 100})
    result = ins.compute_peak_hours(baseline)
    assert len(result) == 3
    assert result[0]["hour"] == 8
    assert result[0]["mean_power_w"] == pytest.approx(1200.0)
    assert result[1]["hour"] == 12
    assert result[2]["hour"] == 18


def test_compute_peak_hours_fewer_than_3() -> None:
    baseline = _make_baseline({10: 300, 14: 500})
    result = ins.compute_peak_hours(baseline)
    assert len(result) == 2


def test_compute_peak_hours_label_format() -> None:
    baseline = _make_baseline({9: 400})
    result = ins.compute_peak_hours(baseline)
    assert result[0]["label"] == "09:00"


def test_compute_peak_hours_averages_across_dow() -> None:
    """Hour 8 appears on two days — result should be their average."""
    baseline = {
        "8_0": {"mean_power": 1000.0, "n_samples": 5},
        "8_1": {"mean_power": 500.0, "n_samples": 5},
    }
    result = ins.compute_peak_hours(baseline)
    assert result[0]["mean_power_w"] == pytest.approx(750.0)


def test_compute_peak_hours_ignores_invalid_keys() -> None:
    baseline = {
        "bad": {"mean_power": 9999.0},
        "8_0": {"mean_power": 400.0},
    }
    result = ins.compute_peak_hours(baseline)
    assert len(result) == 1
    assert result[0]["hour"] == 8


# ---------------------------------------------------------------------------
# compute_top_consumers
# ---------------------------------------------------------------------------


def test_compute_top_consumers_empty() -> None:
    assert ins.compute_top_consumers({}) == []


def test_compute_top_consumers_returns_power_entities_only() -> None:
    entity_bl = _make_entity_baselines(
        {
            "sensor.kitchen_sensor_electric_consumed_w": {h: 400.0 for h in range(24)},
            "binary_sensor.hallway_motion": {h: 0.5 for h in range(24)},
            "sensor.outdoor_temperature": {h: 20.0 for h in range(24)},
        }
    )
    result = ins.compute_top_consumers(entity_bl)
    assert len(result) == 1
    assert result[0]["entity_id"] == "sensor.kitchen_sensor_electric_consumed_w"


def test_compute_top_consumers_top5_limit() -> None:
    entity_bl = _make_entity_baselines(
        {f"sensor.device_{i}_electric_consumed_w": {h: float(i * 100) for h in range(24)} for i in range(1, 8)}
    )
    result = ins.compute_top_consumers(entity_bl)
    assert len(result) == 5
    # Highest wattage device first
    assert result[0]["mean_w"] >= result[1]["mean_w"]


def test_compute_top_consumers_excludes_low_wattage() -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.ghost_electric_consumed_w": {h: 0.5 for h in range(24)}}
    )
    assert ins.compute_top_consumers(entity_bl) == []


def test_compute_top_consumers_strips_internal_keys() -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.fan_electric_consumed_w": {h: 200.0 for h in range(24)}}
    )
    entity_bl["_z_score_run"] = {"timestamp": "2026-01-01", "top5": []}
    result = ins.compute_top_consumers(entity_bl)
    assert len(result) == 1


def test_compute_top_consumers_name_formatting() -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.bathroom_lights_electric_consumed_w": {h: 80.0 for h in range(24)}}
    )
    result = ins.compute_top_consumers(entity_bl)
    assert result[0]["name"] == "Bathroom Lights Electric Consumed W"


# ---------------------------------------------------------------------------
# compute_waste
# ---------------------------------------------------------------------------


def test_compute_waste_empty() -> None:
    assert ins.compute_waste({}) == []


def test_compute_waste_detects_off_peak_consumers() -> None:
    # Device draws 50 W around the clock — on-peak AND off-peak
    hour_means = {h: 50.0 for h in range(24)}
    entity_bl = _make_entity_baselines(
        {"sensor.always_on_electric_consumed_w": hour_means}
    )
    result = ins.compute_waste(entity_bl)
    assert len(result) == 1
    assert result[0]["off_peak_mean_w"] == pytest.approx(50.0, abs=1)


def test_compute_waste_ignores_off_peak_only_devices_below_threshold() -> None:
    # Device draws 5 W off-peak (below 10 W threshold)
    hour_means = {h: 5.0 for h in range(24)}
    entity_bl = _make_entity_baselines(
        {"sensor.tiny_load_electric_consumed_w": hour_means}
    )
    assert ins.compute_waste(entity_bl) == []


def test_compute_waste_excludes_non_power_entities() -> None:
    hour_means = {h: 999.0 for h in range(24)}
    entity_bl = _make_entity_baselines({"sensor.outdoor_temperature": hour_means})
    assert ins.compute_waste(entity_bl) == []


def test_compute_waste_sorted_descending() -> None:
    entity_bl = _make_entity_baselines(
        {
            "sensor.small_load_electric_consumed_w": {h: 15.0 for h in range(24)},
            "sensor.big_load_electric_consumed_w": {h: 80.0 for h in range(24)},
        }
    )
    result = ins.compute_waste(entity_bl)
    assert result[0]["off_peak_mean_w"] >= result[1]["off_peak_mean_w"]


# ---------------------------------------------------------------------------
# compute_solar_ratio
# ---------------------------------------------------------------------------


def test_compute_solar_ratio_no_solar() -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.kitchen_sensor_electric_consumed_w": {h: 400.0 for h in range(24)}}
    )
    assert ins.compute_solar_ratio(entity_bl) is None


def test_compute_solar_ratio_solar_entity_found() -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.solar_panel_roof_fore_energy_watts": {h: 500.0 for h in range(12)}}
    )
    result = ins.compute_solar_ratio(entity_bl)
    assert result is not None
    assert result["solar_entity"] == "sensor.solar_panel_roof_fore_energy_watts"
    assert "solar_mean_w" in result


def test_compute_solar_ratio_zero_generation() -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.solar_panel_roof_fore_energy_watts": {h: 0.0 for h in range(12)}}
    )
    result = ins.compute_solar_ratio(entity_bl)
    assert result is not None
    assert result["ratio"] is None
    assert result.get("reason") == "no_solar_generation"


def test_compute_solar_ratio_with_load_entity() -> None:
    """Solar = 1000 W, load = 800 W → self-consumption = 800/1000 = 0.8."""
    entity_bl = _make_entity_baselines(
        {
            "sensor.solar_panel_roof_fore_energy_watts": {h: 1000.0 for h in range(6, 18)},
            "sensor.mastervolt_total_load": {h: 800.0 for h in range(6, 18)},
        }
    )
    result = ins.compute_solar_ratio(entity_bl)
    assert result is not None
    assert result["ratio"] == pytest.approx(0.8, abs=0.01)


def test_compute_solar_ratio_without_load_entity() -> None:
    """When no load entity is found, ratio should be None."""
    entity_bl = _make_entity_baselines(
        {"sensor.solar_panel_roof_fore_energy_watts": {h: 500.0 for h in range(6, 18)}}
    )
    result = ins.compute_solar_ratio(entity_bl)
    assert result is not None
    assert result["ratio"] is None


# ---------------------------------------------------------------------------
# compute_insights (integration)
# ---------------------------------------------------------------------------


def test_compute_insights_no_files(tmp_path: Path) -> None:
    result = ins.compute_insights(str(tmp_path))
    assert result["peak_hours"] == []
    assert result["top_consumers"] == []
    assert result["estimated_waste"] == []
    assert result["solar_self_consumption"] is None


def test_compute_insights_with_data(tmp_path: Path) -> None:
    baseline = _make_baseline({7: 500, 8: 1200, 18: 900})
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))

    entity_bl = _make_entity_baselines(
        {
            "sensor.kitchen_sensor_electric_consumed_w": {h: 400.0 for h in range(24)},
            "sensor.solar_panel_roof_fore_energy_watts": {h: 600.0 for h in range(6, 18)},
            "sensor.mastervolt_total_load": {h: 800.0 for h in range(24)},
        }
    )
    (tmp_path / "entity_baselines.json").write_text(json.dumps(entity_bl))

    result = ins.compute_insights(str(tmp_path))

    assert len(result["peak_hours"]) <= 3
    assert result["peak_hours"][0]["hour"] == 8
    assert len(result["top_consumers"]) >= 1
    assert result["solar_self_consumption"] is not None


def test_compute_insights_strips_internal_keys(tmp_path: Path) -> None:
    entity_bl = _make_entity_baselines(
        {"sensor.fan_electric_consumed_w": {h: 200.0 for h in range(24)}}
    )
    entity_bl["_z_score_run"] = {"timestamp": "2026-01-01", "top5": []}
    (tmp_path / "entity_baselines.json").write_text(json.dumps(entity_bl))

    result = ins.compute_insights(str(tmp_path))
    assert len(result["top_consumers"]) == 1


def test_compute_insights_uses_env_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    result = ins.compute_insights()  # no explicit data_dir
    assert "peak_hours" in result
    assert result["peak_hours"] == []


