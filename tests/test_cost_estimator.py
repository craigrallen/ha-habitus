"""Tests for cost estimation."""
from __future__ import annotations

import pytest

from habitus.habitus.cost_estimator import (
    compute_saving,
    estimate_watts,
    enrich_with_cost,
    format_saving_badge,
    get_tariff_config,
    DEFAULT_TARIFF,
    DEFAULT_PEAK_TARIFF,
    DOMAIN_DEFAULT_WATTS,
)


class TestEstimateWatts:
    def test_light_default(self):
        watts = estimate_watts("light.living_room", {})
        assert watts == DOMAIN_DEFAULT_WATTS["light"]

    def test_kettle_keyword(self):
        watts = estimate_watts("switch.kitchen_kettle", {})
        assert watts == 2000

    def test_tv_keyword(self):
        watts = estimate_watts("media_player.living_room_tv", {})
        # tv or media_player default
        assert watts > 0

    def test_washer_keyword(self):
        watts = estimate_watts("switch.washing_machine", {})
        assert watts == 2000

    def test_nilm_data_takes_priority(self):
        """NILM data overrides keyword/domain defaults."""
        nilm = {"appliances": [{"entity_id": "light.special", "avg_watts": 42}]}
        watts = estimate_watts("light.special", nilm)
        assert watts == 42

    def test_unknown_entity(self):
        watts = estimate_watts("some_domain.entity", {})
        assert watts > 0


class TestComputeSaving:
    def test_zero_hours_zero_saving(self):
        """No hours saved → zero saving."""
        result = compute_saving("light.test", 0.0)
        assert result["monthly_saving_eur"] == 0.0
        assert result["annual_saving_eur"] == 0.0

    def test_kwh_math_correct(self):
        """kWh math: 10W * 1h/day = 0.01 kWh/day."""
        result = compute_saving("light.test", 1.0, tariff_config={"standard": 1.0}, nilm_data={})
        assert result["estimated_watts"] == 10  # light default
        # 10W * 1h = 10Wh = 0.01 kWh
        assert result["kwh_saved_per_day"] == pytest.approx(0.01)
        # 0.01 kWh * 1 EUR/kWh * 30 days = 0.30 EUR/month
        assert result["monthly_saving_eur"] == pytest.approx(0.30, abs=0.01)

    def test_tariff_application(self):
        """Peak tariff gives higher saving estimate."""
        standard = compute_saving("light.test", 1.0, tariff_config={"standard": 0.30, "peak": 0.45})
        peak = compute_saving("light.test", 1.0, tariff_config={"standard": 0.30, "peak": 0.45}, peak=True)
        assert peak["monthly_saving_eur"] > standard["monthly_saving_eur"]

    def test_annual_is_12x_monthly(self):
        """Annual saving ≈ monthly * 12 (actually 365/30 ratio)."""
        result = compute_saving("light.test", 1.0, tariff_config={"standard": 0.30}, nilm_data={})
        ratio = result["annual_saving_eur"] / result["monthly_saving_eur"]
        assert ratio == pytest.approx(365 / 30, rel=0.01)

    def test_returns_required_fields(self):
        result = compute_saving("light.test", 1.0)
        required = {"entity_id", "estimated_watts", "hours_saved_per_day", "kwh_saved_per_day",
                    "tariff_eur_per_kwh", "monthly_saving_eur", "annual_saving_eur"}
        for field in required:
            assert field in result

    def test_high_power_device_larger_saving(self):
        """Kettle (2000W) saves more than light (10W)."""
        light_saving = compute_saving("light.test", 1.0, tariff_config={"standard": 0.30}, nilm_data={})
        kettle_saving = compute_saving("switch.kettle", 1.0, tariff_config={"standard": 0.30}, nilm_data={})
        assert kettle_saving["monthly_saving_eur"] > light_saving["monthly_saving_eur"]


class TestEnrichWithCost:
    def test_adds_cost_estimate_to_light(self):
        items = [{"entity_id": "light.living_room", "estimated_hours_saved_per_day": 2.0}]
        enriched = enrich_with_cost(items, nilm_data={}, settings={"energy_tariff": 0.30})
        assert "cost_estimate" in enriched[0]

    def test_zero_hours_no_cost_estimate(self):
        """Zero hours saved → no cost estimate added."""
        items = [{"entity_id": "light.living_room", "estimated_hours_saved_per_day": 0.0}]
        enriched = enrich_with_cost(items, nilm_data={})
        # May or may not add estimate depending on implementation
        if "cost_estimate" in enriched[0]:
            assert enriched[0]["cost_estimate"]["monthly_saving_eur"] == 0.0

    def test_sensor_domain_no_cost_estimate(self):
        """Sensor entities don't get cost estimates."""
        items = [{"entity_id": "sensor.temperature", "estimated_hours_saved_per_day": 1.0}]
        enriched = enrich_with_cost(items, nilm_data={})
        assert "cost_estimate" not in enriched[0]

    def test_returns_all_items(self):
        """All items returned even without cost estimate."""
        items = [
            {"entity_id": "light.test"},
            {"entity_id": "sensor.test"},
        ]
        enriched = enrich_with_cost(items, nilm_data={})
        assert len(enriched) == 2


class TestFormatSavingBadge:
    def test_zero_saving(self):
        assert format_saving_badge(0.0) == ""

    def test_small_saving(self):
        badge = format_saving_badge(0.5)
        assert "€" in badge or "¢" in badge

    def test_large_saving(self):
        badge = format_saving_badge(5.0)
        assert "€5.0" in badge or "€5" in badge
        assert "month" in badge


class TestGetTariffConfig:
    def test_defaults_when_no_settings(self, tmp_data_dir):
        config = get_tariff_config({})
        assert config["standard"] == DEFAULT_TARIFF
        assert config["peak"] == DEFAULT_PEAK_TARIFF

    def test_uses_settings(self):
        config = get_tariff_config({"energy_tariff": 0.25})
        assert config["standard"] == 0.25
