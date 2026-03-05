"""Tests for cost estimation integration in suggestions and automation gaps."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCostEnrichmentInSuggestions:
    """Cost estimates should be added to energy-consuming entity suggestions."""

    def test_suggestion_with_power_entity_gets_cost_estimate(self, tmp_data_dir: Path):
        """A suggestion with a light entity gets cost_estimate populated."""
        from habitus.habitus import cost_estimator

        suggestions = [
            {
                "id": "sug-1",
                "title": "Turn off lights when away",
                "entity_id": "light.living_room",
                "applicable": True,
            }
        ]
        result = cost_estimator.enrich_with_cost(suggestions)
        assert len(result) == 1
        # light domain should produce a cost estimate
        assert "cost_estimate" in result[0]
        assert result[0]["cost_estimate"]["monthly_saving_eur"] > 0

    def test_suggestion_with_switch_entity_gets_cost_estimate(self, tmp_data_dir: Path):
        """A suggestion with a switch entity gets cost_estimate populated."""
        from habitus.habitus import cost_estimator

        suggestions = [
            {
                "id": "sug-2",
                "title": "Auto-off switch",
                "entity_id": "switch.heater",
                "applicable": True,
            }
        ]
        result = cost_estimator.enrich_with_cost(suggestions)
        assert len(result) == 1
        assert "cost_estimate" in result[0]
        assert result[0]["cost_estimate"]["monthly_saving_eur"] > 0

    def test_suggestion_with_non_energy_entity_no_cost_estimate(self, tmp_data_dir: Path):
        """A suggestion with a sensor/non-power entity does not get cost_estimate."""
        from habitus.habitus import cost_estimator

        suggestions = [
            {
                "id": "sug-3",
                "title": "Temperature alert",
                "entity_id": "sensor.temperature",
                "applicable": True,
            }
        ]
        result = cost_estimator.enrich_with_cost(suggestions)
        assert len(result) == 1
        # sensor domain should NOT produce a cost estimate
        assert "cost_estimate" not in result[0]

    def test_suggestion_without_entity_id_graceful(self, tmp_data_dir: Path):
        """A suggestion with no entity_id does not crash."""
        from habitus.habitus import cost_estimator

        suggestions = [
            {
                "id": "sug-4",
                "title": "Generic suggestion",
                "applicable": True,
            }
        ]
        result = cost_estimator.enrich_with_cost(suggestions)
        assert len(result) == 1
        assert "cost_estimate" not in result[0]


class TestCostEnrichmentInGaps:
    """Cost estimates should be added to automation gaps with energy entities."""

    def test_gap_with_light_entity_gets_monthly_saving(self, tmp_data_dir: Path):
        """A gap with a light entity gets monthly_saving_eur > 0."""
        from habitus.habitus import cost_estimator

        gaps = [
            {
                "suggestion": "Turn off lights when nobody home",
                "entities": ["light.kitchen"],
                "entity_id": "light.kitchen",
                "status": "missing",
            }
        ]
        result = cost_estimator.enrich_with_cost(gaps, entity_field="entity_id")
        assert len(result) == 1
        assert "cost_estimate" in result[0]
        assert result[0]["cost_estimate"]["monthly_saving_eur"] > 0

    def test_gap_with_non_energy_entity_graceful(self, tmp_data_dir: Path):
        """A gap with a binary_sensor entity does not crash."""
        from habitus.habitus import cost_estimator

        gaps = [
            {
                "suggestion": "Alert on motion after midnight",
                "entities": ["binary_sensor.kitchen_motion"],
                "entity_id": "binary_sensor.kitchen_motion",
                "status": "missing",
            }
        ]
        result = cost_estimator.enrich_with_cost(gaps, entity_field="entity_id")
        assert len(result) == 1
        # binary_sensor not in energy domains, no cost estimate
        assert "cost_estimate" not in result[0]

    def test_cost_estimate_fields_present(self, tmp_data_dir: Path):
        """cost_estimate dict has expected keys."""
        from habitus.habitus import cost_estimator

        items = [{"entity_id": "light.bedroom", "estimated_hours_saved_per_day": 2.0}]
        result = cost_estimator.enrich_with_cost(items)
        assert "cost_estimate" in result[0]
        ce = result[0]["cost_estimate"]
        assert "monthly_saving_eur" in ce
        assert "annual_saving_eur" in ce
        assert "estimated_watts" in ce


class TestPatternsCostIntegration:
    """Verify patterns.run() enriches suggestions before saving."""

    def test_patterns_run_enriches_suggestions(self, tmp_data_dir: Path):
        """patterns.run() should call enrich_with_cost on generated suggestions."""
        from habitus.habitus import patterns as pattern_engine

        with patch("habitus.habitus.patterns.discover_patterns", return_value={}), \
             patch("habitus.habitus.patterns.generate_suggestions", return_value=[
                 {"id": "s1", "title": "Test", "entity_id": "light.living_room", "applicable": True}
             ]), \
             patch("habitus.habitus.cost_estimator.enrich_with_cost") as mock_enrich:
            mock_enrich.side_effect = lambda items, **kw: items  # passthrough

            import pandas as pd
            import numpy as np
            df = pd.DataFrame({"hour": pd.date_range("2025-01-01", periods=10, freq="h")})
            pattern_engine.run(df, stat_ids=["light.living_room"])
            # enrich_with_cost should have been called
            assert mock_enrich.called


class TestAutomationGapCostIntegration:
    """Verify automation_gap.analyse() enriches gaps before returning."""

    def test_analyse_enriches_gaps(self, tmp_data_dir: Path):
        """automation_gap.analyse() should call enrich_with_cost on gaps."""
        import asyncio
        from habitus.habitus import automation_gap

        suggestions = [
            {"title": "Light off when away", "description": "Turn off lights when nobody home",
             "entity_id": "light.living_room", "yaml": "automation:\n  alias: test\n  trigger: []\n  action: []"}
        ]

        with patch("habitus.habitus.automation_gap._fetch_automations", return_value=[]), \
             patch("habitus.habitus.automation_gap._fetch_all_states", return_value=[]), \
             patch("habitus.habitus.cost_estimator.enrich_with_cost") as mock_enrich:
            mock_enrich.side_effect = lambda items, **kw: items  # passthrough

            asyncio.run(automation_gap.analyse("http://localhost", "token", suggestions))
            assert mock_enrich.called
