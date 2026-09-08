from __future__ import annotations

import pytest

from habitus.habitus import automation_gap, web


@pytest.mark.asyncio
async def test_gap_preserves_structured_yaml_and_entities(monkeypatch):
    suggestions = [
        {
            "id": "occupancy_lights",
            "title": "Occupancy Lights",
            "description": "Turn on living room lights when hallway motion is detected",
            "entities": ["binary_sensor.hallway_motion", "light.living_room_ceiling"],
            "yaml": "alias: \"Habitus — Occupancy lights\"\ntrigger: []\naction: []\nmode: single",
        }
    ]

    monkeypatch.setattr(automation_gap, "_fetch_automations", lambda *_: [])
    monkeypatch.setattr(
        automation_gap,
        "_fetch_all_states",
        lambda *_: [
            {"entity_id": "binary_sensor.hallway_motion"},
            {"entity_id": "light.living_room_ceiling"},
        ],
    )

    result = await automation_gap.analyse("http://ha.local", "token", suggestions, auto_scores=[])
    assert result["gaps"]
    gap = result["gaps"][0]
    assert gap["status"] == "missing"
    assert gap["suggestion"] == "Occupancy Lights"
    assert gap["entities"] == ["binary_sensor.hallway_motion", "light.living_room_ceiling"]
    assert gap["ha_automation_yaml"] == suggestions[0]["yaml"]


@pytest.mark.asyncio
async def test_gap_missing_items_include_ha_yaml(monkeypatch):
    suggestions = [
        {
            "id": "night_standby",
            "title": "Nighttime Standby",
            "description": "Reduce standby usage overnight",
            "entities": ["switch.living_room_tv_plug"],
        }
    ]

    monkeypatch.setattr(automation_gap, "_fetch_automations", lambda *_: [])
    monkeypatch.setattr(
        automation_gap,
        "_fetch_all_states",
        lambda *_: [{"entity_id": "switch.living_room_tv_plug"}],
    )

    result = await automation_gap.analyse("http://ha.local", "token", suggestions, auto_scores=[])
    assert result["gaps"][0]["status"] == "missing"
    assert "ha_automation_yaml" in result["gaps"][0]
    assert "switch.living_room_tv_plug" in result["gaps"][0]["ha_automation_yaml"]


@pytest.mark.asyncio
async def test_gap_matches_existing_by_yaml_semantics(monkeypatch):
    suggestions = [
        {
            "id": "occupancy_lights",
            "title": "Occupancy Lights",
            "description": "Turn on hallway light when motion is detected",
            "entities": ["binary_sensor.hallway_motion", "light.hallway_ceiling"],
            "yaml": (
                "alias: \"Occupancy Lights\"\n"
                "trigger:\n"
                "  - platform: state\n"
                "    entity_id: binary_sensor.hallway_motion\n"
                "    to: \"on\"\n"
                "action:\n"
                "  - service: light.turn_on\n"
                "    target:\n"
                "      entity_id: light.hallway_ceiling\n"
            ),
        }
    ]

    monkeypatch.setattr(
        automation_gap,
        "_fetch_automations",
        lambda *_: [
            {
                "entity_id": "automation.evening_hallway",
                "alias": "Evening Hallway",
                "trigger": [
                    {"platform": "state", "entity_id": "binary_sensor.hallway_motion", "to": "on"}
                ],
                "action": [
                    {"service": "light.turn_on", "target": {"entity_id": "light.hallway_ceiling"}}
                ],
                "state": "on",
                "last_triggered": None,
            }
        ],
    )
    monkeypatch.setattr(
        automation_gap,
        "_fetch_all_states",
        lambda *_: [
            {"entity_id": "binary_sensor.hallway_motion"},
            {"entity_id": "light.hallway_ceiling"},
        ],
    )

    result = await automation_gap.analyse("http://ha.local", "token", suggestions, auto_scores=[])
    gap = result["gaps"][0]
    assert gap["status"] == "exists_working"
    assert gap["matched_automation"] == "automation.evening_hallway"
    assert gap["match_score"] >= 45


def test_gap_cards_keep_add_to_ha_flow():
    assert "function addGapToHA(yamlId, btnId)" in web.PAGE
    assert "onclick=\"addGapToHA('" in web.PAGE
    assert "api/add_automation" in web.PAGE
