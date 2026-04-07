"""Lovelace Dashboard Generator — build optimised HA dashboard from usage patterns.

Analyses entity usage frequency + area grouping + domain.
Generates valid Lovelace YAML dashboard.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

import yaml as _yaml

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

DASHBOARD_PATH = os.path.join(DATA_DIR, "dashboard.yaml")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
BATTERY_PATH = os.path.join(DATA_DIR, "battery_status.json")
HA_AUTOMATIONS_PATH = os.path.join(DATA_DIR, "ha_automations.json")

# Domains considered as "controls" (quick access)
CONTROL_DOMAINS = {"light", "switch", "media_player", "climate", "fan", "cover", "lock", "input_boolean"}

# Domains for sensor display
SENSOR_DOMAINS = {"sensor", "binary_sensor"}

# Power/energy sensor keywords
ENERGY_KEYWORDS = {"power", "energy", "watt", "kwh", "consumption", "solar"}


def _load_json(path: str, default: Any = None) -> Any:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _entity_friendly_name(entity_id: str, attrs: dict[str, Any] | None = None) -> str:
    """Get a friendly display name for an entity."""
    if attrs and attrs.get("friendly_name"):
        return attrs["friendly_name"]
    return entity_id.split(".")[-1].replace("_", " ").title()


def _infer_area(entity_id: str, attrs: dict[str, Any] | None = None) -> str:
    """Infer area from entity attributes or entity_id."""
    if attrs:
        area = attrs.get("area") or attrs.get("room") or ""
        if area:
            return area

    eid_lower = entity_id.lower()
    rooms = [
        "living_room", "bedroom", "master_bedroom", "kitchen", "bathroom",
        "hallway", "office", "garage", "garden", "front", "back", "basement",
        "dining", "laundry", "utility",
    ]
    for room in rooms:
        if room in eid_lower:
            return room.replace("_", " ").title()

    return "Other"


def _is_energy_entity(entity_id: str) -> bool:
    """Check if entity is energy/power related."""
    eid_lower = entity_id.lower()
    return any(kw in eid_lower for kw in ENERGY_KEYWORDS)


def _is_battery_entity(entity_id: str) -> bool:
    """Check if entity is battery-related."""
    return "battery" in entity_id.lower()


def _build_entity_card(entity_id: str, name: str | None = None) -> dict[str, Any]:
    """Build a basic entity card."""
    domain = entity_id.split(".")[0]
    card: dict[str, Any] = {"type": "tile", "entity": entity_id}
    if name:
        card["name"] = name
    return card


def _build_entities_card(title: str, entities: list[str]) -> dict[str, Any]:
    """Build an entities card with multiple entities."""
    return {
        "type": "entities",
        "title": title,
        "entities": [{"entity": eid} for eid in entities],
    }


def _build_area_card(area: str, entities: list[str]) -> dict[str, Any]:
    """Build a card for a specific area."""
    return {
        "type": "entities",
        "title": area,
        "entities": [{"entity": eid} for eid in entities],
    }


def generate_dashboard(
    states: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an optimised Lovelace dashboard configuration.

    Args:
        states: HA state list. If None, loads from baseline.
        baseline: Baseline activity data for usage frequency.

    Returns:
        Lovelace dashboard config dict.
    """
    if baseline is None:
        baseline = _load_json(BASELINE_PATH, {})

    battery_data = _load_json(BATTERY_PATH, {})
    automations_raw = _load_json(HA_AUTOMATIONS_PATH, [])

    # Build entity info from states or baseline
    entity_info: dict[str, dict[str, Any]] = {}

    if states:
        for s in states:
            eid = s.get("entity_id", "")
            if eid:
                entity_info[eid] = {
                    "entity_id": eid,
                    "domain": eid.split(".")[0],
                    "state": s.get("state", ""),
                    "attrs": s.get("attributes", {}),
                    "area": _infer_area(eid, s.get("attributes")),
                    "frequency": 0,
                }

    # Enrich with baseline frequency data
    freq_data = baseline.get("entity_frequencies", {}) if baseline else {}
    for eid, freq in freq_data.items():
        if eid in entity_info:
            entity_info[eid]["frequency"] = freq
        elif eid:
            domain = eid.split(".")[0]
            entity_info[eid] = {
                "entity_id": eid,
                "domain": domain,
                "state": "",
                "attrs": {},
                "area": _infer_area(eid),
                "frequency": freq,
            }

    # Sort entities by frequency
    control_entities = [
        e for e in entity_info.values()
        if e["domain"] in CONTROL_DOMAINS
    ]
    control_entities.sort(key=lambda e: e.get("frequency", 0), reverse=True)

    # Group by area
    area_entities: dict[str, list[str]] = defaultdict(list)
    for e in control_entities:
        area_entities[e["area"]].append(e["entity_id"])

    # Find energy entities
    energy_entities = [
        eid for eid in entity_info
        if _is_energy_entity(eid)
    ]

    # Find battery entities
    battery_entities_list = battery_data.get("batteries", [])
    battery_entity_ids = [b["entity_id"] for b in battery_entities_list if b.get("alert") != "ok"]

    # Build cards
    cards = []

    # Section 1: Quick Controls (high-frequency entities)
    top_entities = [e["entity_id"] for e in control_entities[:8]]
    if top_entities:
        cards.append({
            "type": "horizontal-stack",
            "title": "Quick Controls",
            "cards": [_build_entity_card(eid) for eid in top_entities[:4]],
        })
        if len(top_entities) > 4:
            cards.append({
                "type": "horizontal-stack",
                "cards": [_build_entity_card(eid) for eid in top_entities[4:8]],
            })

    # Section 2: Per-room sections
    for area, area_eids in sorted(area_entities.items()):
        if area == "Other" and len(area_eids) > 10:
            continue  # skip large "Other" bucket
        if area_eids:
            cards.append(_build_area_card(area, area_eids[:10]))

    # Section 3: Energy section
    if energy_entities:
        cards.append({
            "type": "entities",
            "title": "⚡ Energy",
            "entities": [{"entity": eid} for eid in energy_entities[:8]],
        })

    # Section 4: Battery section
    if battery_entity_ids:
        cards.append({
            "type": "entities",
            "title": "🔋 Battery Status",
            "entities": [{"entity": eid} for eid in battery_entity_ids[:10]],
        })

    # Section 5: Automations card
    if automations_raw:
        if isinstance(automations_raw, list):
            auto_list = automations_raw[:10]
        else:
            auto_list = automations_raw.get("automations", [])[:10]

        auto_aliases = [a.get("alias", "") for a in auto_list if a.get("alias")]
        if auto_aliases:
            cards.append({
                "type": "markdown",
                "title": "🤖 Active Automations",
                "content": "\n".join(f"• {alias}" for alias in auto_aliases[:10]),
            })

    dashboard = {
        "title": "Habitus Dashboard",
        "views": [
            {
                "title": "Home",
                "path": "home",
                "icon": "mdi:home",
                "cards": cards,
            }
        ],
    }

    return dashboard


def generate_dashboard_yaml(
    states: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
) -> str:
    """Generate dashboard as YAML string."""
    dashboard = generate_dashboard(states, baseline)
    return _yaml.dump(dashboard, default_flow_style=False, allow_unicode=True)


def run(
    states: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run dashboard generator and save output.

    Returns:
        Dict with dashboard config and yaml string.
    """
    dashboard = generate_dashboard(states, baseline)
    yaml_str = _yaml.dump(dashboard, default_flow_style=False, allow_unicode=True)

    result = {
        "dashboard": dashboard,
        "yaml": yaml_str,
        "card_count": len(dashboard.get("views", [{}])[0].get("cards", [])),
    }

    os.makedirs(os.environ.get("DATA_DIR", "/data"), exist_ok=True)
    with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "dashboard.yaml"), "w") as f:
        f.write(yaml_str)

    log.info("Dashboard generator: %d cards generated", result["card_count"])
    return result


def load_dashboard_yaml() -> str:
    """Load cached dashboard YAML."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "dashboard.yaml")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "dashboard.yaml")) as f:
                return f.read()
    except Exception:
        pass
    return ""
