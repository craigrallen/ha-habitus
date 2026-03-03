"""Automation Builder — generate HA automation YAML from discovered patterns and scenes.

Takes discovered scenes and behavioral patterns and produces ready-to-install
Home Assistant automation configs with proper YAML, descriptions, and confidence.
"""

import datetime
import json
import logging
import os
from typing import Any

import requests

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
SMART_SUGGESTIONS_PATH = os.path.join(DATA_DIR, "smart_suggestions.json")
HA_AUTOMATIONS_PATH = os.path.join(DATA_DIR, "ha_automations.json")


def _ha_headers() -> dict[str, str]:
    """Return HA API auth headers."""
    token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _ha_url() -> str:
    return os.environ.get("HA_URL", "http://supervisor/core")


def fetch_ha_automations() -> list[dict[str, Any]]:
    """Fetch all user automations from HA REST API.

    Returns list of automation dicts with id, alias, description, triggers, etc.
    """
    try:
        r = requests.get(
            f"{_ha_url()}/api/states",
            headers=_ha_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            log.warning("Failed to fetch HA states: %d", r.status_code)
            return []

        automations = []
        for entity in r.json():
            if not entity["entity_id"].startswith("automation."):
                continue
            attrs = entity.get("attributes", {})
            automations.append({
                "entity_id": entity["entity_id"],
                "state": entity["state"],
                "alias": attrs.get("friendly_name", entity["entity_id"]),
                "last_triggered": attrs.get("last_triggered"),
                "current_state": entity["state"],
                "id": attrs.get("id", entity["entity_id"].split(".", 1)[1]),
            })
        log.info("Fetched %d HA automations", len(automations))
        return automations
    except Exception as e:
        log.warning("Could not fetch HA automations: %s", e)
        return []


def save_ha_automations(automations: list[dict[str, Any]]) -> None:
    """Cache fetched HA automations to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HA_AUTOMATIONS_PATH, "w") as f:
        json.dump({
            "automations": automations,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "count": len(automations),
        }, f, indent=2, default=str)


def load_ha_automations() -> list[dict[str, Any]]:
    """Load cached HA automations."""
    try:
        if os.path.exists(HA_AUTOMATIONS_PATH):
            with open(HA_AUTOMATIONS_PATH) as f:
                data = json.load(f)
            return data.get("automations", [])
    except Exception:
        pass
    return []


def _entities_overlap(suggestion_entities: list[str], ha_automations: list[dict]) -> str | None:
    """Check if a suggestion's entities overlap with existing HA automations.

    Returns the alias of the overlapping automation, or None.
    """
    sug_set = set(suggestion_entities)
    for auto in ha_automations:
        alias = auto.get("alias", "").lower()
        eid = auto.get("entity_id", "")
        # Check if any of the suggestion's entities appear in the automation name/id
        for ent in sug_set:
            short_name = ent.split(".")[1].replace("_", " ").lower()
            if short_name in alias or short_name in eid:
                return auto["alias"]
    return None


def build_scene_yaml(scene: dict[str, Any]) -> str:
    """Generate HA scene YAML from a detected scene."""
    entities_block = ""
    for eid, state in scene.get("entity_states", {}).items():
        domain = eid.split(".")[0]
        if domain == "light":
            entities_block += f"      {eid}:\n        state: \"{state}\"\n"
        else:
            entities_block += f"      {eid}:\n        state: \"{state}\"\n"

    safe_name = scene["name"].lower().replace(" ", "_").replace("&", "and")
    return f"""scene:
  - name: "Habitus — {scene['name']}"
    id: habitus_{safe_name}
    entities:
{entities_block.rstrip()}"""


def build_scene_automation_yaml(scene: dict[str, Any]) -> str:
    """Generate HA automation YAML that triggers a discovered scene at its peak time."""
    time_pattern = scene.get("time_pattern", {})
    peak_hour = time_pattern.get("peak_hour", 18)
    days = time_pattern.get("days", "daily")
    safe_name = scene["name"].lower().replace(" ", "_").replace("&", "and")

    # Build trigger
    trigger = f"""    - platform: time
      at: "{peak_hour:02d}:00:00\""""

    # Build condition based on day pattern
    condition = ""
    if days == "weekdays":
        condition = """
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]"""
    elif days == "weekends":
        condition = """
  condition:
    - condition: time
      weekday: [sat, sun]"""

    # Build action — turn on each entity
    actions = ""
    for eid in scene.get("entities", []):
        domain = eid.split(".")[0]
        service = f"{domain}.turn_on"
        actions += f"""
    - service: {service}
      target:
        entity_id: {eid}"""

    return f"""automation:
  alias: "Habitus — {scene['name']}"
  description: "Auto-detected scene: {scene.get('description', scene['name'])}"
  trigger:
{trigger}{condition}
  action:{actions}"""


def build_time_automation_yaml(
    alias: str,
    description: str,
    entities: list[str],
    hour: int,
    days: str = "daily",
    action: str = "turn_on",
) -> str:
    """Generate time-based automation YAML."""
    condition = ""
    if days == "weekdays":
        condition = """
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]"""
    elif days == "weekends":
        condition = """
  condition:
    - condition: time
      weekday: [sat, sun]"""

    actions = ""
    for eid in entities:
        domain = eid.split(".")[0]
        service = f"{domain}.{action}"
        actions += f"""
    - service: {service}
      target:
        entity_id: {eid}"""

    return f"""automation:
  alias: "{alias}"
  description: "{description}"
  trigger:
    - platform: time
      at: "{hour:02d}:00:00"{condition}
  action:{actions}"""


def generate_smart_suggestions(
    scenes: list[dict[str, Any]],
    patterns: dict[str, Any] | None = None,
    existing_suggestions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate merged smart suggestions from scenes and patterns.

    Combines scene-based automation suggestions with existing pattern-based
    suggestions, deduplicates, and adds overlap info with existing HA automations.

    Args:
        scenes: Discovered scenes from scene_detector.
        patterns: Discovered patterns dict (from patterns.json).
        existing_suggestions: Current suggestions from patterns.py.

    Returns:
        List of smart suggestion dicts.
    """
    ha_automations = fetch_ha_automations()
    save_ha_automations(ha_automations)

    suggestions: list[dict[str, Any]] = []

    # ── Scene-based suggestions ──
    for scene in scenes:
        scene_yaml = build_scene_automation_yaml(scene)
        overlap = _entities_overlap(scene.get("entities", []), ha_automations)

        suggestion = {
            "id": f"smart_scene_{scene['id']}",
            "title": f"Create \"{scene['name']}\" Scene",
            "description": scene.get("description", ""),
            "confidence": scene.get("confidence", 50),
            "category": "scene",
            "applicable": True,
            "entities": scene.get("entities", []),
            "time_pattern": scene.get("time_pattern", {}),
            "occurrences": scene.get("occurrences", 0),
            "yaml": scene_yaml,
            "scene_yaml": build_scene_yaml(scene),
            "overlap_automation": overlap,
            "source": "scene_detector",
        }
        suggestions.append(suggestion)

    # ── Pattern-based suggestions (from existing suggestions, re-tagged) ──
    if existing_suggestions:
        for sug in existing_suggestions:
            # Skip if already in smart suggestions
            if any(s["id"] == sug.get("id") for s in suggestions):
                continue
            sug_copy = dict(sug)
            sug_copy["source"] = "patterns"
            sug_copy.setdefault("entities", [])
            sug_copy.setdefault("overlap_automation", None)
            suggestions.append(sug_copy)

    # Sort by confidence descending
    suggestions.sort(key=lambda s: -s.get("confidence", 0))

    return suggestions


def save_smart_suggestions(suggestions: list[dict[str, Any]]) -> None:
    """Save merged smart suggestions."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SMART_SUGGESTIONS_PATH, "w") as f:
        json.dump({
            "suggestions": suggestions,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "count": len(suggestions),
            "scene_count": sum(1 for s in suggestions if s.get("source") == "scene_detector"),
            "pattern_count": sum(1 for s in suggestions if s.get("source") == "patterns"),
        }, f, indent=2, default=str)
    log.info("Saved %d smart suggestions to %s", len(suggestions), SMART_SUGGESTIONS_PATH)


def load_smart_suggestions() -> dict[str, Any]:
    """Load cached smart suggestions."""
    try:
        if os.path.exists(SMART_SUGGESTIONS_PATH):
            with open(SMART_SUGGESTIONS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"suggestions": [], "count": 0}
