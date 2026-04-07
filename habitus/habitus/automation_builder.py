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


def build_motion_automation_yaml(
    alias: str,
    description: str,
    trigger_entity: str,
    action_entities: list[str],
    hour_start: int | None = None,
    hour_end: int | None = None,
) -> str:
    """Generate motion/presence-triggered automation YAML.

    Creates automations like: "When motion detected in living room after 18:00,
    turn on living room lights and TV."
    """
    condition = ""
    if hour_start is not None and hour_end is not None:
        condition = f"""
  condition:
    - condition: time
      after: "{hour_start:02d}:00:00"
      before: "{hour_end:02d}:00:00\""""

    actions = ""
    for eid in action_entities:
        domain = eid.split(".")[0]
        service = f"{domain}.turn_on"
        actions += f"""
    - service: {service}
      target:
        entity_id: {eid}"""

    return f"""automation:
  alias: "{alias}"
  description: "{description}"
  trigger:
    - platform: state
      entity_id: {trigger_entity}
      to: "on"{condition}
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

    # ── Motion/presence-triggered suggestions ──
    # If a scene contains binary_sensor (motion/presence) co-occurring with
    # lights/switches/media, generate a motion-triggered automation
    for scene in scenes:
        trigger_entities = [e for e in scene.get("entities", [])
                          if (e.startswith("binary_sensor.") and
                          any(kw in e.lower() for kw in ("motion", "occupancy", "presence", "pir", "door", "window", "contact", "opening")))
                          or (e.startswith("climate.") and
                          any(kw in e.lower() for kw in ("heater", "radiator", "thermostat", "heating", "hvac")))]
        action_entities = [e for e in scene.get("entities", [])
                         if not e.startswith(("binary_sensor.", "person.", "device_tracker."))]
        if trigger_entities and action_entities:
            tp = scene.get("time_pattern", {})
            for trigger in trigger_entities:
                trigger_name = trigger.split(".")[1].replace("_", " ").title()
                alias = f"Habitus — {trigger_name} → {scene['name']}"
                yaml = build_motion_automation_yaml(
                    alias=alias,
                    description=f"When {trigger_name} detects activity, activate {scene['name']}",
                    trigger_entity=trigger,
                    action_entities=action_entities,
                    hour_start=tp.get("peak_hour", 18) - 2 if tp.get("peak_hour") else None,
                    hour_end=tp.get("peak_hour", 22) + 2 if tp.get("peak_hour") else None,
                )
                overlap = _entities_overlap(action_entities, ha_automations)
                suggestions.append({
                    "id": f"smart_motion_{scene['id']}_{trigger.split('.')[-1]}",
                    "title": f"When {trigger_name}: activate {scene['name']}",
                    "description": f"Motion/presence trigger → lights and devices you typically use together",
                    "confidence": min(scene.get("confidence", 50) + 10, 100),
                    "category": "motion",
                    "applicable": True,
                    "entities": action_entities,
                    "trigger": trigger,
                    "time_pattern": tp,
                    "yaml": yaml,
                    "overlap_automation": overlap,
                    "source": "scene_detector",
                })

    # ── Heater/climate as presence indicator ──
    # When someone adjusts a heater/thermostat, it implies they're in that room.
    # Suggest: "When bedroom heater turns on → turn on bedroom lights"
    for scene in scenes:
        climate_triggers = [e for e in scene.get("entities", [])
                           if e.startswith("climate.") and
                           any(kw in e.lower() for kw in ("heater", "radiator", "thermostat", "heating", "hvac"))]
        action_entities = [e for e in scene.get("entities", [])
                         if e.startswith(("light.", "switch.", "media_player.", "fan."))
                         and e not in climate_triggers]
        if climate_triggers and action_entities:
            tp = scene.get("time_pattern", {})
            for clim in climate_triggers:
                clim_name = clim.split(".")[1].replace("_", " ").title()
                alias = f"Habitus — {clim_name} On → {scene['name']}"
                yaml = build_motion_automation_yaml(
                    alias=alias,
                    description=f"When {clim_name} is adjusted (someone in room), activate {scene['name']}",
                    trigger_entity=clim,
                    action_entities=action_entities,
                    hour_start=tp.get("peak_hour", 16) - 2 if tp.get("peak_hour") else None,
                    hour_end=tp.get("peak_hour", 23) + 1 if tp.get("peak_hour") else None,
                )
                overlap = _entities_overlap(action_entities, ha_automations)
                suggestions.append({
                    "id": f"smart_climate_{scene['id']}_{clim.split('.')[-1]}",
                    "title": f"When {clim_name} activates: set up {scene['name']}",
                    "description": f"Heater/thermostat change implies presence — auto-activate room scene",
                    "confidence": min(scene.get("confidence", 50) + 5, 100),
                    "category": "presence",
                    "applicable": True,
                    "entities": action_entities,
                    "trigger": clim,
                    "time_pattern": tp,
                    "yaml": yaml,
                    "overlap_automation": overlap,
                    "source": "scene_detector",
                })

    # ── Door/window-triggered suggestions ──
    for scene in scenes:
        door_entities = [e for e in scene.get("entities", [])
                        if e.startswith("binary_sensor.") and
                        any(kw in e.lower() for kw in ("door", "window", "contact", "opening"))]
        action_entities = [e for e in scene.get("entities", [])
                         if not e.startswith(("binary_sensor.", "person.", "device_tracker."))]
        if door_entities and action_entities:
            tp = scene.get("time_pattern", {})
            for door in door_entities:
                door_name = door.split(".")[1].replace("_", " ").title()
                alias = f"Habitus — {door_name} Opens → {scene['name']}"
                yaml = build_door_automation_yaml(
                    alias=alias,
                    description=f"When {door_name} opens, activate {scene['name']}",
                    door_entity=door,
                    action_entities=action_entities,
                    hour_start=tp.get("peak_hour", 18) - 2 if tp.get("peak_hour") else None,
                    hour_end=tp.get("peak_hour", 22) + 2 if tp.get("peak_hour") else None,
                )
                overlap = _entities_overlap(action_entities, ha_automations)
                suggestions.append({
                    "id": f"smart_door_{scene['id']}_{door.split('.')[-1]}",
                    "title": f"When {door_name} opens: activate {scene['name']}",
                    "description": f"Door/window trigger → lights and devices you typically use after opening",
                    "confidence": min(scene.get("confidence", 50) + 5, 100),
                    "category": "door",
                    "applicable": True,
                    "entities": action_entities,
                    "trigger": door,
                    "time_pattern": tp,
                    "yaml": yaml,
                    "overlap_automation": overlap,
                    "source": "scene_detector",
                })

    # ── Person home/away suggestions ──
    # Find person entities that co-occur with many lights being turned off/on
    person_entities = [e for s in scenes for e in s.get("entities", [])
                      if e.startswith("person.")]
    if person_entities:
        # Collect all action entities across all scenes
        all_action_entities = list({
            e for s in scenes for e in s.get("entities", [])
            if e.startswith(("light.", "switch.", "media_player.", "fan.", "climate."))
        })
        if all_action_entities:
            for person in set(person_entities):
                person_name = person.split(".")[1].replace("_", " ").title()
                # "Leaving home" automation
                leave_yaml = build_presence_automation_yaml(
                    alias=f"Habitus — {person_name} Leaves → All Off",
                    description=f"Turn off lights and devices when {person_name} leaves home",
                    person_entity=person,
                    action_entities=all_action_entities,
                    trigger_state="not_home",
                    action="turn_off",
                )
                suggestions.append({
                    "id": f"smart_presence_leave_{person.split('.')[-1]}",
                    "title": f"When {person_name} leaves home: turn everything off",
                    "description": f"Automatically turn off lights and devices when everyone leaves",
                    "confidence": 70,
                    "category": "presence",
                    "applicable": True,
                    "entities": all_action_entities,
                    "trigger": person,
                    "yaml": leave_yaml,
                    "overlap_automation": _entities_overlap(all_action_entities, ha_automations),
                    "source": "scene_detector",
                })
                # "Arriving home" automation — turn on most-used scene
                if scenes:
                    top_scene = scenes[0]
                    arrive_entities = [e for e in top_scene.get("entities", [])
                                      if not e.startswith(("binary_sensor.", "person.", "device_tracker."))]
                    if arrive_entities:
                        arrive_yaml = build_presence_automation_yaml(
                            alias=f"Habitus — {person_name} Arrives → Welcome",
                            description=f"Welcome {person_name} home with their favourite scene",
                            person_entity=person,
                            action_entities=arrive_entities,
                            trigger_state="home",
                            action="turn_on",
                        )
                        suggestions.append({
                            "id": f"smart_presence_arrive_{person.split('.')[-1]}",
                            "title": f"When {person_name} arrives home: welcome scene",
                            "description": f"Turn on {top_scene['name']} when {person_name} comes home",
                            "confidence": 65,
                            "category": "presence",
                            "applicable": True,
                            "entities": arrive_entities,
                            "trigger": person,
                            "yaml": arrive_yaml,
                            "overlap_automation": _entities_overlap(arrive_entities, ha_automations),
                            "source": "scene_detector",
                        })

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


def build_presence_automation_yaml(
    alias: str,
    description: str,
    person_entity: str,
    action_entities: list[str],
    trigger_state: str = "not_home",
    action: str = "turn_off",
) -> str:
    """Generate person home/away automation YAML.

    E.g. "When Craig leaves home, turn off all lights."
    """
    actions = ""
    for eid in action_entities:
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
    - platform: state
      entity_id: {person_entity}
      to: "{trigger_state}"
      for: "00:05:00"
  action:{actions}"""


def build_door_automation_yaml(
    alias: str,
    description: str,
    door_entity: str,
    action_entities: list[str],
    hour_start: int | None = None,
    hour_end: int | None = None,
) -> str:
    """Generate door/window-triggered automation YAML.

    E.g. "When front door opens after 18:00, turn on hallway lights."
    """
    condition = ""
    if hour_start is not None and hour_end is not None:
        condition = f"""
  condition:
    - condition: time
      after: "{hour_start:02d}:00:00"
      before: "{hour_end:02d}:00:00\""""

    actions = ""
    for eid in action_entities:
        domain = eid.split(".")[0]
        service = f"{domain}.turn_on"
        actions += f"""
    - service: {service}
      target:
        entity_id: {eid}"""

    return f"""automation:
  alias: "{alias}"
  description: "{description}"
  trigger:
    - platform: state
      entity_id: {door_entity}
      to: "on"
      from: "off"{condition}
  action:{actions}"""
