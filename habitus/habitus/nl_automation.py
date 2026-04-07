"""Natural Language Automation Creator — local pattern matching, no LLM.

Accepts a natural language description and generates HA automation YAML.
Patterns: time trigger, state trigger, condition-based, notification.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger("habitus")

# ── Time pattern matchers ──────────────────────────────────────────────────

# Match: "at 7am", "at 07:30", "at 7:30 am", "every morning at 7"
_TIME_PATTERNS = [
    re.compile(r"at\s+(\d{1,2}):(\d{2})\s*(am|pm)?", re.IGNORECASE),
    re.compile(r"at\s+(\d{1,2})\s*(am|pm)", re.IGNORECASE),
    re.compile(r"every\s+(?:morning|day|night|evening)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE),
    re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.IGNORECASE),
]

# Named times
_NAMED_TIMES = {
    "morning": "07:00:00",
    "sunrise": None,  # use sun trigger
    "sunset": None,  # use sun trigger
    "noon": "12:00:00",
    "midnight": "00:00:00",
    "evening": "18:00:00",
    "night": "22:00:00",
    "bedtime": "22:30:00",
}

# ── State/entity patterns ──────────────────────────────────────────────────

_ENTITY_KEYWORDS = {
    "light": ["light", "lamp", "bulb", "lights"],
    "switch": ["switch", "plug", "outlet", "socket"],
    "climate": ["heater", "heating", "thermostat", "ac", "air conditioning", "climate", "hvac"],
    "media_player": ["tv", "television", "speaker", "music", "media", "netflix"],
    "fan": ["fan", "extractor"],
    "cover": ["blind", "shade", "curtain", "shutter", "cover", "roller"],
    "lock": ["lock", "door lock"],
    "input_boolean": ["mode", "flag", "toggle"],
    "binary_sensor": ["door", "window", "motion", "sensor", "presence"],
    "person": ["person", "me", "craig", "owner", "everyone", "nobody", "someone"],
}

_TRIGGER_WORDS = {
    "state_on": ["turns on", "turns on", "switched on", "activated", "opens", "opened", "enters", "arrives", "comes home"],
    "state_off": ["turns off", "switched off", "deactivated", "closes", "closed", "leaves", "goes away"],
    "motion": ["motion detected", "motion", "movement detected", "someone moves"],
    "no_motion": ["no motion", "no movement", "nobody moves", "empty"],
    "time": ["at", "every", "daily", "morning", "evening", "night", "sunset", "sunrise", "noon", "midnight", "bedtime"],
    "temperature_above": ["temperature above", "temp above", "hotter than", "warmer than", "exceeds"],
    "temperature_below": ["temperature below", "temp below", "colder than", "cooler than"],
    "nobody_home": ["nobody home", "no one home", "everyone leaves", "away", "not home", "empty house"],
    "someone_home": ["someone home", "arrives home", "comes home", "returns home"],
}

_ACTION_WORDS = {
    "turn_on": ["turn on", "switch on", "enable", "activate", "start", "open"],
    "turn_off": ["turn off", "switch off", "disable", "deactivate", "stop", "close", "shut"],
    "toggle": ["toggle", "switch"],
    "notify": ["notify", "alert", "send notification", "message", "remind", "tell me"],
    "set_temperature": ["set temperature", "set temp", "heat to", "cool to", "set to"],
    "lock": ["lock", "secure"],
    "unlock": ["unlock"],
}

_CONDITION_WORDS = {
    "only_if": ["only if", "only when", "if", "when", "provided that"],
    "unless": ["unless", "except when", "not when", "but not when"],
}


def _parse_time(text: str) -> str | None:
    """Extract time string from text. Returns HH:MM:SS or None."""
    # Check named times first
    lower = text.lower()
    for name, time_str in _NAMED_TIMES.items():
        if name in lower and time_str:
            return time_str

    # Pattern: HH:MM optional am/pm
    m = re.search(r'\bat\s+(\d{1,2}):(\d{2})\s*(am|pm)?', lower)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{max(0,min(23,hour)):02d}:{max(0,min(59,minute)):02d}:00"

    # Pattern: at Xam or at Xpm
    m = re.search(r'\bat\s+(\d{1,2})\s*(am|pm)\b', lower)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{max(0,min(23,hour)):02d}:00:00"

    # Pattern: every ... at Xam/pm or at X:XX
    m = re.search(r'\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', lower)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{max(0,min(23,hour)):02d}:{max(0,min(59,minute)):02d}:00"

    return None


def _is_sun_trigger(text: str) -> str | None:
    """Check if text mentions sunrise/sunset. Returns 'sunrise', 'sunset', or None."""
    lower = text.lower()
    if "sunset" in lower:
        return "sunset"
    if "sunrise" in lower:
        return "sunrise"
    return None


def _extract_entities(text: str) -> list[dict[str, str]]:
    """Extract entity references from text.

    Returns list of {domain, keyword, entity_id_guess} dicts.
    """
    lower = text.lower()
    found = []

    for domain, keywords in _ENTITY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                # Try to extract specific entity name
                # e.g. "living room light" → light.living_room
                # Find the keyword in context
                idx = lower.find(kw)
                context_start = max(0, idx - 30)
                context_end = min(len(lower), idx + len(kw) + 30)
                context = lower[context_start:context_end]

                # Build a guess at entity_id
                entity_id_guess = _guess_entity_id(domain, kw, context, text)

                found.append({
                    "domain": domain,
                    "keyword": kw,
                    "entity_id": entity_id_guess,
                })
                break  # one match per domain

    return found


def _guess_entity_id(domain: str, keyword: str, context: str, full_text: str) -> str:
    """Guess a plausible entity_id from context."""
    # Look for room qualifiers
    room_words = [
        "living room", "bedroom", "kitchen", "bathroom", "hallway",
        "office", "garage", "garden", "front", "back", "master",
    ]
    for room in room_words:
        if room in context:
            room_slug = room.replace(" ", "_")
            return f"{domain}.{room_slug}_{keyword.replace(' ', '_')}"

    # Check for "all" qualifier
    if "all" in context:
        return f"{domain}.all"

    # Default to domain + keyword
    return f"{domain}.{keyword.replace(' ', '_')}"


def _extract_action(text: str) -> dict[str, str]:
    """Extract the action from text."""
    lower = text.lower()

    # Check notify first (higher priority to avoid "open" clash)
    priority_order = ["notify", "set_temperature", "lock", "unlock", "toggle", "turn_off", "turn_on"]
    all_actions = {**_ACTION_WORDS}

    for action in priority_order:
        phrases = all_actions.get(action, [])
        for phrase in phrases:
            if phrase in lower:
                return {"action": action, "phrase": phrase}

    return {"action": "turn_on", "phrase": ""}


def _extract_trigger_type(text: str) -> dict[str, Any]:
    """Determine the trigger type from text."""
    lower = text.lower()

    # Check for sun trigger
    sun = _is_sun_trigger(lower)
    if sun:
        return {"type": "sun", "event": sun}

    # Check for time trigger
    time_str = _parse_time(text)
    if time_str:
        return {"type": "time", "at": time_str}

    # Check for state triggers
    for trigger_type, phrases in _TRIGGER_WORDS.items():
        if trigger_type.startswith("state_") or trigger_type in ("motion", "no_motion", "nobody_home", "someone_home"):
            for phrase in phrases:
                if phrase in lower:
                    return {"type": "state", "kind": trigger_type, "phrase": phrase}

    # Check for temperature
    for trigger_type in ("temperature_above", "temperature_below"):
        for phrase in _TRIGGER_WORDS[trigger_type]:
            if phrase in lower:
                # Try to extract the temperature value
                temp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°?c|degrees|°)?", lower[lower.find(phrase) + len(phrase):])
                temp_val = float(temp_match.group(1)) if temp_match else 20.0
                return {
                    "type": "numeric_state",
                    "kind": trigger_type,
                    "value": temp_val,
                }

    return {"type": "state", "kind": "state_on", "phrase": ""}


def _extract_condition(text: str) -> dict[str, Any] | None:
    """Extract condition from text."""
    lower = text.lower()

    if "nobody home" in lower or "no one home" in lower or "everyone away" in lower:
        return {"type": "state", "entity_id": "group.all_persons", "state": "not_home"}

    if "someone home" in lower or "anyone home" in lower:
        return {"type": "state", "entity_id": "group.all_persons", "state": "home"}

    if "night" in lower and "condition" not in lower:
        return {"type": "time", "after": "22:00:00", "before": "07:00:00"}

    if "daytime" in lower or "during day" in lower:
        return {"type": "sun", "condition": "above_horizon"}

    return None


def _build_yaml(intent: dict[str, Any]) -> str:
    """Build HA automation YAML from parsed intent."""
    trigger = intent.get("trigger", {})
    action = intent.get("action", {})
    entities = intent.get("entities", [])
    condition = intent.get("condition")
    alias = intent.get("alias", "Habitus NL Automation")

    # Build trigger YAML
    trigger_yaml = ""
    trigger_type = trigger.get("type", "state")

    if trigger_type == "time":
        trigger_yaml = f"""  - platform: time
    at: "{trigger.get('at', '07:00:00')}" """
    elif trigger_type == "sun":
        trigger_yaml = f"""  - platform: sun
    event: {trigger.get('event', 'sunset')} """
    elif trigger_type == "numeric_state":
        kind = trigger.get("kind", "temperature_above")
        temp_entity = "sensor.indoor_temperature"
        above_below = "above" if "above" in kind else "below"
        trigger_yaml = f"""  - platform: numeric_state
    entity_id: {temp_entity}
    {above_below}: {trigger.get('value', 20)} """
    else:
        # State trigger
        kind = trigger.get("kind", "state_on")
        trigger_entity = entities[0]["entity_id"] if entities else "binary_sensor.trigger"
        to_state = "on" if kind in ("state_on", "motion", "someone_home") else "off"
        trigger_yaml = f"""  - platform: state
    entity_id: {trigger_entity}
    to: "{to_state}" """

    # Build condition YAML
    condition_yaml = ""
    if condition:
        ctype = condition.get("type", "state")
        if ctype == "state":
            condition_yaml = f"""
condition:
  - condition: state
    entity_id: {condition.get('entity_id', 'group.all_persons')}
    state: "{condition.get('state', 'home')}" """
        elif ctype == "time":
            condition_yaml = f"""
condition:
  - condition: time
    after: "{condition.get('after', '22:00:00')}"
    before: "{condition.get('before', '07:00:00')}" """
        elif ctype == "sun":
            condition_yaml = f"""
condition:
  - condition: sun
    {condition.get('condition', 'above_horizon')}: true """

    # Build action YAML
    action_type = action.get("action", "turn_on")
    action_entities = entities[1:] if len(entities) > 1 else entities

    if not action_entities:
        action_entities = [{"entity_id": "light.living_room"}]

    action_yaml = ""
    if action_type == "notify":
        action_yaml = """
  - service: notify.notify
    data:
      title: "Habitus"
      message: "Automation triggered" """
    elif action_type == "set_temperature":
        action_yaml = f"""
  - service: climate.set_temperature
    target:
      entity_id: {action_entities[0]['entity_id']}
    data:
      temperature: 21 """
    elif action_type in ("lock", "unlock"):
        action_yaml = f"""
  - service: lock.{action_type}
    target:
      entity_id: {action_entities[0]['entity_id']} """
    else:
        entity_list = "\n          - ".join(e["entity_id"] for e in action_entities)
        service = f"{action_entities[0]['entity_id'].split('.')[0]}.{action_type}"
        action_yaml = f"""
  - service: {service}
    target:
      entity_id:
        - {entity_list} """

    yaml_str = f"""automation:
  alias: "{alias}"
  description: "Generated by Habitus NL Automation Creator"
  trigger:
{trigger_yaml}
{condition_yaml.strip()}
  action:{action_yaml}
  mode: single
"""
    return yaml_str.strip()


def parse_intent(text: str) -> dict[str, Any]:
    """Parse a natural language automation description into structured intent.

    Args:
        text: Natural language description.

    Returns:
        Dict with trigger, action, entities, condition, alias, confidence,
        clarification_needed, generated_yaml.
    """
    if not text or not text.strip():
        return {
            "trigger": {},
            "action": {},
            "entities": [],
            "condition": None,
            "alias": "Unnamed Automation",
            "confidence": 0.0,
            "clarification_needed": ["Please describe what should trigger the automation and what it should do."],
            "generated_yaml": "",
        }

    lower = text.strip().lower()
    clarifications = []

    # Extract components
    trigger = _extract_trigger_type(text)
    action = _extract_action(text)
    entities = _extract_entities(text)
    condition = _extract_condition(text)

    # Build alias from text
    alias = text.strip()[:80]
    alias = re.sub(r"[^a-zA-Z0-9 _\-]", "", alias).strip()
    if not alias:
        alias = "Habitus NL Automation"

    # Compute confidence
    confidence = 0.5  # base confidence
    if trigger.get("type") in ("time", "sun"):
        confidence += 0.2  # time/sun triggers are clear
    if entities:
        confidence += 0.15
    if action.get("action") != "turn_on" or action.get("phrase"):
        confidence += 0.1
    if condition:
        confidence += 0.05

    confidence = min(0.95, confidence)

    # Identify what needs clarification
    if not entities:
        clarifications.append("Which entity/device should this automation control?")
    if trigger.get("type") == "state" and not entities:
        clarifications.append("Which entity should trigger this automation?")
    if "time" in lower and not trigger.get("at") and trigger.get("type") != "sun":
        clarifications.append("At what specific time should this run?")

    # Build full intent
    intent: dict[str, Any] = {
        "trigger": trigger,
        "action": action,
        "entities": entities,
        "condition": condition,
        "alias": alias,
        "confidence": round(confidence, 2),
        "clarification_needed": clarifications,
    }

    # Generate YAML
    intent["generated_yaml"] = _build_yaml(intent)

    return intent
