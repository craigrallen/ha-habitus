"""Cross-domain conflict detection — finds wasteful or contradictory states.

Monitors combinations of entity states that don't make sense together:
- Window open + heating on → wasting energy
- Nobody home + lights on → forgot to turn off
- AC on + heating on → fighting each other
- Door open for >10 min + climate active → losing conditioned air
- Outdoor temp warm + heating on → unnecessary heating
- Outdoor temp cold + windows open → losing heat

These aren't anomalies in the ML sense — they're logical conflicts that
a smart home should catch and suggest fixing.
"""

import datetime
import json
import logging
import os
import sqlite3
from typing import Any

import requests

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
CONFLICTS_PATH = os.path.join(DATA_DIR, "conflicts.json")
HA_DB = "/homeassistant/home-assistant_v2.db"

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))


# ── Conflict Rules ──────────────────────────────────────────────────────────

WINDOW_KEYWORDS = ("window", "fenster", "fönster", "ventana")
DOOR_KEYWORDS = ("door", "dörr", "tür", "puerta", "entry", "entrance")
OPENING_KEYWORDS = WINDOW_KEYWORDS + DOOR_KEYWORDS

HEATING_KEYWORDS = ("heater", "heating", "radiator", "thermostat", "hvac", "boiler")
COOLING_KEYWORDS = ("ac", "air_conditioning", "cooling", "aircon")
CLIMATE_KEYWORDS = HEATING_KEYWORDS + COOLING_KEYWORDS

TEMP_KEYWORDS = ("temperature", "temp", "outdoor_temp", "outside_temp", "weather_temp")

# Comfortable outdoor temp range (°C) — above this, heating is probably unnecessary
WARM_THRESHOLD_C = 18.0
# Below this, windows open is probably losing heat
COLD_THRESHOLD_C = 10.0


def _fetch_current_states() -> dict[str, dict[str, Any]]:
    """Fetch all current entity states from HA REST API."""
    headers = {"Authorization": f"Bearer {HA_TOKEN}"}
    try:
        r = requests.get(f"{HA_URL}/api/states", headers=headers, timeout=10)
        if r.status_code == 200:
            return {s["entity_id"]: s for s in r.json()}
    except Exception as e:
        log.warning("Failed to fetch HA states: %s", e)
    return {}


def _is_on(state: str) -> bool:
    """Check if a state value means 'active/on/open'."""
    return state.lower() in ("on", "open", "heat", "cool", "auto", "heating", "cooling", "home")


def _is_open(state: str) -> bool:
    """Check if a binary sensor state means 'open'."""
    return state.lower() in ("on", "open")


def _matches_keywords(entity_id: str, keywords: tuple[str, ...]) -> bool:
    """Check if entity_id contains any of the keywords."""
    eid_lower = entity_id.lower()
    return any(kw in eid_lower for kw in keywords)


def _get_outdoor_temp(states: dict[str, dict]) -> float | None:
    """Try to find outdoor temperature from available sensors."""
    # Look for explicit outdoor temp sensors
    for eid, s in states.items():
        if not eid.startswith("sensor."):
            continue
        eid_lower = eid.lower()
        if ("outdoor" in eid_lower or "outside" in eid_lower or "weather" in eid_lower) and "temp" in eid_lower:
            try:
                return float(s["state"])
            except (ValueError, TypeError):
                continue

    # Fallback: weather entity
    for eid, s in states.items():
        if eid.startswith("weather."):
            attrs = s.get("attributes", {})
            temp = attrs.get("temperature")
            if temp is not None:
                try:
                    return float(temp)
                except (ValueError, TypeError):
                    pass
    return None


def _get_person_states(states: dict[str, dict]) -> dict[str, str]:
    """Get all person entity states (home/not_home/etc)."""
    return {
        eid: s["state"]
        for eid, s in states.items()
        if eid.startswith("person.")
    }


def detect_conflicts(states: dict[str, dict] | None = None) -> list[dict[str, Any]]:
    """Detect current cross-domain conflicts.

    Returns list of conflict dicts with description, severity, suggestion, and
    optional automation YAML to fix it.
    """
    if states is None:
        states = _fetch_current_states()

    if not states:
        return []

    conflicts: list[dict[str, Any]] = []
    now = datetime.datetime.now(datetime.UTC)

    # Categorise entities by type
    open_windows = []
    open_doors = []
    active_heating = []
    active_cooling = []
    lights_on = []
    media_on = []
    switches_on = []

    for eid, s in states.items():
        state_val = s.get("state", "")
        eid_lower = eid.lower()

        if eid.startswith("binary_sensor.") and _is_open(state_val):
            if _matches_keywords(eid, WINDOW_KEYWORDS):
                open_windows.append(eid)
            elif _matches_keywords(eid, DOOR_KEYWORDS):
                open_doors.append(eid)

        if eid.startswith("climate.") and _is_on(state_val):
            if state_val.lower() in ("heat", "heating", "auto") or _matches_keywords(eid, HEATING_KEYWORDS):
                active_heating.append(eid)
            if state_val.lower() in ("cool", "cooling") or _matches_keywords(eid, COOLING_KEYWORDS):
                active_cooling.append(eid)

        if eid.startswith("light.") and state_val.lower() == "on":
            lights_on.append(eid)

        if eid.startswith("media_player.") and state_val.lower() in ("playing", "on", "paused"):
            media_on.append(eid)

        if eid.startswith("switch.") and state_val.lower() == "on" and _matches_keywords(eid, HEATING_KEYWORDS):
            active_heating.append(eid)

    outdoor_temp = _get_outdoor_temp(states)
    persons = _get_person_states(states)
    anyone_home = any(v.lower() == "home" for v in persons.values())

    # ── Rule 1: Window open + heating on ──
    if open_windows and active_heating:
        window_names = [w.split(".")[-1].replace("_", " ").title() for w in open_windows]
        heater_names = [h.split(".")[-1].replace("_", " ").title() for h in active_heating]
        yaml = _build_conflict_yaml(
            alias="Turn off heating when window opens",
            trigger_entity=open_windows[0],
            trigger_state="on",
            action_entities=active_heating,
            action="turn_off",
        )
        conflicts.append({
            "id": "window_heating",
            "severity": "high",
            "icon": "🪟🔥",
            "title": "Window open while heating is on",
            "description": f"{', '.join(window_names)} open — but {', '.join(heater_names)} still heating. Wasting energy.",
            "suggestion": "Turn off heating when windows are open, or close the windows.",
            "entities": open_windows + active_heating,
            "yaml": yaml,
            "est_waste_w": 500,  # rough estimate
        })

    # ── Rule 2: Window open + AC on ──
    if open_windows and active_cooling:
        yaml = _build_conflict_yaml(
            alias="Turn off AC when window opens",
            trigger_entity=open_windows[0],
            trigger_state="on",
            action_entities=active_cooling,
            action="turn_off",
        )
        conflicts.append({
            "id": "window_cooling",
            "severity": "high",
            "icon": "🪟❄️",
            "title": "Window open while AC is running",
            "description": "Cooling the outdoors! Window open with air conditioning active.",
            "suggestion": "Turn off AC when windows are open.",
            "entities": open_windows + active_cooling,
            "yaml": yaml,
            "est_waste_w": 800,
        })

    # ── Rule 3: Heating + cooling simultaneously ──
    if active_heating and active_cooling:
        conflicts.append({
            "id": "heat_cool_fight",
            "severity": "critical",
            "icon": "🔥❄️",
            "title": "Heating and cooling running simultaneously",
            "description": "Heating and AC are fighting each other. Pick one!",
            "suggestion": "Disable either heating or cooling.",
            "entities": active_heating + active_cooling,
            "yaml": "",
            "est_waste_w": 1500,
        })

    # ── Rule 4: Nobody home + lights/media on ──
    if persons and not anyone_home and (lights_on or media_on):
        wasted = lights_on + media_on
        wasted_names = [e.split(".")[-1].replace("_", " ").title() for e in wasted[:5]]
        yaml = _build_conflict_yaml(
            alias="Turn off everything when nobody home",
            trigger_entity=list(persons.keys())[0] if persons else "person.unknown",
            trigger_state="not_home",
            action_entities=wasted,
            action="turn_off",
        )
        conflicts.append({
            "id": "nobody_home_lights",
            "severity": "medium",
            "icon": "🏠💡",
            "title": f"Nobody home but {len(wasted)} device(s) still on",
            "description": f"Everyone is away but still running: {', '.join(wasted_names)}{'...' if len(wasted) > 5 else ''}",
            "suggestion": "Create an 'away mode' automation to turn off non-essential devices.",
            "entities": list(persons.keys()) + wasted,
            "yaml": yaml,
            "est_waste_w": len(wasted) * 30,
        })

    # ── Rule 5: Warm outside + heating on ──
    if outdoor_temp is not None and outdoor_temp > WARM_THRESHOLD_C and active_heating:
        conflicts.append({
            "id": "warm_outside_heating",
            "severity": "medium",
            "icon": "☀️🔥",
            "title": f"It's {outdoor_temp:.0f}°C outside — heating still on",
            "description": f"Outdoor temperature is {outdoor_temp:.1f}°C (above {WARM_THRESHOLD_C}°C) but heating is active.",
            "suggestion": "Turn off heating or lower the setpoint. Open a window instead!",
            "entities": active_heating,
            "yaml": "",
            "est_waste_w": 400,
        })

    # ── Rule 6: Cold outside + windows open ──
    if outdoor_temp is not None and outdoor_temp < COLD_THRESHOLD_C and open_windows:
        conflicts.append({
            "id": "cold_outside_windows",
            "severity": "low",
            "icon": "🥶🪟",
            "title": f"It's {outdoor_temp:.0f}°C outside — windows still open",
            "description": f"Outdoor temperature is {outdoor_temp:.1f}°C but windows are open. Losing heat.",
            "suggestion": "Close windows to conserve heat.",
            "entities": open_windows,
            "yaml": "",
            "est_waste_w": 200,
        })

    # ── Rule 7: Door open >10 min with climate active ──
    if open_doors and (active_heating or active_cooling):
        # Check if door has been open for a while (check last_changed)
        for door_eid in open_doors:
            s = states.get(door_eid, {})
            last_changed = s.get("last_changed", "")
            if last_changed:
                try:
                    changed_dt = datetime.datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
                    open_minutes = (now - changed_dt).total_seconds() / 60
                    if open_minutes > 10:
                        door_name = door_eid.split(".")[-1].replace("_", " ").title()
                        conflicts.append({
                            "id": f"door_open_long_{door_eid.split('.')[-1]}",
                            "severity": "medium",
                            "icon": "🚪🌡️",
                            "title": f"{door_name} open for {int(open_minutes)} minutes",
                            "description": f"{door_name} has been open {int(open_minutes)} min while climate control is active.",
                            "suggestion": f"Close {door_name} or pause climate control.",
                            "entities": [door_eid] + active_heating + active_cooling,
                            "yaml": "",
                            "est_waste_w": 300,
                        })
                except (ValueError, TypeError):
                    pass

    return conflicts


def _build_conflict_yaml(
    alias: str,
    trigger_entity: str,
    trigger_state: str,
    action_entities: list[str],
    action: str,
) -> str:
    """Build automation YAML to resolve a conflict."""
    actions = ""
    for eid in action_entities:
        domain = eid.split(".")[0]
        service = f"{domain}.{action}"
        actions += f"""
    - service: {service}
      target:
        entity_id: {eid}"""

    return f"""automation:
  alias: "Habitus Conflict Fix — {alias}"
  description: "Auto-generated by Habitus conflict detector"
  trigger:
    - platform: state
      entity_id: {trigger_entity}
      to: "{trigger_state}"
      for: "00:02:00"
  action:{actions}
    - service: notify.notify
      data:
        title: "🏠 Habitus"
        message: "{alias}"
"""


def save_conflicts(conflicts: list[dict[str, Any]]) -> None:
    """Save detected conflicts."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFLICTS_PATH, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "count": len(conflicts),
            "total_est_waste_w": sum(c.get("est_waste_w", 0) for c in conflicts),
            "conflicts": conflicts,
        }, f, indent=2, default=str)
    if conflicts:
        log.info("Conflict detector: %d active conflict(s), ~%dW estimated waste",
                 len(conflicts), sum(c.get("est_waste_w", 0) for c in conflicts))


def load_conflicts() -> dict[str, Any]:
    """Load cached conflicts."""
    try:
        if os.path.exists(CONFLICTS_PATH):
            with open(CONFLICTS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"conflicts": [], "count": 0, "total_est_waste_w": 0}
