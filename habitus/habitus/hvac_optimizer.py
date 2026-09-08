"""Occupancy-aware HVAC pre-conditioning suggestions.

Uses discovered daily-routine patterns and motion/presence data to
predict when rooms will be occupied, then computes the optimal HVAC
start time so the target temperature is reached just before arrival.
"""

import json
import logging
import math
import os
from typing import Any

import yaml

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
HVAC_SCHEDULE_PATH = os.path.join(DATA_DIR, "hvac_schedule.json")
PATTERNS_PATH = os.path.join(DATA_DIR, "patterns.json")


def _load_patterns() -> dict[str, Any]:
    """Load discovered routine patterns from disk.

    Returns:
        Patterns dict, or empty dict if unavailable.
    """
    if not os.path.exists(PATTERNS_PATH):
        return {}
    try:
        with open(PATTERNS_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load patterns.json: %s", exc)
        return {}


def predict_arrival(patterns: dict, current_hour: int) -> dict:
    """Predict when the user will next arrive or become active.

    Scans the ``daily_routine`` section of discovered patterns for the
    next activity window after *current_hour*.

    Args:
        patterns: Parsed patterns.json content.
        current_hour: Current hour of day (0-23).

    Returns:
        Dict with ``predicted_hour``, ``confidence``, and ``source``.
        Returns a low-confidence fallback if no pattern is found.
    """
    daily = patterns.get("daily_routine", [])
    if not daily:
        # No patterns — fall back to a generic morning start
        return {"predicted_hour": 7, "confidence": 0.1, "source": "fallback"}

    # Find the next activity block after current_hour
    best: dict[str, Any] | None = None
    for block in daily:
        hour = block.get("hour", 0)
        activity = block.get("activity_level", 0.0)
        # Look for transition from low to high activity
        if (
            hour > current_hour
            and activity > 0.5
            and (best is None or hour < best.get("predicted_hour", 24))
        ):
            best = {
                "predicted_hour": hour,
                "confidence": min(1.0, activity),
                "source": "daily_routine",
            }

    if best is None:
        # Wrap around to next day — find first activity block
        for block in sorted(daily, key=lambda b: b.get("hour", 0)):
            if block.get("activity_level", 0.0) > 0.5:
                best = {
                    "predicted_hour": block["hour"],
                    "confidence": min(1.0, block["activity_level"]) * 0.8,
                    "source": "daily_routine_next_day",
                }
                break

    if best is None:
        return {"predicted_hour": 7, "confidence": 0.1, "source": "fallback"}

    return best


def suggest_preconditioning(
    current_temp: float,
    target_temp: float,
    arrival_hour: int,
    heat_rate_deg_per_hour: float = 1.5,
) -> dict:
    """Compute when to start HVAC so the target temp is met by arrival.

    Args:
        current_temp: Current indoor temperature in the climate entity's
            native unit.
        target_temp: Desired temperature at arrival time.
        arrival_hour: Expected arrival hour (0-23).
        heat_rate_deg_per_hour: How fast the HVAC changes temperature,
            in degrees per hour.  Positive for both heating and cooling.

    Returns:
        Dict with ``start_hour``, ``minutes_before_arrival``,
        ``estimated_energy_kwh``, and ``automation_yaml``.
    """
    temp_delta = abs(target_temp - current_temp)
    if temp_delta < 0.5:
        return {
            "start_hour": arrival_hour,
            "minutes_before_arrival": 0,
            "estimated_energy_kwh": 0.0,
            "automation_yaml": "",
            "note": "Already at target temperature.",
        }

    hours_needed = temp_delta / max(heat_rate_deg_per_hour, 0.1)
    minutes_needed = math.ceil(hours_needed * 60)
    start_hour_raw = arrival_hour - hours_needed
    start_hour = int(start_hour_raw) % 24
    start_minute = int((start_hour_raw - int(start_hour_raw)) * 60) % 60

    # Rough energy estimate: ~3 kW average HVAC draw
    avg_hvac_kw = 3.0
    estimated_kwh = round(hours_needed * avg_hvac_kw, 2)

    mode = "heat" if target_temp > current_temp else "cool"
    time_str = f"{start_hour:02d}:{abs(start_minute):02d}:00"

    automation = {
        "alias": f"Habitus HVAC Pre-{mode}",
        "description": (
            f"Start {mode}ing {minutes_needed} min before predicted arrival "
            f"to reach {target_temp} degrees."
        ),
        "trigger": [{"platform": "time", "at": time_str}],
        "action": [
            {
                "service": "climate.set_hvac_mode",
                "target": {"entity_id": "climate.thermostat"},
                "data": {"hvac_mode": mode},
            },
            {
                "service": "climate.set_temperature",
                "target": {"entity_id": "climate.thermostat"},
                "data": {"temperature": target_temp},
            },
        ],
    }
    automation_yaml = yaml.dump([automation], default_flow_style=False, sort_keys=False)

    result = {
        "start_hour": start_hour,
        "minutes_before_arrival": minutes_needed,
        "estimated_energy_kwh": estimated_kwh,
        "automation_yaml": automation_yaml,
    }

    # Persist schedule
    atomic_write(HVAC_SCHEDULE_PATH, result)
    log.info(
        "HVAC pre-conditioning: start at %s, %d min before arrival (mode=%s)",
        time_str,
        minutes_needed,
        mode,
    )
    return result


def get_room_schedule(features_hourly: dict) -> dict:
    """Determine which rooms are occupied at which hours.

    Inspects hourly feature data for motion and presence sensors to
    build a per-room, per-hour occupancy map.

    Args:
        features_hourly: Dict mapping hour (int or str) to a dict of
            entity_id -> value pairs.  Motion/presence values > 0
            indicate occupancy.

    Returns:
        Dict mapping room name to a list of occupied hours, e.g.
        ``{"living_room": [7, 8, 18, 19, 20]}``.
    """
    room_hours: dict[str, list[int]] = {}

    for hour_key, entities in features_hourly.items():
        try:
            hour = int(hour_key)
        except (ValueError, TypeError):
            continue

        for entity_id, value in entities.items():
            if not isinstance(entity_id, str):
                continue
            # Only look at motion and occupancy sensors
            is_motion = "motion" in entity_id.lower() or "occupancy" in entity_id.lower()
            is_presence = "presence" in entity_id.lower()
            if not (is_motion or is_presence):
                continue

            try:
                val = float(value)
            except (ValueError, TypeError):
                continue

            if val > 0:
                # Extract room name from entity_id
                room = _extract_room(entity_id)
                if room not in room_hours:
                    room_hours[room] = []
                if hour not in room_hours[room]:
                    room_hours[room].append(hour)

    # Sort hours for readability
    for room in room_hours:
        room_hours[room].sort()

    return room_hours


def _extract_room(entity_id: str) -> str:
    """Extract a human-readable room name from an entity ID.

    Args:
        entity_id: HA entity ID like ``binary_sensor.living_room_motion``.

    Returns:
        Room name like ``"living_room"``.  Falls back to ``"unknown"``
        if no room can be parsed.
    """
    # Strip domain prefix
    parts = entity_id.split(".", 1)
    obj_id = parts[1] if len(parts) > 1 else parts[0]

    # Remove common suffixes
    for suffix in ("_motion", "_occupancy", "_presence", "_sensor", "_pir"):
        if obj_id.endswith(suffix):
            obj_id = obj_id[: -len(suffix)]
            break

    return obj_id if obj_id else "unknown"
