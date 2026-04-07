"""Adaptive Room Thresholds — per-area anomaly sensitivity.

Instead of a single global threshold, learns appropriate sensitivity
per room/zone based on entity characteristics:
- Garage motion at 3 AM: high sensitivity (security concern)
- Bathroom lights at 3 AM: low sensitivity (normal usage)
- Battery voltage: always high sensitivity (safety critical)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
THRESHOLDS_PATH = os.path.join(DATA_DIR, "room_thresholds.json")

# Default zone sensitivity multipliers (1.0 = normal, <1 = more sensitive, >1 = less sensitive)
DEFAULT_ZONE_SENSITIVITY: dict[str, float] = {
    "bedroom": 0.8,  # Slightly more sensitive (sleep disruption)
    "bathroom": 1.5,  # Less sensitive (erratic usage patterns)
    "kitchen": 1.0,  # Normal
    "living_room": 1.0,  # Normal
    "garage": 0.6,  # More sensitive (security)
    "outdoor": 0.5,  # Most sensitive (security)
    "hallway": 0.9,  # Slightly more sensitive
    "office": 1.0,  # Normal
    "utility": 1.3,  # Less sensitive (mechanical equipment)
    "bilge": 0.4,  # Very sensitive (safety critical)
    "engine": 0.5,  # Very sensitive (safety critical)
}

# Time-of-day sensitivity overrides
NIGHT_HOURS = set(range(23, 24)) | set(range(0, 6))


def _detect_zone(entity_id: str) -> str:
    """Infer the zone/room from the entity ID.

    Args:
        entity_id: HA entity ID.

    Returns:
        Zone name string (e.g. ``"kitchen"``, ``"garage"``).
    """
    eid = entity_id.lower()
    for zone in DEFAULT_ZONE_SENSITIVITY:
        if zone in eid:
            return zone
    # Check common HA naming patterns
    zone_keywords = {
        "bedroom": ["bed", "bedroom", "master", "sleep"],
        "bathroom": ["bath", "shower", "toilet", "wc"],
        "kitchen": ["kitchen", "cooker", "oven", "fridge", "dishwash"],
        "living_room": ["living", "lounge", "tv", "sofa", "media"],
        "garage": ["garage", "carport"],
        "outdoor": ["outdoor", "garden", "patio", "deck", "exterior", "front_door"],
        "hallway": ["hall", "corridor", "entry", "foyer"],
        "office": ["office", "desk", "study"],
        "utility": ["utility", "laundry", "pump", "boiler"],
        "bilge": ["bilge"],
        "engine": ["engine", "motor"],
    }
    for zone, keywords in zone_keywords.items():
        if any(kw in eid for kw in keywords):
            return zone
    return "default"


def get_threshold_multiplier(entity_id: str, hour: int | None = None) -> float:
    """Get the anomaly threshold multiplier for an entity.

    Lower multiplier = more sensitive (stricter detection).
    Higher multiplier = less sensitive (more lenient).

    Args:
        entity_id: HA entity ID.
        hour: Current hour (0-23). If None, uses zone default only.

    Returns:
        Multiplier for the entity's z-score threshold.
    """
    # Check user overrides first
    overrides = _load_overrides()
    if entity_id in overrides:
        return overrides[entity_id]

    zone = _detect_zone(entity_id)
    base = DEFAULT_ZONE_SENSITIVITY.get(zone, 1.0)

    # Night-time adjustment: outdoor/garage entities become more sensitive
    if hour is not None and hour in NIGHT_HOURS:
        if zone in ("outdoor", "garage", "hallway"):
            base *= 0.6  # 40% more sensitive at night
        elif zone in ("bathroom", "kitchen"):
            base *= 1.3  # 30% more lenient at night (normal nighttime activity)

    # Safety-critical entities always get maximum sensitivity
    eid = entity_id.lower()
    safety_keywords = [
        "battery",
        "soc",
        "bilge",
        "smoke",
        "co2",
        "co_",
        "leak",
        "water_level",
        "fire",
        "alarm",
        "shore_power",
    ]
    if any(kw in eid for kw in safety_keywords):
        base = min(base, 0.5)

    return round(base, 2)


def _load_overrides() -> dict[str, float]:
    """Load user threshold overrides from disk."""
    try:
        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH) as f:
                data: dict[str, Any] = json.load(f)
                result: dict[str, float] = data.get("overrides", {})
                return result
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def set_override(entity_id: str, multiplier: float) -> dict[str, Any]:
    """Set a user override for an entity's threshold multiplier.

    Args:
        entity_id: HA entity ID.
        multiplier: Threshold multiplier (0.1 to 5.0).

    Returns:
        Updated thresholds data.
    """
    multiplier = max(0.1, min(5.0, multiplier))
    try:
        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH) as f:
                data: dict[str, Any] = json.load(f)
        else:
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}

    overrides = data.get("overrides", {})
    overrides[entity_id] = multiplier
    data["overrides"] = overrides
    atomic_write(THRESHOLDS_PATH, data)
    log.info("Threshold override: %s → %.2f", entity_id, multiplier)
    return data


def remove_override(entity_id: str) -> dict[str, Any]:
    """Remove a user threshold override.

    Args:
        entity_id: HA entity ID.

    Returns:
        Updated thresholds data.
    """
    try:
        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH) as f:
                data: dict[str, Any] = json.load(f)
        else:
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}

    overrides = data.get("overrides", {})
    overrides.pop(entity_id, None)
    data["overrides"] = overrides
    atomic_write(THRESHOLDS_PATH, data)
    return data


def get_zone_map(entity_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Map entities to their detected zones and sensitivity settings.

    Args:
        entity_ids: List of HA entity IDs.

    Returns:
        Dict mapping entity_id to zone info.
    """
    result: dict[str, dict[str, Any]] = {}
    overrides = _load_overrides()
    for eid in entity_ids:
        zone = _detect_zone(eid)
        multiplier = get_threshold_multiplier(eid)
        result[eid] = {
            "zone": zone,
            "multiplier": multiplier,
            "has_override": eid in overrides,
            "is_safety_critical": multiplier <= 0.5,
        }
    return result
