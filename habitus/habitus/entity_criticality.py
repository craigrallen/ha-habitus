"""Entity Criticality Weighting — context-aware anomaly scoring.

Assigns priority weights to entities based on:
1. Domain (battery > climate > light)
2. Keywords (shore_power, battery, bilge = critical)
3. User overrides (persisted preferences)

This prevents false alarms from low-priority entities (bathroom lights)
from drowning out genuinely critical issues (battery voltage).
"""

import json
import logging
import os

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PRIORITY_PATH = os.path.join(DATA_DIR, "entity_priorities.json")

# Default criticality weights by domain
DOMAIN_WEIGHTS = {
    # Critical systems (score × 2.0)
    "binary_sensor": 1.5,  # Alarms, alerts, security
    "sensor": 1.0,  # Default
    # Safety-critical sensor patterns
    "battery": 2.0,
    "bilge": 2.5,
    "shore_power": 1.8,
    "water_level": 2.0,
    "smoke": 3.0,
    "co": 3.0,
    "leak": 2.5,
    # Important but not critical
    "climate": 1.2,
    "energy": 1.0,
    "power": 1.0,
    # Low priority (score × 0.3)
    "light": 0.3,
    "media_player": 0.2,
    "switch": 0.5,
    "automation": 0.4,
}

# Critical keywords (boost priority if found in entity_id or friendly_name)
CRITICAL_KEYWORDS = [
    # Safety
    ("battery", 2.0),
    ("bilge", 2.5),
    ("pump", 1.5),
    ("shore_power", 1.8),
    ("shore power", 1.8),
    ("water_tank", 1.5),
    ("overflow", 2.0),
    ("leak", 2.5),
    ("smoke", 3.0),
    ("co2", 2.5),
    ("carbon_monoxide", 3.0),
    ("fire", 3.0),
    ("alarm", 2.0),
    # Power systems
    ("inverter", 1.5),
    ("charger", 1.3),
    ("voltage", 1.4),
    ("current", 1.2),
    # Low priority (reduce weight)
    ("light", 0.3),
    ("lamp", 0.3),
    ("led", 0.3),
    ("bulb", 0.3),
    ("bedroom", 0.4),
    ("bathroom", 0.3),
    ("kitchen_light", 0.3),
    ("living_room_light", 0.3),
]

# User-defined priority levels
PRIORITY_LEVELS = {
    "critical": 3.0,
    "high": 2.0,
    "medium": 1.0,
    "low": 0.5,
    "muted": 0.1,
}


def get_entity_weight(entity_id: str, friendly_name: str = "") -> float:
    """Calculate criticality weight for an entity.

    Args:
        entity_id: Entity ID (e.g., sensor.battery_voltage).
        friendly_name: Human-readable name (e.g., "House Battery Voltage").

    Returns:
        Weight multiplier (0.1 = ignore, 1.0 = normal, 3.0 = critical).
    """
    # Check user overrides first
    overrides = load_user_priorities()
    if entity_id in overrides:
        return PRIORITY_LEVELS.get(overrides[entity_id], 1.0)

    # Extract domain
    domain = entity_id.split(".")[0] if "." in entity_id else "sensor"
    weight = DOMAIN_WEIGHTS.get(domain, 1.0)

    # Check keywords in entity_id and friendly_name
    search_text = f"{entity_id} {friendly_name}".lower()

    for keyword, keyword_weight in CRITICAL_KEYWORDS:
        if keyword in search_text:
            # Use highest matching keyword weight
            weight = max(weight, keyword_weight)

    return weight


def load_user_priorities() -> dict:
    """Load user-defined entity priorities from disk.

    Returns:
        Dict mapping entity_id → priority level ("critical"|"high"|"medium"|"low"|"muted").
    """
    if not os.path.exists(PRIORITY_PATH):
        return {}

    try:
        with open(PRIORITY_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load entity priorities: {e}")
        return {}


def save_user_priorities(priorities: dict) -> None:
    """Save user-defined entity priorities to disk."""
    try:
        from .utils import atomic_write

        atomic_write(PRIORITY_PATH, priorities)
        log.info(f"Saved {len(priorities)} entity priority overrides")
    except Exception as e:
        log.warning(f"Failed to save entity priorities: {e}")


def set_entity_priority(entity_id: str, priority: str) -> None:
    """Set user priority for an entity.

    Args:
        entity_id: Entity to configure.
        priority: One of "critical", "high", "medium", "low", "muted".
    """
    if priority not in PRIORITY_LEVELS:
        raise ValueError(
            f"Invalid priority: {priority}. Must be one of {list(PRIORITY_LEVELS.keys())}"
        )

    priorities = load_user_priorities()

    if priority == "medium":
        # Remove override (revert to auto-detection)
        priorities.pop(entity_id, None)
    else:
        priorities[entity_id] = priority

    save_user_priorities(priorities)


def get_all_entity_weights(entities: list[dict]) -> dict:
    """Get weights for a list of entities.

    Args:
        entities: List of dicts with 'entity_id' and optional 'friendly_name'.

    Returns:
        Dict mapping entity_id → weight.
    """
    weights = {}
    for ent in entities:
        eid = ent.get("entity_id", "")
        name = ent.get("friendly_name", "") or ent.get("name", "")
        if eid:
            weights[eid] = get_entity_weight(eid, name)

    return weights


def apply_criticality_weighting(anomalies: list[dict]) -> list[dict]:
    """Apply criticality weighting to anomaly scores.

    Args:
        anomalies: List of anomaly dicts with 'entity_id', 'name', 'z_score'.

    Returns:
        Same list with added 'criticality_weight' and adjusted 'weighted_z_score'.
    """
    for anom in anomalies:
        eid = anom.get("entity_id", "")
        name = anom.get("name", "")
        z_score = anom.get("z_score", 0)

        weight = get_entity_weight(eid, name)
        anom["criticality_weight"] = weight
        anom["weighted_z_score"] = z_score * weight

        # Add priority label for UI
        if weight >= 2.5:
            anom["priority_label"] = "Critical"
        elif weight >= 1.5:
            anom["priority_label"] = "High"
        elif weight >= 0.8:
            anom["priority_label"] = "Medium"
        else:
            anom["priority_label"] = "Low"

    return anomalies
