"""Entity Ignore List — users can permanently exclude noisy sensors from scoring.

Persisted to ``entity_ignore.json`` in DATA_DIR. The anomaly scoring
pipeline checks this list before scoring entities, and the UI provides
add/remove controls.
"""

import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
IGNORE_PATH = os.path.join(DATA_DIR, "entity_ignore.json")


def _load() -> dict[str, Any]:
    """Load the ignore list from disk."""
    try:
        if os.path.exists(IGNORE_PATH):
            with open(IGNORE_PATH) as f:
                data: dict[str, Any] = json.load(f)
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"entities": [], "updated_at": ""}


def get_ignored_entities() -> list[str]:
    """Return the list of ignored entity IDs.

    Returns:
        List of entity ID strings that should be excluded from anomaly scoring.
    """
    result: list[str] = _load().get("entities", [])
    return result


def is_ignored(entity_id: str) -> bool:
    """Check if an entity is on the ignore list.

    Args:
        entity_id: HA entity ID to check.

    Returns:
        True if the entity should be excluded from scoring.
    """
    return entity_id in get_ignored_entities()


def add_entity(entity_id: str, reason: str = "") -> dict[str, Any]:
    """Add an entity to the ignore list.

    Args:
        entity_id: HA entity ID to ignore.
        reason: Optional reason for ignoring (displayed in UI).

    Returns:
        Updated ignore list data dict.
    """
    import datetime

    data = _load()
    entities = data.get("entities", [])
    if entity_id not in entities:
        entities.append(entity_id)
        data["entities"] = entities
        data["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        # Track reasons
        reasons = data.get("reasons", {})
        if reason:
            reasons[entity_id] = reason
        data["reasons"] = reasons
        atomic_write(IGNORE_PATH, data)
        log.info("Entity ignored: %s (%s)", entity_id, reason or "no reason")
    return data


def remove_entity(entity_id: str) -> dict[str, Any]:
    """Remove an entity from the ignore list.

    Args:
        entity_id: HA entity ID to stop ignoring.

    Returns:
        Updated ignore list data dict.
    """
    import datetime

    data = _load()
    entities = data.get("entities", [])
    if entity_id in entities:
        entities.remove(entity_id)
        data["entities"] = entities
        data["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        reasons = data.get("reasons", {})
        reasons.pop(entity_id, None)
        data["reasons"] = reasons
        atomic_write(IGNORE_PATH, data)
        log.info("Entity un-ignored: %s", entity_id)
    return data


def get_ignore_list() -> dict[str, Any]:
    """Return the full ignore list with reasons.

    Returns:
        Dict with ``entities`` (list), ``reasons`` (dict), ``updated_at`` (str).
    """
    return _load()
