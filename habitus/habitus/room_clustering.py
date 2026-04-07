"""Room-level clustering for automation suggestions.

Enriches suggestion descriptions with room-specific context by combining:
1. HA area registry (authoritative — fetched via ha_areas.fetch_areas)
2. Entity name heuristics (fallback when area API is unavailable)

Used by patterns.py and scene_analysis.py to produce suggestions like:
  "Turn off living room lights when everyone leaves"
instead of generic entity IDs.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

log = logging.getLogger("habitus")

# -----------------------------------------------------------------------
# Room keyword heuristics (fallback when HA area API unavailable)
# -----------------------------------------------------------------------

ROOM_KEYWORDS: list[tuple[str, str]] = [
    # (pattern, canonical_room_name)
    ("living_room|lounge|family_room|sitting_room", "Living Room"),
    ("master_bedroom|master_bed", "Master Bedroom"),
    ("guest_bedroom|guest_bed|spare_bedroom|spare_bed", "Guest Bedroom"),
    ("kids_room|kids_bed|child_room|nursery|playroom", "Kids Room"),
    ("bedroom|bed_room", "Bedroom"),
    ("kitchen|galley", "Kitchen"),
    ("bathroom|bath_room|ensuite|en_suite|shower", "Bathroom"),
    ("hallway|hall|corridor|entry|foyer|entrance", "Hallway"),
    ("office|study|den|workspace|home_office", "Office"),
    ("dining|dining_room", "Dining Room"),
    ("garage|workshop", "Garage"),
    ("garden|patio|deck|terrace|balcony|outdoor|outside", "Garden"),
    ("laundry|utility_room", "Laundry"),
    # Boat-specific
    ("wheelhouse|helm|bridge", "Wheelhouse"),
    ("engine_room|engine", "Engine Room"),
    ("salon|saloon|cabin", "Cabin"),
    ("cockpit|aft_deck", "Cockpit"),
    ("foredeck|fore_deck", "Foredeck"),
]

# Pre-compiled patterns sorted longest-first to prefer specific matches
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), room)
    for pattern, room in sorted(ROOM_KEYWORDS, key=lambda x: -len(x[0]))
]


def infer_room_from_entity(entity_id: str) -> str | None:
    """Infer room name from entity_id using keyword heuristics.

    Returns canonical room name or None if no match found.

    Args:
        entity_id: HA entity ID like 'light.living_room_ceiling'.
    """
    name_part = entity_id.split(".")[-1] if "." in entity_id else entity_id
    for pattern, room in _COMPILED_PATTERNS:
        if pattern.search(name_part):
            return room
    return None


def get_entity_area(entity_id: str) -> str | None:
    """Get HA-configured area for entity, falling back to heuristic."""
    try:
        from . import ha_areas
        area = ha_areas.get_entity_area(entity_id)
        if area:
            return area
    except Exception:
        pass
    return infer_room_from_entity(entity_id)


def get_rooms_for_entities(entity_ids: list[str]) -> list[str]:
    """Get unique room names for a list of entity IDs.

    Priority:
    1. HA configured areas (via ha_areas module)
    2. Entity name heuristics
    3. Returns empty list if no rooms can be determined

    Args:
        entity_ids: List of HA entity IDs.

    Returns:
        Deduplicated list of room names ordered by frequency.
    """
    room_counts: Counter = Counter()

    # Pass 1: HA configured areas (authoritative)
    try:
        from . import ha_areas
        ha_result = ha_areas.get_entities_rooms(entity_ids)
        if ha_result:
            return ha_result
    except Exception as e:
        log.debug("ha_areas unavailable, using heuristics: %s", e)

    # Pass 2: Heuristic fallback
    for eid in entity_ids:
        room = infer_room_from_entity(eid)
        if room:
            room_counts[room] += 1

    return [r for r, _ in room_counts.most_common()]


def enrich_suggestion_with_rooms(
    suggestion: dict[str, Any],
    entity_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Add room context to a suggestion dict.

    Adds/updates:
    - suggestion['rooms']: list of room names
    - suggestion['room_label']: human-friendly room string for display

    Args:
        suggestion: Existing suggestion dict (modified in-place and returned).
        entity_ids: Entities to determine rooms from. If None, tries
                    suggestion['entities'] or suggestion['entity_ids'].

    Returns:
        The (mutated) suggestion dict.
    """
    if entity_ids is None:
        entity_ids = suggestion.get("entities") or suggestion.get("entity_ids") or []

    rooms = get_rooms_for_entities(entity_ids)
    suggestion["rooms"] = rooms

    if rooms:
        if len(rooms) == 1:
            suggestion["room_label"] = rooms[0]
        elif len(rooms) == 2:
            suggestion["room_label"] = f"{rooms[0]} and {rooms[1]}"
        else:
            suggestion["room_label"] = f"{rooms[0]}, {rooms[1]}, and more"
    else:
        suggestion["room_label"] = ""

    return suggestion


def describe_with_room(action: str, entity_ids: list[str]) -> str:
    """Build a room-aware description for an automation suggestion.

    Args:
        action: Short action description like "turn off lights".
        entity_ids: Entities involved.

    Returns:
        Room-aware string like "Turn off Living Room lights when everyone leaves".
    """
    rooms = get_rooms_for_entities(entity_ids)
    if not rooms:
        return action.capitalize()
    room_str = rooms[0] if len(rooms) == 1 else " and ".join(rooms[:2])
    return f"{action.capitalize()} in {room_str}"


def cluster_entities_by_room(
    entity_ids: list[str],
) -> dict[str, list[str]]:
    """Group entity IDs by their inferred room.

    Args:
        entity_ids: List of entity IDs to cluster.

    Returns:
        Dict mapping room_name → [entity_ids]. Entities with no room
        are grouped under the key '(unknown)'.
    """
    clusters: dict[str, list[str]] = {}
    for eid in entity_ids:
        room = get_entity_area(eid) or "(unknown)"
        clusters.setdefault(room, []).append(eid)
    return clusters
