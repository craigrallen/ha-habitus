"""Home Assistant area (room) registry integration.

Fetches the entity→area mapping from HA's template API so that
scene detection, automation suggestions, and conflict detection can
use actual configured rooms instead of guessing from entity names.
"""

import json
import logging
import os
from typing import Any

import requests

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
AREAS_CACHE_PATH = os.path.join(DATA_DIR, "ha_areas.json")

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))


def fetch_areas() -> dict[str, Any]:
    """Fetch area list and entity→area mapping from HA.

    Returns:
        {
            "areas": [{"id": "...", "name": "Kitchen", "entity_count": 158}, ...],
            "entity_to_area": {"light.kitchen_ceiling": "Kitchen", ...},
        }
    """
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

    # Get area list with names
    try:
        r = requests.post(
            f"{HA_URL}/api/template",
            headers=headers,
            json={"template": "{% for a in areas() %}{{ a }}|{{ area_name(a) }}|{{ area_entities(a) | length }}\n{% endfor %}"},
            timeout=10,
        )
        if r.status_code != 200:
            log.warning("Failed to fetch areas: HTTP %d", r.status_code)
            return _load_cache()

        areas = []
        for line in r.text.strip().split("\n"):
            parts = line.strip().split("|")
            if len(parts) >= 3:
                areas.append({
                    "id": parts[0],
                    "name": parts[1],
                    "entity_count": int(parts[2]) if parts[2].isdigit() else 0,
                })
    except Exception as e:
        log.warning("Area list fetch failed: %s", e)
        return _load_cache()

    # Get entity→area mapping
    entity_to_area: dict[str, str] = {}
    try:
        r = requests.post(
            f"{HA_URL}/api/template",
            headers=headers,
            json={"template": "{% for a in areas() %}{% for e in area_entities(a) %}{{ e }}|{{ area_name(a) }}\n{% endfor %}{% endfor %}"},
            timeout=30,
        )
        if r.status_code == 200:
            for line in r.text.strip().split("\n"):
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    entity_to_area[parts[0]] = parts[1]
    except Exception as e:
        log.warning("Entity-area mapping fetch failed: %s", e)

    result = {
        "areas": areas,
        "entity_to_area": entity_to_area,
        "area_count": len(areas),
        "mapped_entities": len(entity_to_area),
    }

    # Cache
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(AREAS_CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("HA areas: %d areas, %d entities mapped", len(areas), len(entity_to_area))

    return result


def get_entity_area(entity_id: str) -> str | None:
    """Look up the HA-configured area for an entity. Uses cache."""
    data = _load_cache()
    return data.get("entity_to_area", {}).get(entity_id)


def get_entities_rooms(entity_ids: list[str]) -> list[str]:
    """Get unique room names for a list of entity IDs from HA areas.

    Returns list of area names (deduplicated, ordered by frequency).
    """
    data = _load_cache()
    e2a = data.get("entity_to_area", {})
    from collections import Counter
    room_counts = Counter(e2a[eid] for eid in entity_ids if eid in e2a)
    return [r for r, _ in room_counts.most_common()]


def _load_cache() -> dict[str, Any]:
    """Load cached area data."""
    try:
        if os.path.exists(AREAS_CACHE_PATH):
            with open(AREAS_CACHE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"areas": [], "entity_to_area": {}, "area_count": 0, "mapped_entities": 0}
