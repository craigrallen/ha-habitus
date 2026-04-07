"""Scene Detector — mine entity co-occurrence patterns to find implicit scenes.

Discovers groups of entities that change state together within a time window,
detects time-of-day patterns, and names scenes automatically.

Uses the HA SQLite database directly for fast state history access.
"""

import datetime
import json
import logging
import os
from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import Any

from .ha_db import managed_read_connection, resolve_ha_db_path, table_exists

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
SCENES_PATH = os.path.join(DATA_DIR, "scenes.json")

# Co-occurrence window in seconds
CO_OCCURRENCE_WINDOW = 300  # 5 minutes
# Minimum times entities must co-occur to form a scene
MIN_CO_OCCURRENCES = 5
# Minimum confidence to report a scene
MIN_CONFIDENCE = 40
# Domains we care about for scene detection
SCENE_DOMAINS = ("light", "switch", "media_player", "fan", "cover", "climate", "input_boolean")
# Trigger domains — entities that indicate user presence/activity (used as automation triggers)
TRIGGER_DOMAINS = ("binary_sensor", "person", "device_tracker")
# Presence-inferring keywords — state changes on these entities imply someone is in the room
# (e.g. heater turned up/on → person in that room)
PRESENCE_HINT_KEYWORDS = ("heater", "radiator", "thermostat", "heating", "climate", "hvac")


def _get_state_changes(days: int = 30) -> list[dict[str, Any]]:
    """Fetch state changes from HA SQLite database directly.

    Returns list of dicts with entity_id, state, last_changed (as datetime).
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        log.warning("HA database not found in standard paths — falling back to empty")
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        with managed_read_connection(db_path) as conn:
            if conn is None:
                return []

            # HA uses states_meta for entity_id mapping (HA 2023.4+)
            has_states_meta = table_exists(conn, "states_meta")

            if has_states_meta:
                query = """
                    SELECT sm.entity_id, s.state, s.last_changed_ts
                    FROM states s
                    JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                    WHERE s.last_changed_ts > ?
                    AND (
                        sm.entity_id LIKE 'light.%'
                        OR sm.entity_id LIKE 'switch.%'
                        OR sm.entity_id LIKE 'media_player.%'
                        OR sm.entity_id LIKE 'fan.%'
                        OR sm.entity_id LIKE 'cover.%'
                        OR sm.entity_id LIKE 'climate.%'
                        OR sm.entity_id LIKE 'input_boolean.%'
                        OR sm.entity_id LIKE 'binary_sensor.%'
                        OR sm.entity_id LIKE 'person.%'
                        OR sm.entity_id LIKE 'device_tracker.%'
                    )
                    ORDER BY s.last_changed_ts
                """
            else:
                query = """
                    SELECT entity_id, state, last_changed_ts
                    FROM states
                    WHERE last_changed_ts > ?
                    AND (
                        entity_id LIKE 'light.%'
                        OR entity_id LIKE 'switch.%'
                        OR entity_id LIKE 'media_player.%'
                        OR entity_id LIKE 'fan.%'
                        OR entity_id LIKE 'cover.%'
                        OR entity_id LIKE 'climate.%'
                        OR entity_id LIKE 'input_boolean.%'
                        OR entity_id LIKE 'binary_sensor.%'
                        OR entity_id LIKE 'person.%'
                        OR entity_id LIKE 'device_tracker.%'
                    )
                    ORDER BY last_changed_ts
                """

            rows = conn.execute(query, (cutoff_ts,)).fetchall()

        results = []
        for entity_id, state, ts in rows:
            if ts is None:
                continue
            if state in ("unavailable", "unknown", ""):
                continue
            results.append({
                "entity_id": entity_id,
                "state": state,
                "timestamp": float(ts),
                "dt": datetime.datetime.fromtimestamp(float(ts), tz=datetime.UTC),
            })
        log.info("Scene detector: loaded %d state changes from HA database", len(results))
        return results
    except Exception as e:
        log.error("Failed to read HA database: %s", e)
        return []


def _find_co_occurrences(
    changes: list[dict[str, Any]],
    window_s: int = CO_OCCURRENCE_WINDOW,
    strict_sorted: bool = False,
) -> dict[tuple[str, str], list[float]]:
    """Find groups of entities that change state within the same time window.

    Uses a two-pointer sliding window to avoid rescanning the full tail of
    activations for every event.

    Returns dict mapping entity pairs to occurrence timestamps.

    Args:
        changes: Input events. Expected to be sorted by ascending ``timestamp``.
        window_s: Co-occurrence window in seconds.
        strict_sorted: When ``True``, validate that ``changes`` are monotonic by
            timestamp and fail fast with ``ValueError`` if not.
    """
    if not changes:
        return {}

    if strict_sorted:
        last_ts = float(changes[0].get("timestamp", 0.0))
        for idx, change in enumerate(changes[1:], start=1):
            ts = float(change.get("timestamp", 0.0))
            if ts < last_ts:
                raise ValueError(
                    f"changes must be sorted by timestamp for _find_co_occurrences (index {idx})"
                )
            last_ts = ts

    # Group "on" transitions (state going to on/playing/open/home)
    on_states = {"on", "playing", "open", "home", "heat", "cool", "auto", "above_horizon"}
    activations = [c for c in changes if c["state"].lower() in on_states]

    if len(activations) < 2:
        return {}

    pair_occurrences: dict[tuple[str, str], list[float]] = defaultdict(list)

    # Track current window [i, end) with monotonically increasing end pointer.
    entity_counts: dict[str, int] = defaultdict(int)
    end = 0
    n = len(activations)

    for i in range(n):
        group_ts = activations[i]["timestamp"]

        if end < i:
            end = i

        while end < n and (activations[end]["timestamp"] - group_ts) <= window_s:
            entity_counts[activations[end]["entity_id"]] += 1
            end += 1

        if len(entity_counts) >= 2:
            sorted_entities = sorted(entity_counts)
            for a, b in combinations(sorted_entities, 2):
                pair_occurrences[(a, b)].append(group_ts)

        current_entity = activations[i]["entity_id"]
        entity_counts[current_entity] -= 1
        if entity_counts[current_entity] <= 0:
            del entity_counts[current_entity]

    return pair_occurrences


def _cluster_pairs_to_scenes(
    pair_occurrences: dict[tuple[str, str], list[float]],
    min_co: int = MIN_CO_OCCURRENCES,
) -> list[set[str]]:
    """Cluster frequently co-occurring pairs into larger scene groups.

    Uses a greedy graph-based approach: pairs with enough co-occurrences
    form edges, then connected components become scenes.
    """
    # Filter to pairs that co-occur often enough
    frequent_pairs = {
        pair: times for pair, times in pair_occurrences.items()
        if len(times) >= min_co
    }

    if not frequent_pairs:
        return []

    # Build adjacency graph
    adjacency: dict[str, set[str]] = defaultdict(set)
    pair_counts: dict[tuple[str, str], int] = {}
    for (a, b), times in frequent_pairs.items():
        adjacency[a].add(b)
        adjacency[b].add(a)
        pair_counts[(a, b)] = len(times)

    # Find connected components (scenes)
    visited: set[str] = set()
    scenes: list[set[str]] = []

    for entity in adjacency:
        if entity in visited:
            continue
        # BFS
        component: set[str] = set()
        queue = deque([entity])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(component) >= 2:
            # Limit scene size to avoid giant meaningless clusters
            if len(component) <= 12:
                scenes.append(component)
            else:
                # Split large components by finding the densest sub-groups
                _split_large_component(component, pair_counts, scenes)

    return scenes


def _split_large_component(
    component: set[str],
    pair_counts: dict[tuple[str, str], int],
    scenes: list[set[str]],
    max_size: int = 8,
) -> None:
    """Split a large component into smaller scene groups by co-occurrence strength."""
    entities = sorted(component)
    # Score each entity by total co-occurrence count
    entity_scores: dict[str, int] = defaultdict(int)
    for (a, b), count in pair_counts.items():
        if a in component and b in component:
            entity_scores[a] += count
            entity_scores[b] += count

    # Greedy: start from highest-scored entity, add most connected neighbors
    used: set[str] = set()
    sorted_by_score = sorted(entities, key=lambda e: entity_scores.get(e, 0), reverse=True)

    for seed in sorted_by_score:
        if seed in used:
            continue
        group = {seed}
        used.add(seed)
        # Add neighbors sorted by co-occurrence strength with seed
        neighbors = []
        for e in entities:
            if e in used:
                continue
            pair_key = tuple(sorted([seed, e]))
            count = pair_counts.get(pair_key, 0)
            if count > 0:
                neighbors.append((e, count))
        neighbors.sort(key=lambda x: -x[1])
        for e, _ in neighbors[:max_size - 1]:
            group.add(e)
            used.add(e)
        if len(group) >= 2:
            scenes.append(group)


def _analyze_time_patterns(
    entities: set[str],
    changes: list[dict[str, Any]],
    window_s: int = CO_OCCURRENCE_WINDOW,
) -> dict[str, Any]:
    """Analyze when a scene's entities tend to activate together.

    Returns time-of-day pattern, day-of-week pattern, and typical states.
    """
    on_states = {"on", "playing", "open", "home", "heat", "cool", "auto"}

    # Find windows where multiple scene entities activated
    entity_activations: dict[str, list[float]] = defaultdict(list)
    for c in changes:
        if c["entity_id"] in entities and c["state"].lower() in on_states:
            entity_activations[c["entity_id"]].append(c["timestamp"])

    # Find moments where 2+ scene entities activated within window
    all_times = []
    for eid, times in entity_activations.items():
        for t in times:
            all_times.append((t, eid))
    all_times.sort()

    scene_moments: list[float] = []
    i = 0
    while i < len(all_times):
        window_entities: set[str] = {all_times[i][1]}
        j = i + 1
        while j < len(all_times) and (all_times[j][0] - all_times[i][0]) <= window_s:
            window_entities.add(all_times[j][1])
            j += 1
        if len(window_entities & entities) >= 2:
            scene_moments.append(all_times[i][0])
            i = j  # skip past this window
        else:
            i += 1

    if not scene_moments:
        return {"count": 0}

    # Analyze hour distribution
    hours = [datetime.datetime.fromtimestamp(t, tz=datetime.UTC).hour for t in scene_moments]
    hour_counts = Counter(hours)
    peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 18

    # Day of week distribution
    days = [datetime.datetime.fromtimestamp(t, tz=datetime.UTC).weekday() for t in scene_moments]
    day_counts = Counter(days)
    weekday_count = sum(day_counts.get(d, 0) for d in range(5))
    weekend_count = sum(day_counts.get(d, 0) for d in range(5, 7))

    # Time clustering — find the most concentrated 2-hour window
    best_window_count = 0
    best_window_start = 18
    for h in range(24):
        window_count = hour_counts.get(h, 0) + hour_counts.get((h + 1) % 24, 0)
        if window_count > best_window_count:
            best_window_count = window_count
            best_window_start = h

    # Determine time-of-day label
    if best_window_start < 6:
        time_label = "Night"
    elif best_window_start < 12:
        time_label = "Morning"
    elif best_window_start < 17:
        time_label = "Afternoon"
    else:
        time_label = "Evening"

    # Day pattern
    total = weekday_count + weekend_count
    if total > 0:
        weekday_ratio = weekday_count / total
        if weekday_ratio > 0.8:
            day_pattern = "weekdays"
        elif weekday_ratio < 0.3:
            day_pattern = "weekends"
        else:
            day_pattern = "daily"
    else:
        day_pattern = "daily"

    return {
        "count": len(scene_moments),
        "peak_hour": peak_hour,
        "best_window_start": best_window_start,
        "best_window_end": (best_window_start + 2) % 24,
        "time_label": time_label,
        "day_pattern": day_pattern,
        "weekday_ratio": round(weekday_count / max(total, 1), 2),
        "hour_distribution": dict(hour_counts),
    }


# Known room keywords — matched against entity IDs, friendly names, and HA areas
ROOM_KEYWORDS = [
    "living_room", "living room", "lounge", "family_room",
    "bedroom", "master_bedroom", "guest_bedroom", "kids_room",
    "kitchen", "galley",
    "bathroom", "bath", "shower", "ensuite",
    "hallway", "hall", "corridor", "entry", "foyer", "entrance",
    "office", "study", "den", "workspace",
    "dining", "dining_room",
    "garage", "workshop",
    "garden", "patio", "deck", "terrace", "balcony",
    "laundry", "utility",
    "nursery", "playroom",
    # Boat-specific
    "wheelhouse", "engine_room", "salon", "saloon", "cabin", "cockpit",
    "foredeck", "aft_deck", "anchor", "helm",
]

# Hoisted once to avoid per-call sorting inside _extract_room.
_ROOM_KEYWORDS_SORTED = tuple(sorted(ROOM_KEYWORDS, key=len, reverse=True))


def _extract_room(entity_id: str) -> str | None:
    """Extract room name from an entity ID using keyword matching.

    Checks the entity name part (after the dot) for known room keywords.
    Returns the matched room name in title case, or None.
    """
    name_part = entity_id.split(".")[-1].lower()

    # Try longest match first (e.g. "living_room" before "room")
    for kw in _ROOM_KEYWORDS_SORTED:
        kw_underscore = kw.replace(" ", "_")
        if kw_underscore in name_part or kw in name_part:
            return kw.replace("_", " ").title()
    return None


def _extract_rooms_from_entities(entities: set[str]) -> list[str]:
    """Extract unique room names from a set of entity IDs.

    Priority: 1) HA configured areas, 2) keyword matching, 3) common prefix.
    """
    rooms: list[str] = []
    seen: set[str] = set()

    # Pass 0: HA configured areas (authoritative)
    try:
        from . import ha_areas
        ha_rooms = ha_areas.get_entities_rooms(list(entities))
        for r in ha_rooms:
            if r.lower() not in seen:
                seen.add(r.lower())
                rooms.append(r)
    except Exception:
        pass  # ha_areas not available or cache empty — fall through

    if rooms:
        return rooms[:3]  # HA areas found — use them

    # Pass 1: keyword matching (fallback)
    for eid in sorted(entities):
        room = _extract_room(eid)
        if room and room.lower() not in seen:
            seen.add(room.lower())
            rooms.append(room)

    # Pass 2: if no rooms found, try extracting the device/area name
    # by finding the common prefix among entity name parts
    if not rooms:
        name_parts = []
        for eid in entities:
            part = eid.split(".")[-1]
            # Remove numeric suffixes and common generic suffixes
            for suffix in ("_light", "_lamp", "_main", "_ceiling", "_switch", "_plug",
                           "_media_player", "_fan", "_cover", "_strip", "_dimmer",
                           "_sensor", "_motion", "_door", "_window", "_contact",
                           "_power", "_energy", "_temperature", "_humidity",
                           "_1", "_2", "_3", "_left", "_right", "_on", "_off"):
                part = part.removesuffix(suffix)
            if part:
                name_parts.append(part)

        if name_parts:
            # Find most common prefix token
            from collections import Counter
            tokens = []
            for p in name_parts:
                tokens.extend(p.split("_")[:2])  # first 2 tokens
            if tokens:
                common = Counter(tokens).most_common(1)
                if common and common[0][1] >= 2:
                    room = common[0][0].replace("_", " ").title()
                    if room.lower() not in seen and len(room) > 2:
                        rooms.append(room)

    return rooms[:3]  # Max 3 rooms


def _name_scene(entities: set[str], time_info: dict[str, Any]) -> str:
    """Auto-generate a human-readable scene name from entities and time pattern."""
    rooms = _extract_rooms_from_entities(entities)

    time_label = time_info.get("time_label", "")
    room_str = " & ".join(rooms) if rooms else "Home"

    if time_label:
        return f"{time_label} {room_str}"
    return room_str


def _compute_scene_confidence(entities: set[str], time_info: dict[str, Any]) -> int:
    """Compute calibrated confidence score for a scene candidate."""
    freq_score = min(100, time_info["count"] * 3)

    best_window_start = time_info.get("best_window_start", 0)
    total_in_peak = sum(
        time_info.get("hour_distribution", {}).get((best_window_start + offset) % 24, 0)
        for offset in range(3)
    )
    time_consistency = (
        (total_in_peak / max(time_info["count"], 1)) * 100
        if time_info["count"] > 0
        else 0
    )

    domains_in_scene = {e.split(".")[0] for e in entities}
    signal_types = set()
    for d in domains_in_scene:
        if d in ("light",):
            signal_types.add("lighting")
        elif d in ("switch", "input_boolean"):
            signal_types.add("switching")
        elif d in ("media_player",):
            signal_types.add("media")
        elif d in ("climate", "fan"):
            signal_types.add("comfort")
        elif d in ("cover",):
            signal_types.add("covers")
        elif d in ("binary_sensor",):
            signal_types.add("presence")
        elif d in ("person", "device_tracker"):
            signal_types.add("location")

    diversity_bonus = min(20, (len(signal_types) - 1) * 10) if len(signal_types) > 1 else 0
    return int(round(min(95, (freq_score * 0.35) + (time_consistency * 0.45) + diversity_bonus)))


def detect_scenes(days: int = 30) -> list[dict[str, Any]]:
    """Main entry point: detect implicit scenes from HA state history.

    Args:
        days: How many days of history to analyze.

    Returns:
        List of scene dicts with entities, time patterns, confidence, name.
    """
    changes = _get_state_changes(days)
    if not changes:
        log.info("Scene detector: no state changes found")
        return []

    log.info("Scene detector: analyzing %d state changes over %d days", len(changes), days)

    # Find co-occurring entity pairs
    pair_occurrences = _find_co_occurrences(changes)
    log.info("Scene detector: found %d co-occurring pairs", len(pair_occurrences))

    # Cluster into scenes
    scenes = _cluster_pairs_to_scenes(pair_occurrences)
    log.info("Scene detector: clustered into %d potential scenes", len(scenes))

    # Analyze and score each scene
    results = []
    for idx, entities in enumerate(scenes):
        time_info = _analyze_time_patterns(entities, changes)
        if time_info["count"] < MIN_CO_OCCURRENCES:
            continue

        name = _name_scene(entities, time_info)

        confidence = _compute_scene_confidence(entities, time_info)

        if confidence < MIN_CONFIDENCE:
            continue

        sorted_entities = sorted(entities)

        # Build entity states for scene creation
        entity_states = {}
        for eid in sorted_entities:
            domain = eid.split(".")[0]
            if domain in ("light", "switch", "fan", "input_boolean"):
                entity_states[eid] = "on"
            elif domain == "media_player":
                entity_states[eid] = "playing"
            elif domain == "cover":
                entity_states[eid] = "open"
            elif domain == "climate":
                entity_states[eid] = "auto"

        rooms = _extract_rooms_from_entities(set(sorted_entities))
        scene = {
            "id": f"scene_{idx}",
            "name": name,
            "rooms": rooms,
            "entities": sorted_entities,
            "entity_states": entity_states,
            "confidence": confidence,
            "occurrences": time_info["count"],
            "time_pattern": {
                "peak_hour": time_info.get("peak_hour"),
                "window": f"{time_info.get('best_window_start', 0):02d}:00–{time_info.get('best_window_end', 2):02d}:00",
                "label": time_info.get("time_label", ""),
                "days": time_info.get("day_pattern", "daily"),
            },
            "description": _build_description(name, sorted_entities, time_info),
        }
        results.append(scene)

    # Sort by confidence descending
    results.sort(key=lambda s: -s["confidence"])
    log.info("Scene detector: found %d scenes above confidence threshold", len(results))
    return results


def _build_description(
    name: str, entities: list[str], time_info: dict[str, Any]
) -> str:
    """Build a human-readable description of a discovered scene."""
    count = time_info.get("count", 0)
    peak = time_info.get("peak_hour", 18)
    days = time_info.get("day_pattern", "daily")
    n_ents = len(entities)

    # Friendly entity names
    friendly = [e.split(".")[1].replace("_", " ").title() for e in entities[:4]]
    ent_str = ", ".join(friendly)
    if n_ents > 4:
        ent_str += f" and {n_ents - 4} more"

    day_str = {
        "weekdays": "on weekdays",
        "weekends": "on weekends",
        "daily": "most days",
    }.get(days, "regularly")

    return (
        f"We noticed {ent_str} activate together around "
        f"{peak:02d}:00 {day_str}. "
        f"Detected {count} times in the last 30 days — "
        f"this looks like an implicit \"{name}\" scene."
    )


def save(scenes: list[dict[str, Any]]) -> None:
    """Save detected scenes to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCENES_PATH, "w") as f:
        json.dump(
            {
                "scenes": scenes,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "count": len(scenes),
            },
            f,
            indent=2,
            default=str,
        )
    log.info("Scene detector: saved %d scenes to %s", len(scenes), SCENES_PATH)


def load() -> list[dict[str, Any]]:
    """Load previously detected scenes."""
    try:
        if os.path.exists(SCENES_PATH):
            with open(SCENES_PATH) as f:
                data = json.load(f)
            return data.get("scenes", [])
    except Exception:
        pass
    return []
