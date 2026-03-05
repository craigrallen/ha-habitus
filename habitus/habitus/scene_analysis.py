"""Scene Analysis — improvement suggestions for HA scenes.

Compares configured HA scenes against learned co-occurrence patterns to:
- Identify entities frequently activated with scene entities that aren't in the scene.
- Suggest automation triggers based on time/occupancy patterns.

Output is a list of scene improvement suggestions cached in DATA_DIR/scene_analysis.json.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any

import requests
import yaml

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
SCENE_ANALYSIS_PATH = os.path.join(DATA_DIR, "scene_analysis.json")
SCENES_PATH = os.path.join(DATA_DIR, "scenes.json")

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))

# Minimum co-occurrence count to flag an entity as "missing"
MIN_CO_OCCURRENCE_THRESHOLD = 3
# Minimum ratio of co-occurrences vs scene activations to flag
MIN_CO_OCCURRENCE_RATIO = 0.4
# Domains used in scenes
SCENE_DOMAINS = ("light", "switch", "media_player", "fan", "cover", "climate", "input_boolean")


# ---------------------------------------------------------------------------
# HA data fetching
# ---------------------------------------------------------------------------


def _ha_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def fetch_ha_scenes() -> list[dict[str, Any]]:
    """Fetch configured HA scenes via REST API.

    Returns list of scene state objects with entity_id, attributes, state.
    Falls back to empty list on error.
    """
    try:
        r = requests.get(
            f"{HA_URL}/api/states",
            headers=_ha_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            log.warning("fetch_ha_scenes: HTTP %d", r.status_code)
            return []
        states = r.json()
        return [s for s in states if s.get("entity_id", "").startswith("scene.")]
    except Exception as e:
        log.warning("fetch_ha_scenes failed: %s", e)
        return []


def _parse_scene_entities(scene_state: dict[str, Any]) -> list[str]:
    """Extract entity_ids from a scene state object."""
    attrs = scene_state.get("attributes", {})
    # HA scenes store entities in 'entity_id' attribute (may be list or dict)
    entity_data = attrs.get("entity_id", [])
    if isinstance(entity_data, list):
        return [str(e) for e in entity_data]
    if isinstance(entity_data, dict):
        return list(entity_data.keys())
    return []


# ---------------------------------------------------------------------------
# Co-occurrence data from discovered scenes
# ---------------------------------------------------------------------------


def _load_discovered_scenes() -> list[dict[str, Any]]:
    """Load habitus-discovered implicit scenes from DATA_DIR/scenes.json."""
    try:
        if os.path.exists(SCENES_PATH):
            with open(SCENES_PATH) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("scenes", [])
                if isinstance(data, list):
                    return data
    except Exception as e:
        log.warning("_load_discovered_scenes failed: %s", e)
    return []


def _build_co_occurrence_index(
    discovered_scenes: list[dict[str, Any]],
) -> dict[str, Counter]:
    """Build entity → Counter(co-occurring entity → occurrence_count) index.

    Uses habitus discovered scenes as the source of co-occurrence knowledge.
    Each discovered scene contributes pair-wise co-occurrences weighted by
    its occurrence count.
    """
    index: dict[str, Counter] = defaultdict(Counter)
    for scene in discovered_scenes:
        entities = scene.get("entities", [])
        occurrences = scene.get("occurrences", 1)
        for i, a in enumerate(entities):
            for b in entities:
                if a != b:
                    index[a][b] += occurrences
    return index


# ---------------------------------------------------------------------------
# Trigger suggestion generation
# ---------------------------------------------------------------------------


def _build_trigger_yaml(scene_name: str, trigger_config: dict[str, Any]) -> str:
    """Build a YAML automation snippet for a suggested trigger."""
    trigger_type = trigger_config.get("type", "time")
    trigger_time = trigger_config.get("time", "18:00:00")
    trigger_day_pattern = trigger_config.get("days", "daily")

    if trigger_type == "time":
        automation = {
            "alias": f"Habitus: Trigger {scene_name}",
            "trigger": [{"platform": "time", "at": trigger_time}],
            "condition": [],
            "action": [{"service": "scene.turn_on", "target": {"entity_id": "scene.placeholder"}}],
            "mode": "single",
        }
        if trigger_day_pattern and trigger_day_pattern != "daily":
            days_map = {"weekdays": [0, 1, 2, 3, 4], "weekends": [5, 6]}
            mapped = days_map.get(trigger_day_pattern)
            if mapped is not None:
                automation["condition"] = [
                    {
                        "condition": "time",
                        "weekday": [
                            ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][d]
                            for d in mapped
                        ],
                    }
                ]
    elif trigger_type == "state":
        entity = trigger_config.get("entity_id", "binary_sensor.occupancy")
        automation = {
            "alias": f"Habitus: Trigger {scene_name}",
            "trigger": [{"platform": "state", "entity_id": entity, "to": "on"}],
            "condition": [],
            "action": [{"service": "scene.turn_on", "target": {"entity_id": "scene.placeholder"}}],
            "mode": "single",
        }
    else:
        return ""

    try:
        return yaml.dump(automation, default_flow_style=False, allow_unicode=True)
    except Exception:
        return ""


def _suggest_triggers(
    discovered_scene: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Generate trigger suggestions from a matching discovered scene's time pattern."""
    if not discovered_scene:
        return []

    suggestions = []
    time_pattern = discovered_scene.get("time_pattern", {})
    peak_hour = time_pattern.get("peak_hour")
    day_pattern = time_pattern.get("days", "daily")
    scene_name = discovered_scene.get("name", "Scene")

    if peak_hour is not None:
        trigger_time = f"{int(peak_hour):02d}:00:00"
        trigger_cfg = {"type": "time", "time": trigger_time, "days": day_pattern}
        yaml_snippet = _build_trigger_yaml(scene_name, trigger_cfg)
        suggestions.append({
            "description": (
                f"Trigger at {trigger_time} ({day_pattern}) "
                f"— matches observed activation pattern"
            ),
            "trigger_yaml": yaml_snippet,
            "confidence": min(95, int(discovered_scene.get("confidence", 60))),
            "source": "time_pattern",
        })

    return suggestions


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _improvement_score(
    missing_entity_count: int,
    trigger_count: int,
    base_confidence: int,
) -> int:
    """Compute improvement score 0-100 based on how many enhancements are found."""
    score = base_confidence
    score += min(20, missing_entity_count * 7)
    score += min(10, trigger_count * 5)
    return min(100, score)


def analyse_scenes(
    ha_scenes: list[dict[str, Any]] | None = None,
    discovered_scenes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Analyse HA configured scenes and generate improvement suggestions.

    Args:
        ha_scenes: List of HA scene states (fetched from /api/states).
                   If None, fetches from HA REST API.
        discovered_scenes: Habitus-discovered implicit scenes. If None, loads from disk.

    Returns:
        List of suggestion dicts.
    """
    if ha_scenes is None:
        ha_scenes = fetch_ha_scenes()
    if discovered_scenes is None:
        discovered_scenes = _load_discovered_scenes()

    co_index = _build_co_occurrence_index(discovered_scenes)

    # Build a set of all entities across discovered scenes for quick lookup
    all_discovered_entity_sets = [
        set(s.get("entities", [])) for s in discovered_scenes
    ]

    results: list[dict[str, Any]] = []

    for ha_scene in ha_scenes:
        scene_id = ha_scene.get("entity_id", "")
        scene_name = ha_scene.get("attributes", {}).get("friendly_name", scene_id)
        scene_entities = set(_parse_scene_entities(ha_scene))

        # Find missing entities: co-occurring with scene entities but not in the scene
        missing: dict[str, dict[str, Any]] = {}
        for eid in scene_entities:
            if eid not in co_index:
                continue
            for co_entity, co_count in co_index[eid].most_common(10):
                if co_entity in scene_entities:
                    continue  # already in scene
                domain = co_entity.split(".")[0] if "." in co_entity else ""
                if domain not in SCENE_DOMAINS:
                    continue
                # Compute co-occurrence ratio: fraction of times this entity
                # activated alongside this scene entity
                total_for_eid = sum(co_index[eid].values()) or 1
                ratio = co_count / total_for_eid
                if co_count >= MIN_CO_OCCURRENCE_THRESHOLD and ratio >= MIN_CO_OCCURRENCE_RATIO:
                    existing = missing.get(co_entity)
                    if existing is None or co_count > existing["co_count"]:
                        missing[co_entity] = {
                            "entity_id": co_entity,
                            "co_count": co_count,
                            "ratio": round(ratio, 2),
                            "rationale": (
                                f"Activates together with {eid} in {co_count} "
                                f"observed patterns ({round(ratio * 100)}% of the time)"
                            ),
                            "confidence_pct": round(min(99, ratio * 100)),
                        }

        missing_list = sorted(missing.values(), key=lambda x: -x["co_count"])

        # Find best matching discovered scene for trigger suggestions
        best_match: dict[str, Any] | None = None
        best_overlap = 0
        for ds in discovered_scenes:
            ds_entities = set(ds.get("entities", []))
            overlap = len(scene_entities & ds_entities)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = ds

        suggested_triggers = _suggest_triggers(best_match) if best_match else []

        score = _improvement_score(
            len(missing_list),
            len(suggested_triggers),
            best_match.get("confidence", 40) if best_match else 40,
        )

        why = []
        if missing_list:
            why.append(
                f"{len(missing_list)} entity/entities frequently activated "
                f"with this scene but not included"
            )
        if suggested_triggers:
            why.append(
                f"Activation time pattern detected — automation trigger suggested"
            )
        if not why:
            why.append("No improvement suggestions found based on current learned patterns")

        results.append({
            "scene_id": scene_id,
            "scene_name": scene_name,
            "scene_entities": sorted(scene_entities),
            "missing_entities": missing_list,
            "suggested_triggers": suggested_triggers,
            "improvement_score": score,
            "why_suggested": "; ".join(why),
        })

    # Sort by improvement score descending
    results.sort(key=lambda r: -r["improvement_score"])
    return results


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def load_cached_analysis() -> list[dict[str, Any]] | None:
    """Load cached scene analysis result. Returns None if not found."""
    try:
        if os.path.exists(SCENE_ANALYSIS_PATH):
            with open(SCENE_ANALYSIS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_analysis(results: list[dict[str, Any]]) -> None:
    """Persist scene analysis results to DATA_DIR/scene_analysis.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCENE_ANALYSIS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("scene_analysis: saved %d suggestions", len(results))


def run_scene_analysis(force: bool = False) -> list[dict[str, Any]]:
    """Run scene analysis, using cache unless force=True.

    Called by the full training cycle or on-demand via API.
    """
    if not force:
        cached = load_cached_analysis()
        if cached is not None:
            return cached

    results = analyse_scenes()
    save_analysis(results)
    return results
