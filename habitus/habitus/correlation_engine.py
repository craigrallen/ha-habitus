"""Deep correlation mining across all sensor data.

Finds statistically significant relationships between entity state changes:
- Temporal correlations: A happens → B follows within N minutes
- Co-occurrence: A and B change together regularly
- Conditional: A happens when condition C is true
- Causal chains: A → B → C sequences

This is intentionally intensive — it's where the gold is.
Runs in the training thread to avoid blocking the web UI.
"""

import datetime
import json
import logging
import math
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from .ha_db import resolve_ha_db_path

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
CORRELATIONS_PATH = os.path.join(DATA_DIR, "correlations.json")

# Only analyse actionable domains (skip sensors — they're triggers, not actions)
ACTION_DOMAINS = ("light", "switch", "media_player", "climate", "fan", "cover", "input_boolean")
TRIGGER_DOMAINS = ("binary_sensor", "sensor", "person", "device_tracker")

# Temporal window for "A caused B"
CAUSE_WINDOW_SEC = 600  # 10 minutes
# Minimum co-occurrences to count as a correlation
MIN_COOCCURRENCES = 5
# Minimum lift (how much more likely than random) to be interesting
MIN_LIFT = 2.0
# Maximum entities to analyse (memory/time constraint for Odroid)
MAX_ENTITIES = 200
# Batch size for DB queries
BATCH_DAYS = 7


def _get_state_change_events(days: int = 30) -> dict[str, list[tuple[float, str]]]:
    """Get all state change events grouped by entity_id.

    Returns: {entity_id: [(timestamp, new_state), ...]}
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return {}

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    all_domains = ACTION_DOMAINS + TRIGGER_DOMAINS
    like_clauses = " OR ".join(f"sm.entity_id LIKE '{d}.%'" for d in all_domains)

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None

        events: dict[str, list[tuple[float, str]]] = defaultdict(list)

        if has_meta:
            rows = conn.execute(f"""
                SELECT sm.entity_id, s.state, s.last_changed_ts
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE s.last_changed_ts > ?
                AND ({like_clauses})
                AND s.state NOT IN ('unavailable', 'unknown', '')
                ORDER BY s.last_changed_ts
            """, (cutoff_ts,)).fetchall()
        else:
            like_clauses_plain = " OR ".join(f"entity_id LIKE '{d}.%'" for d in all_domains)
            rows = conn.execute(f"""
                SELECT entity_id, state, last_changed_ts
                FROM states
                WHERE last_changed_ts > ?
                AND ({like_clauses_plain})
                AND state NOT IN ('unavailable', 'unknown', '')
                ORDER BY last_changed_ts
            """, (cutoff_ts,)).fetchall()

        conn.close()

        for eid, state, ts in rows:
            events[eid].append((ts, state))

        log.info("Correlation engine: loaded %d events across %d entities",
                 sum(len(v) for v in events.values()), len(events))
        return dict(events)
    except Exception as e:
        log.warning("Failed to load state events: %s", e)
        return {}


def _compute_temporal_correlations(
    events: dict[str, list[tuple[float, str]]],
    entity_to_area: dict[str, str],
) -> list[dict[str, Any]]:
    """Find pairs where A changing state is followed by B changing state within CAUSE_WINDOW_SEC.

    Uses lift (observed / expected) to filter out noise.
    """
    # Focus on entities with enough activity
    active_entities = {
        eid: evts for eid, evts in events.items()
        if len(evts) >= MIN_COOCCURRENCES
    }

    # Limit to MAX_ENTITIES most active
    if len(active_entities) > MAX_ENTITIES:
        sorted_by_activity = sorted(active_entities.items(), key=lambda x: -len(x[1]))
        active_entities = dict(sorted_by_activity[:MAX_ENTITIES])

    entity_ids = list(active_entities.keys())
    n_entities = len(entity_ids)
    total_period = 0

    # Calculate total observation period
    all_ts = [ts for evts in active_entities.values() for ts, _ in evts]
    if all_ts:
        total_period = max(all_ts) - min(all_ts)
    if total_period <= 0:
        return []

    # For each entity, create a set of "active timestamps" (rounded to minute)
    entity_minutes: dict[str, set[int]] = {}
    for eid, evts in active_entities.items():
        entity_minutes[eid] = {int(ts // 60) for ts, _ in evts}

    # Pairwise correlation via temporal proximity
    correlations = []
    total_pairs = n_entities * (n_entities - 1) // 2
    log.info("Correlation engine: analysing %d entity pairs (%d entities)", total_pairs, n_entities)

    # Pre-compute event rates (events per minute)
    event_rates = {eid: len(evts) / (total_period / 60) for eid, evts in active_entities.items()}

    checked = 0
    window_minutes = CAUSE_WINDOW_SEC // 60

    for i, eid_a in enumerate(entity_ids):
        minutes_a = entity_minutes[eid_a]
        rate_a = event_rates[eid_a]

        for j in range(i + 1, n_entities):
            eid_b = entity_ids[j]
            minutes_b = entity_minutes[eid_b]
            rate_b = event_rates[eid_b]

            # Count A→B: how many times B fires within window_minutes after A
            a_then_b = 0
            for m in minutes_a:
                for offset in range(1, window_minutes + 1):
                    if (m + offset) in minutes_b:
                        a_then_b += 1
                        break

            if a_then_b < MIN_COOCCURRENCES:
                continue

            # Expected co-occurrences if independent
            expected = len(minutes_a) * rate_b * (window_minutes)
            if expected <= 0:
                continue

            lift = a_then_b / expected
            if lift < MIN_LIFT:
                continue

            # Confidence: P(B follows | A happened)
            confidence = a_then_b / len(minutes_a)

            room_a = entity_to_area.get(eid_a, "")
            room_b = entity_to_area.get(eid_b, "")

            correlations.append({
                "entity_a": eid_a,
                "entity_b": eid_b,
                "name_a": eid_a.split(".")[-1].replace("_", " ").title(),
                "name_b": eid_b.split(".")[-1].replace("_", " ").title(),
                "room_a": room_a,
                "room_b": room_b,
                "direction": "a_then_b",
                "cooccurrences": a_then_b,
                "total_a": len(minutes_a),
                "confidence": round(confidence, 3),
                "lift": round(lift, 2),
                "same_room": room_a == room_b and room_a != "",
                "cross_room": room_a != room_b and room_a != "" and room_b != "",
            })

            checked += 1

    correlations.sort(key=lambda c: -c["lift"])
    log.info("Correlation engine: found %d significant correlations from %d pairs", len(correlations), checked)
    return correlations


def _classify_correlation(corr: dict) -> str:
    """Classify what kind of relationship this correlation represents."""
    a_domain = corr["entity_a"].split(".")[0]
    b_domain = corr["entity_b"].split(".")[0]

    if a_domain in TRIGGER_DOMAINS and b_domain in ACTION_DOMAINS:
        return "trigger_action"  # Sensor triggers device
    if a_domain in ACTION_DOMAINS and b_domain in ACTION_DOMAINS:
        if corr["same_room"]:
            return "room_routine"  # Same room, both actions
        return "cross_room_routine"  # Different rooms
    if a_domain == "person" or "presence" in corr["entity_a"]:
        return "presence_driven"
    if "temperature" in corr["entity_a"] or "humidity" in corr["entity_a"]:
        return "climate_response"
    return "general"


def _build_automation_suggestion(corr: dict) -> dict[str, Any] | None:
    """Generate an automation suggestion from a strong correlation."""
    if corr["confidence"] < 0.5 or corr["lift"] < 3.0:
        return None

    b_domain = corr["entity_b"].split(".")[0]
    if b_domain not in ACTION_DOMAINS:
        return None

    category = _classify_correlation(corr)
    desc_map = {
        "trigger_action": f"When {corr['name_a']} activates, {corr['name_b']} usually follows ({corr['confidence']:.0%} of the time)",
        "room_routine": f"In {corr['room_a']}: {corr['name_a']} and {corr['name_b']} go together ({corr['confidence']:.0%})",
        "cross_room_routine": f"{corr['name_a']} ({corr['room_a']}) → {corr['name_b']} ({corr['room_b']}) ({corr['confidence']:.0%})",
        "presence_driven": f"When {corr['name_a']} detects presence, {corr['name_b']} follows ({corr['confidence']:.0%})",
        "climate_response": f"Climate change in {corr['name_a']} correlates with {corr['name_b']} ({corr['confidence']:.0%})",
        "general": f"{corr['name_a']} → {corr['name_b']} ({corr['confidence']:.0%}, {corr['lift']}× more likely than random)",
    }

    # Generate YAML
    a_domain = corr["entity_a"].split(".")[0]
    trigger_yaml = ""
    if a_domain == "binary_sensor":
        trigger_yaml = f"""    - platform: state
      entity_id: {corr['entity_a']}
      to: "on" """
    elif a_domain in ACTION_DOMAINS:
        trigger_yaml = f"""    - platform: state
      entity_id: {corr['entity_a']}
      to: "on" """
    elif a_domain == "sensor":
        trigger_yaml = f"""    - platform: state
      entity_id: {corr['entity_a']}"""
    elif a_domain == "person":
        trigger_yaml = f"""    - platform: state
      entity_id: {corr['entity_a']}
      to: "home" """
    else:
        trigger_yaml = f"""    - platform: state
      entity_id: {corr['entity_a']}"""

    yaml = f"""automation:
  alias: "Habitus Correlation — {corr['name_a']} → {corr['name_b']}"
  description: "{desc_map.get(category, desc_map['general'])}"
  trigger:
{trigger_yaml}
  action:
    - service: notify.notify
      data:
        title: "🔗 Habitus — Correlation detected"
        message: "{corr['name_a']} just activated. Turn on {corr['name_b']}?"
        data:
          actions:
            - action: "habitus_corr_approve_{corr['entity_b'].replace('.','_')}"
              title: "✅ Yes"
            - action: "habitus_corr_dismiss_{corr['entity_b'].replace('.','_')}"
              title: "❌ No"
"""

    return {
        "correlation": corr,
        "category": category,
        "description": desc_map.get(category, desc_map["general"]),
        "yaml": yaml,
        "confidence": round(corr["confidence"] * 100),
        "lift": corr["lift"],
    }


def run_correlation_analysis(entity_to_area: dict[str, str], days: int = 30) -> dict[str, Any]:
    """Run full correlation mining pipeline.

    This is CPU-intensive — should run in background training thread.
    """
    log.info("Starting deep correlation analysis (%d days)...", days)
    events = _get_state_change_events(days=days)
    if not events:
        return {"correlations": [], "suggestions": []}

    correlations = _compute_temporal_correlations(events, entity_to_area)

    # Classify and generate suggestions
    for corr in correlations:
        corr["category"] = _classify_correlation(corr)

    suggestions = []
    for corr in correlations:
        sug = _build_automation_suggestion(corr)
        if sug:
            suggestions.append(sug)

    # Top insights
    top_by_lift = correlations[:20]
    cross_room = [c for c in correlations if c.get("cross_room")][:10]
    climate_driven = [c for c in correlations if c.get("category") == "climate_response"][:10]

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total_correlations": len(correlations),
        "actionable_suggestions": len(suggestions),
        "top_correlations": top_by_lift,
        "cross_room_patterns": cross_room,
        "climate_responses": climate_driven,
        "suggestions": suggestions[:30],  # Cap at 30 most useful
        "stats": {
            "entities_analysed": len(events),
            "total_events": sum(len(v) for v in events.values()),
            "days": days,
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CORRELATIONS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("Correlation analysis complete: %d correlations, %d suggestions",
             len(correlations), len(suggestions))
    return result


def load_correlations() -> dict[str, Any]:
    """Load cached correlations."""
    try:
        if os.path.exists(CORRELATIONS_PATH):
            with open(CORRELATIONS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"correlations": [], "suggestions": [], "total_correlations": 0}
