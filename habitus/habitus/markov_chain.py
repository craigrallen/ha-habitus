"""Markov Chain for next-action prediction.

"After turning on kitchen lights, what does the user do next 80% of the time?"
Pure Python, no external dependencies. Lightweight but powerful.
"""

import datetime
import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from .ha_db import resolve_ha_db_path

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MARKOV_PATH = os.path.join(DATA_DIR, "markov_model.json")

ACTION_DOMAINS = ("light", "switch", "media_player", "climate", "fan", "cover",
                  "binary_sensor", "input_boolean")
# Max time between events to count as a transition (seconds)
MAX_TRANSITION_SEC = 300  # 5 minutes
# Minimum transitions to include in model
MIN_TRANSITIONS = 3
# Minimum probability to surface as a prediction
MIN_PREDICTION_PROB = 0.3


def build_markov_model(entity_to_area: dict[str, str], days: int = 30) -> dict[str, Any]:
    """Build first-order Markov chain from state change sequences.

    Transition: (entity_A:state_A) → (entity_B:state_B) with probability.
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return {"transitions": {}, "predictions": []}

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    like_clauses = " OR ".join(f"sm.entity_id LIKE '{d}.%'" for d in ACTION_DOMAINS)

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(f"""
            SELECT sm.entity_id, s.state, s.last_changed_ts
            FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE s.last_changed_ts > ?
            AND ({like_clauses})
            AND s.state IN ('on', 'off', 'heat', 'cool', 'auto', 'playing', 'paused',
                            'home', 'not_home', 'open', 'closed')
            ORDER BY s.last_changed_ts
        """, (cutoff_ts,)).fetchall()
        conn.close()
    except Exception as e:
        log.warning("markov: DB query failed: %s", e)
        return {"transitions": {}, "predictions": []}

    if len(rows) < 10:
        return {"transitions": {}, "predictions": []}

    # Build transition counts
    transition_counts: dict[str, Counter] = defaultdict(Counter)
    state_counts: Counter = Counter()

    prev_event = None
    prev_ts = 0.0

    for eid, state, ts in rows:
        event = f"{eid}:{state}"

        if prev_event and (ts - prev_ts) <= MAX_TRANSITION_SEC:
            if event != prev_event:  # Skip self-transitions
                transition_counts[prev_event][event] += 1
                state_counts[prev_event] += 1

        prev_event = event
        prev_ts = ts

    # Convert to probabilities
    transitions: dict[str, list[dict]] = {}
    predictions: list[dict] = []

    for from_event, targets in transition_counts.items():
        total = state_counts[from_event]
        if total < MIN_TRANSITIONS:
            continue

        from_eid, from_state = from_event.rsplit(":", 1)
        from_room = entity_to_area.get(from_eid, "")
        from_name = from_eid.split(".")[-1].replace("_", " ").title()

        top_targets = []
        for to_event, count in targets.most_common(5):
            prob = count / total
            if prob < MIN_PREDICTION_PROB:
                continue

            to_eid, to_state = to_event.rsplit(":", 1)
            to_room = entity_to_area.get(to_eid, "")
            to_name = to_eid.split(".")[-1].replace("_", " ").title()

            entry = {
                "entity_id": to_eid,
                "name": to_name,
                "state": to_state,
                "room": to_room,
                "probability": round(prob, 3),
                "count": count,
            }
            top_targets.append(entry)

            # Surface as a prediction
            if prob >= 0.5 and to_eid.split(".")[0] in ("light", "switch", "media_player", "climate", "fan", "cover"):
                predictions.append({
                    "trigger": {
                        "entity_id": from_eid,
                        "name": from_name,
                        "state": from_state,
                        "room": from_room,
                    },
                    "prediction": entry,
                    "probability": round(prob, 3),
                    "total_observations": total,
                    "description": f"After {from_name} → {from_state}, you usually {to_name} → {to_state} ({prob:.0%})",
                    "suggestion": f"You just turned {from_state} {from_name} — want me to set {to_name} to {to_state}?",
                })

        if top_targets:
            transitions[from_event] = top_targets

    predictions.sort(key=lambda p: -p["probability"])

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total_events": len(rows),
        "transition_states": len(transitions),
        "prediction_count": len(predictions),
        "transitions": transitions,
        "predictions": predictions[:30],
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MARKOV_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("markov: %d transition states, %d actionable predictions from %d events",
             len(transitions), len(predictions), len(rows))
    return result
