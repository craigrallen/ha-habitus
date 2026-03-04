"""Room-aware predictive automation.

When a user enters a room, predict what they'll want based on:
- Historical patterns: what they usually do in this room at this time
- Day of week: weekday vs weekend routines differ
- Recent context: what they were doing before (sequence)
- Season/weather: heating needs change with temperature

Flow:
1. Motion/presence detects room entry
2. Query: "When user enters [room] at [time] on [day], what happens next?"
3. Build conditional probability: P(action | room, time, day)
4. If confidence > threshold → send actionable HA notification
5. User approves → actions execute. User ignores → we learn from that too.

Uses a Markov chain approach: given current state (room + time + day),
what are the most likely next entity state changes?
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
PREDICTIONS_PATH = os.path.join(DATA_DIR, "room_predictions.json")
PREDICTION_LOG_PATH = os.path.join(DATA_DIR, "prediction_log.json")

# Minimum times a pattern must occur to be suggested
MIN_OCCURRENCES = 3
# Minimum probability to trigger a suggestion
MIN_CONFIDENCE = 0.6
# Time window: actions within N minutes of room entry count as "caused by" the entry
ACTION_WINDOW_MIN = 10
# How many days of history to analyse
DEFAULT_DAYS = 30


def _get_room_entry_events(entity_to_area: dict[str, str], days: int = DEFAULT_DAYS) -> list[dict]:
    """Find room entry events from motion sensors, door sensors, and presence.

    A "room entry" = motion sensor turns on, or door sensor opens,
    in a room with a known HA area assignment.
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    # Find motion/presence sensors that are assigned to rooms
    trigger_entities = {}
    for eid, area in entity_to_area.items():
        if eid.startswith("binary_sensor."):
            eid_lower = eid.lower()
            if any(kw in eid_lower for kw in ("motion", "occupancy", "presence", "pir")):
                trigger_entities[eid] = area

    if not trigger_entities:
        log.info("No motion/presence sensors with area assignments found")
        return []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None

        entries = []
        for eid, area in trigger_entities.items():
            if has_meta:
                rows = conn.execute("""
                    SELECT s.state, s.last_changed_ts
                    FROM states s
                    JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                    WHERE sm.entity_id = ? AND s.last_changed_ts > ? AND s.state = 'on'
                    ORDER BY s.last_changed_ts
                """, (eid, cutoff_ts)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT state, last_changed_ts
                    FROM states WHERE entity_id = ? AND last_changed_ts > ? AND state = 'on'
                    ORDER BY last_changed_ts
                """, (eid, cutoff_ts)).fetchall()

            for _, ts in rows:
                dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
                entries.append({
                    "room": area,
                    "timestamp": ts,
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                    "is_weekend": dt.weekday() >= 5,
                    "trigger_entity": eid,
                })

        conn.close()
        entries.sort(key=lambda e: e["timestamp"])
        log.info("Found %d room entry events across %d trigger sensors", len(entries), len(trigger_entities))
        return entries
    except Exception as e:
        log.warning("Failed to get room entry events: %s", e)
        return []


def _get_actions_after_entry(entry_ts: float, room: str, entity_to_area: dict[str, str]) -> list[dict]:
    """Find entity state changes that happened within ACTION_WINDOW_MIN after a room entry.

    Only includes entities in the same room (or unassigned entities matching room keywords).
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return []

    window_end = entry_ts + ACTION_WINDOW_MIN * 60

    # Get entities in this room
    room_entities = {eid for eid, area in entity_to_area.items() if area == room}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None

        if has_meta:
            rows = conn.execute("""
                SELECT sm.entity_id, s.state, s.last_changed_ts
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE s.last_changed_ts > ? AND s.last_changed_ts <= ?
                AND (
                    sm.entity_id LIKE 'light.%'
                    OR sm.entity_id LIKE 'switch.%'
                    OR sm.entity_id LIKE 'media_player.%'
                    OR sm.entity_id LIKE 'climate.%'
                    OR sm.entity_id LIKE 'fan.%'
                    OR sm.entity_id LIKE 'cover.%'
                )
                ORDER BY s.last_changed_ts
            """, (entry_ts, window_end)).fetchall()
        else:
            rows = conn.execute("""
                SELECT entity_id, state, last_changed_ts
                FROM states
                WHERE last_changed_ts > ? AND last_changed_ts <= ?
                AND (
                    entity_id LIKE 'light.%'
                    OR entity_id LIKE 'switch.%'
                    OR entity_id LIKE 'media_player.%'
                    OR entity_id LIKE 'climate.%'
                    OR entity_id LIKE 'fan.%'
                    OR entity_id LIKE 'cover.%'
                )
                ORDER BY last_changed_ts
            """, (entry_ts, window_end)).fetchall()

        conn.close()

        actions = []
        for eid, state, ts in rows:
            # Only include entities in the same room
            if eid in room_entities and state in ("on", "off", "heat", "cool", "auto", "playing"):
                actions.append({
                    "entity_id": eid,
                    "state": state,
                    "delay_sec": round(ts - entry_ts),
                })
        return actions
    except Exception as e:
        log.warning("Failed to get post-entry actions: %s", e)
        return []


def build_room_model(entity_to_area: dict[str, str], days: int = DEFAULT_DAYS) -> dict[str, Any]:
    """Build the predictive model: P(actions | room, time_slot, day_type).

    For each room, groups entry events by time slot (2-hour windows) and
    day type (weekday/weekend), then counts which actions follow most often.
    """
    entries = _get_room_entry_events(entity_to_area, days=days)
    if not entries:
        return {"rooms": {}, "total_entries": 0}

    # Sample entries to avoid processing millions (take every Nth)
    if len(entries) > 5000:
        step = len(entries) // 5000
        entries = entries[::step]

    # Build conditional action counts
    # Key: (room, time_slot, is_weekend)
    # Value: Counter of (entity_id, state) tuples
    action_counts: dict[tuple, Counter] = defaultdict(Counter)
    entry_counts: dict[tuple, int] = defaultdict(int)

    for entry in entries:
        room = entry["room"]
        time_slot = entry["hour"] // 2  # 2-hour windows: 0-1, 2-3, ..., 22-23
        is_weekend = entry["is_weekend"]
        key = (room, time_slot, is_weekend)

        entry_counts[key] += 1

        actions = _get_actions_after_entry(entry["timestamp"], room, entity_to_area)
        for action in actions:
            action_key = (action["entity_id"], action["state"])
            action_counts[key][action_key] += 1

    # Convert to probability model
    rooms_model: dict[str, list[dict]] = defaultdict(list)

    for (room, time_slot, is_weekend), counter in action_counts.items():
        n_entries = entry_counts[(room, time_slot, is_weekend)]
        if n_entries < MIN_OCCURRENCES:
            continue

        hour_start = time_slot * 2
        hour_end = hour_start + 2
        day_type = "weekend" if is_weekend else "weekday"

        # Get top actions by probability
        top_actions = []
        for (eid, state), count in counter.most_common(10):
            probability = count / n_entries
            if probability >= MIN_CONFIDENCE:
                eid_name = eid.split(".")[-1].replace("_", " ").title()
                top_actions.append({
                    "entity_id": eid,
                    "name": eid_name,
                    "state": state,
                    "probability": round(probability, 2),
                    "occurrences": count,
                    "total_entries": n_entries,
                })

        if top_actions:
            rooms_model[room].append({
                "time_window": f"{hour_start:02d}:00–{hour_end:02d}:00",
                "hour_start": hour_start,
                "hour_end": hour_end,
                "day_type": day_type,
                "entry_count": n_entries,
                "predicted_actions": top_actions,
                "description": _build_prediction_description(room, top_actions, hour_start, day_type),
            })

    return {
        "rooms": dict(rooms_model),
        "total_entries": len(entries),
        "total_predictions": sum(len(v) for v in rooms_model.values()),
    }


def _build_prediction_description(room: str, actions: list[dict], hour: int, day_type: str) -> str:
    """Build human-readable prediction description."""
    time_labels = {
        0: "late night", 2: "early morning", 4: "dawn", 6: "morning",
        8: "mid-morning", 10: "late morning", 12: "midday", 14: "afternoon",
        16: "late afternoon", 18: "evening", 20: "late evening", 22: "night",
    }
    time_label = time_labels.get(hour, "")
    action_strs = [f"{a['name']} → {a['state']} ({a['probability']:.0%})" for a in actions[:3]]
    return f"When you enter {room} on {day_type} {time_label}s, you usually: {', '.join(action_strs)}"


def generate_prediction_automations(model: dict[str, Any], entity_to_area: dict[str, str]) -> list[dict]:
    """Generate HA automations that predict user needs on room entry.

    Each automation:
    - Triggers on motion/presence in the room
    - Checks time window and day type
    - Sends an actionable notification: "Entering Kitchen at 18:00 — turn on lights and extractor?"
    - If approved, executes the actions
    """
    automations = []

    for room, predictions in model.get("rooms", {}).items():
        # Find motion sensors for this room
        triggers = [
            eid for eid, area in entity_to_area.items()
            if area == room and eid.startswith("binary_sensor.") and
            any(kw in eid.lower() for kw in ("motion", "occupancy", "presence", "pir"))
        ]
        if not triggers:
            continue

        for pred in predictions:
            actions = pred["predicted_actions"]
            if not actions:
                continue

            safe_room = room.lower().replace(" ", "_")
            time_slot = f"{pred['hour_start']:02d}_{pred['hour_end']:02d}"
            day_type = pred["day_type"]

            # Build action descriptions for notification
            action_descs = [f"{a['name']} → {a['state']}" for a in actions[:5]]
            msg = f"Entering {room} — want me to: {', '.join(action_descs)}?"

            # Build YAML for the predicted actions
            action_yaml = ""
            for a in actions:
                domain = a["entity_id"].split(".")[0]
                if a["state"] in ("on", "heat", "auto", "playing"):
                    service = f"{domain}.turn_on"
                else:
                    service = f"{domain}.turn_off"
                action_yaml += f"""
    - service: {service}
      target:
        entity_id: {a['entity_id']}"""

            # Condition for time window and day type
            day_condition = ""
            if day_type == "weekday":
                day_condition = """
    - condition: time
      weekday: [mon, tue, wed, thu, fri]"""
            elif day_type == "weekend":
                day_condition = """
    - condition: time
      weekday: [sat, sun]"""

            yaml = f"""automation:
  alias: "Habitus Predict — {room} ({pred['time_window']} {day_type})"
  description: "{pred['description']}"
  trigger:
    - platform: state
      entity_id: {triggers[0]}
      to: "on"
  condition:
    - condition: time
      after: "{pred['hour_start']:02d}:00:00"
      before: "{pred['hour_end']:02d}:00:00"{day_condition}
  action:
    - service: notify.notify
      data:
        title: "🧠 Habitus — {room}"
        message: "{msg}"
        data:
          actions:
            - action: "habitus_approve_{safe_room}_{time_slot}"
              title: "✅ Yes, do it"
            - action: "habitus_dismiss_{safe_room}_{time_slot}"
              title: "❌ Not now"
"""

            avg_confidence = sum(a["probability"] for a in actions) / len(actions)
            automations.append({
                "id": f"predict_{safe_room}_{time_slot}_{day_type}",
                "room": room,
                "time_window": pred["time_window"],
                "day_type": day_type,
                "actions": actions,
                "description": pred["description"],
                "confidence": round(avg_confidence * 100),
                "entry_count": pred["entry_count"],
                "yaml": yaml,
                "action_yaml": action_yaml.strip(),
                "category": "predictive",
            })

    automations.sort(key=lambda a: -a["confidence"])
    return automations


def run_room_prediction(entity_to_area: dict[str, str], days: int = DEFAULT_DAYS) -> dict[str, Any]:
    """Run full room prediction pipeline."""
    model = build_room_model(entity_to_area, days=days)
    automations = generate_prediction_automations(model, entity_to_area)

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "model": model,
        "automations": automations,
        "room_count": len(model.get("rooms", {})),
        "prediction_count": len(automations),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if automations:
        log.info("Room predictor: %d predictions across %d rooms from %d entry events",
                 len(automations), len(model.get("rooms", {})), model.get("total_entries", 0))
    return result


def load_predictions() -> dict[str, Any]:
    """Load cached predictions."""
    try:
        if os.path.exists(PREDICTIONS_PATH):
            with open(PREDICTIONS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"automations": [], "prediction_count": 0}
