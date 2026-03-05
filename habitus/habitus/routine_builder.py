"""Routine Builder — mine temporal sequences from learned patterns.

Detects: entity A changes → within N minutes → entity B changes → entity C changes
Minimum sequence length: 3 steps.
Minimum frequency: occurred on 5+ separate days.
Clusters sequences by time-of-day: morning, evening, night, arrival, departure.
Generates HA script/automation YAML for each detected routine.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from collections import defaultdict
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

ROUTINES_PATH = os.path.join(DATA_DIR, "routines.json")
EVENTS_PATH = os.path.join(DATA_DIR, "events.json")

# Sequence mining config
MAX_STEP_GAP_MINUTES = 10    # max time gap between consecutive steps
MIN_SEQUENCE_LENGTH = 3      # minimum steps in a routine
MIN_FREQUENCY_DAYS = 5       # must occur on at least 5 separate days
MAX_SEQUENCE_LENGTH = 8      # cap to avoid noise

# Time clusters (hour ranges, inclusive)
TIME_CLUSTERS = {
    "morning": (5, 9),
    "evening": (17, 22),
    "night": (22, 25),  # 22-01 (25 = 1am next day)
    "midday": (11, 14),
}


def _classify_time_cluster(hour: int) -> str:
    """Classify an hour into a time-of-day cluster."""
    for cluster, (start, end) in TIME_CLUSTERS.items():
        if end > 24:
            if hour >= start or hour < (end - 24):
                return cluster
        elif start <= hour < end:
            return cluster
    return "other"


def _detect_arrival_departure(steps: list[dict[str, Any]]) -> str | None:
    """Detect if a sequence looks like arrival or departure."""
    entities = [s.get("entity_id", "") for s in steps]
    entity_str = " ".join(entities).lower()
    states = [s.get("state", "") for s in steps]

    has_presence = any("person." in e or "presence" in e or "occupancy" in e for e in entities)
    has_door = any("door" in e for e in entities)

    if has_presence:
        # Check if person goes to 'home' (arrival) or 'not_home' (departure)
        for step in steps:
            if "person." in step.get("entity_id", ""):
                if step.get("state", "") == "home":
                    return "arrival"
                elif step.get("state", "") in ("not_home", "away"):
                    return "departure"
    return None


def mine_sequences(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mine temporal sequences from a list of entity state-change events.

    Args:
        events: List of {entity_id, state, timestamp} dicts, sorted by timestamp.

    Returns:
        List of detected sequence patterns with frequency and time clustering.
    """
    if not events:
        return []

    # Sort by timestamp
    try:
        events_sorted = sorted(events, key=lambda e: e.get("timestamp", ""))
    except TypeError:
        events_sorted = events

    # Group events by day
    days_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events_sorted:
        ts = event.get("timestamp", "")
        if ts:
            try:
                if isinstance(ts, str):
                    dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
                else:
                    dt = ts
                day_key = dt.strftime("%Y-%m-%d")
                days_events[day_key].append({**event, "_dt": dt})
            except (ValueError, TypeError):
                pass

    # Find sequences within each day using sliding window
    # Key: tuple of (entity_id, state) pairs → list of (day, start_hour)
    sequence_occurrences: dict[tuple, list[tuple[str, int]]] = defaultdict(list)

    for day, day_evts in days_events.items():
        day_evts_sorted = sorted(day_evts, key=lambda e: e["_dt"])
        n = len(day_evts_sorted)

        for i in range(n - MIN_SEQUENCE_LENGTH + 1):
            start_evt = day_evts_sorted[i]
            start_dt = start_evt["_dt"]
            sequence = [(start_evt.get("entity_id", ""), start_evt.get("state", ""))]

            for j in range(i + 1, min(i + MAX_SEQUENCE_LENGTH, n)):
                evt = day_evts_sorted[j]
                evt_dt = evt["_dt"]
                gap_minutes = (evt_dt - start_dt).total_seconds() / 60
                if gap_minutes > MAX_STEP_GAP_MINUTES * (len(sequence)):
                    break
                sequence.append((evt.get("entity_id", ""), evt.get("state", "")))

                if len(sequence) >= MIN_SEQUENCE_LENGTH:
                    seq_key = tuple(sequence[:])
                    sequence_occurrences[seq_key].append((day, start_dt.hour))

    # Filter by minimum frequency
    detected = []
    seen_keys: set[tuple] = set()

    for seq_key, occurrences in sequence_occurrences.items():
        unique_days = set(day for day, _ in occurrences)
        if len(unique_days) < MIN_FREQUENCY_DAYS:
            continue

        # Avoid duplicate sub-sequences
        skip = False
        for existing_key in seen_keys:
            if len(existing_key) > len(seq_key):
                # Check if seq_key is a prefix of existing_key
                if existing_key[:len(seq_key)] == seq_key:
                    skip = True
                    break
        if skip:
            continue

        seen_keys.add(seq_key)

        # Calculate time cluster
        hours = [hour for _, hour in occurrences]
        avg_hour = sum(hours) / len(hours) if hours else 12
        time_cluster = _classify_time_cluster(int(avg_hour))

        # Check for arrival/departure
        steps = [{"entity_id": s[0], "state": s[1]} for s in seq_key]
        special_cluster = _detect_arrival_departure(steps)
        if special_cluster:
            time_cluster = special_cluster

        confidence = min(0.95, 0.5 + (len(unique_days) - MIN_FREQUENCY_DAYS) * 0.05)

        detected.append({
            "steps": steps,
            "frequency_days": len(unique_days),
            "time_cluster": time_cluster,
            "avg_hour": round(avg_hour, 1),
            "confidence": round(confidence, 2),
        })

    # Sort by frequency descending
    detected.sort(key=lambda r: r["frequency_days"], reverse=True)
    return detected[:50]  # cap at 50 routines


def generate_routine_yaml(routine: dict[str, Any]) -> str:
    """Generate HA script YAML for a detected routine."""
    steps = routine.get("steps", [])
    time_cluster = routine.get("time_cluster", "routine")
    avg_hour = routine.get("avg_hour", 12)

    alias = f"Habitus Routine — {time_cluster.title()} sequence"

    actions = ""
    for i, step in enumerate(steps):
        eid = step.get("entity_id", "")
        state = step.get("state", "")
        domain = eid.split(".")[0] if "." in eid else "homeassistant"

        if state in ("on", "open", "home"):
            service = f"{domain}.turn_on"
        elif state in ("off", "closed", "not_home", "away"):
            service = f"{domain}.turn_off"
        else:
            service = f"{domain}.turn_on"

        delay = "" if i == 0 else "\n    - delay: '00:00:30'"
        actions += f"{delay}\n    - service: {service}\n      target:\n        entity_id: {eid}"

    # Use time trigger based on avg_hour
    hour = int(avg_hour)
    minute = int((avg_hour - hour) * 60)
    time_str = f"{hour:02d}:{minute:02d}:00"

    return f"""script:
  {alias.lower().replace(' ', '_').replace('—', '').replace('  ', '_')}:
    alias: "{alias}"
    description: "Auto-generated by Habitus routine builder (confidence: {routine.get('confidence', 0):.0%}, seen {routine.get('frequency_days', 0)} days)"
    sequence:{actions}

# To trigger automatically at typical time:
automation:
  alias: "Run {alias}"
  trigger:
    - platform: time
      at: "{time_str}"
  action:
    - service: script.turn_on
      data:
        entity_id: script.{alias.lower().replace(' ', '_').replace('—', '').replace('  ', '_')}
"""


def run(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run routine builder and save results.

    Args:
        events: Optional list of entity events. If None, loads from EVENTS_PATH.

    Returns:
        Dict with routines list.
    """
    if events is None:
        events = []
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "events.json")):
            try:
                with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "events.json")) as f:
                    data = json.load(f)
                    events = data if isinstance(data, list) else data.get("events", [])
            except Exception as e:
                log.warning("Failed to load events: %s", e)

    sequences = mine_sequences(events)

    routines = []
    for seq in sequences:
        yaml_str = generate_routine_yaml(seq)
        routines.append({
            **seq,
            "generated_yaml": yaml_str,
        })

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total": len(routines),
        "routines": routines,
    }

    from .utils import atomic_write as _atomic_write  # noqa: PLC0415
    _atomic_write(os.path.join(os.environ.get("DATA_DIR", "/data"), "routines.json"), result)

    log.info("Routine builder: %d routines detected", len(routines))
    return result


def load_routines() -> dict[str, Any]:
    """Load cached routines."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "routines.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "routines.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {"total": 0, "routines": []}
