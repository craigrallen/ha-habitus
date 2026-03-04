"""Routine prediction from environmental signals.

Detects recurring events from indirect sensor data and suggests
pre-emptive automations:

- **Shower/bath detection**: Humidity spike >10% in bathroom → shower event
  → Learn typical shower time → suggest pre-heating water 1 hour before
- **Cooking detection**: Kitchen temperature/humidity rise → cooking event
  → Learn meal times → suggest turning on extractor fan
- **Sleep detection**: Bedroom lights off + no motion + phone charging
  → bedtime → suggest night mode automation
- **Wake detection**: First motion + lights on after sleep period
  → wake time → suggest morning routine (lights, heating, coffee)

All detection is from indirect signals — no cameras, no microphones.
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
ROUTINES_PATH = os.path.join(DATA_DIR, "routines.json")

# Humidity spike threshold (percentage points above rolling average)
HUMIDITY_SPIKE_THRESHOLD = 10.0
# Minimum spike duration (minutes) to count as a shower
MIN_SHOWER_DURATION_MIN = 5
# Maximum spike duration — longer is probably not a shower
MAX_SHOWER_DURATION_MIN = 45
# How far ahead to suggest pre-heating (minutes)
PREHEAT_MINUTES = 60

# Room keywords
BATHROOM_KEYWORDS = ("bathroom", "bath", "shower", "ensuite", "wc", "toilet", "washroom", "badrum", "dusch")
KITCHEN_KEYWORDS = ("kitchen", "galley", "kök")
BEDROOM_KEYWORDS = ("bedroom", "bed_room", "master_bed", "sovrum")


def _find_humidity_sensors() -> dict[str, str]:
    """Find humidity sensors and classify by room.

    Returns dict of {entity_id: room_type} where room_type is
    'bathroom', 'kitchen', 'bedroom', or 'other'.
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return {}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None

        if has_meta:
            rows = conn.execute("""
                SELECT DISTINCT sm.entity_id
                FROM states_meta sm
                WHERE (sm.entity_id LIKE 'sensor.%humidity%'
                    OR sm.entity_id LIKE 'sensor.%hum%')
                AND sm.entity_id NOT LIKE '%battery%'
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT DISTINCT entity_id
                FROM states
                WHERE (entity_id LIKE 'sensor.%humidity%'
                    OR entity_id LIKE 'sensor.%hum%')
                AND entity_id NOT LIKE '%battery%'
                LIMIT 30
            """).fetchall()
        conn.close()

        sensors = {}
        for (eid,) in rows:
            eid_lower = eid.lower()
            if any(kw in eid_lower for kw in BATHROOM_KEYWORDS):
                sensors[eid] = "bathroom"
            elif any(kw in eid_lower for kw in KITCHEN_KEYWORDS):
                sensors[eid] = "kitchen"
            elif any(kw in eid_lower for kw in BEDROOM_KEYWORDS):
                sensors[eid] = "bedroom"
            else:
                sensors[eid] = "other"

        log.info("Found %d humidity sensors: %s", len(sensors),
                 {v: sum(1 for x in sensors.values() if x == v) for v in set(sensors.values())})
        return sensors
    except Exception as e:
        log.warning("Failed to find humidity sensors: %s", e)
        return {}


def _get_humidity_history(entity_id: str, days: int = 30) -> list[tuple[float, float]]:
    """Get humidity history as [(timestamp, value), ...] from HA database."""
    db_path = resolve_ha_db_path()
    if not db_path:
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None

        if has_meta:
            rows = conn.execute("""
                SELECT s.state, s.last_changed_ts
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                ORDER BY s.last_changed_ts
            """, (entity_id, cutoff_ts)).fetchall()
        else:
            rows = conn.execute("""
                SELECT state, last_changed_ts
                FROM states
                WHERE entity_id = ? AND last_changed_ts > ?
                ORDER BY last_changed_ts
            """, (entity_id, cutoff_ts)).fetchall()
        conn.close()

        readings = []
        for state_val, ts in rows:
            try:
                val = float(state_val)
                if 0 <= val <= 100:  # valid humidity range
                    readings.append((ts, val))
            except (ValueError, TypeError):
                continue
        return readings
    except Exception as e:
        log.warning("Failed to get humidity history for %s: %s", entity_id, e)
        return []


def detect_humidity_spikes(readings: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Detect humidity spikes indicating shower/bath/cooking events.

    Uses a rolling baseline (1-hour trailing average) and looks for
    readings that exceed the baseline by HUMIDITY_SPIKE_THRESHOLD.
    """
    if len(readings) < 20:
        return []

    spikes = []
    in_spike = False
    spike_start = None
    spike_peak = 0.0
    baseline = readings[0][1]  # initial baseline

    for i, (ts, val) in enumerate(readings):
        # Update rolling baseline (exponential moving average, slow)
        baseline = baseline * 0.98 + val * 0.02

        delta = val - baseline

        if not in_spike and delta > HUMIDITY_SPIKE_THRESHOLD:
            in_spike = True
            spike_start = ts
            spike_peak = val

        elif in_spike:
            if val > spike_peak:
                spike_peak = val

            # Spike ends when humidity drops back near baseline
            if delta < HUMIDITY_SPIKE_THRESHOLD * 0.3:
                duration_min = (ts - spike_start) / 60 if spike_start else 0
                if MIN_SHOWER_DURATION_MIN <= duration_min <= MAX_SHOWER_DURATION_MIN:
                    spike_dt = datetime.datetime.fromtimestamp(spike_start, tz=datetime.UTC)
                    spikes.append({
                        "start_ts": spike_start,
                        "end_ts": ts,
                        "start": spike_dt.isoformat(),
                        "hour": spike_dt.hour,
                        "minute": spike_dt.minute,
                        "day_of_week": spike_dt.weekday(),
                        "duration_min": round(duration_min, 1),
                        "peak_humidity": round(spike_peak, 1),
                        "baseline_humidity": round(baseline, 1),
                        "spike_magnitude": round(spike_peak - baseline, 1),
                    })
                in_spike = False
                spike_start = None
                spike_peak = 0.0

    return spikes


def analyse_routine(spikes: list[dict[str, Any]], room: str) -> dict[str, Any] | None:
    """Analyse spike events to find recurring routine patterns.

    Returns routine info with typical time, frequency, and suggested automation.
    """
    if len(spikes) < 3:
        return None

    # Group by hour
    hour_counts = Counter(s["hour"] for s in spikes)
    if not hour_counts:
        return None

    peak_hour = hour_counts.most_common(1)[0][0]
    peak_count = hour_counts[peak_hour]

    # Also check adjacent hours
    adjacent_count = peak_count
    for h in [peak_hour - 1, peak_hour + 1]:
        adjacent_count += hour_counts.get(h % 24, 0)

    # Frequency
    total_days = max(1, (spikes[-1]["start_ts"] - spikes[0]["start_ts"]) / 86400)
    events_per_day = len(spikes) / total_days

    # Day pattern
    day_counts = Counter(s["day_of_week"] for s in spikes)
    weekday_count = sum(day_counts.get(d, 0) for d in range(5))
    weekend_count = sum(day_counts.get(d, 0) for d in (5, 6))
    weekday_pct = weekday_count / max(len(spikes), 1)

    if weekday_pct > 0.8:
        day_pattern = "weekdays"
    elif weekday_pct < 0.3:
        day_pattern = "weekends"
    else:
        day_pattern = "daily"

    # Average minute within peak hour for precision
    peak_minutes = [s["minute"] for s in spikes if s["hour"] == peak_hour]
    avg_minute = int(sum(peak_minutes) / len(peak_minutes)) if peak_minutes else 0

    # Confidence: how concentrated are events around the peak time?
    concentration = adjacent_count / len(spikes)
    confidence = int(min(95, concentration * 100 * min(1, events_per_day)))

    # Activity type based on room
    if room == "bathroom":
        activity = "shower" if sum(s["duration_min"] for s in spikes) / len(spikes) < 20 else "bath"
        icon = "🚿" if activity == "shower" else "🛁"
        suggestion_action = "pre-heat water"
    elif room == "kitchen":
        activity = "cooking"
        icon = "🍳"
        suggestion_action = "turn on extractor fan"
    else:
        activity = "humidity_event"
        icon = "💧"
        suggestion_action = "ventilate"

    # Pre-heat time: 1 hour before typical event
    preheat_hour = (peak_hour - 1) % 24
    preheat_time = f"{preheat_hour:02d}:{avg_minute:02d}:00"

    # Build automation YAML
    yaml = _build_preheat_yaml(
        activity=activity,
        room=room,
        preheat_time=preheat_time,
        day_pattern=day_pattern,
    )

    return {
        "activity": activity,
        "icon": icon,
        "room": room,
        "typical_time": f"{peak_hour:02d}:{avg_minute:02d}",
        "preheat_time": preheat_time,
        "day_pattern": day_pattern,
        "events_per_day": round(events_per_day, 1),
        "total_events": len(spikes),
        "avg_duration_min": round(sum(s["duration_min"] for s in spikes) / len(spikes), 1),
        "avg_peak_humidity": round(sum(s["peak_humidity"] for s in spikes) / len(spikes), 1),
        "confidence": confidence,
        "suggestion": f"{suggestion_action} at {preheat_time} ({day_pattern})",
        "yaml": yaml,
        "recent_events": spikes[-5:],
    }


def _build_preheat_yaml(activity: str, room: str, preheat_time: str, day_pattern: str) -> str:
    """Build pre-heat automation YAML."""
    room_title = room.replace("_", " ").title()
    condition = ""
    if day_pattern == "weekdays":
        condition = """
  condition:
    - condition: time
      weekday: [mon, tue, wed, thu, fri]"""
    elif day_pattern == "weekends":
        condition = """
  condition:
    - condition: time
      weekday: [sat, sun]"""

    if activity in ("shower", "bath"):
        return f"""automation:
  alias: "Habitus — Pre-heat water for {room_title} {activity}"
  description: "Detected regular {activity} at this time. Pre-heats water 1 hour before."
  trigger:
    - platform: time
      at: "{preheat_time}"{condition}
  action:
    - service: water_heater.set_operation_mode
      data:
        operation_mode: "on"
    - service: notify.notify
      data:
        title: "🚿 Habitus"
        message: "Pre-heating water for your {activity} — {room_title}"
"""
    elif activity == "cooking":
        return f"""automation:
  alias: "Habitus — Kitchen prep for cooking time"
  description: "Detected regular cooking at this time."
  trigger:
    - platform: time
      at: "{preheat_time}"{condition}
  action:
    - service: notify.notify
      data:
        title: "🍳 Habitus"
        message: "Cooking time approaching — kitchen is ready"
"""
    return ""


def run_routine_prediction(days: int = 30) -> dict[str, Any]:
    """Run full routine prediction pipeline.

    1. Find humidity sensors and classify by room
    2. Get history for each sensor
    3. Detect spikes (shower/bath/cooking events)
    4. Analyse patterns to find recurring routines
    5. Generate pre-emptive automation suggestions
    """
    sensors = _find_humidity_sensors()
    if not sensors:
        log.info("No humidity sensors found for routine prediction")
        return {"routines": [], "events": 0}

    routines = []
    total_events = 0

    for eid, room in sensors.items():
        readings = _get_humidity_history(eid, days=days)
        if not readings:
            continue

        spikes = detect_humidity_spikes(readings)
        if not spikes:
            continue

        total_events += len(spikes)
        routine = analyse_routine(spikes, room)
        if routine:
            routine["sensor"] = eid
            routines.append(routine)

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "sensors_scanned": len(sensors),
        "total_events": total_events,
        "routines": routines,
    }

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ROUTINES_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if routines:
        log.info("Routine predictor: %d routines from %d events across %d sensors",
                 len(routines), total_events, len(sensors))
    return result


def load_routines() -> dict[str, Any]:
    """Load cached routine predictions."""
    try:
        if os.path.exists(ROUTINES_PATH):
            with open(ROUTINES_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"routines": [], "total_events": 0}
