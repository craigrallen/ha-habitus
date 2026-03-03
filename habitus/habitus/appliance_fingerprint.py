"""Appliance fingerprinting via power spike detection.

Monitors watt sensors for instantaneous power changes and learns to recognise
individual appliances by their unique power signatures:

- **Oven**: ~2000-3000W step, stays on 20-90 min
- **Hob/cooktop**: ~1200-2500W per element, cycling on/off
- **Kettle**: ~1800-2200W spike, 2-5 min duration
- **Washing machine**: ~400-2000W with cycle pattern
- **Dishwasher**: ~1200-1800W with heat/wash phases
- **Hair dryer**: ~1000-2000W, 5-15 min
- **Toaster**: ~800-1200W, 1-4 min
- **Microwave**: ~800-1500W, 30s-10 min
- **Water heater**: ~2000-3000W step, cycling

Uses Non-Intrusive Load Monitoring (NILM) principles:
1. Detect step changes (edges) in aggregate power
2. Cluster similar step sizes into appliance signatures
3. Track duration + time-of-day to identify which appliance
4. Build a library of known appliances over time
"""

import datetime
import json
import logging
import os
import sqlite3
from collections import defaultdict
from typing import Any

import numpy as np

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
FINGERPRINTS_PATH = os.path.join(DATA_DIR, "appliance_fingerprints.json")
HA_DB = "/homeassistant/home-assistant_v2.db"

# Known appliance power signatures (watts) — used as hints for classification
KNOWN_SIGNATURES: dict[str, dict[str, Any]] = {
    "oven": {"min_w": 1800, "max_w": 3500, "min_duration_min": 10, "max_duration_min": 120, "icon": "🔥", "shape": "steady"},
    "hob_element": {"min_w": 1000, "max_w": 2500, "min_duration_min": 3, "max_duration_min": 60, "icon": "🍳", "shape": "cycling"},
    "kettle": {"min_w": 1500, "max_w": 2400, "min_duration_min": 1, "max_duration_min": 6, "icon": "☕", "shape": "steady"},
    "washing_machine": {"min_w": 300, "max_w": 2200, "min_duration_min": 30, "max_duration_min": 180, "icon": "👕", "shape": "phased"},
    "dishwasher": {"min_w": 1000, "max_w": 2000, "min_duration_min": 45, "max_duration_min": 180, "icon": "🍽️", "shape": "phased"},
    "hair_dryer": {"min_w": 800, "max_w": 2200, "min_duration_min": 2, "max_duration_min": 20, "icon": "💇", "shape": "steady"},
    "toaster": {"min_w": 700, "max_w": 1400, "min_duration_min": 1, "max_duration_min": 5, "icon": "🍞", "shape": "steady"},
    "microwave": {"min_w": 700, "max_w": 1500, "min_duration_min": 0.5, "max_duration_min": 15, "icon": "📡", "shape": "steady"},
    "water_heater": {"min_w": 1800, "max_w": 3500, "min_duration_min": 10, "max_duration_min": 120, "icon": "🚿", "shape": "cycling"},
    "space_heater": {"min_w": 500, "max_w": 2500, "min_duration_min": 15, "max_duration_min": 480, "icon": "🌡️", "shape": "cycling"},
    "vacuum": {"min_w": 400, "max_w": 1800, "min_duration_min": 5, "max_duration_min": 60, "icon": "🧹", "shape": "steady"},
    "iron": {"min_w": 1000, "max_w": 2500, "min_duration_min": 5, "max_duration_min": 60, "icon": "👔", "shape": "cycling"},
    # Heat pump: high inrush spike → gradually decreasing as target temp approached
    "heat_pump": {"min_w": 800, "max_w": 4000, "min_duration_min": 15, "max_duration_min": 480, "icon": "♨️", "shape": "decaying"},
    # Electric radiator: fixed wattage cycling (thermostat on/off/on/off)
    "electric_radiator": {"min_w": 500, "max_w": 2000, "min_duration_min": 10, "max_duration_min": 480, "icon": "🔲", "shape": "cycling"},
    # Underfloor heating: low power, very long duration
    "underfloor_heating": {"min_w": 200, "max_w": 1500, "min_duration_min": 30, "max_duration_min": 720, "icon": "🏠", "shape": "steady"},
    # Immersion heater: high power, medium duration, steady
    "immersion_heater": {"min_w": 2000, "max_w": 3500, "min_duration_min": 20, "max_duration_min": 180, "icon": "🔥", "shape": "steady"},
}

# Power shape types:
# "steady" — constant power throughout (oven, kettle, microwave)
# "cycling" — on/off/on/off at regular intervals (heaters, iron, hob)
# "decaying" — high initial power that gradually decreases (heat pump, compressor)
# "phased" — distinct phases at different power levels (washing machine, dishwasher)

# Minimum watt change to count as a "step" (filters noise)
MIN_STEP_WATTS = 150
# Maximum gap between readings to consider them contiguous (seconds)
MAX_GAP_SECONDS = 300


def detect_power_steps(entity_id: str, days: int = 30) -> list[dict[str, Any]]:
    """Detect step changes in a power sensor's history.

    Reads directly from HA SQLite database for speed.
    Returns list of detected steps with magnitude, direction, timestamp.
    """
    if not os.path.exists(HA_DB):
        log.warning("HA database not found at %s — using REST API fallback", HA_DB)
        return _detect_steps_rest(entity_id, days)

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None

        if has_meta:
            query = """
                SELECT s.state, s.last_changed_ts
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                ORDER BY s.last_changed_ts
            """
        else:
            query = """
                SELECT state, last_changed_ts
                FROM states
                WHERE entity_id = ? AND last_changed_ts > ?
                ORDER BY last_changed_ts
            """

        rows = conn.execute(query, (entity_id, cutoff_ts)).fetchall()
        conn.close()
    except Exception as e:
        log.warning("DB query failed for %s: %s", entity_id, e)
        return []

    # Parse into numeric values
    readings = []
    for state_val, ts in rows:
        try:
            w = float(state_val)
            if 0 <= w <= 25000:  # sanity cap
                readings.append((ts, w))
        except (ValueError, TypeError):
            continue

    if len(readings) < 10:
        return []

    # Detect step changes (edges)
    steps = []
    for i in range(1, len(readings)):
        ts_prev, w_prev = readings[i - 1]
        ts_curr, w_curr = readings[i]

        # Skip if gap too large (sensor went offline)
        if ts_curr - ts_prev > MAX_GAP_SECONDS:
            continue

        delta = w_curr - w_prev
        if abs(delta) >= MIN_STEP_WATTS:
            steps.append({
                "timestamp": ts_curr,
                "time": datetime.datetime.fromtimestamp(ts_curr, tz=datetime.UTC).isoformat(),
                "delta_w": round(delta, 1),
                "from_w": round(w_prev, 1),
                "to_w": round(w_curr, 1),
                "direction": "up" if delta > 0 else "down",
            })

    return steps


def _detect_steps_rest(entity_id: str, days: int) -> list[dict[str, Any]]:
    """Fallback: detect steps via HA REST API."""
    # Minimal fallback — DB is preferred
    return []


def pair_steps_into_events(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair up-steps with their corresponding down-steps to form appliance events.

    An event = appliance turned on (step up) then turned off (step down of similar magnitude).
    """
    events = []
    up_steps = [s for s in steps if s["direction"] == "up"]

    for up in up_steps:
        magnitude = abs(up["delta_w"])
        # Find matching down-step within reasonable time window
        best_match = None
        best_diff = float("inf")
        for down in steps:
            if down["direction"] != "down":
                continue
            if down["timestamp"] <= up["timestamp"]:
                continue
            # Must be within 8 hours
            elapsed = down["timestamp"] - up["timestamp"]
            if elapsed > 8 * 3600:
                continue
            # Magnitude should be similar (within 30%)
            down_mag = abs(down["delta_w"])
            mag_diff = abs(magnitude - down_mag) / max(magnitude, 1)
            if mag_diff < 0.3 and mag_diff < best_diff:
                best_diff = mag_diff
                best_match = down

        if best_match:
            duration_min = (best_match["timestamp"] - up["timestamp"]) / 60
            events.append({
                "start": up["time"],
                "end": best_match["time"],
                "start_ts": up["timestamp"],
                "end_ts": best_match["timestamp"],
                "power_w": round(magnitude, 0),
                "duration_min": round(duration_min, 1),
                "hour": datetime.datetime.fromtimestamp(up["timestamp"], tz=datetime.UTC).hour,
                "day_of_week": datetime.datetime.fromtimestamp(up["timestamp"], tz=datetime.UTC).weekday(),
            })

    return events


def detect_power_shape(readings_during_event: list[float]) -> str:
    """Classify the power shape of an event from its readings.

    Returns one of: 'steady', 'cycling', 'decaying', 'phased', 'unknown'.
    """
    if len(readings_during_event) < 3:
        return "unknown"

    arr = np.array(readings_during_event)
    mean_val = np.mean(arr)
    if mean_val < 10:
        return "unknown"

    # Coefficient of variation — steady loads have low CV
    cv = np.std(arr) / mean_val if mean_val > 0 else 0

    # Check for decaying pattern: first third avg > last third avg by >30%
    third = max(1, len(arr) // 3)
    first_avg = np.mean(arr[:third])
    last_avg = np.mean(arr[-third:])

    if first_avg > 0 and (first_avg - last_avg) / first_avg > 0.30:
        return "decaying"

    # Check for cycling: count zero-crossings of (value - mean)
    centered = arr - mean_val
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    cycling_rate = crossings / len(arr)

    if cycling_rate > 0.15 and cv > 0.3:
        return "cycling"

    # Check for phased: distinct plateaus
    if cv > 0.4 and cycling_rate < 0.1:
        return "phased"

    if cv < 0.2:
        return "steady"

    return "unknown"


def classify_event(event: dict[str, Any]) -> dict[str, Any]:
    """Classify a power event against known appliance signatures.

    Returns the event dict with added classification fields.
    Uses both power/duration matching AND power shape analysis.
    """
    power = event["power_w"]
    duration = event["duration_min"]
    shape = event.get("power_shape", "unknown")

    matches = []
    for name, sig in KNOWN_SIGNATURES.items():
        if sig["min_w"] <= power <= sig["max_w"] and sig["min_duration_min"] <= duration <= sig["max_duration_min"]:
            # Score: how central is this event within the signature range?
            power_center = (sig["min_w"] + sig["max_w"]) / 2
            dur_center = (sig["min_duration_min"] + sig["max_duration_min"]) / 2
            power_score = 1 - abs(power - power_center) / (sig["max_w"] - sig["min_w"])
            dur_score = 1 - abs(duration - dur_center) / (sig["max_duration_min"] - sig["min_duration_min"])
            # Shape bonus: if power shape matches the signature's expected shape, boost score
            shape_bonus = 0.2 if shape == sig.get("shape") else 0.0
            score = (power_score + dur_score) / 2 + shape_bonus
            matches.append((name, min(score, 1.0), sig["icon"]))

    if matches:
        matches.sort(key=lambda m: -m[1])
        event["appliance"] = matches[0][0]
        event["confidence"] = round(matches[0][1] * 100)
        event["icon"] = matches[0][2]
        event["alternatives"] = [{"name": m[0], "confidence": round(m[1] * 100)} for m in matches[1:3]]
    else:
        event["appliance"] = "unknown"
        event["confidence"] = 0
        event["icon"] = "❓"
        event["alternatives"] = []

    return event


def cluster_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster similar events to build appliance fingerprint library.

    Groups events by power magnitude (±20%) and typical duration to find
    recurring appliance patterns.
    """
    if not events:
        return []

    # Sort by power
    sorted_events = sorted(events, key=lambda e: e["power_w"])

    clusters: list[dict[str, Any]] = []
    used = set()

    for i, event in enumerate(sorted_events):
        if i in used:
            continue
        cluster_events_list = [event]
        used.add(i)

        for j in range(i + 1, len(sorted_events)):
            if j in used:
                continue
            other = sorted_events[j]
            # Within 20% power range
            if abs(other["power_w"] - event["power_w"]) / max(event["power_w"], 1) < 0.2:
                cluster_events_list.append(other)
                used.add(j)
            elif other["power_w"] > event["power_w"] * 1.3:
                break  # sorted, so no more matches

        if len(cluster_events_list) >= 2:
            powers = [e["power_w"] for e in cluster_events_list]
            durations = [e["duration_min"] for e in cluster_events_list]
            hours = [e["hour"] for e in cluster_events_list]

            # Most common classification
            appliance_votes = defaultdict(int)
            for e in cluster_events_list:
                if "appliance" in e:
                    appliance_votes[e["appliance"]] += 1
            top_appliance = max(appliance_votes, key=appliance_votes.get) if appliance_votes else "unknown"
            icon = KNOWN_SIGNATURES.get(top_appliance, {}).get("icon", "❓")

            # Most active hours
            hour_counts = defaultdict(int)
            for h in hours:
                hour_counts[h] += 1
            peak_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:3]

            clusters.append({
                "appliance": top_appliance,
                "icon": icon,
                "avg_power_w": round(float(np.mean(powers))),
                "min_power_w": round(float(np.min(powers))),
                "max_power_w": round(float(np.max(powers))),
                "avg_duration_min": round(float(np.mean(durations)), 1),
                "event_count": len(cluster_events_list),
                "peak_hours": peak_hours,
                "daily_avg": round(len(cluster_events_list) / 30, 1),
                "est_monthly_kwh": round(
                    float(np.mean(powers)) * float(np.mean(durations)) / 60 * len(cluster_events_list) / 1000, 1
                ),
            })

    clusters.sort(key=lambda c: -c["event_count"])
    return clusters


def run_fingerprinting(power_entities: list[str] | None = None, days: int = 30) -> dict[str, Any]:
    """Run full appliance fingerprinting pipeline.

    1. Find all watt sensors (or use provided list)
    2. Detect step changes on aggregate / per-circuit sensors
    3. Pair into on/off events
    4. Classify against known signatures
    5. Cluster into recurring patterns

    Returns dict with fingerprints, events, and summary.
    """
    if power_entities is None:
        power_entities = _find_power_entities()

    all_events = []
    per_entity_events: dict[str, list] = {}

    for eid in power_entities:
        steps = detect_power_steps(eid, days=days)
        if not steps:
            continue
        # Also get raw readings for power shape analysis
        raw_readings = _get_raw_readings(eid, days=days)
        events = pair_steps_into_events(steps)
        # Annotate events with power shape from raw readings
        for evt in events:
            readings_in_event = [
                w for ts, w in raw_readings
                if evt["start_ts"] <= ts <= evt["end_ts"]
            ]
            if len(readings_in_event) >= 3:
                evt["power_shape"] = detect_power_shape(readings_in_event)
            else:
                evt["power_shape"] = "unknown"
        classified = [classify_event(e) for e in events]
        if classified:
            per_entity_events[eid] = classified
            all_events.extend(classified)

    clusters = cluster_events(all_events)

    # Summary
    total_events = len(all_events)
    known_events = sum(1 for e in all_events if e.get("appliance") != "unknown")

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "entities_scanned": len(power_entities),
        "total_events": total_events,
        "identified_events": known_events,
        "identification_rate": round(known_events / max(total_events, 1) * 100),
        "appliances": clusters,
        "recent_events": sorted(all_events, key=lambda e: e.get("start_ts", 0), reverse=True)[:20],
        "per_entity": {eid: len(evts) for eid, evts in per_entity_events.items()},
    }

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FINGERPRINTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(
        "Appliance fingerprinting: %d events from %d sensors, %d identified (%d%%)",
        total_events, len(power_entities), known_events,
        result["identification_rate"],
    )

    return result


def _get_raw_readings(entity_id: str, days: int = 30) -> list[tuple[float, float]]:
    """Get raw (timestamp, watts) readings for power shape analysis."""
    if not os.path.exists(HA_DB):
        return []
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        has_meta = cursor.fetchone() is not None
        if has_meta:
            rows = conn.execute("""
                SELECT s.state, s.last_changed_ts FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                ORDER BY s.last_changed_ts
            """, (entity_id, cutoff_ts)).fetchall()
        else:
            rows = conn.execute("""
                SELECT state, last_changed_ts FROM states
                WHERE entity_id = ? AND last_changed_ts > ?
                ORDER BY last_changed_ts
            """, (entity_id, cutoff_ts)).fetchall()
        conn.close()
        result = []
        for state_val, ts in rows:
            try:
                w = float(state_val)
                if 0 <= w <= 25000:
                    result.append((ts, w))
            except (ValueError, TypeError):
                continue
        return result
    except Exception:
        return []


def _find_power_entities() -> list[str]:
    """Find watt power sensors from HA database."""
    if not os.path.exists(HA_DB):
        return []

    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states_meta'"
        )
        if cursor.fetchone():
            rows = conn.execute("""
                SELECT DISTINCT sm.entity_id
                FROM states_meta sm
                WHERE sm.entity_id LIKE 'sensor.%'
                AND (
                    sm.entity_id LIKE '%_power'
                    OR sm.entity_id LIKE '%_w'
                    OR sm.entity_id LIKE '%_wattage'
                    OR sm.entity_id LIKE '%_consumption_w'
                )
                AND sm.entity_id NOT LIKE '%_kwh%'
                AND sm.entity_id NOT LIKE '%_kvar%'
                AND sm.entity_id NOT LIKE '%_voltage%'
                AND sm.entity_id NOT LIKE '%_current%'
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT DISTINCT entity_id
                FROM states
                WHERE entity_id LIKE 'sensor.%'
                AND (
                    entity_id LIKE '%_power'
                    OR entity_id LIKE '%_w'
                    OR entity_id LIKE '%_wattage'
                    OR entity_id LIKE '%_consumption_w'
                )
                AND entity_id NOT LIKE '%_kwh%'
                AND entity_id NOT LIKE '%_kvar%'
                AND entity_id NOT LIKE '%_voltage%'
                AND entity_id NOT LIKE '%_current%'
                LIMIT 50
            """).fetchall()
        conn.close()
        entities = [r[0] for r in rows]
        log.info("Found %d power sensors for fingerprinting", len(entities))
        return entities
    except Exception as e:
        log.warning("Failed to find power entities: %s", e)
        return []


def load_fingerprints() -> dict[str, Any]:
    """Load cached fingerprint results."""
    try:
        if os.path.exists(FINGERPRINTS_PATH):
            with open(FINGERPRINTS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"appliances": [], "recent_events": [], "total_events": 0}
