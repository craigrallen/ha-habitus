"""Guest Mode Detection — analyse recent activity vs. baseline to detect guests.

Guest signals:
- Unusual activity time spread (active at unusual hours)
- More simultaneous device activations than baseline
- Unfamiliar motion patterns at night
- Lights in rooms not usually used

Computes guest_probability (0-1) with contributing factors.
If > 0.6: suggests Guest Mode automation pack.
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

GUEST_MODE_PATH = os.path.join(DATA_DIR, "guest_mode.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")

GUEST_PROBABILITY_THRESHOLD = 0.6

# Domains we count as "devices"
ACTIVE_DOMAINS = ("light", "switch", "media_player", "binary_sensor")

# "Guest" rooms — rooms with lights/sensors not usually used
GUEST_ROOM_KEYWORDS = ("guest", "spare", "extra", "second", "visitor", "hallway", "entrance")
PRIVATE_KEYWORDS = ("bedroom", "master", "office", "study", "bathroom", "toilet")

# Night hours (22-06)
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 7))


def _load_baseline() -> dict[str, Any]:
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "baseline.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "baseline.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _hour_spread(activity_hours: list[int]) -> float:
    """Compute normalized spread of activity across 24 hours (0=concentrated, 1=spread)."""
    if not activity_hours:
        return 0.0
    hour_set = set(activity_hours)
    return len(hour_set) / 24.0


def _night_activity_ratio(activity_hours: list[int]) -> float:
    """Fraction of activity that occurs during night hours."""
    if not activity_hours:
        return 0.0
    night = sum(1 for h in activity_hours if h in NIGHT_HOURS)
    return night / len(activity_hours)


def _concurrent_activations(events: list[dict[str, Any]], window_seconds: int = 30) -> float:
    """Average number of entities activated concurrently within a time window."""
    if not events:
        return 0.0

    # Sort by time
    timed = []
    for evt in events:
        ts = evt.get("timestamp", "")
        try:
            if isinstance(ts, str):
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, (int, float)):
                dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
            else:
                continue
            timed.append(dt)
        except (ValueError, TypeError):
            pass

    if len(timed) < 2:
        return 1.0

    timed.sort()
    concurrent_counts = []
    for i, t in enumerate(timed):
        window_end = t + datetime.timedelta(seconds=window_seconds)
        count = sum(1 for t2 in timed if t <= t2 <= window_end)
        concurrent_counts.append(count)

    return sum(concurrent_counts) / len(concurrent_counts)


def _unusual_rooms_active(events: list[dict[str, Any]], baseline: dict[str, Any]) -> list[str]:
    """Find rooms with activity that are unusually active vs. baseline."""
    # Extract unique entities active in recent events
    recent_entities: set[str] = set()
    for evt in events:
        eid = evt.get("entity_id", "")
        if eid:
            recent_entities.add(eid)

    # Baseline entities (entities that appear in baseline patterns)
    baseline_entities: set[str] = set()
    for entry in baseline.get("entities", []):
        eid = entry if isinstance(entry, str) else entry.get("entity_id", "")
        if eid:
            baseline_entities.add(eid)

    # Find entities active now but rarely in baseline
    unusual = []
    for eid in recent_entities:
        eid_lower = eid.lower()
        if any(kw in eid_lower for kw in GUEST_ROOM_KEYWORDS):
            unusual.append(eid)
        elif eid not in baseline_entities and any(kw in eid_lower for kw in ("light.", "switch.")):
            unusual.append(eid)

    return unusual


def compute_guest_probability(
    recent_events: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute guest probability from recent events vs baseline.

    Args:
        recent_events: List of {entity_id, state, timestamp} from last 7 days.
        baseline: Baseline activity patterns. If None, loads from BASELINE_PATH.

    Returns:
        Dict with guest_probability, factors, and suggested actions.
    """
    if baseline is None:
        baseline = _load_baseline()

    factors: dict[str, float] = {}

    # Factor 1: Activity time spread
    hours = []
    for evt in recent_events:
        ts = evt.get("timestamp", "")
        try:
            if isinstance(ts, str):
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, (int, float)):
                dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
            else:
                continue
            hours.append(dt.hour)
        except (ValueError, TypeError):
            pass

    spread = _hour_spread(hours)
    baseline_spread = baseline.get("typical_hour_spread", 0.5)
    if spread > baseline_spread * 1.3:
        factors["unusual_time_spread"] = min(0.3, (spread - baseline_spread) * 0.5)

    # Factor 2: Night activity
    night_ratio = _night_activity_ratio(hours)
    baseline_night = baseline.get("typical_night_ratio", 0.05)
    if night_ratio > baseline_night * 2:
        factors["night_activity"] = min(0.3, (night_ratio - baseline_night) * 2)

    # Factor 3: Concurrent activations
    concurrent = _concurrent_activations(recent_events)
    baseline_concurrent = baseline.get("typical_concurrent", 1.5)
    if concurrent > baseline_concurrent * 1.5:
        factors["high_concurrent_activations"] = min(0.2, (concurrent - baseline_concurrent) * 0.1)

    # Factor 4: Unusual rooms active
    unusual_rooms = _unusual_rooms_active(recent_events, baseline)
    if unusual_rooms:
        factors["unusual_rooms_active"] = min(0.3, len(unusual_rooms) * 0.1)

    # Factor 5: More total events than usual
    daily_avg = len(recent_events) / 7
    baseline_daily = baseline.get("typical_daily_events", 50)
    if baseline_daily > 0 and daily_avg > baseline_daily * 1.4:
        factors["elevated_activity"] = min(0.2, (daily_avg / baseline_daily - 1) * 0.2)

    guest_probability = min(1.0, sum(factors.values()))

    result: dict[str, Any] = {
        "guest_probability": round(guest_probability, 3),
        "factors": factors,
        "unusual_rooms": unusual_rooms,
    }

    if guest_probability > GUEST_PROBABILITY_THRESHOLD:
        result["suggestions"] = _generate_guest_suggestions(unusual_rooms)

    return result


def _generate_guest_suggestions(unusual_rooms: list[str]) -> list[dict[str, Any]]:
    """Generate Guest Mode automation suggestions."""
    suggestions = []

    # 1. Disable private automations
    suggestions.append({
        "type": "disable_private",
        "title": "Disable private automations",
        "description": "Disable bedroom/presence-based automations while guests are present",
        "generated_yaml": """automation:
  alias: "Habitus — Disable private automations (Guest Mode)"
  trigger:
    - platform: state
      entity_id: input_boolean.guest_mode
      to: "on"
  action:
    - service: automation.turn_off
      target:
        entity_id:
          - automation.bedroom_lights_presence
          - automation.master_bedroom_sleep_mode
    - service: notify.notify
      data:
        title: "Guest Mode Active"
        message: "Private automations disabled for guest visit"
""",
    })

    # 2. Enable hallway night lights
    suggestions.append({
        "type": "guest_night_lights",
        "title": "Enable guest-friendly night lights",
        "description": "Turn on hallway and bathroom night lights for guest navigation",
        "generated_yaml": """automation:
  alias: "Habitus — Guest night lights"
  trigger:
    - platform: state
      entity_id: input_boolean.guest_mode
      to: "on"
  action:
    - service: light.turn_on
      target:
        entity_id: light.hallway
      data:
        brightness_pct: 20
        color_temp: 400
""",
    })

    # 3. Bathroom fan on motion
    suggestions.append({
        "type": "guest_bathroom_fan",
        "title": "Bathroom fan on motion (guest comfort)",
        "description": "Automatically activate bathroom fan when motion detected",
        "generated_yaml": """automation:
  alias: "Habitus — Guest bathroom fan"
  trigger:
    - platform: state
      entity_id: binary_sensor.bathroom_motion
      to: "on"
  condition:
    - condition: state
      entity_id: input_boolean.guest_mode
      state: "on"
  action:
    - service: switch.turn_on
      target:
        entity_id: switch.bathroom_fan
    - delay: "00:15:00"
    - service: switch.turn_off
      target:
        entity_id: switch.bathroom_fan
""",
    })

    return suggestions


def run(
    recent_events: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run guest mode detection and save results.

    Returns:
        Full guest mode analysis result.
    """
    if recent_events is None:
        recent_events = []

    result = compute_guest_probability(recent_events, baseline)
    result["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()

    from .utils import atomic_write as _atomic_write  # noqa: PLC0415
    _atomic_write(os.path.join(os.environ.get("DATA_DIR", "/data"), "guest_mode.json"), result)

    log.info("Guest mode: probability=%.2f, factors=%s", result["guest_probability"], result.get("factors", {}))
    return result


def load_guest_mode() -> dict[str, Any]:
    """Load cached guest mode analysis."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "guest_mode.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "guest_mode.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {"guest_probability": 0.0, "factors": {}}
