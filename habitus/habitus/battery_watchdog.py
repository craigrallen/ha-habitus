"""Battery Watchdog — monitor battery levels and drain rates.

Fetches all HA entities where entity_id matches *battery* or device_class=battery.
Tracks current level, trend (drain per day), estimated_days_remaining.
Groups by area/room.
Alert levels: ok (>30%), low (10-30%), critical (<10%).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import requests

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

BATTERY_PATH = os.path.join(DATA_DIR, "battery_status.json")

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))

# Alert thresholds (%)
CRITICAL_THRESHOLD = 10
LOW_THRESHOLD = 30

# History days for drain rate
HISTORY_DAYS = 7


def _ha_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def _fetch_battery_entities(
    states: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all battery-related entities from HA states."""
    if states is None:
        try:
            r = requests.get(f"{HA_URL}/api/states", headers=_ha_headers(), timeout=10)
            if r.status_code == 200:
                states = r.json()
            else:
                return []
        except Exception as e:
            log.warning("Failed to fetch HA states: %s", e)
            return []

    battery_entities = []
    for state in states:
        eid = state.get("entity_id", "")
        attrs = state.get("attributes", {})
        device_class = attrs.get("device_class", "")
        unit = attrs.get("unit_of_measurement", "")

        is_battery = (
            "battery" in eid.lower()
            or device_class == "battery"
            or unit == "%"  # many battery sensors report in %
            and "battery" in eid.lower()
        )

        if is_battery and eid.startswith("sensor."):
            battery_entities.append(state)

    return battery_entities


def _fetch_battery_history(entity_id: str) -> list[dict[str, Any]]:
    """Fetch recent history for a battery entity."""
    try:
        end = datetime.datetime.now(datetime.UTC)
        start = end - datetime.timedelta(days=HISTORY_DAYS)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        url = f"{HA_URL}/api/history/period/{start_str}"
        params = {
            "filter_entity_id": entity_id,
            "end_time": end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        }
        r = requests.get(url, headers=_ha_headers(), params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data[0], list):
                return data[0]
    except Exception as e:
        log.debug("Failed to fetch history for %s: %s", entity_id, e)
    return []


def _compute_drain_rate(history: list[dict[str, Any]]) -> float | None:
    """Compute battery drain rate in % per day from history.

    Returns None if insufficient data.
    """
    readings: list[tuple[datetime.datetime, float]] = []

    for entry in history:
        state = entry.get("state", "")
        last_changed = entry.get("last_changed", "")
        try:
            level = float(state)
            dt = datetime.datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
            readings.append((dt, level))
        except (ValueError, TypeError):
            pass

    if len(readings) < 2:
        return None

    readings.sort(key=lambda r: r[0])

    # Get first and last valid readings
    first_dt, first_level = readings[0]
    last_dt, last_level = readings[-1]

    time_diff_days = (last_dt - first_dt).total_seconds() / 86400
    if time_diff_days < 0.1:
        return None

    level_drop = first_level - last_level  # positive = draining
    drain_per_day = level_drop / time_diff_days

    return round(drain_per_day, 2)


def classify_battery(entity_id: str, level: float) -> str:
    """Classify battery alert level."""
    if level < CRITICAL_THRESHOLD:
        return "critical"
    elif level < LOW_THRESHOLD:
        return "low"
    return "ok"


def _infer_area(entity_id: str, attrs: dict[str, Any]) -> str:
    """Infer area/room from entity_id or attributes."""
    area = attrs.get("area") or attrs.get("room") or ""
    if area:
        return area

    # Try to infer from entity_id
    eid_parts = entity_id.replace("sensor.", "").split("_")
    room_keywords = {
        "living": "Living Room",
        "bedroom": "Bedroom",
        "kitchen": "Kitchen",
        "bathroom": "Bathroom",
        "hallway": "Hallway",
        "office": "Office",
        "garage": "Garage",
        "garden": "Garden",
        "front": "Front",
        "back": "Back",
        "outdoor": "Outdoor",
        "door": "Door",
        "window": "Window",
    }
    for part in eid_parts:
        if part.lower() in room_keywords:
            return room_keywords[part.lower()]

    return "Unknown"


def run_battery_check(
    states: list[dict[str, Any]] | None = None,
    fetch_history: bool = False,
) -> dict[str, Any]:
    """Run battery watchdog check.

    Args:
        states: Optional list of HA states. Fetched from HA if None.
        fetch_history: Whether to fetch history for drain rate calculation.

    Returns:
        Battery status report.
    """
    battery_entities = _fetch_battery_entities(states)

    batteries = []
    by_area: dict[str, list[str]] = {}

    for entity_state in battery_entities:
        eid = entity_state.get("entity_id", "")
        state_val = entity_state.get("state", "")
        attrs = entity_state.get("attributes", {})
        friendly_name = attrs.get("friendly_name", eid.replace("sensor.", "").replace("_", " ").title())

        try:
            level = float(state_val)
        except (ValueError, TypeError):
            continue

        if not (0 <= level <= 100):
            continue

        alert = classify_battery(eid, level)
        area = _infer_area(eid, attrs)

        # Drain rate from history (optional)
        drain_per_day: float | None = None
        estimated_days_remaining: float | None = None

        if fetch_history:
            history = _fetch_battery_history(eid)
            drain_per_day = _compute_drain_rate(history)

        if drain_per_day is not None and drain_per_day > 0:
            estimated_days_remaining = round(level / drain_per_day, 1)
        elif drain_per_day is not None and drain_per_day <= 0:
            # Battery charging or stable
            estimated_days_remaining = None

        battery_info = {
            "entity_id": eid,
            "friendly_name": friendly_name,
            "level": level,
            "alert": alert,
            "area": area,
            "drain_per_day": drain_per_day,
            "estimated_days_remaining": estimated_days_remaining,
            "last_updated": entity_state.get("last_updated", ""),
        }

        batteries.append(battery_info)

        # Group by area
        if area not in by_area:
            by_area[area] = []
        by_area[area].append(eid)

    # Sort by criticality: critical first, then low, then ok
    alert_order = {"critical": 0, "low": 1, "ok": 2}
    batteries.sort(key=lambda b: (alert_order.get(b["alert"], 3), b["level"]))

    # Summary counts
    summary = {
        "critical": sum(1 for b in batteries if b["alert"] == "critical"),
        "low": sum(1 for b in batteries if b["alert"] == "low"),
        "ok": sum(1 for b in batteries if b["alert"] == "ok"),
    }

    report = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total": len(batteries),
        "summary": summary,
        "batteries": batteries,
        "by_area": by_area,
    }
    return report


def save_battery_status(report: dict[str, Any]) -> None:
    """Save battery status report."""
    os.makedirs(os.environ.get("DATA_DIR", "/data"), exist_ok=True)
    with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "battery_status.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(
        "Battery watchdog: %d batteries — critical=%d, low=%d, ok=%d",
        report["total"],
        report["summary"].get("critical", 0),
        report["summary"].get("low", 0),
        report["summary"].get("ok", 0),
    )


def load_battery_status() -> dict[str, Any]:
    """Load cached battery status."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "battery_status.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "battery_status.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {"total": 0, "summary": {"critical": 0, "low": 0, "ok": 0}, "batteries": []}
