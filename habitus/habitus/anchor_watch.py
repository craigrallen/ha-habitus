"""Anchor Watch — monitor for anchor drag and water ingress.

Tracks the vessel's GPS position relative to a set anchor point and
monitors onboard systems (bilge pump, battery, water level) for
safety while at anchor.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import UTC, datetime
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ANCHOR_PATH = os.path.join(DATA_DIR, "anchor_watch.json")

_EARTH_RADIUS_M = 6_371_000.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points in meters.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in meters.
    """
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the initial bearing from point 1 to point 2.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Bearing in degrees (0-360).
    """
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - (math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _load_watch() -> dict[str, Any]:
    """Load anchor watch state from disk.

    Returns:
        Watch state dict or empty dict if not found.
    """
    try:
        with open(ANCHOR_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def start_watch(lat: float, lon: float, radius_m: float = 30) -> dict[str, Any]:
    """Set anchor position and alert radius.

    Args:
        lat: Anchor latitude in decimal degrees.
        lon: Anchor longitude in decimal degrees.
        radius_m: Alert radius in meters (default 30).

    Returns:
        Watch state dict with anchor position and status.
    """
    state: dict[str, Any] = {
        "active": True,
        "anchor_lat": lat,
        "anchor_lon": lon,
        "radius_m": radius_m,
        "started_at": datetime.now(UTC).isoformat(),
        "dragging": False,
        "alerts": [],
    }
    atomic_write(ANCHOR_PATH, state)
    log.info("Anchor watch started at %.6f, %.6f (radius %dm)", lat, lon, radius_m)
    return state


def check_position(current_lat: float, current_lon: float) -> dict[str, Any]:
    """Compute distance from anchor and check for drag.

    Args:
        current_lat: Current GPS latitude.
        current_lon: Current GPS longitude.

    Returns:
        Dict with ``distance_m``, ``bearing_deg``, ``dragging``, ``alert``.

    Raises:
        RuntimeError: If no anchor watch is active.
    """
    watch = _load_watch()
    if not watch.get("active"):
        raise RuntimeError("No anchor watch is active")

    anchor_lat = watch["anchor_lat"]
    anchor_lon = watch["anchor_lon"]
    radius = watch["radius_m"]

    dist = _haversine(anchor_lat, anchor_lon, current_lat, current_lon)
    bear = _bearing(anchor_lat, anchor_lon, current_lat, current_lon)
    dragging = dist > radius

    result: dict[str, Any] = {
        "distance_m": round(dist, 1),
        "bearing_deg": round(bear, 1),
        "dragging": dragging,
        "alert": dragging,
    }

    if dragging:
        log.warning(
            "ANCHOR DRAG detected! Distance %.1fm exceeds radius %.1fm",
            dist,
            radius,
        )
        watch["dragging"] = True
        watch.setdefault("alerts", []).append(
            {
                "type": "drag",
                "distance_m": round(dist, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        atomic_write(ANCHOR_PATH, watch)

    return result


def _numeric_state(state: dict[str, Any]) -> float | None:
    """Extract numeric value from a HA state dict.

    Args:
        state: Single HA state dictionary.

    Returns:
        Float value or None if not parseable.
    """
    try:
        return float(state.get("state", ""))
    except (ValueError, TypeError):
        return None


def check_systems(ha_states: list[dict[str, Any]]) -> dict[str, Any]:
    """Check bilge pump, battery drain, and water level.

    Args:
        ha_states: List of HA state dicts (``/api/states`` format).

    Returns:
        Dict with ``bilge_active``, ``battery_drain_w``, ``water_level_ok``,
        and ``alerts`` list.
    """
    bilge_active = False
    battery_drain_w = 0.0
    water_level_ok = True
    alerts: list[str] = []

    for s in ha_states:
        eid = s.get("entity_id", "").lower()
        val = _numeric_state(s)

        if "bilge" in eid:
            if s.get("state", "").lower() in ("on", "1", "true") or (val is not None and val > 0):
                bilge_active = True
                alerts.append(f"Bilge pump active: {eid}")

        elif "battery" in eid and "drain" in eid and val is not None:
            battery_drain_w = val
            if val > 50:
                alerts.append(f"High battery drain: {val}W on {eid}")

        elif "water_level" in eid and s.get("state", "").lower() in ("high", "alert", "on"):
            water_level_ok = False
            alerts.append(f"Water level alert: {eid}")

    return {
        "bilge_active": bilge_active,
        "battery_drain_w": battery_drain_w,
        "water_level_ok": water_level_ok,
        "alerts": alerts,
    }


def get_watch_status() -> dict[str, Any]:
    """Return combined anchor position and systems status.

    Returns:
        Full watch state from disk, or inactive status if no watch set.
    """
    watch = _load_watch()
    if not watch.get("active"):
        return {"active": False, "message": "No anchor watch is active"}
    return watch


def stop_watch() -> dict[str, Any]:
    """Stop the anchor watch.

    Returns:
        Final watch state with ``active`` set to False.
    """
    watch = _load_watch()
    watch["active"] = False
    watch["stopped_at"] = datetime.now(UTC).isoformat()
    atomic_write(ANCHOR_PATH, watch)
    log.info("Anchor watch stopped")
    return watch
