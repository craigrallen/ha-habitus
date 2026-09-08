"""Passage Mode — detect underway state and switch monitoring profile.

Marine-specific module that detects whether the vessel is underway
based on GPS speed, engine RPM, and shore power status, then adjusts
anomaly detection thresholds accordingly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
PASSAGE_PATH = os.path.join(DATA_DIR, "passage_state.json")

# Entity ID substrings used to identify marine sensors
_GPS_KEYS = ("gps", "speed")
_ENGINE_KEYS = ("engine", "rpm")
_SHORE_KEYS = ("shore_power",)

# Threshold multipliers by entity category during passage
_PASSAGE_THRESHOLDS: dict[str, float] = {
    "bilge": 0.3,
    "battery": 0.4,
    "engine": 0.5,
}
_PASSAGE_DEFAULT_THRESHOLD = 1.5


def _match_entity(entity_id: str, keywords: tuple[str, ...]) -> bool:
    """Check whether *entity_id* contains any of the given keywords.

    Args:
        entity_id: Home Assistant entity identifier.
        keywords: Substrings to match against.

    Returns:
        True if any keyword is found in *entity_id*.
    """
    eid = entity_id.lower()
    return any(k in eid for k in keywords)


def _numeric_state(state: dict[str, Any]) -> float | None:
    """Extract a numeric value from a HA state dict.

    Args:
        state: Single HA state dictionary with ``entity_id`` and ``state``.

    Returns:
        Float value or None if not parseable.
    """
    try:
        return float(state.get("state", ""))
    except (ValueError, TypeError):
        return None


def detect_underway(ha_states: list[dict[str, Any]]) -> bool:
    """Determine if the vessel is underway based on sensor signals.

    Checks three independent signals:
    1. GPS speed > 1 knot
    2. Engine RPM > 0
    3. Shore power disconnected (== 0)

    Any two of the three signals being active means underway.

    Args:
        ha_states: List of HA state dicts (``/api/states`` format).

    Returns:
        True if underway conditions are met.
    """
    gps_speed = False
    engine_running = False
    shore_off = False

    for s in ha_states:
        eid = s.get("entity_id", "")
        val = _numeric_state(s)
        if val is None:
            continue

        if _match_entity(eid, _GPS_KEYS) and val > 1.0:
            gps_speed = True
        elif _match_entity(eid, _ENGINE_KEYS) and val > 0:
            engine_running = True
        elif _match_entity(eid, _SHORE_KEYS) and val == 0:
            shore_off = True

    signals = sum([gps_speed, engine_running, shore_off])
    return signals >= 2


def get_state() -> dict[str, Any]:
    """Return the current passage state from disk.

    Returns:
        Dict with keys: ``underway``, ``detected_at``, ``mode``, ``signals``.
        Returns defaults if no state file exists.
    """
    try:
        with open(PASSAGE_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "underway": False,
            "detected_at": None,
            "mode": "harbor",
            "signals": [],
        }


def activate_passage() -> dict[str, Any]:
    """Switch to passage monitoring mode.

    Increases bilge pump sensitivity and enables battery drain tracking.

    Returns:
        Updated passage state dict.
    """
    state = {
        "underway": True,
        "detected_at": datetime.now(UTC).isoformat(),
        "mode": "passage",
        "signals": ["manual_or_detected"],
        "thresholds": {
            "bilge": _PASSAGE_THRESHOLDS["bilge"],
            "battery": _PASSAGE_THRESHOLDS["battery"],
            "engine": _PASSAGE_THRESHOLDS["engine"],
            "default": _PASSAGE_DEFAULT_THRESHOLD,
        },
    }
    atomic_write(PASSAGE_PATH, state)
    log.info("Passage mode activated")
    return state


def deactivate_passage() -> dict[str, Any]:
    """Return to harbor monitoring mode.

    Restores default anomaly detection thresholds.

    Returns:
        Updated passage state dict.
    """
    state = {
        "underway": False,
        "detected_at": datetime.now(UTC).isoformat(),
        "mode": "harbor",
        "signals": [],
        "thresholds": {},
    }
    atomic_write(PASSAGE_PATH, state)
    log.info("Passage mode deactivated — harbor mode restored")
    return state


def get_passage_thresholds(entity_id: str) -> float:
    """Return the threshold multiplier for an entity during passage.

    During passage mode, safety-critical sensors (bilge, battery, engine)
    get lower multipliers (more sensitive), while non-critical sensors
    get a relaxed multiplier.

    Args:
        entity_id: Home Assistant entity identifier.

    Returns:
        Threshold multiplier float. Lower = more sensitive.
    """
    current = get_state()
    if not current.get("underway"):
        return 1.0

    eid = entity_id.lower()
    for category, threshold in _PASSAGE_THRESHOLDS.items():
        if category in eid:
            return threshold
    return _PASSAGE_DEFAULT_THRESHOLD
