"""Vacation Mode Intelligence — auto-detect absence and switch to security profile.

Detects extended vacancy using motion sensor history and switches to a
tighter anomaly threshold for doors/motion while relaxing thresholds
for power/temperature.
"""

import datetime
import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
VACATION_PATH = os.path.join(DATA_DIR, "vacation_state.json")

# Hours of no motion before vacation mode activates
VACANCY_TRIGGER_HOURS = 24
# Security-sensitive entity domains get tighter thresholds
SECURITY_DOMAINS = {"binary_sensor", "lock", "alarm_control_panel"}
# Domains to relax during vacation (less aggressive scoring)
RELAXED_DOMAINS = {"sensor", "climate", "light", "media_player", "switch"}


def _load_state() -> dict[str, Any]:
    """Load vacation state from disk."""
    try:
        if os.path.exists(VACATION_PATH):
            with open(VACATION_PATH) as f:
                data: dict[str, Any] = json.load(f)
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"active": False, "activated_at": None, "manual": False}


def is_active() -> bool:
    """Check if vacation mode is currently active.

    Returns:
        True if vacation mode is active (auto-detected or manually set).
    """
    active: bool = _load_state().get("active", False)
    return active


def get_state() -> dict[str, Any]:
    """Return the full vacation mode state.

    Returns:
        Dict with ``active``, ``activated_at``, ``manual``,
        ``hours_vacant``, ``security_entities``.
    """
    return _load_state()


def activate(manual: bool = False) -> dict[str, Any]:
    """Activate vacation mode.

    Args:
        manual: True if user-triggered, False if auto-detected.

    Returns:
        Updated vacation state.
    """
    now = datetime.datetime.now(datetime.UTC)
    state = {
        "active": True,
        "activated_at": now.isoformat(),
        "manual": manual,
    }
    atomic_write(VACATION_PATH, state)
    log.info("Vacation mode activated (manual=%s)", manual)
    return state


def deactivate() -> dict[str, Any]:
    """Deactivate vacation mode.

    Returns:
        Updated vacation state.
    """
    state = {"active": False, "activated_at": None, "manual": False}
    atomic_write(VACATION_PATH, state)
    log.info("Vacation mode deactivated")
    return state


def check_auto_vacation(entity_anomalies: list[dict]) -> bool:
    """Check if vacation mode should auto-activate based on motion patterns.

    Looks for the ``vacancy_pattern`` in patterns.json and activates
    when the longest no-motion window exceeds the trigger threshold.

    Args:
        entity_anomalies: Current entity anomaly list (for context).

    Returns:
        True if vacation mode was just activated.
    """
    state = _load_state()
    if state.get("manual"):
        return False  # Don't override manual control

    try:
        patterns_path = os.path.join(DATA_DIR, "patterns.json")
        if not os.path.exists(patterns_path):
            return False
        with open(patterns_path) as f:
            patterns = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    vacancy = patterns.get("vacancy_pattern", {})
    max_no_motion = vacancy.get("max_no_motion_hours", 0)

    if max_no_motion >= VACANCY_TRIGGER_HOURS and not state.get("active"):
        activate(manual=False)
        return True

    if (
        max_no_motion < VACANCY_TRIGGER_HOURS // 2
        and state.get("active")
        and not state.get("manual")
    ):
        deactivate()
        return False

    return False


def get_threshold_multiplier(entity_id: str) -> float:
    """Get the anomaly threshold multiplier for an entity during vacation mode.

    Security-sensitive entities get a 0.5x multiplier (tighter detection).
    Non-security entities get a 2.0x multiplier (more relaxed).

    Args:
        entity_id: HA entity ID.

    Returns:
        Multiplier for the entity's anomaly threshold. 1.0 when vacation
        mode is inactive.
    """
    if not is_active():
        return 1.0

    domain = entity_id.split(".")[0] if "." in entity_id else ""
    eid_lower = entity_id.lower()

    # Security-sensitive keywords
    security_keywords = ["door", "window", "lock", "motion", "occupancy", "alarm", "camera"]
    is_security = domain in SECURITY_DOMAINS or any(kw in eid_lower for kw in security_keywords)

    if is_security:
        return 0.5  # Tighter — more sensitive during absence
    if domain in RELAXED_DOMAINS:
        return 2.0  # More lenient — power/temp changes expected when away
    return 1.0
