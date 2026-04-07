"""Onboarding — first-run detection and wizard state management.

Handles multi-step onboarding flow:
1. Welcome
2. Confirm HA connection
3. Set energy tariff
4. Choose notification preferences
5. Initial training
6. Mark complete
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

ONBOARDING_PATH = os.path.join(DATA_DIR, "onboarding_complete.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

TOTAL_STEPS = 6
STEP_NAMES = [
    "welcome",
    "ha_connection",
    "energy_tariff",
    "notification_prefs",
    "initial_training",
    "complete",
]


def is_complete() -> bool:
    """Check if onboarding has been completed."""
    return os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "onboarding_complete.json"))


def get_status() -> dict[str, Any]:
    """Get current onboarding status.

    Returns:
        Dict with complete, current_step, total_steps, step_names.
    """
    if is_complete():
        try:
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "onboarding_complete.json")) as f:
                data = json.load(f)
        except Exception:
            data = {}
        return {
            "complete": True,
            "completed_at": data.get("completed_at", ""),
            "skipped": data.get("skipped", False),
            "current_step": TOTAL_STEPS,
            "total_steps": TOTAL_STEPS,
            "step_names": STEP_NAMES,
        }

    return {
        "complete": False,
        "completed_at": None,
        "skipped": False,
        "current_step": 0,
        "total_steps": TOTAL_STEPS,
        "step_names": STEP_NAMES,
    }


def complete_onboarding(
    tariff: float | None = None,
    tariff_peak: float | None = None,
    tariff_offpeak: float | None = None,
    notification_prefs: dict[str, Any] | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    """Mark onboarding as complete and save settings.

    Args:
        tariff: Standard energy tariff EUR/kWh.
        tariff_peak: Peak tariff EUR/kWh.
        tariff_offpeak: Off-peak tariff EUR/kWh.
        notification_prefs: Dict of notification preferences.
        skipped: Whether user skipped onboarding.

    Returns:
        Onboarding completion record.
    """
    os.makedirs(os.environ.get("DATA_DIR", "/data"), exist_ok=True)

    # Save energy tariff and preferences to settings
    if tariff is not None or tariff_peak is not None or tariff_offpeak is not None or notification_prefs is not None:
        settings = _load_settings()
        if tariff is not None:
            settings["energy_tariff"] = tariff
        if tariff_peak is not None:
            settings["energy_tariff_peak"] = tariff_peak
        if tariff_offpeak is not None:
            settings["energy_tariff_offpeak"] = tariff_offpeak
        if notification_prefs is not None:
            settings["notification_prefs"] = notification_prefs
        _save_settings(settings)

    # Write completion marker
    record: dict[str, Any] = {
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "skipped": skipped,
        "steps_completed": TOTAL_STEPS if not skipped else 0,
    }

    with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "onboarding_complete.json"), "w") as f:
        json.dump(record, f, indent=2)

    log.info("Onboarding complete (skipped=%s)", skipped)
    return record


def reset_onboarding() -> None:
    """Reset onboarding (remove completion marker)."""
    _path = os.path.join(os.environ.get("DATA_DIR", "/data"), "onboarding_complete.json")
    if os.path.exists(_path):
        os.remove(_path)
    log.info("Onboarding reset")


def _load_settings() -> dict[str, Any]:
    """Load current settings."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "settings.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "settings.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_settings(settings: dict[str, Any]) -> None:
    """Save settings."""
    os.makedirs(os.environ.get("DATA_DIR", "/data"), exist_ok=True)
    with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
