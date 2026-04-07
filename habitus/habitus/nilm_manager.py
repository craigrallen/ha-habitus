"""NILM Appliance Management — manual override, refinement, and appliance library."""

import json
import logging
import os
from datetime import datetime

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
LIBRARY_PATH = os.path.join(DATA_DIR, "nilm_appliance_library.json")
OVERRIDES_PATH = os.path.join(DATA_DIR, "nilm_overrides.json")


def get_appliance_library() -> dict:
    """Load user's appliance library (known device signatures)."""
    # Try user library first
    if os.path.exists(LIBRARY_PATH):
        try:
            with open(LIBRARY_PATH) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load appliance library: {e}")

    # Fall back to boat_appliances.json from repo
    boat_lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "boat_appliances.json")
    ref_lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "reference_appliances.json")

    appliances = []

    # Load boat-specific appliances (highest priority)
    if os.path.exists(boat_lib_path):
        try:
            with open(boat_lib_path) as f:
                data = json.load(f)
                boat_apps = data.get("boat_appliances", [])
                for app in boat_apps:
                    app["source"] = "boat_specific"
                appliances.extend(boat_apps)
        except Exception as e:
            log.warning(f"Failed to load boat appliances library: {e}")

    # Load reference library (PLAID + UK-DALE)
    if os.path.exists(ref_lib_path):
        try:
            with open(ref_lib_path) as f:
                data = json.load(f)
                appliances.extend(data.get("appliances", []))
        except Exception as e:
            log.warning(f"Failed to load reference appliances library: {e}")

    return {
        "appliances": appliances,
        "metadata": {
            "total_appliances": len(appliances),
            "boat_specific": sum(1 for a in appliances if a.get("source") == "boat_specific"),
            "reference": sum(1 for a in appliances if a.get("source") in ["PLAID", "UK-DALE"]),
        },
    }


def save_appliance_library(library: dict) -> None:
    """Save appliance library."""
    try:
        from .utils import atomic_write

        atomic_write(LIBRARY_PATH, library)
    except Exception as e:
        log.warning(f"Failed to save appliance library: {e}")


def get_overrides() -> dict:
    """Load NILM manual overrides (re-labels, merges, deletions)."""
    if not os.path.exists(OVERRIDES_PATH):
        return {"overrides": []}

    try:
        with open(OVERRIDES_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load NILM overrides: {e}")
        return {"overrides": []}


def save_overrides(overrides: dict) -> None:
    """Save NILM overrides."""
    try:
        from .utils import atomic_write

        atomic_write(OVERRIDES_PATH, overrides)
    except Exception as e:
        log.warning(f"Failed to save NILM overrides: {e}")


def add_override(
    slot_name: str, action: str, **kwargs  # "relabel", "merge", "delete", "confirm", "split"
) -> dict:
    """Add a manual override for a NILM appliance slot.

    Args:
        slot_name: Original appliance slot name (e.g., "Dishwasher").
        action: Override action type.
        **kwargs: Action-specific parameters (new_label, merge_with, etc.).

    Returns:
        Updated overrides dict.
    """
    overrides = get_overrides()

    override = {
        "slot_name": slot_name,
        "action": action,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        **kwargs,
    }

    # Remove existing override for this slot if any
    overrides["overrides"] = [
        o for o in overrides.get("overrides", []) if o.get("slot_name") != slot_name
    ]

    overrides["overrides"].append(override)
    save_overrides(overrides)

    return overrides


def apply_overrides(nilm_data: dict) -> dict:
    """Apply manual overrides to NILM disaggregation results.

    Args:
        nilm_data: Raw NILM disaggregation data.

    Returns:
        Modified NILM data with overrides applied.
    """
    overrides = get_overrides()
    appliances = nilm_data.get("discovered_appliances", [])

    for override in overrides.get("overrides", []):
        slot_name = override.get("slot_name")
        action = override.get("action")

        if action == "relabel":
            # Rename appliance
            for app in appliances:
                if app.get("name") == slot_name:
                    app["name"] = override.get("new_label", slot_name)
                    app["user_confirmed"] = True

        elif action == "delete":
            # Remove false positive
            appliances = [a for a in appliances if a.get("name") != slot_name]

        elif action == "merge":
            # Combine two appliance slots
            merge_with = override.get("merge_with")
            target = next((a for a in appliances if a.get("name") == merge_with), None)
            source = next((a for a in appliances if a.get("name") == slot_name), None)

            if target and source:
                # Sum events and energy
                target["events"] = target.get("events", 0) + source.get("events", 0)
                target["avg_kwh"] = (
                    target.get("avg_kwh", 0) * target.get("events", 1)
                    + source.get("avg_kwh", 0) * source.get("events", 1)
                ) / (target.get("events", 1) + source.get("events", 1))
                appliances = [a for a in appliances if a.get("name") != slot_name]

        elif action == "confirm":
            # Lock in identification
            for app in appliances:
                if app.get("name") == slot_name:
                    app["user_confirmed"] = True

    nilm_data["discovered_appliances"] = appliances
    return nilm_data


def suggest_appliance_matches(wattage: float, runtime_min: float = None) -> list:
    """Suggest appliance matches from library based on wattage and runtime.

    Args:
        wattage: Average wattage of discovered appliance.
        runtime_min: Average runtime in minutes (optional).

    Returns:
        List of matching appliances from library, sorted by likelihood.
    """
    library = get_appliance_library()
    appliances = library.get("appliances", [])

    matches = []
    for app in appliances:
        watt_range = app.get("watt_range", [0, 9999])
        if watt_range[0] <= wattage <= watt_range[1]:
            # Calculate match score (0-100)
            typical_watt = app.get("typical_watt", wattage)
            watt_diff = abs(wattage - typical_watt)
            watt_score = max(0, 100 - watt_diff / typical_watt * 100) if typical_watt > 0 else 0

            # Boost score for boat-specific devices
            source_bonus = 0
            if app.get("source") == "boat_specific":
                source_bonus = 20

            matches.append(
                {
                    "name": app.get("name"),
                    "category": app.get("category"),
                    "typical_watt": typical_watt,
                    "watt_range": watt_range,
                    "match_score": round(min(100, watt_score + source_bonus), 1),
                    "phase_hint": app.get("phase_hint"),
                    "notes": app.get("notes", ""),
                    "source": app.get("source", ""),
                }
            )

    # Sort by match score (boat-specific devices boosted)
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches[:10]  # Top 10 matches


def get_appliance_details(slot_name: str, nilm_data: dict) -> dict:
    """Get detailed pattern analysis for a specific appliance slot.

    Args:
        slot_name: Appliance slot name.
        nilm_data: NILM disaggregation data.

    Returns:
        Dict with detailed pattern info (wattage range, runtime, phase, etc.).
    """
    appliances = nilm_data.get("discovered_appliances", [])
    appliance = next((a for a in appliances if a.get("name") == slot_name), None)

    if not appliance:
        return {"error": "Appliance not found"}

    # Extract pattern details
    return {
        "name": slot_name,
        "wattage_avg": appliance.get("avg_watt", 0),
        "wattage_min": appliance.get("min_watt", 0),
        "wattage_max": appliance.get("max_watt", 0),
        "runtime_avg_min": appliance.get("avg_runtime_min", 0),
        "events": appliance.get("events", 0),
        "energy_per_cycle_kwh": appliance.get("avg_kwh", 0),
        "match_confidence": appliance.get("match", 0),
        "phase": appliance.get("phase", "Unknown"),
        "time_pattern": appliance.get("time_pattern", {}),  # Hour-of-day histogram
        "user_confirmed": appliance.get("user_confirmed", False),
    }
