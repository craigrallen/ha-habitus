"""Energy Insights — derives peak hours, top consumers, waste, and solar self-consumption.

Reads stored baseline artefacts (baseline.json, entity_baselines.json) and computes
four dashboard metrics without any live HA calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")

# Active-hours window: 07:00–21:59
ACTIVE_HOURS: frozenset[int] = frozenset(range(7, 22))

# Keywords that identify power-measurement (watt) entities
_POWER_KEYWORDS: tuple[str, ...] = ("_w", "watt", "power", "electric", "load")

# Minimum mean wattage to be counted as a consumer
_MIN_CONSUMER_W: float = 1.0

# Minimum off-peak mean wattage to count as waste
_WASTE_THRESHOLD_W: float = 10.0


def _load_json(path: str, default: Any) -> Any:
    """Load a JSON file, returning *default* on any error.

    Args:
        path: Absolute path to the JSON file.
        default: Value returned when the file is missing or unparseable.

    Returns:
        Parsed JSON value or *default*.
    """
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def _is_power_entity(entity_id: str) -> bool:
    """Return True when *entity_id* looks like a power (watt) sensor.

    Args:
        entity_id: Full HA entity id, e.g. ``sensor.kitchen_electric_consumed_w``.

    Returns:
        True if the entity appears to measure power in watts.
    """
    eid = entity_id.lower()
    return any(kw in eid for kw in _POWER_KEYWORDS)


def _is_solar_entity(entity_id: str) -> bool:
    """Return True when *entity_id* looks like a solar-production sensor.

    Args:
        entity_id: Full HA entity id.

    Returns:
        True if the entity appears to measure solar power output.
    """
    eid = entity_id.lower()
    return "solar" in eid and _is_power_entity(eid)


def _entity_hour_means(slots: dict[str, Any]) -> dict[int, float]:
    """Collapse hour×dow baseline slots into a mean value per hour-of-day.

    Args:
        slots: Dict keyed by ``"{hour}_{dow}"`` with ``{"mean": float, ...}`` values.

    Returns:
        Dict mapping hour-of-day (0–23) → average mean across all days-of-week.
    """
    hour_sums: dict[int, list[float]] = {}
    for key, val in slots.items():
        if not isinstance(val, dict):
            continue
        parts = key.split("_")
        if len(parts) < 2:
            continue
        try:
            hour = int(parts[0])
        except ValueError:
            continue
        mean = val.get("mean", 0.0)
        hour_sums.setdefault(hour, []).append(float(mean))
    return {h: sum(vs) / len(vs) for h, vs in hour_sums.items() if vs}


def compute_peak_hours(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the top-3 peak usage hours from the aggregate baseline.

    Args:
        baseline: Contents of *baseline.json*, keyed by ``"{hour}_{dow}"`` with
            ``{"mean_power": float, ...}`` values.

    Returns:
        List of up to 3 dicts ``{"hour": int, "mean_power_w": float, "label": str}``,
        sorted by *mean_power_w* descending.
    """
    hour_sums: dict[int, list[float]] = {}
    for key, val in baseline.items():
        if not isinstance(val, dict):
            continue
        parts = key.split("_")
        if len(parts) < 2:
            continue
        try:
            hour = int(parts[0])
        except ValueError:
            continue
        mean_power = val.get("mean_power", 0.0)
        hour_sums.setdefault(hour, []).append(float(mean_power))

    hourly: list[dict[str, Any]] = []
    for hour, values in hour_sums.items():
        mean_w = sum(values) / len(values)
        hourly.append(
            {
                "hour": hour,
                "mean_power_w": round(mean_w, 1),
                "label": f"{hour:02d}:00",
            }
        )
    hourly.sort(key=lambda x: x["mean_power_w"], reverse=True)
    return hourly[:3]


def compute_top_consumers(entity_baselines: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the top-5 power-consuming entities by overall mean wattage.

    Args:
        entity_baselines: Contents of *entity_baselines.json*, with internal
            ``_*`` keys already stripped.

    Returns:
        List of up to 5 dicts ``{"entity_id": str, "name": str, "mean_w": float}``,
        sorted by *mean_w* descending.
    """
    consumers: list[dict[str, Any]] = []
    for eid, slots in entity_baselines.items():
        if not isinstance(slots, dict):
            continue
        if not _is_power_entity(eid):
            continue
        hour_means = _entity_hour_means(slots)
        if not hour_means:
            continue
        overall_mean = sum(hour_means.values()) / len(hour_means)
        if overall_mean < _MIN_CONSUMER_W:
            continue
        name = eid.split(".")[-1].replace("_", " ").title()
        consumers.append(
            {
                "entity_id": eid,
                "name": name,
                "mean_w": round(overall_mean, 1),
            }
        )
    consumers.sort(key=lambda x: x["mean_w"], reverse=True)
    return consumers[:5]


def compute_waste(entity_baselines: dict[str, Any]) -> list[dict[str, Any]]:
    """Identify power entities drawing significant wattage outside active hours.

    A device is flagged as "waste" when its mean power during off-peak hours
    (outside ``ACTIVE_HOURS``) exceeds ``_WASTE_THRESHOLD_W`` and it also has
    measurable on-peak consumption, suggesting it should not be running overnight.

    Args:
        entity_baselines: Contents of *entity_baselines.json*, with internal
            ``_*`` keys already stripped.

    Returns:
        List of dicts ``{"entity_id": str, "name": str, "off_peak_mean_w": float,
        "on_peak_mean_w": float}``, sorted by *off_peak_mean_w* descending.
    """
    waste: list[dict[str, Any]] = []
    for eid, slots in entity_baselines.items():
        if not isinstance(slots, dict):
            continue
        if not _is_power_entity(eid):
            continue
        on_vals: list[float] = []
        off_vals: list[float] = []
        for key, val in slots.items():
            if not isinstance(val, dict):
                continue
            parts = key.split("_")
            if len(parts) < 2:
                continue
            try:
                hour = int(parts[0])
            except ValueError:
                continue
            m = float(val.get("mean", 0.0))
            if hour in ACTIVE_HOURS:
                on_vals.append(m)
            else:
                off_vals.append(m)
        if not off_vals:
            continue
        off_mean = sum(off_vals) / len(off_vals)
        on_mean = sum(on_vals) / len(on_vals) if on_vals else 0.0
        if off_mean >= _WASTE_THRESHOLD_W and on_mean >= _MIN_CONSUMER_W:
            name = eid.split(".")[-1].replace("_", " ").title()
            waste.append(
                {
                    "entity_id": eid,
                    "name": name,
                    "off_peak_mean_w": round(off_mean, 1),
                    "on_peak_mean_w": round(on_mean, 1),
                }
            )
    waste.sort(key=lambda x: x["off_peak_mean_w"], reverse=True)
    return waste


def compute_solar_ratio(entity_baselines: dict[str, Any]) -> dict[str, Any] | None:
    """Compute solar self-consumption ratio when a solar sensor is present.

    Self-consumption is estimated as ``min(solar_mean, load_mean)`` / ``solar_mean``
    across all shared hour-of-day slots.  When no load entity is identified the
    ratio is ``None`` but the solar entity and its mean output are still returned.

    Args:
        entity_baselines: Contents of *entity_baselines.json*, with internal
            ``_*`` keys already stripped.

    Returns:
        Dict ``{"ratio": float | None, "solar_entity": str, "solar_mean_w": float}``
        or ``None`` if no solar sensor is found.
    """
    solar_eid: str | None = None
    for eid in entity_baselines:
        if _is_solar_entity(eid):
            solar_eid = eid
            break
    if solar_eid is None:
        return None

    solar_slots = entity_baselines[solar_eid]
    if not isinstance(solar_slots, dict):
        return None
    solar_hour_means = _entity_hour_means(solar_slots)

    total_solar = sum(solar_hour_means.values())
    solar_mean_w = round(total_solar / max(len(solar_hour_means), 1), 1)

    if total_solar <= 0:
        return {
            "ratio": None,
            "solar_entity": solar_eid,
            "solar_mean_w": solar_mean_w,
            "reason": "no_solar_generation",
        }

    # Try to find a total load entity for self-consumption calculation
    load_hour_means: dict[int, float] = {}
    _load_keywords = ("total_load", "mastervolt_total", "grid_import", "house_load", "mains")
    for eid, slots in entity_baselines.items():
        if eid == solar_eid or not isinstance(slots, dict):
            continue
        eid_lower = eid.lower()
        if any(kw in eid_lower for kw in _load_keywords):
            load_hour_means = _entity_hour_means(slots)
            break

    if load_hour_means:
        total_self_consumed = sum(
            min(solar_hour_means.get(h, 0.0), load_hour_means.get(h, 0.0)) for h in solar_hour_means
        )
        ratio: float | None = round(total_self_consumed / total_solar, 3)
    else:
        ratio = None

    return {
        "ratio": ratio,
        "solar_entity": solar_eid,
        "solar_mean_w": solar_mean_w,
    }


def compute_insights(data_dir: str | None = None) -> dict[str, Any]:
    """Compute all energy insights from stored baseline artefacts.

    Reads *baseline.json* and *entity_baselines.json* from *data_dir* and
    derives four dashboard metrics without any live HA API calls.

    Args:
        data_dir: Directory containing the baseline JSON files.  Defaults to the
            ``DATA_DIR`` environment variable (``/data`` in the container).

    Returns:
        Dict with keys:
        - ``peak_hours``: top-3 peak usage hours with wattage.
        - ``top_consumers``: top-5 power entities by mean wattage.
        - ``estimated_waste``: entities drawing power outside active hours.
        - ``solar_self_consumption``: solar ratio dict or ``None``.
    """
    resolved_dir = data_dir or os.environ.get("DATA_DIR", "/data")

    baseline = _load_json(os.path.join(resolved_dir, "baseline.json"), {})
    entity_baselines_raw = _load_json(os.path.join(resolved_dir, "entity_baselines.json"), {})
    # Strip internal runtime keys (e.g. "_z_score_run")
    entity_baselines: dict[str, Any] = {
        k: v for k, v in entity_baselines_raw.items() if not k.startswith("_")
    }

    return {
        "peak_hours": compute_peak_hours(baseline),
        "top_consumers": compute_top_consumers(entity_baselines),
        "estimated_waste": compute_waste(entity_baselines),
        "solar_self_consumption": compute_solar_ratio(entity_baselines),
    }
