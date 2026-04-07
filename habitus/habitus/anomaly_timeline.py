"""Anomaly timeline — reconstruct what happened during an anomaly event."""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
TIMELINE_PATH = os.path.join(DATA_DIR, "anomaly_timeline.json")
ENTITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")


def build_timeline(
    anomaly_history: list[dict[str, Any]],
    window_hours: int = 4,
) -> list[dict[str, Any]]:
    """Build an ordered timeline of sensor events around anomaly periods.

    For each anomaly event in the history, extracts which sensors deviated,
    in what order, by how much, and when.

    Args:
        anomaly_history: List of anomaly event dicts. Each should contain
            at least ``timestamp``, ``entity_id``, ``value``, ``baseline_mean``
            (or ``mean``), and ``baseline_std`` (or ``std``).
        window_hours: How many hours around each anomaly to consider when
            building the narrative. Defaults to 4.

    Returns:
        Ordered list of timeline entry dicts with keys: timestamp, entity_id,
        name, event_type, value, baseline, deviation_pct, description.
    """
    if not anomaly_history:
        return []

    timeline: list[dict[str, Any]] = []

    for event in anomaly_history:
        ts = event.get("timestamp", "")
        eid = event.get("entity_id", "unknown")
        value = event.get("value")
        mean = event.get("baseline_mean") or event.get("mean", 0)
        std = event.get("baseline_std") or event.get("std", 1)

        if value is None:
            continue

        try:
            val = float(value)
            mean_f = float(mean)
            std_f = float(std) if std else 1.0
        except (ValueError, TypeError):
            continue

        # Calculate deviation percentage from baseline
        if mean_f != 0:
            deviation_pct = round(((val - mean_f) / abs(mean_f)) * 100.0, 1)
        elif std_f > 0:
            deviation_pct = round(((val - mean_f) / std_f) * 100.0, 1)
        else:
            deviation_pct = 0.0

        event_type = _classify_event(val, mean_f, std_f)
        name = _friendly_name(eid)
        description = _describe_event(eid, name, val, mean_f, deviation_pct, event_type)

        timeline.append(
            {
                "timestamp": ts,
                "entity_id": eid,
                "name": name,
                "event_type": event_type,
                "value": round(val, 2),
                "baseline": round(mean_f, 2),
                "deviation_pct": deviation_pct,
                "description": description,
            }
        )

    # Sort by timestamp
    timeline.sort(key=lambda e: e.get("timestamp", ""))
    return timeline


def _classify_event(value: float, mean: float, std: float) -> str:
    """Classify an anomaly event by direction and magnitude.

    Args:
        value: Observed sensor value.
        mean: Baseline mean.
        std: Baseline standard deviation.

    Returns:
        Event type string: "spike", "drop", "sustained_high",
        "sustained_low", or "deviation".
    """
    if std <= 0:
        return "deviation"

    z = (value - mean) / std

    if z > 3.0:
        return "spike"
    if z < -3.0:
        return "drop"
    if z > 2.0:
        return "sustained_high"
    if z < -2.0:
        return "sustained_low"
    return "deviation"


def _friendly_name(entity_id: str) -> str:
    """Derive a human-friendly name from an entity ID.

    Args:
        entity_id: Home Assistant entity ID (e.g., "sensor.kitchen_power").

    Returns:
        Cleaned-up name (e.g., "Kitchen Power").
    """
    # Strip domain prefix
    name = entity_id.split(".", 1)[1] if "." in entity_id else entity_id

    # Replace underscores and title-case
    return name.replace("_", " ").title()


def _describe_event(
    entity_id: str,
    name: str,
    value: float,
    mean: float,
    deviation_pct: float,
    event_type: str,
) -> str:
    """Generate a human-readable description for a timeline event.

    Args:
        entity_id: The HA entity ID.
        name: Friendly name of the entity.
        value: Observed value.
        mean: Baseline mean.
        deviation_pct: Percentage deviation from baseline.
        event_type: Classified event type.

    Returns:
        Plain English description of the event.
    """
    direction = "above" if value > mean else "below"
    abs_pct = abs(deviation_pct)

    type_labels = {
        "spike": "sudden spike",
        "drop": "sudden drop",
        "sustained_high": "sustained high reading",
        "sustained_low": "sustained low reading",
        "deviation": "unusual reading",
    }
    label = type_labels.get(event_type, "unusual reading")

    return f"{name}: {label} at {value:.1f} ({abs_pct:.0f}% {direction} " f"baseline of {mean:.1f})"


def get_latest_timeline() -> dict[str, Any]:
    """Load the most recent entity anomalies and build a timeline.

    Reads ``entity_anomalies.json`` and builds a timeline from the current
    anomaly data.

    Returns:
        Dict with ``timestamp``, ``timeline`` list, and ``summary`` string.
        Returns empty dict if no anomaly data is available.
    """
    if not os.path.exists(ENTITY_ANOMALIES_PATH):
        return {}

    try:
        with open(ENTITY_ANOMALIES_PATH) as f:
            anomaly_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    # Extract the anomaly list — handle both flat list and nested structures
    events: list[dict[str, Any]]
    if isinstance(anomaly_data, list):
        events = anomaly_data
    elif isinstance(anomaly_data, dict):
        events = anomaly_data.get("anomalies") or anomaly_data.get("entities") or []
    else:
        return {}

    if not events:
        return {}

    timeline = build_timeline(events)
    summary = format_timeline_text(timeline)

    return {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "timeline": timeline,
        "event_count": len(timeline),
        "summary": summary,
    }


def format_timeline_text(timeline: list[dict[str, Any]]) -> str:
    """Format a timeline into a human-readable narrative.

    Args:
        timeline: Ordered list of timeline entry dicts.

    Returns:
        Multi-line narrative string describing the sequence of events.
    """
    if not timeline:
        return "No anomaly events to report."

    lines: list[str] = []
    lines.append(f"Anomaly timeline ({len(timeline)} event(s)):")

    # Group by event type for a summary header
    type_counts: dict[str, int] = {}
    for entry in timeline:
        et = entry.get("event_type", "unknown")
        type_counts[et] = type_counts.get(et, 0) + 1

    type_summary = ", ".join(f"{count} {etype}" for etype, count in type_counts.items())
    lines.append(f"  Summary: {type_summary}")
    lines.append("")

    for i, entry in enumerate(timeline, 1):
        ts = entry.get("timestamp", "?")
        desc = entry.get("description", "")
        lines.append(f"  {i}. [{ts}] {desc}")

    return "\n".join(lines)


def save(timeline_data: dict[str, Any]) -> None:
    """Persist anomaly timeline to disk.

    Args:
        timeline_data: Timeline dict (typically from ``get_latest_timeline``).
    """
    atomic_write(TIMELINE_PATH, timeline_data)
    count = timeline_data.get("event_count", 0)
    log.info("Anomaly timeline saved: %d event(s)", count)


def load() -> dict[str, Any]:
    """Load anomaly timeline from disk.

    Returns:
        Previously saved timeline dict, or empty dict if none found.
    """
    if not os.path.exists(TIMELINE_PATH):
        return {}
    try:
        with open(TIMELINE_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
