"""Natural Language Anomaly Explanations — convert z-scores to plain English.

Transforms raw anomaly metrics into human-readable descriptions so users
understand *why* Habitus flagged something rather than just seeing a score.
"""

from __future__ import annotations

import logging

log = logging.getLogger("habitus")

_DAY_NAMES = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"]


def severity_label(z_score: float) -> str:
    """Map a z-score to a human-readable severity tier.

    Args:
        z_score: Absolute z-score value.

    Returns:
        One of ``"normal"``, ``"elevated"``, ``"anomaly"``, ``"critical"``,
        ``"extreme"``.
    """
    z = abs(z_score)
    if z < 2:
        return "normal"
    if z < 3:
        return "elevated"
    if z < 4:
        return "anomaly"
    if z < 5:
        return "critical"
    return "extreme"


def format_time_slot(slot: str) -> str:
    """Convert a ``"HH_DOW"`` slot string to human-readable text.

    Args:
        slot: Slot string like ``"14_3"`` (hour 14, day-of-week 3 = Thursday).

    Returns:
        Formatted string like ``"2:00 PM on Thursdays"``.
    """
    try:
        parts = slot.split("_")
        hour = int(parts[0])
        dow = int(parts[1]) if len(parts) > 1 else -1
    except (ValueError, IndexError):
        return slot

    if hour == 0:
        time_str = "12:00 AM"
    elif hour < 12:
        time_str = f"{hour}:00 AM"
    elif hour == 12:
        time_str = "12:00 PM"
    else:
        time_str = f"{hour - 12}:00 PM"

    if 0 <= dow <= 6:
        return f"{time_str} on {_DAY_NAMES[dow]}"
    return time_str


def _friendly_name(entity_id: str, name: str | None = None) -> str:
    """Extract a friendly display name from an entity ID or provided name."""
    if name:
        return name
    if "." in entity_id:
        return entity_id.split(".", 1)[1].replace("_", " ").title()
    return entity_id.replace("_", " ").title()


def explain_entity_anomaly(anomaly: dict) -> str:
    """Generate a plain English explanation for a single entity anomaly.

    Args:
        anomaly: Dict with keys ``entity_id``, ``name``, ``z_score``,
            ``current_value``, ``baseline_mean``, ``baseline_std``,
            ``direction``, ``slot``, ``sensor_type``.

    Returns:
        A single human-readable sentence describing the anomaly.
    """
    name = _friendly_name(anomaly.get("entity_id", ""), anomaly.get("name"))
    z = anomaly.get("z_score", 0)
    current = anomaly.get("current_value")
    mean = anomaly.get("baseline_mean")
    direction = anomaly.get("direction", "high")
    slot = anomaly.get("slot", "")
    sensor_type = anomaly.get("sensor_type", "gauge")

    time_ctx = ""
    if slot:
        time_ctx = f" at {format_time_slot(slot)}"

    # Binary sensors
    if sensor_type == "binary":
        state = "on" if current else "off"
        expected = "on" if (mean or 0) > 0.5 else "off"
        if state == expected:
            return f"{name} is {state}, which is normal{time_ctx}."
        if current == 0 and mean and mean > 0.5:
            return f"{name} has been off — it's normally active{time_ctx}."
        return f"{name} is unexpectedly {state}{time_ctx} " f"(normally {expected})."

    # Gauge / accumulating / setpoint
    if current is not None and mean is not None:
        ratio = current / mean if mean and mean != 0 else 0
        unit = _guess_unit(anomaly.get("entity_id", ""))

        if direction == "high":
            if ratio > 2:
                return (
                    f"{name} is {current:.0f}{unit} — "
                    f"{ratio:.1f}x higher than the usual "
                    f"{mean:.0f}{unit}{time_ctx}."
                )
            return f"{name} is {current:.0f}{unit}, above the usual " f"{mean:.0f}{unit}{time_ctx}."
        else:
            if mean != 0 and current / mean < 0.5:
                return (
                    f"{name} dropped to {current:.0f}{unit} — "
                    f"well below the typical {mean:.0f}{unit}{time_ctx}."
                )
            return f"{name} is {current:.0f}{unit}, below the usual " f"{mean:.0f}{unit}{time_ctx}."

    # Fallback
    sev = severity_label(z)
    return f"{name} shows {sev} deviation (z={z:.1f}){time_ctx}."


def _guess_unit(entity_id: str) -> str:
    """Guess the unit suffix from the entity ID."""
    eid = entity_id.lower()
    if "_w" in eid or "watt" in eid or "power" in eid:
        return "W"
    if "temperature" in eid or "_temp" in eid:
        return "\u00b0C"
    if "humidity" in eid:
        return "%"
    if "soc" in eid or "battery" in eid:
        return "%"
    if "kwh" in eid or "energy" in eid:
        return " kWh"
    return ""


def explain_overall_score(score: int, entity_anomalies: list[dict], total_entities: int = 0) -> str:
    """Generate a 2-3 sentence summary of the overall anomaly state.

    Args:
        score: Global anomaly score (0-100).
        entity_anomalies: List of entity anomaly dicts.
        total_entities: Total number of tracked entities.

    Returns:
        A concise plain English summary.
    """
    n_anom = len(entity_anomalies)

    if score < 30:
        if total_entities:
            return (
                f"Your home is behaving normally (score: {score}/100). "
                f"All {total_entities} tracked sensors are within expected ranges."
            )
        return f"Everything looks normal (score: {score}/100). No anomalies detected."

    if score < 60:
        top = entity_anomalies[:2]
        descs = [explain_entity_anomaly(a) for a in top]
        joined = " ".join(descs)
        return f"Moderate anomaly detected (score: {score}/100). " f"{joined}"

    # High / critical
    top = entity_anomalies[:3]
    descs = [explain_entity_anomaly(a) for a in top]
    joined = " ".join(descs)
    label = "Critical" if score >= 80 else "Significant"
    return (
        f"{label} anomaly (score: {score}/100). "
        f"{n_anom} sensor{'s' if n_anom != 1 else ''} significantly "
        f"outside normal patterns. {joined}"
    )


def explain_drift(drift_data: dict) -> str:
    """Generate a plain English description of routine drift.

    Args:
        drift_data: Dict from ``drift.json`` with ``drifts`` list
            and ``summary`` string.

    Returns:
        Human-readable drift description.
    """
    drifts = drift_data.get("drifts", [])
    sig = [d for d in drifts if d.get("significant")]

    if not sig:
        return "No significant changes in your daily routines."

    parts = []
    for d in sig:
        metric = d.get("metric", "")
        diff_min = abs(d.get("diff_min", 0))
        direction = d.get("direction", "later")

        if "morning" in metric:
            parts.append(
                f"Your morning routine has shifted {diff_min:.0f} minutes "
                f"{direction} over the past 2 weeks."
            )
        elif "bedtime" in metric:
            parts.append(f"Bedtime has moved {diff_min:.0f} minutes {direction} recently.")
        elif "peak" in metric:
            parts.append(f"Peak activity has shifted {diff_min:.0f} minutes {direction}.")
        elif "lights" in metric:
            diff_h = abs(d.get("diff", 0))
            more_less = "more" if d.get("diff", 0) > 0 else "fewer"
            parts.append(f"Lights are on {diff_h:.1f} hours {more_less} per day than usual.")

    return " ".join(parts) if parts else drift_data.get("summary", "Routine drift detected.")
