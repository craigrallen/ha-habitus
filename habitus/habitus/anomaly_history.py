"""Anomaly History Tracking — persist anomaly events for trend analysis."""

import datetime
import json
import logging
import os

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
HISTORY_PATH = os.path.join(DATA_DIR, "anomaly_history.json")

# Keep last 7 days of anomaly events
RETENTION_DAYS = 7


def record_anomalies(anomaly_score: float, entity_anomalies: list) -> None:
    """Record current anomaly state to history.

    Args:
        anomaly_score: Overall anomaly score (0-100).
        entity_anomalies: List of entity anomaly dicts from anomaly_breakdown.
    """
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now - datetime.timedelta(days=RETENTION_DAYS)

    # Load existing history
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load anomaly history: {e}")
            history = []

    # Prune old entries
    history = [
        h
        for h in history
        if datetime.datetime.fromisoformat(h["timestamp"].replace("Z", "+00:00")) > cutoff
    ]

    # Add new entry (only if score is elevated or entities flagged)
    if anomaly_score >= 40 or entity_anomalies:
        event = {
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "anomaly_score": anomaly_score,
            "entity_count": len(entity_anomalies),
            "entities": [
                {
                    "entity_id": e.get("entity_id", e.get("name", "")),
                    "name": e.get("name", ""),
                    "z_score": e.get("z_score", 0),
                    "severity": (
                        "critical"
                        if e.get("z_score", 0) >= 4
                        else "warning" if e.get("z_score", 0) >= 3 else "info"
                    ),
                    "description": e.get("description", ""),
                }
                for e in entity_anomalies[:10]  # Keep top 10 per event
            ],
        }
        history.append(event)

    # Save
    try:
        from .utils import atomic_write as _atomic_write

        _atomic_write(HISTORY_PATH, history)
        log.debug(f"Anomaly history updated: {len(history)} events in last {RETENTION_DAYS}d")
    except Exception as e:
        log.warning(f"Failed to save anomaly history: {e}")


def get_history() -> list:
    """Load anomaly history.

    Returns:
        List of anomaly events, newest first.
    """
    if not os.path.exists(HISTORY_PATH):
        return []

    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    except Exception as e:
        log.warning(f"Failed to load anomaly history: {e}")
        return []


def get_patterns() -> dict:
    """Analyze anomaly history for recurring patterns.

    Returns:
        Dict with pattern analysis (recurring entities, time patterns, etc.).
    """
    history = get_history()
    if not history:
        return {"patterns": [], "recurring_entities": []}

    # Count entity occurrences
    entity_counts = {}
    hourly_counts = [0] * 24
    daily_counts = [0] * 7

    for event in history:
        ts = datetime.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        hourly_counts[ts.hour] += 1
        daily_counts[ts.weekday()] += 1

        for ent in event.get("entities", []):
            eid = ent.get("entity_id", "")
            if eid:
                entity_counts[eid] = entity_counts.get(eid, 0) + 1

    # Recurring entities (appeared in 3+ events)
    recurring = [
        {"entity_id": eid, "occurrences": count}
        for eid, count in entity_counts.items()
        if count >= 3
    ]
    recurring.sort(key=lambda x: x["occurrences"], reverse=True)

    # Peak hours (hours with >avg anomalies)
    avg_hourly = sum(hourly_counts) / 24
    peak_hours = [h for h in range(24) if hourly_counts[h] > avg_hourly * 1.5]

    # Peak days
    avg_daily = sum(daily_counts) / 7
    peak_days = [
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]
        for d in range(7)
        if daily_counts[d] > avg_daily * 1.5
    ]

    return {
        "patterns": [
            (
                {
                    "type": "time_pattern",
                    "description": f"Anomalies peak at hours: {', '.join(f'{h:02d}:00' for h in peak_hours)}",
                    "hours": peak_hours,
                }
                if peak_hours
                else None
            ),
            (
                {
                    "type": "day_pattern",
                    "description": f"Anomalies more frequent on: {', '.join(peak_days)}",
                    "days": peak_days,
                }
                if peak_days
                else None
            ),
        ],
        "recurring_entities": recurring[:10],
        "total_events": len(history),
        "avg_score": sum(e["anomaly_score"] for e in history) / len(history) if history else 0,
    }
