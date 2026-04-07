"""Anomaly feedback system.

Users can confirm or dismiss anomalies. Confirmed anomalies feed back into
the model to improve detection. Dismissed anomalies widen the "normal" band.

Optional: anonymous data sharing for community model improvement.
"""

import datetime
import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "anomaly_feedback.json")


def _load_feedback() -> dict[str, Any]:
    try:
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "entries": [],
        "stats": {"confirmed": 0, "dismissed": 0, "total": 0},
        "sharing_enabled": False,
    }


def _save_feedback(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def record_feedback(anomaly_id: str, action: str, entity_id: str = "",
                    score: float = 0, details: str = "") -> dict:
    """Record user feedback on an anomaly.

    action: "confirmed" | "dismissed" | "false_positive"
    """
    data = _load_feedback()

    entry = {
        "id": anomaly_id,
        "action": action,
        "entity_id": entity_id,
        "score": score,
        "details": details,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    data["entries"].append(entry)

    # Update stats
    data["stats"]["total"] += 1
    if action == "confirmed":
        data["stats"]["confirmed"] += 1
    elif action in ("dismissed", "false_positive"):
        data["stats"]["dismissed"] += 1

    # Keep last 500 entries
    if len(data["entries"]) > 500:
        data["entries"] = data["entries"][-500:]

    _save_feedback(data)
    log.info("Anomaly feedback: %s → %s (entity=%s, score=%.1f)", anomaly_id, action, entity_id, score)
    return entry


def get_feedback_stats() -> dict:
    """Get feedback statistics for model tuning."""
    data = _load_feedback()
    entries = data.get("entries", [])

    # Per-entity feedback
    entity_stats: dict[str, dict] = {}
    for e in entries:
        eid = e.get("entity_id", "unknown")
        if eid not in entity_stats:
            entity_stats[eid] = {"confirmed": 0, "dismissed": 0}
        if e["action"] == "confirmed":
            entity_stats[eid]["confirmed"] += 1
        else:
            entity_stats[eid]["dismissed"] += 1

    # Entities that are mostly false positives → widen their normal band
    false_positive_entities = [
        eid for eid, stats in entity_stats.items()
        if stats["dismissed"] > 3 and stats["dismissed"] / (stats["confirmed"] + stats["dismissed"]) > 0.8
    ]

    return {
        "stats": data.get("stats", {}),
        "false_positive_entities": false_positive_entities,
        "entity_stats": entity_stats,
        "sharing_enabled": data.get("sharing_enabled", False),
    }


def set_sharing(enabled: bool):
    """Toggle anonymous data sharing."""
    data = _load_feedback()
    data["sharing_enabled"] = enabled
    _save_feedback(data)


def get_anonymous_export() -> dict | None:
    """Export anonymised anomaly data for community sharing.

    Strips all identifying info — only entity domains, scores, and feedback.
    Returns None if sharing is disabled.
    """
    data = _load_feedback()
    if not data.get("sharing_enabled"):
        return None

    anonymised = []
    for e in data.get("entries", []):
        eid = e.get("entity_id", "")
        domain = eid.split(".")[0] if "." in eid else "unknown"
        anonymised.append({
            "domain": domain,
            "score": e.get("score", 0),
            "action": e.get("action", ""),
            "hour": None,  # Could extract from timestamp if needed
        })

    return {
        "version": "1.0",
        "entries": anonymised,
        "stats": data.get("stats", {}),
    }
