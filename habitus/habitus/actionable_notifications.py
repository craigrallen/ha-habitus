"""Actionable Notifications — HA companion app with action buttons.

Sends rich notifications via HA mobile_app service with action buttons:
- Mark Normal (suppress this anomaly for 30 days)
- View Details (deep link to Habitus dashboard)
- Snooze 24h (suppress all alerts for 24 hours)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import requests  # type: ignore[import-untyped]

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SNOOZE_PATH = os.path.join(DATA_DIR, "notification_snooze.json")


def _is_snoozed() -> bool:
    """Check if notifications are currently snoozed."""
    try:
        if os.path.exists(SNOOZE_PATH):
            with open(SNOOZE_PATH) as f:
                data = json.load(f)
            until = data.get("snoozed_until", "")
            if until:
                snooze_end = datetime.datetime.fromisoformat(until)
                if datetime.datetime.now(datetime.UTC) < snooze_end:
                    return True
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return False


def snooze(hours: int = 24) -> dict[str, str]:
    """Snooze all notifications for the specified hours.

    Args:
        hours: Number of hours to snooze (default 24).

    Returns:
        Dict with snooze end time.
    """
    until = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=hours)
    data = {
        "snoozed_until": until.isoformat(),
        "snoozed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    atomic_write(SNOOZE_PATH, data)
    log.info("Notifications snoozed until %s", until.isoformat())
    return {"snoozed_until": until.isoformat()}


def unsnooze() -> dict[str, str]:
    """Cancel notification snooze."""
    if os.path.exists(SNOOZE_PATH):
        os.remove(SNOOZE_PATH)
    log.info("Notification snooze cancelled")
    return {"snoozed_until": ""}


def send_anomaly_alert(
    ha_url: str,
    ha_token: str,
    notify_service: str,
    score: int,
    top_anomaly: str,
    entity_id: str = "",
) -> bool:
    """Send an actionable anomaly notification via HA.

    If the notify service supports mobile_app actions (iOS/Android companion),
    includes Mark Normal, View Details, and Snooze buttons.

    Args:
        ha_url: Home Assistant base URL.
        ha_token: Long-lived access token.
        notify_service: HA notify service (e.g. ``"notify.mobile_app_phone"``).
        score: Current anomaly score (0-100).
        top_anomaly: Description of the top anomaly.
        entity_id: Entity ID of the top anomaly (for feedback action).

    Returns:
        True if notification was sent successfully.
    """
    if _is_snoozed():
        log.debug("Notification snoozed — skipping anomaly alert")
        return False

    severity = "critical" if score >= 80 else "warning" if score >= 60 else "info"
    title = f"Habitus — {severity.title()} Anomaly ({score}/100)"
    message = f"{top_anomaly}\n\nScore: {score}/100"

    # Build payload with mobile_app actions
    payload: dict[str, Any] = {
        "title": title,
        "message": message,
        "data": {
            "tag": "habitus_anomaly",
            "group": "habitus",
            "importance": "high" if score >= 70 else "default",
            "actions": [
                {
                    "action": "HABITUS_MARK_NORMAL",
                    "title": "Mark Normal",
                    "uri": "/api/hassio_ingress/local_habitus/api/feedback",
                },
                {
                    "action": "HABITUS_VIEW",
                    "title": "View Details",
                    "uri": "/habitus",
                },
                {
                    "action": "HABITUS_SNOOZE",
                    "title": "Snooze 24h",
                },
            ],
        },
    }

    # Add entity-specific feedback data
    if entity_id:
        payload["data"]["entity_id"] = entity_id

    try:
        resp = requests.post(
            f"{ha_url}/api/services/{notify_service.replace('.', '/')}",
            headers={"Authorization": f"Bearer {ha_token}"},
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            log.info("Actionable notification sent: %s (score=%d)", severity, score)
            return True
        log.warning("Notification failed: HTTP %d", resp.status_code)
        return False
    except requests.RequestException as e:
        log.warning("Failed to send notification: %s", e)
        return False


def handle_action(action: str, entity_id: str = "") -> dict[str, Any]:
    """Handle a mobile app notification action callback.

    Args:
        action: Action identifier from HA (e.g. ``"HABITUS_MARK_NORMAL"``).
        entity_id: Entity ID associated with the action.

    Returns:
        Result dict with status.
    """
    if action == "HABITUS_MARK_NORMAL" and entity_id:
        from . import feedback as _fb

        now = datetime.datetime.now(datetime.UTC)
        hour = now.hour
        dow = now.weekday()
        _fb.record_feedback(
            anomaly_id=f"{entity_id}_{hour}_{dow}",
            action="dismissed",
            entity_id=entity_id,
            details="Marked normal via notification",
        )
        return {"ok": True, "action": "marked_normal", "entity_id": entity_id}

    if action == "HABITUS_SNOOZE":
        result = snooze(24)
        return {"ok": True, "action": "snoozed", **result}

    if action == "HABITUS_VIEW":
        return {"ok": True, "action": "view"}

    return {"ok": False, "error": f"Unknown action: {action}"}
