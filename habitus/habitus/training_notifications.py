"""Training Notifications — notify Home Assistant when training completes or fails.

Sends persistent notifications via the HA REST API so the user knows
when Habitus model training finishes (or errors out) without checking logs.
"""

from __future__ import annotations

import logging
from typing import Any

import requests  # type: ignore[import-untyped]

log = logging.getLogger("habitus")

_TIMEOUT_SEC = 10


def send_training_complete(
    ha_url: str,
    ha_token: str,
    notify_service: str,
    training_days: int,
    duration_sec: float,
    entity_count: int,
) -> bool:
    """Notify HA that model training completed successfully.

    Args:
        ha_url: Home Assistant base URL (e.g. ``http://supervisor/core``).
        ha_token: Long-lived access token or Supervisor token.
        notify_service: HA notify service name (e.g. ``notify.mobile_app``).
        training_days: Number of days of history used for training.
        duration_sec: Training duration in seconds.
        entity_count: Number of entities in the trained model.

    Returns:
        True if the notification was sent successfully, False otherwise.
    """
    title = "Habitus Training Complete"
    message = (
        f"Model training finished in {duration_sec:.1f}s. "
        f"Trained on {training_days} days of data across "
        f"{entity_count} entities."
    )

    return _send_notification(ha_url, ha_token, notify_service, title, message)


def send_training_failed(
    ha_url: str,
    ha_token: str,
    notify_service: str,
    error: str,
) -> bool:
    """Notify HA that model training failed.

    Args:
        ha_url: Home Assistant base URL.
        ha_token: Long-lived access token or Supervisor token.
        notify_service: HA notify service name.
        error: Error message or description of the failure.

    Returns:
        True if the notification was sent successfully, False otherwise.
    """
    title = "Habitus Training Failed"
    message = f"Model training encountered an error: {error}"

    return _send_notification(ha_url, ha_token, notify_service, title, message)


def _send_notification(
    ha_url: str,
    ha_token: str,
    notify_service: str,
    title: str,
    message: str,
) -> bool:
    """Send a notification via the HA REST API.

    Args:
        ha_url: Home Assistant base URL.
        ha_token: Access token.
        notify_service: Service name (e.g. ``notify.mobile_app``).
        title: Notification title.
        message: Notification body.

    Returns:
        True on success, False on failure.
    """
    url = f"{ha_url.rstrip('/')}/api/services/{notify_service.replace('.', '/')}"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "title": title,
        "message": message,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT_SEC)
        resp.raise_for_status()
        log.info("Notification sent: %s", title)
        return True
    except requests.RequestException as exc:
        log.error("Failed to send notification '%s': %s", title, exc)
        return False
