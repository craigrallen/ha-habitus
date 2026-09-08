"""Sound Alert — browser notification sound configuration for dashboard tablets.

Provides a configuration API for enabling/disabling browser-based alert
sounds on Lovelace dashboard tablets. The actual sound playback is handled
by the frontend JavaScript; this module manages the server-side config.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
_CONFIG_PATH = os.path.join(DATA_DIR, "sound_alert_config.json")

_DEFAULT_THRESHOLD = 70
_DEFAULT_SOUND_URL = "/local/habitus/alert.mp3"


def _load_config() -> dict[str, Any]:
    """Load sound alert configuration from disk.

    Returns:
        Config dict with ``enabled``, ``threshold``, and ``sound_url`` keys.
    """
    try:
        with open(_CONFIG_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "enabled": False,
            "threshold": _DEFAULT_THRESHOLD,
            "sound_url": _DEFAULT_SOUND_URL,
        }


def get_alert_config() -> dict[str, Any]:
    """Return the current sound alert configuration.

    Returns:
        Dict with keys: ``enabled`` (bool), ``threshold`` (int),
        ``sound_url`` (str).
    """
    return _load_config()


def set_alert_config(enabled: bool, threshold: int = _DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Update the sound alert configuration.

    Args:
        enabled: Whether browser alert sounds are enabled.
        threshold: Anomaly score threshold (0-100) that triggers the sound.
            Defaults to 70.

    Returns:
        Updated config dict.
    """
    config = _load_config()
    config["enabled"] = enabled
    config["threshold"] = max(0, min(100, threshold))
    atomic_write(_CONFIG_PATH, config)
    log.info("Sound alert config updated: enabled=%s, threshold=%d", enabled, threshold)
    return config
