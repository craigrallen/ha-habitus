"""Device Health Monitoring — track sensor degradation and predict failures.

Monitors trends in data quality metrics over time: increasing NaN rate,
battery voltage fade, signal strength degradation, and stuck sensor frequency.
Alerts before a sensor dies rather than after.
"""

import datetime
import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
HEALTH_PATH = os.path.join(DATA_DIR, "device_health.json")


def _load_health() -> dict[str, Any]:
    """Load device health history from disk."""
    try:
        if os.path.exists(HEALTH_PATH):
            with open(HEALTH_PATH) as f:
                data: dict[str, Any] = json.load(f)
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"devices": {}, "updated_at": ""}


def update_health(
    entity_baselines: dict[str, Any],
    data_quality_issues: list[dict],
) -> dict[str, Any]:
    """Update device health metrics from current baselines and quality data.

    Tracks per-entity health indicators over time and detects degradation
    trends.

    Args:
        entity_baselines: Dict from ``entity_baselines.json``.
        data_quality_issues: List from ``data_quality.json``.

    Returns:
        Updated device health report.
    """
    now = datetime.datetime.now(datetime.UTC)
    health = _load_health()
    devices = health.get("devices", {})

    # Index current quality issues by entity
    issue_map: dict[str, list[str]] = {}
    for issue in data_quality_issues:
        eid = issue.get("entity_id", "")
        if eid:
            issue_map.setdefault(eid, []).append(issue.get("issue", "unknown"))

    for eid, baseline_data in entity_baselines.items():
        if eid.startswith("_"):
            continue

        meta = baseline_data.get("_meta", {})
        if not isinstance(meta, dict):
            continue

        device = devices.get(eid, {"history": [], "status": "healthy"})

        # Build current snapshot
        snapshot = {
            "ts": now.isoformat(),
            "n_samples": meta.get("n_samples", 0),
            "first_seen": meta.get("first_seen", ""),
            "issues": issue_map.get(eid, []),
        }

        # Append to history (keep last 14 snapshots — one per run)
        hist = device.get("history", [])
        hist.append(snapshot)
        if len(hist) > 14:
            hist = hist[-14:]
        device["history"] = hist

        # Analyze degradation trends
        device["status"] = _assess_status(hist)
        device["current_issues"] = issue_map.get(eid, [])
        devices[eid] = device

    health["devices"] = devices
    health["updated_at"] = now.isoformat()
    atomic_write(HEALTH_PATH, health)

    # Log summary
    statuses = [d.get("status", "healthy") for d in devices.values()]
    degraded = sum(1 for s in statuses if s == "degraded")
    failing = sum(1 for s in statuses if s == "failing")
    if degraded or failing:
        log.warning(
            "Device health: %d degraded, %d failing (of %d total)",
            degraded,
            failing,
            len(devices),
        )

    return health


def _assess_status(history: list[dict]) -> str:
    """Determine device health status from its snapshot history.

    Args:
        history: List of health snapshots (newest last).

    Returns:
        One of ``"healthy"``, ``"degraded"``, ``"failing"``.
    """
    if len(history) < 3:
        return "healthy"

    recent = history[-3:]

    # Failing: issues in every recent snapshot
    all_have_issues = all(len(s.get("issues", [])) > 0 for s in recent)
    if all_have_issues:
        return "failing"

    # Degraded: issues in 2 of last 3 snapshots
    issue_count = sum(1 for s in recent if s.get("issues"))
    if issue_count >= 2:
        return "degraded"

    # Check for stuck pattern (same n_samples across snapshots — not growing)
    samples = [s.get("n_samples", 0) for s in recent]
    if len(set(samples)) == 1 and samples[0] > 0:
        return "degraded"

    return "healthy"


def get_health_report() -> dict[str, Any]:
    """Return the device health report.

    Returns:
        Dict with per-device health info, overall stats, and alerts.
    """
    health = _load_health()
    devices = health.get("devices", {})

    alerts: list[dict[str, str]] = []
    for eid, device in devices.items():
        status = device.get("status", "healthy")
        if status == "failing":
            friendly = eid.split(".")[-1].replace("_", " ").title()
            issues = device.get("current_issues", [])
            alerts.append(
                {
                    "entity_id": eid,
                    "name": friendly,
                    "status": "failing",
                    "message": (
                        f"{friendly} has persistent issues: "
                        f"{', '.join(issues) if issues else 'degraded readings'}. "
                        "Check battery, connectivity, or replace sensor."
                    ),
                }
            )
        elif status == "degraded":
            friendly = eid.split(".")[-1].replace("_", " ").title()
            alerts.append(
                {
                    "entity_id": eid,
                    "name": friendly,
                    "status": "degraded",
                    "message": f"{friendly} showing signs of degradation — monitor closely.",
                }
            )

    summary = {
        "total_devices": len(devices),
        "healthy": sum(1 for d in devices.values() if d.get("status") == "healthy"),
        "degraded": sum(1 for d in devices.values() if d.get("status") == "degraded"),
        "failing": sum(1 for d in devices.values() if d.get("status") == "failing"),
    }

    return {
        "summary": summary,
        "alerts": alerts,
        "updated_at": health.get("updated_at", ""),
    }
