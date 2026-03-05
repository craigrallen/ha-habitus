"""Integration Health — stale entity and integration health monitoring.

For every entity in HA states:
- Check last_updated — flag as stale based on domain thresholds
- Check for unavailable/unknown state
- Group by integration (inferred from entity_id prefix or domain)
- Compute integration health score per domain (% entities healthy)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from collections import defaultdict
from typing import Any

import requests

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

INTEGRATION_HEALTH_PATH = os.path.join(DATA_DIR, "integration_health.json")

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))

# Staleness thresholds by domain (seconds)
STALE_THRESHOLDS: dict[str, int] = {
    "sensor": 24 * 3600,           # 24 hours
    "binary_sensor": 7 * 24 * 3600, # 7 days
    "switch": 7 * 24 * 3600,        # 7 days
    "light": 7 * 24 * 3600,         # 7 days
    "climate": 24 * 3600,            # 24 hours
    "motion": 3600,                  # 1 hour (special case for motion sensors)
    "person": 7 * 24 * 3600,         # 7 days
    "media_player": 24 * 3600,       # 24 hours
    "cover": 7 * 24 * 3600,          # 7 days
    "input_boolean": 30 * 24 * 3600, # 30 days
    "automation": 7 * 24 * 3600,     # 7 days
    "default": 7 * 24 * 3600,        # 7 days default
}

UNAVAILABLE_STATES = {"unavailable", "unknown"}


def _ha_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def _fetch_states() -> list[dict[str, Any]]:
    """Fetch all HA states."""
    try:
        r = requests.get(f"{HA_URL}/api/states", headers=_ha_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning("Failed to fetch HA states: %s", e)
    return []


def _get_stale_threshold(entity_id: str, domain: str) -> int:
    """Get staleness threshold in seconds for an entity."""
    eid_lower = entity_id.lower()

    # Motion sensors have shorter threshold
    if "motion" in eid_lower:
        return STALE_THRESHOLDS["motion"]

    return STALE_THRESHOLDS.get(domain, STALE_THRESHOLDS["default"])


def _infer_integration(entity_id: str) -> str:
    """Infer integration name from entity_id prefix or domain."""
    domain = entity_id.split(".")[0]
    eid_lower = entity_id.lower()

    # Known integration prefixes
    integration_patterns = {
        "hue": "Philips Hue",
        "sonos": "Sonos",
        "nest": "Google Nest",
        "ring": "Ring",
        "zwave": "Z-Wave",
        "zigbee": "Zigbee",
        "mqtt": "MQTT",
        "homekit": "HomeKit",
        "tuya": "Tuya",
        "shelly": "Shelly",
        "ikea": "IKEA",
        "fibaro": "Fibaro",
        "deconz": "deCONZ",
        "tasmota": "Tasmota",
        "esphome": "ESPHome",
        "owntracks": "OwnTracks",
        "unifi": "UniFi",
        "modbus": "Modbus",
        "mastervolt": "Mastervolt",
    }

    for pattern, name in integration_patterns.items():
        if pattern in eid_lower:
            return name

    # Fall back to domain as integration name
    return domain.replace("_", " ").title()


def check_entity_health(
    state: dict[str, Any],
    now: datetime.datetime,
) -> dict[str, Any]:
    """Check health of a single entity state.

    Returns:
        Dict with entity_id, status (healthy/stale/unavailable), reason.
    """
    entity_id = state.get("entity_id", "")
    domain = entity_id.split(".")[0]
    state_val = state.get("state", "")
    last_updated = state.get("last_updated", "")

    # Check unavailable/unknown
    if state_val.lower() in UNAVAILABLE_STATES:
        return {
            "entity_id": entity_id,
            "domain": domain,
            "state": state_val,
            "status": "unavailable",
            "reason": f"Entity is {state_val}",
            "last_updated": last_updated,
            "age_hours": None,
            "integration": _infer_integration(entity_id),
        }

    # Check staleness
    age_hours: float | None = None
    if last_updated:
        try:
            lu_dt = datetime.datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            age_seconds = (now - lu_dt).total_seconds()
            age_hours = round(age_seconds / 3600, 1)
            threshold = _get_stale_threshold(entity_id, domain)

            if age_seconds > threshold:
                return {
                    "entity_id": entity_id,
                    "domain": domain,
                    "state": state_val,
                    "status": "stale",
                    "reason": f"Not updated in {age_hours:.1f} hours (threshold: {threshold / 3600:.0f}h)",
                    "last_updated": last_updated,
                    "age_hours": age_hours,
                    "integration": _infer_integration(entity_id),
                }
        except (ValueError, TypeError):
            pass

    return {
        "entity_id": entity_id,
        "domain": domain,
        "state": state_val,
        "status": "healthy",
        "reason": "",
        "last_updated": last_updated,
        "age_hours": age_hours,
        "integration": _infer_integration(entity_id),
    }


def run_integration_health_check(
    states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run integration health check on all HA entities.

    Args:
        states: Optional list of HA state dicts. Fetched from HA if None.

    Returns:
        Health report with stale_entities, unavailable_entities, integration_scores.
    """
    if states is None:
        states = _fetch_states()

    now = datetime.datetime.now(datetime.UTC)

    all_results = [check_entity_health(s, now) for s in states]

    stale_entities = [r for r in all_results if r["status"] == "stale"]
    unavailable_entities = [r for r in all_results if r["status"] == "unavailable"]
    healthy_entities = [r for r in all_results if r["status"] == "healthy"]

    # Compute per-domain health scores
    domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "healthy": 0})
    for result in all_results:
        domain = result["domain"]
        domain_counts[domain]["total"] += 1
        if result["status"] == "healthy":
            domain_counts[domain]["healthy"] += 1

    integration_scores: dict[str, dict[str, Any]] = {}
    for domain, counts in domain_counts.items():
        total = counts["total"]
        healthy = counts["healthy"]
        score = round(healthy / total * 100, 1) if total > 0 else 100.0
        integration_scores[domain] = {
            "domain": domain,
            "total": total,
            "healthy": healthy,
            "score": score,
            "label": _score_label(score),
        }

    # Overall score
    total_entities = len(all_results)
    total_healthy = len(healthy_entities)
    overall_score = round(total_healthy / total_entities * 100, 1) if total_entities > 0 else 100.0

    report = {
        "timestamp": now.isoformat(),
        "total_entities": total_entities,
        "overall_score": overall_score,
        "stale_count": len(stale_entities),
        "unavailable_count": len(unavailable_entities),
        "healthy_count": total_healthy,
        "stale_entities": sorted(stale_entities, key=lambda e: e.get("age_hours") or 0, reverse=True)[:50],
        "unavailable_entities": unavailable_entities[:50],
        "integration_scores": integration_scores,
    }
    return report


def _score_label(score: float) -> str:
    """Convert score to label."""
    if score >= 90:
        return "good"
    elif score >= 70:
        return "fair"
    return "poor"


def save_integration_health(report: dict[str, Any]) -> None:
    """Save integration health report."""
    os.makedirs(os.environ.get("DATA_DIR", "/data"), exist_ok=True)
    with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "integration_health.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(
        "Integration health: score=%.1f%%, stale=%d, unavailable=%d",
        report["overall_score"],
        report["stale_count"],
        report["unavailable_count"],
    )


def load_integration_health() -> dict[str, Any]:
    """Load cached integration health report."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "integration_health.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "integration_health.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "total_entities": 0,
        "overall_score": 0.0,
        "stale_entities": [],
        "unavailable_entities": [],
        "integration_scores": {},
    }
