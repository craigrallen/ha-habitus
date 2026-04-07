"""Automation Health — dead and over-triggering automation detection.

For each HA automation, checks last_triggered from HA states API attributes.
Flags automations as:
- dead: never triggered, or not triggered in >30 days, or trigger entity missing
- stale: triggered but >7 days ago
- over_triggering: triggered >50x in last 7 days
- healthy: triggered recently and not over-triggering
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import requests

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

HEALTH_PATH = os.path.join(DATA_DIR, "automation_health.json")
HA_AUTOMATIONS_PATH = os.path.join(DATA_DIR, "ha_automations.json")

HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))

DEAD_DAYS = 30          # not triggered in >30 days → dead
STALE_DAYS = 7          # not triggered in >7 days → stale
OVER_TRIGGER_COUNT = 50  # >50 triggers in 7 days → over-triggering


def _ha_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def _fetch_automation_states() -> dict[str, dict[str, Any]]:
    """Fetch all automation.* states from HA."""
    try:
        r = requests.get(f"{HA_URL}/api/states", headers=_ha_headers(), timeout=10)
        if r.status_code == 200:
            return {
                s["entity_id"]: s
                for s in r.json()
                if s["entity_id"].startswith("automation.")
            }
    except Exception as e:
        log.warning("Failed to fetch HA automation states: %s", e)
    return {}


def _fetch_automation_history(entity_id: str, days: int = 7) -> list[dict[str, Any]]:
    """Fetch state history for an automation over the past N days."""
    try:
        end = datetime.datetime.now(datetime.UTC)
        start = end - datetime.timedelta(days=days)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        url = f"{HA_URL}/api/history/period/{start_str}"
        params = {"filter_entity_id": entity_id, "end_time": end.strftime("%Y-%m-%dT%H:%M:%S+00:00")}
        r = requests.get(url, headers=_ha_headers(), params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data[0], list):
                return data[0]
    except Exception as e:
        log.debug("Failed to fetch history for %s: %s", entity_id, e)
    return []


def _load_ha_automations() -> list[dict[str, Any]]:
    """Load HA automations from cache file."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations.json")) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get("automations", [])
    except Exception:
        pass
    return []


def _count_triggers_7d(history: list[dict[str, Any]]) -> int:
    """Count how many times an automation was triggered in the last 7 days."""
    count = 0
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
    for entry in history:
        state = entry.get("state", "")
        if state.lower() in ("on", "triggered"):
            last_changed = entry.get("last_changed", "")
            try:
                dt = datetime.datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
                if dt > cutoff:
                    count += 1
            except (ValueError, TypeError):
                pass
    return count


def classify_automation(
    automation: dict[str, Any],
    state_data: dict[str, Any] | None,
    history: list[dict[str, Any]],
    all_entity_ids: set[str],
) -> dict[str, Any]:
    """Classify a single automation's health status."""
    alias = automation.get("alias") or automation.get("id", "unknown")
    entity_id = f"automation.{alias.lower().replace(' ', '_').replace('-', '_')}"

    now = datetime.datetime.now(datetime.UTC)
    last_triggered: str | None = None
    days_since_trigger: float | None = None

    if state_data:
        attrs = state_data.get("attributes", {})
        last_triggered = attrs.get("last_triggered")
        if last_triggered:
            try:
                lt_dt = datetime.datetime.fromisoformat(last_triggered.replace("Z", "+00:00"))
                days_since_trigger = (now - lt_dt).total_seconds() / 86400
            except (ValueError, TypeError):
                last_triggered = None

    trigger_count_7d = _count_triggers_7d(history)

    # Check if trigger entities exist
    trigger_entities_exist = True
    triggers = automation.get("trigger", [])
    if isinstance(triggers, dict):
        triggers = [triggers]
    for trig in (triggers or []):
        eid = trig.get("entity_id")
        if eid and isinstance(eid, str) and eid not in all_entity_ids:
            trigger_entities_exist = False
            break

    # Classify
    if not trigger_entities_exist:
        status = "dead"
        recommendation = "Trigger entity no longer exists in HA. Disable or update this automation."
    elif last_triggered is None:
        status = "dead"
        recommendation = "This automation has never been triggered. Check its trigger conditions."
    elif days_since_trigger is not None and days_since_trigger > DEAD_DAYS:
        status = "dead"
        recommendation = f"Not triggered in {days_since_trigger:.0f} days. Consider disabling or reviewing."
    elif days_since_trigger is not None and days_since_trigger > STALE_DAYS:
        status = "stale"
        recommendation = f"Not triggered in {days_since_trigger:.0f} days. Monitor or review trigger conditions."
    elif trigger_count_7d > OVER_TRIGGER_COUNT:
        status = "over_triggering"
        recommendation = f"Triggered {trigger_count_7d}x in the last 7 days. Add conditions or debounce to reduce noise."
    else:
        status = "healthy"
        recommendation = "Automation is working as expected."

    return {
        "alias": alias,
        "entity_id": entity_id,
        "status": status,
        "last_triggered": last_triggered,
        "days_since_trigger": round(days_since_trigger, 1) if days_since_trigger is not None else None,
        "trigger_count_7d": trigger_count_7d,
        "trigger_entities_exist": trigger_entities_exist,
        "recommendation": recommendation,
    }


def run_health_check(
    automations: list[dict[str, Any]] | None = None,
    states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run full automation health check.

    Args:
        automations: Optional list of automation dicts (loaded from HA_AUTOMATIONS_PATH if None).
        states: Optional dict of HA states keyed by entity_id (fetched from HA if None).

    Returns:
        Health report dict with per-automation status and summary.
    """
    if automations is None:
        automations = _load_ha_automations()

    if states is None:
        states = _fetch_automation_states()

    all_entity_ids = set(states.keys())

    results = []
    for auto in automations:
        alias = auto.get("alias") or auto.get("id", "")
        entity_id_key = f"automation.{alias.lower().replace(' ', '_').replace('-', '_')}"

        # Find matching state entry
        state_data = states.get(entity_id_key)
        if state_data is None:
            # Try to find by friendly_name
            for eid, s in states.items():
                fn = s.get("attributes", {}).get("friendly_name", "")
                if fn.lower() == alias.lower():
                    state_data = s
                    break

        history = _fetch_automation_history(entity_id_key) if state_data else []

        result = classify_automation(auto, state_data, history, all_entity_ids)
        results.append(result)

    # Summary
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    report = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total": len(results),
        "summary": status_counts,
        "automations": results,
    }
    return report


def save_health(report: dict[str, Any]) -> None:
    """Save health report to cache file."""
    from .utils import atomic_write as _atomic_write  # noqa: PLC0415
    _atomic_write(os.path.join(os.environ.get("DATA_DIR", "/data"), "automation_health.json"), report)
    log.info(
        "Automation health: %d total — %s",
        report["total"],
        ", ".join(f"{k}={v}" for k, v in report.get("summary", {}).items()),
    )


def load_health() -> dict[str, Any]:
    """Load cached health report."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "automation_health.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "automation_health.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {"total": 0, "summary": {}, "automations": []}
