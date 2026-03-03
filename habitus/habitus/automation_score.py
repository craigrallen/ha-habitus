"""Automation Effectiveness Scoring — score HA automations by override rate."""

import datetime
import json
import logging
import os
from typing import Any

import aiohttp

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SCORES_PATH = os.path.join(DATA_DIR, "automation_scores.json")

# Only score automations with enough triggers
MIN_TRIGGERS = 10
# Time window to look for manual override after automation trigger (seconds)
OVERRIDE_WINDOW_S = 300  # 5 minutes


async def score_all(ha_url: str | None = None, ha_token: str | None = None) -> list[Any]:
    """Score all HA automations by override rate.

    Args:
        ha_url: Home Assistant URL.
        ha_token: Long-lived access token.

    Returns:
        List of automation score dicts.
    """
    ha_url = ha_url or os.environ.get("HA_URL", "http://supervisor/core")
    ha_token = ha_token or os.environ.get("SUPERVISOR_TOKEN", "")
    headers = {"Authorization": f"Bearer {ha_token}"}

    try:
        async with aiohttp.ClientSession() as session:
            # Get all automation entities
            async with session.get(
                f"{ha_url}/api/states", headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    log.warning("Could not fetch states: %d", resp.status)
                    return []
                states = await resp.json()

            automations = [s for s in states if s["entity_id"].startswith("automation.")]
            if not automations:
                return []

            log.info("Scoring %d automations", len(automations))
            results = []

            # Get logbook for the past 7 days
            end = datetime.datetime.now(datetime.UTC)
            start = end - datetime.timedelta(days=7)
            start_iso = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")

            for auto in automations:
                auto_id = auto["entity_id"]
                friendly = auto.get("attributes", {}).get(
                    "friendly_name", auto_id.split(".")[-1].replace("_", " ").title()
                )

                try:
                    # Get automation trigger events from logbook
                    async with session.get(
                        f"{ha_url}/api/logbook/{start_iso}",
                        headers=headers,
                        params={
                            "entity": auto_id,
                            "end_time": end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        },
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        log_entries = await resp.json()

                    # Count trigger events
                    triggers = [
                        e
                        for e in log_entries
                        if e.get("state") == "on"
                        or "triggered" in str(e.get("message", "")).lower()
                    ]

                    if len(triggers) < MIN_TRIGGERS:
                        continue

                    # Get target entities from automation attributes
                    target_entities = _extract_target_entities(auto)
                    if not target_entities:
                        # Can't determine override rate without targets
                        results.append(
                            {
                                "entity_id": auto_id,
                                "name": friendly,
                                "triggers_7d": len(triggers),
                                "overrides": 0,
                                "override_rate": 0,
                                "score": 100,
                                "note": "No target entities detected — assumed effective",
                            }
                        )
                        continue

                    # Check for manual overrides after each trigger
                    overrides = 0
                    for trigger in triggers:
                        trigger_time = trigger.get("when")
                        if not trigger_time:
                            continue
                        # Look for manual changes to target entities within OVERRIDE_WINDOW_S
                        for target_eid in target_entities:
                            try:
                                async with session.get(
                                    f"{ha_url}/api/logbook/{trigger_time}",
                                    headers=headers,
                                    params={
                                        "entity": target_eid,
                                        "end_time": _add_seconds(trigger_time, OVERRIDE_WINDOW_S),
                                    },
                                    timeout=aiohttp.ClientTimeout(total=5),
                                ) as resp2:
                                    if resp2.status != 200:
                                        continue
                                    target_logs = await resp2.json()

                                manual = [e for e in target_logs if _is_manual_change(e, auto_id)]
                                if manual:
                                    overrides += 1
                                    break  # One override per trigger is enough
                            except Exception:
                                continue

                    override_rate = round(overrides / len(triggers) * 100) if triggers else 0
                    score = max(0, 100 - override_rate)

                    results.append(
                        {
                            "entity_id": auto_id,
                            "name": friendly,
                            "triggers_7d": len(triggers),
                            "overrides": overrides,
                            "override_rate": override_rate,
                            "score": score,
                        }
                    )

                except Exception as e:
                    log.warning("Error scoring %s: %s", auto_id, e)
                    continue

            results.sort(key=lambda x: x["score"])
            return results

    except Exception as e:
        log.warning("Automation scoring failed: %s", e)
        return []


def _extract_target_entities(auto_state: dict) -> list:
    """Extract target entity IDs from automation attributes."""
    attrs = auto_state.get("attributes", {})
    targets = set()

    # Check last_triggered context or entity_id references in attributes
    for key in ("entity_id", "target"):
        val = attrs.get(key)
        if isinstance(val, str) and "." in val:
            targets.add(val)
        elif isinstance(val, list):
            targets.update(v for v in val if isinstance(v, str) and "." in v)

    return list(targets)


def _is_manual_change(entry: dict, automation_id: str) -> bool:
    """Check if a logbook entry is a manual (non-automation) change."""
    context = entry.get("context_entity_id", "")
    if context == automation_id:
        return False  # This change was made by the automation itself
    # If context is empty or from another source, it's likely manual
    source = entry.get("context_user_id", "") or entry.get("context_event_type", "")
    return bool(source) or not context


def _add_seconds(iso_str: str, seconds: int) -> str:
    """Add seconds to an ISO datetime string."""
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        result = dt + datetime.timedelta(seconds=seconds)
        return result.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except Exception:
        return iso_str


def save(results: list) -> None:
    """Save automation scores to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCORES_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Automation scores saved: %d automations scored", len(results))


def load() -> list[Any]:
    """Load automation scores from disk."""
    if not os.path.exists(SCORES_PATH):
        return []
    try:
        with open(SCORES_PATH) as f:
            data: list[Any] = json.load(f)
            return data
    except Exception:
        return []
