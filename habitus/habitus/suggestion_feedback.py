"""Suggestion usefulness feedback loop.

Records when a user adds, dismisses, or removes a suggested automation.
On the next training run, feedback is loaded to:
- Boost confidence of suggestions with action='add' (user found it useful)
- Suppress suggestions with action='dismiss' (user rejected it)

Storage: DATA_DIR/feedback.json
Schema:
    {
        "entries": [
            {"suggestion_id": "...", "action": "add|dismiss|remove", "timestamp": "..."},
            ...
        ]
    }
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
SUGGESTION_FEEDBACK_PATH = os.path.join(DATA_DIR, "suggestion_feedback.json")

VALID_ACTIONS = {"add", "dismiss", "remove"}
MAX_ENTRIES = 1000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load() -> dict[str, Any]:
    try:
        if os.path.exists(SUGGESTION_FEEDBACK_PATH):
            with open(SUGGESTION_FEEDBACK_PATH) as f:
                return json.load(f)
    except Exception as e:
        log.warning("suggestion_feedback load failed: %s", e)
    return {"entries": []}


def _save(data: dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SUGGESTION_FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_feedback(suggestion_id: str, action: str) -> dict[str, Any]:
    """Record a user action on a suggestion.

    Args:
        suggestion_id: Stable suggestion identifier (e.g. automation alias/id).
        action: One of 'add', 'dismiss', 'remove'.

    Returns:
        The recorded feedback entry.

    Raises:
        ValueError: If action is not one of the valid actions.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action '{action}'. Must be one of {sorted(VALID_ACTIONS)}")

    entry: dict[str, Any] = {
        "suggestion_id": suggestion_id,
        "action": action,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    data = _load()
    data["entries"].append(entry)

    # Trim to keep last MAX_ENTRIES
    if len(data["entries"]) > MAX_ENTRIES:
        data["entries"] = data["entries"][-MAX_ENTRIES:]

    _save(data)
    log.info("suggestion_feedback: %s → %s", suggestion_id, action)
    return entry


def get_feedback_summary() -> dict[str, Any]:
    """Return per-suggestion feedback summary for use during training.

    Returns:
        {
            "boosts": {suggestion_id: count},    # action='add' — increase confidence
            "suppressions": {suggestion_id: count},  # action='dismiss' — decrease confidence
            "removals": {suggestion_id: count},  # action='remove'
            "total_entries": int,
        }
    """
    data = _load()
    entries = data.get("entries", [])

    boosts: dict[str, int] = {}
    suppressions: dict[str, int] = {}
    removals: dict[str, int] = {}

    for entry in entries:
        sid = entry.get("suggestion_id", "")
        act = entry.get("action", "")
        if act == "add":
            boosts[sid] = boosts.get(sid, 0) + 1
        elif act == "dismiss":
            suppressions[sid] = suppressions.get(sid, 0) + 1
        elif act == "remove":
            removals[sid] = removals.get(sid, 0) + 1

    return {
        "boosts": boosts,
        "suppressions": suppressions,
        "removals": removals,
        "total_entries": len(entries),
    }


def apply_feedback_to_suggestions(
    suggestions: list[dict[str, Any]],
    boost_factor: float = 5.0,
    suppress_factor: float = 10.0,
    suppress_threshold: int = 2,
) -> list[dict[str, Any]]:
    """Adjust suggestion confidence scores based on recorded feedback.

    Suggestions with action='add' get a confidence boost.
    Suggestions with action='dismiss' (≥ suppress_threshold times) get suppressed.

    Args:
        suggestions: List of suggestion dicts (each must have 'id' or 'automation_id').
        boost_factor: Confidence points to add per 'add' action.
        suppress_factor: Confidence points to subtract per 'dismiss' action.
        suppress_threshold: Minimum dismiss count before suppression kicks in.

    Returns:
        Modified list (dicts mutated in-place).
    """
    summary = get_feedback_summary()
    boosts = summary["boosts"]
    suppressions = summary["suppressions"]

    for sug in suggestions:
        sid = sug.get("id") or sug.get("automation_id") or ""
        if not sid:
            continue

        boost_count = boosts.get(sid, 0)
        suppress_count = suppressions.get(sid, 0)

        confidence = float(sug.get("confidence", 50))

        if boost_count > 0:
            confidence = min(99, confidence + boost_factor * boost_count)
            sug["feedback_boosted"] = True

        if suppress_count >= suppress_threshold:
            confidence = max(0, confidence - suppress_factor * suppress_count)
            sug["feedback_suppressed"] = True

        sug["confidence"] = round(confidence)

    # Re-sort by confidence after feedback adjustment
    suggestions.sort(key=lambda s: -s.get("confidence", 0))
    return suggestions


def get_dismissed_ids() -> set[str]:
    """Return set of suggestion IDs that have been dismissed ≥ 2 times."""
    summary = get_feedback_summary()
    return {
        sid
        for sid, count in summary["suppressions"].items()
        if count >= 2
    }
