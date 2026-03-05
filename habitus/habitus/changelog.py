"""Automation Changelog — track changes to HA automations over time.

Diffs ha_automations.json snapshots to detect added/removed/modified automations.
Also logs Habitus add-to-HA events and automation last_triggered updates.
Stores in DATA_DIR/changelog.json (append-only, max 500 entries).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

CHANGELOG_PATH = os.path.join(DATA_DIR, "changelog.json")
HA_AUTOMATIONS_PATH = os.path.join(DATA_DIR, "ha_automations.json")
HA_AUTOMATIONS_PREV_PATH = os.path.join(DATA_DIR, "ha_automations_prev.json")

MAX_ENTRIES = 500

# Change types
CHANGE_ADDED = "added"
CHANGE_REMOVED = "removed"
CHANGE_MODIFIED = "modified"
CHANGE_TRIGGERED = "triggered"
CHANGE_HABITUS_ADD = "habitus_add"
CHANGE_HABITUS_REMOVE = "habitus_remove"

# Icons for change types
CHANGE_ICONS = {
    CHANGE_ADDED: "➕",
    CHANGE_REMOVED: "❌",
    CHANGE_MODIFIED: "✏️",
    CHANGE_TRIGGERED: "▶️",
    CHANGE_HABITUS_ADD: "🤖",
    CHANGE_HABITUS_REMOVE: "🗑️",
}


def _load_changelog() -> list[dict[str, Any]]:
    """Load existing changelog entries."""
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "changelog.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "changelog.json")) as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        pass
    return []


def _save_changelog(entries: list[dict[str, Any]]) -> None:
    """Save changelog entries (trimmed to MAX_ENTRIES)."""
    # Keep most recent entries
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]
    from .utils import atomic_write as _atomic_write  # noqa: PLC0415
    _atomic_write(os.path.join(os.environ.get("DATA_DIR", "/data"), "changelog.json"), entries)


def _automation_key(automation: dict[str, Any]) -> str:
    """Get a stable key for an automation."""
    return automation.get("id") or automation.get("alias", "") or str(id(automation))


def _automation_signature(automation: dict[str, Any]) -> str:
    """Get a signature for change detection."""
    import hashlib
    sig_data = {
        "alias": automation.get("alias", ""),
        "trigger": automation.get("trigger"),
        "condition": automation.get("condition"),
        "action": automation.get("action"),
    }
    return hashlib.md5(json.dumps(sig_data, sort_keys=True, default=str).encode()).hexdigest()[:8]


def diff_automations(
    prev: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Diff two automation lists and return change entries.

    Args:
        prev: Previous automation list.
        current: Current automation list.

    Returns:
        List of change entry dicts.
    """
    prev_by_key = {_automation_key(a): a for a in prev}
    current_by_key = {_automation_key(a): a for a in current}

    changes = []
    now = datetime.datetime.now(datetime.UTC).isoformat()

    # Detect added
    for key, auto in current_by_key.items():
        if key not in prev_by_key:
            changes.append({
                "type": CHANGE_ADDED,
                "icon": CHANGE_ICONS[CHANGE_ADDED],
                "alias": auto.get("alias", key),
                "automation_id": key,
                "description": f"Automation '{auto.get('alias', key)}' was added",
                "timestamp": now,
            })

    # Detect removed
    for key, auto in prev_by_key.items():
        if key not in current_by_key:
            changes.append({
                "type": CHANGE_REMOVED,
                "icon": CHANGE_ICONS[CHANGE_REMOVED],
                "alias": auto.get("alias", key),
                "automation_id": key,
                "description": f"Automation '{auto.get('alias', key)}' was removed",
                "timestamp": now,
            })

    # Detect modified
    for key in set(prev_by_key) & set(current_by_key):
        prev_sig = _automation_signature(prev_by_key[key])
        curr_sig = _automation_signature(current_by_key[key])
        if prev_sig != curr_sig:
            auto = current_by_key[key]
            changes.append({
                "type": CHANGE_MODIFIED,
                "icon": CHANGE_ICONS[CHANGE_MODIFIED],
                "alias": auto.get("alias", key),
                "automation_id": key,
                "description": f"Automation '{auto.get('alias', key)}' was modified",
                "timestamp": now,
            })

    return changes


def append_entry(entry: dict[str, Any]) -> None:
    """Append a single entry to the changelog."""
    entries = _load_changelog()
    entry.setdefault("timestamp", datetime.datetime.now(datetime.UTC).isoformat())
    entry.setdefault("icon", CHANGE_ICONS.get(entry.get("type", ""), "📝"))
    entries.append(entry)
    _save_changelog(entries)
    log.debug("Changelog: %s — %s", entry.get("type"), entry.get("description", ""))


def append_entries(new_entries: list[dict[str, Any]]) -> None:
    """Append multiple entries to the changelog."""
    if not new_entries:
        return
    entries = _load_changelog()
    now = datetime.datetime.now(datetime.UTC).isoformat()
    for entry in new_entries:
        entry.setdefault("timestamp", now)
        entry.setdefault("icon", CHANGE_ICONS.get(entry.get("type", ""), "📝"))
        entries.append(entry)
    _save_changelog(entries)
    log.info("Changelog: appended %d entries", len(new_entries))


def log_habitus_add(alias: str, automation_id: str | None = None) -> None:
    """Log a Habitus-initiated automation add."""
    append_entry({
        "type": CHANGE_HABITUS_ADD,
        "icon": CHANGE_ICONS[CHANGE_HABITUS_ADD],
        "alias": alias,
        "automation_id": automation_id or alias,
        "description": f"Habitus added automation '{alias}' to HA",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    })


def log_habitus_remove(alias: str, automation_id: str | None = None) -> None:
    """Log a Habitus-initiated automation removal."""
    append_entry({
        "type": CHANGE_HABITUS_REMOVE,
        "icon": CHANGE_ICONS[CHANGE_HABITUS_REMOVE],
        "alias": alias,
        "automation_id": automation_id or alias,
        "description": f"Habitus removed automation '{alias}' from HA",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    })


def log_triggered(alias: str, automation_id: str | None = None, last_triggered: str | None = None) -> None:
    """Log an automation trigger event."""
    append_entry({
        "type": CHANGE_TRIGGERED,
        "icon": CHANGE_ICONS[CHANGE_TRIGGERED],
        "alias": alias,
        "automation_id": automation_id or alias,
        "description": f"Automation '{alias}' was triggered",
        "last_triggered": last_triggered,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    })


def run_diff_and_log() -> list[dict[str, Any]]:
    """Run diff between previous and current automations and log changes.

    Returns:
        List of new change entries.
    """
    current: list[dict[str, Any]] = []
    prev: list[dict[str, Any]] = []

    # Load current automations
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations.json")) as f:
                data = json.load(f)
                current = data if isinstance(data, list) else data.get("automations", [])
    except Exception as e:
        log.warning("Failed to load current automations: %s", e)

    # Load previous snapshot
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations_prev.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations_prev.json")) as f:
                data = json.load(f)
                prev = data if isinstance(data, list) else data.get("automations", [])
    except Exception as e:
        log.warning("Failed to load previous automations: %s", e)

    changes = diff_automations(prev, current)

    if changes:
        append_entries(changes)

    # Save current as new previous snapshot
    from .utils import atomic_write as _atomic_write  # noqa: PLC0415
    _atomic_write(os.path.join(os.environ.get("DATA_DIR", "/data"), "ha_automations_prev.json"), current)

    return changes


def load_changelog(limit: int | None = None) -> list[dict[str, Any]]:
    """Load changelog entries.

    Args:
        limit: Maximum number of entries to return (most recent first).

    Returns:
        List of changelog entries.
    """
    entries = _load_changelog()
    entries_sorted = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)
    if limit:
        return entries_sorted[:limit]
    return entries_sorted
