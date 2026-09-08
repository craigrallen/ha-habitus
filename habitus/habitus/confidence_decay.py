"""Confidence decay — reduce suggestion confidence when underlying patterns weaken."""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
CONFIDENCE_DECAY_PATH = os.path.join(DATA_DIR, "confidence_decay.json")

# Drift threshold in minutes — patterns that drifted more than this are weakened
DRIFT_THRESHOLD_MIN: float = 30.0

# Maximum confidence reduction per decay cycle
MAX_DECAY_PCT: float = 0.40

# Maximum confidence boost per cycle
MAX_BOOST_PCT: float = 0.10

# Minimum confidence floor — never decay below this
MIN_CONFIDENCE: float = 0.05


def compute_pattern_strength(
    pattern_key: str,
    patterns: dict[str, Any],
    drift_data: dict[str, Any],
) -> float:
    """Compute how strongly a pattern still holds, from 0.0 to 1.0.

    Combines pattern frequency/consistency with drift information to produce
    a single strength score.

    Args:
        pattern_key: Key identifying the pattern (e.g., "morning_routine",
            "bedtime", or an entity-based pattern name).
        patterns: Discovered patterns dict (from ``patterns.json``).
        drift_data: Drift analysis dict (from ``drift.json``).

    Returns:
        Float between 0.0 (pattern completely gone) and 1.0 (pattern solid).
    """
    strength = 1.0

    # Check if pattern exists in the patterns data
    pattern_info = _find_pattern(pattern_key, patterns)
    if pattern_info is None:
        return 0.0

    # Factor 1: Pattern consistency / frequency
    consistency = pattern_info.get("consistency", pattern_info.get("frequency", 0.5))
    try:
        consistency = float(consistency)
    except (ValueError, TypeError):
        consistency = 0.5
    consistency = max(0.0, min(1.0, consistency))
    strength *= consistency

    # Factor 2: Drift impact
    drift_penalty = _drift_penalty(pattern_key, drift_data)
    strength *= 1.0 - drift_penalty

    # Factor 3: Recency — if pattern has age info, newer is stronger
    days_old = pattern_info.get("days_since_last_seen", 0)
    try:
        days_old = float(days_old)
    except (ValueError, TypeError):
        days_old = 0.0
    if days_old > 14:
        recency_factor = max(0.3, 1.0 - (days_old - 14) / 60.0)
        strength *= recency_factor

    return round(max(0.0, min(1.0, strength)), 3)


def _find_pattern(pattern_key: str, patterns: dict[str, Any]) -> dict[str, Any] | None:
    """Look up a pattern by key in the patterns data structure.

    Args:
        pattern_key: Pattern identifier to search for.
        patterns: Full patterns dict.

    Returns:
        Pattern info dict if found, None otherwise.
    """
    # Direct key lookup
    if pattern_key in patterns:
        val = patterns[pattern_key]
        return val if isinstance(val, dict) else {"value": val}

    # Search in nested "patterns" or "routines" lists
    for container_key in ("patterns", "routines", "sequences"):
        container = patterns.get(container_key, [])
        if isinstance(container, list):
            for item in container:
                if isinstance(item, dict):
                    item_key = item.get("key") or item.get("name") or item.get("id", "")
                    if item_key == pattern_key:
                        return item
        elif isinstance(container, dict) and pattern_key in container:
            val = container[pattern_key]
            return val if isinstance(val, dict) else {"value": val}

    return None


def _drift_penalty(pattern_key: str, drift_data: dict[str, Any]) -> float:
    """Calculate a penalty factor (0.0 to 1.0) based on drift data.

    Args:
        pattern_key: Pattern identifier.
        drift_data: Drift analysis dict.

    Returns:
        Penalty between 0.0 (no drift) and 1.0 (severe drift).
    """
    drifts = drift_data.get("drifts", [])
    if not drifts:
        return 0.0

    # Map pattern keys to drift metric names
    key_lower = pattern_key.lower()
    relevant_drifts: list[dict[str, Any]] = []

    for drift in drifts:
        if not isinstance(drift, dict):
            continue
        metric = str(drift.get("metric", "")).lower()
        summary = str(drift.get("summary", "")).lower()

        # Match pattern keys to drift metrics by keyword overlap
        matched = (
            any(kw in key_lower and kw in metric for kw in ("morning", "bedtime", "peak", "light"))
            or key_lower in summary
        )

        if matched:
            relevant_drifts.append(drift)

    if not relevant_drifts:
        # No specific drift data — check if any drift is significant
        significant = [d for d in drifts if d.get("significant")]
        if significant:
            return 0.1  # Small general penalty
        return 0.0

    # Calculate penalty from the most severe relevant drift
    max_penalty = 0.0
    for drift in relevant_drifts:
        diff_min = abs(float(drift.get("diff_min", 0)))
        diff_hours = abs(float(drift.get("diff", 0)))

        if diff_min > 0:
            # Scale: 30 min drift = 0.3, 60 min = 0.6, 90+ min = capped at 0.8
            penalty = min(diff_min / 100.0, 0.8)
        elif diff_hours > 0:
            penalty = min(diff_hours / 2.0, 0.8)
        else:
            penalty = 0.1 if drift.get("significant") else 0.0

        max_penalty = max(max_penalty, penalty)

    return max_penalty


def decay_confidences(
    suggestions: list[dict[str, Any]],
    patterns: dict[str, Any],
    drift_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decay or boost suggestion confidences based on pattern strength.

    For each suggestion, checks whether its source pattern still holds.
    Weakened patterns cause a 10-40% confidence reduction; strengthened
    patterns get up to a 10% boost.

    Args:
        suggestions: List of suggestion dicts. Each should have a
            ``confidence`` key and either ``pattern_key``, ``source_pattern``,
            or ``trigger`` to identify the underlying pattern.
        patterns: Discovered patterns dict.
        drift_data: Drift analysis dict.

    Returns:
        Updated list of suggestions with ``original_confidence`` and
        ``decayed_confidence`` fields added.
    """
    updated: list[dict[str, Any]] = []

    for suggestion in suggestions:
        entry = dict(suggestion)
        original = float(entry.get("confidence", 0.5))
        entry["original_confidence"] = round(original, 3)

        # Identify the source pattern
        pattern_key = (
            entry.get("pattern_key") or entry.get("source_pattern") or entry.get("trigger", "")
        )

        if not pattern_key:
            # No pattern link — keep original confidence
            entry["decayed_confidence"] = round(original, 3)
            entry["decay_reason"] = "no pattern link"
            updated.append(entry)
            continue

        strength = compute_pattern_strength(pattern_key, patterns, drift_data)

        if strength >= 0.8:
            # Pattern is strong — apply a small boost
            boost = min(original * MAX_BOOST_PCT, MAX_BOOST_PCT)
            new_confidence = min(original + boost, 1.0)
            reason = f"pattern strong ({strength:.2f}), boosted"
        elif strength >= 0.5:
            # Pattern is moderate — small decay
            decay_factor = 0.1 + 0.1 * (1.0 - strength)
            new_confidence = original * (1.0 - decay_factor)
            reason = f"pattern moderate ({strength:.2f}), minor decay"
        elif strength >= 0.2:
            # Pattern is weak — significant decay
            decay_factor = 0.2 + 0.2 * (1.0 - strength)
            decay_factor = min(decay_factor, MAX_DECAY_PCT)
            new_confidence = original * (1.0 - decay_factor)
            reason = f"pattern weak ({strength:.2f}), decayed"
        else:
            # Pattern is very weak or gone — maximum decay
            new_confidence = original * (1.0 - MAX_DECAY_PCT)
            reason = f"pattern very weak ({strength:.2f}), max decay"

        new_confidence = max(new_confidence, MIN_CONFIDENCE)
        entry["decayed_confidence"] = round(new_confidence, 3)
        entry["pattern_strength"] = strength
        entry["decay_reason"] = reason

        updated.append(entry)

    return updated


def save(decay_report: list[dict[str, Any]]) -> None:
    """Persist confidence decay report to disk.

    Args:
        decay_report: List of updated suggestion dicts with decay info.
    """
    payload = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "suggestions": decay_report,
        "count": len(decay_report),
        "decayed_count": sum(
            1
            for s in decay_report
            if s.get("decayed_confidence", 1) < s.get("original_confidence", 0)
        ),
        "boosted_count": sum(
            1
            for s in decay_report
            if s.get("decayed_confidence", 0) > s.get("original_confidence", 1)
        ),
    }
    atomic_write(CONFIDENCE_DECAY_PATH, payload)
    log.info(
        "Confidence decay saved: %d suggestion(s), %d decayed, %d boosted",
        payload["count"],
        payload["decayed_count"],
        payload["boosted_count"],
    )


def load() -> dict[str, Any]:
    """Load confidence decay report from disk.

    Returns:
        Previously saved decay report dict, or empty dict if none found.
    """
    if not os.path.exists(CONFIDENCE_DECAY_PATH):
        return {}
    try:
        with open(CONFIDENCE_DECAY_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
