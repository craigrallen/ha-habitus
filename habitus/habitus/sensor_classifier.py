"""Sensor type classification for Habitus entities.

Classifies every HA entity into one of five sensor types based on the
``state_class`` HA attribute and/or history pattern analysis.  Downstream
scoring modules consume ``sensor_type`` to apply type-appropriate algorithms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger("habitus")

# Sensor type constants — used as literal values in JSON artefacts
ACCUMULATING = "accumulating"
BINARY = "binary"
GAUGE = "gauge"
EVENT = "event"
SETPOINT = "setpoint"

ALL_SENSOR_TYPES: frozenset[str] = frozenset({ACCUMULATING, BINARY, GAUGE, EVENT, SETPOINT})


def classify_sensor(
    entity_id: str,
    state_class: str | None = None,
    history: Sequence[float] | None = None,
) -> str:
    """Classify an HA entity into a sensor type.

    Classification priority:

    1. HA ``state_class == "total_increasing"`` → ``accumulating``
    2. All history values exclusively in ``{0, 1}`` → ``binary``
    3. History monotonically increasing (≥80% non-decreasing, net upward) → ``accumulating``
    4. ≥70% of values near the floor with occasional higher spikes → ``event``
    5. 2–5 distinct values (rounded to 1 dp) → ``setpoint``
    6. Default → ``gauge``

    Args:
        entity_id: HA entity ID (used only for debug logging).
        state_class: HA ``state_class`` attribute (e.g. ``"total_increasing"``).
        history: Ordered sequence of numeric sensor values for pattern analysis.

    Returns:
        One of ``"accumulating"``, ``"binary"``, ``"gauge"``, ``"event"``,
        ``"setpoint"``.
    """
    # Rule 1: HA state_class attribute takes priority
    if state_class == "total_increasing":
        log.debug("classify_sensor: %s → accumulating (state_class)", entity_id)
        return ACCUMULATING

    if not history:
        log.debug("classify_sensor: %s → gauge (no history)", entity_id)
        return GAUGE

    vals = [float(v) for v in history if v is not None]
    if len(vals) < 2:
        return GAUGE

    # Rule 2: Binary — values are exclusively 0 or 1
    unique_raw = set(vals)
    if unique_raw <= {0.0, 1.0}:
        log.debug("classify_sensor: %s → binary", entity_id)
        return BINARY

    # Rule 3: Accumulating — monotonically increasing history
    if _is_monotonically_increasing(vals):
        log.debug("classify_sensor: %s → accumulating (history)", entity_id)
        return ACCUMULATING

    min_val = min(vals)
    max_val = max(vals)
    value_range = max_val - min_val

    # Rule 4: Event — brief spikes returning to a near-zero floor
    # Condition: ≥70% of samples within 1% of the floor AND max well above floor
    if value_range > 0:
        floor_tolerance = 0.01 * value_range + 0.01
        near_floor = sum(1 for v in vals if abs(v - min_val) <= floor_tolerance)
        floor_fraction = near_floor / len(vals)
        if floor_fraction >= 0.7 and max_val > min_val + 0.5:
            log.debug("classify_sensor: %s → event", entity_id)
            return EVENT

    # Rule 5: Setpoint — small number of distinct fixed values
    unique_rounded = {round(v, 1) for v in vals}
    if 2 <= len(unique_rounded) <= 5:
        log.debug("classify_sensor: %s → setpoint", entity_id)
        return SETPOINT

    # Default: Gauge — continuous bounded variation
    log.debug("classify_sensor: %s → gauge (default)", entity_id)
    return GAUGE


def classify_entities_from_ha_states(ha_states: list[dict]) -> dict[str, str]:
    """Classify entities using the HA ``state_class`` attribute.

    Iterates over the HA ``/api/states`` response and uses ``state_class``
    where present.  Entities without ``state_class`` default to ``gauge``
    (history-based refinement happens separately via :func:`classify_sensor`).

    Args:
        ha_states: List of HA state objects from ``GET /api/states``.

    Returns:
        Mapping of ``entity_id`` → sensor type string.
    """
    result: dict[str, str] = {}
    for state in ha_states:
        eid = state.get("entity_id", "")
        if not eid:
            continue
        attrs = state.get("attributes", {})
        state_class = attrs.get("state_class")
        result[eid] = classify_sensor(eid, state_class=state_class)
    return result


def _is_monotonically_increasing(vals: Sequence[float]) -> bool:
    """Return True if the sequence is predominantly non-decreasing with a net upward trend.

    Additional guards prevent event sensors (mostly-zero with occasional spikes)
    from being falsely classified as accumulating.

    Args:
        vals: Ordered sequence of float values.

    Returns:
        ``True`` if ≥80% of consecutive steps are non-decreasing, the net
        change from first to last value is positive, the overall range is
        non-trivial, and the sequence does not spend the majority of time
        at its floor value (which would indicate an event sensor, not an
        accumulator).
    """
    if len(vals) < 3:
        return False

    min_v = min(vals)
    max_v = max(vals)
    value_range = max_v - min_v

    # Guard: reject if ≥50% of values cluster near the minimum.
    # Real accumulators keep rising; event sensors reset to their floor repeatedly.
    if value_range > 0:
        floor_tolerance = 0.01 * value_range + 0.01
        near_floor = sum(1 for v in vals if abs(v - min_v) <= floor_tolerance)
        if near_floor / len(vals) >= 0.5:
            return False

    steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    non_decreasing = sum(1 for s in steps if s >= 0)
    fraction = non_decreasing / len(steps)
    net_change = vals[-1] - vals[0]
    return fraction >= 0.8 and net_change > 0 and value_range > 0
