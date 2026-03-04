from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from habitus.habitus.scene_detector import _find_co_occurrences


def _baseline_find_co_occurrences(changes: list[dict], window_s: int) -> dict[tuple[str, str], list[float]]:
    """Reference implementation mirroring pre-optimization behavior."""
    on_states = {"on", "playing", "open", "home", "heat", "cool", "auto", "above_horizon"}
    activations = [c for c in changes if c["state"].lower() in on_states]

    pair_occurrences: dict[tuple[str, str], list[float]] = defaultdict(list)
    n = len(activations)
    for i in range(n):
        group_entities = {activations[i]["entity_id"]}
        group_ts = activations[i]["timestamp"]
        j = i + 1
        while j < n and (activations[j]["timestamp"] - group_ts) <= window_s:
            if activations[j]["entity_id"] != activations[i]["entity_id"]:
                group_entities.add(activations[j]["entity_id"])
            j += 1

        if len(group_entities) >= 2:
            for a, b in combinations(sorted(group_entities), 2):
                pair_occurrences[(a, b)].append(group_ts)

    return pair_occurrences


def test_find_co_occurrences_matches_reference_behavior() -> None:
    changes = [
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 0.0},
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 10.0},
        {"entity_id": "switch.tv", "state": "on", "timestamp": 20.0},
        {"entity_id": "climate.lounge", "state": "heat", "timestamp": 120.0},
        {"entity_id": "switch.tv", "state": "off", "timestamp": 170.0},  # ignored
        {"entity_id": "media_player.lounge", "state": "playing", "timestamp": 200.0},
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 250.0},
        {"entity_id": "binary_sensor.motion", "state": "on", "timestamp": 251.0},
    ]

    expected = _baseline_find_co_occurrences(changes, window_s=120)
    actual = _find_co_occurrences(changes, window_s=120)

    assert actual == expected


def test_find_co_occurrences_ignores_windows_without_distinct_entities() -> None:
    changes = [
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 0.0},
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 10.0},
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 20.0},
    ]

    assert _find_co_occurrences(changes, window_s=60) == {}


def test_find_co_occurrences_handles_empty_and_non_activation_input() -> None:
    assert _find_co_occurrences([], window_s=60) == {}
    assert _find_co_occurrences(
        [{"entity_id": "switch.tv", "state": "off", "timestamp": 1.0}],
        window_s=60,
    ) == {}
