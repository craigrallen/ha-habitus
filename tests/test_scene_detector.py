from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from habitus.habitus import scene_detector
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


def test_find_co_occurrences_strict_sorted_guard_raises_for_unsorted_input() -> None:
    changes = [
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 10.0},
        {"entity_id": "switch.tv", "state": "on", "timestamp": 9.0},
    ]

    try:
        _find_co_occurrences(changes, window_s=60, strict_sorted=True)
        assert False, "Expected ValueError for unsorted timestamps"
    except ValueError as exc:
        assert "sorted" in str(exc)


def test_find_co_occurrences_strict_sorted_guard_accepts_sorted_input() -> None:
    changes = [
        {"entity_id": "light.kitchen", "state": "on", "timestamp": 9.0},
        {"entity_id": "switch.tv", "state": "on", "timestamp": 10.0},
    ]

    result = _find_co_occurrences(changes, window_s=60, strict_sorted=True)
    assert result == {("light.kitchen", "switch.tv"): [9.0]}


def _baseline_cluster_pairs_to_scenes(
    pair_occurrences: dict[tuple[str, str], list[float]],
    min_co: int,
) -> list[set[str]]:
    frequent_pairs = {pair: times for pair, times in pair_occurrences.items() if len(times) >= min_co}
    if not frequent_pairs:
        return []

    adjacency: dict[str, set[str]] = defaultdict(set)
    pair_counts: dict[tuple[str, str], int] = {}
    for (a, b), times in frequent_pairs.items():
        adjacency[a].add(b)
        adjacency[b].add(a)
        pair_counts[(a, b)] = len(times)

    visited: set[str] = set()
    scenes: list[set[str]] = []

    for entity in adjacency:
        if entity in visited:
            continue

        component: set[str] = set()
        queue = [entity]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(component) >= 2:
            if len(component) <= 12:
                scenes.append(component)
            else:
                scene_detector._split_large_component(component, pair_counts, scenes)

    return scenes


def test_cluster_pairs_matches_reference_behavior() -> None:
    entities = [f"light.room_{idx}" for idx in range(16)]
    pair_occurrences: dict[tuple[str, str], list[float]] = {}
    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            pair_occurrences[(a, b)] = [1.0, 2.0, 3.0, 4.0, 5.0]

    expected = _baseline_cluster_pairs_to_scenes(pair_occurrences, min_co=5)
    actual = scene_detector._cluster_pairs_to_scenes(pair_occurrences, min_co=5)

    assert {frozenset(scene) for scene in actual} == {frozenset(scene) for scene in expected}


def test_detect_scenes_confidence_counts_wrapped_midnight_window(monkeypatch) -> None:
    monkeypatch.setattr(
        scene_detector,
        "_get_state_changes",
        lambda days: [{"entity_id": "light.kitchen", "state": "on", "timestamp": 1.0}],
    )
    monkeypatch.setattr(
        scene_detector,
        "_find_co_occurrences",
        lambda changes: {("light.kitchen", "switch.tv"): [1.0] * 10},
    )
    monkeypatch.setattr(
        scene_detector,
        "_cluster_pairs_to_scenes",
        lambda pairs: [{"light.kitchen", "switch.tv"}],
    )
    monkeypatch.setattr(
        scene_detector,
        "_analyze_time_patterns",
        lambda entities, changes: {
            "count": 10,
            "peak_hour": 23,
            "best_window_start": 23,
            "best_window_end": 1,
            "time_label": "Night",
            "day_pattern": "daily",
            "hour_distribution": {23: 4, 0: 3, 1: 2, 12: 1},
        },
    )

    scenes = scene_detector.detect_scenes(days=30)

    assert len(scenes) == 1
    assert scenes[0]["confidence"] == 60


def test_extract_room_uses_hoisted_keyword_cache(monkeypatch) -> None:
    monkeypatch.setattr(scene_detector, "_ROOM_KEYWORDS_SORTED", ("galley", "kitchen"))

    assert scene_detector._extract_room("light.boat_galley_main") == "Galley"


def test_room_keyword_cache_prefers_longest_keyword() -> None:
    sorted_keywords = scene_detector._ROOM_KEYWORDS_SORTED
    assert sorted_keywords[0] == "master_bedroom"
    assert sorted_keywords.index("master_bedroom") < sorted_keywords.index("bedroom")
