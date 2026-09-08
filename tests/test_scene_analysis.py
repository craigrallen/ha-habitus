"""Tests for scene_analysis module.

Covers:
- Missing entity detection from co-occurrence patterns.
- Empty suggestions when no learned data exists.
- Trigger suggestion generation from time patterns.
- Score computation.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from habitus.habitus.scene_analysis import (
    _build_co_occurrence_index,
    _improvement_score,
    _suggest_triggers,
    analyse_scenes,
    load_cached_analysis,
    save_analysis,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ha_scene(
    entity_id: str,
    friendly_name: str,
    entities: list[str],
) -> dict:
    """Build a mock HA scene state dict."""
    return {
        "entity_id": entity_id,
        "state": "scening",
        "attributes": {
            "friendly_name": friendly_name,
            "entity_id": entities,
        },
    }


def _make_discovered_scene(
    name: str,
    entities: list[str],
    occurrences: int = 10,
    confidence: int = 70,
    peak_hour: int = 20,
    day_pattern: str = "daily",
) -> dict:
    """Build a mock habitus-discovered scene dict."""
    return {
        "id": f"scene_{name.lower().replace(' ', '_')}",
        "name": name,
        "entities": entities,
        "occurrences": occurrences,
        "confidence": confidence,
        "time_pattern": {
            "peak_hour": peak_hour,
            "window": f"{peak_hour:02d}:00-{peak_hour + 2:02d}:00",
            "label": "evening",
            "days": day_pattern,
        },
    }


# ---------------------------------------------------------------------------
# Co-occurrence index
# ---------------------------------------------------------------------------


def test_co_occurrence_index_builds_correctly() -> None:
    """Index maps each entity to its co-occurring partners with counts."""
    scenes = [
        _make_discovered_scene("Movie Time", ["light.living", "media_player.tv", "light.ambient"], occurrences=10),
    ]
    index = _build_co_occurrence_index(scenes)
    # light.living should co-occur with media_player.tv
    assert "light.living" in index
    assert index["light.living"]["media_player.tv"] == 10
    assert index["media_player.tv"]["light.living"] == 10


def test_co_occurrence_index_empty_scenes() -> None:
    """Empty discovered scenes produce empty index."""
    index = _build_co_occurrence_index([])
    assert index == {}


# ---------------------------------------------------------------------------
# Missing entity detection
# ---------------------------------------------------------------------------


def test_missing_entity_flagged_for_scene() -> None:
    """Entity frequently co-activated with scene entities but not in scene gets flagged."""
    ha_scene = _make_ha_scene(
        "scene.movie_time",
        "Movie Time",
        ["light.living", "media_player.tv"],  # light.ambient missing
    )
    discovered = [
        _make_discovered_scene(
            "Movie Time",
            ["light.living", "media_player.tv", "light.ambient"],
            occurrences=15,
        )
    ]
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=discovered)

    assert len(results) == 1
    suggestion = results[0]
    missing_ids = [m["entity_id"] for m in suggestion["missing_entities"]]
    assert "light.ambient" in missing_ids


def test_entity_already_in_scene_not_flagged() -> None:
    """Entity already in the scene is NOT flagged as missing."""
    ha_scene = _make_ha_scene(
        "scene.evening",
        "Evening",
        ["light.living", "light.ambient"],
    )
    discovered = [
        _make_discovered_scene(
            "Evening",
            ["light.living", "light.ambient"],
            occurrences=20,
        )
    ]
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=discovered)
    assert len(results) == 1
    assert results[0]["missing_entities"] == []


def test_non_scene_domain_entity_not_flagged() -> None:
    """Entities from non-scene domains (binary_sensor, person) are not flagged as missing."""
    ha_scene = _make_ha_scene(
        "scene.morning",
        "Morning",
        ["light.kitchen"],
    )
    # binary_sensor co-occurs but should not be suggested for scene
    discovered = [
        _make_discovered_scene(
            "Morning",
            ["light.kitchen", "binary_sensor.motion"],
            occurrences=20,
        )
    ]
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=discovered)
    assert len(results) == 1
    missing_ids = [m["entity_id"] for m in results[0]["missing_entities"]]
    assert "binary_sensor.motion" not in missing_ids


# ---------------------------------------------------------------------------
# No learned data — graceful empty result
# ---------------------------------------------------------------------------


def test_no_discovered_scenes_gives_empty_suggestions() -> None:
    """Scene with no learned data gives empty missing entities and triggers."""
    ha_scene = _make_ha_scene(
        "scene.unlearned",
        "Unlearned",
        ["light.bedroom"],
    )
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=[])
    assert len(results) == 1
    assert results[0]["missing_entities"] == []
    assert results[0]["suggested_triggers"] == []


def test_no_ha_scenes_gives_empty_results() -> None:
    """No HA scenes → empty result list."""
    results = analyse_scenes(ha_scenes=[], discovered_scenes=[])
    assert results == []


def test_ha_scenes_with_empty_entity_list() -> None:
    """HA scene with no entities doesn't crash and gives empty suggestions."""
    ha_scene = _make_ha_scene("scene.empty", "Empty Scene", [])
    discovered = [_make_discovered_scene("Something", ["light.a", "light.b"], occurrences=10)]
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=discovered)
    assert len(results) == 1
    assert results[0]["missing_entities"] == []


# ---------------------------------------------------------------------------
# Trigger suggestion generation
# ---------------------------------------------------------------------------


def test_trigger_suggestion_from_time_pattern() -> None:
    """Trigger suggestion generated from discovered scene time pattern."""
    discovered_scene = _make_discovered_scene(
        "Movie Time",
        ["light.living", "media_player.tv"],
        occurrences=20,
        confidence=75,
        peak_hour=20,
        day_pattern="weekdays",
    )
    suggestions = _suggest_triggers(discovered_scene)
    assert len(suggestions) >= 1
    trigger = suggestions[0]
    assert "20:00:00" in trigger["description"]
    assert trigger["trigger_yaml"] != ""
    assert "platform: time" in trigger["trigger_yaml"]


def test_trigger_suggestion_none_when_no_discovered_scene() -> None:
    """No trigger suggestions when no matching discovered scene."""
    suggestions = _suggest_triggers(None)
    assert suggestions == []


def test_trigger_suggestion_includes_day_pattern() -> None:
    """Trigger suggestion for weekdays includes day condition in YAML."""
    discovered_scene = _make_discovered_scene(
        "Morning",
        ["light.kitchen"],
        peak_hour=7,
        day_pattern="weekdays",
    )
    suggestions = _suggest_triggers(discovered_scene)
    assert len(suggestions) >= 1
    assert "weekdays" in suggestions[0]["description"]
    # YAML should reference weekday condition
    yaml_str = suggestions[0]["trigger_yaml"]
    assert "condition" in yaml_str or "weekday" in yaml_str


def test_trigger_suggestion_for_weekends() -> None:
    """Trigger suggestion for weekend pattern."""
    discovered_scene = _make_discovered_scene(
        "Lazy Sunday",
        ["light.bedroom", "media_player.tv"],
        peak_hour=10,
        day_pattern="weekends",
    )
    suggestions = _suggest_triggers(discovered_scene)
    assert len(suggestions) >= 1
    assert "10:00:00" in suggestions[0]["description"]


def test_ha_scene_gets_trigger_from_matching_discovered_scene() -> None:
    """Full analyse_scenes generates trigger suggestions from matching discovered scene."""
    ha_scene = _make_ha_scene(
        "scene.movie",
        "Movie Time",
        ["light.living", "media_player.tv"],
    )
    discovered = [
        _make_discovered_scene(
            "Movie Time",
            ["light.living", "media_player.tv", "light.ambient"],
            peak_hour=20,
            day_pattern="weekdays",
        )
    ]
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=discovered)
    assert len(results) == 1
    assert len(results[0]["suggested_triggers"]) >= 1
    assert "20:00:00" in results[0]["suggested_triggers"][0]["description"]


# ---------------------------------------------------------------------------
# Improvement score
# ---------------------------------------------------------------------------


def test_improvement_score_increases_with_missing_entities() -> None:
    """More missing entities → higher improvement score."""
    base = _improvement_score(0, 0, 50)
    with_one = _improvement_score(1, 0, 50)
    with_three = _improvement_score(3, 0, 50)
    assert with_one > base
    assert with_three > with_one


def test_improvement_score_increases_with_triggers() -> None:
    """More triggers → higher improvement score."""
    base = _improvement_score(0, 0, 50)
    with_trigger = _improvement_score(0, 1, 50)
    assert with_trigger > base


def test_improvement_score_capped_at_100() -> None:
    """Score never exceeds 100."""
    score = _improvement_score(20, 10, 100)
    assert score <= 100


def test_improvement_score_minimum_zero_range() -> None:
    """Score is non-negative."""
    score = _improvement_score(0, 0, 0)
    assert score >= 0


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_has_required_fields() -> None:
    """Each suggestion dict has all required fields."""
    ha_scene = _make_ha_scene("scene.test", "Test", ["light.a"])
    results = analyse_scenes(ha_scenes=[ha_scene], discovered_scenes=[])
    assert len(results) == 1
    r = results[0]
    assert "scene_id" in r
    assert "scene_name" in r
    assert "missing_entities" in r
    assert "suggested_triggers" in r
    assert "improvement_score" in r
    assert "why_suggested" in r


def test_results_sorted_by_improvement_score_descending() -> None:
    """Results sorted by improvement_score descending."""
    ha_scenes = [
        _make_ha_scene("scene.a", "A", ["light.a"]),
        _make_ha_scene("scene.b", "B", ["light.a", "light.b"]),
    ]
    discovered = [
        _make_discovered_scene("Discovered", ["light.a", "light.b", "switch.c"], occurrences=30),
    ]
    results = analyse_scenes(ha_scenes=ha_scenes, discovered_scenes=discovered)
    scores = [r["improvement_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def test_save_and_load_cached_analysis(tmp_path, monkeypatch) -> None:
    """save_analysis → load_cached_analysis round-trips correctly."""
    import habitus.habitus.scene_analysis as sa

    monkeypatch.setattr(sa, "SCENE_ANALYSIS_PATH", str(tmp_path / "scene_analysis.json"))
    monkeypatch.setattr(sa, "DATA_DIR", str(tmp_path))

    data = [{"scene_id": "scene.test", "score": 80}]
    save_analysis(data)

    loaded = load_cached_analysis()
    assert loaded == data


def test_load_cached_analysis_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    """load_cached_analysis returns None when file doesn't exist."""
    import habitus.habitus.scene_analysis as sa

    monkeypatch.setattr(sa, "SCENE_ANALYSIS_PATH", str(tmp_path / "nonexistent.json"))
    result = load_cached_analysis()
    assert result is None
