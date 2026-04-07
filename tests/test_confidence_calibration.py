"""Confidence calibration tests.

Asserts that:
1. Suggestions trained on more data have equal or higher confidence than sparse data.
2. Monotonic behavior across edge cases (empty history, 1 day, 30 days, 90 days).
3. Confidence is bounded [0, 100].
4. Feedback boosts/suppressions are calibrated correctly.
"""
from __future__ import annotations

import math
import pytest

from habitus.habitus.scene_detector import _compute_scene_confidence, _analyze_time_patterns
from habitus.habitus.scene_analysis import _improvement_score
from habitus.habitus.suggestion_feedback import apply_feedback_to_suggestions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_time_info(
    count: int,
    peak_hour: int = 20,
    hour_distribution: dict | None = None,
) -> dict:
    """Build a minimal time_info dict for _compute_scene_confidence."""
    if hour_distribution is None:
        # Concentrate all observations in the peak hour
        hour_distribution = {peak_hour: count}
    return {
        "count": count,
        "peak_hour": peak_hour,
        "best_window_start": peak_hour,
        "best_window_end": (peak_hour + 2) % 24,
        "hour_distribution": hour_distribution,
        "day_pattern": "daily",
        "time_label": "evening",
    }


def _make_entities(domains: list[str]) -> set[str]:
    """Make a set of entity IDs with given domains."""
    return {f"{d}.test_entity_{i}" for i, d in enumerate(domains)}


def _make_suggestion(sid: str, confidence: float) -> dict:
    return {"id": sid, "confidence": confidence, "automation_id": sid}


# ---------------------------------------------------------------------------
# Scene confidence: more data = same or higher confidence
# ---------------------------------------------------------------------------


def test_confidence_increases_with_more_occurrences() -> None:
    """More occurrences → same or higher confidence score."""
    entities = _make_entities(["light", "media_player"])
    scores = []
    for count in [1, 5, 10, 20, 33]:
        ti = _make_time_info(count)
        scores.append(_compute_scene_confidence(entities, ti))
    # Verify monotonically non-decreasing
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"Confidence should not decrease: count_{i+1}={scores[i]} > count_{i+2}={scores[i+1]}"
        )


def test_confidence_zero_occurrences() -> None:
    """Zero occurrences → confidence of 0 or very low."""
    entities = _make_entities(["light"])
    ti = _make_time_info(count=0)
    score = _compute_scene_confidence(entities, ti)
    assert score == 0 or score < 20, f"Expected near-zero confidence for 0 occurrences, got {score}"


def test_confidence_single_day_lower_than_month() -> None:
    """1 day of data → lower confidence than 30 days of data."""
    entities = _make_entities(["light", "switch"])
    ti_1d = _make_time_info(count=1)
    ti_30d = _make_time_info(count=15)  # ~15 occurrences in 30 days
    score_1d = _compute_scene_confidence(entities, ti_1d)
    score_30d = _compute_scene_confidence(entities, ti_30d)
    assert score_1d <= score_30d, (
        f"30-day confidence ({score_30d}) should be >= 1-day confidence ({score_1d})"
    )


def test_confidence_30_vs_90_days() -> None:
    """90 days worth of data → higher or equal confidence than 30 days."""
    entities = _make_entities(["light"])
    ti_30d = _make_time_info(count=10)
    ti_90d = _make_time_info(count=30)  # 3× more data
    score_30d = _compute_scene_confidence(entities, ti_30d)
    score_90d = _compute_scene_confidence(entities, ti_90d)
    assert score_30d <= score_90d, (
        f"90-day confidence ({score_90d}) should be >= 30-day confidence ({score_30d})"
    )


def test_confidence_bounded_0_to_100() -> None:
    """Confidence score always in [0, 100] regardless of inputs."""
    test_cases = [
        (0, {}),
        (1000, {20: 1000}),
        (1, {0: 1}),
        (50, {h: 2 for h in range(24)}),
    ]
    entities = _make_entities(["light", "switch", "media_player"])
    for count, hour_dist in test_cases:
        ti = _make_time_info(count, hour_distribution=hour_dist or {20: count})
        score = _compute_scene_confidence(entities, ti)
        assert 0 <= score <= 100, f"Score {score} out of [0, 100] range for count={count}"


def test_confidence_not_nan_or_inf() -> None:
    """Confidence should never be NaN or Inf."""
    entities = _make_entities(["light"])
    for count in [0, 1, 5, 100]:
        ti = _make_time_info(count)
        score = _compute_scene_confidence(entities, ti)
        assert not math.isnan(score), f"Got NaN for count={count}"
        assert not math.isinf(score), f"Got Inf for count={count}"


# ---------------------------------------------------------------------------
# Diversity bonus: mixed domains boost confidence
# ---------------------------------------------------------------------------


def test_confidence_higher_with_diverse_domains() -> None:
    """Scenes with diverse entity domains get a diversity bonus."""
    ti = _make_time_info(count=10)
    score_single = _compute_scene_confidence(_make_entities(["light"]), ti)
    score_diverse = _compute_scene_confidence(
        _make_entities(["light", "media_player", "climate"]), ti
    )
    assert score_diverse >= score_single, (
        f"Diverse domains ({score_diverse}) should score >= single domain ({score_single})"
    )


# ---------------------------------------------------------------------------
# Time consistency: concentrated activations boost confidence
# ---------------------------------------------------------------------------


def test_time_consistency_affects_confidence() -> None:
    """Concentrated time distribution gives higher confidence than spread activations."""
    count = 20
    # All 20 in one hour → high time consistency
    concentrated = _make_time_info(count, hour_distribution={20: count})
    # Spread evenly → low time consistency
    spread = _make_time_info(count, hour_distribution={h: 1 for h in range(20)})

    entities = _make_entities(["light"])
    score_conc = _compute_scene_confidence(entities, concentrated)
    score_spread = _compute_scene_confidence(entities, spread)
    # Concentrated should be >= spread
    assert score_conc >= score_spread, (
        f"Concentrated ({score_conc}) should be >= spread ({score_spread})"
    )


# ---------------------------------------------------------------------------
# Improvement score calibration (scene_analysis)
# ---------------------------------------------------------------------------


def test_improvement_score_monotonic_with_missing_entities() -> None:
    """Improvement score monotonically increases with more missing entities (up to cap)."""
    base_conf = 60
    scores = [_improvement_score(n, 0, base_conf) for n in range(6)]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"Improvement score not monotonic: {scores[i]} > {scores[i + 1]} at n={i}"
        )


def test_improvement_score_monotonic_with_triggers() -> None:
    """Improvement score monotonically increases with more trigger suggestions (up to cap)."""
    base_conf = 60
    scores = [_improvement_score(0, n, base_conf) for n in range(4)]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"Improvement score not monotonic: {scores[i]} > {scores[i + 1]} at n={i}"
        )


def test_improvement_score_bounded() -> None:
    """Improvement score always in [0, 100]."""
    for n_missing in range(0, 10):
        for n_triggers in range(0, 5):
            for conf in [0, 40, 70, 100]:
                score = _improvement_score(n_missing, n_triggers, conf)
                assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Feedback calibration (suggestion_feedback)
# ---------------------------------------------------------------------------


def test_feedback_boost_increases_confidence() -> None:
    """'add' feedback → confidence increases."""
    sug = _make_suggestion("morning_lights", 60.0)
    from habitus.habitus.suggestion_feedback import apply_feedback_to_suggestions
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        import habitus.habitus.suggestion_feedback as sf
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            from habitus.habitus.suggestion_feedback import record_feedback
            record_feedback("morning_lights", "add")
            record_feedback("morning_lights", "add")
            result = apply_feedback_to_suggestions([{"id": "morning_lights", "confidence": 60.0}])
            assert result[0]["confidence"] > 60, "Add feedback should boost confidence"
            assert result[0].get("feedback_boosted") is True
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


def test_feedback_dismiss_suppresses_confidence() -> None:
    """Repeated 'dismiss' feedback → confidence decreases."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            sf.record_feedback("noisy_suggestion", "dismiss")
            sf.record_feedback("noisy_suggestion", "dismiss")
            sf.record_feedback("noisy_suggestion", "dismiss")
            result = sf.apply_feedback_to_suggestions([{"id": "noisy_suggestion", "confidence": 70.0}])
            assert result[0]["confidence"] < 70, "Dismiss feedback should reduce confidence"
            assert result[0].get("feedback_suppressed") is True
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


def test_feedback_single_dismiss_no_suppression() -> None:
    """Single dismiss (below threshold of 2) → no suppression."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            sf.record_feedback("single_dismiss", "dismiss")
            result = sf.apply_feedback_to_suggestions([{"id": "single_dismiss", "confidence": 60.0}])
            # Single dismiss (< threshold of 2) → no change
            assert result[0]["confidence"] == 60
            assert not result[0].get("feedback_suppressed", False)
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


def test_feedback_no_effect_on_unknown_suggestion() -> None:
    """Feedback for unknown suggestion ID → no change to other suggestions."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            sf.record_feedback("other_suggestion", "dismiss")
            sf.record_feedback("other_suggestion", "dismiss")
            result = sf.apply_feedback_to_suggestions([{"id": "my_suggestion", "confidence": 75.0}])
            assert result[0]["confidence"] == 75
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


def test_feedback_confidence_never_exceeds_99() -> None:
    """Boosted confidence is capped at 99."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            for _ in range(20):  # Many boosts
                sf.record_feedback("popular_sug", "add")
            result = sf.apply_feedback_to_suggestions([{"id": "popular_sug", "confidence": 90.0}])
            assert result[0]["confidence"] <= 99
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


def test_feedback_confidence_never_below_zero() -> None:
    """Suppressed confidence is floored at 0."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            for _ in range(20):  # Many dismissals
                sf.record_feedback("bad_sug", "dismiss")
            result = sf.apply_feedback_to_suggestions([{"id": "bad_sug", "confidence": 50.0}])
            assert result[0]["confidence"] >= 0
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_suggestion_list_no_crash() -> None:
    """apply_feedback_to_suggestions with empty list → empty list, no crash."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            result = sf.apply_feedback_to_suggestions([])
            assert result == []
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path


def test_suggestions_without_id_field_handled_gracefully() -> None:
    """Suggestions missing 'id' field are skipped safely."""
    import tempfile, os
    import habitus.habitus.suggestion_feedback as sf

    with tempfile.TemporaryDirectory() as tmp:
        original_path = sf.SUGGESTION_FEEDBACK_PATH
        sf.SUGGESTION_FEEDBACK_PATH = os.path.join(tmp, "feedback.json")
        try:
            sug = {"confidence": 65.0, "description": "No ID here"}
            result = sf.apply_feedback_to_suggestions([sug])
            assert result[0]["confidence"] == 65.0
        finally:
            sf.SUGGESTION_FEEDBACK_PATH = original_path
