"""Tests for adaptive IsolationForest contamination (TASK-020)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from habitus.habitus.main import (
    contamination_for_days,
    contamination_tier_name,
    should_retrain_for_tier_change,
    train_model,
    FEATURE_COLS,
)


# ── contamination_for_days ─────────────────────────────────────────────────────


class TestContaminationForDays:
    def test_warmup_tier_below_7_days(self):
        assert contamination_for_days(0) == 0.005
        assert contamination_for_days(1) == 0.005
        assert contamination_for_days(6) == 0.005

    def test_early_tier_7_to_13_days(self):
        assert contamination_for_days(7) == 0.01
        assert contamination_for_days(10) == 0.01
        assert contamination_for_days(13) == 0.01

    def test_growing_tier_14_to_29_days(self):
        assert contamination_for_days(14) == 0.02
        assert contamination_for_days(20) == 0.02
        assert contamination_for_days(29) == 0.02

    def test_mature_tier_30_to_89_days(self):
        assert contamination_for_days(30) == 0.04
        assert contamination_for_days(60) == 0.04
        assert contamination_for_days(89) == 0.04

    def test_established_tier_90_plus_days(self):
        assert contamination_for_days(90) == 0.05
        assert contamination_for_days(180) == 0.05
        assert contamination_for_days(365) == 0.05

    def test_values_are_monotonically_non_decreasing(self):
        breakpoints = [0, 7, 14, 30, 90]
        values = [contamination_for_days(d) for d in breakpoints]
        for a, b in zip(values, values[1:]):
            assert b >= a

    def test_all_values_in_valid_range(self):
        for days in [0, 5, 7, 10, 14, 20, 30, 60, 90, 365]:
            c = contamination_for_days(days)
            assert 0.0 < c <= 1.0


# ── contamination_tier_name ────────────────────────────────────────────────────


class TestContaminationTierName:
    def test_warmup_below_7(self):
        assert contamination_tier_name(0) == "warmup"
        assert contamination_tier_name(6) == "warmup"

    def test_early_7_to_13(self):
        assert contamination_tier_name(7) == "early"
        assert contamination_tier_name(13) == "early"

    def test_growing_14_to_29(self):
        assert contamination_tier_name(14) == "growing"
        assert contamination_tier_name(29) == "growing"

    def test_mature_30_to_89(self):
        assert contamination_tier_name(30) == "mature"
        assert contamination_tier_name(89) == "mature"

    def test_established_90_plus(self):
        assert contamination_tier_name(90) == "established"
        assert contamination_tier_name(365) == "established"

    def test_returns_string(self):
        for days in [0, 7, 14, 30, 90]:
            assert isinstance(contamination_tier_name(days), str)

    def test_tier_matches_contamination_tier(self):
        """Every distinct contamination value corresponds to a unique tier name."""
        seen: dict[float, str] = {}
        for days in [0, 7, 14, 30, 90]:
            c = contamination_for_days(days)
            name = contamination_tier_name(days)
            if c in seen:
                assert seen[c] == name
            seen[c] = name


# ── should_retrain_for_tier_change ─────────────────────────────────────────────


class TestShouldRetrainForTierChange:
    def test_no_stored_tier_returns_false(self):
        assert should_retrain_for_tier_change({}, 30) is False

    def test_empty_tier_returns_false(self):
        assert should_retrain_for_tier_change({"contamination_tier": ""}, 30) is False

    def test_same_tier_returns_false(self):
        state = {"contamination_tier": "warmup"}
        assert should_retrain_for_tier_change(state, 5) is False

    def test_tier_advance_returns_true(self):
        state = {"contamination_tier": "warmup"}
        # days=7 → "early", stored is "warmup" → tier change
        assert should_retrain_for_tier_change(state, 7) is True

    def test_warmup_to_established_returns_true(self):
        state = {"contamination_tier": "warmup"}
        assert should_retrain_for_tier_change(state, 365) is True

    def test_early_to_growing_returns_true(self):
        state = {"contamination_tier": "early"}
        assert should_retrain_for_tier_change(state, 14) is True

    def test_mature_to_established_returns_true(self):
        state = {"contamination_tier": "mature"}
        assert should_retrain_for_tier_change(state, 90) is True

    def test_established_stays_established(self):
        state = {"contamination_tier": "established"}
        assert should_retrain_for_tier_change(state, 365) is False

    def test_growing_stays_growing(self):
        state = {"contamination_tier": "growing"}
        assert should_retrain_for_tier_change(state, 20) is False


# ── train_model uses adaptive contamination ────────────────────────────────────


class TestTrainModelContamination:
    def _make_features(self) -> pd.DataFrame:
        """Create a minimal feature matrix for train_model()."""
        import datetime

        hours = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
        df = pd.DataFrame({"hour": hours})
        for col in FEATURE_COLS:
            df[col] = 0.0
        df["hour_of_day"] = df["hour"].dt.hour
        df["day_of_week"] = df["hour"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
        df["month"] = df["hour"].dt.month
        return df

    def test_warmup_contamination(self):
        """train_model with 5 days should use contamination=0.005."""
        features = self._make_features()
        captured: list[float] = []

        from sklearn.ensemble import IsolationForest

        orig_iso = IsolationForest

        def patched_iso(**kwargs):
            captured.append(kwargs.get("contamination", -1))
            return orig_iso(**kwargs)

        with patch("sklearn.ensemble.IsolationForest", side_effect=patched_iso):
            train_model(features, training_days=5)

        assert len(captured) == 1
        assert captured[0] == pytest.approx(0.005)

    def test_established_contamination(self):
        """train_model with 120 days should use contamination=0.05."""
        features = self._make_features()
        captured: list[float] = []

        from sklearn.ensemble import IsolationForest

        orig_iso = IsolationForest

        def patched_iso(**kwargs):
            captured.append(kwargs.get("contamination", -1))
            return orig_iso(**kwargs)

        with patch("sklearn.ensemble.IsolationForest", side_effect=patched_iso):
            train_model(features, training_days=120)

        assert len(captured) == 1
        assert captured[0] == pytest.approx(0.05)

    def test_default_training_days_zero_uses_warmup(self):
        """Default training_days=0 should give warmup contamination."""
        features = self._make_features()
        captured: list[float] = []

        from sklearn.ensemble import IsolationForest

        orig_iso = IsolationForest

        def patched_iso(**kwargs):
            captured.append(kwargs.get("contamination", -1))
            return orig_iso(**kwargs)

        with patch("sklearn.ensemble.IsolationForest", side_effect=patched_iso):
            train_model(features)

        assert captured[0] == pytest.approx(0.005)

    def test_returns_model_and_scaler(self):
        """train_model should return (model, scaler) regardless of days."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        features = self._make_features()
        model, scaler = train_model(features, training_days=60)
        assert isinstance(model, IsolationForest)
        assert isinstance(scaler, StandardScaler)


# ── contamination_tier stored in run_state.json ────────────────────────────────


class TestContaminationTierInState:
    def test_tier_stored_after_first_run(self, tmp_data_dir):
        """After computing training_days, contamination_tier should be derivable."""
        state_path = tmp_data_dir / "run_state.json"
        # Simulate what run() does when saving state after training
        training_days = 45  # mature tier
        state = {
            "training_days": training_days,
            "contamination_tier": contamination_tier_name(training_days),
        }
        state_path.write_text(json.dumps(state))

        loaded = json.loads(state_path.read_text())
        assert loaded["contamination_tier"] == "mature"

    def test_tier_change_detected_from_stored_state(self, tmp_data_dir):
        """When stored tier is stale, should_retrain_for_tier_change returns True."""
        state = {"training_days": 30, "contamination_tier": "growing"}
        # 30 days → "mature", stored is "growing" → retrain needed
        assert should_retrain_for_tier_change(state, 30) is True

    def test_no_retrain_when_tier_current(self, tmp_data_dir):
        state = {"training_days": 45, "contamination_tier": "mature"}
        assert should_retrain_for_tier_change(state, 45) is False
