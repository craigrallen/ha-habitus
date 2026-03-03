"""Tests for seasonal model training and scoring."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habitus.habitus.seasonal import (
    current_season,
    load_seasonal_bundle,
    score_with_best_model,
    seasonal_status,
    train_seasonal_models,
)


# ---------------------------------------------------------------------------
# current_season — Northern Hemisphere (default)
# ---------------------------------------------------------------------------


class TestCurrentSeason:
    def test_winter(self):
        from freezegun import freeze_time

        with freeze_time("2025-01-15"):
            assert current_season() == "winter"

    def test_summer(self):
        from freezegun import freeze_time

        with freeze_time("2025-07-01"):
            assert current_season() == "summer"

    def test_spring(self):
        from freezegun import freeze_time

        with freeze_time("2025-04-10"):
            assert current_season() == "spring"

    def test_autumn(self):
        from freezegun import freeze_time

        with freeze_time("2025-10-20"):
            assert current_season() == "autumn"

    def test_december_is_winter(self):
        from freezegun import freeze_time

        with freeze_time("2025-12-25"):
            assert current_season() == "winter"


# ---------------------------------------------------------------------------
# current_season — hemisphere-aware
# ---------------------------------------------------------------------------


class TestHemisphereAwareness:
    """Southern-Hemisphere season flipping via explicit parameter."""

    def test_south_january_is_summer(self):
        from freezegun import freeze_time

        with freeze_time("2025-01-15"):
            assert current_season(hemisphere="south") == "summer"

    def test_south_july_is_winter(self):
        from freezegun import freeze_time

        with freeze_time("2025-07-01"):
            assert current_season(hemisphere="south") == "winter"

    def test_south_april_is_autumn(self):
        from freezegun import freeze_time

        with freeze_time("2025-04-10"):
            assert current_season(hemisphere="south") == "autumn"

    def test_south_october_is_spring(self):
        from freezegun import freeze_time

        with freeze_time("2025-10-20"):
            assert current_season(hemisphere="south") == "spring"

    def test_north_explicit_parameter(self):
        from freezegun import freeze_time

        with freeze_time("2025-07-01"):
            assert current_season(hemisphere="north") == "summer"

    def test_hemisphere_env_var_south(self, monkeypatch):
        import habitus.habitus.seasonal as s
        from freezegun import freeze_time

        monkeypatch.setattr(s, "HEMISPHERE", "south")
        with freeze_time("2025-01-15"):
            # January is summer in Southern Hemisphere; no explicit param →
            # function falls back to module-level HEMISPHERE
            assert current_season() == "summer"


# ---------------------------------------------------------------------------
# train_seasonal_models
# ---------------------------------------------------------------------------


class TestTrainSeasonalModels:
    def test_trains_available_seasons(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        assert len(trained) > 0

    def test_saves_pickle_files(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        for season in trained:
            assert (tmp_data_dir / f"model_{season}.pkl").exists()
            assert (tmp_data_dir / f"scaler_{season}.pkl").exists()

    def test_skips_seasons_with_insufficient_data(self, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        # Only 10 rows — below the 72h threshold for any season
        tiny = pd.DataFrame(
            {
                "hour": pd.date_range("2025-01-01", periods=10, freq="h"),
                "hour_of_day": range(10),
                "day_of_week": [0] * 10,
                "is_weekend": [0] * 10,
                "month": [1] * 10,
                "total_power_w": [100.0] * 10,
                "avg_temp_c": [20.0] * 10,
                "sensor_changes": [5] * 10,
            }
        )
        trained = train_seasonal_models(tiny)
        assert trained == []

    def test_saves_seasonal_bundle(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        assert len(trained) > 0
        assert (tmp_data_dir / "seasonal_models.pkl").exists()

    def test_no_bundle_when_no_seasons_trained(self, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        tiny = pd.DataFrame(
            {
                "month": [1] * 5,
                "hour_of_day": [0] * 5,
                "day_of_week": [0] * 5,
                "is_weekend": [0] * 5,
                "total_power_w": [100.0] * 5,
                "avg_temp_c": [20.0] * 5,
                "sensor_changes": [1] * 5,
            }
        )
        train_seasonal_models(tiny)
        assert not (tmp_data_dir / "seasonal_models.pkl").exists()


# ---------------------------------------------------------------------------
# load_seasonal_bundle
# ---------------------------------------------------------------------------


class TestSeasonalBundle:
    """Tests for the combined seasonal_models.pkl bundle."""

    def test_load_empty_dir(self, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        models, scalers = load_seasonal_bundle()
        assert models == {}
        assert scalers == {}

    def test_bundle_contains_trained_seasons(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        models, scalers = load_seasonal_bundle()
        for season in trained:
            assert season in models
            assert season in scalers

    def test_bundle_not_contaminated_by_other_keys(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        train_seasonal_models(sample_features)
        models, scalers = load_seasonal_bundle()
        # Should never contain extra season names
        valid = {"winter", "spring", "summer", "autumn"}
        assert set(models.keys()).issubset(valid)
        assert set(scalers.keys()).issubset(valid)


# ---------------------------------------------------------------------------
# seasonal_status
# ---------------------------------------------------------------------------


class TestSeasonalStatus:
    def test_empty_dir(self, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        status = seasonal_status()
        assert all(v is False for v in status.values())

    def test_after_training(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        status = seasonal_status()
        for season in trained:
            assert status[season] is True

    def test_untrained_seasons_remain_false(self, sample_features, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        status = seasonal_status()
        untrained = [season for season in status if season not in trained]
        for season in untrained:
            assert status[season] is False


# ---------------------------------------------------------------------------
# score_with_best_model
# ---------------------------------------------------------------------------


class TestScoreWithBestModel:
    """Tests for score_with_best_model."""

    @staticmethod
    def _make_feature_vec() -> np.ndarray:
        """Build a (1, N) feature array matching _get_feature_cols() order."""
        from habitus.habitus.seasonal import _get_feature_cols

        n = len(_get_feature_cols())
        # Populate the first 7 core columns; pad remainder with zeros
        row = [0.0] * n
        row[0] = 8.0    # hour_of_day
        row[1] = 1.0    # day_of_week
        row[2] = 0.0    # is_weekend
        row[3] = 1.0    # month
        row[4] = 500.0  # total_power_w
        row[5] = 20.0   # avg_temp_c
        row[6] = 3.0    # sensor_changes
        return np.array([row])

    def test_uses_seasonal_model_when_available(self, sample_features, tmp_data_dir):
        from freezegun import freeze_time

        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        trained = train_seasonal_models(sample_features)
        assert "winter" in trained, "sample_features must include enough winter data"
        with freeze_time("2025-01-15"):  # winter → uses model_winter.pkl
            score, used = score_with_best_model(self._make_feature_vec())
        assert 0 <= score <= 100
        assert used == "winter"

    def test_falls_back_to_main_model(self, sample_features, tmp_data_dir):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        # Build a main model only — no seasonal models present
        feature_cols = s._get_feature_cols()
        cols = [c for c in feature_cols if c in sample_features.columns]
        X = sample_features[cols].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(Xs)
        with open(tmp_data_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(tmp_data_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        # Freeze to a season that definitely has no model file
        from freezegun import freeze_time

        with freeze_time("2025-06-15"):  # summer — sample has no summer data
            score, used = score_with_best_model(self._make_feature_vec())
        assert 0 <= score <= 100
        assert used == "main"

    def test_returns_zero_on_error(self, tmp_data_dir):
        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        # No models exist — FileNotFoundError propagates to except block
        score, used = score_with_best_model(self._make_feature_vec())
        assert score == 0
        assert used == "error"

    def test_score_range_is_bounded(self, sample_features, tmp_data_dir):
        from freezegun import freeze_time

        import habitus.habitus.seasonal as s

        s.DATA_DIR = str(tmp_data_dir)
        train_seasonal_models(sample_features)
        with freeze_time("2025-01-15"):
            for _ in range(5):
                score, _ = score_with_best_model(self._make_feature_vec())
                assert 0 <= score <= 100
