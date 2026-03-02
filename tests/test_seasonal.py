"""Tests for seasonal model training and scoring."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habitus.habitus.seasonal import (
    current_season,
    train_seasonal_models,
    score_with_best_model,
    seasonal_status,
)


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
        tiny = pd.DataFrame({
            "hour": pd.date_range("2025-01-01", periods=10, freq="h"),
            "hour_of_day": range(10),
            "day_of_week": [0]*10,
            "is_weekend": [0]*10,
            "month": [1]*10,
            "total_power_w": [100.0]*10,
            "avg_temp_c": [20.0]*10,
            "sensor_changes": [5]*10,
        })
        trained = train_seasonal_models(tiny)
        assert trained == []


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
