"""Tests for activity_hmm.py module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestLabelState:
    """Tests for _label_state function."""

    def test_sleeping_label(self):
        """Low activity at 2am → sleeping."""
        from habitus.habitus.activity_hmm import _label_state
        # lights, switches, motion, media, climate, doors, people, changes, hour
        centroid = np.array([0, 0, 0, 0, 0, 0, 1, 0, 2])
        result = _label_state(centroid)
        assert result == "sleeping"

    def test_away_label(self):
        """No people, no motion → away."""
        from habitus.habitus.activity_hmm import _label_state
        centroid = np.array([0, 0, 0, 0, 0, 0, 0, 0, 14])
        result = _label_state(centroid)
        assert result == "away"

    def test_relaxing_label(self):
        """Media + lights at 20:00 → relaxing."""
        from habitus.habitus.activity_hmm import _label_state
        centroid = np.array([2, 1, 1, 2, 0, 0, 1, 5, 20])
        result = _label_state(centroid)
        assert result == "relaxing"

    def test_working_label(self):
        """Lights + motion, no media, 10am → working."""
        from habitus.habitus.activity_hmm import _label_state
        centroid = np.array([1, 1, 2, 0, 0, 0, 1, 5, 10])
        result = _label_state(centroid)
        assert result == "working"

    def test_idle_label_fallback(self):
        """Ambiguous state returns some label."""
        from habitus.habitus.activity_hmm import _label_state
        centroid = np.array([1, 0, 1, 0, 0, 0, 1, 2, 14])
        result = _label_state(centroid)
        assert isinstance(result, str)
        assert len(result) > 0


class TestBuildObservationMatrix:
    def test_returns_empty_no_db(self, tmp_data_dir: Path):
        """_build_observation_matrix returns empty array when DB unavailable."""
        from habitus.habitus.activity_hmm import _build_observation_matrix
        with patch("habitus.habitus.activity_hmm.resolve_ha_db_path", return_value=None):
            X, timestamps = _build_observation_matrix({"sensor.motion": "living_room"}, days=7)
        assert isinstance(X, np.ndarray)
        assert len(X) == 0
        assert timestamps == []

    def test_returns_empty_when_db_error(self, tmp_data_dir: Path):
        """_build_observation_matrix returns empty when DB raises error."""
        import sqlite3
        from habitus.habitus.activity_hmm import _build_observation_matrix
        with patch("habitus.habitus.activity_hmm.resolve_ha_db_path", return_value="/fake/db.db"), \
             patch("sqlite3.connect", side_effect=sqlite3.OperationalError("no db")):
            X, timestamps = _build_observation_matrix({}, days=7)
        assert len(X) == 0


class TestTrainActivityModel:
    def test_returns_empty_no_db(self, tmp_data_dir: Path):
        """train_activity_model returns empty states when DB unavailable."""
        from habitus.habitus.activity_hmm import train_activity_model
        with patch("habitus.habitus.activity_hmm.resolve_ha_db_path", return_value=None):
            result = train_activity_model({})
        assert isinstance(result, dict)
        assert result.get("states") == [] or result.get("states") is None
        assert result.get("current_state") is None

    def test_returns_empty_for_insufficient_data(self, tmp_data_dir: Path):
        """train_activity_model with < 24 observations returns empty states."""
        from habitus.habitus.activity_hmm import train_activity_model
        import numpy as np

        small_X = np.zeros((10, 9))
        with patch("habitus.habitus.activity_hmm._build_observation_matrix",
                   return_value=(small_X, ["ts1"] * 10)):
            result = train_activity_model({})
        assert result["states"] == []

    def test_trains_with_synthetic_data_kmeans(self, tmp_data_dir: Path):
        """train_activity_model with enough data runs KMeans fallback."""
        from habitus.habitus import activity_hmm
        import numpy as np

        # 100 observations × 9 features
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 3, (100, 9))
        X[:, -1] = [h % 24 for h in range(100)]  # hour column
        timestamps = [f"2025-01-01T{h%24:02d}:00:00Z" for h in range(100)]
        hmm_path = str(tmp_data_dir / "activity_states.json")

        with patch("habitus.habitus.activity_hmm._build_observation_matrix",
                   return_value=(X, timestamps)), \
             patch.object(activity_hmm, "HMM_PATH", hmm_path), \
             patch.object(activity_hmm, "DATA_DIR", str(tmp_data_dir)):
            result = activity_hmm.train_activity_model({})

        assert isinstance(result, dict)
        assert "states" in result
        assert isinstance(result["states"], list)

    def test_writes_result_to_file(self, tmp_data_dir: Path):
        """train_activity_model writes result to HMM_PATH."""
        from habitus.habitus import activity_hmm
        import json
        import numpy as np

        X = np.zeros((30, 9))
        X[:, -1] = [h % 24 for h in range(30)]
        hmm_path = str(tmp_data_dir / "activity_states.json")

        with patch("habitus.habitus.activity_hmm._build_observation_matrix",
                   return_value=(X, ["ts"] * 30)), \
             patch.object(activity_hmm, "HMM_PATH", hmm_path), \
             patch.object(activity_hmm, "DATA_DIR", str(tmp_data_dir)):
            result = activity_hmm.train_activity_model({})

        assert (tmp_data_dir / "activity_states.json").exists()
        loaded = json.loads((tmp_data_dir / "activity_states.json").read_text())
        assert "states" in loaded
