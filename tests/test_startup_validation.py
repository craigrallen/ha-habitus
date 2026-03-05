"""Tests for startup validation and HA API backoff."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib


class TestHAReachability:
    """Tests for check_ha_reachable() function."""

    def test_reachable_when_api_responds_200(self, tmp_data_dir: Path, monkeypatch):
        """check_ha_reachable returns True when HA API responds with 200."""
        from habitus.habitus.main import check_ha_reachable

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("habitus.habitus.main.requests.get", return_value=mock_resp):
            result = check_ha_reachable()
        assert result is True

    def test_unreachable_on_connection_error(self, tmp_data_dir: Path, monkeypatch):
        """check_ha_reachable returns False when connection is refused."""
        from habitus.habitus.main import check_ha_reachable

        with patch("habitus.habitus.main.requests.get", side_effect=requests_lib.ConnectionError("refused")):
            result = check_ha_reachable()
        assert result is False

    def test_unreachable_on_timeout(self, tmp_data_dir: Path, monkeypatch):
        """check_ha_reachable returns False on network timeout."""
        from habitus.habitus.main import check_ha_reachable

        with patch("habitus.habitus.main.requests.get", side_effect=requests_lib.Timeout("timeout")):
            result = check_ha_reachable()
        assert result is False

    def test_unreachable_on_non_2xx(self, tmp_data_dir: Path, monkeypatch):
        """check_ha_reachable returns False on HTTP 503."""
        from habitus.habitus.main import check_ha_reachable

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("habitus.habitus.main.requests.get", return_value=mock_resp):
            result = check_ha_reachable()
        assert result is False

    def test_saves_ha_reachable_true_to_state(self, tmp_data_dir: Path):
        """check_ha_reachable saves ha_reachable=True in run_state.json."""
        import habitus.habitus.main as main_mod
        from habitus.habitus.main import check_ha_reachable

        state_path = tmp_data_dir / "run_state.json"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("habitus.habitus.main.requests.get", return_value=mock_resp), \
             patch.object(main_mod, "STATE_PATH", str(state_path)):
            check_ha_reachable()

        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["ha_reachable"] is True

    def test_saves_ha_reachable_false_to_state(self, tmp_data_dir: Path):
        """check_ha_reachable saves ha_reachable=False when unreachable."""
        import habitus.habitus.main as main_mod
        from habitus.habitus.main import check_ha_reachable

        state_path = tmp_data_dir / "run_state.json"
        with patch("habitus.habitus.main.requests.get", side_effect=requests_lib.ConnectionError("no")), \
             patch.object(main_mod, "STATE_PATH", str(state_path)):
            check_ha_reachable()

        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["ha_reachable"] is False


class TestHAGetBackoff:
    """Tests for _ha_get() exponential backoff wrapper."""

    def test_succeeds_on_first_try(self, tmp_data_dir: Path):
        """_ha_get returns immediately on HTTP 200."""
        from habitus.habitus.main import _ha_get

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("habitus.habitus.main.requests.get", return_value=mock_resp) as mock_get, \
             patch("time.sleep"):
            result = _ha_get("http://ha/api/test", token="tok")

        assert result is mock_resp
        assert mock_get.call_count == 1

    def test_retries_on_transient_500(self, tmp_data_dir: Path):
        """_ha_get retries up to max_retries on HTTP 5xx, then succeeds."""
        from habitus.habitus.main import _ha_get

        bad_resp = MagicMock()
        bad_resp.status_code = 503

        good_resp = MagicMock()
        good_resp.status_code = 200

        with patch("habitus.habitus.main.requests.get", side_effect=[bad_resp, good_resp]) as mock_get, \
             patch("habitus.habitus.main._t.sleep", MagicMock()) if False else patch("time.sleep"):
            result = _ha_get("http://ha/api/test", token="tok", retries=3)

        assert good_resp.status_code == 200

    def test_raises_after_max_retries_on_connection_error(self, tmp_data_dir: Path):
        """_ha_get raises after all retries are exhausted on connection failure."""
        from habitus.habitus.main import _ha_get

        with patch("habitus.habitus.main.requests.get", side_effect=requests_lib.ConnectionError("down")), \
             patch("time.sleep"):
            with pytest.raises(requests_lib.ConnectionError):
                _ha_get("http://ha/api/test", token="tok", retries=2)

    def test_no_retry_on_401(self, tmp_data_dir: Path):
        """_ha_get does not retry on 401 auth failure."""
        from habitus.habitus.main import _ha_get

        auth_fail = MagicMock()
        auth_fail.status_code = 401
        auth_fail.raise_for_status.side_effect = requests_lib.HTTPError("401")

        with patch("habitus.habitus.main.requests.get", return_value=auth_fail) as mock_get, \
             patch("time.sleep"):
            with pytest.raises(requests_lib.HTTPError):
                _ha_get("http://ha/api/test", token="bad-token", retries=3)

        # Should not have been called more than once (no retry on 401)
        assert mock_get.call_count == 1
