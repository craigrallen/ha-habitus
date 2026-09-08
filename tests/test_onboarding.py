"""Tests for onboarding wizard."""
from __future__ import annotations

import json
import os

import pytest

from habitus.habitus.onboarding import (
    is_complete,
    get_status,
    complete_onboarding,
    reset_onboarding,
    TOTAL_STEPS,
    STEP_NAMES,
)


class TestIsComplete:
    def test_not_complete_when_no_file(self, tmp_data_dir):
        """Not complete when onboarding_complete.json doesn't exist."""
        assert not is_complete()

    def test_complete_when_file_exists(self, tmp_data_dir):
        """Complete when onboarding_complete.json exists."""
        path = os.path.join(str(tmp_data_dir), "onboarding_complete.json")
        with open(path, "w") as f:
            json.dump({"completed_at": "2025-01-01"}, f)
        assert is_complete()


class TestGetStatus:
    def test_status_not_complete(self, tmp_data_dir):
        """Status shows not complete."""
        status = get_status()
        assert status["complete"] is False
        assert status["current_step"] == 0
        assert status["total_steps"] == TOTAL_STEPS

    def test_status_complete(self, tmp_data_dir):
        """Status shows complete after completion."""
        complete_onboarding()
        status = get_status()
        assert status["complete"] is True
        assert status["current_step"] == TOTAL_STEPS

    def test_status_has_step_names(self, tmp_data_dir):
        """Status includes step names."""
        status = get_status()
        assert "step_names" in status
        assert len(status["step_names"]) == TOTAL_STEPS

    def test_status_completed_at_set(self, tmp_data_dir):
        """Completion timestamp is set after completing."""
        complete_onboarding()
        status = get_status()
        assert status["completed_at"] is not None


class TestCompleteOnboarding:
    def test_creates_completion_file(self, tmp_data_dir):
        """complete_onboarding creates the marker file."""
        record = complete_onboarding()
        assert is_complete()
        assert "completed_at" in record

    def test_saves_tariff_to_settings(self, tmp_data_dir):
        """Tariff saved to settings.json."""
        complete_onboarding(tariff=0.25, tariff_peak=0.40, tariff_offpeak=0.12)
        settings_path = os.path.join(str(tmp_data_dir), "settings.json")
        assert os.path.exists(settings_path)
        with open(settings_path) as f:
            settings = json.load(f)
        assert settings["energy_tariff"] == 0.25
        assert settings["energy_tariff_peak"] == 0.40
        assert settings["energy_tariff_offpeak"] == 0.12

    def test_saves_notification_prefs(self, tmp_data_dir):
        """Notification prefs saved to settings.json."""
        prefs = {"suggestions": True, "anomalies": False}
        complete_onboarding(notification_prefs=prefs)
        settings_path = os.path.join(str(tmp_data_dir), "settings.json")
        with open(settings_path) as f:
            settings = json.load(f)
        assert settings["notification_prefs"] == prefs

    def test_skipped_flag(self, tmp_data_dir):
        """Skipped flag is stored."""
        complete_onboarding(skipped=True)
        path = os.path.join(str(tmp_data_dir), "onboarding_complete.json")
        with open(path) as f:
            data = json.load(f)
        assert data["skipped"] is True

    def test_complete_without_params(self, tmp_data_dir):
        """complete_onboarding works without any optional params."""
        record = complete_onboarding()
        assert is_complete()
        assert record is not None


class TestResetOnboarding:
    def test_reset_removes_file(self, tmp_data_dir):
        """Reset removes completion file."""
        complete_onboarding()
        assert is_complete()
        reset_onboarding()
        assert not is_complete()

    def test_reset_idempotent(self, tmp_data_dir):
        """Reset can be called multiple times without error."""
        reset_onboarding()
        reset_onboarding()

    def test_status_after_reset(self, tmp_data_dir):
        """Status shows not complete after reset."""
        complete_onboarding()
        reset_onboarding()
        status = get_status()
        assert status["complete"] is False


class TestApiEndpointResponse:
    """Simulate what the API endpoints should return."""

    def test_status_response_structure(self, tmp_data_dir):
        """Status response has all required fields."""
        status = get_status()
        required = {"complete", "current_step", "total_steps", "step_names"}
        for field in required:
            assert field in status

    def test_complete_response_structure(self, tmp_data_dir):
        """Complete response has required fields."""
        record = complete_onboarding()
        assert "completed_at" in record
        assert "skipped" in record
