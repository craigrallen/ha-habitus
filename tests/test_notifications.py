"""Tests for TASK-004: HA Notification Integration.

Covers send_notification, persistent_notification, format_digest_message,
should_send_digest, and send_daily_digest with mocked HA REST calls.
"""
from __future__ import annotations

import datetime
import json

import pytest
import responses as responses_lib

from habitus.habitus import main


@pytest.fixture(autouse=True)
def patch_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a clean, testable notification environment for every test."""
    monkeypatch.setattr(main, "NOTIFY_ON", True)
    monkeypatch.setattr(main, "NOTIFY_SVC", "notify.notify")
    monkeypatch.setattr(main, "HA_URL", "http://localhost:8123")
    monkeypatch.setattr(main, "HA_TOKEN", "test-token")
    monkeypatch.setattr(main, "DAILY_DIGEST", True)
    monkeypatch.setattr(main, "DAILY_DIGEST_HOUR", 8)


# ── send_notification ─────────────────────────────────────────────────────────


class TestSendNotification:
    @responses_lib.activate
    def test_posts_to_notify_service(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            json={"result": "ok"},
            status=200,
        )
        main.send_notification("Test Title", "Test message")
        assert len(responses_lib.calls) == 1
        payload = json.loads(responses_lib.calls[0].request.body)
        assert payload["title"] == "Test Title"
        assert payload["message"] == "Test message"

    def test_skips_when_notify_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "NOTIFY_ON", False)
        # No responses registered — any HTTP call would raise ConnectionError
        main.send_notification("Title", "Message")  # must return silently

    @responses_lib.activate
    def test_network_error_does_not_raise(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            body=ConnectionError("network down"),
        )
        main.send_notification("Title", "Message")  # must swallow the error

    @responses_lib.activate
    def test_custom_notify_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "NOTIFY_SVC", "notify.mobile_app_phone")
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/mobile_app_phone",
            json={},
            status=200,
        )
        main.send_notification("Title", "Message")
        assert len(responses_lib.calls) == 1

    @responses_lib.activate
    def test_request_carries_bearer_token(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            json={},
            status=200,
        )
        main.send_notification("T", "M")
        auth = responses_lib.calls[0].request.headers.get("Authorization", "")
        assert auth == "Bearer test-token"


# ── persistent_notification ───────────────────────────────────────────────────


class TestPersistentNotification:
    @responses_lib.activate
    def test_dismisses_then_creates(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/persistent_notification/dismiss",
            json={},
            status=200,
        )
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/persistent_notification/create",
            json={},
            status=200,
        )
        main.persistent_notification("test_id", "Test Title", "Test message")
        assert len(responses_lib.calls) == 2
        dismiss_payload = json.loads(responses_lib.calls[0].request.body)
        assert dismiss_payload["notification_id"] == "test_id"
        create_payload = json.loads(responses_lib.calls[1].request.body)
        assert create_payload["title"] == "Test Title"
        assert create_payload["message"] == "Test message"

    @responses_lib.activate
    def test_network_error_does_not_raise(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/persistent_notification/dismiss",
            body=ConnectionError("down"),
        )
        main.persistent_notification("test_id", "Title", "Message")  # must not propagate


# ── format_digest_message ─────────────────────────────────────────────────────


class TestFormatDigestMessage:
    def test_includes_score_sensors_days(self) -> None:
        msg = main.format_digest_message(75, [], [], 90, 42)
        assert "75/100" in msg
        assert "90" in msg
        assert "42" in msg

    def test_with_anomalies_and_suggestions(self) -> None:
        anomalies = [
            {"description": "Bathroom power is 340W — baseline Mon 20:00 is 45W ±12W"},
            {"description": "Kitchen motion at 03:00 — unusually high"},
        ]
        suggestions = [{"title": "Morning lights automation"}, {"title": "Away mode"}]
        msg = main.format_digest_message(75, anomalies, suggestions, 90, 42)
        assert "Bathroom" in msg
        assert "2 automation suggestions" in msg

    def test_no_anomalies_shows_normal(self) -> None:
        msg = main.format_digest_message(10, [], [], 30, 15)
        assert "No anomalies detected" in msg
        assert "10/100" in msg

    def test_single_suggestion_singular_noun(self) -> None:
        msg = main.format_digest_message(50, [], [{"title": "Morning lights"}], 60, 20)
        assert "1 automation suggestion available" in msg
        assert "suggestions" not in msg

    def test_no_suggestions_omits_suggestion_line(self) -> None:
        msg = main.format_digest_message(50, [{"description": "High power"}], [], 60, 20)
        assert "High power" in msg
        assert "suggestion" not in msg

    def test_truncates_anomalies_to_top_3(self) -> None:
        anomalies = [{"description": f"Entity {i}"} for i in range(5)]
        msg = main.format_digest_message(80, anomalies, [], 90, 50)
        assert "Entity 0" in msg
        assert "Entity 2" in msg
        assert "Entity 3" not in msg

    def test_falls_back_to_entity_key(self) -> None:
        anomalies = [{"entity": "sensor.foo", "z_score": 4.2}]
        msg = main.format_digest_message(60, anomalies, [], 45, 10)
        assert "sensor.foo" in msg


# ── should_send_digest ────────────────────────────────────────────────────────


class TestShouldSendDigest:
    _DIGEST_TIME = datetime.datetime(2026, 3, 3, 8, 0, tzinfo=datetime.timezone.utc)
    _OFF_HOUR = datetime.datetime(2026, 3, 3, 15, 0, tzinfo=datetime.timezone.utc)

    def test_returns_false_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "DAILY_DIGEST", False)
        assert not main.should_send_digest({}, self._DIGEST_TIME)

    def test_returns_false_wrong_hour(self) -> None:
        assert not main.should_send_digest({}, self._OFF_HOUR)

    def test_returns_true_correct_hour_not_sent_today(self) -> None:
        assert main.should_send_digest({}, self._DIGEST_TIME)

    def test_returns_false_already_sent_today(self) -> None:
        state = {"last_digest_date": "2026-03-03"}
        assert not main.should_send_digest(state, self._DIGEST_TIME)

    def test_returns_true_sent_yesterday(self) -> None:
        state = {"last_digest_date": "2026-03-02"}
        assert main.should_send_digest(state, self._DIGEST_TIME)

    def test_custom_digest_hour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "DAILY_DIGEST_HOUR", 20)
        evening = datetime.datetime(2026, 3, 3, 20, 0, tzinfo=datetime.timezone.utc)
        assert main.should_send_digest({}, evening)
        morning = datetime.datetime(2026, 3, 3, 8, 0, tzinfo=datetime.timezone.utc)
        assert not main.should_send_digest({}, morning)


# ── send_daily_digest ─────────────────────────────────────────────────────────


class TestSendDailyDigest:
    _NOW = datetime.datetime(2026, 3, 3, 8, 0, tzinfo=datetime.timezone.utc)

    @responses_lib.activate
    def test_sends_notification_and_updates_state(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            json={},
            status=200,
        )
        updated = main.send_daily_digest({}, 75, [], [], 90, 42, self._NOW)
        assert updated["last_digest_date"] == "2026-03-03"
        assert len(responses_lib.calls) == 1

    @responses_lib.activate
    def test_does_not_send_twice_same_day(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            json={},
            status=200,
        )
        state = main.send_daily_digest({}, 50, [], [], 30, 10, self._NOW)
        state = main.send_daily_digest(state, 50, [], [], 30, 10, self._NOW)
        assert len(responses_lib.calls) == 1

    def test_does_not_mutate_input_state_when_no_digest(self) -> None:
        off_hour = datetime.datetime(2026, 3, 3, 15, 0, tzinfo=datetime.timezone.utc)
        original: dict = {"some_key": "value"}
        main.send_daily_digest(original, 50, [], [], 30, 10, off_hour)
        assert "last_digest_date" not in original

    @responses_lib.activate
    def test_returns_new_dict_does_not_mutate_on_send(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            json={},
            status=200,
        )
        original: dict = {"some_key": "value"}
        updated = main.send_daily_digest(original, 50, [], [], 30, 10, self._NOW)
        assert "last_digest_date" not in original
        assert "last_digest_date" in updated

    @responses_lib.activate
    def test_digest_message_content_in_request(self) -> None:
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8123/api/services/notify/notify",
            json={},
            status=200,
        )
        main.send_daily_digest(
            {},
            85,
            [{"description": "Kitchen surge detected"}],
            [{"title": "Turn off lights"}],
            45,
            30,
            self._NOW,
        )
        payload = json.loads(responses_lib.calls[0].request.body)
        assert "Daily Digest" in payload["title"]
        assert "85/100" in payload["message"]
        assert "Kitchen surge" in payload["message"]

    def test_returns_original_state_when_no_digest(self) -> None:
        off_hour = datetime.datetime(2026, 3, 3, 15, 0, tzinfo=datetime.timezone.utc)
        original: dict = {"key": "val"}
        result = main.send_daily_digest(original, 50, [], [], 30, 10, off_hour)
        assert result is original
