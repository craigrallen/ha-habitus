"""Tests for Lovelace card registration logic and publish_dashboard_entities.

TASK-006 acceptance: Python-side registration tests; ruff + black + mypy pass.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import responses as responses_lib

from habitus.habitus import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_ha_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch HA connection globals for every test in this module."""
    monkeypatch.setattr(main, "HA_URL", "http://localhost:8123")
    monkeypatch.setattr(main, "HA_TOKEN", "test-token")
    monkeypatch.setattr(main, "HA_WS", "ws://localhost:8123/api/websocket")
    monkeypatch.setattr(main, "NOTIFY_ON", False)  # suppress notifications


def _make_ws(recv_seq: list[str]) -> AsyncMock:
    """Build a mock WebSocket that replays *recv_seq* for each recv() call."""
    ws = MagicMock()
    ws.recv = AsyncMock(side_effect=recv_seq)
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _mock_connect(ws: MagicMock) -> AsyncMock:
    """Return an AsyncMock for websockets.connect that resolves to *ws*."""
    return AsyncMock(return_value=ws)


# ---------------------------------------------------------------------------
# _register_lovelace_card
# ---------------------------------------------------------------------------


class TestRegisterLovelaceCard:
    """Tests for the async _register_lovelace_card() function."""

    @pytest.mark.asyncio
    async def test_registers_resource_when_not_present(self) -> None:
        """Resource must be created when it is absent from the HA resource list."""
        ws = _make_ws([
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            # lovelace/resources → empty list
            json.dumps({"id": 101, "type": "result", "success": True, "result": []}),
            # lovelace/resources/create → ok
            json.dumps({"id": 102, "type": "result", "success": True}),
            # lovelace/config → no views
            json.dumps({"id": 103, "type": "result", "success": True, "result": {}}),
        ])

        with patch("habitus.habitus.main.websockets.connect", _mock_connect(ws)):
            await main._register_lovelace_card()

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
        types = [m.get("type") for m in sent]
        assert "lovelace/resources/create" in types

    @pytest.mark.asyncio
    async def test_skips_resource_when_already_registered(self) -> None:
        """Resource must NOT be created when it already exists."""
        existing = [{"url": "/local/habitus/habitus-card.js?v=old", "res_type": "module"}]
        ws = _make_ws([
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({"id": 101, "type": "result", "success": True, "result": existing}),
            # lovelace/config
            json.dumps({"id": 102, "type": "result", "success": True, "result": {}}),
        ])

        with patch("habitus.habitus.main.websockets.connect", _mock_connect(ws)):
            await main._register_lovelace_card()

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
        types = [m.get("type") for m in sent]
        assert "lovelace/resources/create" not in types

    @pytest.mark.asyncio
    async def test_adds_card_to_dashboard_when_absent(self) -> None:
        """habitus-card must be inserted into view[0] when not already there."""
        dash_config = {"views": [{"title": "Home", "cards": [{"type": "weather-forecast"}]}]}
        ws = _make_ws([
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({"id": 101, "type": "result", "success": True, "result": []}),
            json.dumps({"id": 102, "type": "result", "success": True}),
            json.dumps({"id": 103, "type": "result", "success": True, "result": dash_config}),
            json.dumps({"id": 104, "type": "result", "success": True}),
        ])

        with patch("habitus.habitus.main.websockets.connect", _mock_connect(ws)):
            await main._register_lovelace_card()

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
        save_msg = next((m for m in sent if m.get("type") == "lovelace/config/save"), None)
        assert save_msg is not None
        cards = save_msg["config"]["views"][0]["cards"]
        assert any("habitus-card" in c.get("type", "") for c in cards)

    @pytest.mark.asyncio
    async def test_does_not_add_card_when_already_present(self) -> None:
        """habitus-card must NOT be inserted when it already exists in view[0]."""
        dash_config = {
            "views": [{"title": "Home", "cards": [{"type": "custom:habitus-card"}]}]
        }
        ws = _make_ws([
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({"id": 101, "type": "result", "success": True, "result": []}),
            json.dumps({"id": 102, "type": "result", "success": True}),
            json.dumps({"id": 103, "type": "result", "success": True, "result": dash_config}),
        ])

        with patch("habitus.habitus.main.websockets.connect", _mock_connect(ws)):
            await main._register_lovelace_card()

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
        types = [m.get("type") for m in sent]
        assert "lovelace/config/save" not in types

    @pytest.mark.asyncio
    async def test_aborts_on_auth_failure(self) -> None:
        """Should return silently when HA returns auth_invalid."""
        ws = _make_ws([
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_invalid"}),
        ])

        with patch("habitus.habitus.main.websockets.connect", _mock_connect(ws)):
            await main._register_lovelace_card()  # must not raise

        # Only auth message sent; no resource or config calls
        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
        types = [m.get("type") for m in sent]
        assert "lovelace/resources" not in types

    @pytest.mark.asyncio
    async def test_swallows_connection_error(self) -> None:
        """Should not raise when WebSocket connection fails."""
        with patch(
            "habitus.habitus.main.websockets.connect",
            AsyncMock(side_effect=ConnectionRefusedError("refused")),
        ):
            await main._register_lovelace_card()  # must not raise

    @pytest.mark.asyncio
    async def test_resource_url_contains_js_path(self) -> None:
        """Created resource URL must reference habitus-card.js."""
        ws = _make_ws([
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({"id": 101, "type": "result", "success": True, "result": []}),
            json.dumps({"id": 102, "type": "result", "success": True}),
            json.dumps({"id": 103, "type": "result", "success": True, "result": {}}),
        ])

        with patch("habitus.habitus.main.websockets.connect", _mock_connect(ws)):
            await main._register_lovelace_card()

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
        create_msg = next(
            (m for m in sent if m.get("type") == "lovelace/resources/create"), None
        )
        assert create_msg is not None
        assert "habitus-card.js" in create_msg.get("url", "")
        assert create_msg.get("res_type") == "module"


# ---------------------------------------------------------------------------
# publish_dashboard_entities
# ---------------------------------------------------------------------------


class TestPublishDashboardEntities:
    """Tests for publish_dashboard_entities()."""

    @responses_lib.activate
    def test_publishes_top_anomaly_sensor(self) -> None:
        """Publishes sensor.habitus_top_anomaly from entity_anomalies[0]."""
        self._mock_all_posts()
        anomalies = [
            {"description": "Kitchen power is 800W — baseline is 200W ±30W", "entity_id": "sensor.kitchen"}
        ]
        main.publish_dashboard_entities(50, anomalies, [])

        urls = [c.request.url for c in responses_lib.calls]
        assert any("sensor.habitus_top_anomaly" in u for u in urls)

    @responses_lib.activate
    def test_publishes_none_detected_when_no_anomalies(self) -> None:
        """Publishes 'None detected' when anomaly list is empty."""
        self._mock_all_posts()
        main.publish_dashboard_entities(10, [], [])

        bodies = [json.loads(c.request.body) for c in responses_lib.calls
                  if "habitus_top_anomaly" in c.request.url]
        assert bodies and bodies[0]["state"] == "None detected"

    @responses_lib.activate
    def test_publishes_up_to_3_suggestion_sensors(self) -> None:
        """Publishes suggestion_1 through suggestion_3 from suggestions list."""
        self._mock_all_posts()
        suggestions = [
            {"title": "Turn off lights at sunrise", "description": "...", "confidence": 85},
            {"title": "Peak tariff alert", "description": "...", "confidence": 78},
            {"title": "Vacancy security alert", "description": "...", "confidence": 65},
            {"title": "Extra suggestion", "description": "...", "confidence": 55},
        ]
        main.publish_dashboard_entities(30, [], suggestions)

        urls = [c.request.url for c in responses_lib.calls]
        assert any("habitus_suggestion_1" in u for u in urls)
        assert any("habitus_suggestion_2" in u for u in urls)
        assert any("habitus_suggestion_3" in u for u in urls)
        # 4th suggestion must NOT be published (only 3 max)
        assert not any("habitus_suggestion_4" in u for u in urls)

    @responses_lib.activate
    def test_creates_persistent_notification_on_high_score(self) -> None:
        """Sends a persistent notification when score >= 90."""
        self._mock_all_posts()
        anomalies = [{"description": "Extreme power spike", "entity_id": "sensor.load"}]
        main.publish_dashboard_entities(95, anomalies, [])

        urls = [c.request.url for c in responses_lib.calls]
        assert any("persistent_notification/create" in u for u in urls)

    @responses_lib.activate
    def test_dismisses_notification_on_normal_score(self) -> None:
        """Dismisses habitus_anomaly notification when score < 90."""
        self._mock_all_posts()
        main.publish_dashboard_entities(30, [], [])

        urls = [c.request.url for c in responses_lib.calls]
        assert any("persistent_notification/dismiss" in u for u in urls)

    @responses_lib.activate
    def test_suggestion_sensor_state_is_title(self) -> None:
        """Suggestion sensor state must be the suggestion title string."""
        self._mock_all_posts()
        suggestions = [{"title": "Morning lights automation", "description": "desc", "confidence": 80}]
        main.publish_dashboard_entities(20, [], suggestions)

        sug_calls = [c for c in responses_lib.calls if "habitus_suggestion_1" in c.request.url]
        assert sug_calls
        body = json.loads(sug_calls[0].request.body)
        assert body["state"] == "Morning lights automation"

    def _mock_all_posts(self) -> None:
        """Register catch-all POST and GET mocks for HA REST endpoints."""
        responses_lib.add(
            responses_lib.POST,
            responses_lib.matchers.re.compile(r"http://localhost:8123/api/.*"),
            json={"result": "ok"},
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            "http://localhost:8123/api/states",
            json=[],
            status=200,
        )
