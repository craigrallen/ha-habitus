"""Integration tests for Add/Remove HA automation flow.

Tests the full HTTP flow including HA REST API mocking, error handling,
alias normalization, and ID uniqueness logic.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from habitus.habitus import web


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_YAML = """\
automation:
  alias: "Habitus Morning Routine"
  trigger:
    - platform: time
      at: "07:00:00"
  condition: []
  action:
    - service: light.turn_on
      target:
        entity_id: light.bedroom_ceiling
"""

VALID_YAML_SIMPLE = """\
alias: "Test Automation Alpha"
trigger:
  - platform: state
    entity_id: binary_sensor.front_door
    to: "on"
action:
  - service: notify.notify
    data:
      message: "Front door opened"
"""


def _mock_get_states(monkeypatch, entities: list[dict] | None = None):
    """Mock requests.get to return a list of HA automation entities."""
    if entities is None:
        entities = []

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = entities
    mock_resp.text = ""
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_resp)
    return mock_resp


def _mock_post_ha(monkeypatch, status_code: int = 201):
    """Mock requests.post to simulate HA config endpoint."""
    posted = {"url": None, "body": None}
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = "ok" if status_code < 300 else f"Error {status_code}"

    def _fake_post(url, headers, json, timeout):
        posted["url"] = url
        posted["body"] = json
        return mock_resp

    monkeypatch.setattr("requests.post", _fake_post)
    return posted, mock_resp


def _mock_delete_ha(monkeypatch, status_code: int = 204):
    """Mock requests.delete to simulate HA config endpoint."""
    deleted = {"url": None}
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = "" if status_code < 300 else f"Not found: {status_code}"

    def _fake_delete(url, headers, timeout):
        deleted["url"] = url
        return mock_resp

    monkeypatch.setattr("requests.delete", _fake_delete)
    return deleted, mock_resp


# ---------------------------------------------------------------------------
# POST /api/add_automation — valid YAML
# ---------------------------------------------------------------------------


def test_add_automation_valid_yaml_calls_ha_create(monkeypatch) -> None:
    """POST valid YAML → HA create endpoint called, returns {ok: true}."""
    _mock_get_states(monkeypatch, [])
    posted, _ = _mock_post_ha(monkeypatch, status_code=201)

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": VALID_YAML})
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "automation_id" in payload
    assert payload["automation_id"] == "habitus_morning_routine"
    assert posted["url"] is not None
    assert "habitus_morning_routine" in posted["url"]
    assert posted["body"]["alias"] == "Habitus Morning Routine"


def test_add_automation_simple_yaml_no_wrapper(monkeypatch) -> None:
    """YAML without outer 'automation:' key still works."""
    _mock_get_states(monkeypatch, [])
    posted, _ = _mock_post_ha(monkeypatch, status_code=200)

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": VALID_YAML_SIMPLE})
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["automation_id"] == "test_automation_alpha"


# ---------------------------------------------------------------------------
# POST /api/add_automation — invalid/empty YAML
# ---------------------------------------------------------------------------


def test_add_automation_empty_yaml_returns_error() -> None:
    """Empty yaml → {ok: false, error: ...}."""
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": ""})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "error" in payload


def test_add_automation_missing_yaml_key_returns_error() -> None:
    """Missing yaml key in body → error."""
    resp = web.app.test_client().post("/api/add_automation", json={})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False


def test_add_automation_invalid_yaml_syntax_returns_error() -> None:
    """Malformed YAML → {ok: false, error: 'invalid YAML: ...'}."""
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": "trigger: [unclosed"})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "invalid YAML" in payload["error"]


def test_add_automation_yaml_without_alias_returns_error() -> None:
    """Valid YAML but missing alias → {ok: false, error: contains 'alias'}."""
    no_alias = "trigger:\n  - platform: time\n    at: '07:00'\naction: []\n"
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": no_alias})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "alias" in payload["error"]


def test_add_automation_yaml_empty_string_payload() -> None:
    """Whitespace-only yaml → error."""
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": "   "})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False


# ---------------------------------------------------------------------------
# POST /api/remove_automation — valid automation_id
# ---------------------------------------------------------------------------


def test_remove_automation_valid_id_calls_ha_delete(monkeypatch) -> None:
    """POST valid automation_id → HA delete endpoint called, returns {ok: true}."""
    deleted, _ = _mock_delete_ha(monkeypatch, status_code=204)

    resp = web.app.test_client().post(
        "/api/remove_automation",
        json={"entity_id": "automation.habitus_morning_routine"},
    )
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "habitus_morning_routine" in deleted["url"]


def test_remove_automation_via_alias(monkeypatch) -> None:
    """Can remove by alias field instead of entity_id."""
    deleted, _ = _mock_delete_ha(monkeypatch, status_code=204)

    resp = web.app.test_client().post(
        "/api/remove_automation",
        json={"alias": "Habitus Morning Routine"},
    )
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "habitus_morning_routine" in deleted["url"]


def test_remove_automation_missing_id_returns_error() -> None:
    """Neither entity_id nor alias → 400."""
    resp = web.app.test_client().post("/api/remove_automation", json={})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False


# ---------------------------------------------------------------------------
# HA returns non-200 — error surfacing
# ---------------------------------------------------------------------------


def test_add_automation_ha_returns_500_surfaces_error(monkeypatch) -> None:
    """When HA returns 500, Habitus returns {ok: false, error: 'HA 500: ...'}."""
    _mock_get_states(monkeypatch, [])
    mock_err = MagicMock()
    mock_err.status_code = 500
    mock_err.text = "Internal Server Error"

    def _fail_post(url, headers, json, timeout):
        return mock_err

    monkeypatch.setattr("requests.post", _fail_post)

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": VALID_YAML})
    payload = resp.get_json()

    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "500" in payload["error"]


def test_remove_automation_ha_returns_404_surfaces_error(monkeypatch) -> None:
    """HA 404 on delete → Habitus returns {ok: false, error: 'automation not found'}."""
    deleted, _ = _mock_delete_ha(monkeypatch, status_code=404)

    resp = web.app.test_client().post(
        "/api/remove_automation",
        json={"entity_id": "automation.nonexistent_thing"},
    )
    payload = resp.get_json()

    assert resp.status_code == 404
    assert payload["ok"] is False
    assert "not found" in payload["error"]


def test_remove_automation_ha_returns_500_surfaces_error(monkeypatch) -> None:
    """HA 500 on delete → Habitus returns error."""
    _, mock_resp = _mock_delete_ha(monkeypatch, status_code=500)

    resp = web.app.test_client().post(
        "/api/remove_automation",
        json={"entity_id": "automation.habitus_test"},
    )
    payload = resp.get_json()

    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "500" in payload["error"]


def test_add_automation_ha_connection_error_returns_500(monkeypatch) -> None:
    """Network error when calling HA → 500 response."""
    _mock_get_states(monkeypatch, [])

    def _connection_fail(url, headers, json, timeout):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr("requests.post", _connection_fail)

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": VALID_YAML})
    payload = resp.get_json()

    assert resp.status_code == 500
    assert payload["ok"] is False
    assert "failed to add automation" in payload["error"]


# ---------------------------------------------------------------------------
# Alias normalization and ID uniqueness
# ---------------------------------------------------------------------------


def test_alias_normalization_special_chars(monkeypatch) -> None:
    """Alias with special chars gets slugified to valid ID."""
    _mock_get_states(monkeypatch, [])
    posted, _ = _mock_post_ha(monkeypatch, status_code=201)

    special_yaml = """\
alias: "Habitus — Evening Mode (v2)!"
trigger:
  - platform: time
    at: "18:00:00"
action:
  - service: light.turn_on
    target:
      entity_id: light.living_room
"""
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": special_yaml})
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    # Should only contain lowercase alphanumerics and underscores
    import re
    assert re.match(r"^[a-z0-9_]+$", payload["automation_id"])


def test_alias_uniqueness_increments_when_collision(monkeypatch) -> None:
    """When automation_id already exists in HA, a suffix _2 is appended."""
    _mock_get_states(monkeypatch, [{"entity_id": "automation.habitus_morning_routine"}])
    posted, _ = _mock_post_ha(monkeypatch, status_code=201)

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": VALID_YAML})
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["automation_id"] == "habitus_morning_routine_2"


def test_alias_uniqueness_increments_triple_collision(monkeypatch) -> None:
    """When _2 also exists, increment to _3."""
    existing = [
        {"entity_id": "automation.habitus_morning_routine"},
        {"entity_id": "automation.habitus_morning_routine_2"},
    ]
    _mock_get_states(monkeypatch, existing)
    posted, _ = _mock_post_ha(monkeypatch, status_code=201)

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": VALID_YAML})
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["automation_id"] == "habitus_morning_routine_3"


def test_normalize_automation_id_strips_automation_prefix() -> None:
    """_normalize_automation_id strips 'automation.' prefix."""
    from habitus.habitus.web import _normalize_automation_id
    assert _normalize_automation_id("automation.my_test") == "my_test"
    assert _normalize_automation_id("automation.Habitus Evening Mode") == "habitus_evening_mode"
    assert _normalize_automation_id("My Automation") == "my_automation"
    assert _normalize_automation_id("") == ""


def test_unique_alias_id_no_collision() -> None:
    """_unique_alias_id returns base slug when no collision."""
    from habitus.habitus.web import _unique_alias_id
    result = _unique_alias_id("Evening Mode", set())
    assert result == "evening_mode"


def test_unique_alias_id_with_collision() -> None:
    """_unique_alias_id increments suffix on collision."""
    from habitus.habitus.web import _unique_alias_id
    existing = {"evening_mode", "evening_mode_2"}
    result = _unique_alias_id("Evening Mode", existing)
    assert result == "evening_mode_3"


def test_remove_automation_normalizes_unicode_dash(monkeypatch) -> None:
    """Entity IDs with em-dashes and special chars get normalized."""
    deleted, _ = _mock_delete_ha(monkeypatch, status_code=204)

    resp = web.app.test_client().post(
        "/api/remove_automation",
        json={"entity_id": "automation.Habitus — Evening Mode"},
    )
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["automation_id"] == "habitus_evening_mode"
