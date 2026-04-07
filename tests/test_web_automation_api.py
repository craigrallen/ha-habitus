from __future__ import annotations

from habitus.habitus import web


def test_add_automation_rejects_invalid_yaml() -> None:
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": "not: [valid"})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "invalid YAML" in payload["error"]


def test_add_automation_requires_alias_and_structure() -> None:
    bad_yaml = "trigger: []\naction: []"
    resp = web.app.test_client().post("/api/add_automation", json={"yaml": bad_yaml})
    payload = resp.get_json()
    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "alias" in payload["error"]


def test_add_automation_normalizes_duplicate_alias(monkeypatch) -> None:
    calls: list[str] = []

    class _Resp:
        def __init__(self, status_code: int, body=None):
            self.status_code = status_code
            self._body = body or []
            self.text = ""

        def json(self):
            return self._body

    def _fake_get(url, headers, timeout):
        assert url.endswith("/api/states")
        return _Resp(200, [{"entity_id": "automation.habitus_evening_mode"}])

    def _fake_post(url, headers, json, timeout):
        calls.append(url)
        return _Resp(201)

    monkeypatch.setattr("requests.get", _fake_get)
    monkeypatch.setattr("requests.post", _fake_post)

    yaml_payload = (
        "automation:\n"
        "  alias: \"Habitus Evening Mode\"\n"
        "  trigger:\n"
        "    - platform: time\n"
        "      at: \"18:00:00\"\n"
        "  action:\n"
        "    - service: light.turn_on\n"
        "      target:\n"
        "        entity_id: light.living_room_ceiling\n"
    )

    resp = web.app.test_client().post("/api/add_automation", json={"yaml": yaml_payload})
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["automation_id"] == "habitus_evening_mode_2"
    assert calls and calls[0].endswith("/api/config/automation/config/habitus_evening_mode_2")


def test_remove_automation_normalizes_entity_id(monkeypatch) -> None:
    called = {"url": ""}

    class _Resp:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.text = ""

    def _fake_delete(url, headers, timeout):
        called["url"] = url
        return _Resp(204)

    monkeypatch.setattr("requests.delete", _fake_delete)

    resp = web.app.test_client().post(
        "/api/remove_automation",
        json={"entity_id": "automation.Habitus — Evening Mode"},
    )
    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["automation_id"] == "habitus_evening_mode"
    assert called["url"].endswith("/api/config/automation/config/habitus_evening_mode")


def test_suggestion_ui_renders_why_and_badges() -> None:
    assert "Why now:" in web.PAGE
    assert "confidence_rationale" in web.PAGE
    assert "status_badges" in web.PAGE
