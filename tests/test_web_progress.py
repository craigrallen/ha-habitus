from __future__ import annotations

import json
import os
import time
from pathlib import Path

from habitus.habitus import web


def test_api_progress_stale_recovery_preserves_recent_metrics_and_adds_last_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    state_path = tmp_path / "run_state.json"

    progress_path.write_text(
        json.dumps(
            {
                "running": True,
                "phase": "fetching",
                "pct": 42,
                "done": 12,
                "total": 40,
                "rows": 9876,
            }
        )
    )
    last_run = "2026-03-04T23:58:00+00:00"
    state_path.write_text(json.dumps({"last_run": last_run}))

    stale_seconds = 10
    old = time.time() - (stale_seconds + 5)
    os.utime(progress_path, (old, old))

    monkeypatch.setenv("HABITUS_PROGRESS_STALE_SEC", str(stale_seconds))
    monkeypatch.setattr(web, "PROGRESS_PATH", str(progress_path))
    monkeypatch.setattr(web, "STATE_PATH", str(state_path))

    client = web.app.test_client()
    resp = client.get("/api/progress")
    assert resp.status_code == 200

    payload = resp.get_json()
    assert payload["running"] is False
    assert payload["phase"] == "idle"
    assert payload["stale_recovered"] is True
    assert payload["rows"] == 9876
    assert payload["done"] == 12
    assert payload["total"] == 40
    assert payload["last_run"] == last_run


def test_training_log_js_uses_last_run_complete_state() -> None:
    assert "statusEl.textContent = 'Last run complete';" in web.PAGE
    assert "const completedPhase = (lastCompleted && lastCompleted.phase) || 'complete';" in web.PAGE
    assert "last run complete · ${completedPhase}" in web.PAGE
    assert "`[${stamp}] idle`" not in web.PAGE


def test_api_progress_normalizes_idle_payload_from_state_when_file_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    state_path = tmp_path / "run_state.json"

    state_path.write_text(
        json.dumps(
            {
                "last_run": "2026-03-04T23:58:00+00:00",
                "last_completed_progress": {
                    "phase": "training",
                    "completed_at": "2026-03-04T23:57:00+00:00",
                    "rows": 1234,
                },
            }
        )
    )

    monkeypatch.setattr(web, "PROGRESS_PATH", str(progress_path))
    monkeypatch.setattr(web, "STATE_PATH", str(state_path))

    payload = web.app.test_client().get("/api/progress").get_json()

    assert payload["running"] is False
    assert payload["phase"] == "idle"
    assert payload["pct"] == 100
    assert payload["done"] == 0
    assert payload["total"] == 0
    assert payload["rows"] == 0
    assert payload["last_run"] == "2026-03-04T23:58:00+00:00"
    assert payload["last_completed_progress"]["phase"] == "training"


def test_api_progress_recovers_when_progress_says_running_but_trainer_isnt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    state_path = tmp_path / "run_state.json"

    progress_path.write_text(
        json.dumps(
            {
                "running": True,
                "phase": "fetching",
                "pct": 100,
                "done": 891,
                "total": 891,
                "rows": 1798126,
            }
        )
    )
    state_path.write_text("{}")

    old = time.time() - 10
    os.utime(progress_path, (old, old))

    monkeypatch.setenv("HABITUS_PROGRESS_DEAD_GRACE_SEC", "1")
    monkeypatch.setattr(web, "PROGRESS_PATH", str(progress_path))
    monkeypatch.setattr(web, "STATE_PATH", str(state_path))
    monkeypatch.setattr(web._trainer, "is_running", lambda: False)

    payload = web.app.test_client().get("/api/progress").get_json()

    assert payload["running"] is False
    assert payload["phase"] == "idle"
    assert payload["stale_recovered"] is True


def test_api_progress_does_not_recover_immediately_when_file_is_fresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    state_path = tmp_path / "run_state.json"

    progress_path.write_text(
        json.dumps(
            {
                "running": True,
                "phase": "fetching",
                "pct": 30,
                "done": 100,
                "total": 300,
                "rows": 12000,
            }
        )
    )
    state_path.write_text("{}")

    monkeypatch.setenv("HABITUS_PROGRESS_DEAD_GRACE_SEC", "120")
    monkeypatch.setattr(web, "PROGRESS_PATH", str(progress_path))
    monkeypatch.setattr(web, "STATE_PATH", str(state_path))
    monkeypatch.setattr(web._trainer, "is_running", lambda: False)

    payload = web.app.test_client().get("/api/progress").get_json()

    assert payload["running"] is True
    assert payload["phase"] == "fetching"


def test_api_full_train_returns_immediately_with_background_thread(monkeypatch) -> None:
    started = {"count": 0}

    class _DummyThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            started["count"] += 1

    monkeypatch.setenv("HABITUS_DAYS", "90")
    monkeypatch.setattr("threading.Thread", _DummyThread)

    client = web.app.test_client()
    resp = client.post("/api/full_train")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "started" in payload["message"]
    assert started["count"] == 1


def test_api_progress_normalizes_running_payload_and_clamps_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    state_path = tmp_path / "run_state.json"

    progress_path.write_text(
        json.dumps(
            {
                "running": True,
                "phase": "idle",
                "pct": 170,
                "done": "3",
                "total": "2",
                "rows": "-8",
                "elapsed_min": "1.5",
                "eta_min": "oops",
            }
        )
    )
    state_path.write_text("{}")

    monkeypatch.setattr(web, "PROGRESS_PATH", str(progress_path))
    monkeypatch.setattr(web, "STATE_PATH", str(state_path))
    monkeypatch.setattr(web._trainer, "is_running", lambda: True)

    payload = web.app.test_client().get("/api/progress").get_json()

    assert payload["running"] is True
    assert payload["phase"] == "fetching"
    assert payload["pct"] == 100
    assert payload["done"] == 2
    assert payload["total"] == 2
    assert payload["rows"] == 0
    assert payload["elapsed_min"] == 1.5
    assert payload["eta_min"] == 0.0
