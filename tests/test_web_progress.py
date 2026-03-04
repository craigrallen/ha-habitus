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
    assert "`[${stamp}] idle`" not in web.PAGE
