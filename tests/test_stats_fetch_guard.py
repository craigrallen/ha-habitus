from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from habitus.habitus import main


def test_clamp_fetch_window_by_row_budget_clamps_deterministically() -> None:
    start_iso = "2026-01-01T00:00:00+00:00"
    end_iso = "2026-02-01T00:00:00+00:00"

    clamped_start, was_clamped, info = main.clamp_fetch_window_by_row_budget(
        start_iso,
        end_iso,
        entity_count=10,
        row_budget=1000,
        min_window_days=7,
    )

    assert was_clamped is True
    assert info["requested_hours"] == 31 * 24
    assert info["max_hours"] == 7 * 24
    assert clamped_start == "2026-01-25T00:00:00+00:00"


def test_clamp_fetch_window_by_row_budget_disabled() -> None:
    start_iso = "2026-01-01T00:00:00+00:00"
    end_iso = "2026-02-01T00:00:00+00:00"

    clamped_start, was_clamped, _ = main.clamp_fetch_window_by_row_budget(
        start_iso,
        end_iso,
        entity_count=100,
        row_budget=0,
        min_window_days=1,
    )

    assert was_clamped is False
    assert clamped_start == start_iso


def test_fetch_stats_sqlite_respects_row_budget(monkeypatch, tmp_path) -> None:
    db = tmp_path / "ha_stats.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT)")
    conn.execute(
        "CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, mean REAL, sum REAL)"
    )
    conn.execute("INSERT INTO statistics_meta (id, statistic_id) VALUES (1, 'sensor.a')")
    conn.execute("INSERT INTO statistics_meta (id, statistic_id) VALUES (2, 'sensor.b')")

    base = int(dt.datetime(2026, 1, 1, tzinfo=dt.UTC).timestamp())
    for hour in range(120):
        ts = base + hour * 3600
        conn.execute(
            "INSERT INTO statistics (metadata_id, start_ts, mean, sum) VALUES (1, ?, ?, ?)",
            (ts, float(hour), None),
        )
        conn.execute(
            "INSERT INTO statistics (metadata_id, start_ts, mean, sum) VALUES (2, ?, ?, ?)",
            (ts, float(hour), None),
        )

    conn.commit()
    conn.close()

    monkeypatch.setenv("HABITUS_HA_DB", str(db))
    monkeypatch.setenv("HABITUS_FETCH_ROW_BUDGET", "50")
    monkeypatch.setenv("HABITUS_FETCH_MIN_WINDOW_DAYS", "1")

    df = main.fetch_stats_sqlite(
        ["sensor.a", "sensor.b"],
        "2026-01-01T00:00:00+00:00",
        "2026-01-06T00:00:00+00:00",
    )

    assert len(df) == 50
    counts = df["entity_id"].value_counts().to_dict()
    assert counts.get("sensor.a") == 25
    assert counts.get("sensor.b") == 25


@pytest.mark.asyncio
async def test_fetch_stats_api_respects_row_budget(monkeypatch) -> None:
    class FakeWS:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        async def recv(self) -> str:
            points_a = [{"start": 1_700_000_000 + i * 3600, "mean": i, "sum": None} for i in range(20)]
            points_b = [{"start": 1_700_000_000 + i * 3600, "mean": i, "sum": None} for i in range(20)]
            return json.dumps({"result": {"sensor.b": points_b, "sensor.a": points_a}})

        async def close(self) -> None:
            return None

    fake_ws = FakeWS()

    async def _fake_ws_connect():
        return fake_ws

    monkeypatch.setenv("HABITUS_FORCE_API", "true")
    monkeypatch.setenv("HABITUS_FETCH_ROW_BUDGET", "10")
    monkeypatch.setenv("HABITUS_FETCH_MIN_WINDOW_DAYS", "1")
    monkeypatch.setattr(main, "ws_connect", _fake_ws_connect)

    df = await main.fetch_stats(
        ["sensor.a", "sensor.b"],
        "2025-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00+00:00",
    )

    assert len(df) == 10
    assert all(eid == "sensor.a" for eid in df["entity_id"].tolist())

    sent_start_time = fake_ws.sent[0]["start_time"]
    assert sent_start_time == "2025-12-31T00:00:00+00:00"
