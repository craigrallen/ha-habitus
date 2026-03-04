from __future__ import annotations

from habitus.habitus import main


def test_mark_last_completed_progress_persists_explicit_contract() -> None:
    state: dict = {}

    main.mark_last_completed_progress(
        state,
        "training",
        done=20,
        total=20,
        rows=12345,
        pct=100,
        extra={"training_days": 30},
        completed_at="2026-03-04T23:59:00+00:00",
    )

    checkpoint = state["last_completed_progress"]
    assert checkpoint["phase"] == "training"
    assert checkpoint["completed_at"] == "2026-03-04T23:59:00+00:00"
    assert checkpoint["done"] == 20
    assert checkpoint["total"] == 20
    assert checkpoint["rows"] == 12345
    assert checkpoint["pct"] == 100
    assert checkpoint["extra"]["training_days"] == 30


def test_mark_last_completed_progress_clamps_invalid_values() -> None:
    state: dict = {}

    main.mark_last_completed_progress(
        state,
        "fetching",
        done=-9,
        total=-3,
        rows=-11,
        pct=170,
        extra={"window_days": 60},
    )

    checkpoint = state["last_completed_progress"]
    assert checkpoint["phase"] == "fetching"
    assert checkpoint["done"] == 0
    assert checkpoint["total"] == 0
    assert checkpoint["rows"] == 0
    assert checkpoint["pct"] == 100
    assert checkpoint["extra"]["window_days"] == 60
