from __future__ import annotations

import importlib

from habitus.habitus import main as main_module


def test_invalid_integer_env_values_fallback_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("HABITUS_ANOMALY_THRESHOLD", "not-an-int")
    monkeypatch.setenv("HABITUS_DAILY_DIGEST_HOUR", "oops")
    monkeypatch.setenv("HABITUS_MIN_SCORING_DAYS", "bad")
    monkeypatch.setenv("HABITUS_FETCH_ROW_BUDGET", "nope")
    monkeypatch.setenv("HABITUS_FETCH_MIN_WINDOW_DAYS", "nah")

    mod = importlib.reload(main_module)

    assert mod.THRESHOLD == 70
    assert mod.DAILY_DIGEST_HOUR == 8
    assert mod.MIN_SCORING_DAYS == 7
    assert mod.FETCH_ROW_BUDGET == 1_000_000
    assert mod.FETCH_MIN_WINDOW_DAYS == 7
    assert mod._fetch_row_budget() == 1_000_000
    assert mod._fetch_min_window_days() == 7
