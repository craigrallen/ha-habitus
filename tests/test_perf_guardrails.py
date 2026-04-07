from __future__ import annotations

import logging

from habitus.habitus import main


def test_summarize_perf_guardrail_computes_rates_and_budget() -> None:
    summary = main.summarize_perf_guardrail(
        "fetch_sqlite",
        2.5,
        rows=500,
        entities=10,
        warn_ms=2000,
    )

    assert summary["stage"] == "fetch_sqlite"
    assert summary["elapsed_ms"] == 2500
    assert summary["rows"] == 500
    assert summary["entities"] == 10
    assert summary["rows_per_sec"] == 200.0
    assert summary["rows_per_entity"] == 50.0
    assert summary["warn_exceeded"] is True


def test_log_perf_guardrail_emits_warning_when_threshold_exceeded(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="habitus"):
        summary = main.log_perf_guardrail(
            "build_features",
            1.5,
            rows=150,
            entities=5,
            warn_ms=1000,
        )

    assert summary["warn_exceeded"] is True
    assert "Perf[build_features]: 1500ms" in caplog.text
    assert "guardrail 1000ms exceeded" in caplog.text
