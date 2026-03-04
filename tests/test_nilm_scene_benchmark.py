from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "nilm_scene_benchmark.py"
    spec = importlib.util.spec_from_file_location("nilm_scene_benchmark", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_failures_absolute_limits() -> None:
    bench = _load_benchmark_module()
    summary = {
        "nilm_ms": {"p95": 12.0},
        "scene_ms": {"p95": 25.0},
    }

    failed = bench._threshold_failures(
        summary,
        max_nilm_ms=10.0,
        max_scene_ms=20.0,
        baseline=None,
        regression_tolerance_pct=15.0,
    )

    assert any("NILM p95" in line for line in failed)
    assert any("Scene p95" in line for line in failed)


def test_threshold_failures_regression_gate() -> None:
    bench = _load_benchmark_module()
    summary = {
        "nilm_ms": {"p95": 9.5},
        "scene_ms": {"p95": 33.0},
    }
    baseline = {
        "nilm_ms": {"p95": 8.0},
        "scene_ms": {"p95": 30.0},
    }

    failed = bench._threshold_failures(
        summary,
        max_nilm_ms=100.0,
        max_scene_ms=100.0,
        baseline=baseline,
        regression_tolerance_pct=10.0,
    )

    assert any("NILM regression" in line for line in failed)
    assert not any("Scene regression" in line for line in failed)


def test_print_markdown_includes_regression_gate_info() -> None:
    bench = _load_benchmark_module()
    summary = {
        "iterations": 5,
        "nilm_ms": {"mean": 3.0, "p95": 4.0, "max": 4.5},
        "scene_ms": {"mean": 10.0, "p95": 11.0, "max": 11.2},
        "last_run": {
            "nilm": {"edges": 1, "paired_events": 1, "clusters": 1},
            "scene": {"pair_groups": 1, "scenes": 1},
        },
    }
    baseline = {
        "nilm_ms": {"p95": 8.0},
        "scene_ms": {"p95": 30.0},
    }

    md = bench._print_markdown(summary, baseline=baseline, regression_tolerance_pct=15.0)
    assert "NILM regression gate" in md
    assert "Scene regression gate" in md
