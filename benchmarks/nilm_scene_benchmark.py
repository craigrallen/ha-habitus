#!/usr/bin/env python3
"""Synthetic performance benchmark for NILM and scene analytics pipelines."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from habitus.habitus.nilm_disaggregator import (
    _cluster_events,
    _detect_edges,
    _estimate_current_breakdown,
    _match_to_appliances,
    _pair_edges,
)
from habitus.habitus.scene_detector import (
    _analyze_time_patterns,
    _cluster_pairs_to_scenes,
    _find_co_occurrences,
)


def _build_nilm_readings(samples: int = 7200) -> list[tuple[float, float]]:
    rng = random.Random(42)
    start_ts = 1_700_000_000.0
    readings: list[tuple[float, float]] = []
    for idx in range(samples):
        t = start_ts + idx
        base = 260.0 + 15.0 * math.sin(idx / 200.0)
        kettle = 2000.0 if 600 <= (idx % 1800) <= 780 else 0.0
        hvac = 900.0 if (idx // 900) % 2 else 0.0
        fridge = 120.0 if (idx % 420) < 150 else 0.0
        noise = rng.uniform(-10.0, 10.0)
        readings.append((t, max(0.0, base + kettle + hvac + fridge + noise)))
    return readings


def _build_scene_changes(events: int = 8000) -> list[dict[str, Any]]:
    rng = random.Random(7)
    start_ts = 1_700_000_000.0
    entities = [
        "light.kitchen_main",
        "light.lounge_lamp",
        "switch.tv_power",
        "media_player.lounge_tv",
        "climate.living_room",
        "binary_sensor.hall_motion",
        "person.craig",
    ]
    states = ["on", "playing", "home", "heat", "auto"]
    changes: list[dict[str, Any]] = []
    for idx in range(events):
        ts = start_ts + idx * 4
        scene_bias = idx % 18 in (0, 1, 2, 3, 4)
        if scene_bias:
            eid = rng.choice(entities[:5])
            state = rng.choice(states[:3])
        else:
            eid = rng.choice(entities)
            state = rng.choice(states)
        changes.append({"entity_id": eid, "state": state, "timestamp": ts})
    return changes


def _time_call(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


def _run_once(readings: list[tuple[float, float]], changes: list[dict[str, Any]]) -> dict[str, Any]:
    edges, t_edges = _time_call(_detect_edges, readings)
    paired, t_pair = _time_call(_pair_edges, edges)
    clusters, t_cluster = _time_call(_cluster_events, paired)
    matched, t_match = _time_call(_match_to_appliances, clusters)
    _, t_breakdown = _time_call(_estimate_current_breakdown, readings, matched)

    co_occ, t_co = _time_call(_find_co_occurrences, changes)
    scenes, t_scene_cluster = _time_call(_cluster_pairs_to_scenes, co_occ)
    _, t_pattern = _time_call(
        _analyze_time_patterns,
        scenes[0] if scenes else {"light.kitchen_main", "switch.tv_power"},
        changes,
    )

    return {
        "nilm_ms": t_edges + t_pair + t_cluster + t_match + t_breakdown,
        "scene_ms": t_co + t_scene_cluster + t_pattern,
        "nilm": {
            "edges": len(edges),
            "paired_events": len(paired),
            "clusters": len(clusters),
            "matched": len(matched),
        },
        "scene": {
            "pair_groups": len(co_occ),
            "scenes": len(scenes),
        },
        "stages_ms": {
            "detect_edges": round(t_edges, 2),
            "pair_edges": round(t_pair, 2),
            "cluster_events": round(t_cluster, 2),
            "match_appliances": round(t_match, 2),
            "estimate_breakdown": round(t_breakdown, 2),
            "find_co_occurrences": round(t_co, 2),
            "cluster_scenes": round(t_scene_cluster, 2),
            "analyze_pattern": round(t_pattern, 2),
        },
    }


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    nilm = [r["nilm_ms"] for r in runs]
    scene = [r["scene_ms"] for r in runs]
    return {
        "iterations": len(runs),
        "nilm_ms": {
            "mean": round(statistics.mean(nilm), 2),
            "p95": round(sorted(nilm)[max(0, int(len(nilm) * 0.95) - 1)], 2),
            "max": round(max(nilm), 2),
        },
        "scene_ms": {
            "mean": round(statistics.mean(scene), 2),
            "p95": round(sorted(scene)[max(0, int(len(scene) * 0.95) - 1)], 2),
            "max": round(max(scene), 2),
        },
        "last_run": runs[-1],
    }


def _print_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## NILM + Scene benchmark",
            "",
            f"- Iterations: {summary['iterations']}",
            (
                "- NILM ms (mean/p95/max): "
                f"{summary['nilm_ms']['mean']}/{summary['nilm_ms']['p95']}/{summary['nilm_ms']['max']}"
            ),
            (
                "- Scene ms (mean/p95/max): "
                f"{summary['scene_ms']['mean']}/{summary['scene_ms']['p95']}/{summary['scene_ms']['max']}"
            ),
            f"- Last run NILM edges/paired/clusters: "
            f"{summary['last_run']['nilm']['edges']}/{summary['last_run']['nilm']['paired_events']}/{summary['last_run']['nilm']['clusters']}",
            f"- Last run scene pair-groups/scenes: "
            f"{summary['last_run']['scene']['pair_groups']}/{summary['last_run']['scene']['scenes']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json-out", default="benchmark-results/nilm_scene_benchmark.json")
    parser.add_argument("--summary-out", default="benchmark-results/nilm_scene_benchmark.md")
    parser.add_argument("--max-nilm-ms", type=float, default=1800.0)
    parser.add_argument("--max-scene-ms", type=float, default=1200.0)
    args = parser.parse_args()

    readings = _build_nilm_readings()
    changes = _build_scene_changes()

    # Warmup pass
    _run_once(readings, changes)

    runs = [_run_once(readings, changes) for _ in range(args.iterations)]
    summary = _summarize(runs)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = _print_markdown(summary)
    Path(args.summary_out).write_text(markdown, encoding="utf-8")
    print(markdown)

    failed = []
    if summary["nilm_ms"]["p95"] > args.max_nilm_ms:
        failed.append(f"NILM p95 {summary['nilm_ms']['p95']}ms > {args.max_nilm_ms}ms")
    if summary["scene_ms"]["p95"] > args.max_scene_ms:
        failed.append(f"Scene p95 {summary['scene_ms']['p95']}ms > {args.max_scene_ms}ms")

    if failed:
        for line in failed:
            print(f"FAIL: {line}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
