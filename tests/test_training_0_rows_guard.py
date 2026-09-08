"""Regression tests: 0-rows fetch guard — training must not silently complete with empty data.

These tests verify that:
1. When fetch returns 0 rows, run() writes fetch_failed to progress.json (not complete/idle).
2. When entity lists are empty, run() writes fetch_failed to progress.json.
3. The stale progress.json guard clears stale locks and allows a fresh run.
4. write_fetch_failed writes the correct structure.
5. set_progress preserves started_at across ticks.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

import habitus.habitus.main as main_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_progress(data_dir: Path) -> dict:
    p = data_dir / "progress.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _stub_run_prerequisites(monkeypatch: pytest.MonkeyPatch, tmp_data_dir: Path,
                             stat_ids: list[str], behavioral_ids: list[str]) -> None:
    """Patch all network-touching functions so run() can be tested in isolation."""
    # Prevent energy entity auto-detection from hitting the network
    monkeypatch.setenv("HABITUS_POWER_ENTITY", "sensor.fake_power")
    monkeypatch.setenv("HABITUS_ENERGY_GRID", "sensor.fake_grid")

    monkeypatch.setattr(main_mod, "get_stat_ids", AsyncMock(return_value=(stat_ids, len(stat_ids))))
    monkeypatch.setattr(main_mod, "get_behavioral_entity_ids", AsyncMock(return_value=behavioral_ids))
    monkeypatch.setattr(main_mod, "get_ha_entity_count", MagicMock(return_value=len(stat_ids)))
    monkeypatch.setattr(main_mod, "send_notification", MagicMock())
    monkeypatch.setattr(main_mod, "should_retrain_for_tier_change", MagicMock(return_value=False))


# ---------------------------------------------------------------------------
# Test 1: fetch returns 0 rows → phase must be fetch_failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_writes_fetch_failed_when_fetch_returns_zero_rows(
    tmp_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If fetch_stats returns an empty DataFrame, progress.json must show fetch_failed."""
    progress_path = str(tmp_data_dir / "progress.json")

    _stub_run_prerequisites(monkeypatch, tmp_data_dir,
                            stat_ids=["sensor.a", "sensor.b"], behavioral_ids=[])

    # fetch_stats returns empty DataFrame → simulates DB unreachable / wrong path
    monkeypatch.setattr(main_mod, "fetch_stats", AsyncMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(main_mod, "fetch_recent_raw_history", MagicMock(return_value=pd.DataFrame()))

    with patch.object(main_mod, "PROGRESS_PATH", progress_path):
        with patch.object(main_mod, "DATA_DIR", str(tmp_data_dir)):
            with patch.object(main_mod, "STATE_PATH", str(tmp_data_dir / "run_state.json")):
                await main_mod.run(days_history=30, mode="full")

    progress = _read_progress(tmp_data_dir)
    assert progress.get("phase") == "fetch_failed", (
        f"Expected phase='fetch_failed' but got {progress.get('phase')!r}. "
        f"Full progress: {progress}"
    )
    assert progress.get("running") is False, (
        "progress.json must have running=false after fetch_failed"
    )


# ---------------------------------------------------------------------------
# Test 2: entity list empty → fetch_failed (not silent return)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_writes_fetch_failed_when_entity_list_is_empty(
    tmp_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If both stat_ids and behavioral_entity_ids are empty, write fetch_failed."""
    progress_path = str(tmp_data_dir / "progress.json")

    _stub_run_prerequisites(monkeypatch, tmp_data_dir, stat_ids=[], behavioral_ids=[])

    with patch.object(main_mod, "PROGRESS_PATH", progress_path):
        with patch.object(main_mod, "DATA_DIR", str(tmp_data_dir)):
            with patch.object(main_mod, "STATE_PATH", str(tmp_data_dir / "run_state.json")):
                await main_mod.run(days_history=30, mode="full")

    progress = _read_progress(tmp_data_dir)
    assert progress.get("phase") == "fetch_failed", (
        f"Expected phase='fetch_failed' when entity list is empty but got {progress.get('phase')!r}. "
        f"Full progress: {progress}"
    )
    assert progress.get("running") is False


# ---------------------------------------------------------------------------
# Test 3: stale progress.json is cleared and run proceeds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_progress_lock_is_cleared_and_run_proceeds(
    tmp_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale progress.json (>15 min old, running=true) must be cleared so the run starts."""
    progress_path = str(tmp_data_dir / "progress.json")

    # Write a stale progress.json — older than 15 minutes
    stale_started_at = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=20)
    ).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    p = tmp_data_dir / "progress.json"
    p.write_text(json.dumps({
        "running": True,
        "phase": "fetching",
        "started_at": stale_started_at,
    }))
    # Also backdate the file mtime
    old_mtime = time.time() - 25 * 60  # 25 minutes ago
    os.utime(p, (old_mtime, old_mtime))

    _stub_run_prerequisites(monkeypatch, tmp_data_dir,
                            stat_ids=["sensor.a"], behavioral_ids=[])
    # fetch returns empty so we get a clean fetch_failed exit (not a full train)
    monkeypatch.setattr(main_mod, "fetch_stats", AsyncMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(main_mod, "fetch_recent_raw_history", MagicMock(return_value=pd.DataFrame()))

    with patch.object(main_mod, "PROGRESS_PATH", progress_path):
        with patch.object(main_mod, "DATA_DIR", str(tmp_data_dir)):
            with patch.object(main_mod, "STATE_PATH", str(tmp_data_dir / "run_state.json")):
                await main_mod.run(days_history=30, mode="full")

    progress = _read_progress(tmp_data_dir)
    # The stale lock must have been cleared — the run was NOT stuck on the old lock
    # After the run: progress reflects the outcome (fetch_failed here since fetch is mocked empty)
    assert progress.get("phase") in ("fetch_failed", "stale_aborted", "idle", "complete"), (
        f"Unexpected phase after stale-lock clear: {progress.get('phase')!r}"
    )
    # The run must have proceeded past the stale lock (phase != original "fetching" with running=True)
    assert not (progress.get("phase") == "fetching" and progress.get("running") is True), (
        "Stale progress.json lock was NOT cleared — run was blocked by stale running=true state"
    )


# ---------------------------------------------------------------------------
# Test 4: write_fetch_failed writes correct structure
# ---------------------------------------------------------------------------

def test_write_fetch_failed_writes_correct_structure(
    tmp_data_dir: Path,
) -> None:
    """write_fetch_failed must write running=false, phase=fetch_failed, reason, failed_at."""
    progress_path = str(tmp_data_dir / "progress.json")

    with patch.object(main_mod, "PROGRESS_PATH", progress_path):
        main_mod.write_fetch_failed("test reason: 0 rows returned")

    progress = _read_progress(tmp_data_dir)
    assert progress["phase"] == "fetch_failed"
    assert progress["running"] is False
    assert "reason" in progress
    assert "test reason" in progress["reason"]
    assert "failed_at" in progress


# ---------------------------------------------------------------------------
# Test 5: set_progress preserves started_at across ticks
# ---------------------------------------------------------------------------

def test_set_progress_preserves_started_at(
    tmp_data_dir: Path,
) -> None:
    """started_at written on first set_progress call must survive subsequent calls."""
    progress_path = str(tmp_data_dir / "progress.json")

    with patch.object(main_mod, "PROGRESS_PATH", progress_path):
        # First call — started_at should be set fresh
        main_mod.set_progress("fetching", 0, 100, 0)
        progress1 = _read_progress(tmp_data_dir)
        assert "started_at" in progress1, "set_progress must write started_at"
        original_started_at = progress1["started_at"]

        # Second call — started_at must be preserved (not overwritten)
        main_mod.set_progress("fetching", 50, 100, 5000)
        progress2 = _read_progress(tmp_data_dir)
        assert progress2.get("started_at") == original_started_at, (
            "set_progress must preserve started_at across ticks within the same run"
        )
