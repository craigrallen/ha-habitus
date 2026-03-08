"""Tests for per-phase NILM edge detection and correlation."""
from __future__ import annotations

import datetime

import pytest

from habitus.habitus.nilm_disaggregator import correlate_phase_edges, _make_phase_label


def _dt(offset_sec: float) -> datetime.datetime:
    """Helper: datetime relative to a fixed epoch for test reproducibility."""
    base = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    return base + datetime.timedelta(seconds=offset_sec)


class TestMakePhaseLabel:
    def test_single(self):
        assert _make_phase_label("single", ["L1"]) == "L1 (single phase)"

    def test_two_phase_400v(self):
        assert _make_phase_label("two_phase_400v", ["L1", "L2"]) == "L1+L2 (400V two-phase)"

    def test_two_phase_mixed(self):
        assert _make_phase_label("two_phase_mixed", ["L2", "L3"]) == "L2+L3 (mixed)"

    def test_three_phase(self):
        assert _make_phase_label("three_phase", ["L1", "L2", "L3"]) == "L1+L2+L3 (three-phase)"

    def test_empty_phases(self):
        # Should not raise; falls back gracefully
        label = _make_phase_label("single", [])
        assert "single" in label


class TestSinglePhaseDetection:
    def test_single_phase_detection(self):
        """Edge on L1 only → phase_type='single'."""
        edges_by_phase = {
            "L1": [(_dt(0), 1200.0)],
            "L2": [],
            "L3": [],
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        grp = groups[0]
        assert grp["phase_type"] == "single"
        assert grp["phases"] == ["L1"]
        assert grp["total_delta_w"] == pytest.approx(1200.0)

    def test_single_phase_l3(self):
        """Edge on L3 only → phase_type='single', phases=['L3']."""
        edges_by_phase = {
            "L1": [],
            "L2": [],
            "L3": [(_dt(0), 800.0)],
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        assert groups[0]["phase_type"] == "single"
        assert groups[0]["phases"] == ["L3"]


class TestTwoPhase400vDetection:
    def test_two_phase_400v_detection(self):
        """Simultaneous equal edges on L1+L2 → phase_type='two_phase_400v'."""
        edges_by_phase = {
            "L1": [(_dt(0), 1000.0)],
            "L2": [(_dt(5), 1000.0)],  # 5 seconds later, same magnitude
            "L3": [],
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        # Should be merged into one group
        assert len(groups) == 1
        grp = groups[0]
        assert grp["phase_type"] == "two_phase_400v"
        assert sorted(grp["phases"]) == ["L1", "L2"]
        assert grp["total_delta_w"] == pytest.approx(2000.0)

    def test_two_phase_near_equal_tolerance(self):
        """Edges within ±15% magnitude still classified as two_phase_400v."""
        edges_by_phase = {
            "L1": [(_dt(0), 1000.0)],
            "L2": [(_dt(10), 920.0)],  # ratio = 0.92 > 0.85 threshold
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        assert groups[0]["phase_type"] == "two_phase_400v"


class TestUnqualTwoPhase:
    def test_unequal_two_phase(self):
        """Edges on L1+L2 with very different magnitudes → phase_type='two_phase_mixed'."""
        edges_by_phase = {
            "L1": [(_dt(0), 2000.0)],
            "L2": [(_dt(15), 500.0)],  # ratio = 0.25 < 0.85 threshold
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        grp = groups[0]
        assert grp["phase_type"] == "two_phase_mixed"
        assert sorted(grp["phases"]) == ["L1", "L2"]


class TestThreePhaseDetection:
    def test_three_phase_detection(self):
        """Edges on all 3 phases within window → phase_type='three_phase'."""
        edges_by_phase = {
            "L1": [(_dt(0), 1500.0)],
            "L2": [(_dt(20), 1500.0)],
            "L3": [(_dt(40), 1500.0)],
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        grp = groups[0]
        assert grp["phase_type"] == "three_phase"
        assert sorted(grp["phases"]) == ["L1", "L2", "L3"]
        assert grp["total_delta_w"] == pytest.approx(4500.0)


class TestPhaseCorrelationWindow:
    def test_edges_outside_window_not_correlated(self):
        """Edges >120s apart on different phases must NOT be correlated."""
        edges_by_phase = {
            "L1": [(_dt(0), 1000.0)],
            "L2": [(_dt(200), 1000.0)],  # 200s > 120s window
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        # Should produce two separate single-phase groups
        assert len(groups) == 2
        phase_types = {g["phase_type"] for g in groups}
        assert phase_types == {"single"}

    def test_edges_exactly_at_window_boundary_correlated(self):
        """Edges exactly at window boundary (120s) ARE correlated (≤ window)."""
        edges_by_phase = {
            "L1": [(_dt(0), 1000.0)],
            "L2": [(_dt(120), 1000.0)],  # exactly at boundary
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        assert groups[0]["phase_type"] == "two_phase_400v"

    def test_empty_phases_no_error(self):
        """Empty input produces no groups and no exception."""
        groups = correlate_phase_edges({"L1": [], "L2": [], "L3": []}, window_sec=120)
        assert groups == []

    def test_multiple_independent_events(self):
        """Multiple well-separated single-phase events stay independent."""
        edges_by_phase = {
            "L1": [(_dt(0), 1000.0), (_dt(3600), 2000.0)],
            "L2": [],
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 2
        for g in groups:
            assert g["phase_type"] == "single"
            assert g["phases"] == ["L1"]

    def test_per_phase_dict_content(self):
        """per_phase dict correctly maps phase → delta_w."""
        edges_by_phase = {
            "L1": [(_dt(0), 1200.0)],
            "L2": [(_dt(30), 1100.0)],
        }
        groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        assert len(groups) == 1
        pp = groups[0]["per_phase"]
        assert "L1" in pp
        assert "L2" in pp
        assert pp["L1"] == pytest.approx(1200.0)
        assert pp["L2"] == pytest.approx(1100.0)
