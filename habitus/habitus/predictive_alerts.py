"""Predictive anomaly alerts — trend extrapolation from recent entity readings."""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import numpy as np

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PREDICTIVE_ALERTS_PATH = os.path.join(DATA_DIR, "predictive_alerts.json")

# Sigma threshold for flagging a projected value as anomalous
SIGMA_THRESHOLD: float = 2.5

# Minimum readings required for trend fitting
MIN_READINGS: int = 4

# Extrapolation horizons in minutes
HORIZONS_MIN: list[int] = [30, 60]


def _fit_linear_trend(values: list[float]) -> tuple[float, float]:
    """Fit a simple linear trend (slope, intercept) to equally-spaced values.

    Args:
        values: Equally-spaced numeric readings (oldest first).

    Returns:
        Tuple of (slope_per_step, intercept).
    """
    n = len(values)
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    # Use least-squares via polyfit (degree 1)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def _extrapolate(slope: float, intercept: float, n_points: int, steps_ahead: float) -> float:
    """Extrapolate from the fitted trend line.

    Args:
        slope: Slope per step from the linear fit.
        intercept: Intercept of the linear fit.
        n_points: Number of data points used for the fit.
        steps_ahead: Number of steps into the future to extrapolate.

    Returns:
        Projected value at the future time step.
    """
    future_x = (n_points - 1) + steps_ahead
    return slope * future_x + intercept


def _severity_label(deviation_sigma: float) -> str:
    """Map a sigma deviation to a human-readable severity label.

    Args:
        deviation_sigma: How many standard deviations the projected value is
            away from the baseline mean.

    Returns:
        One of "low", "medium", "high", or "critical".
    """
    abs_dev = abs(deviation_sigma)
    if abs_dev >= 5.0:
        return "critical"
    if abs_dev >= 4.0:
        return "high"
    if abs_dev >= 3.0:
        return "medium"
    return "low"


def predict_anomalies(
    entity_baselines: dict[str, Any],
    recent_readings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Predict anomalies 30-60 min ahead using trend extrapolation.

    For each entity with enough recent readings, fits a linear trend on the
    last 6 data points and extrapolates forward. If the projected value
    crosses the 2.5-sigma threshold from the entity baseline, an alert is
    created.

    Args:
        entity_baselines: Per-entity baseline dict, keyed by entity_id.
            Each value should contain ``mean`` and ``std`` (or ``sigma``) keys.
        recent_readings: List of dicts, each with at least ``entity_id``,
            ``value``, and ``timestamp`` keys. Should be sorted oldest-first
            per entity.

    Returns:
        List of alert dicts with keys: entity_id, current_value,
        predicted_value_30m, predicted_value_60m, baseline_mean, threshold,
        severity, message.
    """
    # Group readings by entity
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for reading in recent_readings:
        eid = reading.get("entity_id", "")
        if not eid:
            continue
        by_entity.setdefault(eid, []).append(reading)

    alerts: list[dict[str, Any]] = []

    for eid, readings in by_entity.items():
        baseline = entity_baselines.get(eid)
        if not baseline:
            continue

        mean = baseline.get("mean")
        std = baseline.get("std") or baseline.get("sigma")
        if mean is None or std is None or std <= 0:
            continue

        # Sort by timestamp and take last 6
        sorted_readings = sorted(readings, key=lambda r: r.get("timestamp", ""))
        values = []
        for r in sorted_readings[-6:]:
            try:
                values.append(float(r["value"]))
            except (KeyError, ValueError, TypeError):
                continue

        if len(values) < MIN_READINGS:
            continue

        slope, intercept = _fit_linear_trend(values)
        n = len(values)
        current_value = values[-1]

        # Assume readings are ~10 min apart; steps_ahead = horizon / interval
        interval_min = 10.0
        predicted: dict[int, float] = {}
        triggered = False
        max_sigma = 0.0

        for horizon in HORIZONS_MIN:
            steps = horizon / interval_min
            proj = _extrapolate(slope, intercept, n, steps)
            predicted[horizon] = proj
            sigma_dev = abs(proj - mean) / std
            if sigma_dev >= SIGMA_THRESHOLD:
                triggered = True
                max_sigma = max(max_sigma, sigma_dev)

        if not triggered:
            continue

        threshold = mean + SIGMA_THRESHOLD * std
        severity = _severity_label(max_sigma)
        message = format_prediction(
            {
                "entity_id": eid,
                "predicted_value_30m": predicted.get(30, current_value),
                "predicted_value_60m": predicted.get(60, current_value),
                "baseline_mean": mean,
                "severity": severity,
            }
        )

        alerts.append(
            {
                "entity_id": eid,
                "current_value": round(current_value, 2),
                "predicted_value_30m": round(predicted.get(30, current_value), 2),
                "predicted_value_60m": round(predicted.get(60, current_value), 2),
                "baseline_mean": round(mean, 2),
                "threshold": round(threshold, 2),
                "severity": severity,
                "message": message,
            }
        )

    return alerts


def format_prediction(alert: dict[str, Any]) -> str:
    """Format a predictive alert into plain English.

    Args:
        alert: Alert dict with entity_id, predicted_value_30m,
            predicted_value_60m, baseline_mean, and severity keys.

    Returns:
        Human-readable description of the predicted anomaly.
    """
    eid = alert.get("entity_id", "unknown")
    pred_30 = alert.get("predicted_value_30m")
    pred_60 = alert.get("predicted_value_60m")
    mean = alert.get("baseline_mean", 0)
    severity = alert.get("severity", "low")

    # Pick whichever horizon deviates more for the narrative
    if pred_60 is not None and pred_30 is not None:
        dev_30 = abs(pred_30 - mean) if pred_30 is not None else 0
        dev_60 = abs(pred_60 - mean) if pred_60 is not None else 0
        if dev_60 > dev_30:
            eta = "~60 min"
            proj = pred_60
        else:
            eta = "~30 min"
            proj = pred_30
    elif pred_30 is not None:
        eta = "~30 min"
        proj = pred_30
    else:
        eta = "~60 min"
        proj = pred_60

    direction = "peak" if (proj or 0) > mean else "dip"
    return (
        f"{eid}: trajectory suggests unusual {direction} in {eta} "
        f"(projected {proj:.1f} vs baseline {mean:.1f}, severity: {severity})"
    )


def save(alerts: list[dict[str, Any]]) -> None:
    """Persist predictive alerts to disk.

    Args:
        alerts: List of alert dicts from ``predict_anomalies``.
    """
    payload = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "alerts": alerts,
        "count": len(alerts),
    }
    atomic_write(PREDICTIVE_ALERTS_PATH, payload)
    log.info("Predictive alerts saved: %d alert(s)", len(alerts))


def load() -> dict[str, Any]:
    """Load predictive alerts from disk.

    Returns:
        Previously saved alerts dict, or empty dict if none found.
    """
    if not os.path.exists(PREDICTIVE_ALERTS_PATH):
        return {}
    try:
        with open(PREDICTIVE_ALERTS_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
