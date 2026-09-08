"""Comfort Intelligence — multi-dimensional comfort scoring.

Combines temperature, humidity, and lighting into a single 0-100 comfort
score. Learns preferred conditions per time-of-day and suggests
pre-conditioning automations.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import pandas as pd  # noqa: TC002

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
COMFORT_PATH = os.path.join(DATA_DIR, "comfort_report.json")

# Ideal indoor ranges (configurable via env)
TEMP_IDEAL_MIN = float(os.environ.get("HABITUS_COMFORT_TEMP_MIN", "20"))
TEMP_IDEAL_MAX = float(os.environ.get("HABITUS_COMFORT_TEMP_MAX", "23"))
HUMIDITY_IDEAL_MIN = float(os.environ.get("HABITUS_COMFORT_HUMIDITY_MIN", "40"))
HUMIDITY_IDEAL_MAX = float(os.environ.get("HABITUS_COMFORT_HUMIDITY_MAX", "60"))


def _temp_comfort(temp_c: float) -> float:
    """Score temperature comfort (0-1). 1.0 = ideal range."""
    if TEMP_IDEAL_MIN <= temp_c <= TEMP_IDEAL_MAX:
        return 1.0
    if temp_c < TEMP_IDEAL_MIN:
        return max(0, 1 - (TEMP_IDEAL_MIN - temp_c) / 10)
    return max(0, 1 - (temp_c - TEMP_IDEAL_MAX) / 10)


def _humidity_comfort(humidity_pct: float) -> float:
    """Score humidity comfort (0-1). 1.0 = ideal range."""
    if HUMIDITY_IDEAL_MIN <= humidity_pct <= HUMIDITY_IDEAL_MAX:
        return 1.0
    if humidity_pct < HUMIDITY_IDEAL_MIN:
        return max(0, 1 - (HUMIDITY_IDEAL_MIN - humidity_pct) / 30)
    return max(0, 1 - (humidity_pct - HUMIDITY_IDEAL_MAX) / 30)


def compute_comfort(features: pd.DataFrame) -> dict[str, Any]:
    """Compute comfort scores from feature data.

    Args:
        features: Hourly feature DataFrame with at least ``avg_temp_c``
            column. Optional: humidity, lights_on columns.

    Returns:
        Comfort report with current score, hourly profile, and
        recommendations.
    """
    if features.empty or "avg_temp_c" not in features.columns:
        return {"error": "Need temperature data for comfort scoring"}

    has_hour = "hour_of_day" in features.columns

    # Current comfort (latest readings)
    latest = features.iloc[-1]
    temp = float(latest["avg_temp_c"])
    temp_score = _temp_comfort(temp)

    humidity_score = 1.0
    humidity_val = None
    for col in ("humidity", "avg_humidity", "indoor_humidity"):
        if col in features.columns:
            humidity_val = float(latest[col])
            humidity_score = _humidity_comfort(humidity_val)
            break

    light_score = 1.0  # Default — no lighting penalty
    if "lights_on" in features.columns and has_hour:
        hour = int(latest["hour_of_day"])
        lights_on = bool(latest["lights_on"])
        if hour >= 23 or hour < 6:
            light_score = 0.7 if lights_on else 1.0  # Lights on at night reduces comfort
        elif 8 <= hour <= 18:
            light_score = 1.0  # Daytime — no penalty either way

    # Weighted composite (temperature dominates)
    current_score = int((temp_score * 0.50 + humidity_score * 0.30 + light_score * 0.20) * 100)

    # Hourly comfort profile
    hourly_profile: dict[int, float] = {}
    if has_hour:
        for hour_val, grp in features.groupby("hour_of_day"):
            avg_t = float(grp["avg_temp_c"].mean())
            hourly_profile[int(hour_val)] = round(_temp_comfort(avg_t) * 100, 0)

    # Find worst comfort hours
    worst_hours: list[dict[str, Any]] = []
    if hourly_profile:
        sorted_hours = sorted(hourly_profile.items(), key=lambda x: x[1])
        for h, score in sorted_hours[:3]:
            if score < 70:
                worst_hours.append({"hour": h, "score": int(score)})

    # Learned preferences (what temp does the user keep when home?)
    preferences: dict[str, Any] = {}
    if has_hour and "total_power_w" in features.columns:
        # Active hours = power above baseline = user is home and active
        active = features[features["total_power_w"] > features["total_power_w"].quantile(0.3)]
        if not active.empty:
            preferences["preferred_temp_c"] = round(float(active["avg_temp_c"].mean()), 1)
            preferences["preferred_temp_range"] = [
                round(float(active["avg_temp_c"].quantile(0.25)), 1),
                round(float(active["avg_temp_c"].quantile(0.75)), 1),
            ]

    # Recommendations
    recs: list[str] = []
    if temp < TEMP_IDEAL_MIN - 2:
        recs.append(
            f"Temperature is {temp:.1f}\u00b0C — consider pre-heating "
            f"to reach {TEMP_IDEAL_MIN:.0f}\u00b0C before your active hours."
        )
    elif temp > TEMP_IDEAL_MAX + 2:
        recs.append(
            f"Temperature is {temp:.1f}\u00b0C — ventilation or cooling " "would improve comfort."
        )
    if humidity_val is not None and humidity_val > HUMIDITY_IDEAL_MAX + 10:
        recs.append(f"Humidity is {humidity_val:.0f}% — consider a dehumidifier " "or ventilation.")
    if not recs:
        recs.append("Comfort levels are within ideal range.")

    report: dict[str, Any] = {
        "current_score": current_score,
        "temperature": {
            "value_c": round(temp, 1),
            "score": int(temp_score * 100),
            "ideal_range": [TEMP_IDEAL_MIN, TEMP_IDEAL_MAX],
        },
        "humidity": {
            "value_pct": round(humidity_val, 1) if humidity_val is not None else None,
            "score": int(humidity_score * 100),
            "ideal_range": [HUMIDITY_IDEAL_MIN, HUMIDITY_IDEAL_MAX],
        },
        "hourly_profile": hourly_profile,
        "worst_hours": worst_hours,
        "preferences": preferences,
        "recommendations": recs,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    atomic_write(COMFORT_PATH, report)
    log.info("Comfort score: %d/100 (temp %.1f\u00b0C)", current_score, temp)
    return report


def load() -> dict[str, Any]:
    """Load the latest comfort report from disk."""
    try:
        if os.path.exists(COMFORT_PATH):
            with open(COMFORT_PATH) as f:
                data: dict[str, Any] = json.load(f)
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}
