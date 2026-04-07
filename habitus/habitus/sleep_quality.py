"""Sleep Quality Inference — estimate sleep patterns from motion, lights, and power.

No wearable required. Uses:
- Motion sensor silence (no motion = likely asleep)
- Light state (lights off = likely asleep)
- Power drop to overnight baseline
- Consistency across nights

Output: sleep_report.json with sleep/wake estimates, consistency score,
and disruption events.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SLEEP_REPORT_PATH = os.path.join(DATA_DIR, "sleep_report.json")

# Hours considered candidate sleep window
SLEEP_WINDOW_START = 21  # 9 PM
SLEEP_WINDOW_END = 10  # 10 AM next day

# Minimum hours of inactivity to count as sleep
MIN_SLEEP_HOURS = 4


def infer_sleep(features: pd.DataFrame) -> dict[str, Any]:
    """Infer sleep patterns from feature data.

    Analyzes motion, lights, and power columns to estimate nightly
    sleep/wake times and quality metrics.

    Args:
        features: Hourly feature DataFrame with ``hour_of_day``,
            ``motion_events``, ``lights_on``, ``total_power_w`` columns.

    Returns:
        Sleep report dict with nightly estimates and aggregate metrics.
    """
    if features.empty or "hour_of_day" not in features.columns:
        return {"error": "Insufficient data for sleep inference"}

    has_motion = "motion_events" in features.columns
    has_lights = "lights_on" in features.columns
    has_power = "total_power_w" in features.columns

    if not (has_motion or has_lights):
        return {"error": "Need motion_events or lights_on columns"}

    # Build activity score per hour (0 = inactive, 1 = fully active)
    activity = pd.Series(0.0, index=features.index)
    if has_motion:
        motion_norm = features["motion_events"] / max(features["motion_events"].max(), 1)
        activity += motion_norm * 0.5
    if has_lights:
        activity += features["lights_on"] * 0.3
    if has_power:
        night_baseline = features.loc[
            features["hour_of_day"].isin(range(1, 6)), "total_power_w"
        ].mean()
        if pd.notna(night_baseline) and night_baseline > 0:
            power_above = (features["total_power_w"] - night_baseline).clip(lower=0)
            power_norm = power_above / max(power_above.max(), 1)
            activity += power_norm * 0.2

    features = features.copy()
    features["_activity"] = activity.clip(0, 1)

    # Group by date and find sleep windows
    if "hour" in features.columns:
        features["_date"] = pd.to_datetime(features["hour"]).dt.date
    else:
        # Synthesize date from index
        n = len(features)
        days = n // 24
        if days < 2:
            return {"error": "Need at least 2 days of data"}
        dates = []
        for d in range(days):
            dates.extend([d] * 24)
        features["_date"] = dates[:n]

    nights: list[dict[str, Any]] = []
    unique_dates = sorted(features["_date"].unique())

    for i, date in enumerate(unique_dates[:-1]):
        # Evening window: this date 21:00 → next date 10:00
        day_data = features[features["_date"] == date]
        next_data = features[features["_date"] == unique_dates[i + 1]]

        evening = day_data[day_data["hour_of_day"] >= SLEEP_WINDOW_START]
        morning = next_data[next_data["hour_of_day"] < SLEEP_WINDOW_END]

        if evening.empty and morning.empty:
            continue

        night_window = pd.concat([evening, morning])
        if len(night_window) < 6:
            continue

        # Find sleep onset: first hour with activity < 0.15
        sleep_start = None
        for _, row in night_window.iterrows():
            if row["_activity"] < 0.15:
                sleep_start = int(row["hour_of_day"])
                break

        # Find wake time: last hour with activity > 0.3 in morning
        wake_time = None
        for _, row in morning.iloc[::-1].iterrows():
            if row["_activity"] > 0.3:
                wake_time = int(row["hour_of_day"])
                break

        if sleep_start is None:
            continue

        # Disruptions: hours during sleep with activity > 0.2
        if wake_time and sleep_start:
            # Sleep duration (handle midnight wrap)
            if sleep_start > 12:
                duration = (24 - sleep_start) + (wake_time or 7)
            else:
                duration = (wake_time or 7) - sleep_start
            duration = max(0, min(duration, 14))
        else:
            duration = 0

        disruptions = int((night_window["_activity"] > 0.2).sum()) - 1
        disruptions = max(0, disruptions)

        if duration >= MIN_SLEEP_HOURS:
            nights.append(
                {
                    "date": str(date),
                    "sleep_onset_hour": sleep_start,
                    "wake_hour": wake_time,
                    "duration_hours": round(duration, 1),
                    "disruptions": disruptions,
                    "quality_score": _quality_score(duration, disruptions, sleep_start),
                }
            )

    if not nights:
        return {"error": "Could not detect sleep patterns", "nights": []}

    # Aggregate metrics
    durations = [n["duration_hours"] for n in nights]
    onsets = [n["sleep_onset_hour"] for n in nights]
    qualities = [n["quality_score"] for n in nights]

    # Consistency: low std in onset time = consistent routine
    onset_std = float(np.std(onsets)) if len(onsets) > 1 else 0
    consistency = max(0, min(100, int(100 - onset_std * 20)))

    report: dict[str, Any] = {
        "nights": nights[-14:],  # Keep last 2 weeks
        "avg_duration_hours": round(float(np.mean(durations)), 1),
        "avg_onset_hour": round(float(np.mean(onsets)), 1),
        "avg_quality_score": round(float(np.mean(qualities)), 0),
        "consistency_score": consistency,
        "total_nights_analysed": len(nights),
        "recommendation": _recommendation(
            float(np.mean(durations)), consistency, float(np.mean(qualities))
        ),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    atomic_write(SLEEP_REPORT_PATH, report)
    log.info(
        "Sleep analysis: %d nights, avg %.1fh, consistency %d%%",
        len(nights),
        report["avg_duration_hours"],
        consistency,
    )
    return report


def _quality_score(duration: float, disruptions: int, onset_hour: int) -> int:
    """Compute a 0-100 sleep quality score.

    Args:
        duration: Sleep duration in hours.
        onset_hour: Hour of sleep onset (0-23).
        disruptions: Number of disruption events.

    Returns:
        Quality score 0-100.
    """
    score = 50.0

    # Duration: 7-9h is optimal
    if 7 <= duration <= 9:
        score += 25
    elif 6 <= duration < 7 or 9 < duration <= 10:
        score += 15
    elif duration < 6:
        score -= 10
    else:
        score += 5

    # Disruptions
    score -= disruptions * 5

    # Onset consistency (22-23 is optimal)
    if 22 <= onset_hour <= 23:
        score += 15
    elif onset_hour >= 21 or onset_hour == 0:
        score += 10
    elif onset_hour >= 1:
        score -= 5

    return max(0, min(100, int(score)))


def _recommendation(avg_duration: float, consistency: int, avg_quality: float) -> str:
    """Generate a sleep recommendation based on metrics."""
    if avg_quality >= 75 and consistency >= 70:
        return "Sleep patterns look healthy — consistent timing and adequate duration."
    parts: list[str] = []
    if avg_duration < 7:
        parts.append(f"Average sleep is {avg_duration:.1f}h — consider targeting 7-8 hours.")
    if consistency < 50:
        parts.append("Sleep timing varies significantly — a consistent bedtime improves quality.")
    if avg_quality < 50:
        parts.append("Multiple disruptions detected — check for noise or comfort issues.")
    return " ".join(parts) if parts else "Sleep patterns are within normal range."


def load() -> dict[str, Any]:
    """Load the latest sleep report from disk."""
    try:
        if os.path.exists(SLEEP_REPORT_PATH):
            with open(SLEEP_REPORT_PATH) as f:
                data: dict[str, Any] = json.load(f)
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}
