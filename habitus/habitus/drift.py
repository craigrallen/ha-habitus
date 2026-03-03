"""Routine Drift Detection — detect slow changes in household routines."""

import datetime
import json
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DRIFT_PATH = os.path.join(DATA_DIR, "drift.json")

# Minimum days of data needed
MIN_DAYS = 30
DRIFT_THRESHOLD_MIN = 45


def detect_drift(df: pd.DataFrame) -> dict:
    """Compare recent 14-day window vs baseline 30-60 days ago.

    Args:
        df: Feature dataframe with 'hour' column and activity features.

    Returns:
        Dict with drift metrics or {"reason": "..."} if insufficient data.
    """
    if df.empty or "hour" not in df.columns:
        return {"reason": "No data available"}

    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    date_range = (df["hour"].max() - df["hour"].min()).days
    if date_range < MIN_DAYS:
        return {"reason": f"Only {date_range} days of data, need {MIN_DAYS}"}

    now = df["hour"].max()
    recent_start = now - pd.Timedelta(days=14)
    baseline_end = now - pd.Timedelta(days=30)
    baseline_start = now - pd.Timedelta(days=60)

    recent = df[df["hour"] >= recent_start]
    baseline = df[(df["hour"] >= baseline_start) & (df["hour"] < baseline_end)]

    if len(recent) < 24 * 7 or len(baseline) < 24 * 7:
        return {"reason": "Not enough data in recent or baseline window"}

    metrics = {}
    drifts = []

    # First motion hour (proxy: first hour with motion_events > 0)
    for label, window in [("recent", recent), ("baseline", baseline)]:
        daily = window.copy()
        daily["date"] = daily["hour"].dt.date

        motion_col = "motion_events" if "motion_events" in daily.columns else None
        lights_col = "lights_on" if "lights_on" in daily.columns else None

        first_motion_hours = []
        last_motion_hours = []
        peak_hours = []
        lights_hours = []

        for _, day_data in daily.groupby("date"):
            day_data = day_data.sort_values("hour")

            if motion_col and day_data[motion_col].sum() > 0:
                active = day_data[day_data[motion_col] > 0]
                first_motion_hours.append(active["hour"].iloc[0].hour + active["hour"].iloc[0].minute / 60)
                last_motion_hours.append(active["hour"].iloc[-1].hour + active["hour"].iloc[-1].minute / 60)

            # Peak activity hour (by sensor_changes)
            if "sensor_changes" in day_data.columns and day_data["sensor_changes"].sum() > 0:
                peak_idx = day_data["sensor_changes"].idxmax()
                peak_hours.append(day_data.loc[peak_idx, "hour"].hour)

            if lights_col:
                lights_hours.append(day_data[lights_col].sum())

        metrics[label] = {
            "first_motion_hour": np.mean(first_motion_hours) if first_motion_hours else None,
            "last_motion_hour": np.mean(last_motion_hours) if last_motion_hours else None,
            "peak_activity_hour": np.mean(peak_hours) if peak_hours else None,
            "avg_lights_on_hours": np.mean(lights_hours) if lights_hours else None,
        }

    result = {"timestamp": datetime.datetime.now(datetime.UTC).isoformat(), "drifts": []}

    drift_names = {
        "first_motion_hour": ("morning_drift_min", "Morning routine"),
        "last_motion_hour": ("bedtime_drift_min", "Bedtime"),
        "peak_activity_hour": ("peak_drift_min", "Peak activity"),
        "avg_lights_on_hours": ("lights_drift_hours", "Lights-on duration"),
    }

    for metric_key, (drift_key, friendly) in drift_names.items():
        r_val = metrics["recent"].get(metric_key)
        b_val = metrics["baseline"].get(metric_key)
        if r_val is None or b_val is None:
            continue

        if metric_key == "avg_lights_on_hours":
            diff_hours = r_val - b_val
            result["drifts"].append({
                "metric": drift_key,
                "diff": round(diff_hours, 1),
                "unit": "hours",
                "significant": abs(diff_hours) > 0.75,
            })
        else:
            diff_min = (r_val - b_val) * 60
            direction = "later" if diff_min > 0 else "earlier"
            significant = abs(diff_min) >= DRIFT_THRESHOLD_MIN
            entry = {
                "metric": drift_key,
                "diff_min": round(diff_min),
                "direction": direction,
                "significant": significant,
            }
            if significant:
                entry["summary"] = f"{friendly} has shifted {abs(round(diff_min))} min {direction} over recent weeks"
                drifts.append(entry)
            result["drifts"].append(entry)

    # Overall summary
    sig = [d for d in result["drifts"] if d.get("significant")]
    if sig:
        main = max(sig, key=lambda x: abs(x.get("diff_min", 0)))
        result["summary"] = main.get("summary", "Routine drift detected")
        result["direction"] = main.get("direction", "unknown")
    else:
        result["summary"] = "No significant routine drift detected"

    return result


def save(data: dict) -> None:
    """Save drift results to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DRIFT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Drift analysis saved: %s", data.get("summary", ""))


def load() -> dict:
    """Load drift results from disk."""
    if not os.path.exists(DRIFT_PATH):
        return {}
    try:
        with open(DRIFT_PATH) as f:
            return json.load(f)
    except Exception:
        return {}
