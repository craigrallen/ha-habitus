"""Weather-aware energy forecasting.

Uses HA weather entities (local, no external API calls) + historical energy data
to predict tomorrow's energy usage with uncertainty bands.

"Tomorrow: 6°C, rain — expected 18.2 kWh ± 2.3 kWh (heating will dominate)"
"""

import datetime
import json
import logging
import math
import os
import sqlite3
from collections import defaultdict
from typing import Any

import numpy as np

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
FORECAST_PATH = os.path.join(DATA_DIR, "energy_forecast.json")
HA_DB = "/homeassistant/home-assistant_v2.db"

# Default entities
WEATHER_ENTITY = "weather.forecast_home"  # HA default weather integration
OUTDOOR_TEMP_PATTERNS = ("outdoor_temp", "outside_temp", "weather_temp", "exterior_temp")


def _get_daily_energy_and_weather(days: int = 90) -> list[dict]:
    """Get daily energy consumption paired with weather data.

    Returns list of {date, kwh, avg_temp, min_temp, max_temp, day_of_week, month}.
    """
    if not os.path.exists(HA_DB):
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)

        # Find energy entity (cumulative kWh meter)
        energy_candidates = conn.execute("""
            SELECT DISTINCT sm.entity_id FROM states_meta sm
            WHERE (sm.entity_id LIKE '%consumption_kwh%'
                   OR sm.entity_id LIKE '%energy_kwh%'
                   OR sm.entity_id LIKE '%grid%kwh%')
            AND sm.entity_id LIKE 'sensor.%'
        """).fetchall()

        energy_eid = None
        for (eid,) in energy_candidates:
            if "shore_power" in eid or "grid" in eid or "consumption" in eid:
                energy_eid = eid
                break
        if not energy_eid and energy_candidates:
            energy_eid = energy_candidates[0][0]

        # Find outdoor temperature sensor
        temp_candidates = conn.execute("""
            SELECT DISTINCT sm.entity_id FROM states_meta sm
            WHERE sm.entity_id LIKE 'sensor.%'
            AND (sm.entity_id LIKE '%outdoor%temp%'
                 OR sm.entity_id LIKE '%outside%temp%'
                 OR sm.entity_id LIKE '%exterior%temp%'
                 OR sm.entity_id LIKE '%weather%temp%')
        """).fetchall()

        temp_eid = None
        for (eid,) in temp_candidates:
            temp_eid = eid
            break

        if not energy_eid:
            conn.close()
            log.warning("energy_forecast: no energy entity found")
            return []

        # Get energy readings
        energy_rows = conn.execute("""
            SELECT s.state, s.last_changed_ts FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE sm.entity_id = ? AND s.last_changed_ts > ?
            ORDER BY s.last_changed_ts
        """, (energy_eid, cutoff_ts)).fetchall()

        # Get temperature readings
        temp_rows = []
        if temp_eid:
            temp_rows = conn.execute("""
                SELECT s.state, s.last_changed_ts FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                ORDER BY s.last_changed_ts
            """, (temp_eid, cutoff_ts)).fetchall()

        conn.close()
    except Exception as e:
        log.warning("energy_forecast: DB query failed: %s", e)
        return []

    # Process energy into daily totals (difference-based for cumulative meter)
    daily_energy: dict[str, list[float]] = defaultdict(list)
    for state_val, ts in energy_rows:
        try:
            kwh = float(state_val)
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
            date_str = dt.strftime("%Y-%m-%d")
            daily_energy[date_str].append(kwh)
        except (ValueError, TypeError):
            continue

    # Process temperature into daily stats
    daily_temp: dict[str, list[float]] = defaultdict(list)
    for state_val, ts in temp_rows:
        try:
            temp = float(state_val)
            if -50 < temp < 60:
                dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
                date_str = dt.strftime("%Y-%m-%d")
                daily_temp[date_str].append(temp)
        except (ValueError, TypeError):
            continue

    # Merge into daily records
    records = []
    for date_str in sorted(daily_energy.keys()):
        readings = daily_energy[date_str]
        if len(readings) < 2:
            continue
        kwh = max(readings) - min(readings)
        if kwh < 0 or kwh > 200:
            continue

        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        temps = daily_temp.get(date_str, [])

        records.append({
            "date": date_str,
            "kwh": round(kwh, 2),
            "avg_temp": round(np.mean(temps), 1) if temps else None,
            "min_temp": round(min(temps), 1) if temps else None,
            "max_temp": round(max(temps), 1) if temps else None,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "is_weekend": dt.weekday() >= 5,
        })

    log.info("energy_forecast: %d daily records (energy=%s, temp=%s)",
             len(records), energy_eid, temp_eid or "none")
    return records


def run_energy_forecast(days_history: int = 90) -> dict[str, Any]:
    """Build energy forecast model and predict next 7 days.

    Uses a simple regression: kWh ~ f(day_of_week, month, avg_temp, is_weekend)
    with Gaussian Process for uncertainty estimation if sklearn is available.
    """
    records = _get_daily_energy_and_weather(days=days_history)
    if len(records) < 14:
        return {"forecast": [], "model_quality": "insufficient_data"}

    # Build feature matrix
    has_temp = any(r["avg_temp"] is not None for r in records)
    features = []
    targets = []

    for r in records:
        row = [
            r["day_of_week"],
            r["month"],
            1.0 if r["is_weekend"] else 0.0,
        ]
        if has_temp and r["avg_temp"] is not None:
            row.append(r["avg_temp"])
        elif has_temp:
            continue  # Skip rows without temp when we have temp data
        features.append(row)
        targets.append(r["kwh"])

    X = np.array(features)
    y = np.array(targets)

    if len(X) < 14:
        return {"forecast": [], "model_quality": "insufficient_data"}

    # Try Gaussian Process for uncertainty, fall back to Ridge
    forecast_days = []
    model_type = "unknown"

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kernel = ConstantKernel() * RBF() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
        gpr.fit(X_scaled, y)
        model_type = "gaussian_process"

        # Predict next 7 days
        today = datetime.date.today()
        for d in range(1, 8):
            future_date = today + datetime.timedelta(days=d)
            row = [
                future_date.weekday(),
                future_date.month,
                1.0 if future_date.weekday() >= 5 else 0.0,
            ]
            # Use recent average temp as proxy (or seasonal average)
            if has_temp:
                recent_temps = [r["avg_temp"] for r in records[-14:] if r["avg_temp"] is not None]
                row.append(np.mean(recent_temps) if recent_temps else 10.0)

            X_pred = scaler.transform([row])
            mean, std = gpr.predict(X_pred, return_std=True)
            predicted_kwh = max(0, float(mean[0]))
            uncertainty = float(std[0])

            forecast_days.append({
                "date": future_date.isoformat(),
                "day_name": future_date.strftime("%A"),
                "predicted_kwh": round(predicted_kwh, 1),
                "uncertainty_kwh": round(uncertainty, 1),
                "low_kwh": round(max(0, predicted_kwh - 2 * uncertainty), 1),
                "high_kwh": round(predicted_kwh + 2 * uncertainty, 1),
                "is_weekend": future_date.weekday() >= 5,
            })

    except Exception as e:
        log.warning("GP forecast failed, using simple averages: %s", e)
        model_type = "simple_average"

        # Simple fallback: average by day of week
        dow_kwh: dict[int, list[float]] = defaultdict(list)
        for r in records:
            dow_kwh[r["day_of_week"]].append(r["kwh"])

        today = datetime.date.today()
        for d in range(1, 8):
            future_date = today + datetime.timedelta(days=d)
            dow = future_date.weekday()
            values = dow_kwh.get(dow, list(y))
            mean_kwh = np.mean(values) if values else float(np.mean(y))
            std_kwh = np.std(values) if len(values) > 1 else float(np.std(y))

            forecast_days.append({
                "date": future_date.isoformat(),
                "day_name": future_date.strftime("%A"),
                "predicted_kwh": round(mean_kwh, 1),
                "uncertainty_kwh": round(std_kwh, 1),
                "low_kwh": round(max(0, mean_kwh - 2 * std_kwh), 1),
                "high_kwh": round(mean_kwh + 2 * std_kwh, 1),
                "is_weekend": future_date.weekday() >= 5,
            })

    # Temperature-energy correlation
    temp_correlation = None
    if has_temp:
        temps = [r["avg_temp"] for r in records if r["avg_temp"] is not None]
        kwhs = [r["kwh"] for r in records if r["avg_temp"] is not None]
        if len(temps) >= 10:
            corr = float(np.corrcoef(temps, kwhs)[0, 1])
            temp_correlation = {
                "coefficient": round(corr, 3),
                "direction": "inverse" if corr < -0.3 else "positive" if corr > 0.3 else "weak",
                "interpretation": (
                    "Colder days = more energy (heating dominant)" if corr < -0.3
                    else "Warmer days = more energy (cooling dominant)" if corr > 0.3
                    else "Temperature has weak effect on energy usage"
                ),
            }

    # Weekly summary
    total_forecast = sum(f["predicted_kwh"] for f in forecast_days)
    avg_recent = float(np.mean([r["kwh"] for r in records[-7:]])) * 7

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "model_type": model_type,
        "training_days": len(records),
        "forecast": forecast_days,
        "weekly_total_kwh": round(total_forecast, 1),
        "recent_weekly_avg_kwh": round(avg_recent, 1),
        "trend": "up" if total_forecast > avg_recent * 1.1 else "down" if total_forecast < avg_recent * 0.9 else "stable",
        "temperature_correlation": temp_correlation,
        "historical_summary": {
            "avg_daily_kwh": round(float(np.mean(y)), 1),
            "std_daily_kwh": round(float(np.std(y)), 1),
            "min_daily_kwh": round(float(np.min(y)), 1),
            "max_daily_kwh": round(float(np.max(y)), 1),
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FORECAST_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("energy_forecast: %s model, %d-day forecast, weekly=%.1f kWh",
             model_type, len(forecast_days), total_forecast)
    return result
