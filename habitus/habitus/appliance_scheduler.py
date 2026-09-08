"""Appliance scheduling — recommend optimal times for high-draw appliances."""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SCHEDULING_PATH = os.path.join(DATA_DIR, "scheduling_recommendations.json")

# Tariff configuration via environment variables
_PEAK_START: int = int(os.environ.get("HABITUS_PEAK_START", "16"))
_PEAK_END: int = int(os.environ.get("HABITUS_PEAK_END", "21"))
_PEAK_RATE: float = float(os.environ.get("HABITUS_PEAK_RATE", "0.30"))
_OFFPEAK_RATE: float = float(os.environ.get("HABITUS_OFFPEAK_RATE", "0.10"))

TARIFF_SCHEDULE: dict[str, Any] = {
    "peak_start": _PEAK_START,
    "peak_end": _PEAK_END,
    "peak_rate": _PEAK_RATE,
    "offpeak_rate": _OFFPEAK_RATE,
}


def get_tariff_profile() -> dict[str, Any]:
    """Return the current tariff configuration.

    Returns:
        Dict with peak_start, peak_end, peak_rate, and offpeak_rate.
    """
    return {
        "peak_start": int(os.environ.get("HABITUS_PEAK_START", "16")),
        "peak_end": int(os.environ.get("HABITUS_PEAK_END", "21")),
        "peak_rate": float(os.environ.get("HABITUS_PEAK_RATE", "0.30")),
        "offpeak_rate": float(os.environ.get("HABITUS_OFFPEAK_RATE", "0.10")),
    }


def _rate_at_hour(hour: int, tariff: dict[str, Any]) -> float:
    """Return the tariff rate for a given hour of the day.

    Args:
        hour: Hour in 0-23 range.
        tariff: Tariff profile dict.

    Returns:
        Cost rate (currency units per kWh).
    """
    peak_start = tariff["peak_start"]
    peak_end = tariff["peak_end"]
    if peak_start <= hour < peak_end:
        return float(tariff["peak_rate"])
    return float(tariff["offpeak_rate"])


def _cost_for_window(
    start_hour: int,
    duration_min: int,
    power_w: float,
    tariff: dict[str, Any],
    solar_forecast: list[float] | None = None,
) -> tuple[float, float]:
    """Calculate electricity cost and solar coverage for a run window.

    Args:
        start_hour: Hour of day (0-23) when the appliance starts.
        duration_min: Run duration in minutes.
        power_w: Power draw in watts.
        tariff: Tariff profile dict.
        solar_forecast: Optional 24-element list of solar production in watts
            per hour (index = hour of day).

    Returns:
        Tuple of (cost, solar_pct) where cost is in currency units and
        solar_pct is the fraction of energy offset by solar (0.0 to 1.0).
    """
    power_kw = power_w / 1000.0
    total_cost = 0.0
    total_solar_offset_kwh = 0.0
    total_energy_kwh = 0.0

    # Walk through each hour slot the appliance occupies
    remaining_min = duration_min
    current_hour = start_hour

    while remaining_min > 0:
        minutes_in_slot = min(remaining_min, 60)
        fraction = minutes_in_slot / 60.0
        energy_kwh = power_kw * fraction
        total_energy_kwh += energy_kwh

        rate = _rate_at_hour(current_hour % 24, tariff)

        # Solar offset
        solar_w = 0.0
        if solar_forecast and 0 <= (current_hour % 24) < len(solar_forecast):
            solar_w = solar_forecast[current_hour % 24]

        solar_kw = solar_w / 1000.0
        offset_kw = min(solar_kw, power_kw)
        solar_offset_kwh = offset_kw * fraction
        total_solar_offset_kwh += solar_offset_kwh

        net_kwh = max(energy_kwh - solar_offset_kwh, 0.0)
        total_cost += net_kwh * rate

        remaining_min -= minutes_in_slot
        current_hour += 1

    solar_pct = total_solar_offset_kwh / total_energy_kwh if total_energy_kwh > 0 else 0.0
    return total_cost, solar_pct


def recommend_schedule(
    appliance_name: str,
    power_w: float,
    duration_min: int,
    solar_forecast: list[float] | None = None,
) -> dict[str, Any]:
    """Find the cheapest time window to run an appliance.

    Evaluates every possible start hour (0-23) and picks the window with
    the lowest net cost after tariff and solar offset.

    Args:
        appliance_name: Human-readable appliance name (e.g., "Washing Machine").
        power_w: Appliance power draw in watts.
        duration_min: Run duration in minutes.
        solar_forecast: Optional 24-element list of solar production in watts
            per hour (index = hour of day). None if no solar data available.

    Returns:
        Dict with appliance, best_start_hour, estimated_cost, savings_vs_now,
        solar_pct, and reason.
    """
    tariff = get_tariff_profile()
    now_hour = datetime.datetime.now().hour

    # Cost if run right now
    cost_now, solar_now = _cost_for_window(now_hour, duration_min, power_w, tariff, solar_forecast)

    best_hour = now_hour
    best_cost = cost_now
    best_solar = solar_now

    for hour in range(24):
        cost, solar_pct = _cost_for_window(hour, duration_min, power_w, tariff, solar_forecast)
        if cost < best_cost:
            best_cost = cost
            best_hour = hour
            best_solar = solar_pct

    savings = cost_now - best_cost
    reason = _build_reason(best_hour, now_hour, savings, best_solar, tariff)

    return {
        "appliance": appliance_name,
        "best_start_hour": best_hour,
        "estimated_cost": round(best_cost, 4),
        "savings_vs_now": round(savings, 4),
        "solar_pct": round(best_solar, 3),
        "reason": reason,
    }


def _build_reason(
    best_hour: int,
    now_hour: int,
    savings: float,
    solar_pct: float,
    tariff: dict[str, Any],
) -> str:
    """Build a human-readable reason for the scheduling recommendation.

    Args:
        best_hour: Recommended start hour.
        now_hour: Current hour.
        savings: Cost savings vs running now.
        solar_pct: Fraction of energy covered by solar.
        tariff: Tariff profile dict.

    Returns:
        Plain English explanation of why this time is recommended.
    """
    parts: list[str] = []

    if best_hour == now_hour:
        parts.append("Now is already the optimal time")
    else:
        peak_start = tariff["peak_start"]
        peak_end = tariff["peak_end"]
        if peak_start <= now_hour < peak_end and not (peak_start <= best_hour < peak_end):
            parts.append(f"Shift from peak ({now_hour}:00) to off-peak ({best_hour}:00)")
        else:
            parts.append(f"Best start at {best_hour}:00")

    if savings > 0:
        parts.append(f"saves {savings:.3f} currency units")

    if solar_pct > 0.1:
        parts.append(f"{solar_pct:.0%} covered by solar")

    return "; ".join(parts)


def recommend_all(
    nilm_results: dict[str, Any],
    solar_forecast: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Run scheduling recommendations for all detected appliances.

    Args:
        nilm_results: Dict keyed by appliance name, each value containing
            at least ``power_w`` and ``duration_min``.
        solar_forecast: Optional 24-element solar production forecast.

    Returns:
        List of recommendation dicts from ``recommend_schedule``.
    """
    recommendations: list[dict[str, Any]] = []

    for name, info in nilm_results.items():
        power = info.get("power_w")
        duration = info.get("duration_min")
        if power is None or duration is None:
            log.warning("Skipping appliance %s: missing power_w or duration_min", name)
            continue

        try:
            rec = recommend_schedule(
                appliance_name=name,
                power_w=float(power),
                duration_min=int(duration),
                solar_forecast=solar_forecast,
            )
            recommendations.append(rec)
        except (ValueError, TypeError) as exc:
            log.warning("Failed to schedule %s: %s", name, exc)

    return recommendations


def save(recommendations: list[dict[str, Any]]) -> None:
    """Persist scheduling recommendations to disk.

    Args:
        recommendations: List of recommendation dicts.
    """
    payload = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "tariff": get_tariff_profile(),
        "recommendations": recommendations,
        "count": len(recommendations),
    }
    atomic_write(SCHEDULING_PATH, payload)
    log.info("Scheduling recommendations saved: %d appliance(s)", len(recommendations))


def load() -> dict[str, Any]:
    """Load scheduling recommendations from disk.

    Returns:
        Previously saved recommendations dict, or empty dict if none found.
    """
    if not os.path.exists(SCHEDULING_PATH):
        return {}
    try:
        with open(SCHEDULING_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
