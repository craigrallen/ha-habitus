"""Monthly energy budget tracking with projections.

Reads phantom load data to estimate monthly consumption and compares
against a user-defined energy budget (kWh or cost).  Provides daily
burn-rate calculations, overshoot projections, and alert thresholds.
"""

import datetime
import json
import logging
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
BUDGET_PATH = os.path.join(DATA_DIR, "energy_budget.json")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")

# Default cost per kWh when the user hasn't configured one
_DEFAULT_COST_PER_KWH = 0.12


def _load_budget() -> dict[str, Any]:
    """Load the budget config from disk.

    Returns:
        Budget dict, or empty dict if not yet configured.
    """
    if not os.path.exists(BUDGET_PATH):
        return {}
    try:
        with open(BUDGET_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load energy_budget.json: %s", exc)
        return {}


def _load_phantom_data() -> dict[str, Any]:
    """Load phantom load data for energy consumption estimates.

    Returns:
        Phantom loads dict, or empty dict if unavailable.
    """
    if not os.path.exists(PHANTOM_PATH):
        return {}
    try:
        with open(PHANTOM_PATH) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load phantom_loads.json: %s", exc)
        return {}


def set_budget(
    monthly_kwh: float | None = None,
    monthly_cost: float | None = None,
) -> dict:
    """Set or update the monthly energy budget.

    At least one of *monthly_kwh* or *monthly_cost* should be provided.
    If only cost is given, kWh is estimated using a default rate.

    Args:
        monthly_kwh: Monthly energy budget in kilowatt-hours.
        monthly_cost: Monthly energy budget in local currency.

    Returns:
        The persisted budget configuration dict.
    """
    budget = _load_budget()
    if monthly_kwh is not None:
        budget["monthly_kwh"] = monthly_kwh
    if monthly_cost is not None:
        budget["monthly_cost"] = monthly_cost
        if monthly_kwh is None and "monthly_kwh" not in budget:
            budget["monthly_kwh"] = monthly_cost / _DEFAULT_COST_PER_KWH

    budget["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    atomic_write(BUDGET_PATH, budget)
    log.info("Energy budget set: %s", budget)
    return budget


def get_daily_burn_rate(phantom_data: dict) -> float:
    """Compute the average daily energy consumption in kWh.

    Uses the ``total_monthly_kwh`` field from phantom load data and
    divides by the number of days in the current month.

    Args:
        phantom_data: Parsed phantom_loads.json content.

    Returns:
        Estimated daily kWh burn rate.  Returns 0.0 if data is missing.
    """
    monthly_kwh = phantom_data.get("total_monthly_kwh", 0.0)
    if not monthly_kwh:
        # Fall back: sum individual device standby estimates
        devices = phantom_data.get("devices", [])
        total_w = sum(d.get("standby_watts", 0.0) for d in devices)
        # Standby is 24h — convert watts to daily kWh
        return (total_w * 24.0) / 1000.0

    now = datetime.datetime.now(datetime.UTC)
    days_in_month = (
        datetime.date(now.year + (now.month // 12), (now.month % 12) + 1, 1)
        - datetime.date(now.year, now.month, 1)
    ).days
    return monthly_kwh / days_in_month


def get_budget_status() -> dict:
    """Compute current budget status with projections.

    Reads the configured budget and phantom load data, then calculates
    usage so far, remaining budget, projected total, and whether the
    household is on track.

    Returns:
        Dict with keys: ``budget_kwh``, ``used_kwh``, ``remaining_kwh``,
        ``days_remaining``, ``daily_avg``, ``projected_total``,
        ``on_track``, ``pct_used``, ``pct_month_elapsed``,
        ``overshoot_kwh``.  Returns a minimal dict with
        ``error`` if no budget is configured.
    """
    budget = _load_budget()
    if "monthly_kwh" not in budget:
        return {"error": "No budget configured. Call set_budget() first."}

    budget_kwh: float = budget["monthly_kwh"]
    phantom_data = _load_phantom_data()
    daily_avg = get_daily_burn_rate(phantom_data)

    now = datetime.datetime.now(datetime.UTC)
    day_of_month = now.day
    days_in_month = (
        datetime.date(now.year + (now.month // 12), (now.month % 12) + 1, 1)
        - datetime.date(now.year, now.month, 1)
    ).days
    days_remaining = days_in_month - day_of_month

    used_kwh = daily_avg * day_of_month
    projected_total = daily_avg * days_in_month
    remaining_kwh = max(0.0, budget_kwh - used_kwh)
    overshoot_kwh = max(0.0, projected_total - budget_kwh)
    pct_used = (used_kwh / budget_kwh * 100.0) if budget_kwh > 0 else 0.0
    pct_month_elapsed = (day_of_month / days_in_month * 100.0) if days_in_month > 0 else 0.0
    on_track = projected_total <= budget_kwh

    return {
        "budget_kwh": round(budget_kwh, 2),
        "used_kwh": round(used_kwh, 2),
        "remaining_kwh": round(remaining_kwh, 2),
        "days_remaining": days_remaining,
        "daily_avg": round(daily_avg, 2),
        "projected_total": round(projected_total, 2),
        "on_track": on_track,
        "pct_used": round(pct_used, 1),
        "pct_month_elapsed": round(pct_month_elapsed, 1),
        "overshoot_kwh": round(overshoot_kwh, 2),
    }


def should_alert(status: dict) -> bool:
    """Check whether a budget alert should fire.

    Returns ``True`` if the projected total exceeds the budget by more
    than 10 percent.

    Args:
        status: Dict returned by :func:`get_budget_status`.

    Returns:
        ``True`` if over-budget by >10%, ``False`` otherwise.
    """
    budget = status.get("budget_kwh", 0)
    projected = status.get("projected_total", 0)
    if budget <= 0:
        return False
    return projected > budget * 1.10


def format_budget_message(status: dict) -> str:
    """Format a plain-English budget status message.

    Args:
        status: Dict returned by :func:`get_budget_status`.

    Returns:
        Human-readable string describing the current budget situation.
    """
    if "error" in status:
        return status["error"]

    on_track_str = "on track" if status["on_track"] else "over budget"
    msg = (
        f"Energy budget: {status['used_kwh']:.1f} / {status['budget_kwh']:.1f} kWh used "
        f"({status['pct_used']:.0f}% of budget, {status['pct_month_elapsed']:.0f}% "
        f"of month elapsed). "
        f"Daily avg: {status['daily_avg']:.1f} kWh. "
        f"Projected total: {status['projected_total']:.1f} kWh — {on_track_str}."
    )
    if status["overshoot_kwh"] > 0:
        msg += f" Overshoot: +{status['overshoot_kwh']:.1f} kWh."
    return msg
