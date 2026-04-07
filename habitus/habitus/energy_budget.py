"""Smart Energy Budget — auto-computes monthly targets from HA Energy Dashboard history.

Instead of asking the user to guess a budget, derives it from actual
historical consumption. Uses 12-month rolling data from phantom_loads.json
(which reads directly from HA's Energy Dashboard statistics).

Smart features:
- Auto-budget: same-month average from last 1-2 years of history
- Seasonal awareness: winter months get higher budgets than summer
- Trend detection: flags if this month is tracking above/below historical norm
- Day-normalized comparison: compares first N days of this month vs same
  N days in previous months to detect overshoot early
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

KWH_PRICE = float(os.environ.get("HABITUS_KWH_PRICE", "0.30"))
CURRENCY = os.environ.get("HABITUS_CURRENCY", "kr")


def _load_json(path: str, default: Any = None) -> Any:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return default


def _load_budget() -> dict[str, Any]:
    data = _load_json(BUDGET_PATH, {})
    return data if isinstance(data, dict) else {}


def _load_phantom() -> dict[str, Any]:
    data = _load_json(PHANTOM_PATH, {})
    return data if isinstance(data, dict) else {}


def _compute_smart_budget(months_data: list[dict]) -> dict[str, Any]:
    """Compute a smart monthly budget from historical Energy Dashboard data.

    Uses the same month in previous years as primary reference. Falls back
    to the trailing 3-month average if no same-month history exists.

    Args:
        months_data: List of ``{month: "YYYY-MM", kwh: float}`` from
            phantom_loads.json.

    Returns:
        Dict with ``auto_budget_kwh``, ``method``, ``reference_months``,
        ``seasonal_factor``.
    """
    if not months_data:
        return {"auto_budget_kwh": 0, "method": "none", "reference_months": []}

    now = datetime.datetime.now(datetime.UTC)
    current_month_num = now.month
    current_ym = now.strftime("%Y-%m")

    # Group historical months by month number (Jan=1, Dec=12)
    by_month_num: dict[int, list[float]] = {}
    all_kwh: list[float] = []
    for m in months_data:
        ym = m.get("month", "")
        kwh = m.get("kwh", 0)
        if not ym or kwh <= 0 or ym == current_ym:
            continue
        try:
            month_num = int(ym.split("-")[1])
        except (ValueError, IndexError):
            continue
        by_month_num.setdefault(month_num, []).append(kwh)
        all_kwh.append(kwh)

    # Method 1: Same calendar month in previous years
    same_month_history = by_month_num.get(current_month_num, [])
    if same_month_history:
        avg = sum(same_month_history) / len(same_month_history)
        return {
            "auto_budget_kwh": round(avg, 1),
            "method": "same_month_avg",
            "reference_months": [
                m["month"]
                for m in months_data
                if m["month"].endswith(f"-{current_month_num:02d}") and m["month"] != current_ym
            ],
            "reference_values": [round(v, 1) for v in same_month_history],
        }

    # Method 2: Trailing 3-month average (seasonal approximation)
    completed = [m for m in months_data if m.get("kwh", 0) > 0 and m.get("month", "") != current_ym]
    completed.sort(key=lambda x: x["month"], reverse=True)
    recent_3 = completed[:3]
    if recent_3:
        avg = sum(m["kwh"] for m in recent_3) / len(recent_3)
        return {
            "auto_budget_kwh": round(avg, 1),
            "method": "trailing_3mo_avg",
            "reference_months": [m["month"] for m in recent_3],
            "reference_values": [round(m["kwh"], 1) for m in recent_3],
        }

    # Method 3: Overall average
    if all_kwh:
        avg = sum(all_kwh) / len(all_kwh)
        return {
            "auto_budget_kwh": round(avg, 1),
            "method": "overall_avg",
            "reference_months": [m["month"] for m in months_data if m["month"] != current_ym],
            "reference_values": [round(v, 1) for v in all_kwh],
        }

    return {"auto_budget_kwh": 0, "method": "none", "reference_months": []}


def _seasonal_profile(months_data: list[dict]) -> dict[str, Any]:
    """Build a seasonal consumption profile from historical data.

    Returns:
        Dict mapping month names to average kWh, plus highest/lowest months.
    """
    month_names = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    by_month: dict[int, list[float]] = {}
    for m in months_data:
        kwh = m.get("kwh", 0)
        if kwh <= 0:
            continue
        try:
            month_num = int(m["month"].split("-")[1])
        except (ValueError, IndexError):
            continue
        by_month.setdefault(month_num, []).append(kwh)

    profile: dict[str, float] = {}
    for mn, values in sorted(by_month.items()):
        profile[month_names[mn]] = round(sum(values) / len(values), 1)

    if not profile:
        return {"profile": {}, "highest": "", "lowest": ""}

    highest = max(profile, key=profile.get)  # type: ignore[arg-type]
    lowest = min(profile, key=profile.get)  # type: ignore[arg-type]
    return {
        "profile": profile,
        "highest_month": highest,
        "highest_kwh": profile[highest],
        "lowest_month": lowest,
        "lowest_kwh": profile[lowest],
    }


def set_budget(
    monthly_kwh: float | None = None,
    monthly_cost: float | None = None,
) -> dict[str, Any]:
    """Set a manual budget override, or clear to use auto-budget.

    Pass ``monthly_kwh=0`` to clear the manual override and revert to
    the smart auto-budget.

    Args:
        monthly_kwh: Manual monthly energy budget in kWh. 0 = use auto.
        monthly_cost: Monthly cost budget (converted to kWh via KWH_PRICE).

    Returns:
        The persisted budget configuration dict.
    """
    budget = _load_budget()

    if monthly_kwh is not None:
        if monthly_kwh <= 0:
            budget.pop("manual_kwh", None)
        else:
            budget["manual_kwh"] = monthly_kwh

    if monthly_cost is not None and monthly_cost > 0:
        budget["manual_kwh"] = monthly_cost / KWH_PRICE

    budget["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    atomic_write(BUDGET_PATH, budget)
    log.info("Energy budget updated: %s", budget)
    return budget


def get_budget_status() -> dict[str, Any]:
    """Compute current budget status with smart auto-budget and projections.

    Reads HA Energy Dashboard data from phantom_loads.json. If no manual
    budget is set, auto-computes from historical same-month averages.

    Returns:
        Comprehensive budget status with auto-budget, actual consumption,
        projections, seasonal context, and comparison to history.
    """
    budget_cfg = _load_budget()
    phantom = _load_phantom()
    months_data = phantom.get("months", [])

    now = datetime.datetime.now(datetime.UTC)
    current_ym = now.strftime("%Y-%m")
    day_of_month = now.day
    days_in_month = (
        datetime.date(now.year + (now.month // 12), (now.month % 12) + 1, 1)
        - datetime.date(now.year, now.month, 1)
    ).days
    days_remaining = days_in_month - day_of_month

    # Smart auto-budget from historical data
    smart = _compute_smart_budget(months_data)
    auto_budget_kwh = smart.get("auto_budget_kwh", 0)

    # Use manual override if set, otherwise auto-budget
    manual_kwh = budget_cfg.get("manual_kwh")
    budget_kwh = manual_kwh if manual_kwh else auto_budget_kwh
    budget_source = "manual" if manual_kwh else smart.get("method", "none")

    # Actual consumption this month (from Energy Dashboard)
    this_month_kwh = phantom.get("this_month_kwh", 0)
    this_month_avg_daily = phantom.get("this_month_avg_daily", 0)
    last_month_kwh = phantom.get("last_month_kwh", 0)

    # Day-normalized comparison (same first N days)
    same_days = phantom.get("same_days_comparison", {})
    same_days_this = same_days.get("this_month_first_n", 0)
    same_days_last = same_days.get("last_month_first_n", 0)
    same_days_delta_pct = same_days.get("delta_pct")

    # Projections
    projected_total = round(this_month_avg_daily * days_in_month, 1) if this_month_avg_daily else 0
    remaining_kwh = max(0, budget_kwh - this_month_kwh) if budget_kwh else 0
    overshoot_kwh = max(0, projected_total - budget_kwh) if budget_kwh else 0
    pct_used = round(this_month_kwh / budget_kwh * 100, 1) if budget_kwh > 0 else 0
    pct_month = round(day_of_month / days_in_month * 100, 1) if days_in_month else 0
    on_track = projected_total <= budget_kwh if budget_kwh else True

    # Pace comparison: are we burning faster/slower than budget implies?
    budget_daily_target = budget_kwh / days_in_month if budget_kwh and days_in_month else 0
    pace_vs_target = ""
    if budget_daily_target and this_month_avg_daily:
        ratio = this_month_avg_daily / budget_daily_target
        if ratio > 1.15:
            pace_vs_target = f"Burning {(ratio - 1) * 100:.0f}% faster than budget pace"
        elif ratio < 0.85:
            pace_vs_target = f"Running {(1 - ratio) * 100:.0f}% under budget pace"
        else:
            pace_vs_target = "On pace with budget"

    # Cost estimates
    cost_so_far = round(this_month_kwh * KWH_PRICE, 1)
    cost_projected = round(projected_total * KWH_PRICE, 1)
    cost_budget = round(budget_kwh * KWH_PRICE, 1) if budget_kwh else 0

    # Seasonal context
    seasonal = _seasonal_profile(months_data)

    # Month-over-month trend
    trend = ""
    if same_days_delta_pct is not None:
        if same_days_delta_pct > 10:
            trend = f"Up {same_days_delta_pct:.0f}% vs same period last month"
        elif same_days_delta_pct < -10:
            trend = f"Down {abs(same_days_delta_pct):.0f}% vs same period last month"
        else:
            trend = "Roughly flat vs last month"

    return {
        "budget_kwh": round(budget_kwh, 1) if budget_kwh else None,
        "budget_source": budget_source,
        "auto_budget_kwh": round(auto_budget_kwh, 1),
        "smart_budget": smart,
        "used_kwh": round(this_month_kwh, 1),
        "remaining_kwh": round(remaining_kwh, 1),
        "daily_avg": round(this_month_avg_daily, 2),
        "budget_daily_target": round(budget_daily_target, 2),
        "projected_total": projected_total,
        "on_track": on_track,
        "pct_used": pct_used,
        "pct_month_elapsed": pct_month,
        "overshoot_kwh": round(overshoot_kwh, 1),
        "days_remaining": days_remaining,
        "pace": pace_vs_target,
        "trend": trend,
        "cost": {
            "so_far": cost_so_far,
            "projected": cost_projected,
            "budget": cost_budget,
            "currency": CURRENCY,
            "kwh_price": KWH_PRICE,
        },
        "vs_last_month": {
            "last_month_kwh": round(last_month_kwh, 1),
            "same_days_this": round(same_days_this, 1),
            "same_days_last": round(same_days_last, 1),
            "delta_pct": same_days_delta_pct,
        },
        "seasonal": seasonal,
        "month": current_ym,
        "day_of_month": day_of_month,
        "days_in_month": days_in_month,
    }


def should_alert(status: dict[str, Any]) -> bool:
    """Check whether a budget alert should fire.

    Returns True if projected total exceeds budget by >10%.
    """
    budget = status.get("budget_kwh") or 0
    projected = status.get("projected_total") or 0
    if budget <= 0:
        return False
    return bool(projected > budget * 1.10)


def format_budget_message(status: dict[str, Any]) -> str:
    """Format a plain-English budget status message."""
    if not status.get("budget_kwh"):
        return "No budget data — waiting for Energy Dashboard history."

    cost = status.get("cost", {})
    parts = [
        f"Energy: {status['used_kwh']:.0f} / {status['budget_kwh']:.0f} kWh "
        f"({status['pct_used']:.0f}%)",
    ]
    if status.get("pace"):
        parts.append(status["pace"])
    if cost.get("projected"):
        parts.append(f"Projected cost: {cost['projected']} {cost.get('currency', 'kr')}")
    if status.get("trend"):
        parts.append(status["trend"])
    if not status.get("on_track"):
        parts.append(f"Over budget by ~{status['overshoot_kwh']:.0f} kWh")
    return ". ".join(parts) + "."
