"""Cost Estimator — estimate energy savings for suggestions and automations.

Uses energy tariff config (default 0.30 EUR/kWh, peak 0.45, off-peak 0.15).
Estimates wattage from NILM data or domain defaults.
Computes monthly_saving_eur and annual_saving_eur.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _get_data_dir() -> str:
    return os.environ.get("DATA_DIR", DATA_DIR)

SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
NILM_PATH = os.path.join(DATA_DIR, "nilm.json")

# Default tariff (EUR/kWh)
DEFAULT_TARIFF = 0.30
DEFAULT_PEAK_TARIFF = 0.45
DEFAULT_OFFPEAK_TARIFF = 0.15

# Default wattage by domain (W)
DOMAIN_DEFAULT_WATTS: dict[str, int] = {
    "light": 10,
    "switch": 50,
    "media_player": 100,
    "climate": 1500,
    "washer": 2000,
    "kettle": 2000,
    "dishwasher": 1800,
    "dryer": 2500,
    "fan": 50,
    "heater": 1000,
    "tv": 100,
    "standby": 5,
    "vacuum": 30,
    "default": 50,
}

# Keywords for domain inference from entity_id
KEYWORD_WATT_MAP: dict[str, int] = {
    "kettle": 2000,
    "washer": 2000,
    "washing_machine": 2000,
    "dryer": 2500,
    "dishwasher": 1800,
    "oven": 2200,
    "microwave": 1200,
    "tv": 100,
    "television": 100,
    "heater": 1000,
    "radiator": 800,
    "fan": 50,
    "vacuum": 30,
    "charger": 20,
    "fridge": 60,
    "freezer": 80,
    "standby": 5,
}


def _load_settings() -> dict[str, Any]:
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "settings.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "settings.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _load_nilm() -> dict[str, Any]:
    try:
        if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "nilm.json")):
            with open(os.path.join(os.environ.get("DATA_DIR", "/data"), "nilm.json")) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_tariff_config(settings: dict[str, Any] | None = None) -> dict[str, float]:
    """Get energy tariff config from settings or defaults."""
    if settings is None:
        settings = _load_settings()

    return {
        "standard": float(settings.get("energy_tariff", DEFAULT_TARIFF)),
        "peak": float(settings.get("energy_tariff_peak", DEFAULT_PEAK_TARIFF)),
        "off_peak": float(settings.get("energy_tariff_offpeak", DEFAULT_OFFPEAK_TARIFF)),
    }


def estimate_watts(entity_id: str, nilm_data: dict[str, Any] | None = None) -> int:
    """Estimate wattage for an entity.

    Checks NILM data first, then keywords, then domain defaults.

    Args:
        entity_id: HA entity ID.
        nilm_data: NILM appliance data. If None, loads from NILM_PATH.

    Returns:
        Estimated wattage in watts.
    """
    if nilm_data is None:
        nilm_data = _load_nilm()

    # Check NILM appliance signatures
    appliances = nilm_data.get("appliances", [])
    for appl in appliances:
        if appl.get("entity_id") == entity_id:
            watt = appl.get("avg_watts") or appl.get("typical_watts")
            if watt:
                return int(watt)

    eid_lower = entity_id.lower()

    # Check keyword map
    for keyword, watts in KEYWORD_WATT_MAP.items():
        if keyword in eid_lower:
            return watts

    # Fall back to domain default
    domain = entity_id.split(".")[0]
    return DOMAIN_DEFAULT_WATTS.get(domain, DOMAIN_DEFAULT_WATTS["default"])


def compute_saving(
    entity_id: str,
    hours_saved_per_day: float,
    tariff_config: dict[str, float] | None = None,
    nilm_data: dict[str, Any] | None = None,
    peak: bool = False,
) -> dict[str, Any]:
    """Compute energy saving for reducing entity runtime.

    Args:
        entity_id: HA entity ID.
        hours_saved_per_day: How many hours per day the entity will run less.
        tariff_config: Tariff config dict. If None, loads from settings.
        nilm_data: NILM data for wattage lookup.
        peak: Whether to use peak tariff.

    Returns:
        Dict with watts, kwh_per_day, monthly_saving_eur, annual_saving_eur.
    """
    if tariff_config is None:
        tariff_config = get_tariff_config()

    watts = estimate_watts(entity_id, nilm_data)
    kwh_per_day = (watts * max(0.0, hours_saved_per_day)) / 1000.0

    tariff = tariff_config.get("peak" if peak else "standard", DEFAULT_TARIFF)
    daily_saving = kwh_per_day * tariff
    monthly_saving = daily_saving * 30
    annual_saving = daily_saving * 365

    # Payback estimation is not available without hardware cost
    return {
        "entity_id": entity_id,
        "estimated_watts": watts,
        "hours_saved_per_day": hours_saved_per_day,
        "kwh_saved_per_day": round(kwh_per_day, 4),
        "tariff_eur_per_kwh": tariff,
        "monthly_saving_eur": round(monthly_saving, 2),
        "annual_saving_eur": round(annual_saving, 2),
    }


def enrich_with_cost(
    items: list[dict[str, Any]],
    entity_field: str = "entity_id",
    hours_field: str = "estimated_hours_saved_per_day",
    default_hours: float = 1.0,
    settings: dict[str, Any] | None = None,
    nilm_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enrich a list of suggestions/automations with cost_estimate field.

    Args:
        items: List of suggestion/automation/gap dicts.
        entity_field: Field name for entity_id in each item.
        hours_field: Field name for hours_saved_per_day in each item.
        default_hours: Default hours saved per day if not specified.
        settings: Settings dict. If None, loads from SETTINGS_PATH.
        nilm_data: NILM data. If None, loads from NILM_PATH.

    Returns:
        Items list with cost_estimate added where applicable.
    """
    tariff_config = get_tariff_config(settings)
    if nilm_data is None:
        nilm_data = _load_nilm()

    enriched = []
    for item in items:
        item = dict(item)
        entity_id = item.get(entity_field, "")
        hours = float(item.get(hours_field, default_hours))

        if entity_id and hours > 0:
            domain = entity_id.split(".")[0]
            # Only add cost estimate for power-consuming domains
            if domain in ("light", "switch", "media_player", "climate", "fan", "script"):
                cost = compute_saving(entity_id, hours, tariff_config, nilm_data)
                if cost["monthly_saving_eur"] > 0:
                    item["cost_estimate"] = cost

        enriched.append(item)

    return enriched


def format_saving_badge(monthly_eur: float) -> str:
    """Format a saving as a human-readable badge string."""
    if monthly_eur <= 0:
        return ""
    if monthly_eur < 1:
        return f"Saves ~€{monthly_eur * 100:.0f}¢/month"
    return f"Saves ~€{monthly_eur:.1f}/month"
