"""Shared entity identification and explicit unit conversion."""

from __future__ import annotations

import json
import os
import re

SCHEMA_VERSION = 2
UNIT_KINDS = {
    "W": "power",
    "kW": "power",
    "MW": "power",
    "Wh": "energy",
    "kWh": "energy",
    "MWh": "energy",
    "J": "energy",
    "°C": "temperature",
    "°F": "temperature",
    "K": "temperature",
    "A": "current",
    "mA": "current",
    "V": "voltage",
    "mV": "voltage",
    "Pa": "pressure",
    "hPa": "pressure",
    "kPa": "pressure",
    "bar": "pressure",
    "psi": "pressure",
    "L": "volume",
    "m³": "volume",
    "gal": "volume",
}


def identify(entity_id: str, attributes: dict | None = None) -> dict:
    """Resolve an entity using explicit attributes before conservative name hints.

    Args:
        entity_id: Home Assistant entity or statistic identifier.
        attributes: Home Assistant attributes or previously resolved metadata.

    Returns:
        Metadata retaining the reported unit and the identification source.
    """
    attrs = dict(attributes or {})
    explicit = "unit_of_measurement" in attrs
    unit = attrs.get("unit_of_measurement") or ""
    device_class = attrs.get("device_class") or ""
    kind = device_class or UNIT_KINDS.get(unit, "unknown")
    source = attrs.get("identification_source") or (
        "metadata" if explicit or device_class else "name"
    )
    if not explicit and not device_class:
        tokens = set(re.split(r"[._]", entity_id.lower()))
        for token, guessed_unit, guessed_kind in (
            ("kwh", "kWh", "energy"),
            ("wh", "Wh", "energy"),
            ("kw", "kW", "power"),
            ("w", "W", "power"),
            ("watt", "W", "power"),
            ("watts", "W", "power"),
            ("temperature", "°C", "temperature"),
            ("temp", "°C", "temperature"),
            ("humidity", "%", "humidity"),
            ("voltage", "V", "voltage"),
            ("current", "A", "current"),
            ("pressure", "hPa", "pressure"),
        ):
            if token in tokens:
                unit, kind = guessed_unit, guessed_kind
                break
    if kind == "unknown" and not unit:
        source = "unknown"
    attrs.update(unit_of_measurement=unit, quantity=kind, identification_source=source)
    return attrs


def read_overrides(data_dir: str) -> dict:
    """Read locally configured sensor corrections.

    Args:
        data_dir: Add-on data directory.

    Returns:
        Entity overrides, or an empty mapping if unavailable.
    """
    try:
        with open(os.path.join(data_dir, "entity_overrides.json"), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def apply_override(attributes: dict, override: dict | None) -> dict:
    """Apply a persistent correction to entity attributes.

    Args:
        attributes: Reported metadata.
        override: User correction, if any.

    Returns:
        Corrected metadata with its source recorded.
    """
    if not override:
        return dict(attributes)
    corrected = {
        **attributes,
        **override,
        "identification_source": "override",
        "identification_override": override,
    }
    if "unit_of_measurement" in override and override["unit_of_measurement"] in UNIT_KINDS:
        corrected["device_class"] = UNIT_KINDS[override["unit_of_measurement"]]
    return corrected


def power_factor(metadata: dict) -> float | None:
    """Return the multiplier to watts for a confirmed power measurement.

    Args:
        metadata: Resolved entity metadata.

    Returns:
        Scale to watts, or None for unsupported or conflicting measurements.
    """
    if metadata.get("quantity") != "power":
        return None
    return {"W": 1.0, "kW": 1000.0, "MW": 1_000_000.0}.get(metadata.get("unit_of_measurement", ""))


def convert_value(value: float, source: str, target: str) -> float:
    """Convert compatible live readings into the historical baseline unit.

    Args:
        value: Source reading.
        source: Live unit.
        target: Baseline unit.

    Returns:
        Converted reading.

    Raises:
        ValueError: If units are unknown or incompatible.
    """
    if source == target:
        return value
    if source in ("°C", "°F", "K") and target in ("°C", "°F", "K"):
        celsius = temperature_c(value, source)
        return (
            celsius * 9 / 5 + 32
            if target == "°F"
            else celsius + 273.15 if target == "K" else celsius
        )
    for scales in (
        {"W": 1, "kW": 1000, "MW": 1_000_000},
        {"Wh": 1, "kWh": 1000, "MWh": 1_000_000, "J": 1 / 3600},
        {"A": 1, "mA": 0.001},
        {"V": 1, "mV": 0.001},
        {"L": 1, "m³": 1000},
        {"Pa": 1, "hPa": 100, "kPa": 1000, "bar": 100_000},
    ):
        if source in scales and target in scales:
            return value * scales[source] / scales[target]
    raise ValueError(f"Cannot compare {source!r} readings against {target!r} baseline")


def temperature_c(value: float, unit: str) -> float:
    """Convert a supported temperature to Celsius.

    Args:
        value: Temperature reading.
        unit: Explicit source unit.

    Returns:
        Celsius reading.

    Raises:
        ValueError: If the unit is unsupported.
    """
    if unit == "°C":
        return value
    if unit == "°F":
        return (value - 32) * 5 / 9
    if unit == "K":
        return value - 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")
