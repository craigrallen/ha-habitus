"""Shore Power Quality — track voltage and frequency over time.

Monitors shore power quality by recording voltage/frequency readings,
detecting brownouts and voltage sags, and producing a quality score.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
QUALITY_PATH = os.path.join(DATA_DIR, "shore_power_quality.json")

# Rolling history window
_HISTORY_DAYS = 7

# Alert thresholds
_VOLTAGE_WARNING = 210.0
_VOLTAGE_CRITICAL = 200.0
_FREQ_LOW = 49.5
_FREQ_HIGH = 50.5
_SAG_PERCENT = 0.10  # 10% dip from baseline


def _load_data() -> dict[str, Any]:
    """Load quality data from disk.

    Returns:
        Dict with ``readings`` list and optional ``analysis``.
    """
    try:
        with open(QUALITY_PATH) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"readings": [], "analysis": None}


def _prune_old(readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove readings older than the history window.

    Args:
        readings: List of reading dicts with ``timestamp`` keys.

    Returns:
        Pruned list containing only recent readings.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=_HISTORY_DAYS)).isoformat()
    return [r for r in readings if r.get("timestamp", "") >= cutoff]


def record_reading(
    voltage: float,
    frequency: float | None = None,
    timestamp: str | None = None,
) -> None:
    """Append a voltage/frequency reading to rolling history.

    Args:
        voltage: Shore power voltage (e.g. 230.0).
        frequency: Grid frequency in Hz (e.g. 50.0). Optional.
        timestamp: ISO 8601 timestamp. Defaults to now (UTC).
    """
    data = _load_data()
    readings = _prune_old(data.get("readings", []))

    entry: dict[str, Any] = {
        "voltage": voltage,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
    }
    if frequency is not None:
        entry["frequency"] = frequency

    readings.append(entry)
    data["readings"] = readings
    atomic_write(QUALITY_PATH, data)
    log.debug("Recorded shore power: %.1fV", voltage)


def analyze_quality() -> dict[str, Any]:
    """Compute quality metrics from recorded readings.

    Calculates average/min/max voltage, stability percentage, brownout
    and sag event counts, frequency deviation, and an overall quality
    score from 0-100.

    Returns:
        Dict with keys: ``avg_voltage``, ``min_voltage``, ``max_voltage``,
        ``voltage_stability_pct``, ``brownout_count``, ``sag_events``,
        ``frequency_deviation``, ``quality_score``, ``alerts``,
        ``reading_count``.
    """
    data = _load_data()
    readings = _prune_old(data.get("readings", []))

    if not readings:
        return {
            "avg_voltage": 0.0,
            "min_voltage": 0.0,
            "max_voltage": 0.0,
            "voltage_stability_pct": 0.0,
            "brownout_count": 0,
            "sag_events": 0,
            "frequency_deviation": 0.0,
            "quality_score": 0,
            "alerts": ["No readings recorded"],
            "reading_count": 0,
        }

    voltages = [r["voltage"] for r in readings]
    avg_v = sum(voltages) / len(voltages)
    min_v = min(voltages)
    max_v = max(voltages)

    # Stability: percentage of readings within +/-5% of average
    tolerance = avg_v * 0.05
    stable_count = sum(1 for v in voltages if abs(v - avg_v) <= tolerance)
    stability_pct = round((stable_count / len(voltages)) * 100, 1)

    # Brownout: voltage < 200V
    brownout_count = sum(1 for v in voltages if v < _VOLTAGE_CRITICAL)

    # Sag events: voltage dip > 10% from baseline (average)
    sag_threshold = avg_v * (1 - _SAG_PERCENT)
    sag_events = sum(1 for v in voltages if v < sag_threshold)

    # Frequency deviation from 50 Hz
    frequencies = [r["frequency"] for r in readings if "frequency" in r]
    freq_deviation = 0.0
    if frequencies:
        avg_freq = sum(frequencies) / len(frequencies)
        freq_deviation = round(abs(avg_freq - 50.0), 3)

    # Quality score (0-100)
    score = 100.0
    # Penalise voltage instability
    score -= max(0, (100 - stability_pct) * 0.5)
    # Penalise brownouts heavily
    score -= brownout_count * 5
    # Penalise sags
    score -= sag_events * 2
    # Penalise frequency deviation
    score -= freq_deviation * 10
    # Penalise low average voltage
    if avg_v < _VOLTAGE_WARNING:
        score -= (_VOLTAGE_WARNING - avg_v) * 0.5
    quality_score = max(0, min(100, round(score)))

    # Generate alerts
    alerts: list[str] = []
    if min_v < _VOLTAGE_CRITICAL:
        alerts.append(f"CRITICAL: Voltage dropped to {min_v:.1f}V (< {_VOLTAGE_CRITICAL}V)")
    elif min_v < _VOLTAGE_WARNING:
        alerts.append(f"WARNING: Voltage dropped to {min_v:.1f}V (< {_VOLTAGE_WARNING}V)")
    if brownout_count > 0:
        alerts.append(f"Brownout events detected: {brownout_count}")
    if frequencies:
        for freq in frequencies:
            if freq < _FREQ_LOW or freq > _FREQ_HIGH:
                alerts.append(
                    f"WARNING: Frequency {freq:.2f}Hz outside " f"{_FREQ_LOW}-{_FREQ_HIGH}Hz range"
                )
                break

    analysis: dict[str, Any] = {
        "avg_voltage": round(avg_v, 1),
        "min_voltage": round(min_v, 1),
        "max_voltage": round(max_v, 1),
        "voltage_stability_pct": stability_pct,
        "brownout_count": brownout_count,
        "sag_events": sag_events,
        "frequency_deviation": freq_deviation,
        "quality_score": quality_score,
        "alerts": alerts,
        "reading_count": len(readings),
    }

    # Persist analysis alongside readings
    data["readings"] = readings
    data["analysis"] = analysis
    atomic_write(QUALITY_PATH, data)

    return analysis


def get_quality_report() -> dict[str, Any]:
    """Load the latest quality analysis from disk.

    Returns:
        Most recent analysis dict, or empty analysis if none exists.
    """
    data = _load_data()
    analysis: dict[str, Any] | None = data.get("analysis")
    if analysis:
        return analysis
    return analyze_quality()
