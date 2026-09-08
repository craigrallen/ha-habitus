"""Anonymous benchmark comparison system.

Computes household energy and behavioral metrics, anonymises them,
and compares against hardcoded reference ranges for different home
sizes.  Supports optional anonymous sharing if the user has opted in
via the feedback module.
"""

import json
import logging
import math
import os
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
BENCHMARKS_PATH = os.path.join(DATA_DIR, "community_benchmarks.json")
PHANTOM_PATH = os.path.join(DATA_DIR, "phantom_loads.json")
ENTITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")
ENTITY_BASELINES_PATH = os.path.join(DATA_DIR, "entity_baselines.json")

# Reference overnight baseline wattage ranges by home size
REFERENCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "studio": {
        "overnight_baseline_w": (100.0, 200.0),
        "daily_avg_kwh": (5.0, 12.0),
    },
    "apartment": {
        "overnight_baseline_w": (150.0, 300.0),
        "daily_avg_kwh": (10.0, 20.0),
    },
    "house": {
        "overnight_baseline_w": (200.0, 500.0),
        "daily_avg_kwh": (15.0, 35.0),
    },
    "large_house": {
        "overnight_baseline_w": (400.0, 800.0),
        "daily_avg_kwh": (30.0, 60.0),
    },
}


def _load_json(path: str) -> Any:
    """Load a JSON file, returning an empty dict on failure.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        Parsed JSON content, or ``{}`` if the file is missing or corrupt.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return {}


def compute_benchmarks() -> dict:
    """Compute household energy and behavioral metrics.

    Reads phantom loads, entity baselines, and entity anomalies to
    build a summary of key household metrics.

    Returns:
        Dict with ``overnight_baseline_w``, ``daily_avg_kwh``,
        ``entities_tracked``, ``anomaly_rate_per_week``, and
        ``top_consumer_w``.
    """
    phantom = _load_json(PHANTOM_PATH)
    baselines = _load_json(ENTITY_BASELINES_PATH)
    anomalies = _load_json(ENTITY_ANOMALIES_PATH)

    # Overnight baseline from phantom data
    overnight_w = phantom.get("overnight_baseline_watts", 0.0)
    if not overnight_w:
        # Fall back: sum standby watts from devices
        devices = phantom.get("devices", [])
        overnight_w = sum(d.get("standby_watts", 0.0) for d in devices)

    # Daily average kWh
    monthly_kwh = phantom.get("total_monthly_kwh", 0.0)
    daily_avg_kwh = monthly_kwh / 30.0 if monthly_kwh else (overnight_w * 24.0 / 1000.0)

    # Entities tracked
    entities_tracked = len(baselines) if isinstance(baselines, dict) else 0

    # Anomaly rate per week
    anomaly_list = anomalies if isinstance(anomalies, list) else anomalies.get("anomalies", [])
    total_anomalies = len(anomaly_list) if isinstance(anomaly_list, list) else 0
    # Assume anomaly data covers roughly 7 days
    anomaly_rate = float(total_anomalies)

    # Top consumer
    top_consumer_w = 0.0
    devices = phantom.get("devices", [])
    if devices:
        top_consumer_w = max((d.get("standby_watts", 0.0) for d in devices), default=0.0)

    benchmarks = {
        "overnight_baseline_w": round(overnight_w, 1),
        "daily_avg_kwh": round(daily_avg_kwh, 1),
        "entities_tracked": entities_tracked,
        "anomaly_rate_per_week": round(anomaly_rate, 1),
        "top_consumer_w": round(top_consumer_w, 1),
    }

    atomic_write(BENCHMARKS_PATH, benchmarks)
    log.info("Community benchmarks computed: %s", benchmarks)
    return benchmarks


def anonymize(benchmarks: dict) -> dict:
    """Strip identifying info and round values for anonymous sharing.

    Rounds numeric values to the nearest 10 to prevent fingerprinting.

    Args:
        benchmarks: Raw benchmarks dict from :func:`compute_benchmarks`.

    Returns:
        Anonymised copy with rounded values and no identifying keys.
    """
    result: dict[str, Any] = {}
    for key, value in benchmarks.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Round to nearest 10
            result[key] = int(math.ceil(value / 10.0) * 10) if value > 0 else 0
        elif isinstance(value, str):
            # Strip string fields that might identify the household
            continue
        else:
            result[key] = value
    return result


def _estimate_percentile(value: float, ref_low: float, ref_high: float) -> int:
    """Estimate a rough percentile within a reference range.

    Args:
        value: The household's metric value.
        ref_low: Lower bound of the reference range.
        ref_high: Upper bound of the reference range.

    Returns:
        Estimated percentile (0-100).
    """
    if ref_high <= ref_low:
        return 50
    if value <= ref_low:
        return max(5, int((value / ref_low) * 25))
    if value >= ref_high:
        return min(95, int(75 + (value - ref_high) / ref_high * 25))
    # Linear interpolation within range
    fraction = (value - ref_low) / (ref_high - ref_low)
    return int(25 + fraction * 50)


def compare_to_reference(
    benchmarks: dict,
    home_size: str = "house",
) -> dict:
    """Compare household metrics against reference ranges.

    Args:
        benchmarks: Dict from :func:`compute_benchmarks`.
        home_size: One of ``"studio"``, ``"apartment"``, ``"house"``,
            ``"large_house"``.

    Returns:
        Dict mapping each comparable metric to its value, reference
        range, percentile estimate, and status label.
    """
    refs = REFERENCE_RANGES.get(home_size, REFERENCE_RANGES["house"])
    comparisons: dict[str, dict[str, Any]] = {}

    for metric, (ref_low, ref_high) in refs.items():
        value = benchmarks.get(metric, 0.0)
        percentile = _estimate_percentile(value, ref_low, ref_high)

        if percentile < 35:
            status = "below_avg"
        elif percentile > 65:
            status = "above_avg"
        else:
            status = "average"

        comparisons[metric] = {
            "metric": metric,
            "your_value": value,
            "reference_range": [ref_low, ref_high],
            "home_size": home_size,
            "percentile_estimate": percentile,
            "status": status,
        }

    return comparisons


def get_sharing_payload() -> dict | None:
    """Return anonymised benchmark data if sharing is enabled.

    Checks the feedback module's sharing flag.  Returns ``None`` if
    sharing is disabled or feedback data is unavailable.

    Returns:
        Anonymised benchmark dict, or ``None``.
    """
    # Import here to avoid circular imports
    from .feedback import _load_feedback

    feedback = _load_feedback()
    if not feedback.get("sharing_enabled", False):
        log.debug("Anonymous sharing is disabled — not generating payload")
        return None

    benchmarks = compute_benchmarks()
    payload = anonymize(benchmarks)
    payload["source"] = "habitus"
    payload["version"] = "anonymous"
    log.info("Sharing payload generated with %d metrics", len(payload))
    return payload
