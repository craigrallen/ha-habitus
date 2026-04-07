"""CSV Export — download baselines, anomalies, and energy data as CSV.

Provides functions that read existing JSON artifacts and convert them
to CSV-formatted strings for the ``/api/export/<dataset>`` endpoint.
"""

import csv
import io
import json
import logging
import os
from typing import Any

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _load_json(path: str, default: Any = None) -> Any:
    """Read a JSON file, returning *default* on any error."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return default


def export_baselines() -> str:
    """Export entity baselines as CSV.

    Columns: entity_id, slot, mean, std, n_samples, sensor_type.

    Returns:
        CSV string with headers.
    """
    baselines = _load_json(os.path.join(DATA_DIR, "entity_baselines.json"), {})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["entity_id", "slot", "mean", "std", "n_samples", "sensor_type"])

    for eid, data in baselines.items():
        meta = data.get("_meta", {})
        sensor_type = meta.get("sensor_type", "gauge")
        slots = data.get("slots", {})
        for slot_key, slot_data in slots.items():
            writer.writerow(
                [
                    eid,
                    slot_key,
                    round(slot_data.get("mean", 0), 3),
                    round(slot_data.get("std", 0), 3),
                    slot_data.get("n", 0),
                    sensor_type,
                ]
            )

    return buf.getvalue()


def export_anomalies() -> str:
    """Export current entity anomalies as CSV.

    Columns: entity_id, name, z_score, current_value, baseline_mean,
    direction, confidence, severity.

    Returns:
        CSV string with headers.
    """
    from .nl_explanations import severity_label

    data = _load_json(os.path.join(DATA_DIR, "entity_anomalies.json"), {})
    anomalies = data.get("anomalies", data) if isinstance(data, dict) else data
    if isinstance(anomalies, dict):
        anomalies = anomalies.get("anomalies", [])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "entity_id",
            "name",
            "z_score",
            "current_value",
            "baseline_mean",
            "direction",
            "confidence",
            "severity",
        ]
    )

    for a in anomalies:
        writer.writerow(
            [
                a.get("entity_id", a.get("name", "")),
                a.get("name", ""),
                round(a.get("z_score", 0), 2),
                a.get("current_value", ""),
                round(a.get("baseline_mean", 0), 2) if a.get("baseline_mean") else "",
                a.get("direction", ""),
                round(a.get("confidence", 0), 2),
                severity_label(a.get("z_score", 0)),
            ]
        )

    return buf.getvalue()


def export_energy() -> str:
    """Export energy/phantom data as CSV.

    Columns: month, kwh.

    Returns:
        CSV string with headers.
    """
    phantom = _load_json(os.path.join(DATA_DIR, "phantom_loads.json"), {})
    months = phantom.get("months", [])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["month", "kwh"])
    for m in months:
        writer.writerow([m.get("month", ""), m.get("kwh", 0)])

    return buf.getvalue()


def export_anomaly_history() -> str:
    """Export 7-day anomaly history as CSV.

    Columns: timestamp, score, entity_count, top_entity, top_z_score.

    Returns:
        CSV string with headers.
    """
    history = _load_json(os.path.join(DATA_DIR, "anomaly_history.json"), [])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "score", "entity_count", "top_entity", "top_z_score"])

    for event in history:
        entities = event.get("entities", [])
        top = entities[0] if entities else {}
        writer.writerow(
            [
                event.get("timestamp", ""),
                event.get("anomaly_score", 0),
                event.get("entity_count", 0),
                top.get("entity_id", ""),
                round(top.get("z_score", 0), 2),
            ]
        )

    return buf.getvalue()


AVAILABLE_EXPORTS = {
    "baselines": ("entity_baselines.csv", export_baselines),
    "anomalies": ("anomalies.csv", export_anomalies),
    "energy": ("energy_monthly.csv", export_energy),
    "history": ("anomaly_history.csv", export_anomaly_history),
}
