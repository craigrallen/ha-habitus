"""Anomaly clustering — group simultaneous anomalies into incidents.

Reduces alert fatigue by clustering entity-level anomalies that fire
within the same time window into a single "incident" with a probable
root cause label.
"""

import contextlib
import datetime
import json
import logging
import os
import uuid
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
INCIDENTS_PATH = os.path.join(DATA_DIR, "anomaly_incidents.json")
ENTITY_ANOMALIES_PATH = os.path.join(DATA_DIR, "entity_anomalies.json")


# ---------------------------------------------------------------------------
# Cause inference heuristics
# ---------------------------------------------------------------------------

_POWER_KEYWORDS = ("power", "energy", "watt", "kwh", "consumption", "load")
_MOTION_KEYWORDS = ("motion", "occupancy", "presence", "pir")
_LIGHT_KEYWORDS = ("light", "lamp", "brightness", "lux")
_DOOR_KEYWORDS = ("door", "window", "lock", "contact")
_CLIMATE_KEYWORDS = ("temperature", "humidity", "climate", "thermostat", "hvac")


def _has_keyword(entity_id: str, keywords: tuple[str, ...]) -> bool:
    """Check if an entity ID contains any of the given keywords.

    Args:
        entity_id: HA entity ID string.
        keywords: Tuple of lowercase substrings to search for.

    Returns:
        ``True`` if any keyword appears in the lowered entity ID.
    """
    lower = entity_id.lower()
    return any(kw in lower for kw in keywords)


def _infer_cause(entities: list[dict]) -> str:
    """Infer a probable root cause from the set of anomalous entities.

    Uses simple heuristics based on which entity domains / keywords
    appear together.

    Args:
        entities: List of entity anomaly dicts, each with at least
            an ``entity_id`` key.

    Returns:
        Human-readable cause string.
    """
    ids = [e.get("entity_id", "") for e in entities]

    power_count = sum(1 for eid in ids if _has_keyword(eid, _POWER_KEYWORDS))
    motion_count = sum(1 for eid in ids if _has_keyword(eid, _MOTION_KEYWORDS))
    light_count = sum(1 for eid in ids if _has_keyword(eid, _LIGHT_KEYWORDS))
    door_count = sum(1 for eid in ids if _has_keyword(eid, _DOOR_KEYWORDS))
    climate_count = sum(1 for eid in ids if _has_keyword(eid, _CLIMATE_KEYWORDS))

    total = len(ids)

    # All or mostly power entities
    if power_count > 0 and power_count >= total * 0.6:
        return "power event"

    # Motion + lights together suggest arrival
    if motion_count > 0 and light_count > 0:
        return "arrival or departure"

    # Motion + doors
    if motion_count > 0 and door_count > 0:
        return "arrival or departure"

    # Doors only
    if door_count > 0 and door_count >= total * 0.5:
        return "door/window activity"

    # Climate sensors
    if climate_count > 0 and climate_count >= total * 0.5:
        return "climate change"

    # Mixed bag — could be a system restart or broad event
    if total >= 4:
        return "system restart or broad event"

    return "unknown"


def _compute_severity(entities: list[dict]) -> str:
    """Compute incident severity from entity z-scores.

    Args:
        entities: List of entity anomaly dicts with optional ``z_score``
            or ``score`` fields.

    Returns:
        One of ``"low"``, ``"medium"``, ``"high"``.
    """
    scores: list[float] = []
    for e in entities:
        z = e.get("z_score", e.get("score", 0.0))
        with contextlib.suppress(ValueError, TypeError):
            scores.append(float(z))

    if not scores:
        return "low"

    max_score = max(abs(s) for s in scores)
    if max_score >= 4.0:
        return "high"
    if max_score >= 2.5:
        return "medium"
    return "low"


def _parse_timestamp(ts_value: Any) -> datetime.datetime | None:
    """Parse a timestamp from various formats.

    Args:
        ts_value: ISO string, epoch float, or datetime.

    Returns:
        Timezone-aware datetime, or ``None`` if unparseable.
    """
    if isinstance(ts_value, datetime.datetime):
        if ts_value.tzinfo is None:
            return ts_value.replace(tzinfo=datetime.UTC)
        return ts_value

    if isinstance(ts_value, (int, float)):
        return datetime.datetime.fromtimestamp(ts_value, tz=datetime.UTC)

    if isinstance(ts_value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(ts_value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.UTC)
                return dt
            except ValueError:
                continue
    return None


def cluster_anomalies(
    entity_anomalies: list[dict],
    time_window_min: int = 15,
) -> list[dict]:
    """Group entity anomalies that occurred within the same time window.

    Anomalies are sorted by timestamp, then greedily assigned to
    clusters: each new anomaly joins the current cluster if it falls
    within *time_window_min* of the cluster's first timestamp, otherwise
    a new cluster is started.

    Args:
        entity_anomalies: List of per-entity anomaly dicts.  Each must
            have ``entity_id`` and ``timestamp`` keys.
        time_window_min: Maximum minutes between first and last anomaly
            in a cluster.

    Returns:
        List of incident dicts, each containing ``incident_id``,
        ``timestamp``, ``entities``, ``probable_cause``, ``severity``,
        and ``description``.
    """
    if not entity_anomalies:
        return []

    # Attach parsed datetimes
    timed: list[tuple[datetime.datetime, dict]] = []
    for a in entity_anomalies:
        ts = _parse_timestamp(a.get("timestamp"))
        if ts is not None:
            timed.append((ts, a))

    if not timed:
        log.warning("No parseable timestamps in %d anomalies", len(entity_anomalies))
        return []

    timed.sort(key=lambda x: x[0])
    window = datetime.timedelta(minutes=time_window_min)

    clusters: list[list[dict]] = []
    cluster_start: datetime.datetime = timed[0][0]
    current_cluster: list[dict] = []

    for ts, anomaly in timed:
        if ts - cluster_start <= window:
            current_cluster.append(anomaly)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [anomaly]
            cluster_start = ts

    if current_cluster:
        clusters.append(current_cluster)

    # Build incident records
    incidents: list[dict] = []
    for cluster in clusters:
        cause = _infer_cause(cluster)
        severity = _compute_severity(cluster)
        entity_ids = [e.get("entity_id", "unknown") for e in cluster]
        first_ts = _parse_timestamp(cluster[0].get("timestamp"))
        ts_str = first_ts.isoformat() if first_ts else "unknown"

        incident = {
            "incident_id": uuid.uuid4().hex[:12],
            "timestamp": ts_str,
            "entities": entity_ids,
            "entity_count": len(cluster),
            "probable_cause": cause,
            "severity": severity,
            "description": (
                f"{len(cluster)} anomalous entities within "
                f"{time_window_min} min — probable cause: {cause}"
            ),
        }
        incidents.append(incident)

    # Persist
    atomic_write(INCIDENTS_PATH, incidents)
    log.info("Clustered %d anomalies into %d incidents", len(entity_anomalies), len(incidents))
    return incidents


def get_incidents(hours: int = 24) -> list[dict]:
    """Load recent incidents from disk.

    Args:
        hours: Only return incidents from the last *hours* hours.

    Returns:
        List of incident dicts, filtered by recency.
    """
    if not os.path.exists(INCIDENTS_PATH):
        return []

    try:
        with open(INCIDENTS_PATH) as fh:
            all_incidents: list[dict] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load anomaly_incidents.json: %s", exc)
        return []

    if not isinstance(all_incidents, list):
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=hours)
    recent: list[dict] = []
    for inc in all_incidents:
        ts = _parse_timestamp(inc.get("timestamp"))
        if ts is not None and ts >= cutoff:
            recent.append(inc)

    return recent
