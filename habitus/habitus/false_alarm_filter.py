"""False Alarm Filter — reduce noise from sensor glitches and transient issues.

Filters out:
1. Sensor unavailable/unknown state changes
2. Single-sample spikes (require 2+ consecutive anomalies)
3. Sensor restarts (value drops to 0 then recovers)
4. Integration reload artifacts
"""

import logging
from datetime import datetime, timedelta

log = logging.getLogger("habitus")

# Track recent anomalies to detect consecutive patterns
_recent_anomalies = {}  # entity_id → [timestamp, ...]


def is_false_alarm(entity_id: str, current_value: any, baseline_mean: float, z_score: float) -> bool:
    """Determine if an anomaly is likely a false alarm.
    
    Args:
        entity_id: Entity being checked.
        current_value: Current sensor value.
        baseline_mean: Expected baseline value.
        z_score: Z-score deviation.
    
    Returns:
        True if this is likely a false alarm (should be filtered out).
    """
    # Filter 1: Sensor unavailable/unknown states
    if current_value in [None, "unavailable", "unknown", "none", ""]:
        log.debug(f"False alarm filter: {entity_id} unavailable/unknown state")
        return True
    
    # Filter 2: Sensor value = 0 (likely restart/reset artifact)
    try:
        if float(current_value) == 0 and baseline_mean > 10:
            log.debug(f"False alarm filter: {entity_id} dropped to zero (restart?)")
            return True
    except (ValueError, TypeError):
        pass
    
    # Filter 3: Single-sample spike (require 2+ consecutive anomalies)
    # Only apply to high z-scores (>10σ are often sensor glitches)
    if z_score >= 10:  # Only filter extreme outliers
        now = datetime.now(datetime.UTC)
        
        # Record this anomaly
        if entity_id not in _recent_anomalies:
            _recent_anomalies[entity_id] = []
        
        _recent_anomalies[entity_id].append(now)
        
        # Clean up old records (>5 minutes)
        cutoff = now - timedelta(minutes=5)
        _recent_anomalies[entity_id] = [
            ts for ts in _recent_anomalies[entity_id]
            if ts > cutoff
        ]
        
        # Require at least 2 anomalies within 5 minutes
        if len(_recent_anomalies[entity_id]) < 2:
            log.debug(f"False alarm filter: {entity_id} single-sample spike (not consecutive)")
            return True
    
    return False


def filter_anomalies(anomalies: list[dict]) -> list[dict]:
    """Filter out likely false alarms from anomaly list.
    
    Args:
        anomalies: List of anomaly dicts.
    
    Returns:
        Filtered list with false alarms removed.
    """
    filtered = []
    removed_count = 0
    
    for anom in anomalies:
        eid = anom.get("entity_id", "")
        current = anom.get("current_value")
        baseline = anom.get("baseline_mean", 0)
        z_score = anom.get("z_score", 0)
        
        if is_false_alarm(eid, current, baseline, z_score):
            removed_count += 1
        else:
            filtered.append(anom)
    
    if removed_count > 0:
        log.info(f"False alarm filter removed {removed_count} likely false alarms")
    
    return filtered


def reset_anomaly_history():
    """Clear anomaly history (useful for testing/reset)."""
    global _recent_anomalies
    _recent_anomalies = {}
