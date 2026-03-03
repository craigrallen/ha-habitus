"""Non-Intrusive Load Monitoring (NILM) — appliance disaggregation.

Decomposes a single aggregate power meter reading into estimated
per-appliance consumption using a combinatorial optimisation approach
inspired by Hart (1992) and FHMM (Factorial HMM).

Algorithm:
1. Detect step changes (edges) in aggregate power signal
2. Pair ON/OFF edges by magnitude similarity → appliance events
3. Cluster events by power level → discover appliance "slots"
4. Match clusters against known signatures (generic + user-trained)
5. For each time window, estimate which appliances are ON and their contribution

No external NILM libraries needed — runs on numpy + sklearn (already installed).
Designed for 1Hz–1/60Hz data from a single aggregate meter on Odroid N2.
"""

import datetime
import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any

import numpy as np

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
NILM_PATH = os.path.join(DATA_DIR, "nilm_disaggregation.json")
CUSTOM_SIGS_PATH = os.path.join(DATA_DIR, "custom_signatures.json")
HA_DB = "/homeassistant/home-assistant_v2.db"

# Edge detection
MIN_EDGE_WATTS = 100  # Minimum step change to count as an appliance event
STEADY_STATE_SAMPLES = 3  # Samples to average for steady-state detection
MAX_PAIR_WINDOW_SEC = 28800  # 8 hours max ON duration

# Known appliance power levels (watts) for matching
# These are centroids — actual usage matched within ±30%
GENERIC_APPLIANCES = {
    "fridge_freezer": {"power": 120, "icon": "🧊", "always_on": True, "duty_cycle": 0.3},
    "router_modem": {"power": 15, "icon": "📶", "always_on": True, "duty_cycle": 1.0},
    "standby_cluster": {"power": 30, "icon": "🔌", "always_on": True, "duty_cycle": 1.0},
    "led_lighting": {"power": 50, "icon": "💡", "always_on": False},
    "tv_media": {"power": 150, "icon": "📺", "always_on": False},
    "kettle": {"power": 2000, "icon": "☕", "always_on": False},
    "oven": {"power": 2500, "icon": "🔥", "always_on": False},
    "hob_element": {"power": 1500, "icon": "🍳", "always_on": False},
    "washing_machine": {"power": 500, "icon": "👕", "always_on": False},
    "dishwasher": {"power": 1200, "icon": "🍽️", "always_on": False},
    "water_heater": {"power": 2800, "icon": "🚿", "always_on": False},
    "space_heater": {"power": 1000, "icon": "🌡️", "always_on": False},
    "heat_pump": {"power": 2000, "icon": "♨️", "always_on": False},
    "microwave": {"power": 1100, "icon": "📡", "always_on": False},
    "hair_dryer": {"power": 1500, "icon": "💇", "always_on": False},
    "vacuum": {"power": 800, "icon": "🧹", "always_on": False},
    "charger_laptop": {"power": 65, "icon": "💻", "always_on": False},
    "shore_charger": {"power": 1800, "icon": "🔋", "always_on": False},
}


def _get_aggregate_power(entity_id: str, days: int = 7) -> list[tuple[float, float]]:
    """Get aggregate power readings as (timestamp, watts) pairs."""
    if not os.path.exists(HA_DB):
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
        rows = conn.execute("""
            SELECT s.state, s.last_changed_ts FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE sm.entity_id = ? AND s.last_changed_ts > ?
            ORDER BY s.last_changed_ts
        """, (entity_id, cutoff_ts)).fetchall()
        conn.close()

        result = []
        for state_val, ts in rows:
            try:
                w = float(state_val)
                if 0 <= w <= 25000:
                    result.append((ts, w))
            except (ValueError, TypeError):
                continue
        return result
    except Exception as e:
        log.warning("nilm: DB read failed: %s", e)
        return []


def _detect_edges(readings: list[tuple[float, float]]) -> list[dict]:
    """Detect step changes (edges) in aggregate power signal.

    An edge = sudden change in power level that persists for at least
    STEADY_STATE_SAMPLES readings.
    """
    if len(readings) < STEADY_STATE_SAMPLES + 1:
        return []

    edges = []
    timestamps = [r[0] for r in readings]
    watts = [r[1] for r in readings]

    # Compute rolling average for smoothing
    window = min(STEADY_STATE_SAMPLES, len(watts))
    smoothed = np.convolve(watts, np.ones(window) / window, mode='valid')

    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i - 1]
        if abs(delta) >= MIN_EDGE_WATTS:
            ts_idx = i + window - 1
            if ts_idx < len(timestamps):
                edges.append({
                    "timestamp": timestamps[ts_idx],
                    "time": datetime.datetime.fromtimestamp(timestamps[ts_idx], tz=datetime.UTC).isoformat(),
                    "delta_w": round(float(delta), 1),
                    "direction": "up" if delta > 0 else "down",
                    "power_after": round(float(smoothed[i]), 1),
                })

    return edges


def _pair_edges(edges: list[dict]) -> list[dict]:
    """Pair ON edges with matching OFF edges to form appliance events."""
    events = []
    up_edges = [e for e in edges if e["direction"] == "up"]
    down_edges = [e for e in edges if e["direction"] == "down"]
    used_down = set()

    for up in up_edges:
        up_mag = abs(up["delta_w"])
        best = None
        best_diff = float("inf")

        for j, down in enumerate(down_edges):
            if j in used_down:
                continue
            if down["timestamp"] <= up["timestamp"]:
                continue
            elapsed = down["timestamp"] - up["timestamp"]
            if elapsed > MAX_PAIR_WINDOW_SEC:
                continue

            down_mag = abs(down["delta_w"])
            mag_diff = abs(up_mag - down_mag) / max(up_mag, 1)
            if mag_diff < 0.35 and mag_diff < best_diff:
                best_diff = mag_diff
                best = j

        if best is not None:
            down = down_edges[best]
            used_down.add(best)
            duration_min = (down["timestamp"] - up["timestamp"]) / 60
            events.append({
                "start_ts": up["timestamp"],
                "end_ts": down["timestamp"],
                "start": up["time"],
                "end": down["time"],
                "power_w": round(abs(up["delta_w"]), 0),
                "duration_min": round(duration_min, 1),
                "hour": datetime.datetime.fromtimestamp(up["timestamp"], tz=datetime.UTC).hour,
            })

    return events


def _cluster_events(events: list[dict]) -> list[dict]:
    """Cluster appliance events by power level using KMeans.

    Each cluster = one discovered "appliance slot".
    """
    if len(events) < 3:
        return []

    powers = np.array([e["power_w"] for e in events]).reshape(-1, 1)

    from sklearn.cluster import KMeans
    # Choose k: min of 10, or number of distinct 100W buckets
    n_buckets = len(set(int(p[0] // 100) for p in powers))
    k = min(max(2, n_buckets), 10, len(events))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(powers)
    centroids = km.cluster_centers_.flatten()

    clusters = []
    for i in range(k):
        mask = labels == i
        cluster_events = [e for e, m in zip(events, mask) if m]
        if not cluster_events:
            continue

        centroid_w = float(centroids[i])
        durations = [e["duration_min"] for e in cluster_events]
        hours = [e["hour"] for e in cluster_events]

        clusters.append({
            "id": i,
            "centroid_w": round(centroid_w, 0),
            "event_count": len(cluster_events),
            "avg_duration_min": round(float(np.mean(durations)), 1),
            "peak_hours": [h for h, _ in Counter(hours).most_common(3)],
            "total_kwh": round(sum(e["power_w"] * e["duration_min"] / 60 / 1000 for e in cluster_events), 2),
        })

    clusters.sort(key=lambda c: -c["centroid_w"])
    return clusters


def _learn_signatures_from_known_monitors(exclude_entity: str = "", days: int = 30) -> dict[str, dict[str, Any]]:
    """Learn appliance signatures from known HA power monitors (smart plugs etc).

    This treats device-level power sensors as ground truth fingerprints.
    Returns a signature map compatible with _match_to_appliances.
    """
    learned: dict[str, dict[str, Any]] = {}
    if not os.path.exists(HA_DB):
        return learned

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    try:
        conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
        candidates = conn.execute("""
            SELECT DISTINCT sm.entity_id
            FROM states_meta sm
            WHERE sm.entity_id LIKE 'sensor.%'
              AND (
                sm.entity_id LIKE '%_power%'
                OR sm.entity_id LIKE '%_watt%'
                OR sm.entity_id LIKE '%_watts%'
                OR sm.entity_id LIKE '%energy_watts%'
                OR sm.entity_id LIKE '%consumption_w%'
              )
        """).fetchall()

        for (eid,) in candidates:
            if eid == exclude_entity:
                continue
            # Skip known aggregate/system meters
            low = eid.lower()
            if any(k in low for k in (
                'shore_power', 'mastervolt', 'solar_', 'battery_', 'inverter',
                'charger_input_power', 'wind_turbine', 'combined_wattage'
            )):
                continue

            rows = conn.execute("""
                SELECT s.state
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                ORDER BY s.last_changed_ts
            """, (eid, cutoff_ts)).fetchall()

            watts = []
            for (state_val,) in rows:
                try:
                    w = float(state_val)
                    if 0 <= w <= 25000:
                        watts.append(w)
                except (ValueError, TypeError):
                    continue

            if len(watts) < 20:
                continue

            arr = np.array(watts)
            active = arr[arr > 30]  # above standby
            if len(active) < 10:
                continue

            # Robust fingerprint from active periods
            median_w = float(np.median(active))
            p90_w = float(np.percentile(active, 90))
            duty = float(len(active) / len(arr))

            key = eid.replace('sensor.', '').replace('.', '_')
            learned[key] = {
                'power': round(median_w, 1),
                'power_p90': round(p90_w, 1),
                'icon': '🔎',
                'source': 'ha_monitor',
                'entity_id': eid,
                'duty_cycle': round(duty, 3),
            }

            # Promote likely water heater naming so it wins matching
            if any(k in low for k in ('water_heater', 'waterheater', 'boiler', 'varmvatten', 'heater')):
                learned[key]['icon'] = '🚿'
                learned[key]['priority'] = 1

        conn.close()
    except Exception as e:
        log.warning("nilm: failed learning monitor signatures: %s", e)

    return learned


def _match_to_appliances(clusters: list[dict], learned_sigs: dict[str, dict[str, Any]] | None = None) -> list[dict]:
    """Match discovered clusters to known appliance signatures."""
    # Load custom signatures
    custom_sigs = {}
    try:
        if os.path.exists(CUSTOM_SIGS_PATH):
            with open(CUSTOM_SIGS_PATH) as f:
                for sig in json.load(f):
                    custom_sigs[sig["name"]] = {
                        "power": sig.get("peak_delta_w", sig.get("avg_delta_w", 0)),
                        "icon": "🏷️",
                        "source": "user_trained",
                    }
    except Exception:
        pass

    # Merge generic + custom + learned from HA power monitors
    learned_sigs = learned_sigs or {}
    all_sigs = {**GENERIC_APPLIANCES, **custom_sigs, **learned_sigs}

    matched = []
    used_sigs = set()

    for cluster in clusters:
        centroid = cluster["centroid_w"]
        best_name = "unknown"
        best_icon = "❓"
        best_diff = float("inf")
        best_source = "generic"

        for name, sig in all_sigs.items():
            if name in used_sigs:
                continue
            sig_power = sig["power"]
            diff = abs(centroid - sig_power) / max(sig_power, 1)
            if diff < 0.35 and diff < best_diff:
                best_diff = diff
                best_name = name
                best_icon = sig.get("icon", "❓")
                best_source = sig.get("source", "generic")

        if best_name != "unknown":
            used_sigs.add(best_name)

        cluster["appliance"] = best_name.replace("_", " ").title()
        cluster["appliance_key"] = best_name
        cluster["icon"] = best_icon
        cluster["match_confidence"] = round((1 - best_diff) * 100) if best_diff < 1 else 0
        cluster["source"] = best_source
        matched.append(cluster)

    return matched


def _estimate_current_breakdown(readings: list[tuple[float, float]],
                                 matched_clusters: list[dict]) -> list[dict]:
    """Estimate current power breakdown based on recent readings + known appliances.

    Uses the most recent stable power level and decomposes it into
    likely active appliances using a greedy subtraction approach.
    """
    if not readings or not matched_clusters:
        return []

    # Get recent stable power
    recent = [w for _, w in readings[-10:]]
    current_w = float(np.median(recent))

    # Always-on base load
    breakdown = []
    remaining = current_w

    # Sort by power descending — greedy decomposition
    sorted_clusters = sorted(matched_clusters, key=lambda c: -c["centroid_w"])

    for cluster in sorted_clusters:
        appliance_w = cluster["centroid_w"]
        if appliance_w <= remaining + 50:  # 50W tolerance
            breakdown.append({
                "appliance": cluster["appliance"],
                "icon": cluster["icon"],
                "estimated_w": round(min(appliance_w, remaining), 0),
                "confidence": cluster["match_confidence"],
            })
            remaining -= appliance_w
            if remaining < 20:
                break

    if remaining > 20:
        breakdown.append({
            "appliance": "Other / Unidentified",
            "icon": "❓",
            "estimated_w": round(max(0, remaining), 0),
            "confidence": 0,
        })

    return breakdown


def run_disaggregation(power_entity: str = "", days: int = 7) -> dict[str, Any]:
    """Run full NILM disaggregation pipeline.

    1. Load aggregate power data
    2. Detect edges → pair into events → cluster by power level
    3. Match clusters to known/trained appliance signatures
    4. Estimate current real-time breakdown
    """
    if not power_entity:
        # Try to find the main power entity
        power_entity = os.environ.get("HABITUS_POWER_ENTITY", "")
        if not power_entity:
            try:
                state_path = os.path.join(DATA_DIR, "state.json")
                with open(state_path) as f:
                    state = json.load(f)
                power_entity = state.get("user_settings", {}).get("power_entity", "")
            except Exception:
                pass
        if not power_entity:
            # Auto-detect
            if os.path.exists(HA_DB):
                try:
                    conn = sqlite3.connect(f"file:{HA_DB}?mode=ro", uri=True)
                    rows = conn.execute("""
                        SELECT DISTINCT sm.entity_id FROM states_meta sm
                        WHERE sm.entity_id LIKE 'sensor.%'
                        AND (sm.entity_id LIKE '%consumption_w' OR sm.entity_id LIKE '%power_w'
                             OR sm.entity_id LIKE '%electric%w')
                    """).fetchall()
                    conn.close()
                    if rows:
                        power_entity = rows[0][0]
                except Exception:
                    pass

    if not power_entity:
        return {"error": "No power entity configured", "breakdown": []}

    log.info("nilm: running disaggregation on %s (%d days)", power_entity, days)

    readings = _get_aggregate_power(power_entity, days=days)
    if len(readings) < 20:
        return {"error": "Insufficient data", "readings_count": len(readings), "breakdown": []}

    learned_monitor_sigs = _learn_signatures_from_known_monitors(exclude_entity=power_entity, days=min(days, 30))

    edges = _detect_edges(readings)
    events = _pair_edges(edges)
    clusters = _cluster_events(events)
    matched = _match_to_appliances(clusters, learned_sigs=learned_monitor_sigs)
    breakdown = _estimate_current_breakdown(readings, matched)

    # Energy breakdown (last 24h estimation)
    total_kwh_24h = 0
    now = datetime.datetime.now(datetime.UTC).timestamp()
    recent_events = [e for e in events if e["start_ts"] > now - 86400]
    appliance_kwh: dict[str, float] = defaultdict(float)
    for e in recent_events:
        # Find which cluster this event belongs to
        for m in matched:
            if abs(e["power_w"] - m["centroid_w"]) / max(m["centroid_w"], 1) < 0.35:
                kwh = e["power_w"] * e["duration_min"] / 60 / 1000
                appliance_kwh[m["appliance"]] += kwh
                total_kwh_24h += kwh
                break

    energy_breakdown = [
        {"appliance": name, "kwh_24h": round(kwh, 2)}
        for name, kwh in sorted(appliance_kwh.items(), key=lambda x: -x[1])
    ]

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "power_entity": power_entity,
        "days_analysed": days,
        "readings_count": len(readings),
        "edges_detected": len(edges),
        "events_paired": len(events),
        "appliance_slots": len(matched),
        "learned_monitor_signatures": len(learned_monitor_sigs),
        "monitor_signature_entities": sorted([v.get("entity_id", "") for v in learned_monitor_sigs.values() if v.get("entity_id")])[:50],
        "current_breakdown": breakdown,
        "current_total_w": round(float(np.median([w for _, w in readings[-10:]])), 0) if readings else 0,
        "discovered_appliances": matched,
        "energy_24h": energy_breakdown,
        "total_kwh_24h": round(total_kwh_24h, 2),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(NILM_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("nilm: %d edges, %d events, %d appliance slots, current=%.0fW",
             len(edges), len(events), len(matched), result["current_total_w"])
    return result
