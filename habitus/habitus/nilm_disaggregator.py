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
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .ha_db import managed_read_connection, resolve_ha_db_path, table_exists

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
NILM_PATH = os.path.join(DATA_DIR, "nilm_disaggregation.json")
CUSTOM_SIGS_PATH = os.path.join(DATA_DIR, "custom_signatures.json")

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
    """Get aggregate power readings as (timestamp, watts) pairs.

    Primary: SQLite recorder DB
    Fallback: HA history API (for setups where recorder schema/history access differs)
    """
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    # 1) DB path
    db_path = resolve_ha_db_path()
    if db_path:
        try:
            with managed_read_connection(db_path) as conn:
                if conn is None:
                    return []
                rows = conn.execute("""
                    SELECT s.state, s.last_changed_ts FROM states s
                    JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                    WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                    ORDER BY s.last_changed_ts
                """, (entity_id, cutoff_ts)).fetchall()

            result = []
            for state_val, ts in rows:
                try:
                    w = float(state_val)
                    if 0 <= w <= 25000:
                        result.append((ts, w))
                except (ValueError, TypeError):
                    continue
            if result:
                return result
        except Exception as e:
            log.warning("nilm: DB read failed: %s", e)

    # 2) HA API fallback
    try:
        import requests
        ha_url = os.environ.get("HA_URL", "http://supervisor/core")
        token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
        if not token:
            return []

        start_iso = cutoff.isoformat()
        url = f"{ha_url}/api/history/period/{start_iso}"
        params = {
            "filter_entity_id": entity_id,
            "minimal_response": "1",
            "no_attributes": "1",
            "significant_changes_only": "0",
        }
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code != 200:
            return []

        payload = r.json()
        if not payload or not isinstance(payload, list):
            return []

        # Home Assistant returns list-of-lists
        series = payload[0] if payload and isinstance(payload[0], list) else []
        result = []
        for item in series:
            try:
                state_val = item.get("state")
                ts_raw = item.get("last_changed") or item.get("last_updated")
                if state_val is None or ts_raw is None:
                    continue
                w = float(state_val)
                if not (0 <= w <= 25000):
                    continue
                dt = datetime.datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                result.append((dt.timestamp(), w))
            except Exception:
                continue
        if result:
            log.info("nilm: loaded %d readings from HA history API for %s", len(result), entity_id)
        return result
    except Exception as e:
        log.warning("nilm: API fallback failed: %s", e)
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
    n_buckets = len({int(p[0] // 100) for p in powers})
    k = min(max(2, n_buckets), 10, len(events))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(powers)
    centroids = km.cluster_centers_.flatten()

    clusters = []
    for i in range(k):
        mask = labels == i
        cluster_events = [e for e, m in zip(events, mask, strict=False) if m]
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

    Primary source: recorder DB
    Fallback: HA states + history API
    """
    learned: dict[str, dict[str, Any]] = {}
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    def _skip_entity(eid: str) -> bool:
        if eid == exclude_entity:
            return True
        low = eid.lower()
        return any(k in low for k in (
            'shore_power', 'mastervolt', 'solar_', 'battery_', 'inverter',
            'charger_input_power', 'wind_turbine', 'combined_wattage'
        ))

    def _build_sig(eid: str, watts: list[float]):
        if len(watts) < 20:
            return
        arr = np.array(watts)
        active = arr[arr > 30]
        if len(active) < 10:
            return
        median_w = float(np.median(active))
        p90_w = float(np.percentile(active, 90))
        duty = float(len(active) / len(arr))
        key = eid.replace('sensor.', '').replace('.', '_')
        low = eid.lower()
        learned[key] = {
            'power': round(median_w, 1),
            'power_p90': round(p90_w, 1),
            'icon': '🔎',
            'source': 'ha_monitor',
            'entity_id': eid,
            'duty_cycle': round(duty, 3),
        }
        if any(k in low for k in ('water_heater', 'waterheater', 'boiler', 'varmvatten', 'heater')):
            learned[key]['icon'] = '🚿'
            learned[key]['priority'] = 1

    # 1) DB learning
    db_path = resolve_ha_db_path()
    if db_path:
        try:
            with managed_read_connection(db_path) as conn:
                if conn is None:
                    return learned
                has_meta = table_exists(conn, "states_meta")
                if has_meta:
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
                else:
                    candidates = conn.execute("""
                        SELECT DISTINCT entity_id
                        FROM states
                        WHERE entity_id LIKE 'sensor.%'
                          AND (
                            entity_id LIKE '%_power%'
                            OR entity_id LIKE '%_watt%'
                            OR entity_id LIKE '%_watts%'
                            OR entity_id LIKE '%energy_watts%'
                            OR entity_id LIKE '%consumption_w%'
                          )
                    """).fetchall()
                for (eid,) in candidates:
                    if _skip_entity(eid):
                        continue
                    if has_meta:
                        rows = conn.execute("""
                            SELECT s.state
                            FROM states s
                            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                            WHERE sm.entity_id = ? AND s.last_changed_ts > ?
                            ORDER BY s.last_changed_ts
                        """, (eid, cutoff_ts)).fetchall()
                    else:
                        rows = conn.execute("""
                            SELECT state
                            FROM states
                            WHERE entity_id = ? AND last_changed_ts > ?
                            ORDER BY last_changed_ts
                        """, (eid, cutoff_ts)).fetchall()
                    watts = []
                    for (state_val,) in rows:
                        try:
                            w = float(state_val)
                            if 0 <= w <= 25000:
                                watts.append(w)
                        except (ValueError, TypeError):
                            continue
                    _build_sig(eid, watts)
        except Exception as e:
            log.warning("nilm: DB monitor-learning failed: %s", e)

    # 2) API fallback if DB yielded nothing
    if learned:
        return learned

    try:
        import requests
        ha_url = os.environ.get("HA_URL", "http://supervisor/core")
        token = os.environ.get("SUPERVISOR_TOKEN", os.environ.get("HABITUS_HA_TOKEN", ""))
        if not token:
            return learned
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Candidate list from current states
        r = requests.get(f"{ha_url}/api/states", headers=headers, timeout=30)
        if r.status_code != 200:
            return learned
        states = r.json()
        candidates = []
        for s in states:
            eid = s.get('entity_id', '')
            if not eid.startswith('sensor.'):
                continue
            low = eid.lower()
            if any(k in low for k in ('_power', '_watt', '_watts', 'energy_watts', 'consumption_w')) and not _skip_entity(eid):
                candidates.append(eid)

        # Limit for performance, prioritize heater-like names first
        candidates.sort(key=lambda e: 0 if any(k in e.lower() for k in ('water_heater','boiler','varmvatten','heater')) else 1)
        candidates = candidates[:20]

        for eid in candidates:
            url = f"{ha_url}/api/history/period/{cutoff.isoformat()}"
            params = {
                'filter_entity_id': eid,
                'minimal_response': '1',
                'no_attributes': '1',
                'significant_changes_only': '0',
            }
            rr = requests.get(url, headers=headers, params=params, timeout=45)
            if rr.status_code != 200:
                continue
            payload = rr.json()
            series = payload[0] if payload and isinstance(payload, list) and payload and isinstance(payload[0], list) else []
            watts = []
            for item in series:
                try:
                    w = float(item.get('state'))
                    if 0 <= w <= 25000:
                        watts.append(w)
                except Exception:
                    continue
            _build_sig(eid, watts)
    except Exception as e:
        log.warning("nilm: API monitor-learning failed: %s", e)

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


def _auto_detect_power_entity(db_path: str) -> str:
    """Best-effort auto-detect of a likely aggregate power sensor from recorder DB."""
    try:
        with managed_read_connection(db_path) as conn:
            if conn is None:
                return ""
            has_meta = table_exists(conn, "states_meta")

            if has_meta:
                rows = conn.execute("""
                    SELECT DISTINCT sm.entity_id FROM states_meta sm
                    WHERE sm.entity_id LIKE 'sensor.%'
                    AND (sm.entity_id LIKE '%consumption_w' OR sm.entity_id LIKE '%power_w'
                         OR sm.entity_id LIKE '%electric%w')
                """).fetchall()
            else:
                rows = conn.execute("""
                    SELECT DISTINCT s.entity_id FROM states s
                    WHERE s.entity_id LIKE 'sensor.%'
                    AND (s.entity_id LIKE '%consumption_w' OR s.entity_id LIKE '%power_w'
                         OR s.entity_id LIKE '%electric%w')
                """).fetchall()

        if rows:
            return rows[0][0]
    except Exception:
        pass
    return ""


def _make_phase_label(phase_type: str, phases: list[str]) -> str:
    """Build human-readable phase label from phase_type and phase list."""
    if phase_type == "single":
        return f"{phases[0]} (single phase)" if phases else "single phase"
    elif phase_type == "two_phase_400v":
        return f"{'+'.join(phases)} (400V two-phase)" if len(phases) >= 2 else "two-phase 400V"
    elif phase_type == "two_phase_mixed":
        return f"{'+'.join(phases)} (mixed)" if len(phases) >= 2 else "two-phase mixed"
    elif phase_type == "three_phase":
        return "L1+L2+L3 (three-phase)"
    return ""


def correlate_phase_edges(edges_by_phase: dict, window_sec: int = 120) -> list[dict]:
    """Correlate edges detected independently per phase into multi-phase event groups.

    Args:
        edges_by_phase: {"L1": [(ts, delta_w), ...], "L2": [...], "L3": [...]}
            where ts is a datetime or float timestamp.
        window_sec: Maximum seconds between edges to consider correlated.

    Returns:
        List of correlated edge groups with phase attribution, each containing:
        - ts: timestamp of the triggering edge
        - total_delta_w: sum of deltas across all phases
        - phase_type: "single" | "two_phase_400v" | "two_phase_mixed" | "three_phase"
        - phases: sorted list of phase labels involved
        - per_phase: dict of {phase: delta_w}
    """
    all_edges = []
    for phase, edges in edges_by_phase.items():
        for ts, delta in edges:
            all_edges.append({"ts": ts, "delta": delta, "phase": phase})
    all_edges.sort(key=lambda e: e["ts"])

    groups = []
    used: set[int] = set()
    for i, edge in enumerate(all_edges):
        if i in used:
            continue
        group = [edge]
        used.add(i)
        for j, other in enumerate(all_edges[i + 1:], i + 1):
            if j in used:
                continue
            ts_i = edge["ts"]
            ts_j = other["ts"]
            # Support both datetime objects and float timestamps
            if hasattr(ts_i, "total_seconds"):
                diff_sec = abs(ts_j - ts_i)
            else:
                try:
                    diff_sec = abs((ts_j - ts_i).total_seconds())
                except AttributeError:
                    diff_sec = abs(float(ts_j) - float(ts_i))
            if diff_sec <= window_sec and other["phase"] != edge["phase"]:  # different phase only
                group.append(other)
                used.add(j)

        phases_in_group = [e["phase"] for e in group]

        if len(set(phases_in_group)) == 1:
            phase_type = "single"
        elif len(set(phases_in_group)) == 2:
            deltas = [abs(e["delta"]) for e in group]
            ratio = min(deltas) / max(deltas) if max(deltas) > 0 else 0
            phase_type = "two_phase_400v" if ratio > 0.85 else "two_phase_mixed"
        else:
            phase_type = "three_phase"

        groups.append({
            "ts": edge["ts"],
            "total_delta_w": sum(e["delta"] for e in group),
            "phase_type": phase_type,
            "phases": sorted(set(phases_in_group)),
            "per_phase": {e["phase"]: e["delta"] for e in group},
        })
    return groups


def _annotate_clusters_with_phase(
    matched_clusters: list[dict],
    correlated_groups: list[dict],
) -> list[dict]:
    """Annotate matched appliance clusters with phase information.

    For each cluster, find the correlated edge group whose total_delta_w is
    closest to the cluster's centroid_w, within ±35% tolerance.  When a match
    is found, the cluster receives phase_type, phases, and phase_label fields.
    Unmatched clusters get phase_type="single" as a safe default.
    """
    used_groups: set[int] = set()
    annotated = []

    for cluster in matched_clusters:
        centroid = cluster["centroid_w"]
        best_idx = None
        best_diff = float("inf")

        for idx, grp in enumerate(correlated_groups):
            if idx in used_groups:
                continue
            total = abs(grp["total_delta_w"])
            diff = abs(centroid - total) / max(total, 1)
            if diff < 0.35 and diff < best_diff:
                best_diff = diff
                best_idx = idx

        c = cluster.copy()
        if best_idx is not None:
            grp = correlated_groups[best_idx]
            used_groups.add(best_idx)
            c["phase_type"] = grp["phase_type"]
            c["phases"] = grp["phases"]
            c["per_phase"] = grp["per_phase"]
        else:
            c["phase_type"] = "single"
            c["phases"] = ["L1"]
            c["per_phase"] = {}

        c["phase_label"] = _make_phase_label(c["phase_type"], c["phases"])
        annotated.append(c)

    return annotated


def run_disaggregation(power_entity: str = "", days: int = 7) -> dict[str, Any]:
    """Run full NILM disaggregation pipeline.

    1. Load aggregate power data (single or multi-phase)
    2. Detect edges → pair into events → cluster by power level
    3. Match clusters to known/trained appliance signatures
    4. Annotate clusters with per-phase attribution (when multi-phase configured)
    5. Estimate current real-time breakdown
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
            db_path = resolve_ha_db_path()
            if db_path:
                power_entity = _auto_detect_power_entity(db_path)

    if not power_entity:
        return {"error": "No power entity configured", "breakdown": []}

    # Parse comma-separated multi-phase entities
    phase_entities = [e.strip() for e in power_entity.split(",") if e.strip()]
    is_multi_phase = len(phase_entities) > 1

    log.info("nilm: running disaggregation on %s (%d days, phases=%d)",
             power_entity, days, len(phase_entities))

    # ── Aggregate readings (sum of phases for the main pipeline) ──────────────
    if is_multi_phase:
        # Load each phase separately, then sum per timestamp
        all_phase_readings: dict[str, list[tuple[float, float]]] = {}
        for ph_idx, eid in enumerate(phase_entities):
            ph_label = f"L{ph_idx + 1}"
            all_phase_readings[ph_label] = _get_aggregate_power(eid, days=days)

        # Build combined total from all phases
        from collections import defaultdict as _dd
        ts_sums: dict[float, float] = _dd(float)
        for ph_readings in all_phase_readings.values():
            for ts, w in ph_readings:
                ts_sums[ts] += w
        readings: list[tuple[float, float]] = sorted(ts_sums.items())
    else:
        readings = _get_aggregate_power(phase_entities[0] if phase_entities else power_entity, days=days)
        all_phase_readings = {}

    if len(readings) < 20:
        return {"error": "Insufficient data", "readings_count": len(readings), "breakdown": []}

    first_entity = phase_entities[0] if phase_entities else power_entity
    learned_monitor_sigs = _learn_signatures_from_known_monitors(
        exclude_entity=first_entity, days=min(days, 30)
    )

    edges = _detect_edges(readings)
    events = _pair_edges(edges)
    clusters = _cluster_events(events)
    matched = _match_to_appliances(clusters, learned_sigs=learned_monitor_sigs)

    # ── Per-phase correlation (multi-phase only) ───────────────────────────────
    correlated_groups: list[dict] = []
    if is_multi_phase and all_phase_readings:
        edges_by_phase: dict[str, list[tuple[Any, float]]] = {}
        for ph_label, ph_readings in all_phase_readings.items():
            ph_edges = _detect_edges(ph_readings)
            edges_by_phase[ph_label] = [
                (
                    datetime.datetime.fromtimestamp(e["timestamp"], tz=datetime.UTC),
                    e["delta_w"],
                )
                for e in ph_edges
            ]
        correlated_groups = correlate_phase_edges(edges_by_phase, window_sec=120)
        matched = _annotate_clusters_with_phase(matched, correlated_groups)
        log.info("nilm: %d correlated phase-edge groups from %d phases",
                 len(correlated_groups), len(phase_entities))
    else:
        # Single-phase: add default annotations so output schema is consistent
        for c in matched:
            c.setdefault("phase_type", "single")
            c.setdefault("phases", ["L1"])
            c.setdefault("per_phase", {})
            c["phase_label"] = _make_phase_label(c["phase_type"], c["phases"])

    # ── Enrich unnamed slots with device library matches ──────────────────────
    try:
        from . import device_library as _dl  # noqa: PLC0415
        for slot in matched:
            if not slot.get("name") or slot.get("name", "").startswith("Device "):
                _match = _dl.match_wattage_to_device(slot.get("watts", slot.get("centroid_w", 0)))
                if _match:
                    slot["name"] = _match["name"]
                    slot["matched_from_library"] = True
                    slot["library_confidence"] = _match["confidence"]
    except Exception as _e:
        log.debug("Device library enrichment skipped: %s", _e)

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

    # Per-phase current wattage (most recent reading per phase)
    phase_current_w: dict[str, float] = {}
    if is_multi_phase and all_phase_readings:
        for ph_label, ph_readings in all_phase_readings.items():
            if ph_readings:
                recent_ph = [w for _, w in ph_readings[-10:]]
                phase_current_w[ph_label] = round(float(np.median(recent_ph)), 0)

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "power_entity": power_entity,
        "days_analysed": days,
        "readings_count": len(readings),
        "edges_detected": len(edges),
        "events_paired": len(events),
        "appliance_slots": len(matched),
        "learned_monitor_signatures": len(learned_monitor_sigs),
        "monitor_signature_entities": sorted(
            [v.get("entity_id", "") for v in learned_monitor_sigs.values() if v.get("entity_id")]
        )[:50],
        "current_breakdown": breakdown,
        "current_total_w": round(float(np.median([w for _, w in readings[-10:]])), 0) if readings else 0,
        "discovered_appliances": matched,
        "energy_24h": energy_breakdown,
        "total_kwh_24h": round(total_kwh_24h, 2),
        "phase_count": len(phase_entities) if phase_entities else 1,
        "phase_current_w": phase_current_w,
        "correlated_phase_groups": len(correlated_groups),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(NILM_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("nilm: %d edges, %d events, %d appliance slots, current=%.0fW",
             len(edges), len(events), len(matched), result["current_total_w"])
    return result
