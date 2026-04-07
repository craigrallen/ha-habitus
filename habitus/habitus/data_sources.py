"""Dual-source data fetching: merge statistics (long-term) + states (recent) for complete coverage."""

from __future__ import annotations

import datetime
import logging
import sqlite3
from typing import Any

import pandas as pd

log = logging.getLogger("habitus")


def fetch_hybrid(
    entity_ids: list[str],
    cutoff: datetime.datetime,
    now: datetime.datetime,
    db_path: str,
    max_w: float = 25000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch power/energy data from both statistics and states, merge for complete coverage.
    
    Strategy:
    1. Fetch from statistics table (long-term hourly aggregates)
    2. Fetch from states table (recent high-resolution, last 7 days)
    3. Cross-validate overlapping period
    4. Merge: use statistics for bulk history, states for freshest data
    5. Return combined dataframe + quality report
    
    Returns:
        (dataframe, quality_report)
        - dataframe: columns [entity_id, ts, mean, sum]
        - quality_report: {entity_id: {stats_rows, states_rows, consistency, gaps, ...}}
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cutoff_ts = cutoff.timestamp()
    now_ts = now.timestamp()
    recent_cutoff = (now - datetime.timedelta(days=7)).timestamp()
    
    quality = {}
    all_stats_rows = []
    all_states_rows = []
    
    for entity_id in entity_ids:
        # === 1. Fetch from statistics (long-term) ===
        try:
            stats_rows = conn.execute("""
                SELECT sm.statistic_id, st.start_ts, st.mean, st.sum
                FROM statistics st
                JOIN statistics_meta sm ON st.metadata_id = sm.metadata_id
                WHERE sm.statistic_id = ? AND st.start_ts >= ? AND st.start_ts <= ?
                ORDER BY st.start_ts
            """, (entity_id, cutoff_ts, now_ts)).fetchall()
            
            stats_count = len(stats_rows)
            log.debug(f"{entity_id}: {stats_count} rows from statistics")
            
            for eid, ts, mean_val, sum_val in stats_rows:
                all_stats_rows.append({
                    "entity_id": eid,
                    "ts": ts,
                    "mean": mean_val,
                    "sum": sum_val,
                    "source": "statistics"
                })
        except Exception as e:
            log.debug(f"{entity_id}: statistics query failed: {e}")
            stats_count = 0
        
        # === 2. Fetch from states (recent 7 days) ===
        try:
            states_rows = conn.execute("""
                SELECT sm.entity_id, s.last_changed_ts, s.state
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = ? AND s.last_changed_ts >= ?
                ORDER BY s.last_changed_ts
            """, (entity_id, recent_cutoff)).fetchall()
            
            states_count = len(states_rows)
            log.debug(f"{entity_id}: {states_count} rows from states")
            
            # Aggregate states to hourly (to match statistics granularity)
            states_hourly = {}
            for eid, ts, state_val in states_rows:
                try:
                    val = float(state_val)
                    if val < 0 or val > max_w:
                        continue
                    hour_ts = int(ts // 3600 * 3600)
                    if hour_ts not in states_hourly:
                        states_hourly[hour_ts] = []
                    states_hourly[hour_ts].append(val)
                except (ValueError, TypeError):
                    continue
            
            # Convert to mean per hour
            for hour_ts, values in states_hourly.items():
                all_states_rows.append({
                    "entity_id": entity_id,
                    "ts": hour_ts,
                    "mean": sum(values) / len(values),
                    "sum": None,
                    "source": "states"
                })
        except Exception as e:
            log.debug(f"{entity_id}: states query failed: {e}")
            states_count = 0
        
        quality[entity_id] = {
            "stats_rows": stats_count,
            "states_rows": states_count,
            "consistency": None,  # Calculated below
        }
    
    conn.close()
    
    # === 3. Merge: prefer states for recent, statistics for historical ===
    # Convert to dataframes
    stats_df = pd.DataFrame(all_stats_rows) if all_stats_rows else pd.DataFrame()
    states_df = pd.DataFrame(all_states_rows) if all_states_rows else pd.DataFrame()
    
    if stats_df.empty and states_df.empty:
        log.warning("Hybrid fetch: no data from either source")
        return pd.DataFrame(), quality
    
    # Merge strategy:
    # 1. Use all statistics data
    # 2. For recent period (last 7 days), replace with states data (fresher, more accurate)
    if not stats_df.empty:
        stats_df["ts"] = pd.to_datetime(stats_df["ts"], unit="s", utc=True)
        merged = stats_df.copy()
    else:
        merged = pd.DataFrame()
    
    if not states_df.empty:
        states_df["ts"] = pd.to_datetime(states_df["ts"], unit="s", utc=True)
        
        if not merged.empty:
            # Remove recent period from statistics (will be replaced by states)
            recent_dt = now - datetime.timedelta(days=7)
            merged = merged[merged["ts"] < recent_dt]
            
            # Append states data
            merged = pd.concat([merged, states_df], ignore_index=True)
        else:
            # No statistics, use states only
            merged = states_df.copy()
    
    # Sort and deduplicate
    if not merged.empty:
        merged = merged.sort_values(["entity_id", "ts"]).drop_duplicates(subset=["entity_id", "ts"])
        merged = merged.reset_index(drop=True)
        
        log.info(
            f"Hybrid fetch complete: {len(merged)} rows from {len(entity_ids)} entities "
            f"(stats: {len(stats_df)}, states: {len(states_df)})"
        )
    
    # === 4. Calculate consistency where sources overlap ===
    # (For now, just log coverage — full cross-validation in Phase 2)
    for entity_id in entity_ids:
        ent_stats = stats_df[stats_df["entity_id"] == entity_id] if not stats_df.empty else pd.DataFrame()
        ent_states = states_df[states_df["entity_id"] == entity_id] if not states_df.empty else pd.DataFrame()
        
        if not ent_stats.empty and not ent_states.empty:
            # Simple consistency: do both sources have data for recent period?
            recent_dt = now - datetime.timedelta(days=1)
            stats_recent = ent_stats[ent_stats["ts"] >= recent_dt]
            states_recent = ent_states[ent_states["ts"] >= recent_dt]
            
            if not stats_recent.empty and not states_recent.empty:
                # Both have recent data — calculate mean agreement
                stats_mean = stats_recent["mean"].mean()
                states_mean = states_recent["mean"].mean()
                if stats_mean > 0:
                    consistency = 1.0 - abs(stats_mean - states_mean) / stats_mean
                    quality[entity_id]["consistency"] = round(consistency, 3)
    
    return merged, quality
