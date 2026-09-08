# Dual-Source Data Validation Strategy

## Concept
Use both `states` (recent, high-res) and `statistics` (long-term, aggregated) tables as checks and balances to ensure data quality and completeness.

## Current State (3.10.42)
- **Training:** Uses `statistics` table for long-term history (3 years)
- **Energy forecast:** Switched to `statistics` for kWh sensors
- **Issue:** If a sensor has no statistics but has states, we miss recent data
- **Issue:** If statistics are wrong/corrupted, we have no validation

## Proposed Architecture

### 1. Dual Fetch Strategy
For each power/energy sensor:
1. **Fetch from statistics** (primary, long-term)
2. **Fetch from states** (validation, recent 7-10 days)
3. **Cross-validate:** Compare overlapping period
4. **Report discrepancies** → flag in data quality report

### 2. Checks & Balances

**Check 1: Coverage Comparison**
```python
stats_coverage = has_data_in_statistics(entity_id, last_30_days)
states_coverage = has_data_in_states(entity_id, last_7_days)

if states_coverage and not stats_coverage:
    log.warning(f"{entity_id}: has recent states but no statistics — LTS recorder may be disabled")
    
if stats_coverage and not states_coverage:
    log.warning(f"{entity_id}: has statistics but no recent states — sensor may be dead")
```

**Check 2: Value Consistency**
```python
# Compare hourly mean from both sources for last 24-48 hours
stats_mean = get_hourly_mean_from_statistics(entity_id, last_48h)
states_mean = get_hourly_mean_from_states(entity_id, last_48h)

diff_pct = abs(stats_mean - states_mean) / max(stats_mean, 0.1)
if diff_pct > 0.15:  # >15% discrepancy
    log.warning(f"{entity_id}: statistics vs states mismatch — {diff_pct*100:.1f}% difference")
```

**Check 3: Timestamp Freshness**
```python
latest_stat = get_latest_statistic_timestamp(entity_id)
latest_state = get_latest_state_timestamp(entity_id)
age_hours = (now - latest_state).total_seconds() / 3600

if age_hours > 2:
    log.warning(f"{entity_id}: no new data in {age_hours:.1f}h — sensor may be stale")
```

**Check 4: Gap Detection**
```python
# For statistics: gaps in hourly sequence indicate recorder issues
gaps = find_missing_hours(entity_id, last_7_days)
if len(gaps) > 10:
    log.warning(f"{entity_id}: {len(gaps)} missing hours in statistics — recorder may have crashed")
```

### 3. Data Source Priority

**For Training (3+ year history):**
1. **Primary:** `statistics` table (long-term hourly)
2. **Backfill:** `states` table for last 7 days if statistics lag behind
3. **Fallback:** Proxy sensors when both are sparse

**For Real-time Scoring:**
1. **Primary:** `states` table (most current)
2. **Validation:** Compare to latest statistics entry
3. **Alert:** If states/statistics diverge significantly

**For Energy Forecast:**
1. **Primary:** `statistics` table (daily kWh deltas)
2. **Validation:** `states` for last 24h as sanity check
3. **Report:** Flag if sources disagree

### 4. Health Dashboard Section

Add a **Data Quality** tab showing:

**Sensor Health Table:**
| Sensor | States (7d) | Statistics (30d) | Consistency | Status |
|--------|-------------|------------------|-------------|--------|
| inverter_l1 | ✓ 2026-03-20 14:50 | ✓ 2026-03-20 13:00 | 98.5% | 🟢 Good |
| shore_power | ✓ 2026-03-20 14:50 | ✗ No data | N/A | 🟡 No LTS |
| battery_soc | ✓ 2026-03-20 14:50 | ✓ 2026-03-20 13:00 | 76.2% | 🟡 Drift |

**Data Source Coverage:**
```
Power sensors (3):
  ├─ States coverage:      7.2 days (100%)
  ├─ Statistics coverage:  1095 days (100%)
  └─ Overlap consistency:  97.3%

Proxy sensors (5):
  ├─ States coverage:      7.0 days (97%)
  ├─ Statistics coverage:  1089 days (99%)
  └─ Overlap consistency:  95.1%
```

**Anomalies Detected:**
- `sensor.shore_power_l1`: Statistics show 0W for 2026-03-15 but states show 1200W → Recorder issue?
- `sensor.battery_soc`: 12h gap in statistics on 2026-03-18 → HA restart?

### 5. Implementation Plan

**Phase 1 (3.10.43): Data Quality Report**
- Add `/api/data_quality` endpoint
- Check all configured sensors (power, proxy, temp, energy)
- Report: states coverage, statistics coverage, latest timestamps
- Flag: missing data, stale sensors, no statistics

**Phase 2 (3.10.44): Cross-Validation**
- For overlapping periods, compare states vs statistics
- Calculate consistency percentage (hourly mean agreement)
- Report discrepancies in data quality API

**Phase 3 (3.10.45): Smart Merge Strategy**
- Use statistics for bulk history
- Use states for last 7 days (fresher, more accurate)
- Detect and fill recorder gaps with states data

**Phase 4 (3.10.46): UI Dashboard**
- New "Sensor Health" tab
- Table view with status indicators
- Coverage timeline visualization
- Downloadable CSV report

### 6. Benefits

**Reliability:**
- Detect recorder failures early
- Catch sensor death (no new states)
- Identify statistics corruption

**Accuracy:**
- Use highest-res data available (states for recent)
- Validate long-term trends (statistics)
- Cross-check prevents garbage-in-garbage-out

**Debugging:**
- Clear visibility into data sources
- Historical gap analysis
- Reproducible validation reports

**User Trust:**
- Transparent data quality metrics
- Automatic health monitoring
- Proactive alerts for sensor issues

### 7. Example Scenario

**Scenario:** Shore power sensor stopped recording statistics after HA update

**Without dual-source:**
- Training uses only statistics
- Fetches 1095 days, gets 0 rows for last 30 days
- Proxy fallback works but user doesn't know why

**With dual-source:**
1. Fetch statistics: 1065 days of data
2. Fetch states: 7 days of data
3. Detect gap: statistics end 2026-02-18, states start 2026-03-13
4. **Action:** Merge states into training window, fill 30-day gap
5. **Alert:** "shore_power_l1: statistics stopped on 2026-02-18 — check recorder config"

**User sees:**
- Data Quality tab: 🟡 shore_power_l1 missing statistics for 30 days
- Suggestion: "Re-enable LTS recorder for sensor.shore_power_l1"
- Training continues with hybrid data (stats + states)

### 8. Code Pattern

```python
def fetch_with_validation(entity_id: str, days: int) -> pd.DataFrame:
    """Fetch from both sources and cross-validate."""
    
    # Primary: statistics (long-term)
    stats_df = fetch_from_statistics(entity_id, days)
    
    # Validation: states (recent 7 days)
    states_df = fetch_from_states(entity_id, min(days, 7))
    
    # Cross-check overlap
    if not stats_df.empty and not states_df.empty:
        consistency = calculate_consistency(stats_df, states_df)
        if consistency < 0.85:
            log.warning(f"{entity_id}: {consistency*100:.1f}% consistency between sources")
    
    # Merge: use stats for bulk, states for recent
    if not stats_df.empty:
        # Fill recent period with states (more accurate)
        cutoff = datetime.now() - timedelta(days=7)
        merged = stats_df[stats_df.index < cutoff]
        if not states_df.empty:
            merged = pd.concat([merged, states_df])
        return merged.sort_index()
    
    # Fallback: states only if no statistics
    return states_df
```

## Next Steps

1. Implement data quality endpoint (quick win)
2. Add cross-validation checks (moderate effort)
3. Build UI dashboard (polish)
4. Document for users (transparency)

This dual-source strategy makes Habitus production-ready for environments where data quality varies.
