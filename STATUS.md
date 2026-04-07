# Habitus Status — 2026-03-20 17:30 CET

## ✅ Major Accomplishments Today

### Training System Fixed (5 Critical Bugs)
1. **day_name UnboundLocalError** (3.10.39) — Variable scope error prevented ALL training
2. **Stale data_to timestamp** (3.10.40) — Incremental fetch tried to fetch 0 rows  
3. **state.json vs run_state.json** (3.10.43) — Settings saved to wrong file, never loaded
4. **Auto-detection overwrites user settings** (3.10.44) — Config reset every training run
5. **build_features doesn't read user_settings** (3.11.9) — Used env var instead of state file

**Result:** Training completes successfully, 997/997 tests passing ✓

### Local Timeseries Database (3.11.x)
- **Architecture:** SQLite DB with WebSocket collector + HA history backfill
- **Backfill:** 275,376 rows collected from HA history API (10 days)
- **Coverage:** 9 power/solar/battery sensors tracked
- **API Endpoints:** `/api/timeseries/stats`, `/api/timeseries/query`
- **Storage:** `/data/timeseries.db` inside add-on container

**Status:** Backfill works ✓, WebSocket collector not running ✗

### Dual-Source Data Strategy
- **Plan documented:** DUAL_SOURCE_VALIDATION.md
- **Hybrid fetch implemented:** `data_sources.py` merges statistics + states
- **Local DB priority:** Training tries local DB first, falls back to hybrid HA fetch

## ❌ Remaining Issues

### Issue #1: WebSocket Collector Not Running
**Symptom:** Only 2 hourly aggregates in last 42 hours despite 171k raw events expected  
**Root cause:** Collector thread crashes or never connects to HA WebSocket  
**Impact:** Real-time collection doesn't work, only backfill data exists  
**Fix needed:** Debug WebSocket connection, add reconnection logic, error logging

### Issue #2: Power Charts Still 0W
**Symptom:** Training shows "0.1% non-zero" for total_power_w → drops feature  
**Root cause:** Only 28 hourly samples out of 336 hours (14 days) = 8.3%  
**Why:** WebSocket collector not populating continuous data  
**Threshold:** Need >5% non-zero (17+ hours) to keep power features  
**Fix needed:** Get WebSocket collector working OR reduce training window to match available data

### Issue #3: Sparse Hourly Aggregation
**Symptom:** 275k raw events → only 28 hourly aggregates  
**Root cause:** Data concentrated in small time windows (backfill periods)  
**Impact:** Hourly resolution loses most data points  
**Fix needed:** Either collect continuously OR use 1-minute resolution for features

## 📊 Current Data State

**Local DB:**
```
Total rows:     275,376
Time span:      10.0 days (2026-03-10 to 2026-03-20)
Hourly aggregates: 28 (8.3% of expected 336 hours)

Inverter L1:    171,327 events
Inverter L2:    242,033 events  
Inverter L3:    171,000 events (estimated)
Solar:          364,690 events
Shore power:    ~30k events total (L1/L2/L3)
Battery:        59,983 events
```

**HA Statistics Table:**
- Inverter sensors: sparse/missing (only March 1-7 data)
- Solar/battery: years of history via MasterBus

**Training Window:**
- Configured: 14 days
- With power data: 28 hours = 1.17 days
- Behavioral data: 3 years ✓

## 🎯 Action Plan

### Priority 1: Fix WebSocket Collector (Critical)
**Steps:**
1. Add detailed logging to collector thread startup
2. Check if WebSocket connection establishes
3. Add try/except around `connect_and_subscribe`
4. Log each state_changed event received
5. Add automatic reconnection on disconnect
6. Verify SUPERVISOR_TOKEN is valid

**Success criteria:** See "Flushed X records to DB" logs every 10 seconds

### Priority 2: Reduce Training Window (Quick Win)
**Current:** 14 days configured, only 1.2 days have power data  
**Change to:** 2 days (48 hours) to match available data  
**Impact:** 28 hours / 48 hours = 58% non-zero → power features kept!

**Steps:**
```bash
curl -X POST http://172.30.33.7:8099/api/settings \
  -H 'Content-Type: application/json' \
  -d '{"days_history":2}'

curl -X POST http://172.30.33.7:8099/api/full_train
```

**Expected result:** Power charts populate with 779W mean, 2446W max ✓

### Priority 3: Cross-Validation (Data Quality)
**Implement:**
- Compare local DB vs HA statistics for overlapping period
- Flag discrepancies >15%
- Report data quality metrics in UI
- Auto-detect sensor failures (no new data in 2h)

**Refer to:** DUAL_SOURCE_VALIDATION.md Phase 2

### Priority 4: Continuous Collection (Long-term)
**Fix WebSocket collector:**
- Debug connection issues
- Add reconnection logic
- Verify state_changed events flowing
- Test with multiple sensors

**OR alternative:** Poll HA states API every 60 seconds instead of WebSocket

## 🔧 Quick Wins Available Now

### Win #1: Set days_history=2
This will make power charts populate immediately with existing 28-hour dataset.

### Win #2: Manual WebSocket Test
SSH into add-on container, run collector manually with logging to see actual error.

### Win #3: Fallback to Polling
If WebSocket is unreliable, implement 60-second polling of HA states API as alternative.

## 📝 Code Changes Made Today

**Versions deployed:**
- 3.10.39: day_name fix
- 3.10.40: stale data_to fix  
- 3.10.43: state.json → run_state.json
- 3.10.44: preserve user power_entity
- 3.10.45: dual-source hybrid fetch
- 3.11.0: local timeseries DB + collector
- 3.11.1–3.11.9: collector fixes, feature building from local DB

**Key files modified:**
- `habitus/habitus/collector.py` (new) — WebSocket collector + SQLite DB
- `habitus/habitus/data_sources.py` (new) — Hybrid fetch (stats + states)
- `habitus/habitus/main.py` — Feature building from local DB, user_settings fixes
- `habitus/habitus/web.py` — Collector startup, settings endpoints, log instance
- `habitus/habitus/activity.py` — day_name scope fix

**Tests:** 997/997 passing ✓

## 🚀 Next Session Goals

1. **Fix WebSocket collector** (30 min) — add logging, debug connection  
2. **Set days_history=2** (1 min) — quick win to populate charts  
3. **Verify power charts** (5 min) — check if 779W mean/2446W max shows  
4. **Add cross-validation** (1 hour) — compare local DB vs HA statistics  
5. **Implement polling fallback** (30 min) — if WebSocket unreliable

**Expected outcome:** Power charts fully populated, continuous data collection working

## 💾 State Preservation

**Git repo:** `craigrallen/ha-habitus` branch `main`  
**Latest commit:** `61f482e` (3.11.9)  
**HA add-on:** `57582523_habitus` version 3.11.9  
**Database:** `/data/timeseries.db` with 275k rows  
**Settings persist:** `run_state.json` working ✓

**To resume:**
```bash
cd ~/Projects/ha-habitus
git pull origin main
ha apps logs 57582523_habitus  # check collector status
```

## 🎓 Lessons Learned

1. **Env vars vs state files:** User settings MUST come from state.json, not env vars (env only for defaults)
2. **WebSocket reliability:** Need reconnection logic + fallback to polling
3. **Hourly aggregation:** Loses too much resolution for sparse data — use 1-minute
4. **Training window:** Must match available data, not aspirational history
5. **Multiple bugs cascade:** One bug (filename) masked another (env var) masked another (WebSocket)

## 📖 Documentation Created

- `DUAL_SOURCE_VALIDATION.md` — Strategy for states + statistics cross-check
- `INCREMENTAL_TRAINING.md` — Plan for chunked training (1M row limit)
- `STATUS.md` (this file) — Comprehensive state snapshot

---

**Summary:** Local DB infrastructure complete and collecting data via backfill. WebSocket real-time collection needs debugging. Quick win: reduce training window to 2 days → charts will populate. Next: fix WebSocket, add cross-validation, implement polling fallback.
