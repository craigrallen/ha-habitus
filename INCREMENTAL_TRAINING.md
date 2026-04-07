# Incremental Training Plan

## Current Problem
- Training fetches exactly 1,000,000 rows (hard cap to prevent OOM)
- All data loaded into memory at once → crashes on low-RAM devices
- No progress indication during long-running training
- Automation scoring (80 automations) times out → kills add-on → training never completes

## Immediate Wins (3.10.37 ✓ / 3.10.38 in progress)
1. ✅ **Timeout automation scoring** (20s limit) — prevents add-on restart
2. **Smart sampling** — when total rows > 1M, sample every Nth row to fit budget
3. **Batched automation scoring** — process 20 automations at a time with progress
4. **Progress percentages** — "Fetching data: 45% (450k/1M rows)" in UI

## Future: True Incremental Training
**Goal:** Train on devices with <512MB RAM by processing data in chunks.

### Strategy
1. **Chunked data fetch:**
   - Read DB in 100k row chunks
   - Aggregate to hourly features per chunk
   - Merge chunks into final feature matrix

2. **Incremental model fitting:**
   - Replace `IsolationForest` (batch-only) with:
     - `IncrementalPCA` for dimensionality reduction
     - `MiniBatchKMeans` for clustering
     - `SGDOneClassSVM` for anomaly detection
   - Use `partial_fit()` to train on each chunk

3. **Checkpoint/resume:**
   - Save progress after each chunk
   - Resume from last checkpoint if training crashes
   - Mark chunks as "processed" in state.json

4. **Adaptive chunk sizing:**
   - Measure memory usage per chunk
   - Adjust chunk size dynamically (start 100k, reduce if OOM risk)

### Implementation Phases
**Phase 1 (3.10.38):** Smart sampling + progress tracking
**Phase 2 (3.11.x):** Chunked data fetch
**Phase 3 (3.12.x):** Incremental models with `partial_fit`
**Phase 4 (3.13.x):** Checkpoint/resume system

### Why 1,000,000 Rows?
- It's a **safety cap**, not a target
- Current code: `FETCH_ROW_BUDGET = 1_000_000`
- Real row count depends on:
  - Number of sensors (2115 in your case)
  - Training window (1095 days = 3 years)
  - Expected rows: `2115 sensors × 1095 days × 24 hours = 55,566,000 potential rows`
  - Actual: ~1M because most sensors don't have 3 years of history

### Progress Tracking
**Current:**
- `set_progress(phase, done, total, rows, elapsed, eta)` exists
- Called during: data fetch, baseline build, training, scoring
- Saved to `/data/progress.json`
- UI polls this file every 2 seconds

**Needed:**
- More granular updates during data fetch (every 100k rows)
- Sub-phase tracking ("Fetching chunk 5/10")
- Memory usage reporting

## Config Tunables
- `HABITUS_FETCH_ROW_BUDGET` — max rows to fetch (default 1M)
- `HABITUS_SMART_SAMPLING` — enable sampling when over budget (default true)
- `HABITUS_FETCH_CHUNK_SIZE` — rows per chunk (future, default 100k)
- `HABITUS_AUTO_SCORING_BATCH` — automations per scoring batch (future, default 20)
