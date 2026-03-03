# Habitus PRD — Agent Task List

Each task below is a unit of work. The agent picks one, implements it, passes all quality gates, commits, and marks it done in `progress.txt`.

---

## Batch 1 — v1.0 Intelligence Layer (Priority: High)

### TASK-001: Proposed Automations — Pattern-to-YAML Engine
**File:** `habitus/habitus/patterns.py`
**Description:** Extend `patterns.py` to generate ready-to-use HA automation YAML from discovered patterns. Cover:
- Morning routine (lights always on 06:30–07:15 weekdays → suggest automation)
- Peak tariff alert (kitchen power >800W at 07:00–08:30 weekdays)
- Vacancy/security alert (no motion >24h in living area)
- Bilge/marine: bilge temp rises 3°C above baseline
- Shore power drops to 0 while battery <40%
Output: `suggestions.json` with fields: `title`, `description`, `confidence`, `yaml`, `category`
**Tests:** `tests/test_patterns.py` — cover YAML generation and confidence scoring
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-002: Anomaly Breakdown — Per-Entity Z-Score UI Data
**File:** `habitus/habitus/anomaly_breakdown.py`
**Description:** When anomaly score > 40, compute top-5 anomalous entities with:
- Entity name
- Current value
- Baseline mean ± std for this hour/day-of-week
- Plain-English reason: "Bathroom power is 340W — baseline for Monday 20:00 is 45W ±12W"
Output: `entity_baselines.json` updated with z-score breakdown per run
**Tests:** `tests/test_anomaly_breakdown.py`
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-003: Seasonal Model Selection
**File:** `habitus/habitus/seasonal.py`
**Description:** Train and persist 4 IsolationForest models (winter/spring/summer/autumn). Auto-select based on current date hemisphere-aware. Store as `seasonal_models.pkl`.
**Tests:** `tests/test_seasonal.py`
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-004: HA Notification Integration
**File:** `habitus/habitus/main.py`
**Description:** On anomaly score > configured threshold, call HA `notify` service via REST API. Daily digest support (optional config). Config options: `notify_service`, `notify_on_anomaly`, `anomaly_threshold`.
**Tests:** Mock HA REST calls; test digest formatting
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-005: Energy Insights — Dashboard Data Endpoint
**File:** `habitus/habitus/web.py`
**Description:** Add `/api/insights` JSON endpoint returning:
- Peak usage hours (top 3 with wattage)
- Top 5 power consumers by entity
- Estimated waste (devices active outside baseline hours)
- Solar self-consumption ratio (if solar sensor present)
**Tests:** Test the Flask endpoint with mock data
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

## Batch 2 — Lovelace Card (Priority: High)

### TASK-006: Custom Lovelace Card — habitus-card.js
**File:** `habitus/www/habitus-card.js`
**Description:** Complete the Lovelace card implementation showing:
- Animated anomaly score gauge (0–100)
- Top 3 anomaly reasons (entity + reason string)
- Latest suggested automation (title + confidence)
- Last updated timestamp
Auto-register via WebSocket API on first run (no HACS). Serve from `/www/habitus-card.js`.
**Tests:** N/A (JS) — but add registration logic test in Python
**Acceptance:** Card renders; ruff + black + mypy pass on Python side

---

## Batch 3 — Novel ML Features (Priority: Medium)

### TASK-007: Routine Drift Detection
**File:** `habitus/habitus/drift.py`
**Description:** Implement changepoint detection on rolling weekly averages:
- Morning routine start time drift
- Bedtime drift
- Screen time trends
Output: `drift_report.json` with `metric`, `baseline_time`, `current_time`, `delta_minutes`, `trend`
**Tests:** `tests/test_drift.py` with synthetic weekly data
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-008: Phantom Load Hunter
**File:** `habitus/habitus/phantom.py`
**Description:** Identify devices drawing baseline power 24/7 that shouldn't need to. For each:
- Entity name, mean standby wattage, hours/day on
- Annual cost estimate (configurable kWh price via config)
- Suggested automation fix
Output: `phantom_loads.json`
**Tests:** `tests/test_phantom.py`
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-009: Automation Effectiveness Scoring
**File:** `habitus/habitus/automation_score.py`
**Description:** Score existing HA automations:
- Fire count per month
- Manual override rate (automation fires, human action reverses within 2 min)
- Sensor validity at fire time
Output: `automation_scores.json` with score 0–100, issues list, suggested fix per automation
**Tests:** `tests/test_automation_score.py`
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-010: Automation Gap Analysis
**File:** `habitus/habitus/automation_gap.py`
**Description:** Identify patterns in sensor history that have no corresponding automation:
- Recurring manual actions (light toggled at same time 5+ days/week → no automation)
- Recurring power events with no trigger automation
Output: `automation_gaps.json` with gap description + suggested YAML
**Tests:** `tests/test_automation_gap.py`
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

## Batch 4 — Future / Stretch (Priority: Low)

### TASK-011: Guest Detection
Detect guest presence from behavioral patterns alone (no cameras). Unusual room combinations, different activity times. Output: `guest_pattern.json`.

### TASK-012: Sleep Quality Inference
From motion + light + time patterns — no wearable. Consistency score + disruption events. Output: `sleep_report.json`.

### TASK-013: Hidden Correlation Discovery
Find non-obvious causal chains between sensors. Output as "Did you know?" insights.

### TASK-014: Comfort Intelligence
Multi-dimensional comfort score (temp + humidity + CO2 + lighting). Learn optimal comfort conditions. Suggest pre-conditioning automations.

---

---

## Batch 5 — Sensor Intelligence & False Positive Reduction (Priority: High)

### TASK-015: Sensor Type Classifier
**File:** `habitus/habitus/sensor_classifier.py` (new), `habitus/habitus/anomaly_breakdown.py`
**Description:** Classify every entity into a sensor type based on HA `state_class` attribute and history pattern analysis:
- `total_increasing` state_class or monotonically increasing values → `accumulating` (kWh, m³, L)
- Values only ever 0 or 1 → `binary`
- Continuous bounded variation → `gauge`
- Brief spikes returning to zero → `event`
- Values in a fixed setpoint range → `setpoint`
Store `sensor_type` per entity in `entity_baselines.json`. All downstream scoring must consume this classification.
**Tests:** `tests/test_sensor_classifier.py` — cover all 5 types, HA attribute path, history fallback
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-016: Accumulating Sensor Rate-of-Change Baseline
**File:** `habitus/habitus/anomaly_breakdown.py`, `habitus/habitus/main.py`
**Description:** For entities classified as `accumulating` (kWh, gas m³, water L), the absolute value is meaningless as a baseline — it monotonically increases forever. Replace with:
- Compute hourly delta (consumption rate) from the raw history
- Build baseline on delta values, not absolute values
- Anomaly = unusual consumption rate for this hour/day-of-week
- New accumulating entity going from 0 → any value: **never an anomaly** — begin delta accumulation, no scoring for first 24h of deltas
- Store `baseline_type: "rate"` vs `"absolute"` in entity baseline slot
**Tests:** cover delta computation, new-entity bootstrap, anomaly rate scoring
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-017: Binary Sensor Timing & Frequency Scoring
**File:** `habitus/habitus/anomaly_breakdown.py`
**Description:** Binary sensors (motion, door, presence, contact) cannot be scored on raw value (always 0 or 1). Replace z-score with:
- **Expected state**: what fraction of time is this sensor `on` at this hour/day? If baseline=5% but currently on → score proportionally
- **Transition frequency**: how many on→off / off→on events per hour is normal? Flag if >3× baseline frequency
- **Duration**: how long does this sensor typically stay on? Flag if current duration >2× baseline
- Never flag a binary sensor purely because its value is 1 (it's supposed to be 1 sometimes)
**Tests:** cover expected-state scoring, frequency anomaly, duration anomaly, no false positives on normal state
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-018: Per-Entity Cold Start Protection
**File:** `habitus/habitus/anomaly_breakdown.py`, `habitus/habitus/main.py`
**Description:** Replace the crude global 7-day warmup with per-entity intelligence:
- Track `first_seen` timestamp and `n_samples` count per entity in `entity_baselines.json`
- Entity with <7 days since `first_seen`: exempt from scoring entirely, report as `"status": "learning"` in breakdown
- Baseline slot with <10 samples: widen CI (use 2.5σ threshold instead of 2σ), halve z-score weight
- New entity detected in current HA states but absent from baselines: add to `entity_lifecycle.json` with `status: "new"`, do not score
- Entity going from null/missing → any value: bootstrap only, never anomalous
**Tests:** cover new entity detection, slot sample threshold, first_seen gating, reactivation
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-019: Confidence-Weighted Anomaly Score Aggregation
**File:** `habitus/habitus/anomaly_breakdown.py`, `habitus/habitus/main.py`
**Description:** Replace equal-weight entity averaging with confidence-weighted aggregation:
- Per-entity confidence = `min(1.0, days_of_data/30) × min(1.0, slot_n/20) × sensor_type_certainty`
- Global score = `Σ(z_score × confidence) / Σ(confidence)` (weighted average, not equal sum)
- Minimum confidence threshold: entities below 0.1 confidence contribute nothing to global score
- Expose `confidence` field per entity in `/api/anomaly_breakdown` response
- Show confidence badge in UI breakdown: "72% confident — 4 days of data"
**Tests:** cover weighting math, low-confidence exclusion, API field presence
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-020: Adaptive IsolationForest Contamination
**File:** `habitus/habitus/trainer.py`, `habitus/habitus/main.py`
**Description:** IsolationForest's `contamination` parameter determines how aggressively it flags outliers. On day 7 it has almost no baseline — default contamination causes excessive false positives. Ramp it by training age:
- <7 days: `contamination = 0.005` (almost nothing flagged)
- 7–14 days: `contamination = 0.01`
- 14–30 days: `contamination = 0.02`
- 30–90 days: `contamination = 0.04`
- 90+ days: `contamination = 0.05`
Automatically retrain the model when the contamination tier changes (store current tier in `run_state.json`).
**Tests:** cover tier calculation, automatic retrain trigger, contamination value per tier
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-021: Impossible Value & Data Quality Guard
**File:** `habitus/habitus/anomaly_breakdown.py`, `habitus/habitus/main.py`
**Description:** Filter bad sensor data *before* it enters baselines or scoring — never surface as a behavioral anomaly:
- Negative power values → clamp to 0, log warning
- Temperature >85°C or <-60°C → discard, log as bad sensor
- Humidity outside 0–100% → clamp
- Stuck sensor: entity value unchanged for >24h AND it's a gauge type → flag in `data_quality.json` as `stuck`, exclude from model
- Value jump >10× in one hour for a gauge sensor → discard that single point as bad reading
- Report `data_quality.json`: `{entity_id, issue, since, last_valid}` — surfaced in UI as a separate "Sensor Health" section, not as anomalies
**Tests:** cover each filter type, stuck detection, quality file output
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

## Batch 6 — Entity Lifecycle & Feedback (Priority: Medium)

### TASK-022: Entity Lifecycle Tracker
**File:** `habitus/habitus/lifecycle.py` (new)
**Description:** Track the lifecycle of every HA entity Habitus observes:
- New entity first seen → status: `bootstrapping`, no scoring for 7 days
- Entity absent from HA states for 7+ days → status: `inactive`, removed from scoring but baseline preserved
- Inactive entity reappears → status: `reactivating`, restart 3-day confidence ramp
- Persist to `entity_lifecycle.json`: `{entity_id, status, first_seen, last_seen, days_active}`
- Expose lifecycle status in `/api/anomaly_breakdown` per entity
**Tests:** `tests/test_lifecycle.py` — cover all state transitions
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-023: User Feedback Loop — "Mark as Normal"
**File:** `habitus/habitus/web.py`, `habitus/habitus/anomaly_breakdown.py`
**Description:** Allow users to teach the model what's normal for their household:
- POST `/api/feedback` endpoint: `{entity_id, hour_of_day, day_of_week, feedback: "normal" | "real_anomaly"}`
- Store feedback in `feedback.json`
- During scoring: if entity × slot has `normal` feedback → suppress from score for 30 days
- If entity × slot has `real_anomaly` feedback → lower z-score threshold to 1.5σ for that slot
- Settings tab: show feedback entries with "Reset" per-entity option
**Tests:** cover feedback storage, suppression logic, threshold adjustment, reset
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

## Batch 7 — Advanced Detection (Priority: Medium)

### TASK-024: Rate-of-Change Spike Detection
**File:** `habitus/habitus/spike_detector.py` (new)
**Description:** Detect sudden intra-hour jumps that baseline z-scores miss (transient events):
- For each gauge entity, compute delta vs previous hour reading
- Flag if `|delta| > 3 × typical_delta_std` for that hour/day slot
- Spike score (0–100) contributed separately to global anomaly score (weighted 30% spike / 70% baseline)
- Output spike events to `spike_events.json`: `{entity_id, ts, delta, baseline_delta, severity}`
- Show in UI as "sudden change" vs "sustained deviation"
**Tests:** `tests/test_spike_detector.py` — cover normal delta, spike detection, score contribution
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

### TASK-025: Correlated Sensor Cross-Validation
**File:** `habitus/habitus/correlation.py` (new)
**Description:** Before confirming an anomaly, check if correlated sensors agree — reduces false positives from single noisy sensors:
- During training, build a correlation map: which sensor pairs move together? (Pearson r > 0.6)
- Store as `correlation_map.json`: `{entity_a, entity_b, r, lag_hours}`
- At scoring time: if entity A is anomalous but all strongly-correlated partners are normal → reduce A's score contribution by 40%
- If multiple correlated entities are simultaneously anomalous → raise each one's confidence
- Expose correlation context in breakdown: "Corroborated by 2 sensors" / "Uncorroborated"
**Tests:** `tests/test_correlation.py` — cover map building, score adjustment, corroboration flag
**Acceptance:** ruff + black + mypy pass; pytest ≥70% coverage

---

## Completion Signal
When all high-priority tasks (TASK-001 through TASK-006 and TASK-015 through TASK-021) are complete and committed, output:
```
<promise>COMPLETE</promise>
```
