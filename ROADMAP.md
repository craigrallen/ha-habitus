# Habitus Roadmap

## v0.x (current) — Foundation
- [x] Long-term stats ingestion (full HA history)
- [x] IsolationForest anomaly detection
- [x] 4 published HA sensors
- [x] Web UI with ingress + sidebar
- [x] Full Rescan button
- [x] Live training progress
- [x] Incremental runs (no duplicate data)

---

## v1.0 — Intelligence Layer

### 1. Proposed Automations
Habitus analyses discovered patterns and outputs ready-to-use HA automation YAML.

Examples it will generate:
- "Bathroom lights always on 06:30–07:15 weekdays → suggest morning routine automation"
- "Kitchen power spikes >800W every weekday 07:00–08:30 → suggest peak tariff alert"
- "Solar production > 1500W between 11:00–15:00 in summer → suggest battery-priority charging"
- "Bilge temp rises 3°C above baseline → suggest bilge pump check alert"
- "Shore power drops to 0 while battery <40% → suggest low-battery alert"
- "No motion/sensor changes for >24h in living area → suggest vacancy/security alert"

UI: **Proposed Automations tab** — each suggestion shows:
- Plain English explanation of the pattern
- Confidence score
- Copy-to-clipboard YAML
- "Add to HA" button (calls HA REST API to create draft automation)

### 2. Anomaly Breakdown
When score > 40, show *which sensors* are driving it:
- Top 5 anomalous entities with their current vs baseline value
- "Bathroom power is 340W — baseline for Monday 20:00 is 45W ±12W"
- Drill-down per circuit/room

### 3. Pattern Discovery
Persistent learned patterns stored in `patterns.json`:
- Daily routines (wakeup, sleep, meals)
- Weekly cycles (weekday vs weekend)
- Seasonal shifts
- Unusual one-offs flagged separately

### 4. Energy Insights tab
- Peak usage hours (with baseline chart)
- Top 5 power consumers by circuit
- Estimated waste (devices left on outside normal hours)
- Solar self-consumption ratio

### 5. HA Notification Integration
- Configurable anomaly threshold alert via HA `notify` service
- Daily digest (optional) — "Yesterday: 2 anomalies, peak power 2.1kW at 18:30"
- Config option: `notify_service: notify.telegram_openclaw`

### 6. Seasonal Models
- Separate IsolationForest per season (winter/spring/summer/autumn)
- Prevents false anomalies when behaviour legitimately shifts
- Auto-selects correct model based on current date

### 7. Suggested Lovelace Cards
- Auto-generates a Lovelace YAML snippet for an insights card
- Paste into your dashboard to show score + top anomaly reason

---

## v2.0 — Predictive Layer (future)
- Failure prediction (motor degradation, heating inefficiency trends)
- Occupancy inference (are people home?)
- Weather-aware SOC targeting
- Per-room energy attribution
