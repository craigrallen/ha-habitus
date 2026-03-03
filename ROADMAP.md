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

## Next: Custom Lovelace Card (v3.0.0)
- Serve `/www/habitus-card.js` from add-on (maps to `/local/habitus-card.js` in HA)
- Register as Lovelace resource via WebSocket API on first run (no HACS needed)
- Auto-inject card into default dashboard on first install
- Card shows: score gauge, anomaly list, top suggestions, last updated — all inline
- Replaces the manual "add this YAML" step entirely

---

## Novel ML Features — Differentiated from Everything Else

### 🔮 1. Routine Drift Detection
Most systems detect "is this unusual RIGHT NOW". Habitus should also detect SLOW DRIFT over weeks/months.
- "Your morning routine has shifted 47 minutes later over the past 6 weeks"
- "Bedtime has moved from 22:30 to 00:15 since December"
- "The TV is being used 2.3hr/day more than 3 months ago"
- Uses changepoint detection on rolling weekly averages, not just hourly anomalies
- Surfaces as a "Life Rhythm" report — subtle, personal, genuinely useful

### 🔌 2. Appliance Health Fingerprinting  
Each appliance has a power signature — duration, wattage curve, on/off frequency.
- Learn the baseline fingerprint for: fridge, washing machine, dishwasher, HVAC
- Detect degradation: "Fridge compressor running 34% longer than 6 months ago — coils may need cleaning"
- "Dishwasher cycle extended from 58min avg to 74min avg — possible blockage or element degradation"
- "Washing machine motor current spiking 12% higher than baseline — bearing wear possible"
- Real predictive maintenance — nobody does this well at consumer level

### 👻 3. Phantom Load Hunter
- Identify every device drawing power 24/7 that shouldn't need to
- Calculate annual cost in user's local currency (configurable kWh price)
- "These 7 devices waste €127/year on standby — here's how to fix each"
- Ranks by impact, provides specific automation to fix each one
- Completely novel as an automatic discovery (vs manual energy monitoring)

### 🧑‍🤝‍🧑 4. Guest Detection
- Detect when guests are present purely from behavioral patterns (no cameras, no privacy invasion)
- Unusual room usage combinations, different activity times, extra motion in unusual areas
- "Guest pattern detected — suggesting: [adjust thermostat for extra occupant, notify if front door unlocked at night]"
- Infers household size changes from behavior, not presence sensors

### 💤 5. Sleep Quality Inference
- From motion sensors + light patterns + time of activity — no wearable needed
- "Sleep consistency score: 68/100 — irregular bedtimes affecting this"
- "Unusual motion at 03:20 this week — possible sleep disruption"
- Trend over weeks: improving/declining sleep patterns
- Completely privacy-first (local only, never leaves the device)

### ⚡ 6. Automation Effectiveness Scoring
- Score EXISTING HA automations for how effective they actually are
- "This automation fires 180x/month but motion sensor it depends on is only active 23% of fire times — likely a timing issue"
- "Sunset lights automation overridden manually 67% of the time — probably wrong offset"  
- Learns from manual overrides to detect bad automation logic
- Suggests fixes based on actual patterns

### 🌡️ 7. Comfort Intelligence
- Multi-dimensional comfort score: temp + humidity + CO2 + lighting + noise (if available)
- Learn WHEN the home is at its most comfortable (correlate with user activity/presence patterns)
- "Your home is most comfortable at 21°C / 45% RH — currently at 19°C, suggest automation adjustment"
- "Comfort dips every weekday morning between 07:00-08:00 — suggest pre-heating 30 mins earlier"

### 🔗 8. Hidden Correlation Discovery
- Find non-obvious causal chains: "Every time the washing machine runs after 21:00, the bedroom light stays on 40 min longer"
- "Coffee maker before 07:00 correlates with 94% probability of heating boost within 15 min"
- "Front door opening between 17:00-18:30 predicts TV on within 8 minutes — 89% accuracy"
- Present as "Did you know?" insights — things the user never consciously noticed about themselves

