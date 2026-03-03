# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Scene Detector** (`scene_detector.py`) — mines entity co-occurrence patterns from HA SQLite database to find implicit scenes (groups of entities that activate together within a 5-minute window)
- **Automation Builder** (`automation_builder.py`) — generates HA automation YAML from discovered patterns and scenes, checks for overlap with existing HA automations
- **Smart Home tab** — merged Automations and Insights tabs into a single unified view with sub-sections: Discovered Scenes, Suggested Automations, Your Automations, Entity Picker, and all Insights
- **Discovered Scenes** section — shows implicit scenes with entity badges, time patterns, confidence scores, and expandable Scene YAML
- **Entity Picker** widget — searchable, filterable entity browser for lights, switches, media players, climate, covers, fans, scenes, and automations with one-click copy
- **HA Automations** section — displays existing user automations from Home Assistant alongside Habitus suggestions
- All YAML code blocks now behind `<details><summary>` expand buttons for cleaner UI
- Smart suggestions show overlap warnings when they match existing HA automations
- Scene-based suggestions with entity state tracking and time-of-day patterns
- Direct HA SQLite database access for state history (much faster than REST API)
- New API endpoints: `GET /api/scenes`, `GET /api/ha_automations`, `GET /api/smart_suggestions`, `GET /api/entities`
- `mlxtend` added to requirements.txt

### Changed
- "Automations" and "Insights" tabs merged into single "Smart Home" tab
- Suggestion cards now show entity badges, time patterns, and overlap info
- All existing API endpoints remain backward-compatible

## [3.1.0] - 2026-03-03

### Added
- Sensor type classifier with 5-type taxonomy (`accumulating`, `binary`, `gauge`, `event`, `setpoint`) stored in `entity_baselines.json`
- Accumulating sensor rate-of-change baseline scoring
- Binary sensor timing and frequency scoring
- Per-entity cold start protection with lifecycle tracking
- Confidence-weighted anomaly score aggregation
- Adaptive IsolationForest contamination parameter scaled by training age
- Impossible value detection and data quality guard
- Energy insights endpoint (`/api/insights`) with peak hours, top consumers, waste, and solar self-consumption
- Data-driven automation suggestions derived from discovered patterns
- Per-entity z-score breakdown with anomaly threshold gating
- Hemisphere-aware seasonal model selection with bundled seasonal models
- HA notification integration with daily digest
- Lovelace card shows top-3 anomaly reasons and suggestion confidence
- Version exposed in `/api/state` response
- `sensor_type` field in anomaly breakdown API response
- Explicit error logging in progressive training thread
- `CLAUDE.md`, `PRD.md`, `AGENTS.md` for autonomous coding infrastructure

### Changed
- Scene detector now mines motion sensors, presence detection, person entities, and device trackers alongside lights/switches/media
- Motion/presence-triggered automations generated when binary_sensor co-occurs with lights in a scene
- Routine predictor — detects recurring activities from humidity/temperature spikes
- Shower/bath detection from bathroom humidity spikes (>10% above rolling average, 5-45 min duration)
- Cooking detection from kitchen humidity/temperature rises
- Learns typical event times and day patterns (weekday/weekend/daily)
- Pre-heat automation YAML: suggests turning on water heater 1 hour before usual shower time
- Predicted Routines section in Smart Home tab with confidence scores
- Cross-domain conflict detector — catches wasteful/contradictory states in real-time
- 7 conflict rules: window+heating, window+AC, heating+cooling, nobody-home+lights, warm-outside+heating, cold+windows-open, door-open-long+climate
- Estimated wattage waste per conflict
- Fix-it automation YAML for each conflict
- Conflicts section at top of Smart Home tab with severity colours
- Runs on every score cycle (lightweight, real-time)
- Heater/thermostat state changes as implicit presence signal — "someone adjusted the bedroom heater → they are in the bedroom"
- Climate-triggered automations: "when heater turns on, activate room scene"
- Presence-inferring keywords: heater, radiator, thermostat, heating, climate, hvac
- Deep correlation engine — mines all sensor data for statistically significant patterns
- Temporal correlations: A happens → B follows within 10 minutes (lift-based filtering)
- Cross-room pattern detection (e.g., arriving home → kitchen lights)
- Climate-response correlations (temp change → heating activation)
- Pairwise analysis of up to 200 most active entities
- Actionable suggestions with approve/dismiss HA notifications
- Correlation categories: trigger_action, room_routine, cross_room, presence_driven, climate_response
- UI: Discovered Correlations section with lift scores and colour-coded significance
- Room-aware predictive automation — predicts what user wants when entering a room
- Conditional probability model: P(action | room, time_slot, day_type) from motion history
- 2-hour time windows × weekday/weekend for granular predictions
- Actionable HA notifications with approve/dismiss buttons
- 60% minimum confidence threshold to avoid noise
- Predictions shown prominently in Smart Home tab with per-action confidence
- HA area registry integration — uses configured rooms/areas as primary room source
- Entity→area mapping cached from HA template API (18 areas, 963 entities on Craig's system)
- Falls back to keyword matching only when HA areas unavailable
- Room detection from entity names — matches 30+ room keywords (kitchen, bedroom, wheelhouse, salon, etc.)
- Boat-specific room keywords (wheelhouse, engine room, salon, cabin, cockpit, foredeck)
- Fallback room detection via common prefix extraction when no keyword matches
- Each scene now includes a `rooms` field for UI grouping
- Power shape analysis — classifies events as steady/cycling/decaying/phased
- Heat pump signature: high inrush spike → gradual decay as target temp reached
- Electric radiator: cycling on/off pattern at fixed wattage
- Underfloor heating: low power, long duration, steady
- Immersion heater: high power, medium duration, steady
- Shape matching boosts classification confidence (shape match = +20% score)
- Appliance fingerprinting (NILM) — detects oven, hob, kettle, washing machine, etc. from power spike signatures
- Power step detection reads directly from HA SQLite database for speed
- 12 known appliance signatures with power range, duration, and icon
- Appliance Detection section in Smart Home tab with event grid and recent events table
- Door/window sensor triggers — "when front door opens, turn on hallway lights"
- Person home/away automations — "when Craig leaves, turn everything off" / "when Craig arrives, welcome scene"
- Door, window, contact, and opening sensor keywords detected as triggers
- Minimum z-score threshold raised to 3.0σ — deviations below this are normal operating variance
- Exclude non-behavioral sensors from anomaly scoring: crypto prices (xbt/xrp/eth), reactive power (kvar/kvarh), network device memory/CPU utilization

### Fixed
- JSON serialization crashes from numpy `bool_`/`int64` types — added `default=str` to all `json.dump()` calls
- Broken `default=str` insertions in `len()`, `round()`, `.isoformat()`, `.values()` calls — caused `TypeError` crashes during training
- Syntax errors in `anomaly_breakdown.py` from malformed regex replacements
- Syntax errors in `anomaly_breakdown.py` from malformed replacements
- Silent training failures — progressive thread now logs full tracebacks
- Training completing entity baselines but crashing before model fit (root cause of persistent score=100)

### Changed
- Major refactor: 7,000+ lines added across 10+ new modules
- Sensor classifier filters accumulating sensors from anomaly scoring
- 70% test coverage minimum enforced in CI

## [3.0.0] - 2026-03-02

### Added
- `sensor_classifier.py` — classifies all HA sensors into 5 types
- `insights.py` — energy insights engine (overnight baseline, period comparison, cost estimates)
- `anomaly_breakdown.py` — per-entity z-score breakdown with confidence weighting
- `automation_gap.py` — checks if suggested automations exist, flags improvable ones with YAML templates
- `activity_engine.py` — activity feature extraction (lights, motion, presence, media, doors)
- `pattern_engine.py` — pattern discovery across sensor data
- Comprehensive test suite (10+ test files)
- `.claude/` config directory for autonomous coding agents

### Changed
- Version bump from 2.x to 3.0.0 — major architectural change

## [2.78.0] - 2026-03-01

### Added
- Responsive tables — `.table-wrap` horizontal scroll containers on all data tables
- Mobile-friendly spacing and font reductions at `@media (max-width: 700px)`

## [2.77.0] - 2026-03-01

### Added
- Light/dark mode toggle button (🌙/☀️) in header
- Theme preference saved to `localStorage`, defaults to dark

## [2.76.0] - 2026-03-01

### Added
- Light mode CSS variables (`[data-theme="light"]`)

## [2.73.0] - 2026-02-28

### Added
- `/api/full_train` endpoint for direct 365-day training (bypasses progressive)
- Seasonal model metadata files (`meta_{season}.json`) to track training hours

### Fixed
- Seasonal models now extend-not-overwrite — new model only replaces existing if it has MORE data
- `FEATURE_COLS` synced between `main.py` and `seasonal.py` (was mismatched 7 vs 19 columns)
- Rescan now wipes ALL model/json artifacts for clean retrain

### Changed
- Lowered seasonal minimum training threshold from 72h to 48h

## [2.72.0] - 2026-02-28

### Added
- `/api/full_train` endpoint for direct 365-day training

## [2.71.0] - 2026-02-28

### Changed
- Seasonal models only train when ≥180 days of data available
- Prevents progressive 30d runs from wiping full-year seasonal models

## [2.70.0] - 2026-02-28

### Changed
- Training progress indicator changed from blocking full-screen overlay to small dismissible toast in bottom-right corner

## [2.69.0] - 2026-02-28

### Changed
- Anomaly notification threshold raised to ≥90 (was ≥40 — too spammy)
- Suggestions notifications disabled (available in UI only)

## [2.68.0] - 2026-02-28

### Added
- Day-normalized period comparison — partial weeks/months compared by same number of days

## [2.67.0] - 2026-02-28

### Changed
- Renamed "phantom loads" to "overnight baseline" — boat legitimately uses 1.3-1.6 kWh/hour overnight
- 30-day window for overnight baseline calculation (avoids skew from off-shore months)

## [2.66.0] - 2026-02-27

### Changed
- Overnight baseline now uses HA Energy Dashboard statistics API (`recorder/statistics_during_period`)
- Same monthly data HA Energy Dashboard displays — no re-fetching raw history

## [2.65.0] - 2026-02-27

### Changed
- Phantom loads refactored to use grid kWh meter directly — period vs period comparison
- Idle-hour baseline = overnight draw; no watt sensors, no price math

## [2.64.0] - 2026-02-27

### Added
- Auto-pull electricity price from Energy Dashboard (`number_energy_price`)
- 3.0 kr/kWh auto-detected from HA configuration

## [2.63.0] - 2026-02-27

### Added
- Phantom loads reference Energy Dashboard grid kWh — shows annual bill, % wasted
- Sanity-caps phantom totals against real grid data

## [2.62.0] - 2026-02-27

### Fixed
- Default `kwh_price` corrected from 0.30 to 3.0 (kr/kWh, not EUR/kWh)

## [2.61.0] - 2026-02-27

### Fixed
- Phantom loads — strict W-only matching via HA `unit_of_measurement` cache
- Exclude write_rate/kvah/kwh/non-power entities from phantom detection
- 500W per-device cap; configurable kr price and currency

## [2.60.0] - 2026-02-27

### Added
- Warmup grace period — anomaly score stays 0 for first 7 days of training
- UI shows "Warming up (Xd left)" instead of false anomaly scores

## [2.59.0] - 2026-02-27

### Added
- Power sensor selector in Settings tab — lists all watt sensors, saves choice, triggers retrain
- Selection persists across restarts via `state.json`

## [2.58.0] - 2026-02-27

### Fixed
- Leak alert restricted to `binary_sensor` domain with explicit keywords only
- No more false positives on `input_boolean` entities

## [2.57.0] - 2026-02-26

### Added
- Automation gap analyser — checks if suggested automations exist, flags improvable ones with YAML templates

## [2.56.0] - 2026-02-26

### Added
- Auto-detect watt companion sensor (`_w`) from Energy Dashboard kWh entity
- Prefer real-time watts over kWh delta; broader fallback watt sensor search

## [2.55.0] - 2026-02-26

### Added
- Phantom load hunter — detects always-on devices with unexplained draw
- Routine drift detection — spots changes in daily patterns over time
- Automation effectiveness scoring — measures how well automations work

## [2.54.0] - 2026-02-26

### Added
- 4 custom Lovelace card designs (pulse/chip/panel/timeline)
- Auto-installs on first run — no HACS required

## [2.53.0] - 2026-02-26

### Added
- Gas and water smart meter support — auto-detected from HA Energy Dashboard
- `water_l_per_h` and `gas_m3_per_h` as features (19 total)

## [2.52.0] - 2026-02-26

### Added
- Water pump watts and leak binary as features (18 total)
- Instant persistent notification on leak detection

## [2.51.0] - 2026-02-26

### Added
- `grid_kwh_w` as 16th feature column — kWh delta as cross-validation signal

## [2.50.0] - 2026-02-26

### Added
- Auto-detect HA Energy Dashboard entities for accurate total power
- Grid kWh delta → W conversion; fallback to device rate sensors

## [2.49.0] - 2026-02-25

### Fixed
- Use `max()` not `sum()` for `total_power_w` — prevents double-counting overlapping sensors

### Added
- `power_entity` config option for explicit power sensor selection

## [2.48.0] - 2026-02-25

### Added
- `max_power_kw` setting (default 25kW) — configurable via add-on UI, filters broken sensors

## [2.47.0] - 2026-02-25

### Fixed
- Power cap tightened to 25kW (typical max home load)
- `fmtW()` hides values >25kW as "—" in UI

## [2.46.0] - 2026-02-25

### Added
- Anomalies and suggestions auto-surfaced as HA persistent notifications and text sensors

## [2.45.0] - 2026-02-25

### Fixed
- Export `HABITUS_VERSION` from bashio in `run.sh` so version badge shows correctly

## [2.44.0] - 2026-02-25

### Fixed
- Exclude phone/mobile sensors from behavioral analysis
- `fmtW()` applied in patterns view
- Cap `night_baseline` to reasonable values

## [2.43.0] - 2026-02-25

### Fixed
- Version badge no longer hardcoded; hourly table uses `fmtW()`
- Version exposed in `/api/state` response

## [2.42.0] - 2026-02-25

### Fixed
- Human-readable power display — `fmtW()` formats as kW/W, caps at 1MW to filter bad sensors

## [2.41.0] - 2026-02-25

### Fixed
- `build_features` always returns all 15 `FEATURE_COLS` — pads missing activity features with zeros

## [2.40.0] - 2026-02-25

### Fixed
- Rescan wipes all `.pkl` files — stale 7-feature scaler was blocking scoring

## [2.39.0] - 2026-02-25

### Fixed
- `UnboundLocalError` at line 594 — moved `actual_start` logic before `del df`

## [2.38.0] - 2026-02-25

### Fixed
- Progressive training wired into `/api/rescan` endpoint
- Default days reduced from 3650 to 365

## [2.37.0] - 2026-02-24

### Added
- Progressive training 30d→60d→90d→180d→365d — first score in ~3 minutes, expands in background

### Fixed
- Hardcoded version strings removed from `run.sh`
- `DAYS` capped at 365 to prevent OOM on Odroid N2

## [2.36.0] - 2026-02-24

### Fixed
- `days_history` cap applied to all paths — was still using 2000-01-01 fallback in some branches
- Default `days_history` reduced from 3650 to 365 to prevent OOM with 900+ sensors

## [2.35.0] - 2026-02-24

### Fixed
- 3s poll during training (was 30s); live row counter; adaptive refresh rate

## [2.34.0] - 2026-02-24

### Fixed
- Score fallback feature vector matches `FEATURE_COLS` length
- Version badge shows real version
- `data_from` shows actual first data date

## [2.33.0] - 2026-02-24

### Added
- Training-start notification

## [2.32.0] - 2026-02-24

### Added
- Notifications sent when training starts and completes (day count, sensor count, score)

## [2.31.0] - 2026-02-24

### Changed
- UI shows app during training once baselines/model ready
- Full-page overlay only on true first run

## [2.30.0] - 2026-02-24

### Added
- Progressive UI during training — baseline tab unlocks after fetch, score shows after model trains

## [2.29.0] - 2026-02-24

### Added
- Background training thread — web UI always accessible during training
- Training banner at bottom of page; first-run-only overlay

## [2.28.0] - 2026-02-24

### Added
- Per-phase progress labels in UI
- `log.info` before each slow phase (baselines/train/seasonal/patterns)

## [2.27.0] - 2026-02-24

### Changed
- Sensor discovery: replaced keyword allowlist with domain-based filter
- Captures lights/switches/climate/media/covers (~5-10x more sensors)

## [2.26.0] - 2026-02-23

### Fixed
- Force image rebuild (supervisor was serving stale 2.7.0 image)

## [2.20.0] - 2026-02-23

### Fixed
- Dark mode contrast improvements
- Baseline table JS key fix — `parseInt` for hour comparison

## [2.13.0] - 2026-02-23

### Fixed
- Notification tap opens Habitus ingress page instead of HA home

## [2.10.0] - 2026-02-23

### Fixed
- Relative imports in `main.py`
- Ruff/Black/MyPy lint fixes
- `pyproject.toml` ruff lint section corrected

## [2.2.0] - 2026-02-22

### Added
- Activity baseline features (lights, motion, presence, media, doors, weather)
- Full CI pipeline with codecov
- 100% annotated test suite
- Overnight vs continuous training schedule
- Score-only daytime mode
- Settings tab shows schedule status

### Changed
- Resumed semantic versioning from 2.1.0

## [2.1.0] - 2026-02-22

### Added
- Inter font, animated SVG gauge, glassmorphism progress overlay
- Step indicators and refined card styling

## [2.0.0] - 2026-02-22

### Added
- Seasonal models (winter/spring/summer/autumn)
- Entity anomaly breakdown with z-scores
- Boat-specific automations
- 5-tab UI (Overview, Baseline, Patterns, Suggestions, Settings)
- HA notification integration
- Add-to-HA button for automation suggestions

## [1.0.0] - 2026-02-21

### Added
- Proposed Automations tab
- Energy Patterns view
- Pattern Discovery engine
- Add-to-HA button for one-click automation installation

## [0.13.0] - 2026-02-21

### Added
- Activity baseline (lights/motion/presence/media/doors/weather)
- Full CI pipeline
- Codecov integration
- 100% annotated tests

## [0.12.0] - 2026-02-21

### Changed
- Reverted to 0.x.x versioning (pre-release)

## [0.2.0] - 2026-02-20

### Added
- Long-term statistics ingestion via WebSocket — 5 years of hourly behavioral data
- Ingress web UI — insights dashboard with anomaly score, baselines, run state
- Full Rescan button — wipes model artifacts, triggers immediate retrain
- Live training progress bar with rows fetched, elapsed time, ETA
- Rich progress UI with training phase indicator

### Fixed
- Use base-python image (pip3 not in aarch64-base)
- Correct WebSocket URL (`ws://supervisor/core/api/websocket`)
- 30s startup wait; crash recovery
- Unpinned requirements (numpy 2.0.0rc1 incompatible with Python 3.13)

### Changed
- No raw data storage — queries live, keeps only model artifacts

## [0.1.0] - 2026-02-20

### Added
- Initial release
- IsolationForest anomaly detection on hourly behavioral snapshots
- HA add-on with ingress web UI
- MIT license

## [3.5.0] - 2026-03-03

### Added
- **PrefixSpan sequence mining** — discovers ordered routines (hallway → kitchen → kettle → coffee)
- **Markov chain next-action prediction** — "you just turned on X, want me to do Y?" (pure Python)
- **Hidden Markov Model activity states** — home knows it's "sleeping", "cooking", "relaxing" etc (hmmlearn)
- **Weather-aware energy forecasting** — 7-day kWh prediction with Gaussian Process uncertainty bands
- **Dynamic automations** — detects timing drift ("evening routine shifted 30 min later"), calendar-aware
- **Behaviour drift detection** — z-score based, shows when habits are shifting
- **Calendar integration** — reads HA calendar entities to predict schedule impacts
- **Temperature-energy correlation analysis** — "colder days = more energy" quantified
- New API endpoints: `/api/sequences`, `/api/markov`, `/api/activity_states`, `/api/energy_forecast`, `/api/dynamic`
- UI: Activity States, Energy Forecast, Behaviour Drift, Routine Sequences, Next-Action Predictions sections
- New dependencies: `prefixspan`, `hmmlearn`, `ruptures`

## [3.6.0] - 2026-03-03

### Added
- **Geek tab** — advanced ML analysis moved here (HMM, PrefixSpan, Markov, Correlations)
- **Anomaly feedback system** — confirm/dismiss anomalies to train the model
  - Confirmed anomalies tighten detection; dismissed widen the "normal" band
  - Per-entity stats: frequently dismissed entities auto-widen thresholds
- **Anonymous data sharing** (opt-in) — share anonymised anomaly data to improve for everyone
  - Only entity domains, scores, and feedback actions; no names/IPs/identifying info
- **Device training mode** — teach Habitus your specific appliances
  - Select power sensor → Start recording → Turn on device → Stop → Name it
  - Captures power profile, shape, peak/avg watts, inrush detection
  - Custom signatures saved alongside generic fingerprints
- **Configurable history depth** — settings option to go back across ALL HA database history
  - Options: 30d / 90d / 6mo / 1yr / 2yr / 3yr / all (max 10 years)
  - More history = richer patterns but longer training
- Energy Forecast and Behaviour Drift sections moved to Smart Home tab (user-facing)

### Changed
- Smart Home tab: cleaner layout with forecast + drift + predictions + conflicts
- Geek tab: ML models (HMM, PrefixSpan, Markov, Correlations) + Device Training + Feedback

## [3.7.0] - 2026-03-03

### Added
- **NILM disaggregation** — decomposes aggregate power meter into per-appliance estimates
  - Edge detection + ON/OFF pairing + KMeans clustering + signature matching
  - 18 generic appliance signatures (fridge, kettle, oven, heat pump, shore charger, etc.)
  - Custom user-trained signatures merged automatically
  - Real-time power breakdown bar (coloured segments per appliance)
  - 24h energy-by-appliance breakdown
  - "Re-analyse" button for on-demand disaggregation
  - Greedy subtraction decomposition for current breakdown
  - No external NILM libraries — runs on numpy + sklearn (already installed)
- New API endpoints: `/api/nilm`, `/api/nilm/run`
- NILM section in Energy & Patterns tab
