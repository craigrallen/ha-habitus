# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
