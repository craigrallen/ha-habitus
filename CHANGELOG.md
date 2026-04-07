# Changelog

All notable changes to this project will be documented in this file.

<!-- Generated from git history on 2026-04-02 -->

## [Unreleased]


## [2026-03] - 2026-03-01

### Added
- Context-aware anomaly scoring — entity criticality weighting + false alarm filtering + user priority overrides (bathroom lights=0.3×, battery=2×)
- Nuanced anomaly scoring — 5 severity tiers (0-30 elevated, 30-60 anomaly, 60-80 critical, 80-95 critical, 95-100 extreme) instead of binary 0/100
- Comprehensive NILM dataset integration — 127 appliance signatures from PLAID + UK-DALE + REDD + ECO + GREEND (159 total with boat-specific)
- Integrate PLAID + UK-DALE appliance signatures — 54 reference appliances, 86 total with boat-specific (boat devices get +20% match boost)
- Boat appliance library — 30 known devices with wattage ranges, runtime patterns, phase hints for NILM matching
- NILM appliance management API — manual override, relabel, merge, delete, appliance library (backend infrastructure)
- Anomaly history tracking + pattern analysis — 7-day event log, recurring entity detection, time patterns
- Dedicated Anomalies tab — full visibility into current status, active alerts, history, patterns
- Safety-critical automation protection — detect emergency/alarm patterns, flag with badge, prevent deletion
- Hybrid power data — use inverter (new) + proxy backfill (2023-2026 estimated from shore+battery+solar)
- Build power features directly from local DB (bypasses zero-dilution from 3-year behavioral data)
- Collector reads user settings from run_state.json, backfills from HA history API on startup
- Local high-resolution timeseries DB — WebSocket collector + SQLite storage, training prefers local data
- Dual-source data fetching — merge statistics + states for complete power sensor coverage
- Power proxy UI explainer — how merge works + reminder to retrain after saving
- Live automation sync — show Habitus automations in HA with delete button; refresh endpoint
- Solar surplus notification now requires battery >90% SOC + 1.5h headroom to 100%
- Power proxy sensors — shore+battery fallback fills historical power gaps
- Testing mode toggle — cache clear+retrain on every start when on, version-stamped smart invalidation when off
- Add /api/clear_derived_cache endpoint to nuke stale device_library etc.
- Smart plug ground-truth device template library for NILM (v3.10.12)
- UI — phase attribution badges and per-phase power summary panel
- Per-phase weekly profiles in patterns.py
- Per-phase power columns in build_features()
- Per-phase NILM edge detection and correlation

### Changed
- Add Mastervolt Mass Combi Ultra 24/3500 specs to boat metadata
- Debug: more logging for proxy stats fetch result
- Debug: add logging to proxy backfill to diagnose why historical data not loading
- Comprehensive status snapshot and action plan for next session
- Debug: log local DB power fetch exceptions as warning
- Debug: add time range logging for local DB merge
- Incremental training plan + ENABLE_SMART_SAMPLING flag (not yet implemented)
- Backfill CHANGELOG for 3.10.1–3.10.25 (was stuck at 3.1.0)
- Bump version to 3.10.10
- Bump version to 3.10.5
- Add test_training_0_rows_guard.py regression tests
- Bump to 3.10.2
- Bump version to 3.10.0 — major UI reorg + fixes
- Full tab reorganisation — Home/Suggestions/Automations/Energy/Health/Settings
- Deduplicate card/yaml/add-to-HA/fetch patterns, shared JS utilities
- Nilm, scene_detector, routine_predictor, main_utils coverage tests
- Additional coverage tests for main_utils, activity_hmm + atomic write fixes
- Additional mock coverage boost tests for 65% target
- Mock-based coverage for phantom, room_predictor, main fetch functions
- Set coverage fail_under=40 (realistic baseline; main.py/trainer.py require HA DB)
- Confidence calibration assertions for suggestion scoring
- Integration tests for add/remove HA automation flow
- Add GitHub Actions CI workflow with pytest-cov and Codecov
- Add standardized fetch/build_features telemetry budgets
- Hardening(features): tolerate invalid numeric env overrides

### Fixed
- ACTUALLY fix anomaly score 100 with 0 entities — recalculate score from active anomalies only (top 3 z-scores)
- Anomaly score 100 with 0 entities — reset score to 0 when all anomalies dismissed
- Update inverter specs — 3× Mastervolt Mass Combi Ultra in 3-phase cluster (10.5kW continuous, 21kW peak)
- Inverter overload false alarms — add inverter_capacity_w setting (85% threshold), fixes 539W warning on 3500W inverter
- Update heat pump specs — Fujitsu ASYB12LDC 910W cooling / 1220W heating, max 2.3kW, inverter-driven
- Update induction hob specs — Siemens EX675FEC1E 7.4kW max, 4 zones, PowerBoost, 17 power levels
- Add both heater types — 2× 1kW split-phase panel heaters + 2× 2kW portable fan heaters (32 appliances total)
- Correct heater specs — 2× 1kW split-phase electric heaters (not 2kW fan heaters)
- Proxy sensor dropdowns reset on load — removed redundant refresh that cleared selected values
- CRITICAL — prevent auto-detection from overwriting user power_entity settings on every restart
- Proxy stats uses 'mean' column not 'v' (fetch_stats_sqlite schema)
- Use fetch_stats_sqlite (not fetch_stats) with correct params for proxy backfill
- Proxy backfill resolves db_path locally (was undefined in scope)
- Proxy backfill computes own time range (data_from was undefined in build_features scope)
- Proxy backfill queries full training window (not just local DB range) to use years of Z-Wave shore power data
- Reduce false positives — skip sparse baselines during early training, lower contamination thresholds, prefer proxy sensors with full history
- Train ONLY on hours with power data (no zero-dilution from gaps)
- Restrict training hours to local DB time range (prevents power dilution by behavioral-only hours)
- Allow days_history down to 1 day (was minimum 7) for local DB short-window training
- CRITICAL — build_features reads power_entity from user_settings (not env var)
- Local DB + hybrid fetch variable scope errors (cutoff_ts → computed from data range)
- Better error logging for collector startup, remove shadowing os import
- Missing log instance in web.py
- Collector runs in background thread with own event loop (Flask isn't async)
- Training overwrites user's power_entity with auto-detect — now preserves user_settings.power_entity

