# Changelog

All notable changes to Habitus are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [3.1.0] - 2026-04-07

### Added

#### Intelligence Features
- **Data-driven automation suggestions** (TASK-001) — morning lights, peak tariff, vacancy detection, bilge temperature, and shore+battery alerts generated from discovered patterns
- **Per-entity z-score anomaly breakdown** (TASK-002) — top-5 entity anomalies when score > 40, persisted to `entity_baselines.json`
- **Hemisphere-aware seasonal models** (TASK-003) — per-season IsolationForest with `seasonal_models.pkl` bundle and Southern Hemisphere support
- **HA notification integration** (TASK-004) — `send_notification()`, `persistent_notification()`, daily digest with configurable hour
- **Energy insights module** (TASK-005) — peak hours, top-5 consumers, off-peak waste estimation, solar self-consumption ratio via `/api/insights`
- **Enhanced Lovelace cards** (TASK-006) — 4 card types (Pulse, Chip, Detail Panel, Timeline) fetching top-3 anomaly reasons and suggestion confidence from API

#### Sensor Intelligence
- **5-type sensor classifier** (TASK-015) — `accumulating`, `binary`, `gauge`, `event`, `setpoint` via HA `state_class` + history analysis; stored in `_meta.sensor_type`
- **Accumulating sensor rate-of-change** (TASK-016) — hourly delta baselines with 24h bootstrap exemption
- **Binary sensor timing & frequency scoring** (TASK-017) — on-fraction, transition frequency, and duration-based anomaly detection
- **Per-entity cold start protection** (TASK-018) — entities < 7 days exempt from scoring; low-sample slots weighted 0.5x; lifecycle tracking in `entity_lifecycle.json`
- **Confidence-weighted anomaly scores** (TASK-019) — per-entity confidence from age, sample count, and sensor type certainty
- **Adaptive IsolationForest contamination** (TASK-020) — 5-tier ramp (0.005 to 0.05) based on training age; tier changes trigger full retrain
- **Data quality guard** (TASK-021) — filters impossible values (negative power, temperature/humidity bounds, gauge jumps > 10x, stuck sensors >= 24h); `/api/sensor_health` endpoint

#### Infrastructure
- Autonomous coding loop (`ralph.sh`) with PRD-driven task management
- `progress.txt` task completion log
- `codecov.yml` with 70% project target and 60% patch target

### Changed

#### Code Quality Improvements
- **`patterns.py` refactored** — monolithic 580-line `generate_suggestions()` split into 6 focused helpers: `_routine_suggestions()`, `_energy_suggestions()`, `_boat_suggestions()`, `_anomaly_suggestions()`, `_pattern_driven_suggestions()`, `_lovelace_suggestions()`
- **`web.py` decomposed** — 1,263-line inline HTML extracted to `templates/index.html`; now uses Flask `render_template()` with Jinja2 variables (1,554 -> 288 lines)
- **Pickle replaced with joblib** in `seasonal.py` and `main.py` for safer model serialization (no arbitrary code execution on load)
- **`automation_gap.py` async consistency** — replaced synchronous `urllib.request` with `aiohttp` to match the async `analyse()` interface
- **Exception handling narrowed** across 8 modules — broad `except Exception` replaced with specific types (`OSError`, `json.JSONDecodeError`, `FileNotFoundError`, `ValueError`, `EOFError`)
- **Type annotations added** to all public functions in `automation_gap.py`
- **Dependency versions pinned** in `requirements.txt` to match `pyproject.toml` floor versions
- Added `joblib>=1.3` and `aiohttp>=3.9` to both `requirements.txt` and `pyproject.toml`

### Fixed
- **Version mismatch** — `run.sh` displayed hardcoded `v2.27.0` instead of actual version from `config.yaml`; now uses `${HABITUS_VERSION}` from bashio
- **Hardcoded Lovelace ingress path** — `habitus-card.js` had plugin ID `57582523` hardcoded; now auto-detects from script tags, DOM, or window location
- **`discover_patterns()` KeyError** — accessing `hourly.loc[h, ...]` for hours missing from sparse data; now checks `h in hourly.index` first
- Responsive UI for anomaly breakdown table on mobile
- Theme toggle button and JS function

### Tests
- **425 total tests** (up from 0 at project start)
- **25 new edge case tests** in `test_edge_cases.py`:
  - Empty and minimal DataFrame handling
  - NaN and Inf value propagation
  - Corrupt JSON file recovery for 6 modules
  - Contamination tier boundary conditions
  - Season detection across all 12 months
  - Sensor classifier edge cases (short history, all-same values, binary floats)
- Core module coverage: `patterns.py` 96%, `anomaly_breakdown.py` 96%, `seasonal.py` 93%, `sensor_classifier.py` 99%, `insights.py` 90%
