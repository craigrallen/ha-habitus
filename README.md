<p align="center">
  <img src="logo.png" width="100" alt="Habitus">
</p>

<h1 align="center">Habitus</h1>

<p align="center">
  <strong>Behavioral intelligence for Home Assistant — 100% local, zero cloud, zero compromise.</strong>
</p>

Habitus is building a smarter kind of smart home: one that understands behavior, not just triggers. Running fully local in Home Assistant, it learns routines, predicts likely needs, explains energy usage, and adapts automations as life changes — with transparent approvals, user feedback loops, and privacy-by-default architecture. Built in Europe, Habitus treats your data like your home: private property, never platform fuel.

<p align="center">
  <a href="https://github.com/craigrallen/ha-habitus/actions/workflows/ci.yml">
    <img src="https://github.com/craigrallen/ha-habitus/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/craigrallen/ha-habitus">
    <img src="https://codecov.io/gh/craigrallen/ha-habitus/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://github.com/craigrallen/ha-habitus/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="MIT">
  </a>
  <a href="https://buymeacoffee.com/craigrallen">
    <img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow?style=flat&logo=buy-me-a-coffee" alt="Buy Me A Coffee">
  </a>
</p>

---

## 🔒 Local First. Always.

Habitus is built on a single principle: **your home's data belongs to you**.

Every byte of analysis happens on your Home Assistant device. No data is sent to any server, no API calls leave your network, no account is required. Your behavioural patterns — when you wake up, when you're home, what you watch, how you live — stay exactly where they belong.

This isn't a marketing claim. It's architectural:

> Also, yes — the docs hide subtle 70s/80s/90s references. Consider it a side quest.

- **No outbound network calls** from the add-on (only to your own HA instance)
- **No telemetry** of any kind
- **No cloud model** — the ML runs on your hardware
- **Offline-first** — works without internet after install
- **Signal over spectacle** — clarity first, hype last
- **Open source** — audit every line yourself

---

## What Habitus Does

Habitus learns the *living patterns* of your home — not just power consumption, but the full picture of daily life:

### 🏃 Activity Baseline
Beyond simple power usage, Habitus tracks:

| Signal | What it reveals |
|--------|----------------|
| Motion sensors | Movement patterns — corridors, rooms, garden |
| Light switches | Which rooms are occupied, what time |
| Presence sensors | Who is home, when they arrive and leave |
| Media players | TV on, music playing, which room is active |
| Door/window sensors | Arrivals, departures, ventilation habits |
| Person entities | Occupancy fraction, device tracker signals |
| Weather sensors | Outdoor context — cold days mean heating, not anomaly |

The key insight: **power alone lies**. A 2kW spike at 07:00 is normal if the kitchen is active with someone home. It's anomalous if nobody is home, no lights are on, and no motion has been detected.

### 🧠 Anomaly Detection
An IsolationForest model trained on your actual history detects when the combination of power, activity, presence and environmental signals deviates from what's normal *for this time of day, day of week, and season*.

### 🌱 Seasonal Intelligence
Separate models for each season. Cold winter mornings with high heating load won't trigger false anomalies in summer.

### 🤖 Proposed Automations
Habitus generates ready-to-use HA automation YAML based on discovered patterns:
- Morning routines, night modes
- High power alerts calibrated to *your* baseline
- Boat-specific: bilge alerts, battery protection, shore power monitoring
- Occupancy-aware load control
- Daily energy digests

### 📊 Insights Dashboard
A polished web UI served via HA ingress (sidebar button):
- Animated anomaly score gauge
- Per-entity anomaly breakdown (z-score ranked)
- Hourly and seasonal baselines
- One-click Copy YAML / Add to HA for all suggestions

---

## Feature Catalogue (Current)

Below is the practical feature set currently in the repo (see `habitus/CHANGELOG.md` for version-by-version detail).

### Detection & Scoring
- Whole-home anomaly score (`sensor.habitus_anomaly_score`) with thresholded anomaly binary sensor
- Per-entity anomaly breakdown with confidence weighting and sensor-type-aware scoring
- Sensor classifier taxonomy (`accumulating`, `binary`, `gauge`, `event`, `setpoint`)
- Data-quality guards, impossible-value filtering, and cold-start protection
- Adaptive contamination / training-age-aware model behavior

### Behavioral Learning
- Activity baseline engine (motion, lights, doors, media, presence, person/device-tracker signals)
- Routine predictor (humidity/temperature-derived shower, bath, cooking pattern detection)
- Sequence mining (PrefixSpan) for ordered behavior flows
- Markov chain next-action prediction
- Hidden Markov Model (HMM) activity-state inference
- Behavior drift analysis for “what changed” over time

### Energy Intelligence
- Power shape classification (`steady`, `cycling`, `decaying`, `phased`)
- Appliance fingerprinting/event detection from power signatures
- Deep correlation engine for statistically significant cross-entity patterns
- Energy forecast with uncertainty estimates
- Overnight baseline / phantom-load style analysis and cost-aware insights

### Smart Home Automation
- Scene detector (implicit scenes from co-occurrence patterns)
- Automation builder (time/state/scene-based YAML suggestions)
- Automation gap and automation score engines
- Conflict detector (wasteful or contradictory states) with suggested fix-it automations
- Dynamic automation generation with confidence thresholds
- Existing HA automation import + overlap detection against suggested automations

### Home Assistant Integration
- Area registry integration with room-aware predictions
- Room predictor based on area mapping + time/day context
- Notification support for anomalies and actionable insights
- Ingress web app with Smart Home + Geek views and API endpoints for state/insights/suggestions

### Feedback, Training & Controls
- User feedback loop (confirm/dismiss anomalies) to tighten/widen per-entity thresholds
- Device training mode for user-labeled signatures (start/stop capture + naming)
- Configurable history depth windows (30d → all history)
- Full-train endpoint and progressive training flow
- Seasonal model handling that extends knowledge without regressing stronger models

### Privacy & Architecture
- 100% local processing on your HA host
- No required cloud account or outbound telemetry path for core operation
- Stores model artifacts, not raw mirrored history

## Installation

1. In Home Assistant: **Settings → Add-ons → Add-on Store → ⋮ → Repositories**
2. Add: `https://github.com/craigrallen/ha-habitus`
3. Find **Habitus** in the store → **Install**
4. Configure (optional) → **Start**
5. Click the 🧠 icon in the sidebar

**First run** fetches all available HA long-term statistics (can take 20–40 minutes for large installations). Subsequent runs are incremental.

---

## Configuration

```yaml
scan_interval_hours: 6       # How often to re-score (default: 6h)
days_history: 3650           # Training window — set high, HA limits to what it has
notify_service: notify.notify # HA notify service for anomaly alerts
notify_on_anomaly: true       # Send notification when anomaly detected
anomaly_threshold: 70         # Score threshold for anomaly notification (0–100)
```

---

## Published Sensors

| Entity | Description |
|--------|-------------|
| `sensor.habitus_anomaly_score` | 0–100 combined anomaly score |
| `binary_sensor.habitus_anomaly_detected` | `on` when score > threshold |
| `sensor.habitus_training_days` | Days of history used |
| `sensor.habitus_entity_count` | Number of behavioral sensors tracked |

---

## Architecture

```
HA Recorder DB (your data)
        │
        ▼
  WebSocket API          ← No raw data stored locally
  (live query each run)
        │
        ▼
  Feature extraction     ← Power + Activity + Presence + Weather
        │
        ├── IsolationForest (main model)
        ├── IsolationForest (seasonal × 4)
        │
        ▼
  /data (artifacts only) ← ~200KB total
  ├── model.pkl          ← Trained model weights
  ├── scaler.pkl         ← Normalization params
  ├── baseline.json      ← Hourly power norms
  ├── activity_baseline.json  ← Per-sensor activity norms
  ├── entity_baselines.json   ← Per-entity z-score baselines
  ├── patterns.json      ← Discovered routines
  ├── suggestions.json   ← Generated automation YAML
  └── run_state.json     ← Discovery window + last run
        │
        ▼
  HA REST API            ← Publish 4 sensors back to HA
```

HA is the source of truth. Habitus stores only what it *learned*, never what it *read*.

---

## Development

### Prerequisites

```bash
git clone https://github.com/craigrallen/ha-habitus
cd ha-habitus
pip install -e ".[dev]"
```

### Running tests

```bash
pytest                              # All tests with coverage
pytest tests/test_activity.py -v   # Specific module
pytest --co -q                     # List all tests
```

### Linting

```bash
ruff check habitus/habitus/        # Lint
black habitus/habitus/             # Format
mypy habitus/habitus/              # Type check
```

### Code standards

- All public functions must have docstrings (Google style)
- Type annotations required on all function signatures
- New features require tests before merge (70% coverage minimum, enforced by CI)
- Ruff + Black must pass — no exceptions
- PRs require passing CI before merge

### Project structure

```
habitus/
  Dockerfile            — Add-on container
  config.yaml           — HA add-on manifest
  build.yaml            — Build architecture matrix
  requirements.txt      — Runtime dependencies
  run.sh                — Container entrypoint
  logo.png              — Add-on logo
  habitus/
    __init__.py
    main.py             — Orchestration: fetch → train → score → publish
    activity.py         — Activity baseline engine (lights, motion, presence, media)
    anomaly_breakdown.py — Per-entity z-score scoring
    patterns.py         — Pattern discovery + automation suggestion generation
    seasonal.py         — Per-season IsolationForest models
    web.py              — Flask ingress web UI
tests/
  conftest.py           — Shared fixtures
  test_activity.py
  test_anomaly_breakdown.py
  test_patterns.py
  test_seasonal.py
.github/
  workflows/
    ci.yml              — Lint + test + build gate
    release.yml         — Tag → GitHub Release
```

---

## Contributing

Pull requests welcome. Please:

1. Fork and create a feature branch
2. Write tests for new functionality
3. Ensure `pytest`, `ruff check`, and `black --check` all pass locally
4. Open a PR — CI will gate on coverage and lint

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Support

If Habitus is useful, consider buying me a coffee ☕

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/craigrallen)

---

## License

MIT © Craig Allen
