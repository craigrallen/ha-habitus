# Habitus — Agent Instructions

## Project
Home Assistant add-on. Behavioral intelligence layer: learns household patterns, detects anomalies, generates automation suggestions. 100% local ML, no cloud.

## Repository Layout
```
habitus/habitus/     — Python package (main source)
tests/               — pytest test suite
habitus/Dockerfile   — Add-on container
habitus/config.yaml  — HA add-on manifest
habitus/www/         — Lovelace card JS
pyproject.toml       — Build, lint, test config
```

Key modules:
- `main.py`              — Orchestration entry point
- `activity.py`          — Activity baseline (motion, lights, presence, media)
- `anomaly_breakdown.py` — Per-entity z-score scoring
- `patterns.py`          — Pattern discovery + automation YAML generation
- `seasonal.py`          — Per-season IsolationForest models
- `drift.py`             — Routine drift detection (changepoint)
- `phantom.py`           — Phantom load hunter
- `progressive.py`       — Progressive training
- `trainer.py`           — Model training orchestration
- `automation_gap.py`    — Automation gap analysis
- `automation_score.py`  — Existing automation effectiveness scoring
- `web.py`               — Flask ingress web UI

## Quality Gates (must all pass before committing)

```bash
# From repo root:
cd "C:/Users/Widemind/OneDrive/Documents/Claude Code/Habitus"
export PATH="/c/Users/Widemind/AppData/Local/Python/pythoncore-3.14-64/Scripts:$PATH"

# Lint
ruff check habitus/habitus/

# Format check
black --check habitus/habitus/

# Type check
mypy habitus/habitus/

# Tests with coverage
pytest --cov=habitus/habitus --cov-report=term-missing
```

**Minimum 70% test coverage enforced — do not commit below this threshold.**

## Code Standards
- All public functions must have Google-style docstrings
- Type annotations required on all function signatures
- Ruff + Black must pass — no exceptions
- New features require tests before commit

## Commit Style
```
feat: short description of what was added
fix: short description of what was fixed
```
One feature per commit. After committing, append to `progress.txt`.

## Constraints
- No outbound network calls (all data stays local to HA)
- No cloud APIs, no telemetry
- Python 3.11+ only
- scikit-learn IsolationForest is the primary ML model
- Flask serves the web UI via HA ingress
