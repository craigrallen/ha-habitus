# Habitus — Claude Code Instructions

## Project Overview
Habitus is a Home Assistant add-on that learns household behavioral patterns, detects anomalies, and generates automation suggestions. 100% local ML, no cloud, no telemetry.

**Stack:** Python 3.11+, scikit-learn (IsolationForest), pandas, Flask, WebSocket → HA API
**Version:** 2.9.0 | **License:** MIT

---

## Repository Structure
```
habitus/
  habitus/         — Python package (source of truth)
    main.py          Orchestration: fetch → train → score → publish
    activity.py      Activity baseline (motion, lights, presence, media, door)
    anomaly_breakdown.py  Per-entity z-score scoring
    patterns.py      Pattern discovery + automation YAML generation
    seasonal.py      Per-season IsolationForest models
    drift.py         Routine drift / changepoint detection
    phantom.py       Phantom load hunter (standby waste)
    automation_score.py  Existing automation effectiveness scoring
    automation_gap.py    Automation gap discovery
    progressive.py   Progressive training logic
    trainer.py       Model training orchestration
    web.py           Flask ingress web UI + API endpoints
  www/
    habitus-card.js  Custom Lovelace card
  Dockerfile         Add-on container
  config.yaml        HA add-on manifest
  requirements.txt   Runtime dependencies
  run.sh             Container entrypoint
tests/
  conftest.py        Shared fixtures (sample_df, tmp_data_dir, mock_ha_states)
  test_activity.py
  test_anomaly_breakdown.py
  test_patterns.py
  test_seasonal.py
pyproject.toml       Build, lint, type, test config
AGENTS.md            Autonomous agent instructions (Ralph Wiggum)
PRD.md               Task backlog with acceptance criteria
progress.txt         Completed task log
ralph.sh             Autonomous coding loop script
```

---

## Environment
- **OS:** Windows 11 with Git Bash
- **Python:** 3.14.3 (host); 3.11+ inside Docker container
- **Node.js:** installed (for Lovelace card dev)
- **Claude Code CLI:** installed at `~/.npm-global` / npm global

**Always use bash/Unix syntax** — forward slashes, not backslash.

---

## Quality Gates — Run in This Order
```bash
# From repo root (Python Scripts not on PATH by default on this machine):
export PATH="/c/Users/Widemind/AppData/Local/Python/pythoncore-3.14-64/Scripts:$PATH"

ruff check habitus/habitus/
black --check habitus/habitus/
mypy habitus/habitus/
pytest --cov=habitus/habitus --cov-report=term-missing
```

**Rules:**
- All four must pass before any commit
- Minimum 70% test coverage (enforced by CI)
- `web.py` is excluded from coverage (Flask routes — integration tested separately)
- Fix issues rather than suppressing with `# noqa` or `# type: ignore` unless genuinely necessary

---

## Code Standards
- Google-style docstrings on all public functions and classes
- Full type annotations on all function signatures
- `ruff` line length: 100 chars; `black` line length: 100 chars
- No bare `except:` — always catch specific exceptions
- No outbound network calls except to `HA_URL`/`HA_WS` (local HA instance)
- Data artifacts go to `DATA_DIR` (`/data` in container, `tmp_path` in tests)
- HA is source of truth — Habitus stores only what it inferred, never raw history

---

## Testing Patterns
The `conftest.py` provides these fixtures — use them, don't reinvent:
- `tmp_data_dir` — temporary DATA_DIR with env vars patched
- `sample_df` — 90-day hourly DataFrame covering all sensor categories
- `sample_features` — pre-built feature matrix
- `mock_ha_states` — realistic HA `/api/states` response

For HA REST calls: use `responses` library to mock HTTP.
For async code: `pytest-asyncio` is configured with `asyncio_mode = "auto"`.

---

## Git Workflow
```bash
# Stage only changed source files — never git add -A
git add habitus/habitus/filename.py tests/test_filename.py
git commit -m "feat: short description"
# or
git commit -m "fix: short description"
```

- One feature per commit
- Never commit with failing tests or lint
- Never force-push or amend published commits

---

## Key Data Files (runtime, in `/data`)
| File | Contents |
|------|----------|
| `model.pkl` | Main IsolationForest model |
| `scaler.pkl` | Feature scaler |
| `baseline.json` | Hourly power norms |
| `activity_baseline.json` | Per-sensor activity norms |
| `entity_baselines.json` | Per-entity z-score baselines |
| `patterns.json` | Discovered routines |
| `suggestions.json` | Generated automation YAML |
| `phantom_loads.json` | Identified standby wasters |
| `drift_report.json` | Routine drift metrics |
| `automation_scores.json` | Automation effectiveness |
| `automation_gaps.json` | Missing automations |
| `run_state.json` | Discovery window + last run timestamp |

---

## HA Sensor Entities Published
| Entity | Description |
|--------|-------------|
| `sensor.habitus_anomaly_score` | 0–100 combined anomaly score |
| `binary_sensor.habitus_anomaly_detected` | `on` when score > threshold |
| `sensor.habitus_training_days` | Days of history used |
| `sensor.habitus_entity_count` | Number of behavioral sensors tracked |

---

## Autonomous Coding (Ralph Wiggum)
To run autonomous task iteration:
```bash
bash ralph.sh <N>   # N = number of iterations
# Example: bash ralph.sh 3
```
The agent reads `PRD.md` for tasks, appends to `progress.txt` after each commit.

---

## Sub-Agents
Invoke these via `/agent:<name>` or the Task tool for specific work:
- **test-runner** — runs full quality gates and reports failures
- **code-reviewer** — reviews a module for style, coverage, correctness
- **prd-planner** — breaks a new feature into PRD tasks
- **ha-schema-checker** — validates config.yaml and automation YAML output

Sub-agent prompts: `.claude/agents/`

---

## Invoked Automatically
These sub-agents are invoked automatically when relevant:
- **test-runner**: after any code change
- **code-reviewer**: before committing new modules
- **prd-planner**: when user describes a new feature idea without a task

---

## Common Commands
| Slash command | What it does |
|---------------|-------------|
| `/check`      | Run all quality gates |
| `/ralph <N>`  | Start Ralph Wiggum for N iterations |
| `/status`     | Show completed tasks from progress.txt |
| `/review <file>` | Code review a specific module |
| `/plan <feature>` | Turn a feature idea into PRD tasks |
