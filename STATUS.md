# Habitus Status — 2026-07-18

> Refreshed from tracked repository state (git history, pyproject.toml,
> config.yaml, CHANGELOG.md, tests/). This file cannot attest to the health
> of any live/running add-on instance — see **Superseded Content** below for
> where to check that.

## Current Release

| Source | Version |
|---|---|
| `pyproject.toml` `[project].version` | 4.1.4 |
| `habitus/config.yaml` (HA add-on manifest) | 4.1.4 |
| `CHANGELOG.md` latest entry | [4.1.4] - 2026-05-03 |
| `habitus/CHANGELOG.md` latest entry | [4.1.4] - 2026-05-03 |
| `CLAUDE.md` `**Version:**` | 4.1.4 |

All five are aligned as of this refresh. `tests/test_version_alignment.py`
(added in this change) asserts this stays true — see **Version-Alignment
Guard** below.

## Recent History

Per `git log` and `progress.txt`, the most recent completed work was CI/release
hygiene: restoring broken web API endpoints, wiring `full_train` into
`TrainerManager`, updating GitHub Actions to Node 24, and aligning
`pyproject.toml` with the add-on manifest version (the exact drift this
change now guards against). Run `git log --oneline -10` for the current
commit list rather than trusting a copy pasted here.

## Repository Shape (verified by listing/grep)

- `habitus/habitus/` — the Python package (`main.py`, `web.py`, `activity.py`,
  `anomaly_breakdown.py`, `patterns.py`, `seasonal.py`, `drift.py`,
  `nilm_disaggregator.py`, `trainer.py`, and others — behavioral,
  energy/NILM, and automation-suggestion features).
- `tests/` — a pytest suite covering the package, plus `conftest.py` shared
  fixtures (`tmp_data_dir`, `sample_df`, `sample_features`, `mock_ha_states`).
  Run `pytest --collect-only -q | tail -1` for a current test count rather
  than trusting a number written here.
- `.github/workflows/ci.yml` — lint (ruff/black/mypy, mypy non-blocking),
  prettier (non-blocking), test matrix (Python 3.11/3.12/3.13, pytest +
  coverage → Codecov), NILM/scene benchmark with regression gate, and an
  aarch64 Docker build check (no push).
- `.github/workflows/release.yml` — present; not inspected in this pass.

## Quality Gates

Not run to completion as part of this refresh — see the PR/commit evidence
for this change for what was actually verified. Before relying on this file,
run the gates yourself (`/check` or the commands in CLAUDE.md) or check CI
status on the current commit.

## Version-Alignment Guard (added in this change)

`tests/test_version_alignment.py` — four tests, no ML dependencies required:

1. `pyproject.toml` `[project].version` == `habitus/config.yaml` `version`
2. Latest `## [X.Y.Z]` heading in `CHANGELOG.md` == the manifest version
3. Latest `## [X.Y.Z]` heading in `habitus/CHANGELOG.md` == the manifest version
4. `**Version:**` declared in `CLAUDE.md` == the manifest version

This exists because pyproject/manifest version drift already happened once
(fixed 2026-05-03, see CHANGELOG), and CLAUDE.md separately drifted to a
stale `2.9.0` with nothing catching either recurrence. Regular `pytest` runs
(in an environment with project deps installed) will now fail fast if a
version bump only touches some of these files.

## Superseded Content

Everything below this line in the previous STATUS.md (dated 2026-03-20,
version 3.11.9 — WebSocket collector debugging, local timeseries DB backfill
state, "0.1% non-zero power feature" issue) described **live add-on runtime
state on a specific deployment at that time**, not repository state. It is
several major versions and many commits behind current `main` and has been
removed rather than carried forward stale. If that work is still relevant,
check the running add-on directly — this file cannot attest to runtime state.

**To check live deployment state:** `ha apps logs <slug>` / the add-on's own
`/api/state` endpoint on the target Home Assistant instance. Nothing in this
repository can confirm what a running instance is currently doing.
