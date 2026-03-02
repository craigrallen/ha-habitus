# Contributing to Habitus

Thank you for considering a contribution! Habitus is a local-first,
privacy-preserving HA add-on — every contribution must uphold that principle.

## Local First — Non-Negotiable

No code will be merged that:
- Makes outbound network calls to any server outside the local HA instance
- Collects or transmits any user data, usage analytics, or telemetry
- Requires an account, API key, or cloud service to function
- Stores raw HA state data outside the HA device

## Getting Started

```bash
git clone https://github.com/craigrallen/ha-habitus
cd ha-habitus
pip install -e ".[dev]"
```

## Before You Submit

### Tests

All new code must include tests. The CI enforces a 70% coverage floor.

```bash
pytest                     # Must pass
pytest --cov-report=term   # Check your coverage
```

### Lint & Format

```bash
ruff check habitus/habitus/     # Zero warnings required
black habitus/habitus/          # Auto-format
mypy habitus/habitus/           # Type errors fail CI
```

### Docstrings

Every public function must have a Google-style docstring:

```python
def my_function(x: float, y: str) -> dict:
    """Short one-line description.

    Longer explanation if needed. Explain *why*, not just what.
    Local-first note if relevant.

    Args:
        x: What this is.
        y: What this is.

    Returns:
        Description of what is returned.

    Raises:
        ValueError: When and why.
    """
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `test:` — tests only
- `refactor:` — no behaviour change
- `chore:` — build/CI/tooling

## CI Checks

Every PR runs:
1. **Ruff lint** — zero warnings
2. **Black format check** — auto-format must match
3. **MyPy type check** — no type errors
4. **pytest** on Python 3.11, 3.12, 3.13
5. **Codecov** — coverage must not drop below 70%
6. **Docker build** — add-on image must build for aarch64

All checks must pass before merge.

## Architecture Decisions

Before changing the core architecture (data flow, model choice, storage
format), open an issue first. Habitus has strong constraints:

- HA is the source of truth. Raw data is never stored locally.
- /data stores only inference artifacts (model weights, baselines, suggestions)
- All ML runs on the HA device — no remote inference
- Feature extraction must handle missing/unavailable sensor gracefully

## Questions?

Open a GitHub Discussion or Issue.
