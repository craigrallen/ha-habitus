Run all Habitus quality gates in order and report results.

Execute these commands from the repo root:

```bash
ruff check habitus/habitus/
black --check habitus/habitus/
mypy habitus/habitus/
pytest --cov=habitus/habitus --cov-report=term-missing
```

Report:
- Any lint violations with file and line number
- Any formatting issues
- Any type errors
- Test results: pass/fail count, coverage percentage
- Whether the code is ready to commit (all gates pass + coverage ≥70%)

If anything fails, explain what failed and suggest the fix.
