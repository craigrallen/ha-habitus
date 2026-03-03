Do a thorough code review of the specified Habitus module.

The file to review: $ARGUMENTS

If no file is specified, ask the user which module to review.

Review checklist:
1. **Correctness** — Does the logic match its docstring and intent?
2. **Type safety** — Are all annotations present and accurate?
3. **Docstrings** — Do all public functions have Google-style docstrings?
4. **Error handling** — Are exceptions caught specifically (no bare `except`)?
5. **Test coverage** — Are there corresponding tests? Do they cover edge cases?
6. **HA constraints** — No outbound network calls except to HA_URL/HA_WS
7. **Data isolation** — Does it write only to DATA_DIR, never to source tree?
8. **Performance** — Any obvious inefficiencies in pandas/numpy operations?
9. **Style** — Would ruff and black pass without changes?

Output a structured report with:
- Summary (overall quality rating: Good / Needs Work / Requires Refactor)
- Issues found (severity: high / medium / low, file:line, description, suggested fix)
- What's done well
