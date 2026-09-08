# Sensor identification

Habitus now retains Home Assistant sensor attributes and recorder statistics units.
An entity name containing `energy` no longer automatically means kWh. Explicit
units are preserved; compatible live values are converted to the baseline unit.
Power calculations normalize W/kW/MW, and temperature features normalize °C/°F/K.
Counters use consumption per elapsed hour, with the rate unit displayed explicitly.

Open **Sensor identification and corrections** from the dashboard to inspect
reported units, selected behavior, and the identification source. Use the form to
correct a unit or behavior, or restore automatic detection. Unit corrections
reinterpret a misreported unit; they are not a request to rescale the raw data.
Save a correction and run **Full retrain** on the dashboard. A corrected sensor
is paused until its baseline is rebuilt. Corrections persist locally in
`entity_overrides.json`; reported attributes are stored in `entity_metadata.json`.

The first scheduled run after upgrading rebuilds the model and baselines using
the new identification version. Incompatible units are excluded from live scoring.
Unknown units remain unknown. Name hints are used only when metadata is absent.
History-based classification remains a fallback for entities without metadata.

Additional fixes include metadata-aware motion/door identification, correctly
scaled energy insights, seconds/milliseconds support in phantom reports, recovery
of stale data-quality flags during rebuilding, and a nonblocking full-retrain
endpoint. The dashboard no longer requests a font from Google.

Validation commands (Windows development environment):

```powershell
.venv/Scripts/python.exe -m ruff check habitus/habitus/
.venv/Scripts/python.exe -m black --check habitus/habitus/
.venv/Scripts/python.exe -m mypy habitus/habitus/
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --cov=habitus/habitus --cov-report=term-missing
```

The cache plugin is disabled because reading this checkout's existing OneDrive
pytest cache stalled locally. This does not disable tests. Tests use mocked Home
Assistant responses; validation against a running household installation remains
necessary. Repository-wide coverage must reach 70% before committing.

Validated on 2026-09-08: 435 tests passed, including 35 new regression cases.
Ruff, Black, mypy, and Git whitespace checks passed. Overall coverage is 59%,
below the 70% commit gate. The user explicitly authorized committing and opening
a PR after this limitation was reported. Existing
pandas downcasting warnings remain in the test output.

Integration with current main preserves SQLite history and the template dashboard. The metadata snapshot is refreshed before training and attached after combining raw and statistical history. Final merge validation uses Linux CI; two existing atomic-write tests encounter Windows-specific file locks locally.
