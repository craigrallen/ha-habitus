# TASK: Smart Automations — "Read the User's Mind"

## Context
Habitus is a Home Assistant add-on that monitors behavioral patterns. It already has:
- `patterns.py` — basic pattern discovery (hourly profiles, weekly, sleep detection)
- `automation_gap.py` — checks if suggested automations exist in HA
- `automation_score.py` — scores automation effectiveness
- `insights.py` — energy insights
- `web.py` — Flask web UI with tabs: Overview, Anomaly Breakdown, Automations, Energy & Patterns, Insights, Settings

## What to Build

### 1. Scene Detector (`scene_detector.py`)
Mine entity co-occurrence patterns to find "implicit scenes" — groups of entities that change state together.
- Use association rule mining (mlxtend Apriori/FP-Growth) or similar
- Track which lights/switches/media activate together within a time window (e.g. 5 min)
- Detect time-of-day patterns: "Every weekday evening 18-20, these 3 lights turn on together"
- Name scenes automatically: "Evening Living Room", "Morning Kitchen"
- Output: list of discovered scenes with entities, time patterns, confidence scores

### 2. Automation Builder (`automation_builder.py`)
Generate actual HA automation YAML from discovered patterns:
- Time-based: "Turn on X lights at Y time on Z days"
- State-based: "When motion detected in room, turn on lights"
- Scene-based: "Create scene from co-occurring entity states"
- Each suggestion should have: description, confidence %, YAML, entities involved
- Check against HA's existing automations (GET /api/config/automation/config) to avoid duplicates

### 3. Import Existing HA Automations
- Fetch all user automations from HA REST API
- Display them in the UI alongside Habitus suggestions
- Show which Habitus suggestions overlap with existing automations

### 4. Merge UI Tabs
- Merge "Automations" and "Insights" tabs into single **"Smart Home"** tab
- Sub-sections: Discovered Scenes, Suggested Automations, Your Automations, Insights
- ALL YAML code behind `<details><summary>` expand buttons
- Entity picker widget for lights/scenes
- Show implicit scenes prominently: "We noticed you always turn on [X, Y, Z] together at 7pm"

### 5. API Endpoints
- `GET /api/scenes` — discovered implicit scenes
- `GET /api/ha_automations` — user's existing HA automations
- `GET /api/smart_suggestions` — merged suggestions with confidence, YAML, overlap info
- Keep existing endpoints working for backward compat

## Technical Notes
- HA REST API base: use `HA_URL` and `HA_TOKEN` env vars (already in main.py)
- All data files go in `/data/` (DATA_DIR)
- Add `mlxtend` to requirements.txt for association rule mining
- Python 3.13, must compile cleanly
- Keep `default=str` on ALL json.dump calls
- Update CHANGELOG.md with changes
- Do NOT change the version number in config.yaml

## Quality
- Run `python3 -m py_compile` on every changed file
- Test imports work
- Keep the web UI responsive (mobile-friendly, max-width 960px)
- Dark/light mode must still work

## Files to Modify
- `habitus/habitus/scene_detector.py` (NEW)
- `habitus/habitus/automation_builder.py` (NEW)
- `habitus/habitus/web.py` (merge tabs, add expand buttons, entity picker)
- `habitus/habitus/main.py` (wire in new modules)
- `habitus/requirements.txt` (add mlxtend)
- `habitus/CHANGELOG.md` (update)

## DO NOT
- Change version number in config.yaml
- Break existing API endpoints
- Remove existing functionality
- Use scientific notation for power values
