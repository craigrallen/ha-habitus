"""Automation Gap Analyser — checks if Habitus suggestions already have HA automations,
and if so whether they're working well or need improvement."""

import datetime
import json
import logging
import os
import re

import yaml

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
GAP_PATH = os.path.join(DATA_DIR, "automation_gap.json")


# ── Intent patterns ────────────────────────────────────────────────────────────

_INTENT_PATTERNS = [
    {
        "intent": "lights_off_empty_room",
        "keywords": [
            (
                ["turn off", "lights off", "switch off"],
                ["no motion", "empty", "nobody", "unoccupied", "when away"],
            ),
        ],
        "trigger_type": "no_motion",
        "action_type": "turn_off",
        "entity_domains": ["light", "switch"],
    },
    {
        "intent": "reduce_standby",
        "keywords": [
            (
                ["turn off", "standby", "phantom", "reduce", "cut"],
                ["standby", "phantom", "idle", "not in use"],
            ),
        ],
        "trigger_type": "idle",
        "action_type": "turn_off",
        "entity_domains": ["switch", "media_player"],
    },
    {
        "intent": "predictive_heating",
        "keywords": [
            (["pre-heat", "heat before", "warm up", "preheat"], []),
        ],
        "trigger_type": "schedule",
        "action_type": "set_temperature",
        "entity_domains": ["climate"],
    },
    {
        "intent": "security_lock",
        "keywords": [
            (["lock", "secure", "arm"], []),
        ],
        "trigger_type": "departure",
        "action_type": "lock",
        "entity_domains": ["lock", "alarm_control_panel"],
    },
    {
        "intent": "notification",
        "keywords": [
            (["alert", "notify", "notification", "remind"], []),
        ],
        "trigger_type": "event",
        "action_type": "notify",
        "entity_domains": [],
    },
    {
        "intent": "lights_on_motion",
        "keywords": [
            (["turn on", "lights on", "switch on"], ["motion", "presence", "arrive", "enter"]),
        ],
        "trigger_type": "motion",
        "action_type": "turn_on",
        "entity_domains": ["light", "switch"],
    },
    {
        "intent": "media_off_idle",
        "keywords": [
            (["turn off", "switch off"], ["tv", "media", "playing", "idle", "inactive"]),
        ],
        "trigger_type": "idle",
        "action_type": "turn_off",
        "entity_domains": ["media_player", "switch"],
    },
]


def _match_intent(text):
    """Try to match a suggestion text to a known intent."""
    lower = text.lower()
    for pat in _INTENT_PATTERNS:
        for keyword_group in pat["keywords"]:
            primary, secondary = keyword_group
            primary_match = any(kw in lower for kw in primary)
            if not primary_match:
                continue
            if secondary and not any(kw in lower for kw in secondary):
                continue
            return pat
    return None


def _extract_entities_from_text(text, known_entity_ids):
    """Extract entity IDs mentioned (by name fragments) in suggestion text."""
    lower = text.lower()
    matches = []
    for eid in known_entity_ids:
        parts = eid.split(".", 1)
        if len(parts) < 2:
            continue
        domain, name = parts
        friendly = name.replace("_", " ")
        if friendly in lower or name in lower.replace(" ", "_"):
            matches.append(eid)
    return matches


def _pick_known_entity(known_entity_ids, domains=None, keywords=None):
    """Pick best entity from known IDs to avoid generic placeholders."""
    domains = domains or []
    keywords = [k.lower() for k in (keywords or [])]
    candidates = []
    for eid in known_entity_ids:
        if "." not in eid:
            continue
        domain, _obj = eid.split(".", 1)
        if domains and domain not in domains:
            continue
        lower = eid.lower()
        score = 0
        score += sum(4 for kw in keywords if kw in lower)
        if "living" in lower or "hall" in lower or "main" in lower:
            score += 1
        candidates.append((score, eid.lower()))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][1]


def _parse_suggestion(suggestion, known_entity_ids):
    """Parse a suggestion into a structured intent dict.

    Prefers structured fields from suggestion dicts (title/description/entities/yaml)
    so gap analysis can reuse concrete automations instead of generic templates.
    """
    if isinstance(suggestion, dict):
        title = (suggestion.get("title", "") or "").strip()
        raw = (suggestion.get("description", "") or title).strip()
        sug_id = suggestion.get("id", "")
        sug_yaml = (suggestion.get("yaml", "") or "").strip()
        explicit_entities = [
            str(e).strip().lower()
            for e in (suggestion.get("entities") or [])
            if isinstance(e, str) and "." in e
        ]
    else:
        raw = str(suggestion).strip()
        title = ""
        sug_id = ""
        sug_yaml = ""
        explicit_entities = []

    intent_pat = _match_intent((title + "\n" + raw).strip())
    entities = _extract_entities_from_text((title + " " + raw).strip(), known_entity_ids)

    # Prefer explicit entities from structured suggestions when available.
    if explicit_entities:
        entities = explicit_entities

    if intent_pat and intent_pat["entity_domains"] and not explicit_entities:
        entities = [e for e in entities if e.split(".")[0] in intent_pat["entity_domains"]]

    if not entities and intent_pat and intent_pat.get("entity_domains"):
        inferred = _pick_known_entity(
            known_entity_ids,
            domains=intent_pat["entity_domains"],
            keywords=[intent_pat["intent"], title, raw],
        )
        if inferred:
            entities = [inferred]

    if not entities:
        fallback = _pick_known_entity(
            known_entity_ids,
            domains=["light", "switch", "climate", "media_player", "fan", "lock", "input_boolean"],
            keywords=[title, raw],
        )
        if fallback:
            entities = [fallback]

    # Deduplicate while preserving order.
    entities = list(dict.fromkeys(entities))

    semantics = _extract_yaml_semantics(sug_yaml)

    return {
        "id": sug_id,
        "title": title,
        "display": title or raw,
        "raw": raw,
        "yaml": sug_yaml,
        "intent": intent_pat["intent"] if intent_pat else "unknown",
        "trigger_type": intent_pat["trigger_type"] if intent_pat else None,
        "action_type": intent_pat["action_type"] if intent_pat else None,
        "entities": entities,
        "semantics": semantics,
    }


def _fetch_automations(ha_url, ha_token):
    """Fetch automation details from HA. Falls back to /api/states if config API 404s."""
    import urllib.error
    import urllib.request

    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}

    def _get(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    try:
        data = _get(f"{ha_url}/api/config/automation/config")
        if isinstance(data, list):
            log.info("automation_gap: fetched %d automations from config API", len(data))
            result = []
            for a in data:
                alias = a.get("alias", "") or a.get("id", "")
                result.append(
                    {
                        "entity_id": "automation.{}".format(alias.lower().replace(" ", "_")),
                        "alias": alias,
                        "trigger": a.get("trigger", []),
                        "action": a.get("action", []),
                        "state": "off" if a.get("mode") == "disabled" else "on",
                        "last_triggered": None,
                    }
                )
            return result
    except Exception as e:
        log.debug("automation_gap: config API failed (%s), falling back to states", e)

    try:
        states = _get(f"{ha_url}/api/states")
        automations = [s for s in states if s["entity_id"].startswith("automation.")]
        log.info("automation_gap: fetched %d automations from states API", len(automations))
        result = []
        for a in automations:
            attrs = a.get("attributes", {})
            result.append(
                {
                    "entity_id": a["entity_id"],
                    "alias": attrs.get("friendly_name", a["entity_id"]),
                    "trigger": [],
                    "action": [],
                    "state": a.get("state", "on"),
                    "last_triggered": attrs.get("last_triggered"),
                }
            )
        return result
    except Exception as e:
        log.warning("automation_gap: states API failed: %s", e)
        return []


def _fetch_all_states(ha_url, ha_token):
    """Fetch all HA states to get entity IDs."""
    import urllib.request

    try:
        req = urllib.request.Request(
            f"{ha_url}/api/states",
            headers={"Authorization": f"Bearer {ha_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.warning("automation_gap: could not fetch states: %s", e)
        return []


def _extract_entity_refs(obj, refs=None):
    """Recursively extract entity_id strings from trigger/action dicts."""
    if refs is None:
        refs = set()
    if isinstance(obj, str) and "." in obj:
        refs.add(obj.lower())
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_entity_refs(v, refs)
    elif isinstance(obj, list):
        for item in obj:
            _extract_entity_refs(item, refs)
    return refs


def _normalize_id(text: str) -> str:
    norm = (text or "").strip().lower()
    norm = norm.replace("automation.", "", 1)
    norm = re.sub(r"[^a-z0-9_]+", "_", norm)
    norm = re.sub(r"_+", "_", norm)
    return norm.strip("_")


def _keyword_overlap(auto_alias, intent):
    """Score keyword overlap between automation alias and intent name (0-40)."""
    alias_words = set(re.split(r"[\W_]+", auto_alias.lower()))
    intent_words = set(re.split(r"_+", intent.lower()))
    common = alias_words & intent_words
    return min(40, len(common) * 15)


def _extract_yaml_semantics(sug_yaml: str) -> dict[str, set[str]]:
    if not sug_yaml:
        return {"trigger_entities": set(), "action_entities": set(), "services": set(), "platforms": set()}

    try:
        parsed = yaml.safe_load(sug_yaml)
    except Exception:
        return {"trigger_entities": set(), "action_entities": set(), "services": set(), "platforms": set()}

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {"trigger_entities": set(), "action_entities": set(), "services": set(), "platforms": set()}

    auto = parsed.get("automation", parsed)
    if isinstance(auto, list) and auto:
        auto = auto[0]
    if not isinstance(auto, dict):
        return {"trigger_entities": set(), "action_entities": set(), "services": set(), "platforms": set()}

    trigger = auto.get("trigger", [])
    action = auto.get("action", [])
    if isinstance(trigger, dict):
        trigger = [trigger]
    if isinstance(action, dict):
        action = [action]

    trigger_entities = _extract_entity_refs(trigger)
    action_entities = _extract_entity_refs(action)
    services = {
        str(a.get("service", "")).lower()
        for a in action
        if isinstance(a, dict) and a.get("service")
    }
    platforms = {
        str(t.get("platform", "")).lower()
        for t in trigger
        if isinstance(t, dict) and t.get("platform")
    }
    return {
        "trigger_entities": trigger_entities,
        "action_entities": action_entities,
        "services": services,
        "platforms": platforms,
    }


def _match_automation(parsed_sug, automations):
    """Find the best matching automation. Returns (automation, score 0-100)."""
    best = None
    best_score = 0
    sug_entities = {e.lower() for e in parsed_sug["entities"]}
    semantics = parsed_sug.get("semantics", {})
    sug_trigger_entities = set(semantics.get("trigger_entities", set()))
    sug_action_entities = set(semantics.get("action_entities", set()))
    sug_services = set(semantics.get("services", set()))
    sug_platforms = set(semantics.get("platforms", set()))
    sug_alias_norm = _normalize_id(parsed_sug.get("title") or parsed_sug.get("display") or "")

    for auto in automations:
        score = 0
        auto_trigger = auto.get("trigger", [])
        auto_action = auto.get("action", [])
        auto_trigger_entities = _extract_entity_refs(auto_trigger)
        auto_action_entities = _extract_entity_refs(auto_action)
        auto_entities = auto_trigger_entities | auto_action_entities

        if sug_entities and auto_entities:
            overlap = sug_entities & auto_entities
            if overlap:
                score += min(55, len(overlap) * 22)

        if sug_trigger_entities and auto_trigger_entities:
            trigger_overlap = sug_trigger_entities & auto_trigger_entities
            if trigger_overlap:
                score += min(20, len(trigger_overlap) * 10)

        if sug_action_entities and auto_action_entities:
            action_overlap = sug_action_entities & auto_action_entities
            if action_overlap:
                score += min(20, len(action_overlap) * 10)

        auto_services = {
            str(a.get("service", "")).lower()
            for a in auto_action
            if isinstance(a, dict) and a.get("service")
        }
        if sug_services and auto_services:
            service_overlap = sug_services & auto_services
            if service_overlap:
                score += min(15, len(service_overlap) * 8)

        auto_platforms = {
            str(t.get("platform", "")).lower()
            for t in auto_trigger
            if isinstance(t, dict) and t.get("platform")
        }
        if sug_platforms and auto_platforms:
            platform_overlap = sug_platforms & auto_platforms
            if platform_overlap:
                score += min(10, len(platform_overlap) * 5)

        alias = auto.get("alias", auto.get("entity_id", ""))
        if parsed_sug["intent"] != "unknown":
            score += _keyword_overlap(alias, parsed_sug["intent"])

        auto_alias_norm = _normalize_id(alias)
        auto_entity_norm = _normalize_id(auto.get("entity_id", ""))
        if sug_alias_norm and sug_alias_norm in {auto_alias_norm, auto_entity_norm}:
            score += 30

        auto_action_text = json.dumps(auto_action).lower()
        if parsed_sug["action_type"] and parsed_sug["action_type"] in auto_action_text:
            score += 15

        if score > best_score:
            best_score = min(score, 100)
            best = auto

    return best, best_score


def _generate_yaml(parsed_sug):
    """Generate ready-to-use HA automation YAML for a missing suggestion."""
    intent = parsed_sug["intent"]
    entities = parsed_sug["entities"]
    raw = parsed_sug["raw"]

    light_entity = next((e for e in entities if e.startswith("light.")), None) or "light.your_light"
    switch_entity = next((e for e in entities if e.startswith("switch.")), None) or "switch.your_switch"
    climate_entity = next((e for e in entities if e.startswith("climate.")), None) or "climate.your_thermostat"
    lock_entity = next((e for e in entities if e.startswith("lock.")), None) or "lock.your_lock"

    alias = raw[:60].strip().rstrip(".")

    if intent == "lights_off_empty_room":
        motion_sensor = "binary_sensor.{}_motion".format(
            light_entity.split(".")[-1] if "." in light_entity else "room"
        )
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: state\n"
            f"    entity_id: {motion_sensor}\n"
            '    to: "off"\n'
            '    for: "00:15:00"\n'
            "condition: []\n"
            "action:\n"
            "  - service: light.turn_off\n"
            "    target:\n"
            f"      entity_id: {light_entity}\n"
            "mode: single"
        )

    elif intent == "reduce_standby":
        standby_target = (
            next((e for e in entities if e.startswith("switch.")), None)
            or next((e for e in entities if e.startswith("media_player.")), None)
            or switch_entity
        )
        service = "media_player.turn_off" if standby_target.startswith("media_player.") else "switch.turn_off"
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: time\n"
            '    at: "23:00:00"\n'
            "condition: []\n"
            "action:\n"
            f"  - service: {service}\n"
            "    target:\n"
            f"      entity_id: {standby_target}\n"
            "mode: single"
        )

    elif intent == "predictive_heating":
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: time\n"
            '    at: "06:30:00"\n'
            "condition:\n"
            "  - condition: time\n"
            "    weekday:\n"
            "      - mon\n"
            "      - tue\n"
            "      - wed\n"
            "      - thu\n"
            "      - fri\n"
            "action:\n"
            "  - service: climate.set_temperature\n"
            "    target:\n"
            f"      entity_id: {climate_entity}\n"
            "    data:\n"
            "      temperature: 21\n"
            "mode: single"
        )

    elif intent == "security_lock":
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: state\n"
            "    entity_id: binary_sensor.occupancy\n"
            '    to: "off"\n'
            '    for: "00:05:00"\n'
            "condition: []\n"
            "action:\n"
            "  - service: lock.lock\n"
            "    target:\n"
            f"      entity_id: {lock_entity}\n"
            "mode: single"
        )

    elif intent == "lights_on_motion":
        motion_sensor = "binary_sensor.{}_motion".format(
            light_entity.split(".")[-1] if "." in light_entity else "room"
        )
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: state\n"
            f"    entity_id: {motion_sensor}\n"
            '    to: "on"\n'
            "condition: []\n"
            "action:\n"
            "  - service: light.turn_on\n"
            "    target:\n"
            f"      entity_id: {light_entity}\n"
            "mode: single"
        )

    elif intent == "media_off_idle":
        media_entity = entities[0] if entities else "media_player.your_tv"
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: state\n"
            f"    entity_id: {media_entity}\n"
            '    to: "idle"\n'
            '    for: "00:30:00"\n'
            "condition: []\n"
            "action:\n"
            "  - service: media_player.turn_off\n"
            "    target:\n"
            f"      entity_id: {media_entity}\n"
            "mode: single"
        )

    else:
        target = entities[0] if entities else "switch.your_device"
        service = "homeassistant.toggle"
        if target.startswith("light."):
            service = "light.toggle"
        elif target.startswith("switch."):
            service = "switch.toggle"
        elif target.startswith("media_player."):
            service = "media_player.turn_off"
        elif target.startswith("climate."):
            service = "climate.turn_off"
        return (
            f'alias: "{alias}"\n'
            "trigger:\n"
            "  - platform: state\n"
            f"    entity_id: {target}\n"
            "condition: []\n"
            "action:\n"
            f"  - service: {service}\n"
            "    target:\n"
            f"      entity_id: {target}\n"
            "mode: single"
        )


def _improvement_hint(auto, auto_score):
    """Generate a human-readable hint for why an automation is being overridden."""
    override_rate = auto_score.get("override_rate", 0) if auto_score else 0
    hint_parts = [f"Override rate {override_rate}% — "]

    trigger_str = json.dumps(auto.get("trigger", [])).lower()
    if "time" in trigger_str or '"at"' in trigger_str:
        hint_parts.append(
            "trigger may be firing at wrong times — consider adding a time condition "
            "or adjusting the schedule."
        )
    elif "state" in trigger_str:
        hint_parts.append(
            "action may be wrong — check what the user does manually after the automation fires."
        )
    else:
        hint_parts.append(
            "check trigger conditions and make sure they match actual usage patterns."
        )

    return "".join(hint_parts)


async def analyse(ha_url, ha_token, suggestions, auto_scores=None):
    """Run automation gap analysis.

    Args:
        ha_url: Home Assistant URL.
        ha_token: HA long-lived access token.
        suggestions: List of suggestion dicts (or strings) from Habitus.
        auto_scores: Optional automation scores from automation_score.py.

    Returns:
        Gap analysis dict with 'gaps' list and 'summary'.
    """
    auto_scores = auto_scores or []

    try:
        automations = _fetch_automations(ha_url, ha_token)
    except Exception as e:
        log.warning("automation_gap: failed to fetch automations: %s", e)
        automations = []

    try:
        all_states = _fetch_all_states(ha_url, ha_token)
        known_entity_ids = [s["entity_id"] for s in all_states]
    except Exception as e:
        log.warning("automation_gap: failed to fetch entity IDs: %s", e)
        known_entity_ids = []

    score_by_id = {s["entity_id"]: s for s in auto_scores}
    gaps = []
    counts = {"missing": 0, "exists_working": 0, "exists_poor": 0, "exists_disabled": 0}

    for sug in suggestions:
        parsed = _parse_suggestion(sug, known_entity_ids)

        # Keep unknown-intent suggestions if they include a concrete YAML snippet.
        if parsed["intent"] == "unknown" and not parsed.get("yaml"):
            continue

        best_auto, match_score = _match_automation(parsed, automations)

        gap = {
            "suggestion": parsed.get("display") or parsed["raw"],
            "title": parsed.get("title", ""),
            "intent": parsed["intent"],
            "entities": parsed["entities"],
        }

        if best_auto is None or match_score < 25:
            gap["status"] = "missing"
            gap["opportunity"] = True
            # Prefer concrete YAML from the suggestion itself when present.
            gap["ha_automation_yaml"] = parsed.get("yaml") or _generate_yaml(parsed)
            counts["missing"] += 1
        else:
            auto_eid = best_auto.get("entity_id", "")
            auto_score_data = score_by_id.get(auto_eid)
            override_rate = auto_score_data.get("override_rate", 0) if auto_score_data else 0
            auto_state = best_auto.get("state", "on")

            gap["matched_automation"] = auto_eid
            gap["match_score"] = match_score
            gap["match_quality"] = (
                "strong" if match_score >= 70 else "moderate" if match_score >= 45 else "weak"
            )

            if auto_state == "off":
                gap["status"] = "exists_disabled"
                gap["improvement"] = (
                    "Automation '{}' exists but is disabled. Consider enabling it.".format(
                        best_auto.get("alias", auto_eid)
                    )
                )
                counts["exists_disabled"] += 1
            elif override_rate > 40:
                gap["status"] = "exists_poor"
                gap["override_rate"] = override_rate / 100
                gap["improvement"] = _improvement_hint(best_auto, auto_score_data)
                counts["exists_poor"] += 1
            else:
                gap["status"] = "exists_working"
                gap["opportunity"] = False
                if override_rate:
                    gap["override_rate"] = override_rate / 100
                counts["exists_working"] += 1

        gaps.append(gap)

    summary_parts = []
    if counts["missing"]:
        summary_parts.append(
            "{} missing automation{}".format(
                counts["missing"], "s" if counts["missing"] != 1 else ""
            )
        )
    if counts["exists_poor"]:
        summary_parts.append("{} improvable".format(counts["exists_poor"]))
    if counts["exists_disabled"]:
        summary_parts.append("{} disabled".format(counts["exists_disabled"]))
    if counts["exists_working"]:
        summary_parts.append("{} working well".format(counts["exists_working"]))

    summary = ", ".join(summary_parts) if summary_parts else "No actionable suggestions found"

    # Enrich gaps with cost estimates for energy-consuming entities
    try:
        from . import cost_estimator as _ce
        gaps = _ce.enrich_with_cost(gaps, entity_field="entities", default_hours=1.0)
        # Also try primary entity from entities list
        for gap in gaps:
            if "cost_estimate" not in gap:
                ents = gap.get("entities", [])
                if ents:
                    enriched = _ce.enrich_with_cost(
                        [{"entity_id": ents[0]}],
                        entity_field="entity_id",
                        default_hours=1.0,
                    )
                    if enriched and "cost_estimate" in enriched[0]:
                        gap["cost_estimate"] = enriched[0]["cost_estimate"]
    except Exception as e:
        log.warning("Cost enrichment for gaps failed: %s", e)

    return {
        "analysed_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        "gaps": gaps,
        "summary": summary,
    }


def save(result):
    """Save gap analysis to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GAP_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(
        "automation_gap: saved %d gaps — %s",
        len(result.get("gaps", [])),
        result.get("summary", ""),
    )


def load():
    """Load gap analysis from disk."""
    if not os.path.exists(GAP_PATH):
        return {}
    try:
        with open(GAP_PATH) as f:
            return json.load(f)
    except Exception:
        return {}
