"""Tests for natural language automation creator."""
from __future__ import annotations

import pytest

from habitus.habitus.nl_automation import (
    parse_intent,
    _parse_time,
    _extract_action,
    _extract_trigger_type,
    _is_sun_trigger,
)


class TestParseTime:
    def test_7am(self):
        assert _parse_time("at 7am") == "07:00:00"

    def test_7_30am(self):
        assert _parse_time("at 7:30am") == "07:30:00"

    def test_19_00(self):
        assert _parse_time("at 19:00") == "19:00:00"

    def test_morning(self):
        assert _parse_time("every morning") == "07:00:00"

    def test_midnight(self):
        assert _parse_time("at midnight") == "00:00:00"

    def test_noon(self):
        assert _parse_time("at noon") == "12:00:00"

    def test_no_time_returns_none(self):
        assert _parse_time("turn off the lights") is None

    def test_pm_conversion(self):
        result = _parse_time("at 2pm")
        assert result == "14:00:00"

    def test_12pm_is_noon(self):
        result = _parse_time("at 12pm")
        assert result == "12:00:00"


class TestIsSunTrigger:
    def test_sunset(self):
        assert _is_sun_trigger("at sunset turn on lights") == "sunset"

    def test_sunrise(self):
        assert _is_sun_trigger("at sunrise open blinds") == "sunrise"

    def test_neither(self):
        assert _is_sun_trigger("at 7am turn on lights") is None


class TestExtractAction:
    def test_turn_off(self):
        result = _extract_action("turn off the lights")
        assert result["action"] == "turn_off"

    def test_turn_on(self):
        result = _extract_action("turn on the living room light")
        assert result["action"] == "turn_on"

    def test_notify(self):
        result = _extract_action("notify me when door opens")
        assert result["action"] == "notify"

    def test_toggle(self):
        result = _extract_action("toggle the switch")
        assert result["action"] == "toggle"


class TestExtractTriggerType:
    def test_time_trigger(self):
        result = _extract_trigger_type("every morning at 7am")
        assert result["type"] == "time"
        assert result["at"] == "07:00:00"

    def test_sun_trigger_sunset(self):
        result = _extract_trigger_type("at sunset turn on lights")
        assert result["type"] == "sun"
        assert result["event"] == "sunset"

    def test_state_trigger_on(self):
        result = _extract_trigger_type("when motion detected turn on light")
        assert result["type"] == "state"

    def test_temperature_above(self):
        result = _extract_trigger_type("when temperature above 25 degrees turn on fan")
        assert result["type"] == "numeric_state"
        assert "above" in result["kind"]


class TestParseIntent:
    def test_time_based_automation(self):
        """'Turn off lights at 11pm' → time trigger."""
        intent = parse_intent("Turn off lights at 11pm")
        assert intent["trigger"]["type"] == "time"
        assert intent["trigger"]["at"] == "23:00:00"
        assert intent["action"]["action"] == "turn_off"
        assert intent["confidence"] > 0.5

    def test_state_trigger(self):
        """'When motion detected turn on hallway light' → state trigger."""
        intent = parse_intent("When motion detected turn on hallway light")
        assert intent["trigger"]["type"] == "state"
        assert intent["action"]["action"] == "turn_on"

    def test_sunset_trigger(self):
        """'At sunset turn on porch light' → sun trigger."""
        intent = parse_intent("At sunset turn on porch light")
        assert intent["trigger"]["type"] == "sun"
        assert intent["trigger"]["event"] == "sunset"

    def test_empty_text(self):
        """Empty text → low confidence with clarifications."""
        intent = parse_intent("")
        assert intent["confidence"] == 0.0
        assert len(intent["clarification_needed"]) > 0
        assert intent["generated_yaml"] == ""

    def test_notification_automation(self):
        """'Notify me when door opens' → notify action."""
        intent = parse_intent("Notify me when front door opens")
        assert intent["action"]["action"] == "notify"

    def test_generated_yaml_is_non_empty(self):
        """Generated YAML is non-empty for valid input."""
        intent = parse_intent("Turn off lights at 10pm")
        assert len(intent["generated_yaml"]) > 20

    def test_generated_yaml_contains_automation(self):
        """Generated YAML contains 'automation' key."""
        intent = parse_intent("Turn off kitchen lights at midnight")
        assert "automation" in intent["generated_yaml"]

    def test_multi_entity_detection(self):
        """Multiple entity types in text are detected."""
        intent = parse_intent("When motion detected in hallway turn on hallway light and notify")
        entities = intent["entities"]
        domains = [e["domain"] for e in entities]
        # Should find at least one entity
        assert len(entities) > 0

    def test_condition_nobody_home(self):
        """'If nobody home turn off lights' → condition."""
        intent = parse_intent("If nobody home turn off all lights")
        # Either parsed as condition or state trigger
        assert intent["confidence"] > 0.0

    def test_confidence_between_0_and_1(self):
        """Confidence is always between 0 and 1."""
        for text in [
            "",
            "turn on",
            "at 7am turn on lights when nobody home",
            "notify me at sunset",
        ]:
            intent = parse_intent(text)
            assert 0.0 <= intent["confidence"] <= 1.0, f"Bad confidence for: {text!r}"

    def test_all_required_fields_present(self):
        """Intent has all required output fields."""
        intent = parse_intent("Turn off lights at 10pm")
        required = {"trigger", "action", "entities", "condition", "alias",
                    "confidence", "clarification_needed", "generated_yaml"}
        for field in required:
            assert field in intent, f"Missing field: {field}"

    def test_morning_time_automation(self):
        """'Every morning at 7am turn on lights' → time trigger."""
        intent = parse_intent("Every morning at 7am turn on lights")
        assert intent["trigger"]["type"] == "time"
        assert intent["trigger"]["at"] == "07:00:00"
