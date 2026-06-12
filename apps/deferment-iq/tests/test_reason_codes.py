"""Tests for the reason-code taxonomy + classifier."""
from src.reason_codes import classify, classify_rules, is_planned, is_recoverable


def test_rules_classifier_hits_each_cause():
    cases = {
        "ESP tripped on underload, VSD fault F012": "artificial_lift",
        "rod string parted, pulling unit scheduled Tuesday": "artificial_lift",
        "compressor down, no gas takeaway off the pad": "surface_facility",
        "lost power to the pad, main breaker tripped": "power",
        "high line pressure from gathering, curtailed": "gathering_thirdparty",
        "scale buildup restricting flow": "wellbore",
        "scheduled ESP workover": "planned",
        "winter storm freeze-off across the pad": "weather",
        "water cut climbing, well watering out": "reservoir",
    }
    for note, expected in cases.items():
        assert classify_rules(note)[0] == expected, note


def test_vague_note_is_unclassified():
    assert classify_rules("well down, see foreman")[0] == "unclassified"


def test_classify_falls_back_to_rules_without_client():
    # use_llm=True but no client -> must not crash, returns the rules answer
    assert classify("compressor down", use_llm=True, client=None) == "surface_facility"


def test_recoverable_and_planned_flags():
    assert is_recoverable("artificial_lift") and not is_planned("artificial_lift")
    assert is_planned("planned") and not is_recoverable("planned")
    assert not is_recoverable("reservoir")          # watered-out barrels aren't recoverable
    assert not is_recoverable("unclassified")
