"""Deferment reason-code taxonomy + classifier.

Every barrel of deferred (lost) production gets a cause. Operators carry a fixed
reason-code taxonomy and tag each downtime/curtailment event — usually from the
free-text note a pumper/operator typed. This module is the *deterministic* core:
a keyword/rules classifier over that note. An optional LLM classifier (BYOK) can
handle the messy long-tail, but it always falls back to the rules so the app — and
the committed eval — run with no API key.

Design (same as the rest of the suite): the engineering/accounting math is
deterministic and trusted; the LLM only assists on ambiguous text and narrates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonCode:
    key: str
    label: str
    recoverable: bool   # is the lost production recoverable by operator action?
    planned: bool       # planned (expected) vs unplanned loss
    keywords: tuple[str, ...]


# Canonical taxonomy. `recoverable` drives the "recovery opportunity" $ (planned and
# reservoir losses are NOT opportunity — you can't get watered-out or scheduled barrels
# back). Order matters only for display; classification scores all and takes the best.
REASON_CODES: tuple[ReasonCode, ...] = (
    ReasonCode("artificial_lift", "Artificial lift", True, False, (
        "esp", "vsd", "rod pump", "rod string", "rod", "parted", "gas lift", "gaslift",
        "plunger", "pump off", "pumped off", "fillage", "poc", "underload", "overload",
        "gearbox", "downhole pump", "no fluid", "pulling unit", "pumping unit",
        "tubing leak", "intake", "pump failure", "lift")),
    ReasonCode("surface_facility", "Surface facility", True, False, (
        "separator", "compressor", "comp down", "tank", "battery full", "lact",
        "dump valve", "heater treater", "treater", "vru", "facility", "slug catcher",
        "emulsion", "high level", "shut in on facility")),
    ReasonCode("power", "Power / electrical", True, False, (
        "power", "electric", "breaker", "substation", "transformer", "outage",
        "grid", "generator", "genset", "lost power", "no power", "lightning strike")),
    ReasonCode("gathering_thirdparty", "Gathering / 3rd-party", True, False, (
        "line pressure", "high line", "gathering", "midstream", "gas plant", "plant down",
        "takeaway", "curtail", "pipeline", "backpressure", "back pressure", "third party",
        "3rd party", "sales line", "nomination")),
    ReasonCode("wellbore", "Wellbore", True, False, (
        "scale", "paraffin", "wax", "sand", "hole in tubing", "fill", "plugged",
        "restricted", "casing", "hydrate", "asphaltene", "wax cut", "screen out")),
    ReasonCode("planned", "Planned work", False, True, (
        "planned", "scheduled", "workover", "maintenance", "well test", "testing",
        "frac", "completion", "wireline", "slickline", "routine", "turnaround",
        "pm ", "inspection", "rig move")),
    ReasonCode("weather", "Weather / freeze", True, False, (
        "freeze", "frozen", "freeze off", "weather", "winter storm", "ice", "flood",
        "hurricane", "cold front", "froze")),
    ReasonCode("reservoir", "Reservoir", False, False, (
        "water cut", "watering out", "watered out", "depletion", "liquid loading",
        "loading up", "loaded up", "gor", "pressure depletion", "declining inflow")),
)

REASON_BY_KEY: dict[str, ReasonCode] = {rc.key: rc for rc in REASON_CODES}
UNCLASSIFIED = ReasonCode("unclassified", "Unclassified / uncaptured", False, False, ())

LLM_SYSTEM_PROMPT = (
    "You classify an oil & gas production downtime/curtailment note into EXACTLY ONE "
    "reason code. Reply with ONLY the code key, nothing else. Valid keys: "
    + ", ".join(rc.key for rc in REASON_CODES) + ", unclassified."
)


def classify_rules(note: str) -> tuple[str, int]:
    """Deterministic keyword classifier. Returns (reason_key, match_score).

    Scores each reason code by how many of its keywords appear in the note; the
    highest score wins. Zero matches -> 'unclassified'. Whole-word-ish matching
    avoids 'comp'-in-'compliance' style false hits.
    """
    text = (note or "").lower()
    best_key, best_score = "unclassified", 0
    for rc in REASON_CODES:
        score = 0
        for kw in rc.keywords:
            # word-boundary match for short single tokens; substring for phrases
            if " " in kw:
                if kw in text:
                    score += 1
            elif re.search(rf"\b{re.escape(kw)}\b", text):
                score += 1
        if score > best_score:
            best_key, best_score = rc.key, score
    return best_key, best_score


def classify(note: str, use_llm: bool = False, client=None,
             model: str = "claude-sonnet-4-6") -> str:
    """Classify a note into a reason key. Rules by default (deterministic, no key).

    With use_llm=True and an Anthropic client, asks the LLM but falls back to the
    rules classifier on any error or an out-of-taxonomy answer — so it never crashes
    and never returns an invalid code.
    """
    rule_key, _ = classify_rules(note)
    if not use_llm or client is None:
        return rule_key
    try:
        resp = client.messages.create(
            model=model, max_tokens=12, system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": (note or "").strip()[:500]}])
        ans = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
        ans = re.sub(r"[^a-z_]", "", ans)
        if ans in REASON_BY_KEY or ans == "unclassified":
            return ans
        return rule_key
    except Exception:
        return rule_key


def label_for(key: str) -> str:
    return REASON_BY_KEY[key].label if key in REASON_BY_KEY else UNCLASSIFIED.label


# Suggested first intervention per recoverable cause — what the field actually does to
# get the barrels back. Drives the work-queue's `suggested_action` and the AFE deep-link.
SUGGESTED_ACTION: dict[str, str] = {
    "artificial_lift": "ESP / rod-pump workover",
    "surface_facility": "Facility repair (separator/compressor/treater)",
    "power": "Restore power / electrical repair",
    "gathering_thirdparty": "Line-pressure / midstream coordination",
    "wellbore": "Wellbore cleanout (scale/paraffin/sand)",
    "weather": "Freeze protection / winterization",
}


def suggested_action(key: str) -> str:
    return SUGGESTED_ACTION.get(key, "Investigate")


def is_recoverable(key: str) -> bool:
    return REASON_BY_KEY[key].recoverable if key in REASON_BY_KEY else False


def is_planned(key: str) -> bool:
    return REASON_BY_KEY[key].planned if key in REASON_BY_KEY else False
