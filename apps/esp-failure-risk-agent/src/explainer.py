"""Turn model feature contributions into plain-English explanations via Claude.

Design principle (shared with the Daily Production Digest): **detection stays
deterministic, the LLM only narrates.** A rule-based classifier maps the engineered
features to a suspected ESP failure mode FIRST; the LLM is then asked to write the
rationale *for that mode*, so it can't hallucinate a diagnosis the data doesn't
support.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv


class MissingAPIKey(RuntimeError):
    """Raised when an LLM call is attempted without ANTHROPIC_API_KEY set."""


EXPLAINER_SYSTEM_PROMPT = """You are a Senior Production Engineer writing the rationale section of an ESP failure-risk digest. You are given a well's diagnostics, the top features driving its risk score, and a SUSPECTED FAILURE MODE already determined by a deterministic classifier. Write 2-3 sentences (max 60 words) explaining WHY this well is high-risk in the language a field engineer uses.

Style rules:
- Lead with the suspected failure mode you were given (do NOT contradict it; you may add nuance).
- Reference the specific values that triggered the call (e.g., "intake pressure at 28 psi, declining").
- End with a concrete next step (chemical/scale treatment, ESP pull, VSD/frequency adjustment, gas separator, motor/cable megger test).
- No filler, no hedging, no "based on the model".
"""


# ---- deterministic failure-mode classifier --------------------------------

def classify_failure_mode(
    feature_values: dict[str, float], lift: str | None = None
) -> tuple[str, str]:
    """Map an engineered feature row to a suspected failure mode + evidence.

    Returns (mode_label, evidence). Rules are ordered by specificity; the first
    match wins. This is intentionally simple and auditable — it grounds the LLM
    narration and is shown to the user alongside the risk score.

    Detection is deterministic and lift-agnostic (the rules read generic,
    SCADA-derived features). ``lift`` only specializes the *wording*: an ESP — or
    unknown/None — lift reproduces the original ESP phrasing byte-for-byte, while a
    rod-pump / gas-lift / flowing well gets physically correct terminology, since a
    well with no ESP has no intake, motor, VSD, or pump stages to talk about.
    """
    f = feature_values
    g = lambda k, d=0.0: float(f.get(k, d))

    imb_max = g("current_imbalance_max_30d", 3.0)
    imb_days = g("high_imbalance_days_30d")
    bfpd_cv = g("bfpd_cv_30d")
    downtime = g("downtime_days_30d")
    low_intake_days = g("low_intake_days_30d")
    intake_slope = g("intake_p_slope_30d")
    intake_mean = g("intake_p_last7_mean", 130.0)
    amps_slope = g("motor_amps_slope_30d")
    temp_slope = g("motor_temp_slope_30d")
    bfpd_slope = g("bfpd_slope_30d")
    runtime_mean = g("runtime_last7_mean", 99.0)
    freq_slope = g("drive_freq_slope_30d")
    v = dict(imb_max=imb_max, imb_days=imb_days, bfpd_cv=bfpd_cv, downtime=downtime,
             intake_mean=intake_mean, intake_slope=intake_slope, amps_slope=amps_slope,
             temp_slope=temp_slope, bfpd_slope=bfpd_slope, runtime_mean=runtime_mean,
             freq_slope=freq_slope)

    # 1) Electrical / current imbalance — the imbalance channel is the tell.
    if imb_max >= 9 or imb_days >= 2:
        return _phrase_mode("electrical", lift, v)
    # 2) Gas lock / pump-off cycling — flow volatility + runtime cycling, freq rising.
    if bfpd_cv >= 0.15 and downtime >= 3:
        return _phrase_mode("gas_lock", lift, v)
    # 3) Gas interference — smooth intake collapse.
    if low_intake_days >= 2 or (intake_slope <= -1.0 and intake_mean < 90):
        return _phrase_mode("gas_interference", lift, v)
    # 4) Scale / abrasive buildup — amps & temp creeping together, intake holding.
    if amps_slope >= 0.15 and temp_slope >= 0.10:
        return _phrase_mode("scale", lift, v)
    # 5) Downthrust / reservoir decline — rate slumping, runtime down, amps not rising.
    if bfpd_slope <= -8 and runtime_mean < 96 and amps_slope < 0.15:
        return _phrase_mode("downthrust", lift, v)
    return _phrase_mode("unclear", lift, v)


def _phrase_mode(cond: str, lift: str | None, v: dict[str, float]) -> tuple[str, str]:
    """Render (mode_label, evidence) for a detected condition in the language of the
    well's artificial-lift type. ESP / unknown / None lift reproduces the original ESP
    wording exactly; rod-pump, gas-lift, and flowing wells get lift-correct terms and
    next steps (no megger, no VSD frequency, no pump POR / stage count where there is
    no ESP)."""
    esp = lift not in ("Rod pump", "Gas lift", "Flowing")

    if cond == "electrical":
        if esp:
            return ("Electrical — current imbalance / incipient motor short",
                    f"current imbalance peaked at {v['imb_max']:.0f}% ({int(v['imb_days'])} day(s) >8%); "
                    f"megger the motor and cable before the next start.")
        if lift == "Rod pump":
            return ("Electrical — surface motor / drive imbalance",
                    f"current imbalance peaked at {v['imb_max']:.0f}% ({int(v['imb_days'])} day(s) >8%); "
                    "inspect the surface motor, drive, and electrical service.")
        return (f"Electrical signal anomaly — no motor on a {lift.lower()} well",
                f"current-imbalance channel peaked at {v['imb_max']:.0f}% "
                f"({int(v['imb_days'])} day(s) >8%), but this lift has no motor; "
                "verify the SCADA instrumentation.")

    if cond == "gas_lock":
        if esp:
            return ("Gas lock — pump-off cycling",
                    f"production volatile (CV {v['bfpd_cv']:.0%}) with {int(v['downtime'])} low-runtime day(s)"
                    + (f" and drive frequency climbing {v['freq_slope']:+.2f} Hz/d" if v['freq_slope'] > 0.05 else "")
                    + "; review gas handling / install a separator and check VSD pump-off setpoints.")
        if lift == "Rod pump":
            return ("Gas lock / pump pounding",
                    f"production volatile (CV {v['bfpd_cv']:.0%}) with {int(v['downtime'])} low-runtime day(s); "
                    "improve pump fillage (SPM / pump-off controller) and set a gas anchor.")
        if lift == "Gas lift":
            return ("Unstable lift — heading / slugging",
                    f"production volatile (CV {v['bfpd_cv']:.0%}) with {int(v['downtime'])} low-flow day(s); "
                    "stabilize lift-gas injection (rate / valve depth) to damp heading.")
        return ("Slugging / liquid loading",
                f"production volatile (CV {v['bfpd_cv']:.0%}) with {int(v['downtime'])} low-flow day(s); "
                "evaluate a velocity string or artificial-lift conversion.")

    if cond == "gas_interference":
        if esp:
            return ("Gas interference — intake pressure collapse",
                    f"intake at {v['intake_mean']:.0f} psi, trending {v['intake_slope']:+.1f} psi/d; "
                    f"reduce drawdown / adjust frequency and evaluate a gas separator.")
        if lift == "Rod pump":
            return ("Gas interference / fluid pound",
                    f"pump intake at {v['intake_mean']:.0f} psi, trending {v['intake_slope']:+.1f} psi/d; "
                    "set a gas anchor and adjust pump fillage.")
        if lift == "Gas lift":
            return ("Gas interference — unstable lift",
                    f"flowing pressure at {v['intake_mean']:.0f} psi, trending {v['intake_slope']:+.1f} psi/d; "
                    "review lift-gas injection rate and valve performance.")
        return ("Pressure decline / liquid loading",
                f"flowing bottomhole pressure at {v['intake_mean']:.0f} psi, trending "
                f"{v['intake_slope']:+.1f} psi/d; evaluate stimulation or an artificial-lift conversion.")

    if cond == "scale":
        if esp:
            return ("Scale / abrasive buildup",
                    f"motor amps creeping {v['amps_slope']:+.2f} A/d and temperature {v['temp_slope']:+.2f} °F/d "
                    f"with stable intake; schedule a scale-inhibitor squeeze / acid treatment.")
        if lift == "Rod pump":
            return ("Scale / abrasive buildup",
                    f"motor load creeping {v['amps_slope']:+.2f} A/d and temperature {v['temp_slope']:+.2f} °F/d; "
                    "schedule a scale-inhibitor squeeze / acid treatment.")
        return ("Scale / tubing restriction",
                f"surface signature creeping up ({v['amps_slope']:+.2f}/d) with rising temperature "
                f"({v['temp_slope']:+.2f} °F/d); schedule a scale-inhibitor squeeze / acid treatment.")

    if cond == "downthrust":
        if esp:
            return ("Downthrust / declining inflow",
                    f"production sliding {v['bfpd_slope']:+.0f} bbl/d with runtime at {v['runtime_mean']:.0f}%; "
                    f"verify pump is within POR and consider a re-rate or smaller stage count.")
        if lift == "Rod pump":
            return ("Declining inflow / pump underload",
                    f"production sliding {v['bfpd_slope']:+.0f} bbl/d with runtime at {v['runtime_mean']:.0f}%; "
                    "review the dynamometer card / pump fillage and consider a re-rate or downsize.")
        if lift == "Gas lift":
            return ("Declining inflow / reservoir decline",
                    f"production sliding {v['bfpd_slope']:+.0f} bbl/d; optimize lift-gas "
                    "(injection rate / valve depth) and confirm reservoir decline.")
        return ("Declining inflow / reservoir decline",
                f"production sliding {v['bfpd_slope']:+.0f} bbl/d; confirm reservoir decline and "
                "evaluate stimulation or an artificial-lift conversion.")

    return ("Unclear — multiple weak signals",
            "no single dominant signature; review the trend plot and recent well work before acting.")


def explain_well(
    well_id: str,
    risk_score: float,
    feature_values: dict[str, float],
    top_drivers: list[tuple[str, float]],
    suspected_mode: str | None = None,
    model: str = "claude-sonnet-4-6",
    client=None,
) -> str:
    """Generate a plain-English rationale for a single high-risk well.

    Raises ``MissingAPIKey`` (not a bare KeyError) when no client is supplied and
    ANTHROPIC_API_KEY is unset, so callers can degrade gracefully.
    """
    if suspected_mode is None:
        suspected_mode = classify_failure_mode(feature_values)[0]

    if client is None:
        load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKey(
                "ANTHROPIC_API_KEY is not set — set it to generate AI rationales. "
                "The risk scores, drivers, and suspected failure mode work without it.")
        from anthropic import Anthropic
        client = Anthropic(api_key=key)

    drivers_str = "\n".join(
        f"  - {feat}: contribution={contrib:+.2f}, current_value={feature_values.get(feat, 'n/a')}"
        for feat, contrib in top_drivers
    )

    prompt = f"""Well: {well_id}
30-day failure probability: {risk_score:.1%}
Suspected failure mode (deterministic classifier): {suspected_mode}

Top features driving this risk (XGBoost Tree SHAP contributions, positive = increases risk):
{drivers_str}

Write the 2-3 sentence rationale."""

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=EXPLAINER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def top_drivers(contribs_row, k: int = 4) -> list[tuple[str, float]]:
    """Pick the top-k features by absolute contribution (excluding bias)."""
    s = contribs_row.drop("bias")
    return list(s.abs().sort_values(ascending=False).head(k).items())
