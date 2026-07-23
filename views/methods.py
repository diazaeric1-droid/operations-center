"""Data · Methods & Limitations — the model card.

Consolidates the disclosures that are otherwise scattered as fine-print captions
across the console: the economics conventions, the ESP score's honest limits, the
two-datasets provenance, the lift-aware intervention rule, and the committed
backtest numbers. This is the page a sharp PE / ML-aware reviewer looks for — it
pre-empts the "but is this calibrated?" question in one place.
"""
from __future__ import annotations

import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()

    pt.masthead("ops", "Methods & Limitations",
                "How every number on this console is computed, and exactly where the "
                "synthetic demo data stops standing in for a real asset.")
    pt.context_bar([
        ("Deck", c.deck_label()),
        ("Surveillance fleet", c.scada_source_label(c.DISK_TOKEN)),
        ("Stance", "deterministic math · LLM only narrates · honest about limits"),
    ])

    pt.section("Economics — risked NPV & discounting")
    st.markdown(
        "- **Risked NPV = risk × PV(net revenue the intervention protects) − cost.** "
        "The intervention cost is *certain*, so only the upside is chance-weighted. "
        "This is why a well's Risked NPV is always **≤** the AFE's deterministic Net "
        "NPV — that gap is the failure-risk discount, not a discrepancy.\n"
        "- **Net-to-operator throughout.** Revenue is netted by the deck NRI; the "
        "Morning Brief, the email, and every KPI use the same net convention.\n"
        "- **PV10, fixed.** All NPVs discount at an effective **10%/yr** via the AFE "
        "component's certified kernel (`(1+r)^(m/12)`, not the `(1+r/12)^m` that "
        "silently compounds to 10.47%). The deck's discount control does not re-rate "
        "the certified chain math.\n"
        "- **Monte-Carlo** (Action Chain) draws 10,000 trials over the three biggest "
        "uncertainties (incremental rate, uplift decline, realized price); its **base "
        "case** (mean of the inputs) reconciles exactly with the deterministic Net NPV, "
        "and the P50 sits slightly below it because the NPV distribution is right-skewed.\n"
        "- **Uplift horizon — a stated assumption.** Intervention NPVs book a **5-year** "
        "uplift tail declining at 0.6/yr. That fits a capital workover (ESP swap, rod-pump "
        "workover, acid stim) but is **generous for a short-scope job** such as a gas-lift "
        "optimization (often a 1-day slickline visit). The *relative* ranking is "
        "unaffected (every well uses the same horizon); treat the *absolute* NPV on "
        "optimization-type jobs as an upper bound until the horizon is scoped per job "
        "type. Flagged on the Action Chain where these figures appear.")

    pt.section("ESP failure-risk model card")
    ev = _esp_eval()
    if ev:
        m = st.columns(5)
        m[0].metric("AUROC (calibrated OOF)", f"{ev['auroc_cv_mean']:.3f}",
                    f"± {ev['auroc_cv_std']:.3f}", delta_color="off")
        m[1].metric("Brier (calibrated OOF)", f"{ev['brier']:.3f}", "lower is better",
                    delta_color="off")
        m[2].metric("Precision @ top 10%", f"{ev['precision_at_top10pct']:.0%}",
                    f"top {ev['n_flagged_top10pct']} wells", delta_color="off")
        m[3].metric("Recall @ top 10%", f"{ev['recall_at_top10pct']:.0%}",
                    f"of {ev['n_positives']} impaired", delta_color="off")
        m[4].metric("Calibrated", "yes" if ev["calibrated"] else "no",
                    f"{ev['n_positives']}/{ev['n_wells']} impaired", delta_color="off")
        st.caption(
            "**Precision and recall are a pair — read them together.** Precision@10% is "
            f"near-100% because on a fleet where {ev['n_positives']}/{ev['n_wells']} wells "
            "are impaired, the very top of the ranking is almost all true positives — not "
            "an impressive number on its own. Recall@10% is structurally capped: flagging "
            f"only the top {ev['n_flagged_top10pct']} wells can recover at most "
            f"~{ev['n_flagged_top10pct']}/{ev['n_positives']} of the impaired fleet; at the "
            "quartile cut the board actually flags on, recall is higher (full panel on the "
            "Optimization Board). **AUROC and Brier are END-TO-END calibrated out-of-fold** — "
            "each CV fold trains the booster AND Platt-calibrates it before scoring the "
            "held-out wells, exactly as the shipped model does — so the Brier describes the "
            "calibrated probabilities the console actually displays, not a raw booster.")
    st.markdown(
        "The 30-day failure score is a **Platt-calibrated probability** from an XGBoost "
        "model **trained on the digest fleet itself** — the fleet the console scores — "
        "using the generator's ground-truth fault labels. (Earlier it was trained on a "
        "*different* fleet and scored this one out-of-distribution, so the console only "
        "trusted its fleet-relative ranking; the model is now calibrated on this fleet.) "
        "The metrics above are **end-to-end out-of-fold**: each cross-validation fold "
        "trains the booster *and* fits the Platt calibrator before scoring the held-out "
        "wells, so they describe the calibrated pipeline the console displays — not a raw "
        "booster scored without calibration (calibrating in-fold trades a little AUROC for "
        "an honest Brier; that lower AUROC is the one to trust).\n\n"
        "**Read the high AUROC honestly:** it is near-perfect because the synthetic fault "
        "signatures are cleanly separable *by design* — an **upper bound on clean data, "
        "not a real-world claim.** On a real operator's messy historian (missing days, "
        "metering noise, comms dropouts) we'd expect a materially lower number, and *that* "
        "drop — not this synthetic figure — would be the real measure of skill. We don't "
        "quote a specific real-world AUROC because we have no labeled operator historian to "
        "measure one on; treat it as an engineering expectation, not a result. If the model "
        "can't load at all, every well falls back to a baseline risk and the Optimization Board / "
        "Home show a visible **degradation banner** rather than a misleading uniform fleet.")

    pt.section("Lift-aware interventions")
    st.markdown(
        "Recommended interventions are gated by the well's **artificial-lift type**, so "
        "the board never proposes a physically-impossible job (no ESP swap on a "
        "rod-pumped well, no gas-lift optimization on a well with no injection). Because "
        "lift-correct jobs are cheaper, an **opportunity** additionally requires a real "
        "trigger — actively deferring production OR the fleet's elevated-risk quartile — "
        "not merely a cheap intervention that happens to pencil. This ranking is "
        "mirrored bit-for-bit against the PE-Pipeline orchestrator (a parity test pins "
        "them identical).\n\n"
        "**Downtime context, not choke data.** Recommendations for wells in an OPEN "
        "event (state machine NEW/ONGOING) carry a visible *'verify post-restart before "
        "acting'* flag on the Optimization Board, Well 360, and the Action Chain. The "
        "SCADA schema carries **no choke-position channel**, so choke moves cannot be "
        "separated from reservoir/lift losses — a stated limitation (disclosed in-page "
        "where recommendations render), not something the console models.")

    pt.section("NRI conventions — deck vs per-well")
    st.markdown(
        "- **Certified chain/ranking economics** (risked NPV, AFE, Monte-Carlo) use the "
        "**sidebar deck NRI** — one flat, auditable number, matching pe-pipeline parity.\n"
        "- **Roll-up NET views** (Morning Brief unified list, Optimization Board "
        "deferred-$ columns, Deferment Overview) can apply **per-well NRI** — a "
        "deterministic, varied registry default (≈0.73–0.85) with session-only "
        "per-well overrides on **Sources & BYOD**. GROSS (8/8) stays the default "
        "base-management convention.\n"
        "- The registry NRI is **synthetic and illustrative** — real division-order "
        "data replaces it in a deployment.")

    pt.section("Two datasets, no fake join")
    n = len(c.scada_well_ids())
    st.markdown(
        f"- **Surveillance fleet** (Today + Well File): a synthetic **daily SCADA** "
        f"fleet of **{n} wells** with known ground truth. Public production data is "
        "monthly, so daily SCADA must be modeled.\n"
        "- **Loss-accounting book** (Loss Accounting): a synthetic reason-coded "
        "**monthly** fleet with ground-truth causes (so cause attribution / MTTR / the "
        "recovery queue all work).\n"
        "- They are **different datasets at different cadences and the console does not "
        "fake a join** between them. Bring your own daily SCADA or monthly book on "
        "**Sources & BYOD**.")

    pt.section("Backtests — the rankings are scored, not asserted")
    sc = _scorecard(price, nri)
    if sc:
        st.markdown(
            f"- **Board ranking:** scored against the fleet's known seeded faults — "
            f"precision@10 **{sc['at_k'][10]['precision']:.0%}** "
            f"({sc['at_k'][10]['lift']:.1f}× lift over random), "
            f"recall@{sc['n_impaired']} **{sc['recall_at_n_impaired']:.0%}**. Honest, "
            "not a trivial 100% — low-rate failure modes defer few barrels and rank "
            "lower. (Full panel on the Optimization Board.)")
    st.markdown(
        "- **Digest event detector:** backtested with near-threshold decoys so "
        "precision/recall aren't trivially 1.0; a real lead-time/latency metric.\n"
        "- **Deferment cause classifier:** ~92% on a ground-truth-labeled eval "
        "(CI gate fails below 80%).")

    pt.section("What's synthetic (and what that means)")
    st.markdown(
        "Operator, API-14, and well **locations** are illustrative; the map places "
        "each well at its (real Permian) county centroid with a deterministic "
        "within-county jitter. Production signatures are seeded and deterministic — a "
        "clean stand-in that exercises every code path, **not** the messiness of a real "
        "operator's historian. The honest next step is real operator data, not a higher "
        "synthetic score. The engineering math, the economics, and the evals are real "
        "and would run unchanged on a real fleet.")

    theme.references(["npv", "arps", "shap", "deferment"])


def _scorecard(price: float, nri: float):
    import core
    try:
        return core.triage_scorecard(c.board_with_deferred(price, nri))
    except Exception:  # noqa: BLE001
        return None


def _esp_eval():
    import core
    try:
        return core.esp_model_eval()
    except Exception:  # noqa: BLE001
        return None
