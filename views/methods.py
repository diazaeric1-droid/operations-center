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
        "uncertainties (incremental rate, uplift decline, realized price); its P50 "
        "reconciles exactly with the deterministic Net NPV.")

    pt.section("ESP failure-risk model card")
    ev = _esp_eval()
    if ev:
        m = st.columns(4)
        m[0].metric("AUROC (out-of-fold)", f"{ev['auroc_cv_mean']:.3f}",
                    f"± {ev['auroc_cv_std']:.3f}", delta_color="off")
        m[1].metric("Brier (OOF)", f"{ev['brier']:.3f}", "lower is better",
                    delta_color="off")
        m[2].metric("Precision @ top 10%", f"{ev['precision_at_top10pct']:.0%}",
                    delta_color="off")
        m[3].metric("Calibrated", "yes" if ev["calibrated"] else "no",
                    f"{ev['n_positives']}/{ev['n_wells']} impaired", delta_color="off")
    st.markdown(
        "The 30-day failure score is a **Platt-calibrated probability** from an XGBoost "
        "model **trained on the digest fleet itself** — the fleet the console scores — "
        "using the generator's ground-truth fault labels. (Earlier it was trained on a "
        "*different* fleet and scored this one out-of-distribution, so the console only "
        "trusted its fleet-relative ranking; the model is now calibrated on this fleet.) "
        "The metrics above are **out-of-fold** (stratified CV), so they measure "
        "generalization, not memorization.\n\n"
        "**Read the high AUROC honestly:** it is near-perfect because the synthetic "
        "fault signatures are cleanly separable *by design* — it is an **upper bound on "
        "clean data, not a real-world claim.** On a real operator's messy historian we "
        "would expect ~0.85 and treat that drop as the real signal. The displayed "
        "per-well score is the calibrated probability (slightly optimistic vs the OOF "
        "metric on the training wells). If the model can't load at all, every well "
        "falls back to a baseline risk and the Triage Board / Home show a visible "
        "**degradation banner** rather than a misleading uniform fleet.")

    pt.section("Lift-aware interventions")
    st.markdown(
        "Recommended interventions are gated by the well's **artificial-lift type**, so "
        "the board never proposes a physically-impossible job (no ESP swap on a "
        "rod-pumped well, no gas-lift optimization on a well with no injection). Because "
        "lift-correct jobs are cheaper, an **opportunity** additionally requires a real "
        "trigger — actively deferring production OR the fleet's elevated-risk quartile — "
        "not merely a cheap intervention that happens to pencil. This ranking is "
        "mirrored bit-for-bit against the PE-Pipeline orchestrator (a parity test pins "
        "them identical).")

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
            f"- **Triage ranking:** scored against the fleet's known seeded faults — "
            f"precision@10 **{sc['at_k'][10]['precision']:.0%}** "
            f"({sc['at_k'][10]['lift']:.1f}× lift over random), "
            f"recall@{sc['n_impaired']} **{sc['recall_at_n_impaired']:.0%}**. Honest, "
            "not a trivial 100% — low-rate failure modes defer few barrels and rank "
            "lower. (Full panel on the Triage Board.)")
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
