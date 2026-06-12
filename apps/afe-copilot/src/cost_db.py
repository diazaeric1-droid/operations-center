"""Historical cost benchmarks for AFE line items.

In production this would be backed by the operator's actual AFE database
(SAP, Quorum, Oracle, etc.). For the open-source demo, ships with synthetic
Permian-basin-realistic values an engineer would recognize.

Contingency is computed programmatically from the stated percentage of direct
cost (see CONTINGENCY_PCT), so the line-item label can never disagree with the
arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InterventionType = Literal[
    "acid_stimulation",
    "scale_treatment",
    "esp_swap",
    "esp_to_beam_conversion",
    "rod_pump_workover",
    "gas_lift_optimization",
    "paraffin_treatment",
    "p_and_a",
]


@dataclass
class LineItem:
    category: str       # e.g., "Workover rig"
    description: str
    unit: str           # e.g., "day", "lump sum", "bbl"
    qty: float
    unit_cost_usd: float
    vendor: str | None = None
    cost_class: Literal["tangible", "intangible"] = "intangible"

    @property
    def total_usd(self) -> float:
        return self.qty * self.unit_cost_usd


# Categories that are CAPITALIZED equipment (tangible) for tax/AFE purposes; the
# rest are intangible drilling/workover costs (IDC: rig days, services, labor,
# chemicals, supervision) — currently expensed, where tangibles are depreciated.
# Every real AFE shows this split, so finance can model the tax treatment.
TANGIBLE_CATEGORIES: set[str] = {
    "ESP unit", "ESP cable", "VSD + transformer", "Beam pumping unit", "Rod string",
    "Downhole pump", "Surface motor + VFD", "Foundation + pad", "Gas lift valves",
    "Replacement rods", "Wireline plunger",
}


def classify_cost_class(category: str) -> str:
    return "tangible" if category in TANGIBLE_CATEGORIES else "intangible"


# Reference DIRECT-cost line-item templates per intervention (contingency added
# programmatically by lookup_cost_template).
# Costs reflect Delaware/Midland Basin Q1 2026 going rates (synthetic but realistic).
COST_TEMPLATES: dict[InterventionType, list[LineItem]] = {
    "acid_stimulation": [
        LineItem("Workover rig",        "Pulling unit, 4 days",                "day",     4,   18_000, "Permian WOR"),
        LineItem("Coiled tubing unit",  "CTU + pump truck for acid placement", "day",     2,   42_000, "Halliburton"),
        LineItem("Acid system",         "15% HCl + non-emulsifier, 5,000 gal", "lump",    1,   38_000, "Multi-Chem"),
        LineItem("Diverter package",    "Solid diverter w/ ball sealers",      "lump",    1,   14_500, "BJ Energy"),
        LineItem("Wellsite supervision","Company man + co-man, 4 days",        "day",     4,    2_400, None),
        LineItem("Flowback / disposal", "Flowback equipment + SWD trucking",   "day",     5,    6_200, "Liberty"),
    ],
    "scale_treatment": [
        LineItem("Coiled tubing unit",  "CTU for inhibitor squeeze",           "day",     1,   35_000, "Halliburton"),
        LineItem("Scale inhibitor",     "Phosphonate squeeze, 2,500 gal",      "lump",    1,   22_000, "Multi-Chem"),
        LineItem("HCl wash",            "5% HCl pre-wash, 1,500 gal",          "lump",    1,    9_800, "Multi-Chem"),
        LineItem("Wellsite supervision","Company man, 2 days",                 "day",     2,    1_200, None),
        LineItem("Flowback",            "Flowback equipment, 3 days",          "day",     3,    5_400, "Liberty"),
    ],
    "esp_swap": [
        LineItem("Workover rig",        "Pulling unit, 6 days",                "day",     6,   22_000, "Permian WOR"),
        LineItem("ESP unit",            "REDA/Centrilift 538-series, new",     "lump",    1,  185_000, "SLB"),
        LineItem("ESP cable",           "5,000 ft of #4 power cable",          "ft",  5_000,        18, "SLB"),
        LineItem("VSD + transformer",   "Surface VSD package",                 "lump",    1,   42_000, "Yaskawa"),
        LineItem("Tubing inspection",   "Tubing tally + EMI inspection",       "ft",  5_000,         2.5, None),
        LineItem("Wellsite supervision","Company man + co-man, 6 days",        "day",     6,    2_400, None),
    ],
    "esp_to_beam_conversion": [
        LineItem("Workover rig",        "Pulling unit, 7 days",                "day",     7,   22_000, "Permian WOR"),
        LineItem("Beam pumping unit",   "C-228D-200-74, refurbished",          "lump",    1,   95_000, "Lufkin"),
        LineItem("Rod string",          "Grade D fiberglass + steel, 5,000 ft","ft",  5_000,         9, "Norris"),
        LineItem("Downhole pump",       "1.75-in plunger, 12-ft barrel",       "lump",    1,    8_500, "Don-Nan"),
        LineItem("Surface motor + VFD", "20 HP w/ VFD",                        "lump",    1,   28_000, "Toshiba"),
        LineItem("Foundation + pad",    "Concrete pad + anchors",              "lump",    1,   18_500, None),
        LineItem("Wellsite supervision","Company man, 7 days",                 "day",     7,    1_200, None),
    ],
    "rod_pump_workover": [
        LineItem("Workover rig",        "Pulling unit, 3 days",                "day",     3,   18_000, "Permian WOR"),
        LineItem("Rod inspection",      "EMI inspection of full rod string",   "lump",    1,   12_000, None),
        LineItem("Replacement rods",    "Replace failed section, 200 ft",      "ft",    200,         9, "Norris"),
        LineItem("Downhole pump",       "Refurb 1.5-in plunger if needed",     "lump",    1,    5_200, "Don-Nan"),
        LineItem("Wellsite supervision","Company man, 3 days",                 "day",     3,    1_200, None),
    ],
    "gas_lift_optimization": [
        LineItem("Wireline unit",       "Slickline for valve change",          "day",     1,    8_500, "Halliburton"),
        LineItem("Gas lift valves",     "Replace bottom 3 mandrels",           "ea",      3,    2_800, "Camco"),
        LineItem("Injection optimization","Engineering review + model rerun",  "lump",    1,    6_000, None),
        LineItem("Wellsite supervision","Company man, 1 day",                  "day",     1,    1_200, None),
    ],
    "paraffin_treatment": [
        LineItem("Hot oil truck",       "Hot oiler, 80 bbl @ 180°F",           "trip",    1,    7_500, "Select"),
        LineItem("Wireline plunger",    "Inspect + replace plunger",           "lump",    1,    3_500, None),
        LineItem("Paraffin inhibitor",  "Continuous chemical, 30-day supply",  "lump",    1,    4_200, "Multi-Chem"),
        LineItem("Wellsite supervision","Company man, 1 day",                  "day",     1,    1_200, None),
    ],
    "p_and_a": [
        LineItem("Workover rig",        "P&A rig, 5 days",                     "day",     5,   24_000, "Permian WOR"),
        LineItem("Cement",              "Class H cement, multiple plugs",      "sack",  450,        28, "Halliburton"),
        LineItem("Wellbore prep",       "Scrape, junk basket runs",            "lump",    1,   18_500, None),
        LineItem("Wireline plugs",      "Mechanical plug sets",                "ea",      4,    4_500, "Halliburton"),
        LineItem("Surface restoration", "Cellar cleanup, signage",             "lump",    1,   12_000, None),
        LineItem("Regulatory filing",   "RRC W-3 filing + bond release",       "lump",    1,    3_500, None),
    ],
}


# Contingency as a fraction of DIRECT cost, per intervention. Higher-risk jobs
# (conversions, P&A downhole risk) carry 15%.
CONTINGENCY_PCT: dict[InterventionType, float] = {
    "acid_stimulation":       0.10,
    "scale_treatment":        0.10,
    "esp_swap":               0.10,
    "esp_to_beam_conversion": 0.15,
    "rod_pump_workover":      0.10,
    "gas_lift_optimization":  0.10,
    "paraffin_treatment":     0.10,
    "p_and_a":                0.15,
}


def _contingency_line(intervention: InterventionType, direct_items: list[LineItem]) -> LineItem:
    """Build the contingency line as an exact percentage of direct cost."""
    pct = CONTINGENCY_PCT[intervention]
    direct_total = sum(i.total_usd for i in direct_items)
    return LineItem(
        category="Contingency",
        description=f"{pct:.0%} contingency on direct cost",
        unit="lump",
        qty=1,
        unit_cost_usd=round(pct * direct_total),
        vendor=None,
    )


def lookup_cost_template(intervention: InterventionType) -> list[LineItem]:
    """Return the canonical line-item list (direct lines + computed contingency),
    each tagged tangible/intangible for the AFE cost-class rollup."""
    if intervention not in COST_TEMPLATES:
        raise ValueError(f"Unknown intervention: {intervention}. "
                         f"Known: {list(COST_TEMPLATES)}")
    direct = []
    for item in COST_TEMPLATES[intervention]:
        li = LineItem(**item.__dict__)
        li.cost_class = classify_cost_class(li.category)
        direct.append(li)
    return direct + [_contingency_line(intervention, direct)]


def total_estimate(intervention: InterventionType) -> float:
    return sum(item.total_usd for item in lookup_cost_template(intervention))


def cost_rollup(intervention: InterventionType) -> dict[str, float]:
    """Direct / contingency / total plus the tangible-vs-intangible (IDC) split.

    Contingency inherits the tangible:intangible *ratio* of the direct cost so the
    two subtotals still add to the grand total.
    """
    items = lookup_cost_template(intervention)
    direct_items = [i for i in items if i.category != "Contingency"]
    contingency = sum(i.total_usd for i in items if i.category == "Contingency")
    direct = sum(i.total_usd for i in direct_items)
    tangible_direct = sum(i.total_usd for i in direct_items if i.cost_class == "tangible")
    intangible_direct = direct - tangible_direct
    # Spread contingency across the two classes in proportion to direct cost.
    t_frac = (tangible_direct / direct) if direct else 0.0
    tangible = tangible_direct + contingency * t_frac
    intangible = intangible_direct + contingency * (1 - t_frac)
    return {
        "direct": direct,
        "contingency": contingency,
        "total": direct + contingency,
        "tangible": tangible,
        "intangible": intangible,
    }


def benchmark_summary() -> dict[str, float]:
    """Total estimate per intervention type — useful for at-a-glance benchmarking."""
    return {i: total_estimate(i) for i in COST_TEMPLATES}
