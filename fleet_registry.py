"""Canonical Upstream Copilot Suite fleet registry — the single source of truth
for well IDENTITY + METADATA shared across all eight apps.

Vendored byte-identical into each app (like ``theme.py``). Apps join this metadata
onto their OWN production / SCADA / event data by ``well_id`` — every app already
uses the shared ``well_0NN`` convention, so this is purely ADDITIVE enrichment:

- It does NOT generate production data.
- It does NOT touch any app's eval / label / holdout datasets.
- So eval gates (ESP calibration, Deferment reason-code accuracy, PE Copilot blind
  holdout) are completely unaffected.

The fleet is synthetic and Permian-flavored (Midland + Delaware basins, onshore).
Operator / API are ILLUSTRATIVE placeholders for a demo asset — NOT real proprietary
data. Real public production arrives later via the NDIC / Texas-RRC adapter milestone,
which populates this same interface (``get`` / ``as_frame`` / ``enrich``) behind the
scenes so the apps don't change.

A handful of curated "hero wells" carry a consistent cross-app storyline so the same
well reads coherently end-to-end (Monitor → Diagnose → Predict → Quantify → Authorize).
Every other ``well_0NN`` gets deterministic, stable metadata derived from its number,
so a lookup never fails and never changes between runs.

ADDITIVE EXTENSION (2026-07 · PE field-feedback round 1, Operations Center):
``surface_latlon`` (synthetic surface coordinates — county centroid + deterministic
within-county jitter, the exact formula the Operations Center map already used),
``ctb_for`` (deterministic central-tank-battery assignment, clustered by county),
and ``nri_for`` (deterministic per-well net revenue interest, varied across wells
like a real asset), each mirrored as a read-only property on ``WellMeta``
(``lat`` / ``lon`` / ``ctb`` / ``nri``) so ``enrich`` can join them. META_COLUMNS
gained ``ctb`` and ``nri``. Purely additive: no existing field or value changed.
This diverges this copy from sibling repos until they take the same block.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

META_COLUMNS = ("basin", "area", "formation", "lift", "lateral_length_ft", "peer_group",
                "hero", "ctb", "nri")

# Onshore Permian geology (public/generic — not proprietary).
_MIDLAND_FM = ["Wolfcamp A", "Wolfcamp B", "Spraberry (Lower)", "Spraberry (Jo Mill)", "Dean"]
_DELAWARE_FM = ["Bone Spring (2nd)", "Bone Spring (3rd)", "Wolfcamp A", "Avalon", "Wolfcamp X/Y"]
_MIDLAND_AREAS = ["Martin Co., TX", "Midland Co., TX", "Howard Co., TX", "Glasscock Co., TX"]
_DELAWARE_AREAS = ["Reeves Co., TX", "Loving Co., TX", "Ward Co., TX", "Culberson Co., TX"]
# Permian artificial-lift mix skews ESP / rod pump.
_LIFTS = ["ESP", "ESP", "ESP", "Rod pump", "Rod pump", "Gas lift", "Gas lift", "Flowing"]


@dataclass(frozen=True)
class WellMeta:
    well_id: str
    name: str
    basin: str          # "Midland" | "Delaware"
    area: str           # county-level locator
    formation: str
    lift: str           # ESP | Rod pump | Gas lift | Flowing
    api14: str          # synthetic Texas API (state 42) — illustrative only
    first_prod: str     # YYYY-MM
    peer_group: str     # for type-curve peer comparisons
    storyline: str = "" # cross-app narrative ("" unless a hero well)
    hero: bool = False
    lateral_length_ft: int = 9500   # completion lateral length (ft); curated per hero / derived

    def as_dict(self) -> dict:
        return asdict(self)

    # -- additive derived attributes (2026-07) — properties, not fields, so the
    # frozen constructor signature, as_dict(), and every existing caller are
    # untouched; ``enrich`` reaches them via getattr like any field.
    @property
    def lat(self) -> float:
        return surface_latlon(self.well_id)[0]

    @property
    def lon(self) -> float:
        return surface_latlon(self.well_id)[1]

    @property
    def ctb(self) -> str:
        return ctb_for(self.well_id)

    @property
    def nri(self) -> float:
        return nri_for(self.well_id)


# --- curated hero wells (storyline consistent with the apps' demo behavior) ----
# Keyed by the shared well_0NN id. Stories are written to match what each app
# actually surfaces for these wells (e.g. well_013 is the top triage opportunity).
_HERO: dict[str, WellMeta] = {
    "well_007": WellMeta(
        "well_007", "Garza 7H", "Midland", "Howard Co., TX", "Spraberry (Lower)",
        "Rod pump", "42-227-30007", "2022-08", "Midland · Spraberry",
        "Base-decline rod-pump producer with a recent rate step-down — the Daily "
        "Digest's flagged rate-loss event and a Deferment IQ underperformance case.",
        hero=True, lateral_length_ft=7800),
    "well_008": WellMeta(
        "well_008", "Loving 8H", "Delaware", "Loving Co., TX", "Avalon",
        "ESP", "42-301-30008", "2023-01", "Delaware · Avalon",
        "ESP producer in downthrust — sliding gross fluid and falling runtime "
        "(pump operating below its range) — an ESP-swap candidate across the ESP "
        "model, PE Copilot, and AFE Copilot.", hero=True, lateral_length_ft=9600),
    "well_013": WellMeta(
        "well_013", "Martin 13H", "Midland", "Martin Co., TX", "Wolfcamp A",
        "Gas lift", "42-317-30013", "2023-04", "Midland · Wolfcamp A",
        "Gas-lift well with gas interference / unstable lift — the Fleet Triage "
        "Board's top risked-NPV opportunity (gas-lift-optimization candidate) and "
        "the pipeline's flagship.", hero=True, lateral_length_ft=10200),
    "well_022": WellMeta(
        "well_022", "Reeves 22H", "Delaware", "Reeves Co., TX", "Bone Spring (3rd)",
        "ESP", "42-389-30022", "2022-11", "Delaware · Bone Spring",
        "ESP wear with current imbalance — high 30-day failure risk and a near-term "
        "remaining-useful-life well.", hero=True, lateral_length_ft=9900),
    "well_041": WellMeta(
        "well_041", "Ward 41H", "Delaware", "Ward Co., TX", "Wolfcamp A",
        "ESP", "42-475-30041", "2023-06", "Delaware · Wolfcamp A",
        "Late-life ESP failure signature — an ESP-swap authorization case feeding "
        "AFE Copilot and a Capital Optimizer workover candidate.", hero=True, lateral_length_ft=10500),
    "well_048": WellMeta(
        "well_048", "Midland 48H", "Midland", "Midland Co., TX", "Wolfcamp B",
        "ESP", "42-329-30048", "2023-02", "Midland · Wolfcamp B",
        "ESP-swap candidate with degrading thrust — recurring across the predict "
        "and authorize stages.", hero=True, lateral_length_ft=9800),
}


def _suffix(well_id: str) -> int:
    """Numeric suffix of a well_0NN id; 0 if unparseable (deterministic, no raise)."""
    tail = well_id.rsplit("_", 1)[-1]
    digits = "".join(c for c in tail if c.isdigit())
    return int(digits) if digits else 0


def _derive(well_id: str) -> WellMeta:
    """Deterministic, stable metadata for any non-curated well_0NN."""
    n = _suffix(well_id)
    is_midland = (n % 2 == 0)
    basin = "Midland" if is_midland else "Delaware"
    fms = _MIDLAND_FM if is_midland else _DELAWARE_FM
    areas = _MIDLAND_AREAS if is_midland else _DELAWARE_AREAS
    formation = fms[n % len(fms)]
    area = areas[(n // 2) % len(areas)]
    lift = _LIFTS[n % len(_LIFTS)]
    county = 200 + (n % 99)
    api14 = f"42-{county:03d}-{30000 + n:05d}"
    year = 2021 + (n % 4)
    month = 1 + (n % 12)
    lateral = 7500 + (n * 311) % 5001   # deterministic 7,500–12,500 ft
    return WellMeta(
        well_id=well_id, name=f"{area.split(' ')[0]} {n}H", basin=basin, area=area,
        formation=formation, lift=lift, api14=api14,
        first_prod=f"{year}-{month:02d}", peer_group=f"{basin} · {formation.split(' (')[0]}",
        lateral_length_ft=lateral,
    )


def get(well_id: str) -> WellMeta:
    """Registry metadata for a well_id — curated hero data if present, else derived.

    Never raises and never changes between runs (deterministic)."""
    return _HERO.get(well_id) or _derive(well_id)


def hero_wells() -> list[WellMeta]:
    """The curated hero wells, in id order."""
    return [_HERO[k] for k in sorted(_HERO)]


# --- synthetic surface coordinates / CTB / per-well NRI (deterministic, additive) --
# Approximate centroids of the Permian counties the registry uses, so the map is
# geographically honest at the county level. Per-well coordinates are SYNTHETIC —
# centroid + a stable jitter seeded off the well number (the registry carries no
# real surface locations). Same formula the Operations Center map used before this
# block existed, so no well moved when the logic migrated here.
_COUNTY_LATLON: dict[str, tuple[float, float]] = {
    "Martin": (32.30, -101.95), "Midland": (31.87, -102.03),
    "Howard": (32.31, -101.44), "Glasscock": (31.87, -101.52),
    "Reeves": (31.42, -103.69), "Loving": (31.85, -103.58),
    "Ward": (31.51, -103.10), "Culberson": (31.44, -104.52),
}


def _county_of(well_id: str) -> str:
    return get(well_id).area.split(" Co.")[0].strip()


def surface_latlon(well_id: str) -> tuple[float, float]:
    """Deterministic synthetic surface (lat, lon): the well's county centroid plus a
    stable within-county jitter seeded off the well number. Synthetic by design —
    never changes between runs, never raises (unknown counties fall back to a
    mid-Permian anchor)."""
    import numpy as np
    lat0, lon0 = _COUNTY_LATLON.get(_county_of(well_id), (31.8, -102.5))
    rng = np.random.default_rng(_suffix(well_id) * 2654435761 % (2 ** 32))
    return (lat0 + float(rng.uniform(-0.13, 0.13)),
            lon0 + float(rng.uniform(-0.16, 0.16)))


def ctb_for(well_id: str) -> str:
    """Deterministic central-tank-battery (CTB) assignment: two batteries per county,
    wells stably grouped by their number so pads read like a real gathering layout
    (e.g. ``"Reeves CTB-1"``). Purely derived — never changes between runs."""
    n = _suffix(well_id)
    return f"{_county_of(well_id)} CTB-{(n // 8) % 2 + 1}"


def nri_for(well_id: str) -> float:
    """Deterministic per-well net revenue interest, varied across wells like a real
    asset (mixed royalty burdens): a basin-level base (Midland 0.80 / Delaware 0.77)
    plus a seeded per-well adjustment, landing in ≈0.73–0.85. Synthetic and
    illustrative — session overrides on the Operations Center Data page win over
    this default. Never changes between runs."""
    import random as _random
    meta = get(well_id)
    base = 0.80 if meta.basin == "Midland" else 0.77
    rng = _random.Random(_suffix(well_id) * 9176 + 3)
    return round(base + rng.uniform(-0.07, 0.05), 4)


# --- synthetic intervention / workover history (deterministic, additive) -------
# Each workover: (all-in cost $, plausible uplift band bopd). Uplift bands are sized
# so realized $/bopd lands in a defensible ~$2k–8k band (a big ESP swap restores far
# more rate than a gas-lift tweak) instead of the RNG that gave $35k/bopd ESP swaps.
_WORKOVERS: dict[str, tuple[int, int, int]] = {
    "ESP swap":              (523_000, 80, 180),
    "Recompletion":          (410_000, 60, 160),
    "Acid stimulation":      (274_000, 40, 110),
    "Scale treatment":       (274_000, 35,  90),
    "Tubing repair":         ( 96_000, 10,  35),
    "Rod-pump workover":     ( 84_000, 12,  40),
    "Gas-lift optimization": ( 27_000,  8,  30),
}
# Only jobs that physically apply to each lift type (you can't run an ESP swap on a
# rod-pumped or flowing well, or optimize gas lift on a well with no injection).
_LIFT_WORKOVERS: dict[str, list[str]] = {
    "ESP":      ["ESP swap", "Scale treatment", "Acid stimulation", "Tubing repair"],
    "Rod pump": ["Rod-pump workover", "Tubing repair", "Acid stimulation"],
    "Gas lift": ["Gas-lift optimization", "Acid stimulation", "Recompletion"],
    "Flowing":  ["Acid stimulation", "Recompletion", "Tubing repair"],
}


def well_history(well_id: str, as_of: str = "2026-05-29") -> dict:
    """Deterministic synthetic well-work history for a well: when it came online and
    every past intervention (date, type, cost, uplift). Seeded off the well number so
    it never changes between runs. Additive — derives nothing from production data."""
    import random as _random
    from datetime import date

    meta = get(well_id)
    n = _suffix(well_id)
    rng = _random.Random(n * 131 + 7)
    fp = meta.first_prod  # "YYYY-MM"
    try:
        fy, fm = (int(x) for x in fp.split("-"))
    except Exception:  # noqa: BLE001
        fy, fm = 2022, 1
    ay, am = (int(x) for x in as_of.split("-")[:2])
    months_online = max(1, (ay - fy) * 12 + (am - fm))

    # Hero wells carry a richer history; everyone gets 0–3 prior jobs.
    n_jobs = (2 if meta.hero else 0) + rng.randint(0, 2)
    n_jobs = min(n_jobs, max(0, months_online // 6))
    candidates = _LIFT_WORKOVERS.get(meta.lift, list(_WORKOVERS))
    records = []
    for _i in range(n_jobs):
        mo = rng.randint(2, max(3, months_online - 1))
        yy = fy + (fm - 1 + mo) // 12
        mm = (fm - 1 + mo) % 12 + 1
        kind = rng.choice(candidates)               # only jobs valid for this lift
        base, ulo, uhi = _WORKOVERS[kind]
        cost = int(base * rng.uniform(0.85, 1.2))
        uplift = rng.randint(ulo, uhi)              # uplift scaled to the job's size
        records.append({"date": date(yy, mm, 1).isoformat(), "type": kind,
                        "cost_usd": cost, "uplift_bopd": uplift,
                        "result": "restored to type curve" if rng.random() > 0.3
                        else f"+{uplift} bopd uplift"})
    records.sort(key=lambda r: r["date"])
    return {
        "online_since": fp,
        "months_online": months_online,
        "years_online": round(months_online / 12.0, 1),
        "n_workovers": len(records),
        "records": records,
        "last_worked": records[-1]["date"] if records else None,
    }


def as_frame(well_ids):
    """Return a pandas DataFrame of metadata for the given well_ids (pandas lazy-imported)."""
    import pandas as pd
    return pd.DataFrame([get(w).as_dict() for w in well_ids])


def enrich(df, id_col: str = "well_id", columns=META_COLUMNS):
    """Left-join registry metadata columns onto a DataFrame keyed by ``id_col``.

    Additive only — returns a copy with the requested metadata columns added.
    Unknown / non-well_0NN ids get deterministically-derived metadata, so no row
    is dropped and no NaNs are introduced.
    """
    out = df.copy()
    for col in columns:
        out[col] = out[id_col].map(lambda w: getattr(get(str(w)), col))
    return out
