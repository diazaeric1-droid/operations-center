"""Score a live fleet with the trained autoencoder — the single source of truth
shared by the Surveillance Early-Warning tab and the Morning Brief.

Pure (no Streamlit). ``torch`` is required to *score* but NOT to import: callers
gate on ``model_ready()`` first, so the deployed app (no torch) never trips here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import model as _model

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Friendly names for the SCADA channels (the "driver" the model reconstructs worst).
CH_LABEL = {
    "bopd": "oil rate", "bfpd": "fluid rate", "intake_pressure_psi": "intake pressure",
    "motor_temp_f": "motor temp", "motor_amps": "motor amps", "runtime_pct": "runtime",
    "current_imbalance_pct": "current imbalance", "drive_freq_hz": "drive freq",
    "gas_mcfd": "gas rate",
}
ALARM_Z = 3.5   # |robust-z| the rate-drop alarm needs to fire (illustrative)


def model_ready() -> bool:
    """True when torch is installed AND a trained model is on disk."""
    return _model.torch_available() and (ARTIFACTS / "autoencoder.pt").exists()


def _robust_z():
    """The shipped median/MAD detector, for the rate-drop-alarm contrast."""
    import sys
    digest_src = (Path(__file__).resolve().parent.parent / "apps"
                  / "daily-production-digest" / "src")
    if str(digest_src) not in sys.path:
        sys.path.insert(0, str(digest_src))
    from anomaly_detector import robust_z
    return robust_z


def score_fleet_latest(fleet: dict) -> pd.DataFrame:
    """Score each well's latest window by reconstruction error.

    Returns a DataFrame (well, score, driver, maxz) sorted by score desc — empty
    if no well has a full window. Requires torch + a trained model (model_ready()).
    """
    import torch
    ckpt = torch.load(ARTIFACTS / "autoencoder.pt", weights_only=False)
    cfg = ckpt["config"]
    chans, L = cfg["channels"], cfg["length"]
    m = _model.build_model(len(chans), hidden=cfg["hidden"], latent=cfg["latent"],
                           num_layers=cfg["num_layers"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    mean = np.asarray(ckpt["mean"], dtype=float)
    std = np.asarray(ckpt["std"], dtype=float)
    std[std < 1e-9] = 1.0
    robust_z = _robust_z()

    rows = []
    for wid, df in fleet.items():
        if df is None or not set(chans).issubset(df.columns) or len(df) < L:
            continue
        w = df[chans].to_numpy(dtype=float)[-L:]
        xn = (w - mean) / std
        with torch.no_grad():
            recon = m(torch.tensor(xn[None], dtype=torch.float32)).numpy()[0]
        err = (recon - xn) ** 2
        per_ch = err.mean(axis=0)
        zmax = max(abs(robust_z(w[:, j])) for j in range(len(chans)))
        rows.append({"well": str(wid), "score": float(err.mean()),
                     "driver": CH_LABEL.get(chans[int(np.argmax(per_ch))],
                                            chans[int(np.argmax(per_ch))]),
                     "maxz": float(zmax)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def flag_table(df: pd.DataFrame, pct: float = 85.0,
               alarm_z: float = ALARM_Z) -> pd.DataFrame:
    """Add flagged / alarm / deep_only columns to a scored frame.

    flagged   = top (100-pct)% by drift score (the early-warning tier)
    alarm     = the shipped rate-drop alarm would also fire (|z| >= alarm_z)
    deep_only = flagged by the autoencoder but NOT by the rate-drop alarm
                (the slow drift the point alarm can't see — the headline catch)
    """
    if df.empty:
        return df
    df = df.copy()
    cut = float(np.percentile(df["score"], pct))
    df["flagged"] = df["score"] >= cut
    df["alarm"] = df["maxz"] >= alarm_z
    df["deep_only"] = df["flagged"] & ~df["alarm"]
    return df
