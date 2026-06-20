"""Evaluate the autoencoder vs. the shipped robust-z baseline. `python -m dl.evaluate`.

Both detectors score the SAME held-out test windows (healthy + injected ESP
pre-failure drift). We report ROC-AUC and PR-AUC (average precision) for each —
an honest, apples-to-apples answer to "does the deep model earn its keep over the
single-channel z-score the product already ships?".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from . import data as data_mod
from . import model as model_mod

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def _load_model():
    import torch
    ckpt = torch.load(ARTIFACTS / "autoencoder.pt", weights_only=False)
    cfg = ckpt["config"]
    m = model_mod.build_model(len(cfg["channels"]), hidden=cfg["hidden"],
                              latent=cfg["latent"], num_layers=cfg["num_layers"])
    m.load_state_dict(ckpt["state_dict"])
    return m, ckpt


def baseline_robust_z_scores(X_raw: np.ndarray, channels: list) -> np.ndarray:
    """The product's detector as a window anomaly score.

    Reuses ``daily-production-digest``'s exact ``robust_z`` (median/MAD): per
    channel, the robust-z of the window's last day vs. its own preceding days;
    the window score is the max |z| across channels — the strongest single-channel
    signal, which is precisely what a point z-score can see.
    """
    import sys
    digest_src = (Path(__file__).resolve().parent.parent
                  / "apps" / "daily-production-digest" / "src")
    if str(digest_src) not in sys.path:
        sys.path.insert(0, str(digest_src))
    from anomaly_detector import robust_z  # the shipped function

    scores = np.zeros(len(X_raw))
    for i, w in enumerate(X_raw):                      # w: (L, C)
        scores[i] = max(abs(robust_z(w[:, c])) for c in range(w.shape[1]))
    return scores


def evaluate(seed: int = 13, log=print) -> dict:
    model, ckpt = _load_model()
    cfg = ckpt["config"]
    ds = data_mod.build_dataset(length=cfg["length"], stride=cfg["stride"],
                                seed=seed)

    ae_scores = model_mod.reconstruction_error(model, ds.X_test)
    bz_scores = baseline_robust_z_scores(ds.X_test_raw, ds.channels)
    y = ds.y_test

    res = {
        "test_windows": int(len(y)),
        "anomalies": int(y.sum()),
        "autoencoder": {
            "roc_auc": round(float(roc_auc_score(y, ae_scores)), 4),
            "pr_auc": round(float(average_precision_score(y, ae_scores)), 4),
        },
        "robust_z_baseline": {
            "roc_auc": round(float(roc_auc_score(y, bz_scores)), 4),
            "pr_auc": round(float(average_precision_score(y, bz_scores)), 4),
        },
    }
    res["pr_auc_lift"] = round(
        res["autoencoder"]["pr_auc"] - res["robust_z_baseline"]["pr_auc"], 4)
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(res, indent=2))

    log(f"test: {res['test_windows']} windows, {res['anomalies']} injected anomalies")
    log(f"  LSTM autoencoder   ROC-AUC {res['autoencoder']['roc_auc']:.3f}"
        f"   PR-AUC {res['autoencoder']['pr_auc']:.3f}")
    log(f"  robust-z baseline  ROC-AUC {res['robust_z_baseline']['roc_auc']:.3f}"
        f"   PR-AUC {res['robust_z_baseline']['pr_auc']:.3f}")
    verdict = ("autoencoder wins" if res["pr_auc_lift"] > 0.02 else
               "baseline competitive" if abs(res["pr_auc_lift"]) <= 0.02 else
               "baseline wins")
    log(f"  PR-AUC lift {res['pr_auc_lift']:+.3f}  ->  {verdict}")
    return res


if __name__ == "__main__":
    evaluate()
