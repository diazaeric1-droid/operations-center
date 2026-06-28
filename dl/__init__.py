"""Deep anomaly detection — an LSTM autoencoder over the SCADA fleet.

The shipped surveillance path flags rate drops with a robust median/MAD z-score
(``daily-production-digest/src/anomaly_detector.py``): fast, interpretable, and
single-channel. This package asks whether a **deep, unsupervised, multivariate**
model does better on the failure mode the z-score misses — slow, correlated drift
across several SCADA channels (current imbalance creeping up while intake pressure
sags and amps climb) that no single-channel point test catches until it's late.

Method (textbook unsupervised anomaly detection):

    train an LSTM autoencoder ONLY on healthy wells -> it learns "normal" dynamics
      -> reconstruction error on a new window is the anomaly score
        -> evaluate on held-out wells with injected, realistic ESP pre-failure
           drift, head-to-head against the robust-z baseline (ROC-AUC / PR-AUC).

Convention (same as ``esp-failure-risk-agent/src/sequence_model.py``): ``torch``
is an OPTIONAL dependency, this package imports cleanly without it, and nothing
here is wired into the deployed Streamlit app. It's a trainable/evaluable
experiment — `python -m dl.train`, `python -m dl.evaluate` — not a live code path.
"""
from __future__ import annotations

__all__ = ["data", "model", "train", "evaluate"]

# The 9 daily SCADA channels the autoencoder reconstructs (digest fleet schema).
CHANNELS = [
    "bopd", "bfpd", "intake_pressure_psi", "motor_temp_f", "motor_amps",
    "runtime_pct", "current_imbalance_pct", "drive_freq_hz", "gas_mcfd",
]
