# Deep anomaly detection — LSTM autoencoder over the SCADA fleet

A PyTorch LSTM autoencoder for **unsupervised, multivariate** equipment-anomaly
detection, built head-to-head against the median/MAD robust-z detector the
product already ships.

## The question

The shipped surveillance detector
(`apps/daily-production-digest/src/anomaly_detector.py`) flags rate drops with a
**single-channel** robust z-score: it asks "is today's value an outlier vs. this
well's recent baseline?". That's fast, interpretable, and strong on **sudden
step-changes** — but blind to the failure mode that actually precedes most ESP
failures: a **slow, correlated drift** across several channels at once (current
imbalance creeping up while intake pressure sags and motor amps/temp climb and
rate fades). No single day is an outlier, so a point z-score never fires until
it's too late.

Does a deep model close that gap?

## The method (textbook unsupervised anomaly detection)

```
train an LSTM autoencoder ONLY on healthy wells  →  it learns normal dynamics
  →  reconstruction MSE on a new window = anomaly score
    →  evaluate on held-out wells with injected, realistic ESP pre-failure drift,
       vs. the robust-z baseline, on the SAME windows (ROC-AUC / PR-AUC)
```

- **Data**: the 100-well × 400-day digest fleet, 9 SCADA channels per day.
- **Model**: LSTM encoder → 16-d latent → LSTM decoder (`dl/model.py`, ~55k params).
- **Split**: well-disjoint train / val / test (no leakage); scaler fit on train only.
- **Injection**: gradual multivariate ESP-degradation signature ramped over each
  anomalous window (`dl/data.py`), severity randomized subtle→obvious.
- **Baseline**: the product's exact `robust_z` function, scored per window as the
  max |z| across channels — i.e. the strongest signal a point test can see.

## Result (`dl/artifacts/eval_report.json`)

| Detector | ROC-AUC | PR-AUC |
|---|---:|---:|
| **LSTM autoencoder** | **0.998** | **0.976** |
| robust-z baseline (shipped) | 0.532 | 0.177 |

**+0.80 PR-AUC.** Trained in ~25 s on CPU (no GPU); ~22k windows/s at inference.

### The honest read

This is **not** "deep learning strictly dominates." It's a targeted win: the
autoencoder is decisively better on **gradual, multivariate drift** — the regime
the point z-score is blind to by construction (a slow ramp contaminates its own
baseline, so the last day is never an outlier). The z-score remains the right,
cheaper tool for **sudden single-channel step-drops**, its design target. The two
are complementary; the autoencoder covers the surveillance gap the shipped
detector leaves open. Measuring that — rather than asserting the deep model wins
everywhere — is the point. (Compare `esp-failure-risk-agent/src/sequence_model.py`,
which honestly found a supervised Temporal-CNN does *not* beat XGBoost on the same
fleet — too little data. Different task, different answer.)

## Run it

```bash
pip install -r requirements-dl.txt     # torch (CPU is fine)
python -m dl.train                     # ~25s; writes dl/artifacts/autoencoder.pt + report
python -m dl.evaluate                  # writes dl/artifacts/eval_report.json
pytest tests/test_dl.py                # data tests always run; model tests when torch present
```

## Convention

`torch` is an **optional** dependency: every module here imports cleanly without
it (the `RuntimeError` only fires if you actually call the model), and nothing in
`dl/` is imported by the deployed Streamlit app. This is a trainable experiment,
not a live code path — same opt-in discipline as `sequence_model.py`.
