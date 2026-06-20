"""Data prep for the anomaly autoencoder — pure numpy/pandas (no torch).

Pipeline:
    load the 100-well x 400-day x 9-channel digest fleet
      -> split wells into healthy-train / val / test (well-disjoint, no leakage)
        -> slide fixed-length windows over each well's series
          -> standardize per channel (stats fit on TRAIN ONLY)
            -> for the TEST set, inject realistic ESP pre-failure drift into a
               fraction of windows and label them (1 = anomalous, 0 = normal)

Keeping this torch-free means the windowing + injection + scaling are unit-tested
without the heavy dep, and the same arrays feed both the autoencoder and the
robust-z baseline (apples-to-apples eval).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import CHANNELS


@dataclass
class Dataset:
    """Standardized windows ready for the model + the baseline."""
    X_train: np.ndarray        # (n_train, L, C) healthy windows
    X_val: np.ndarray          # (n_val,   L, C) healthy windows
    X_test: np.ndarray         # (n_test,  L, C) standardized (with injections)
    X_test_raw: np.ndarray     # (n_test,  L, C) UN-standardized (for the z baseline)
    y_test: np.ndarray         # (n_test,)       1 = injected anomaly, 0 = normal
    mean: np.ndarray           # (C,) train channel means
    std: np.ndarray            # (C,) train channel stds
    channels: list

    @property
    def shape_str(self) -> str:
        return (f"train {self.X_train.shape} · val {self.X_val.shape} · "
                f"test {self.X_test.shape} ({int(self.y_test.sum())} anomalous)")


def _fleet_dir():
    import core  # authoritative path to the generated digest fleet
    return core.DIGEST_FLEET


def load_fleet(channels: list | None = None) -> dict:
    """{well_id: (T, C) float array} from the digest fleet CSVs."""
    channels = channels or CHANNELS
    out = {}
    for path in sorted(glob.glob(str(_fleet_dir() / "well_*.csv"))):
        df = pd.read_csv(path)
        if not set(channels).issubset(df.columns):
            continue
        wid = path.split("/")[-1].replace(".csv", "")
        out[wid] = df[channels].to_numpy(dtype=np.float64)
    return out


def _windows(series: np.ndarray, length: int, stride: int) -> np.ndarray:
    """(T, C) -> (n_windows, length, C) via a sliding window."""
    n = (len(series) - length) // stride + 1
    if n <= 0:
        return np.empty((0, length, series.shape[1]))
    return np.stack([series[i * stride:i * stride + length]
                     for i in range(n)])


def _inject_failure(window: np.ndarray, rng: np.random.Generator,
                    channels: list) -> np.ndarray:
    """Overlay a realistic ESP pre-failure signature onto a window (copy).

    The failure mode the single-channel z-score is weakest on: a SLOW, CORRELATED
    drift ramping in over the window rather than a one-day spike —
      current imbalance climbs · intake pressure sags · motor amps/temp rise ·
      oil & fluid rates fade.
    Severity is randomized so the test set spans subtle -> obvious.
    """
    w = window.copy()
    L = len(w)
    ramp = np.linspace(0.0, 1.0, L)            # 0 -> 1 over the window (gradual)
    sev = rng.uniform(0.5, 1.5)                # subtle .. obvious
    idx = {c: i for i, c in enumerate(channels)}

    def add(ch, delta):
        if ch in idx:
            w[:, idx[ch]] = w[:, idx[ch]] + delta

    base = w  # for proportional drifts
    add("current_imbalance_pct", ramp * sev * 12.0)                 # +up to ~18%
    add("motor_amps", ramp * sev * 0.18 * np.nanmedian(base[:, idx["motor_amps"]]))
    add("motor_temp_f", ramp * sev * 14.0)                          # heating
    add("intake_pressure_psi",
        -ramp * sev * 0.20 * np.nanmedian(base[:, idx["intake_pressure_psi"]]))
    add("bopd", -ramp * sev * 0.22 * np.nanmedian(base[:, idx["bopd"]]))
    add("bfpd", -ramp * sev * 0.18 * np.nanmedian(base[:, idx["bfpd"]]))
    # small idiosyncratic noise so injected windows aren't all identical
    w = w + rng.normal(0.0, 0.01, size=w.shape) * np.nanstd(base, axis=0)
    return w


def build_dataset(length: int = 30, stride: int = 5, seed: int = 13,
                  test_frac: float = 0.25, val_frac: float = 0.15,
                  anomaly_frac: float = 0.18, channels: list | None = None,
                  max_wells: int | None = None) -> Dataset:
    """Assemble the standardized train/val/test windows + injected test labels."""
    channels = channels or CHANNELS
    fleet = load_fleet(channels)
    wells = sorted(fleet)
    if max_wells:
        wells = wells[:max_wells]
    rng = np.random.default_rng(seed)
    rng.shuffle(wells)

    n_test = max(1, int(len(wells) * test_frac))
    n_val = max(1, int(len(wells) * val_frac))
    test_wells = wells[:n_test]
    val_wells = wells[n_test:n_test + n_val]
    train_wells = wells[n_test + n_val:]

    def stack(ws):
        parts = [_windows(fleet[w], length, stride) for w in ws]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.empty((0, length, len(channels)))

    X_train = stack(train_wells)
    X_val = stack(val_wells)
    X_test_raw = stack(test_wells)

    # inject anomalies into a fraction of TEST windows
    y_test = np.zeros(len(X_test_raw), dtype=int)
    n_anom = int(len(X_test_raw) * anomaly_frac)
    anom_idx = rng.choice(len(X_test_raw), size=n_anom, replace=False)
    X_test_inj = X_test_raw.copy()
    for i in anom_idx:
        X_test_inj[i] = _inject_failure(X_test_raw[i], rng, channels)
        y_test[i] = 1

    # standardize per channel on TRAIN ONLY (no leakage)
    mean = X_train.reshape(-1, len(channels)).mean(axis=0)
    std = X_train.reshape(-1, len(channels)).std(axis=0)
    std[std < 1e-9] = 1.0

    def norm(a):
        return (a - mean) / std

    return Dataset(
        X_train=norm(X_train), X_val=norm(X_val), X_test=norm(X_test_inj),
        X_test_raw=X_test_inj,    # raw (un-normalized) injected windows for z-baseline
        y_test=y_test, mean=mean, std=std, channels=list(channels))
