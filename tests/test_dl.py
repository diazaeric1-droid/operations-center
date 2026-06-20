"""Tests for the deep anomaly-detection package (dl/).

Tiered by dependency:
  * data/injection tests — pure numpy, always run.
  * model + train/eval smoke — skip when torch isn't installed.
The bootstrapped fixture (conftest) guarantees the digest fleet exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from dl import data as data_mod
from dl import model as model_mod

needs_torch = pytest.mark.skipif(not model_mod.torch_available(),
                                 reason="torch not installed")


# --- data (pure, always run) --------------------------------------------------
def test_build_dataset_shapes(bootstrapped):
    ds = data_mod.build_dataset(length=30, stride=10, max_wells=40)
    assert ds.X_train.ndim == 3 and ds.X_train.shape[1:] == (30, 9)
    assert len(ds.X_test) == len(ds.y_test) == len(ds.X_test_raw)
    assert ds.y_test.sum() > 0                      # some anomalies injected
    # standardization: train windows are ~zero-mean / unit-std per channel
    flat = ds.X_train.reshape(-1, 9)
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-3)


def test_injection_raises_failure_signature(bootstrapped):
    """An injected window must show the ESP drift: current imbalance climbs."""
    rng = np.random.default_rng(0)
    fleet = data_mod.load_fleet()
    w = data_mod._windows(next(iter(fleet.values())), 30, 5)[0]
    inj = data_mod._inject_failure(w, rng, data_mod.CHANNELS)
    ci = data_mod.CHANNELS.index("current_imbalance_pct")
    # last-day imbalance is higher after injection (gradual ramp up)
    assert inj[-1, ci] > w[-1, ci]
    assert not np.allclose(inj, w)


def test_dataset_is_reproducible(bootstrapped):
    a = data_mod.build_dataset(length=20, stride=10, seed=5, max_wells=30)
    b = data_mod.build_dataset(length=20, stride=10, seed=5, max_wells=30)
    assert np.array_equal(a.y_test, b.y_test)
    assert np.allclose(a.X_test, b.X_test)


# --- model + training (needs torch) -------------------------------------------
@needs_torch
def test_model_forward_shape():
    m = model_mod.build_model(n_channels=9, hidden=16, latent=8)
    import torch
    x = torch.randn(4, 30, 9)
    out = m(x)
    assert out.shape == (4, 30, 9)


@needs_torch
def test_reconstruction_error_shape():
    m = model_mod.build_model(n_channels=9, hidden=16, latent=8)
    X = np.random.randn(7, 30, 9).astype("float32")
    errs = model_mod.reconstruction_error(m, X)
    assert errs.shape == (7,)
    assert (errs >= 0).all()


@needs_torch
def test_train_and_eval_smoke(bootstrapped, monkeypatch, tmp_path):
    """A tiny train run saves a model; eval scores both detectors.

    Redirect artifacts to tmp so the committed model isn't clobbered.
    """
    from dl import train as train_mod
    from dl import evaluate as eval_mod
    monkeypatch.setattr(train_mod, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(eval_mod, "ARTIFACTS", tmp_path)

    report = train_mod.train(length=20, stride=10, hidden=16, latent=8,
                             epochs=2, max_wells=30, log=lambda *_a, **_k: None)
    assert report["epochs_run"] >= 1
    assert (tmp_path / "autoencoder.pt").exists()

    res = eval_mod.evaluate(log=lambda *_a, **_k: None)
    for det in ("autoencoder", "robust_z_baseline"):
        assert 0.0 <= res[det]["roc_auc"] <= 1.0
        assert 0.0 <= res[det]["pr_auc"] <= 1.0
