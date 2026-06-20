"""Train the LSTM autoencoder on healthy windows. `python -m dl.train`.

Saves to dl/artifacts/:
    autoencoder.pt        model weights + config + the train-fit scaler
    training_report.json  loss history, epochs, and train/inference timing
                          (the compute footprint — useful for the hardware
                          requirements an inference deployment has to plan for)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import model as model_mod

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def train(length: int = 30, stride: int = 5, hidden: int = 64, latent: int = 16,
          num_layers: int = 1, epochs: int = 60, batch: int = 128,
          lr: float = 1e-3, patience: int = 8, seed: int = 13,
          max_wells: int | None = None, log=print) -> dict:
    import torch
    torch.manual_seed(seed)

    ds = data_mod.build_dataset(length=length, stride=stride, seed=seed,
                                max_wells=max_wells)
    log(f"dataset: {ds.shape_str}")

    model = model_mod.build_model(len(ds.channels), hidden=hidden,
                                  latent=latent, num_layers=num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    Xtr = torch.tensor(ds.X_train, dtype=torch.float32)
    Xva = torch.tensor(ds.X_val, dtype=torch.float32)

    history, best_val, best_state, bad = [], float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tr_loss = 0.0
        for i in range(0, len(Xtr), batch):
            xb = Xtr[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(Xtr)

        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xva), Xva).item()
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val - 1e-6:
            best_val, best_state, bad = va_loss, {
                k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep == 1 or ep % 5 == 0 or bad >= patience:
            log(f"  epoch {ep:3d}  train {tr_loss:.5f}  val {va_loss:.5f}"
                f"  (best {best_val:.5f})")
        if bad >= patience:
            log(f"  early stop at epoch {ep} (no val improvement in {patience})")
            break
    train_secs = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # inference throughput (windows/sec) — the deployment-planning number
    t1 = time.time()
    _ = model_mod.reconstruction_error(model, ds.X_val)
    infer_secs = time.time() - t1
    throughput = len(ds.X_val) / infer_secs if infer_secs > 0 else float("nan")

    ARTIFACTS.mkdir(exist_ok=True)
    config = {"length": length, "stride": stride, "hidden": hidden,
              "latent": latent, "num_layers": num_layers, "channels": ds.channels}
    torch.save({"state_dict": model.state_dict(), "config": config,
                "mean": ds.mean, "std": ds.std}, ARTIFACTS / "autoencoder.pt")

    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "dataset": ds.shape_str,
        "n_parameters": int(n_params),
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "train_seconds": round(train_secs, 2),
        "inference_windows_per_sec": round(throughput, 1),
        "config": config,
        "history": history,
    }
    (ARTIFACTS / "training_report.json").write_text(json.dumps(report, indent=2))
    log(f"saved model ({n_params:,} params) + report to {ARTIFACTS}")
    log(f"train {train_secs:.1f}s · inference {throughput:,.0f} windows/s")
    return report


if __name__ == "__main__":
    train()
