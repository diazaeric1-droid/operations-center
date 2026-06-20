"""LSTM autoencoder (PyTorch). Optional dep — imports cleanly without torch.

Encoder LSTM compresses a (L, C) SCADA window to a small latent vector; the
decoder LSTM reconstructs the full window from it. Trained only on healthy
windows, the network learns normal multivariate dynamics, so an abnormal window
reconstructs poorly — the per-window reconstruction MSE is the anomaly score.
"""
from __future__ import annotations

import numpy as np

try:                       # torch is optional; absence must not break import
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:        # pragma: no cover
    _TORCH_AVAILABLE = False


def torch_available() -> bool:
    return _TORCH_AVAILABLE


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install the DL extras:\n"
            "    pip install -r requirements-dl.txt")


if _TORCH_AVAILABLE:

    class LSTMAutoencoder(nn.Module):
        """Sequence-to-sequence LSTM autoencoder for fixed-length windows."""

        def __init__(self, n_channels: int, hidden: int = 64,
                     latent: int = 16, num_layers: int = 1,
                     dropout: float = 0.0):
            super().__init__()
            self.n_channels = n_channels
            self.encoder = nn.LSTM(n_channels, hidden, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(hidden, hidden, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
            self.out = nn.Linear(hidden, n_channels)

        def forward(self, x):                      # x: (B, L, C)
            L = x.shape[1]
            _, (h, _) = self.encoder(x)            # h: (layers, B, hidden)
            z = self.to_latent(h[-1])              # (B, latent)
            seed = self.from_latent(z).unsqueeze(1).repeat(1, L, 1)  # (B, L, hidden)
            dec, _ = self.decoder(seed)            # (B, L, hidden)
            return self.out(dec)                   # (B, L, C)


def build_model(n_channels: int, hidden: int = 64, latent: int = 16,
                num_layers: int = 1, dropout: float = 0.0):
    _require_torch()
    return LSTMAutoencoder(n_channels, hidden=hidden, latent=latent,
                           num_layers=num_layers, dropout=dropout)


def reconstruction_error(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """Per-window mean-squared reconstruction error -> (n,) anomaly scores."""
    _require_torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32)
            recon = model(xb)
            mse = ((recon - xb) ** 2).mean(dim=(1, 2))   # (B,)
            errs.append(mse.cpu().numpy())
    return np.concatenate(errs) if errs else np.empty((0,))
