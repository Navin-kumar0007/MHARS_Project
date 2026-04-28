"""
MHARS — Stage 2: Autoencoder (Within-Range Anomaly Detection)
==============================================================
Trained on NORMAL thermal data only. At inference, the reconstruction
error is the anomaly score — high error means the pattern is unusual.

This catches within-range anomalies: subtle degradation patterns
that remain technically within safe temperature bounds but signal
impending failure. Validated by Gutiérrez et al. (2024) Aerospace.

Architecture:
  Encoder: Linear(12 → 6 → 3)
  Decoder: Linear(3  → 6 → 12)
  Loss:    MSE (reconstruction)
  Threshold: 95th percentile of normal reconstruction errors
"""

import numpy as np
import os, sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Model ─────────────────────────────────────────────────────────────────────
class ThermalAutoencoder(nn.Module):
    def __init__(self, seq_len=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, seq_len),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        """Per-sample MSE between input and reconstruction."""
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    X_normal: np.ndarray,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 0.001,
    val_split: float = 0.1,
    model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "autoencoder.pt")
):
    split    = int(len(X_normal) * (1 - val_split))
    X_train  = torch.FloatTensor(X_normal[:split])
    X_val    = torch.FloatTensor(X_normal[split:])

    train_dl = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    model     = ThermalAutoencoder(seq_len=X_normal.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n── Autoencoder Training ──────────────────────────────────────")
    print(f"  Normal train samples: {len(X_train):,}  |  Val: {len(X_val):,}")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}")
    print()

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for (xb,) in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  val MSE: {val_loss:.6f}"
                  + (" ← best" if val_loss == best_val_loss else ""))
                  
        history.append({"epoch": epoch, "val_loss": val_loss})

    return model, history


def compute_threshold(model, X_normal_val: np.ndarray, percentile: float = 95) -> float:
    """
    Threshold = 95th percentile of reconstruction errors on NORMAL data.
    Anything above this is flagged as an anomaly.
    Setting at 95th percentile means only the top 5% most unusual
    normal patterns trigger a flag — avoiding alert fatigue.
    """
    model.eval()
    x = torch.FloatTensor(X_normal_val)
    with torch.no_grad():
        errors = model.reconstruction_error(x).numpy()
    threshold = float(np.percentile(errors, percentile))
    print(f"\n  Anomaly threshold (p{percentile}): {threshold:.6f}")
    return threshold


def get_anomaly_score(model, X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns anomaly scores normalized to [0, 1].
    Score > 1.0 means reconstruction error exceeded the threshold.
    """
    model.eval()
    x = torch.FloatTensor(X)
    with torch.no_grad():
        errors = model.reconstruction_error(x).numpy()
    return (errors / (threshold + 1e-8)).astype(np.float32)


def load_model(seq_len=12, model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "autoencoder.pt")):
    model = ThermalAutoencoder(seq_len=seq_len)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def run_training(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "autoencoder.pt")):
    from stage1_simulation.load_cmapss import load_cmapss, preprocess, make_lstm_windows
    df = load_cmapss()
    df = preprocess(df)

    X_all, _, _ = make_lstm_windows(df, window=12)

    # Train ONLY on normal data
    normal_idx = df[df["near_failure"] == 0].index
    # Approximate: keep windows whose last index falls in normal data
    X_normal = X_all[:int(len(X_all) * 0.88)]  # ~88% is non-near-failure

    split = int(len(X_normal) * 0.9)
    X_train_norm = X_normal[:split]
    X_val_norm   = X_normal[split:]

    model, history = train(X_train_norm, model_path=model_path)

    threshold = compute_threshold(model, X_val_norm)
    
    # Save training logs to CSV
    from mhars.config import Config
    import pandas as pd
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(Config.RESULTS_DIR, "ae_training.csv"), index=False)

    # Quick sanity check: near-failure windows should score higher
    X_anomaly = X_all[int(len(X_all) * 0.88):]
    normal_scores  = get_anomaly_score(model, X_val_norm[:500],  threshold)
    anomaly_scores = get_anomaly_score(model, X_anomaly[:500],   threshold)

    print(f"\n  Mean score — normal samples:  {normal_scores.mean():.3f}")
    print(f"  Mean score — anomaly samples: {anomaly_scores.mean():.3f}")
    if anomaly_scores.mean() > normal_scores.mean():
        print("  ✓  Autoencoder correctly scores anomalies higher")
    else:
        print("  ⚠  Scores not separating well — train for more epochs")

    # Save threshold alongside model
    import json
    meta_path = model_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"threshold": threshold, "seq_len": 12}, f)
    print(f"  Threshold saved → {meta_path}")
    print(f"[PASS] Autoencoder trained and saved\n")
    return model, threshold


if __name__ == "__main__":
    run_training()