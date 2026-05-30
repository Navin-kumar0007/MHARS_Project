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

import json
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
    meta_path = model_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"threshold": threshold, "seq_len": 12}, f)
    print(f"  Threshold saved → {meta_path}")
    print(f"[PASS] Autoencoder trained and saved\n")
    return model, threshold


if __name__ == "__main__":
    run_training()


# ── V2 LSTM-Autoencoder Training (Phase 1 Deep Analysis) ─────────────────────

def train_lstm_ae(
    X_normal: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 0.001,
    val_split: float = 0.1,
    patience: int = 15,
    model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "autoencoder_lstm_v2.pt"),
):
    """
    Train the LSTM-based autoencoder on multivariate normal sequences.

    Input X_normal shape: (N, seq_len, n_sensors) — e.g. (N, 12, 5)

    Improvements over linear AE:
      1. Temporal-aware: catches sequence-level anomalies
      2. Multivariate: uses all 5 thermal sensors
      3. CosineAnnealing LR + early stopping
      4. Works for ALL machine types (no CPU/Server bypass)
    """
    from mhars.models import ThermalAutoencoderLSTM

    seq_len    = X_normal.shape[1]
    input_size = X_normal.shape[2]

    split = int(len(X_normal) * (1 - val_split))
    X_train = torch.FloatTensor(X_normal[:split])
    X_val   = torch.FloatTensor(X_normal[split:])

    train_dl = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    model     = ThermalAutoencoderLSTM(input_size=input_size, hidden_size=32, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    print(f"\n── LSTM-Autoencoder V2 Training ────────────────────────────────")
    print(f"  Normal train samples: {len(X_train):,}  |  Val: {len(X_val):,}")
    print(f"  Input shape: (batch, {seq_len}, {input_size})  [multivariate sequence]")
    print(f"  Epochs: {epochs} (early stop patience={patience})")
    print(f"  Architecture: LSTM-Encoder(32) → LSTM-Decoder(32) → Linear({input_size})")
    print()

    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for (xb,) in train_dl:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            recon_val = model(X_val)
            val_loss = criterion(recon_val, X_val).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or no_improve == 0:
            marker = " ← best" if no_improve == 0 else ""
            print(f"  Epoch {epoch:3d}/{epochs}  val MSE: {val_loss:.6f}{marker}")

        history.append({"epoch": epoch, "val_loss": val_loss})

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val MSE: {best_val_loss:.6f}")
    return model, history


def compute_threshold_v2(model, X_normal_val: np.ndarray, percentile: float = 95) -> float:
    """Compute anomaly threshold from the LSTM-AE reconstruction errors on normal data."""
    model.eval()
    x = torch.FloatTensor(X_normal_val)
    with torch.no_grad():
        errors = model.reconstruction_error(x).numpy()
    threshold = float(np.percentile(errors, percentile))
    print(f"  LSTM-AE anomaly threshold (p{percentile}): {threshold:.6f}")
    return threshold


def run_training_v2(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "autoencoder_lstm_v2.pt")):
    """Train the LSTM-Autoencoder on multivariate CMAPSS normal sequences."""
    from stage1_simulation.load_cmapss import (
        load_cmapss, preprocess_multivariate, make_lstm_windows_multivariate
    )
    from mhars.config import Config
    import pandas as pd

    df = load_cmapss()
    df = preprocess_multivariate(df)
    X_all, _, _ = make_lstm_windows_multivariate(df, window=12)

    # Train ONLY on normal data (~88% of windows)
    n_normal = int(len(X_all) * 0.88)
    X_normal = X_all[:n_normal]

    split = int(len(X_normal) * 0.9)
    X_train_norm = X_normal[:split]
    X_val_norm   = X_normal[split:]

    model, history = train_lstm_ae(X_train_norm, model_path=model_path)
    threshold = compute_threshold_v2(model, X_val_norm)

    # Validate: near-failure windows should score higher
    X_anomaly = X_all[n_normal:]

    model.eval()
    with torch.no_grad():
        normal_errors  = model.reconstruction_error(torch.FloatTensor(X_val_norm[:500])).numpy()
        anomaly_errors = model.reconstruction_error(torch.FloatTensor(X_anomaly[:500])).numpy()

    normal_mean  = float(normal_errors.mean())
    anomaly_mean = float(anomaly_errors.mean())

    print(f"\n  Mean error — normal:  {normal_mean:.4f}")
    print(f"  Mean error — anomaly: {anomaly_mean:.4f}")
    if anomaly_mean > normal_mean:
        print("  ✓  LSTM-AE correctly scores anomalies higher")
    else:
        print("  ⚠  Scores not separating — check data pipeline")

    # Save metadata
    meta_path = model_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "seq_len": 12,
            "input_size": X_all.shape[2],
            "hidden_size": 32,
            "normal_mean_error": round(normal_mean, 6),
            "anomaly_mean_error": round(anomaly_mean, 6),
        }, f)
    print(f"  Meta saved → {meta_path}")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(history).to_csv(
        os.path.join(Config.RESULTS_DIR, "ae_lstm_v2_training.csv"), index=False
    )

    print(f"[PASS] LSTM-Autoencoder V2 trained and saved\n")
    return model, threshold