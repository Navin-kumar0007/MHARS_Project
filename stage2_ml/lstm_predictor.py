"""
MHARS — Stage 2: LSTM Thermal Predictor
=========================================
12-step sliding window LSTM that predicts the next temperature
reading given the last 12. Validated by Kalla & Smith (2024)
who used this exact window size for motor temperature prediction.

Architecture:
  Input  → LSTM(hidden=64) → Dropout(0.2) → Linear(64→1)
  Loss   → MSE
  Target → RMSE < 2°C on held-out data
"""

import numpy as np
import os, sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stage1_simulation.load_cmapss import load_cmapss, preprocess, make_lstm_windows


# ── Model ─────────────────────────────────────────────────────────────────────
class ThermalLSTM(nn.Module):
    """
    Single-layer LSTM followed by a linear head.
    Kept deliberately simple — complexity can be added in Stage 5
    when we fine-tune per machine type.
    """
    def __init__(self, input_size=1, hidden_size=128, dropout=0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len=12, features=1)
        out, _ = self.lstm(x)
        out     = self.drop(out[:, -1, :])   # take last timestep only
        return self.linear(out).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.001,
    val_split: float = 0.2,
    model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm.pt")
):
    # Train/val split (keep temporal order — do NOT shuffle)
    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Convert to tensors — shape (N, 12, 1)
    def to_tensor(arr):
        return torch.FloatTensor(arr).unsqueeze(-1)

    train_ds = TensorDataset(to_tensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   torch.FloatTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = ThermalLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n── LSTM Training ─────────────────────────────────────────────")
    print(f"  Train windows: {len(X_train):,}  |  Val windows: {len(X_val):,}")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print()

    best_val_rmse = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_rmse = (train_loss / len(X_train)) ** 0.5

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_rmse = (val_loss / len(X_val)) ** 0.5

        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train RMSE: {train_rmse:.4f}  val RMSE: {val_rmse:.4f}"
                  + (" ← best" if val_rmse == best_val_rmse else ""))

    print(f"\n  Best val RMSE: {best_val_rmse:.4f}")

    # Note: RMSE is on normalized data [0,1]. Multiply by temp range (~85°C)
    # to get real-world °C error for reporting.
    rmse_celsius_approx = best_val_rmse * 85
    print(f"  Approx real-world RMSE: ~{rmse_celsius_approx:.1f}°C")
    status = "✓" if rmse_celsius_approx < 2.0 else "⚠  above 2°C target — try hidden_size=128"
    print(f"  {status}")

    return model, best_val_rmse, history


def load_model(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm.pt")):
    model = ThermalLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict_next(model, window_12: np.ndarray) -> float:
    """
    Given 12 consecutive normalized temperature readings,
    return the predicted next value (normalized) without uncertainty.
    """
    x = torch.FloatTensor(window_12).unsqueeze(0).unsqueeze(-1)  # (1,12,1)
    with torch.no_grad():
        return model(x).item()


def predict_mc(model, window_12: np.ndarray, num_samples: int = 30) -> dict:
    """
    Monte Carlo Dropout prediction.
    Runs the forward pass multiple times with dropout enabled
    to estimate epistemic uncertainty.
    """
    x = torch.FloatTensor(window_12).unsqueeze(0).unsqueeze(-1)  # (1,12,1)
    
    # Force dropout layers to be active
    model.train() 
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            predictions.append(model(x).item())
            
    # Restore to eval mode
    model.eval()
    
    preds = np.array(predictions)
    return {
        "mean": float(np.mean(preds)),
        "variance": float(np.var(preds)),
        "std": float(np.std(preds))
    }


def run_training(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm.pt")):
    from stage1_simulation.load_cmapss import load_cmapss, preprocess, make_lstm_windows
    from mhars.config import Config
    import pandas as pd
    
    df = load_cmapss()
    df = preprocess(df)
    X, y, _ = make_lstm_windows(df, window=12)
    model, best_rmse, history = train(X, y, model_path=model_path)
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(Config.RESULTS_DIR, "lstm_training.csv"), index=False)
    
    print(f"[PASS] LSTM trained — saved to {model_path}\n")
    return model, best_rmse, history


if __name__ == "__main__":
    run_training()


# ── V2 Enhanced Training (Phase 1 Deep Analysis) ─────────────────────────────

def train_v2(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    val_split: float = 0.2,
    patience: int = 15,
    model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm_v2.pt"),
):
    """
    Train the enhanced BiLSTM+Attention model on multivariate sensor data.

    Improvements over train():
      1. Multivariate input: X shape (N, 12, 5) instead of (N, 12)
      2. CosineAnnealingLR scheduler for better convergence
      3. Early stopping with patience to prevent overfitting
      4. Gradient clipping for training stability
      5. Reports MAE alongside RMSE
    """
    from mhars.models import ThermalLSTMv2

    # Train/val split (temporal order)
    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # X is already (N, 12, 5) from make_lstm_windows_multivariate
    X_train_t = torch.FloatTensor(X_train)
    X_val_t   = torch.FloatTensor(X_val)
    y_train_t = torch.FloatTensor(y_train)
    y_val_t   = torch.FloatTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t,   y_val_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    input_size = X.shape[2]  # number of sensor channels
    model     = ThermalLSTMv2(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    print(f"\n── BiLSTM+Attention V2 Training ────────────────────────────────")
    print(f"  Train windows: {len(X_train):,}  |  Val windows: {len(X_val):,}")
    print(f"  Input shape:   (batch, {X.shape[1]}, {X.shape[2]})  [multivariate]")
    print(f"  Epochs: {epochs} (early stop patience={patience})  |  LR: {lr}")
    print(f"  Architecture: BiLSTM(hidden=128, layers=2, bidirectional) + Attention")
    print()

    best_val_rmse = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_rmse = (train_loss / len(X_train)) ** 0.5

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                val_mae  += torch.abs(pred - yb).sum().item()
        val_rmse = (val_loss / len(X_val)) ** 0.5
        val_mae  = val_mae / len(X_val)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch,
            "train_rmse": round(train_rmse, 6),
            "val_rmse": round(val_rmse, 6),
            "val_mae": round(val_mae, 6),
            "lr": round(current_lr, 8),
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improve = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or no_improve == 0:
            marker = " ← best" if no_improve == 0 else ""
            print(f"  Epoch {epoch:3d}/{epochs}  train: {train_rmse:.4f}  "
                  f"val RMSE: {val_rmse:.4f}  MAE: {val_mae:.4f}  "
                  f"lr: {current_lr:.2e}{marker}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\n  Best val RMSE: {best_val_rmse:.6f}")
    rmse_celsius_approx = best_val_rmse * 85
    print(f"  Approx real-world RMSE: ~{rmse_celsius_approx:.1f}°C")
    print(f"  Model saved → {model_path}")

    return model, best_val_rmse, history


def run_training_v2(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm_v2.pt")):
    """Train the enhanced BiLSTM+Attention model on multivariate CMAPSS data."""
    from stage1_simulation.load_cmapss import (
        load_cmapss, preprocess_multivariate, make_lstm_windows_multivariate
    )
    from mhars.config import Config
    import pandas as pd

    df = load_cmapss()
    df = preprocess_multivariate(df)
    X, y, unit_ids = make_lstm_windows_multivariate(df, window=12)

    print(f"  Multivariate windows: {X.shape}  (N, window, sensors)")

    model, best_rmse, history = train_v2(X, y, model_path=model_path)

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    import pandas as pd
    pd.DataFrame(history).to_csv(
        os.path.join(Config.RESULTS_DIR, "lstm_v2_training.csv"), index=False
    )

    print(f"[PASS] BiLSTM+Attention V2 trained — saved to {model_path}\n")
    return model, best_rmse, history