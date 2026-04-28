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
    return the predicted next value (normalized).
    """
    x = torch.FloatTensor(window_12).unsqueeze(0).unsqueeze(-1)  # (1,12,1)
    with torch.no_grad():
        return model(x).item()


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