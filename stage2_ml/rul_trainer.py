"""
MHARS — Phase 2: RUL Training Pipeline
========================================
Trains the RULPredictor model on CMAPSS FD001 data with:
  - Piece-wise linear RUL labels (capped at 125 cycles)
  - Huber loss (robust to outliers)
  - CosineAnnealing learning rate scheduler
  - Early stopping (patience=10)
  - NASA asymmetric scoring function evaluation

Usage:
    python -m stage2_ml.rul_trainer
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch required: pip install torch")

from mhars.config import Config
from mhars.models import RULPredictor


def nasa_scoring_function(y_true, y_pred):
    """
    NASA asymmetric scoring function for CMAPSS RUL prediction.
    
    Late predictions (predicting RUL > actual) are penalized much more
    than early predictions, because in real systems, failing to warn
    early enough has catastrophic consequences.
    
    S = Σ (exp(-d/13) - 1)  if d < 0  (early prediction)
      + Σ (exp(d/10) - 1)   if d >= 0 (late prediction)
    
    where d = predicted_rul - actual_rul
    """
    d = y_pred - y_true
    score = 0.0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13.0) - 1
        else:
            score += np.exp(di / 10.0) - 1
    return score


def evaluate_rul(model, test_loader, device="cpu"):
    """
    Evaluate RUL model with RMSE, MAE, and NASA scoring function.
    
    Returns dict with metrics.
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            all_true.extend(yb.numpy())
            all_pred.extend(preds)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    nasa_score = nasa_scoring_function(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "nasa_score": float(nasa_score),
        "n_samples": len(y_true),
        "mean_pred": float(np.mean(y_pred)),
        "mean_true": float(np.mean(y_true)),
    }


def train_rul(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    device="cpu",
    verbose=True,
):
    """
    Train the RUL predictor with Huber loss and CosineAnnealing.
    
    Returns the trained model and training history.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=20.0)  # Huber loss: robust to outlier RUL values

    best_val_rmse = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_rmse": [], "val_mae": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_train += len(xb)
        train_loss /= n_train

        # Validation
        metrics = evaluate_rul(model, val_loader, device)
        val_rmse = metrics["rmse"]
        val_mae = metrics["mae"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — "
                  f"loss={train_loss:.4f}, val_rmse={val_rmse:.2f}, val_mae={val_mae:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def run_training(save_path=None, verbose=True):
    """
    End-to-end RUL training pipeline:
    1. Load CMAPSS data
    2. Create RUL windows with piece-wise labels
    3. Engine-wise train/val/test split
    4. Train RUL predictor
    5. Evaluate on test set
    6. Save model
    """
    if save_path is None:
        save_path = Config.RUL_MODEL_V2

    if verbose:
        print("── RUL Predictor Training Pipeline ────────────────────────────")

    # Load and preprocess
    from stage1_simulation.load_cmapss import (
        load_cmapss, preprocess_multivariate, make_rul_windows
    )
    df = load_cmapss()
    df = preprocess_multivariate(df)

    # Create windows
    X, y_rul, unit_ids = make_rul_windows(
        df, window=Config.LSTM_WINDOW, rul_cap=Config.RUL_MAX_CYCLES
    )
    if verbose:
        print(f"  Dataset: {len(X)} windows, RUL range [{y_rul.min():.0f}, {y_rul.max():.0f}]")

    # Engine-wise split (no data leakage)
    unique_units = np.unique(unit_ids)
    rng = np.random.RandomState(Config.SEED)
    rng.shuffle(unique_units)
    n_train = int(len(unique_units) * 0.7)
    n_val = int(len(unique_units) * 0.15)

    train_units = set(unique_units[:n_train])
    val_units = set(unique_units[n_train:n_train + n_val])
    test_units = set(unique_units[n_train + n_val:])

    train_mask = np.isin(unit_ids, list(train_units))
    val_mask = np.isin(unit_ids, list(val_units))
    test_mask = np.isin(unit_ids, list(test_units))

    X_train, y_train = X[train_mask], y_rul[train_mask]
    X_val, y_val = X[val_mask], y_rul[val_mask]
    X_test, y_test = X[test_mask], y_rul[test_mask]

    if verbose:
        print(f"  Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # DataLoaders
    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=64, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=128,
    )
    test_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=128,
    )

    # Initialize model
    model = RULPredictor(
        input_size=Config.LSTM_INPUT_SIZE_V2,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: RULPredictor ({n_params:,} parameters)")

    # Train
    model, history = train_rul(
        model, train_dl, val_dl,
        epochs=50, lr=1e-3, verbose=verbose,
    )

    # Evaluate on test set
    test_metrics = evaluate_rul(model, test_dl)
    if verbose:
        print(f"\n  ── Test Results ──")
        print(f"  RMSE:        {test_metrics['rmse']:.2f} cycles")
        print(f"  MAE:         {test_metrics['mae']:.2f} cycles")
        print(f"  NASA Score:  {test_metrics['nasa_score']:.0f}")
        print(f"  Test size:   {test_metrics['n_samples']} windows")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    if verbose:
        print(f"\n  ✓  RUL model saved → {save_path}")

    # Save metadata
    meta_path = save_path.replace(".pt", "_meta.json")
    meta = {
        "input_size": Config.LSTM_INPUT_SIZE_V2,
        "hidden_size": 64,
        "num_layers": 2,
        "rul_max_cycles": Config.RUL_MAX_CYCLES,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_nasa_score": test_metrics["nasa_score"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    model, metrics = run_training(verbose=True)
