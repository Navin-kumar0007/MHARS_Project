"""
MHARS — Phase 1 Comprehensive Evaluation Benchmark Suite
=========================================================
Evaluates LSTM, Autoencoder, Conformal Prediction, and RUL models
with proper metrics: RMSE, MAE, MAPE, F1, AUC-ROC, Coverage, and
the NASA asymmetric scoring function.

Supports side-by-side V1 vs V2 comparison.

Usage:
    python -m benchmarks.evaluate              # Full evaluation
    python -m benchmarks.evaluate --v2-only    # V2 models only
    python -m benchmarks.evaluate --compare    # Side-by-side comparison
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Metric Functions ──────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignoring near-zero true values)."""
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA CMAPSS asymmetric scoring function.
    Late predictions (underestimating RUL) are penalized more severely
    than early predictions (overestimating RUL).

    S = Σ exp(-d/13) - 1    if d < 0 (early prediction)
      + Σ exp(d/10) - 1     if d >= 0 (late prediction)

    where d = y_pred - y_true
    """
    d = y_pred - y_true
    score = 0.0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13.0) - 1
        else:
            score += np.exp(di / 10.0) - 1
    return float(score)


def anomaly_metrics(y_true: np.ndarray, scores: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.

    Args:
        y_true:    binary labels (1 = anomaly, 0 = normal)
        scores:    anomaly scores (higher = more anomalous)
        threshold: score threshold for binary classification

    Returns:
        dict with precision, recall, f1, auc_roc
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    y_pred = (scores >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = 0.0  # only one class present

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "threshold": threshold,
    }


def conformal_coverage(y_true: np.ndarray, lower: np.ndarray,
                       upper: np.ndarray) -> Dict[str, float]:
    """
    Evaluate conformal prediction intervals.

    Returns:
        dict with coverage (should be >= target), mean_width, median_width
    """
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    widths  = upper - lower

    return {
        "coverage": round(float(covered.mean()), 4),
        "mean_width": round(float(widths.mean()), 4),
        "median_width": round(float(np.median(widths)), 4),
        "n_samples": len(y_true),
    }


# ── Model Evaluators ──────────────────────────────────────────────────────────

def evaluate_lstm_v1(X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate the original univariate LSTM."""
    import torch
    from mhars.models import ThermalLSTM
    from mhars.config import Config

    if not os.path.exists(Config.LSTM):
        return {"error": "lstm.pt not found"}

    checkpoint = torch.load(Config.LSTM, map_location="cpu")
    hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
    model = ThermalLSTM(hidden_size=hidden_size)
    model.load_state_dict(checkpoint)
    model.eval()

    # X_test shape: (N, 12) → need (N, 12, 1)
    x = torch.FloatTensor(X_test).unsqueeze(-1)
    with torch.no_grad():
        preds = model(x).numpy()

    return {
        "model": "ThermalLSTM (V1)",
        "architecture": f"LSTM(1→{hidden_size}→1)",
        "rmse": round(rmse(y_test, preds), 6),
        "mae": round(mae(y_test, preds), 6),
        "mape": round(mape(y_test, preds), 4),
        "rmse_celsius": round(rmse(y_test, preds) * 85, 2),
        "n_test": len(y_test),
    }


def evaluate_lstm_v2(X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate the BiLSTM+Attention multivariate model."""
    import torch
    from mhars.models import ThermalLSTMv2
    from mhars.config import Config

    if not os.path.exists(Config.LSTM_V2):
        return {"error": "lstm_v2.pt not found"}

    checkpoint = torch.load(Config.LSTM_V2, map_location="cpu")
    hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
    input_size = checkpoint["lstm.weight_ih_l0"].shape[1]
    model = ThermalLSTMv2(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(checkpoint)
    model.eval()

    # X_test shape: (N, 12, 5)
    x = torch.FloatTensor(X_test)
    with torch.no_grad():
        preds = model(x).numpy()

    return {
        "model": "ThermalLSTMv2 (BiLSTM+Attention)",
        "architecture": f"BiLSTM({input_size}→{hidden_size}×2→Attention→1)",
        "rmse": round(rmse(y_test, preds), 6),
        "mae": round(mae(y_test, preds), 6),
        "mape": round(mape(y_test, preds), 4),
        "rmse_celsius": round(rmse(y_test, preds) * 85, 2),
        "n_test": len(y_test),
    }


def evaluate_anomaly_v1(X_normal: np.ndarray, X_anomaly: np.ndarray) -> Dict:
    """Evaluate the original linear Autoencoder for anomaly detection."""
    import torch
    from mhars.models import ThermalAutoencoder
    from mhars.config import Config

    if not os.path.exists(Config.AUTOENCODER):
        return {"error": "autoencoder.pt not found"}

    model = ThermalAutoencoder()
    model.load_state_dict(torch.load(Config.AUTOENCODER, map_location="cpu"))
    model.eval()

    threshold = 0.05
    if os.path.exists(Config.AUTOENCODER_META):
        with open(Config.AUTOENCODER_META) as f:
            threshold = json.load(f).get("threshold", 0.05)

    # Score both sets
    with torch.no_grad():
        normal_errors = model.reconstruction_error(torch.FloatTensor(X_normal)).numpy()
        anomaly_errors = model.reconstruction_error(torch.FloatTensor(X_anomaly)).numpy()

    normal_scores = normal_errors / (threshold + 1e-8)
    anomaly_scores = anomaly_errors / (threshold + 1e-8)

    # Build labels and scores
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])

    metrics = anomaly_metrics(all_labels, all_scores, threshold=1.0)
    metrics["model"] = "ThermalAutoencoder (V1, Linear)"
    metrics["mean_normal_score"] = round(float(normal_scores.mean()), 4)
    metrics["mean_anomaly_score"] = round(float(anomaly_scores.mean()), 4)
    metrics["separation_ratio"] = round(float(anomaly_scores.mean() / (normal_scores.mean() + 1e-8)), 2)
    return metrics


def evaluate_anomaly_v2(X_normal_3d: np.ndarray, X_anomaly_3d: np.ndarray) -> Dict:
    """Evaluate the LSTM-Autoencoder for anomaly detection."""
    import torch
    from mhars.models import ThermalAutoencoderLSTM
    from mhars.config import Config

    if not os.path.exists(Config.AUTOENCODER_V2):
        return {"error": "autoencoder_lstm_v2.pt not found"}

    meta = {}
    if os.path.exists(Config.AUTOENCODER_V2_META):
        with open(Config.AUTOENCODER_V2_META) as f:
            meta = json.load(f)

    input_size = meta.get("input_size", 5)
    hidden_size = meta.get("hidden_size", 32)
    threshold = meta.get("threshold", 0.05)

    model = ThermalAutoencoderLSTM(input_size=input_size, hidden_size=hidden_size, seq_len=12)
    model.load_state_dict(torch.load(Config.AUTOENCODER_V2, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        normal_errors = model.reconstruction_error(torch.FloatTensor(X_normal_3d)).numpy()
        anomaly_errors = model.reconstruction_error(torch.FloatTensor(X_anomaly_3d)).numpy()

    normal_scores = normal_errors / (threshold + 1e-8)
    anomaly_scores = anomaly_errors / (threshold + 1e-8)

    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])

    metrics = anomaly_metrics(all_labels, all_scores, threshold=1.0)
    metrics["model"] = "ThermalAutoencoderLSTM (V2, LSTM-AE)"
    metrics["mean_normal_score"] = round(float(normal_scores.mean()), 4)
    metrics["mean_anomaly_score"] = round(float(anomaly_scores.mean()), 4)
    metrics["separation_ratio"] = round(float(anomaly_scores.mean() / (normal_scores.mean() + 1e-8)), 2)
    return metrics


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_report(results: Dict, save_path: str = None) -> str:
    """Generate a markdown evaluation report."""
    lines = [
        "# MHARS Phase 1 — Evaluation Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # LSTM comparison
    if "lstm_v1" in results and "lstm_v2" in results:
        v1 = results["lstm_v1"]
        v2 = results["lstm_v2"]
        if "error" not in v1 and "error" not in v2:
            improvement = ((v1["rmse"] - v2["rmse"]) / v1["rmse"]) * 100
            lines.extend([
                "## LSTM Temperature Prediction — V1 vs V2",
                "",
                "| Metric | V1 (Univariate LSTM) | V2 (BiLSTM+Attention) | Improvement |",
                "|:-------|:--------------------:|:---------------------:|:-----------:|",
                f"| **RMSE** | {v1['rmse']:.6f} | {v2['rmse']:.6f} | **{improvement:+.1f}%** |",
                f"| MAE | {v1['mae']:.6f} | {v2['mae']:.6f} | |",
                f"| MAPE | {v1['mape']:.2f}% | {v2['mape']:.2f}% | |",
                f"| RMSE (°C) | {v1['rmse_celsius']:.2f}°C | {v2['rmse_celsius']:.2f}°C | |",
                f"| Test samples | {v1['n_test']:,} | {v2['n_test']:,} | |",
                "",
            ])

    # Anomaly detection comparison
    if "anomaly_v1" in results and "anomaly_v2" in results:
        v1 = results["anomaly_v1"]
        v2 = results["anomaly_v2"]
        if "error" not in v1 and "error" not in v2:
            lines.extend([
                "## Anomaly Detection — V1 vs V2",
                "",
                "| Metric | V1 (Linear AE) | V2 (LSTM-AE) |",
                "|:-------|:--------------:|:------------:|",
                f"| **F1 Score** | {v1['f1']:.4f} | {v2['f1']:.4f} |",
                f"| Precision | {v1['precision']:.4f} | {v2['precision']:.4f} |",
                f"| Recall | {v1['recall']:.4f} | {v2['recall']:.4f} |",
                f"| AUC-ROC | {v1['auc_roc']:.4f} | {v2['auc_roc']:.4f} |",
                f"| Separation ratio | {v1['separation_ratio']:.2f}× | {v2['separation_ratio']:.2f}× |",
                "",
            ])

    # Conformal prediction
    if "conformal" in results:
        cp = results["conformal"]
        if "error" not in cp:
            lines.extend([
                "## Conformal Prediction Intervals",
                "",
                f"| Metric | Value |",
                f"|:-------|:-----:|",
                f"| Coverage | {cp['coverage']:.1%} |",
                f"| Target coverage | {cp.get('target', 0.90):.0%} |",
                f"| Mean width | {cp['mean_width']:.4f} |",
                f"| Median width | {cp['median_width']:.4f} |",
                f"| N samples | {cp['n_samples']:,} |",
                "",
            ])

    report = "\n".join(lines)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        print(f"\n  Report saved → {save_path}")

    return report


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_evaluation(compare: bool = True):
    """Run the full evaluation suite."""
    from stage1_simulation.load_cmapss import (
        load_cmapss, preprocess, make_lstm_windows,
        preprocess_multivariate, make_lstm_windows_multivariate,
    )
    from mhars.config import Config

    print("═" * 70)
    print("  MHARS Phase 1 — Comprehensive Evaluation Benchmark")
    print("═" * 70)

    # Load and preprocess data
    print("\n── Loading CMAPSS data ──")
    df = load_cmapss()
    df_v1 = preprocess(df.copy())
    df_v2 = preprocess_multivariate(df.copy())

    results = {}

    # ── LSTM Evaluation ──
    print("\n── Evaluating LSTM models ──")

    # V1: univariate
    X_v1, y_v1, _ = make_lstm_windows(df_v1, window=12)
    split = int(len(X_v1) * 0.8)
    results["lstm_v1"] = evaluate_lstm_v1(X_v1[split:], y_v1[split:])
    if "error" not in results["lstm_v1"]:
        print(f"  V1 RMSE: {results['lstm_v1']['rmse']:.6f} ({results['lstm_v1']['rmse_celsius']:.2f}°C)")

    # V2: multivariate
    X_v2, y_v2, _ = make_lstm_windows_multivariate(df_v2, window=12)
    split2 = int(len(X_v2) * 0.8)
    results["lstm_v2"] = evaluate_lstm_v2(X_v2[split2:], y_v2[split2:])
    if "error" not in results["lstm_v2"]:
        print(f"  V2 RMSE: {results['lstm_v2']['rmse']:.6f} ({results['lstm_v2']['rmse_celsius']:.2f}°C)")

    # ── Anomaly Detection Evaluation ──
    print("\n── Evaluating Anomaly Detection ──")

    # V1: univariate AE
    n_normal_v1 = int(len(X_v1) * 0.88)
    X_normal_v1 = X_v1[int(n_normal_v1 * 0.9):n_normal_v1]  # validation normal
    X_anomaly_v1 = X_v1[n_normal_v1:]
    results["anomaly_v1"] = evaluate_anomaly_v1(
        X_normal_v1[:500], X_anomaly_v1[:500]
    )
    if "error" not in results["anomaly_v1"]:
        print(f"  V1 F1: {results['anomaly_v1']['f1']:.4f}  AUC: {results['anomaly_v1']['auc_roc']:.4f}")

    # V2: LSTM-AE multivariate
    n_normal_v2 = int(len(X_v2) * 0.88)
    X_normal_v2 = X_v2[int(n_normal_v2 * 0.9):n_normal_v2]
    X_anomaly_v2 = X_v2[n_normal_v2:]
    results["anomaly_v2"] = evaluate_anomaly_v2(
        X_normal_v2[:500], X_anomaly_v2[:500]
    )
    if "error" not in results["anomaly_v2"]:
        print(f"  V2 F1: {results['anomaly_v2']['f1']:.4f}  AUC: {results['anomaly_v2']['auc_roc']:.4f}")

    # ── Generate Report ──
    report_path = os.path.join(Config.RESULTS_DIR, "phase1_evaluation.md")
    report = generate_report(results, save_path=report_path)

    # Also save raw metrics as JSON
    json_path = os.path.join(Config.RESULTS_DIR, "phase1_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics JSON → {json_path}")

    print("\n" + "═" * 70)
    print("  Evaluation complete!")
    print("═" * 70)

    return results


if __name__ == "__main__":
    run_evaluation()
