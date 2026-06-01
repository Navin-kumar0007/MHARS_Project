"""
MHARS — Benchmarks: Evaluation Metrics Framework
=================================================
Comprehensive evaluation metrics for publishable benchmarking against
CMAPSS and other predictive maintenance datasets.

Metrics implemented:
  - Regression: RMSE, MAE, MAPE, R²
  - Anomaly Detection: Precision, Recall, F1, AUC-ROC, AUC-PR
  - RUL: NASA Scoring Function (asymmetric), Timeliness
  - Conformal: Coverage Probability, Average Interval Width
  - RL: Energy Efficiency, Safety Violation Rate
"""

import numpy as np
from typing import Dict, Optional, Tuple

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


# ── Regression Metrics ─────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Avoids division by zero."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all regression metrics at once."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }


# ── Anomaly Detection Metrics ─────────────────────────────────────────────────

def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Returns (TP, FP, TN, FN)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP)."""
    tp, fp, _, _ = _binary_confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN)."""
    tp, _, _, fn = _binary_confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 200) -> float:
    """
    Area Under ROC Curve. 
    Implements the trapezoidal rule over threshold sweep (no sklearn dependency).
    
    Args:
        y_true: Binary labels (0/1)
        y_scores: Continuous anomaly scores
        n_thresholds: Number of threshold steps for the curve
    """
    thresholds = np.linspace(y_scores.min() - 1e-4, y_scores.max() + 1e-4, n_thresholds)
    tpr_list, fpr_list = [], []
    
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    
    if total_pos == 0 or total_neg == 0:
        return 0.5  # Undefined, return chance
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)
    
    # Sort by FPR ascending for proper AUC computation
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sorted_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sorted_idx]
    tpr_sorted = tpr_arr[sorted_idx]
    
    return float(_trapz(tpr_sorted, fpr_sorted))


def auc_pr(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 200) -> float:
    """
    Area Under Precision-Recall Curve.
    """
    thresholds = np.linspace(y_scores.min() - 1e-4, y_scores.max() + 1e-4, n_thresholds)
    prec_list, rec_list = [], []
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(p)
        rec_list.append(r)
    
    rec_arr = np.array(rec_list)
    prec_arr = np.array(prec_list)
    sorted_idx = np.argsort(rec_arr)
    rec_sorted = rec_arr[sorted_idx]
    prec_sorted = prec_arr[sorted_idx]
    
    return float(_trapz(prec_sorted, rec_sorted))


def anomaly_report(y_true: np.ndarray, y_pred_binary: np.ndarray,
                   y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute all anomaly detection metrics at once."""
    report = {
        "precision": precision(y_true, y_pred_binary),
        "recall": recall(y_true, y_pred_binary),
        "f1_score": f1_score(y_true, y_pred_binary),
    }
    if y_scores is not None:
        report["auc_roc"] = auc_roc(y_true, y_scores)
        report["auc_pr"] = auc_pr(y_true, y_scores)
    return report


# ── RUL Metrics ────────────────────────────────────────────────────────────────

def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA's asymmetric scoring function for RUL prediction.
    Late predictions are penalized more heavily than early ones.
    
    S = sum_i(
        exp(-d_i / 13) - 1   if d_i < 0  (early prediction)
        exp( d_i / 10) - 1   if d_i >= 0  (late prediction)
    )
    where d_i = y_pred_i - y_true_i
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(scores))


def rul_timeliness(y_true: np.ndarray, y_pred: np.ndarray,
                   early_window: float = 10.0) -> float:
    """
    Fraction of predictions that are within an acceptable early-warning window.
    A good model predicts failure slightly early (within `early_window` cycles).
    
    Returns a score [0, 1] — fraction of predictions in the sweet spot.
    """
    d = y_pred - y_true
    timely = np.sum((-early_window <= d) & (d <= 0))
    return float(timely / len(d)) if len(d) > 0 else 0.0


def rul_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all RUL-specific metrics."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_scoring_function(y_true, y_pred),
        "timeliness_10": rul_timeliness(y_true, y_pred, early_window=10),
        "timeliness_20": rul_timeliness(y_true, y_pred, early_window=20),
        "r_squared": r_squared(y_true, y_pred),
    }


# ── Conformal Prediction Metrics ──────────────────────────────────────────────

def coverage_probability(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Fraction of true values that fall within the predicted interval [lower, upper].
    Should be >= target coverage (e.g., 0.90 for 90% intervals).
    """
    covered = np.sum((y_true >= lower) & (y_true <= upper))
    return float(covered / len(y_true)) if len(y_true) > 0 else 0.0


def average_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of prediction intervals. Smaller is better (more precise)."""
    return float(np.mean(upper - lower))


def conformal_report(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
    """Compute all conformal prediction metrics."""
    return {
        "coverage_probability": coverage_probability(y_true, lower, upper),
        "average_interval_width": average_interval_width(lower, upper),
        "median_interval_width": float(np.median(upper - lower)),
    }


# ── RL / Control Metrics ──────────────────────────────────────────────────────

def safety_violation_rate(temps: np.ndarray, safe_max: float) -> float:
    """Fraction of timesteps where temperature exceeded safe_max."""
    return float(np.mean(temps > safe_max))


def energy_efficiency(fan_speeds: np.ndarray) -> float:
    """
    Energy metric: average fan power consumption (lower = more efficient).
    Fan power scales quadratically with speed (cube law for real fans, 
    but square is a common simplification).
    """
    return float(np.mean(fan_speeds ** 2))


def rl_report(temps: np.ndarray, fan_speeds: np.ndarray,
              safe_max: float, critical: float) -> Dict[str, float]:
    """Compute all RL control metrics."""
    return {
        "safety_violation_rate": safety_violation_rate(temps, safe_max),
        "critical_breach_rate": safety_violation_rate(temps, critical),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "energy_efficiency": energy_efficiency(fan_speeds),
        "mean_fan_speed": float(np.mean(fan_speeds)),
    }


# ── Full Benchmark Runner ─────────────────────────────────────────────────────

def run_full_benchmark(
    lstm_y_true: np.ndarray = None, lstm_y_pred: np.ndarray = None,
    anomaly_y_true: np.ndarray = None, anomaly_y_pred: np.ndarray = None,
    anomaly_scores: np.ndarray = None,
    rul_y_true: np.ndarray = None, rul_y_pred: np.ndarray = None,
    conformal_y_true: np.ndarray = None,
    conformal_lower: np.ndarray = None, conformal_upper: np.ndarray = None,
    rl_temps: np.ndarray = None, rl_fan_speeds: np.ndarray = None,
    safe_max: float = 85.0, critical: float = 100.0,
) -> Dict[str, Dict[str, float]]:
    """
    Run all available benchmarks given the data provided.
    Only computes metrics for which data is supplied.
    """
    results = {}
    
    if lstm_y_true is not None and lstm_y_pred is not None:
        results["lstm_regression"] = regression_report(lstm_y_true, lstm_y_pred)
    
    if anomaly_y_true is not None and anomaly_y_pred is not None:
        results["anomaly_detection"] = anomaly_report(
            anomaly_y_true, anomaly_y_pred, anomaly_scores
        )
    
    if rul_y_true is not None and rul_y_pred is not None:
        results["rul_prediction"] = rul_report(rul_y_true, rul_y_pred)
    
    if conformal_y_true is not None and conformal_lower is not None and conformal_upper is not None:
        results["conformal_prediction"] = conformal_report(
            conformal_y_true, conformal_lower, conformal_upper
        )
    
    if rl_temps is not None and rl_fan_speeds is not None:
        results["rl_control"] = rl_report(rl_temps, rl_fan_speeds, safe_max, critical)
    
    return results


def print_benchmark_report(results: Dict[str, Dict[str, float]]):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 65)
    print("  MHARS BENCHMARK REPORT")
    print("=" * 65)
    
    for section, metrics in results.items():
        print(f"\n── {section.replace('_', ' ').title()} {'─' * (45 - len(section))}")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name:30s}: {value:>10.4f}")
            else:
                print(f"  {name:30s}: {value}")
    
    print("\n" + "=" * 65)
