"""
MHARS — Conformal Prediction for Uncertainty Quantification
=============================================================
Provides statistically guaranteed prediction intervals for LSTM
temperature predictions. If the model says "72.5°C next", conformal
prediction adds "with 90% confidence, it will be between 70.1°C and 74.9°C".

Key properties:
  1. Distribution-free — no assumptions about error distribution
  2. Finite-sample guarantee — coverage holds even with small calibration sets
  3. Adaptive — online quantile update for streaming data

Two modes:
  - Static:   calibrate once on held-out data → fixed interval width
  - Adaptive: update quantile online as new residuals arrive (Gibbs & Candès, 2021)

Usage:
    from mhars.conformal import ConformalPredictor
    cp = ConformalPredictor(coverage=0.90)
    cp.calibrate(residuals_array)
    interval = cp.predict_interval(point_prediction)
    # interval = {"lower": 70.1, "upper": 74.9, "width": 4.8}

Reference:
  - Vovk et al. (2005) Algorithmic Learning in a Random World
  - Gibbs & Candès (2021) Adaptive Conformal Inference
"""

import json
import numpy as np
from typing import Dict, Optional
from collections import deque


class ConformalPredictor:
    """
    Split Conformal Predictor with optional online adaptation.

    Calibrated on absolute residuals |y_true - y_pred| from a held-out set.
    The prediction interval is: [pred - q, pred + q]
    where q is the (1-alpha) quantile of calibration residuals.
    """

    def __init__(self, coverage: float = 0.90, adaptive: bool = True,
                 adaptive_lr: float = 0.005, window_size: int = 200):
        """
        Args:
            coverage:     desired coverage probability (e.g. 0.90 for 90%)
            adaptive:     if True, update quantile online with new residuals
            adaptive_lr:  learning rate for online quantile update
            window_size:  rolling window for adaptive residuals
        """
        assert 0 < coverage < 1, f"Coverage must be in (0,1), got {coverage}"
        self.coverage = coverage
        self.alpha = 1 - coverage
        self.adaptive = adaptive
        self.adaptive_lr = adaptive_lr

        # Calibration state
        self._quantile: Optional[float] = None
        self._calibration_residuals: Optional[np.ndarray] = None

        # Adaptive state
        self._online_residuals: deque = deque(maxlen=window_size)
        self._online_quantile: Optional[float] = None

        # Statistics
        self._n_predictions = 0
        self._n_covered = 0

    def calibrate(self, residuals: np.ndarray) -> float:
        """
        Calibrate the predictor using absolute residuals from a held-out set.

        Args:
            residuals: array of |y_true - y_pred| values from calibration data

        Returns:
            The computed quantile threshold
        """
        self._calibration_residuals = np.sort(np.abs(residuals))
        n = len(self._calibration_residuals)

        # Conformal quantile: ceil((n+1)*(1-alpha)) / n
        # This gives finite-sample coverage guarantee
        q_idx = int(np.ceil((n + 1) * self.coverage)) - 1
        q_idx = min(q_idx, n - 1)  # clamp to valid range

        self._quantile = float(self._calibration_residuals[q_idx])
        self._online_quantile = self._quantile

        print(f"  [Conformal] Calibrated on {n} residuals")
        print(f"  [Conformal] Quantile (coverage={self.coverage:.0%}): {self._quantile:.6f}")
        print(f"  [Conformal] Median residual: {float(np.median(residuals)):.6f}")
        print(f"  [Conformal] Max residual:    {float(np.max(residuals)):.6f}")

        return self._quantile

    def predict_interval(self, point_pred: float) -> Dict[str, float]:
        """
        Compute prediction interval around a point prediction.

        Returns:
            {"lower": float, "upper": float, "width": float, "quantile": float}
        """
        q = self._get_current_quantile()
        return {
            "lower": point_pred - q,
            "upper": point_pred + q,
            "width": 2 * q,
            "quantile": q,
        }

    def update(self, y_true: float, y_pred: float):
        """
        Update the adaptive predictor with a new observation.
        Call this after each prediction when the true value becomes available.

        Uses Gibbs & Candès (2021) online quantile update:
            q_t+1 = q_t + lr * (alpha - I{|residual| > q_t})

        This automatically widens the interval if coverage is too low
        and narrows it if coverage is too high.
        """
        residual = abs(y_true - y_pred)
        self._online_residuals.append(residual)

        # Track coverage
        self._n_predictions += 1
        q = self._get_current_quantile()
        if residual <= q:
            self._n_covered += 1

        # Adaptive quantile update (Gibbs & Candès, 2021)
        if self.adaptive and self._online_quantile is not None:
            # indicator = I{residual > q} = miscoverage indicator
            # When indicator = 1 (uncovered): q += lr*(1 - alpha) → WIDENS interval
            # When indicator = 0 (covered):   q -= lr*alpha       → NARROWS interval
            indicator = 1.0 if residual > self._online_quantile else 0.0
            self._online_quantile += self.adaptive_lr * (indicator - self.alpha)
            self._online_quantile = max(self._online_quantile, 1e-6)  # prevent negative

    def _get_current_quantile(self) -> float:
        """Get the current quantile — adaptive if available, else static."""
        if self.adaptive and self._online_quantile is not None:
            return self._online_quantile
        if self._quantile is not None:
            return self._quantile
        return 0.05  # safe default before calibration

    @property
    def empirical_coverage(self) -> float:
        """Actual coverage achieved so far (should be close to self.coverage)."""
        if self._n_predictions == 0:
            return 0.0
        return self._n_covered / self._n_predictions

    @property
    def is_calibrated(self) -> bool:
        return self._quantile is not None

    def save(self, path: str):
        """Save calibration state to JSON."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "coverage": self.coverage,
            "quantile": self._quantile,
            "online_quantile": self._online_quantile,
            "n_calibration_samples": len(self._calibration_residuals) if self._calibration_residuals is not None else 0,
            "n_predictions": self._n_predictions,
            "n_covered": self._n_covered,
            "empirical_coverage": round(self.empirical_coverage, 4),
            "adaptive": self.adaptive,
            "adaptive_lr": self.adaptive_lr,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"  [Conformal] State saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ConformalPredictor":
        """Load calibration state from JSON."""
        with open(path) as f:
            state = json.load(f)
        cp = cls(
            coverage=state["coverage"],
            adaptive=state.get("adaptive", True),
            adaptive_lr=state.get("adaptive_lr", 0.005),
        )
        cp._quantile = state["quantile"]
        cp._online_quantile = state.get("online_quantile", state["quantile"])
        cp._n_predictions = state.get("n_predictions", 0)
        cp._n_covered = state.get("n_covered", 0)
        return cp

    def __repr__(self):
        status = "calibrated" if self.is_calibrated else "uncalibrated"
        q = self._get_current_quantile()
        return (f"ConformalPredictor({status}, coverage={self.coverage:.0%}, "
                f"quantile={q:.4f}, empirical={self.empirical_coverage:.1%})")


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Conformal Prediction Self-Test ──")
    rng = np.random.default_rng(42)

    # Simulate: model predicts y = x + noise
    n = 500
    y_true_cal = rng.normal(0, 1, n)
    y_pred_cal = y_true_cal + rng.normal(0, 0.3, n)
    residuals = np.abs(y_true_cal - y_pred_cal)

    cp = ConformalPredictor(coverage=0.90, adaptive=True)
    cp.calibrate(residuals)
    print(f"  {cp}")

    # Test coverage on new data
    n_test = 1000
    y_true_test = rng.normal(0, 1, n_test)
    y_pred_test = y_true_test + rng.normal(0, 0.3, n_test)

    for yt, yp in zip(y_true_test, y_pred_test):
        interval = cp.predict_interval(yp)
        cp.update(yt, yp)

    print(f"  Empirical coverage: {cp.empirical_coverage:.1%} (target: 90%)")
    assert cp.empirical_coverage > 0.85, "Coverage too low!"
    print("  ✓  Conformal prediction working correctly")
