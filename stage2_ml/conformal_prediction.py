"""
MHARS — Stage 2: Conformal Prediction
=====================================
Applies split-conformal prediction to the LSTM output.
Provides statistically guaranteed safety bounds for thermal predictions.
"""

import numpy as np

class ConformalPredictor:
    def __init__(self, alpha: float = 0.05):
        """
        alpha: the error rate. e.g., alpha=0.05 gives 95% coverage.
        """
        self.alpha = alpha
        self.q_hat = None
        self.calibrated = False

    def calibrate(self, residuals: np.ndarray):
        """
        Calibrates the predictor using absolute residuals from a validation set.
        residuals: |y_true - y_pred|
        """
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = float(np.quantile(residuals, q_level))
        self.calibrated = True

    def predict_interval(self, point_prediction: float) -> dict:
        """
        Returns the conformal prediction interval for a point prediction.
        """
        if not self.calibrated:
            # Fallback if not calibrated 
            # 0.05 is an arbitrary fallback residual for normalized temp
            self.q_hat = 0.05
            
        lower_bound = point_prediction - self.q_hat
        upper_bound = point_prediction + self.q_hat
        
        return {
            "prediction": float(point_prediction),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "confidence_level": 1.0 - self.alpha,
            "q_hat": self.q_hat
        }
