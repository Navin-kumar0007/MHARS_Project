import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class TrendAnalyzer:
    """
    Statistical Trend Analysis Engine for gradual drift detection.
    Uses CUSUM (Cumulative Sum) and EWMA (Exponentially Weighted Moving Average).
    """
    
    def __init__(self, target_mean: float, std_dev: float, ewma_alpha: float = 0.1,
                 window_size: int = 100, warmup: int = 30):
        self.target_mean = target_mean
        self.std_dev = max(std_dev, 0.01) # Avoid division by zero
        self.alpha = ewma_alpha
        self.window_size = window_size

        # Auto-calibration: learn the real baseline from the first `warmup` samples
        # instead of trusting a fixed target_mean. Prevents permanent false "drift"
        # when the actual operating point differs from the assumed mean.
        self.warmup = warmup
        self._warmup_buf = []
        self._calibrated = False

        # CUSUM state
        self.s_high = 0.0
        self.s_low = 0.0
        self.cusum_history = deque(maxlen=window_size)

        # EWMA state
        self.ewma = target_mean
        self.ewma_history = deque(maxlen=window_size)

        self.raw_history = deque(maxlen=window_size)

    def update(self, value: float) -> Dict[str, float]:
        """
        Process a new value and return trend statistics.
        """
        self.raw_history.append(value)

        # ── Warm-up calibration ──────────────────────────────────────────────
        # During warm-up, collect samples and report no drift. After warm-up,
        # set target_mean/std from observed data so CUSUM measures *deviation
        # from this machine's real baseline*, not from a guessed constant.
        if not self._calibrated:
            self._warmup_buf.append(value)
            self.ewma = value if len(self._warmup_buf) == 1 else self.alpha * value + (1 - self.alpha) * self.ewma
            self.ewma_history.append(self.ewma)
            if len(self._warmup_buf) >= self.warmup:
                import numpy as _np
                self.target_mean = float(_np.mean(self._warmup_buf))
                self.std_dev = max(float(_np.std(self._warmup_buf)), 0.02)
                self._calibrated = True
            return {"ewma": self.ewma, "cusum": 0.0, "trend_score": 0.0, "is_drifting": False}

        # 1. EWMA Update
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma
        self.ewma_history.append(self.ewma)

        # 2. CUSUM Update
        # Standardize the value
        z = (value - self.target_mean) / self.std_dev
        
        # Slack parameter (typically 0.5)
        k = 0.5
        
        self.s_high = max(0.0, self.s_high + z - k)
        self.s_low = max(0.0, self.s_low - z - k)
        
        cusum_val = max(self.s_high, self.s_low)
        self.cusum_history.append(cusum_val)
        
        # Calculate trend score [0, 1] based on CUSUM threshold (h=5 is typical)
        h = 5.0
        trend_score = min(cusum_val / (h * 2.0), 1.0)
        
        return {
            "ewma": self.ewma,
            "cusum": cusum_val,
            "trend_score": trend_score,
            "is_drifting": cusum_val > h
        }
        
    def reset(self, new_target: float, new_std: float):
        self.target_mean = new_target
        self.std_dev = max(new_std, 0.01)
        self.s_high = 0.0
        self.s_low = 0.0
        self.ewma = new_target
        self.cusum_history.clear()
        self.ewma_history.clear()
        self.raw_history.clear()
        # Re-run warm-up calibration after a reset
        self._warmup_buf = []
        self._calibrated = False
