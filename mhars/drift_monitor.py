"""
X.1 — Streaming feature-drift monitor.

Goes beyond the existing CUSUM/EWMA-on-temperature trend check: it watches the
*distribution* of the full feature vector the models consume and flags when the
normal operating regime has shifted enough to warrant retraining (concept
drift). Only NORMAL-operation samples are counted, so transient injected faults
don't masquerade as drift.

Metric: per-feature standardized mean shift |mean_cur - mean_ref| / std_ref,
averaged across features (a lightweight PSI-style divergence). A sustained
breach of the threshold raises `retrain_recommended`.
"""
from __future__ import annotations
from collections import deque
import numpy as np


class DriftMonitor:
    def __init__(self, ref_n: int = 60, cur_n: int = 60, threshold: float = 2.5, sustain: int = 30):
        self.ref_n = ref_n
        self.cur_n = cur_n
        self.threshold = threshold
        self.sustain = sustain
        self._ref: list[np.ndarray] = []
        self._ref_mean = None
        self._ref_std = None
        self._cur = deque(maxlen=cur_n)
        self._breach_streak = 0
        self.drift_score = 0.0
        self.drifting = False
        self.retrain_recommended = False

    def update(self, features, is_normal: bool):
        """Feed one feature vector. Only normal-operation samples shape the
        reference/current distributions. Returns the current drift score."""
        if not is_normal:
            return self.drift_score
        f = np.asarray(features, dtype=np.float64)

        # Phase 1: build a frozen reference from the first ref_n normal samples.
        if self._ref_mean is None:
            self._ref.append(f)
            if len(self._ref) >= self.ref_n:
                R = np.array(self._ref)
                self._ref_mean = R.mean(axis=0)
                self._ref_std = R.std(axis=0) + 1e-6
            return self.drift_score

        # Phase 2: compare the rolling current window to the reference.
        self._cur.append(f)
        if len(self._cur) < max(10, self.cur_n // 2):
            return self.drift_score
        cur_mean = np.array(self._cur).mean(axis=0)
        self.drift_score = float(np.mean(np.abs(cur_mean - self._ref_mean) / self._ref_std))

        if self.drift_score > self.threshold:
            self._breach_streak += 1
        else:
            self._breach_streak = max(0, self._breach_streak - 1)
        self.drifting = self.drift_score > self.threshold
        self.retrain_recommended = self._breach_streak >= self.sustain
        return self.drift_score

    def snapshot(self) -> dict:
        return {
            "drift_score": round(self.drift_score, 3),
            "drifting": bool(self.drifting),
            "retrain_recommended": bool(self.retrain_recommended),
            "reference_ready": self._ref_mean is not None,
            "threshold": self.threshold,
        }
