"""
R1 — Zero-shot foundation-model forecaster + residual anomaly.

Wraps a small pretrained time-series foundation model (Chronos-Bolt) to provide
multi-horizon *quantile* forecasts with NO per-machine training, and a
distribution-free **residual anomaly** score: how far the actual reading falls
outside the model's own predicted p10–p90 band.

Why it matters for MHARS: the bespoke LSTM/AE are trained on the simulator, so
they go out-of-distribution on unseen machines / real hardware (e.g. the Live
mode false-100% detector). A zero-shot foundation model forecasts *any* signal's
next-step band, so the residual anomaly generalises without retraining.

Graceful: if `chronos-forecasting` isn't installed the class raises ImportError,
which callers catch and fall back to the trained models.
"""
from __future__ import annotations
import numpy as np


class FoundationForecaster:
    def __init__(self, model_name: str = "amazon/chronos-bolt-tiny",
                 quantiles=(0.1, 0.5, 0.9)):
        self.model_name = model_name
        self.quantiles = list(quantiles)
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.model_name, device_map="cpu", dtype=torch.float32)
        return self._pipe

    def forecast(self, history, horizon: int) -> dict:
        """history: 1-D array of recent values. Returns p10/p50/p90 arrays (len=horizon)."""
        import torch
        pipe = self._load()
        ctx = torch.tensor(np.asarray(history, dtype=np.float32))
        q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon,
                                      quantile_levels=self.quantiles)
        q = q[0].numpy()  # (horizon, n_quantiles)
        return {"p10": q[:, 0], "p50": q[:, 1], "p90": q[:, 2]}

    @staticmethod
    def residual_anomaly(actual: float, p10: float, p50: float, p90: float) -> float:
        """Distribution-free anomaly score ∈ [0,1]: 0 when the actual reading is
        inside the predicted band, → 1 the further it lies outside, normalized by
        band width. Scale-free and zero-shot → generalises across machines."""
        width = max(p90 - p10, 1e-6)
        if actual > p90:
            z = (actual - p90) / width
        elif actual < p10:
            z = (p10 - actual) / width
        else:
            z = 0.0
        return float(min(1.0, z))
