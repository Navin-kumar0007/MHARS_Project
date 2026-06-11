"""
Anomaly score calibration via Peaks-Over-Threshold (EVT).

Replaces the raw ``error / fixed_threshold`` mapping — which leaves a high,
jittery baseline on normal data and saturates out-of-distribution — with a
calibrated 0..1 score anchored on the *extreme-value* distribution of normal
reconstruction errors.

Method (Siffer et al., 2017 — "Anomaly Detection in Streams with EVT"):
  1. Pick an initialisation threshold ``t`` (high quantile of normal errors).
  2. Fit a Generalised Pareto Distribution (GPD) to the excesses ``e - t`` by
     method-of-moments (no SciPy dependency).
  3. Derive the extreme alarm threshold ``z_q`` for a target risk ``q`` (false
     alarm rate). Above ``z_q`` an observation is a near-certain anomaly.

Calibrated score (monotone, interpretable):
    err <= p50          → 0.0   (typical normal)
    p50  < err <= t     → 0..0.5 (within the normal body)
    t    < err <= z_q   → 0.5..1.0 (in the tail, approaching alarm)
    err  > z_q          → 1.0   (extreme — anomaly)
"""
from __future__ import annotations
import numpy as np


class AnomalyCalibrator:
    def __init__(self, p50: float, t: float, z_q: float, xi: float, beta: float, q: float = 1e-3):
        self.p50 = float(p50)
        self.t = float(t)
        self.z_q = float(z_q)
        self.xi = float(xi)
        self.beta = float(beta)
        self.q = float(q)

    # ── Fit ──────────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, errors: np.ndarray, init_quantile: float = 0.92, q: float = 1e-3) -> "AnomalyCalibrator":
        e = np.asarray(errors, dtype=np.float64)
        e = e[np.isfinite(e)]
        n = len(e)
        p50 = float(np.percentile(e, 50))
        t = float(np.percentile(e, init_quantile * 100))

        excess = e[e > t] - t
        n_t = len(excess)
        # Degenerate tail → fall back to a robust high percentile for z_q.
        if n_t < 20 or excess.var() <= 1e-18:
            z_q = float(max(np.percentile(e, 99.9), e.max() * 1.05))
            return cls(p50, t, z_q, xi=0.0, beta=max(excess.mean() if n_t else 1e-6, 1e-9), q=q)

        m = float(excess.mean())
        v = float(excess.var())
        r = (m * m) / (v + 1e-18)
        xi = 0.5 * (1.0 - r)
        beta = 0.5 * m * (r + 1.0)
        # Numerical guards
        xi = float(np.clip(xi, -0.5, 0.9))
        beta = float(max(beta, 1e-9))

        # SPOT extreme threshold for risk q.
        ratio = (q * n) / max(n_t, 1)
        if abs(xi) < 1e-6:
            z_q = t + beta * np.log(max(n_t / (q * n + 1e-18), 1.0 + 1e-9))
        else:
            z_q = t + (beta / xi) * (ratio ** (-xi) - 1.0)
        # z_q must sit above t and above the observed normal max.
        z_q = float(max(z_q, t * 1.0 + beta, e.max() * 1.02))
        return cls(p50, t, z_q, xi, beta, q)

    # ── Calibrated score ─────────────────────────────────────────────────────
    def score(self, err: float) -> float:
        err = float(err)
        if err <= self.p50:
            return 0.0
        if err <= self.t:
            denom = (self.t - self.p50) or 1e-9
            return float(np.clip(0.5 * (err - self.p50) / denom, 0.0, 0.5))
        if err <= self.z_q:
            denom = (self.z_q - self.t) or 1e-9
            return float(np.clip(0.5 + 0.5 * (err - self.t) / denom, 0.5, 1.0))
        return 1.0

    # ── Serialisation ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"p50": self.p50, "t": self.t, "z_q": self.z_q,
                "xi": self.xi, "beta": self.beta, "q": self.q}

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyCalibrator":
        return cls(d["p50"], d["t"], d["z_q"], d.get("xi", 0.0), d.get("beta", 1e-6), d.get("q", 1e-3))
