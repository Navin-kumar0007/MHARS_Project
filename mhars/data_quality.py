"""
MHARS — Sensor-health / data-quality gate.
==========================================
Cheap preprocessing that runs BEFORE the ML models so a dead, stuck, or wildly
out-of-range sensor lowers pipeline trust instead of silently producing a
confident-but-wrong anomaly score. Industry deployments live and die on this:
most "AI false alarms" in the field are actually bad data, not bad models.

Detects, per tick:
  · NaN / inf            — corrupt reading
  · out-of-range         — beyond absolute physical plausibility
  · spike                — implausible instantaneous jump (|ΔT| too large)
  · flatline / stuck     — sensor frozen (near-zero variance over a window)

Returns a quality score ∈ [0,1] and a list of human-readable flags. The score
feeds urgency-confidence and the LLM context; control decisions are left to the
safety layer (a bad sensor must never *suppress* a real over-temperature).
"""
from __future__ import annotations
from collections import deque
from typing import List, Dict, Any
import math

from mhars.config import Config


class DataQualityMonitor:
    """Stateful, O(1)-per-tick sensor-health checker."""

    def __init__(self):
        self._recent = deque(maxlen=Config.DQ_FLATLINE_WINDOW)
        self._prev_temp: float | None = None

    def check(self, temp_c: float, dT_dt: float | None = None) -> Dict[str, Any]:
        flags: List[str] = []
        quality = 1.0

        # 1. Corrupt values — hard fail, do not let NaN propagate into torch.
        if temp_c is None or math.isnan(temp_c) or math.isinf(temp_c):
            return {
                "ok": False, "quality": 0.0, "flags": ["corrupt: NaN/inf reading"],
                "usable_temp": self._prev_temp if self._prev_temp is not None else 0.0,
                "note": "Corrupt sensor reading — holding last good value.",
            }

        # 2. Out-of-range — beyond absolute physical plausibility.
        if temp_c < Config.DQ_TEMP_MIN or temp_c > Config.DQ_TEMP_MAX:
            flags.append(f"out-of-range: {temp_c:.1f}°C")
            quality = min(quality, 0.15)

        # 3. Spike — implausible instantaneous jump.
        jump = abs(temp_c - self._prev_temp) if self._prev_temp is not None else 0.0
        if jump > Config.DQ_SPIKE_DT:
            flags.append(f"spike: ΔT {jump:.1f}°C/tick")
            quality = min(quality, 0.35)

        # 4. Flatline / stuck sensor — near-zero variance over the window.
        self._recent.append(temp_c)
        if len(self._recent) >= Config.DQ_FLATLINE_WINDOW:
            mean = sum(self._recent) / len(self._recent)
            var = sum((x - mean) ** 2 for x in self._recent) / len(self._recent)
            std = var ** 0.5
            # A live thermal signal always has micro-jitter; exact-zero std over
            # 15 ticks means the sensor is frozen (unless genuinely idle at 0).
            if std < Config.DQ_FLATLINE_STD and abs(mean) > 1e-6:
                flags.append("flatline: sensor may be stuck")
                quality = min(quality, 0.4)

        self._prev_temp = temp_c
        note = "OK" if not flags else "; ".join(flags)
        return {
            "ok": quality >= 0.5,
            "quality": round(quality, 3),
            "flags": flags,
            "usable_temp": temp_c,
            "note": note,
        }
