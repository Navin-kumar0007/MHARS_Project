"""
MHARS — Sensor Reading Schema
===============================
Multi-feature input schema replacing the single-float temp_celsius.

A motor at 72°C under 100% load is fine; a motor at 72°C at idle
is alarming. This dataclass gives the AI pipeline the context it
needs to tell the difference.

Usage:
    from mhars.schemas import SensorReading
    reading = SensorReading(temp_c=72.0, load_pct=0.95)
    result  = system.run(reading)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SensorReading:
    """
    Multi-dimensional sensor snapshot for one timestep.

    Required:
        temp_c        — current temperature in °C

    Optional (auto-computed or defaulted when not available):
        load_pct      — machine load as fraction 0.0–1.0
        ambient_c     — ambient/room temperature °C
        dT_dt         — rate of temperature change °C/sec
                        (auto-computed from ring buffer if None)
        humidity_pct  — relative humidity 0.0–100.0
        vibration_g   — vibration in g-force (from accelerometer)
    """
    temp_c:        float
    load_pct:      float = 0.0
    ambient_c:     float = 25.0
    dT_dt:         Optional[float] = None   # None = auto-compute from history
    humidity_pct:  float = 50.0
    vibration_g:   float = 0.0              # 0.0 = use thermal proxy, >0 = use real sensor
    audio_score:   Optional[float] = None   # None = use thermal proxy, 0-1 = use real sensor
    audio_var:     Optional[float] = None

    def to_feature_vector(self) -> list:
        """Return the feature vector for ML models (dT_dt must be set)."""
        return [
            self.temp_c,
            self.load_pct,
            self.ambient_c,
            self.dT_dt or 0.0,
            self.humidity_pct / 100.0,
            self.vibration_g,
        ]

    @classmethod
    def from_temp_only(cls, temp_c: float) -> "SensorReading":
        """Backwards-compatible constructor from a single temperature."""
        return cls(temp_c=temp_c)
