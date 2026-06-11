import numpy as np
from typing import Dict, Any

class HealthScoreEngine:
    """
    Computes a composite 0-100 health score for a machine based on multiple telemetry signals.
    """
    
    def __init__(self, machine_profile: Dict[str, Any]):
        self.profile = machine_profile
        self._history = []
        
    def compute(self, current_temp: float, anomaly_score: float, 
                rul_minutes: float, vib_score: float, drift_detected: bool) -> Dict[str, Any]:
        """
        Calculate the overall health score and per-component breakdown.
        
        Args:
            current_temp: Current temperature in Celsius.
            anomaly_score: Autoencoder reconstruction error score [0, 1].
            rul_minutes: Remaining useful life in minutes (or None if stable).
            vib_score: Vibration anomaly score [0, 1].
            drift_detected: Boolean flag for concept drift.
            
        Returns:
            Dict containing 'score' (0-100), 'trend', and 'breakdown' (sub-scores).
        """
        # 1. Thermal Health (0-100)
        # Based on proximity to safe_max and critical thresholds
        safe_max = self.profile["safe_max"]
        critical = self.profile["critical"]

        # Guard against degenerate profiles that would cause division by zero.
        safe_span = max(safe_max - 25, 1e-6)
        crit_span = max(critical - safe_max, 1e-6)

        if current_temp <= safe_max:
            # Scale from 100 to 70 as it approaches safe_max
            thermal_score = 100 - 30 * (max(0, current_temp - 25) / safe_span)
        elif current_temp < critical:
            # Scale from 70 to 0 as it approaches critical
            thermal_score = 70 - 70 * ((current_temp - safe_max) / crit_span)
        else:
            thermal_score = 0.0
            
        # Modulate thermal health by overall anomaly score
        thermal_score = thermal_score * (1.0 - (anomaly_score * 0.5))
            
        # 2. Mechanical/Vibration Health (0-100)
        # vib_score is a [0,1] anomaly level; only penalise meaningfully high values
        # so a low synthetic baseline doesn't crush the score on healthy machines.
        mech_score = 100 * (1.0 - min(max(vib_score - 0.2, 0.0) * 1.25, 1.0))
        
        # 3. RUL Health (0-100)
        # If RUL > 60 mins, it's 100. If < 5 mins, it's 0.
        if rul_minutes is None:
            rul_score = 100.0
        else:
            rul_score = np.clip((rul_minutes - 5) / 55.0, 0, 1) * 100
            
        # 4. Drift Penalty
        drift_penalty = 15 if drift_detected else 0
        
        # Weighted Composite
        weights = {
            "thermal": 0.45,
            "mechanical": 0.35,
            "rul": 0.20
        }
        
        overall = (thermal_score * weights["thermal"] + 
                   mech_score * weights["mechanical"] + 
                   rul_score * weights["rul"]) - drift_penalty
                   
        overall = float(np.clip(overall, 0, 100))
        
        # Trend calculation
        self._history.append(overall)
        if len(self._history) > 120:  # Keep ~2 mins of history at 1Hz
            self._history.pop(0)
            
        trend = "stable"
        if len(self._history) >= 10:
            recent_avg = np.mean(self._history[-5:])
            past_avg = np.mean(self._history[-10:-5])
            diff = recent_avg - past_avg
            if diff > 1.5:
                trend = "improving"
            elif diff < -1.5:
                trend = "degrading"
                
        return {
            "score": round(overall, 1),
            "trend": trend,
            "breakdown": {
                "thermal": round(thermal_score, 1),
                "mechanical": round(mech_score, 1),
                "longevity": round(rul_score, 1)
            }
        }
