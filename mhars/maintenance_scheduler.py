from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta

class MaintenanceScheduler:
    """
    Predictive Maintenance Scheduler.
    Determines optimal maintenance windows based on Remaining Useful Life (RUL).
    """
    
    def __init__(self, machine_profile: Dict[str, Any]):
        self.profile = machine_profile
        
    def schedule(self, rul_minutes: Optional[float], health_score: float) -> Dict[str, Any]:
        """
        Calculates maintenance window based on RUL and health.
        
        Args:
            rul_minutes: Estimated RUL in minutes (None if healthy).
            health_score: Composite health score [0, 100].
            
        Returns:
            Dictionary containing urgency, recommended window, and action.
        """
        now = datetime.now()
        
        if rul_minutes is None or rul_minutes > 1440: # > 24 hours
            # System is healthy or RUL is far out
            if health_score < 70:
                return {
                    "status": "Warning",
                    "window_start": (now + timedelta(days=7)).isoformat(),
                    "window_end": (now + timedelta(days=14)).isoformat(),
                    "urgency_level": "low",
                    "action": "Schedule routine inspection in next maintenance cycle."
                }
            return {
                "status": "Healthy",
                "window_start": None,
                "window_end": None,
                "urgency_level": "none",
                "action": "No immediate maintenance required."
            }
            
        elif rul_minutes <= 60: # < 1 hour
            return {
                "status": "Critical",
                "window_start": now.isoformat(),
                "window_end": (now + timedelta(minutes=60)).isoformat(),
                "urgency_level": "critical",
                "action": "Immediate emergency maintenance required to prevent failure."
            }
            
        elif rul_minutes <= 480: # < 8 hours
            return {
                "status": "High",
                "window_start": now.isoformat(),
                "window_end": (now + timedelta(hours=4)).isoformat(),
                "urgency_level": "high",
                "action": "Schedule maintenance within next shift."
            }
            
        else: # 8 - 24 hours
            return {
                "status": "Medium",
                "window_start": (now + timedelta(hours=12)).isoformat(),
                "window_end": (now + timedelta(hours=24)).isoformat(),
                "urgency_level": "medium",
                "action": "Plan maintenance for next off-peak window."
            }
