import os
import time
import json
from typing import Optional, Any

class MetadataManager:
    """
    Manages metadata for machine learning models in MHARS, ensuring
    models are not stale and alerting operators if retraining is needed.
    """
    
    def __init__(self, max_age_days: int = 30, logger: Optional[Any] = None):
        self.max_age_days = max_age_days
        self.max_age_seconds = max_age_days * 24 * 3600
        self.logger = logger

    def check_model_freshness(self, model_path: str, model_name: str) -> bool:
        """
        Check if a model file is older than max_age_days.
        Returns True if fresh, False if stale.
        """
        if not os.path.exists(model_path):
            return False
            
        mtime = os.path.getmtime(model_path)
        age = time.time() - mtime
        
        if age > self.max_age_seconds:
            msg = f"Model {model_name} is stale ({(age / 86400):.1f} days old). Consider retraining."
            print(f"  [WARNING] {msg}")
            
            if self.logger:
                self.logger.warning(json.dumps({
                    "event": "model_stale",
                    "model": model_name,
                    "path": model_path,
                    "age_days": round(age / 86400, 1),
                    "message": msg
                }))
            return False
            
        return True
