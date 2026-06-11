import pytest
import numpy as np
from mhars.health_score import HealthScoreEngine
from mhars.trend_analyzer import TrendAnalyzer
from mhars.share_links import ShareLinkManager
from stage2_ml.mc_dropout import MCDropoutEstimator

# Dummy PyTorch model and tensor for testing MCDropout when Torch is missing
class DummyModel:
    def __init__(self):
        self.training = False
    def train(self): self.training = True
    def eval(self): self.training = False
    def __call__(self, x): return type('obj', (object,), {'item': lambda: 0.75})

def test_health_score_engine():
    profile = {"safe_max": 80.0, "critical": 95.0}
    engine = HealthScoreEngine(profile)
    
    # Healthy case
    res = engine.compute(current_temp=50.0, anomaly_score=0.1, rul_minutes=None, vib_score=0.0, drift_detected=False)
    assert res["score"] > 80.0
    assert res["breakdown"]["thermal"] > 80.0
    
    # Critical case
    res2 = engine.compute(current_temp=90.0, anomaly_score=0.8, rul_minutes=10, vib_score=0.5, drift_detected=True)
    assert res2["score"] < 50.0

def test_trend_analyzer():
    analyzer = TrendAnalyzer(target_mean=0.5, std_dev=0.1)

    # Warm-up + steady baseline: analyzer auto-calibrates to the real mean and
    # must report NO drift on stable data.
    for _ in range(40):
        stats = analyzer.update(0.5)
    assert not stats["is_drifting"]

    # Sustained shift away from the learned baseline → drift detected.
    for _ in range(30):
        stats = analyzer.update(1.2)
    assert stats["is_drifting"]
    assert stats["trend_score"] > 0.0

def test_mc_dropout_fallback():
    # Test the fallback without torch
    estimator = MCDropoutEstimator(num_samples=5)
    dummy_input = [0.1, 0.2, 0.3, 0.4]
    
    res = estimator._fallback_prediction(dummy_input)
    assert res["mean"] == 0.4
    assert res["confidence"] == 85.0
    
def test_share_links(tmp_path):
    # Override path for testing
    import mhars.share_links
    mhars.share_links.SHARE_LINKS_FILE = tmp_path / "test_links.json"
    
    manager = ShareLinkManager()
    token = manager.create_link("admin", "Test Link", expires_in_hours=1)
    
    assert token is not None
    assert len(manager.list_links()) == 1
    
    # Validation
    assert manager.validate_and_record_access(token) == True
    assert manager.validate_and_record_access("invalid") == False
    
    # Revoke
    manager.revoke_link(token)
    assert len(manager.list_links()) == 0
