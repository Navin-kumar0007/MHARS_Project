import pytest
import numpy as np
from stage2_ml.attention_fusion import fuse, interpret

def test_fusion_normal_conditions():
    result = fuse(lstm_score=0.1, ae_score=0.1, if_score=0.1, cnn_score=0.1, audio_score=0.1)
    assert result["global_context_score"] < 0.3
    assert result["urgency"] < 0.3

def test_fusion_lstm_spike():
    result = fuse(lstm_score=0.9, ae_score=0.2, if_score=0.2, cnn_score=0.2, audio_score=0.2)
    assert result["global_context_score"] > 0.3 # Should elevate
    assert result["urgency"] > 0.5 # Should be heavily weighted

def test_fusion_multi_alarm():
    result = fuse(lstm_score=0.8, ae_score=0.9, if_score=0.7, cnn_score=0.8, audio_score=0.8)
    assert result["global_context_score"] > 0.7
    assert result["urgency"] > 0.8

def test_fusion_variance_reduction():
    # When a sensor has high variance (uncertainty), its weight should drop
    result_high_var = fuse(lstm_score=0.5, ae_score=0.5, if_score=0.5, cnn_score=0.9, audio_score=0.5, cnn_var=100.0)
    result_low_var = fuse(lstm_score=0.5, ae_score=0.5, if_score=0.5, cnn_score=0.9, audio_score=0.5, cnn_var=None)
    
    assert result_high_var["global_context_score"] < result_low_var["global_context_score"]

def test_interpret_function():
    assert interpret(0.2) == "Healthy (Routine check)"
    assert interpret(0.6) == "Anomaly Detected (Investigation required)"
    assert interpret(0.9) == "Critical Fault (Immediate maintenance)"
