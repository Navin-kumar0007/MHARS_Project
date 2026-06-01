import pytest
import torch
import numpy as np
from mhars.config import Config
from stage2_ml.tft_predictor import TFTPredictor
from stage3_ai.causal_layer import PhysicsCausalLayer
from stage5_adapter.machine_adapter import MetaLearningAdapter
from stage6_federated.fed_avg import FederatedClient, FederatedServer, run_federated_simulation
from stage1_simulation.digital_twin import DigitalTwin

def test_tft_predictor():
    """Verify TFT predictor shapes and quantile output."""
    model = TFTPredictor(num_vars=5, d_model=16, n_heads=2, num_quantiles=3)
    x = torch.rand(2, 12, 5) # batch=2, seq_len=12, vars=5
    quantiles, var_weights, attn_weights = model(x)
    
    assert quantiles.shape == (2, 3), "Quantile shape mismatch"
    assert var_weights.shape == (2, 12, 5), "Variable weights shape mismatch"
    assert attn_weights.shape == (2, 12, 12), "Attention weights shape mismatch"
    
    # p10 should be less than or equal to p90 roughly (if trained), 
    # but initially it's just random projection. So we just check shape.

def test_physics_causal_layer():
    """Verify physics-informed causal layer outputs reasonable estimates."""
    profile = {"name": "Test", "heat_rate": 2.0, "conv_coeff": 0.05, "thermal_mass_J_K": 20.0, "safe_max": 85.0}
    layer = PhysicsCausalLayer(profile)
    
    # Normal operation
    res_normal = layer.analyze(current_temp=30.0, load=0.5, fan_speed=0.0)
    assert res_normal["fault_probability"] < 0.2
    
    # Unexpected heating (fault)
    res_fault = layer.analyze(current_temp=90.0, load=0.1, fan_speed=1.0)
    assert res_fault["fault_probability"] > 0.8
    assert res_fault["residual"] > 20.0

def test_maml_adapter():
    """Verify MAML adapter completes inner/outer loop without crashing."""
    import os
    if not os.path.exists(Config.LSTM_V2):
        pytest.skip("Base LSTM V2 not found")
        
    adapter = MetaLearningAdapter(Config.LSTM_V2, 0)
    
    # Dummy data
    X_s = np.random.rand(10, 12, 5)
    y_s = np.random.rand(10)
    X_q = np.random.rand(10, 12, 5)
    y_q = np.random.rand(10)
    
    tasks = [(X_s, y_s, X_q, y_q)]
    
    try:
        adapter.meta_train(tasks, meta_epochs=1)
    except Exception as e:
        pytest.fail(f"MAML training failed: {e}")

def test_federated_learning():
    """Verify FedAvg works across multiple simulated clients."""
    model = TFTPredictor(num_vars=5, d_model=16, n_heads=2, num_quantiles=3)
    
    # Client 1
    X1 = np.random.rand(20, 12, 5)
    y1 = np.random.rand(20, 3)
    # Client 2
    X2 = np.random.rand(20, 12, 5)
    y2 = np.random.rand(20, 3)
    
    try:
        global_model = run_federated_simulation(model, [(X1, y1), (X2, y2)], rounds=1, local_epochs=1, is_v2=True)
        assert isinstance(global_model, TFTPredictor)
    except Exception as e:
        pytest.fail(f"FedAvg failed: {e}")

def test_digital_twin():
    """Verify Digital Twin simulate_what_if returns valid trajectory."""
    profile = {"name": "Test", "heat_rate": 2.0, "conv_coeff": 0.05, "thermal_mass_J_K": 20.0, "safe_max": 85.0, "critical": 100.0}
    dt = DigitalTwin(profile)
    
    traj = dt.simulate_what_if(current_temp=80.0, current_load=0.8, current_fan=0.0, action_sequence=["throttle", "throttle"], steps_per_action=5)
    assert len(traj) == 11 # 1 initial + 2 * 5 steps
    assert traj[0] == 80.0
