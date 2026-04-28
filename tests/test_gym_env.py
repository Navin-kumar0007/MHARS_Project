import pytest
import numpy as np
from stage1_simulation.gym_env import ThermalEnv, MACHINE_PROFILES

def test_env_initialization():
    env = ThermalEnv(machine_type_id=1, max_steps=100)
    assert env.machine_type_id == 1
    assert env.profile == MACHINE_PROFILES[1]
    assert env.action_space.n == 5
    # observation space contains: [temp, pred_temp, ae_score, machine_type, time, urgency]
    assert env.observation_space.shape == (6,)

def test_env_reset():
    env = ThermalEnv(machine_type_id=1)
    obs, info = env.reset()
    assert len(obs) == 6
    assert isinstance(info, dict)
    assert "temp" in info

def test_env_step_logic():
    env = ThermalEnv(machine_type_id=1)
    obs, _ = env.reset()
    init_temp = env.temp
    
    # Action 1: Fan+ should decrease temp
    obs, reward, term, trunc, info = env.step(1)
    assert env.temp < init_temp or env.temp == env.profile["min_temp"]
    
def test_env_reward_penalties():
    env = ThermalEnv(machine_type_id=1)
    env.reset()
    
    # Set to critical temp
    env.temp = env.profile["critical"] + 10
    obs, reward, term, trunc, info = env.step(0) # Do nothing
    assert reward < -50 # Heavy penalty for ignoring critical
    assert term == True # Ends episode

def test_env_truncation():
    env = ThermalEnv(machine_type_id=1, max_steps=5)
    env.reset()
    for _ in range(4):
        _, _, term, trunc, _ = env.step(0)
        assert trunc == False
    _, _, term, trunc, _ = env.step(0)
    assert trunc == True
