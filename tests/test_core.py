import pytest
import numpy as np
from mhars import MHARS, MHARSResult
from mhars.config import Config

# Dummy tests to ensure the Core class handles instantiation and basic typing correctly.
# We skip the heavy model loading here by checking if it boots up fast and doesn't crash on standard input.

def test_mhars_initialization():
    system = MHARS(machine_type_id=1, verbose=False, llm_path=None)
    assert system.machine_type_id == 1
    assert system.machine_name == "Pump"
    assert system.profile["critical"] == 95.0

def test_mhars_run_returns_valid_result():
    system = MHARS(machine_type_id=1, verbose=False, llm_path=None)
    result = system.run(temp_celsius=50.0)
    
    assert isinstance(result, MHARSResult)
    assert result.current_temp == 50.0
    assert 0.0 <= result.context_score <= 1.0
    assert 0.0 <= result.urgency <= 1.0
    assert result.route in ["edge", "cloud", "both"]
    assert result.action in Config.ACTIONS

def test_mhars_run_sequence():
    system = MHARS(machine_type_id=1, verbose=False, llm_path=None)
    temps = [45.0, 46.0, 47.0]
    results = system.run_sequence(temps)
    
    assert len(results) == 3
    for res in results:
        assert isinstance(res, MHARSResult)
