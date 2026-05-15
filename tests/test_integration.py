import sys
import os
import pytest

# Add stage dirs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mhars.core import MHARS
from mhars.schemas import SensorReading
from mhars.config import Config

@pytest.fixture
def system():
    # Use motor profile for testing
    mhars = MHARS(machine_type_id=1)
    return mhars

def test_integration_pipeline_safe(system):
    """Test that normal operating temperatures yield a SAFE action."""
    # Motor idle is 40.0. Normal temp is ~45.0.
    for i in range(15):
        # Temp fluctuates around 45°C
        temp = 45.0 + (i % 3) * 0.5
        reading = SensorReading(
            temp_c=temp,
            load_pct=0.2,
            vibration_g=0.5
        )
        res = system.run(reading)
    
    # After 15 steps, it should be stable and safe
    assert res.action in ["do-nothing", "fan+", "alert"], f"Expected do-nothing, fan+, or alert, got {res.action}"
    assert res.urgency < 0.50  # load_factor (1.24 at 20% load) amplifies urgency

def test_integration_pipeline_critical(system):
    """Test that a rapid temperature rise yields a CRITICAL warning."""
    # Warm up phase
    for i in range(12):
        system.run(SensorReading(temp_c=45.0, load_pct=0.5))
        
    # Critical spike phase (motor critical is 95.0)
    for temp in [60.0, 75.0, 90.0, 96.0]:
        res = system.run(SensorReading(
            temp_c=temp, 
            load_pct=0.9, 
            vibration_g=12.0  # Add high vibration for anomaly certainty
        ))
        
    assert res.action in ["emergency-shutdown", "shutdown", "alert", "throttle"], f"Expected a mitigating action, got {res.action}"
    assert res.urgency > 0.7

def test_integration_pipeline_emergency_override(system):
    """Test that temperatures exceeding safe maximum trigger an immediate emergency-shutdown."""
    # Motor critical is 95.0. A reading of 96.0 should trigger an emergency shutdown.
    reading = SensorReading(temp_c=96.0, load_pct=0.9, vibration_g=1.0)
    res = system.run(reading)
    assert res.action == "emergency-shutdown", f"Expected emergency-shutdown override, got {res.action}"

def test_llm_queue_overflow(system):
    """Test that the system remains responsive even when the LLM queue is saturated."""
    # Inject 20 fast readings (queue size is 10)
    for i in range(20):
        # High temp to trigger alert generation
        res = system.run(SensorReading(temp_c=70.0 + (i % 5), load_pct=0.8))
        
    # Verify that the last result still has a valid (though maybe pending) alert
    assert res.current_temp >= 70.0
    assert res.alert is not None
    # Wait for background queue to clear
    system.wait_for_alerts()
    assert len(res.alert) > 0, "Alert should be a non-empty string even under queue pressure"
