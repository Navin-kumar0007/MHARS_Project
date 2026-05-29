"""
MHARS — Stage 3: Causal Reasoning Layer
========================================
Distinguishes expected heating (from load) from unexpected heating (fault).
Helps the LLM write better, highly contextual alerts.
"""

class CausalReasoning:
    def __init__(self, safe_max: float = 85.0):
        self.safe_max = safe_max

    def analyze(self, temp: float, load: float) -> dict:
        """
        Calculates causal bounds.
        temp: Current temperature (Celsius)
        load: Current system load [0, 1]
        """
        # A simple causal heuristic: Expected temperature rises linearly with load
        # Base temp = 30C at 0 load, up to safe_max at 1.0 load
        expected_temp = 30.0 + (self.safe_max - 30.0) * load
        
        # If temp is much higher than expected for this load, it's a fault
        residual = temp - expected_temp
        
        if residual > 15.0:
            root_cause = "External fault or cooling failure"
            causal_score = 1.0 # High anomaly
        elif residual > 5.0:
            root_cause = "Slight inefficiency or ambient heat"
            causal_score = 0.5
        elif temp > self.safe_max * 0.9:
            root_cause = "Expected heating due to high load"
            causal_score = 0.1
        else:
            root_cause = "Normal operation"
            causal_score = 0.0

        return {
            "expected_temp": expected_temp,
            "residual": residual,
            "root_cause_hypothesis": root_cause,
            "fault_probability": causal_score
        }
