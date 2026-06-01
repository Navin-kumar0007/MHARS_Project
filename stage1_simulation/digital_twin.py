"""
MHARS — Stage 1: Digital Twin
=============================
A predictive simulation module to run "what-if" scenarios based on the
current system state without affecting the real environment.
"""

import copy
import numpy as np

class DigitalTwin:
    """
    A stateless digital twin that unrolls physics equations to predict
    future states based on hypothetical action sequences.
    """
    def __init__(self, profile):
        self.profile = profile
        self.ambient_temp = 25.0
        self.heat_rate = self.profile.get("heat_rate", 2.0)
        self.conv_coeff = self.profile.get("conv_coeff", 0.05)
        self.thermal_mass = self.profile.get("thermal_mass_J_K", 20.0)
        self.safe_max = self.profile.get("safe_max", 85.0)
        self.critical = self.profile.get("critical", 100.0)

    def simulate_what_if(self, current_temp, current_load, current_fan, action_sequence, steps_per_action=1):
        """
        Simulates the thermal trajectory given an action sequence.
        
        action_sequence: list of float fan speeds [0, 1] or strings like "shutdown"
        steps_per_action: how many seconds each action is sustained
        
        Returns:
            list of predicted temperatures
        """
        temp = current_temp
        load = current_load
        fan = current_fan
        
        trajectory = [temp]
        
        for action in action_sequence:
            # Parse action
            if isinstance(action, str):
                if action == "shutdown" or action == "emergency-shutdown":
                    load = 0.05
                    fan = 1.0
                elif action == "fan+":
                    fan = min(1.0, fan + 0.3)
                elif action == "throttle":
                    load = max(0.1, load - 0.15)
                elif action == "do-nothing" or action == "alert":
                    pass # fan drifts down, load decays slightly, but we'll hold constant for simplicity
            else:
                fan = float(action)
                
            for _ in range(steps_per_action):
                q_in = self.heat_rate * load * 100.0
                effective_h = self.conv_coeff * (1.0 + 2.0 * fan)
                q_out = effective_h * (temp - self.ambient_temp) * 100.0
                
                dT = (q_in - q_out) / self.thermal_mass
                temp = temp + dT
                temp = np.clip(temp, 15.0, self.critical + 10.0)
                trajectory.append(temp)
                
        return trajectory
