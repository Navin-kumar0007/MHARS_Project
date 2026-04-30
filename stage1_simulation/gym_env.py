"""
MHARS — Stage 1: Thermal Management Gymnasium Environment
==========================================================
A simulation of a machine's thermal state. The RL agent learns
to keep temperature stable by choosing the right action at the
right time — without overcooling (wasteful) or undercooling (dangerous).

State space (6 dimensions):
  [0] current_temp         — current temperature in °C (normalized 0–1)
  [1] predicted_temp_10min — LSTM prediction 10 min ahead (placeholder: rule-based)
  [2] anomaly_score        — Autoencoder reconstruction error (0.0–1.0)
  [3] machine_type_id      — 0=CPU, 1=motor, 2=server, 3=engine
  [4] time_since_action    — seconds since last non-null action (normalized)
  [5] urgency_score        — combination of anomaly + rate of change

Actions (5 discrete):
  0 — do nothing
  1 — increase fan speed by 20%
  2 — reduce load by 15% (throttle)
  3 — send maintenance alert
  4 — emergency shutdown

Reward:
  +1.0  — temperature stayed in safe band this timestep
  -2.0  — intervention triggered when temperature was stable and safe
  -10.0 — temperature exceeded threshold OR machine damage
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Machine profiles ───────────────────────────────────────────────────────────
# Each machine type has different thermal characteristics
MACHINE_PROFILES = {
    0: {"name": "CPU",    "safe_max": 85.0, "critical": 100.0, "idle": 45.0, "load": 80.0, "heat_rate": 2.5},
    1: {"name": "Motor",  "safe_max": 80.0, "critical":  95.0, "idle": 40.0, "load": 70.0, "heat_rate": 1.8},
    2: {"name": "Server", "safe_max": 75.0, "critical":  90.0, "idle": 35.0, "load": 65.0, "heat_rate": 1.5},
    3: {"name": "Engine", "safe_max": 95.0, "critical": 110.0, "idle": 70.0, "load": 100.0, "heat_rate": 3.8},
}


class ThermalEnv(gym.Env):
    """
    Thermal management environment for a single machine.
    Simulates realistic temperature dynamics including:
    - Load spikes (random workload surges)
    - Ambient temperature variation
    - Cooling lag (fan doesn't cool instantly)
    - Degradation over time (machine runs hotter as it ages)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, machine_type_id: int = 0, max_steps: int = 500, render_mode=None):
        super().__init__()

        assert machine_type_id in MACHINE_PROFILES, f"machine_type_id must be 0–3, got {machine_type_id}"
        self.machine_type_id = machine_type_id
        self.profile = MACHINE_PROFILES[machine_type_id]
        self.max_steps = max_steps
        self.render_mode = render_mode

        # ── Action and observation spaces ──────────────────────────────────────
        self.action_space = spaces.Discrete(5)

        # All 6 state dimensions normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32
        )

        # ── Internal state ─────────────────────────────────────────────────────
        self.temp: float = 0.0
        self.fan_speed: float = 0.0              # 0.0–1.0
        self.load_level: float = 0.0             # 0.0–1.0 (simulated workload)
        self.step_count: int = 0
        self.steps_since_action: int = 0
        self.damage_accumulated: float = 0.0     # accumulates when temp > safe_max
        self.temp_history: list[float] = []      # last 12 readings for trend calculation

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        profile = self.profile
        # Start at a random temperature in the idle-to-60%load range
        self.temp = self.np_random.uniform(profile["idle"], profile["idle"] + 15.0)
        self.fan_speed = 0.3           # starts at 30%
        self.load_level = self.np_random.uniform(0.3, 0.7)
        self.step_count = 0
        self.steps_since_action = 0
        self.damage_accumulated = 0.0
        self.temp_history = [self.temp] * 12

        obs = self._get_obs()
        info = {"machine": profile["name"], "temp": self.temp}
        return obs, info

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        profile = self.profile
        prev_temp = self.temp
        action_was_intervention = action in [1, 2, 3, 4]

        # ── Apply action effects ───────────────────────────────────────────────
        if action == 1:   # increase fan
            self.fan_speed = min(1.0, self.fan_speed + 0.20)
            self.steps_since_action = 0
        elif action == 2: # throttle load
            self.load_level = max(0.1, self.load_level - 0.15)
            self.steps_since_action = 0
        elif action == 3: # maintenance alert (no immediate physical effect)
            self.steps_since_action = 0
        elif action == 4: # emergency shutdown (load drops to near-zero)
            self.load_level = 0.05
            self.fan_speed = 1.0
            self.steps_since_action = 0
        else:             # do nothing
            self.steps_since_action += 1
            # Fan speed drifts back toward baseline over time when idle
            self.fan_speed = max(0.2, self.fan_speed - 0.02)

        # ── Simulate temperature dynamics ──────────────────────────────────────
        # Random load spike (10% chance per step)
        if self.np_random.random() < 0.10:
            self.load_level = min(1.0, self.load_level + self.np_random.uniform(0.1, 0.3))

        # Load naturally decays toward baseline
        self.load_level = max(0.2, self.load_level - 0.01)

        # Temperature update:
        # heat_in = load × machine heat rate
        # heat_out = fan speed × cooling coefficient
        # ambient noise = small random fluctuation
        heat_in  = self.load_level * profile["heat_rate"]
        heat_out = self.fan_speed  * profile["heat_rate"] * 1.2   # fan slightly overpowers
        noise    = self.np_random.normal(0, 0.3)
        self.temp = self.temp + heat_in - heat_out + noise
        self.temp = np.clip(self.temp, 15.0, profile["critical"] + 10.0)

        # Track history for trend calculation
        self.temp_history.append(self.temp)
        if len(self.temp_history) > 12:
            self.temp_history.pop(0)

        # Accumulate damage above safe threshold
        if self.temp > profile["safe_max"]:
            self.damage_accumulated += (self.temp - profile["safe_max"]) * 0.01

        self.step_count += 1

        # ── Compute reward ─────────────────────────────────────────────────────
        safe_max = profile["safe_max"]
        temp_was_safe = prev_temp < (safe_max * 0.70)  # was below 70% of threshold

        if self.temp >= profile["critical"] or self.damage_accumulated > 5.0:
            reward = -10.0
            terminated = True
        elif action_was_intervention and temp_was_safe:
            # Penalise unnecessary intervention — the key fix from the original plan
            reward = -2.0
            terminated = False
        elif self.temp <= safe_max:
            reward = +1.0
            terminated = False
        else:
            # Above safe_max but below critical — partial penalty
            excess = (self.temp - safe_max) / (profile["critical"] - safe_max)
            reward = -excess * 3.0
            terminated = False

        truncated = (self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "temp": round(self.temp, 2),
            "fan_speed": round(self.fan_speed, 2),
            "load": round(self.load_level, 2),
            "damage": round(self.damage_accumulated, 3),
            "action_name": ["do-nothing", "fan+", "throttle", "alert", "shutdown"][action],
        }

        if self.render_mode == "human":
            self._render_frame(info)

        return obs, reward, terminated, truncated, info

    # ── Observation builder ────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        profile = self.profile

        # 1. Normalized current temperature
        temp_norm = (self.temp - 15.0) / (profile["critical"] - 15.0)

        # 2. Predicted temperature in 10 steps (simple linear trend extrapolation)
        #    Will be replaced by the real LSTM in Stage 2
        if len(self.temp_history) >= 6:
            trend = (self.temp_history[-1] - self.temp_history[-6]) / 5.0
            predicted = self.temp + (trend * 10)
        else:
            predicted = self.temp
        pred_norm = np.clip((predicted - 15.0) / (profile["critical"] - 15.0), 0, 1)

        # 3. Anomaly score (placeholder: ratio of temp to safe threshold)
        #    Will be replaced by the real Autoencoder in Stage 2
        anomaly = np.clip((self.temp - profile["idle"]) / (profile["safe_max"] - profile["idle"]), 0, 1)

        # 4. Machine type ID (normalized to 0–1)
        machine_norm = self.machine_type_id / 3.0

        # 5. Time since last action (normalized, max 100 steps)
        time_norm = np.clip(self.steps_since_action / 100.0, 0, 1)

        # 6. Urgency score: blend of anomaly + rate of change
        if len(self.temp_history) >= 3:
            rate = (self.temp_history[-1] - self.temp_history[-3]) / 2.0
            rate_norm = np.clip(rate / 5.0, 0, 1)
        else:
            rate_norm = 0.0
        urgency = np.clip(0.6 * anomaly + 0.4 * rate_norm, 0, 1)

        obs = np.array([
            temp_norm, pred_norm, anomaly,
            machine_norm, time_norm, urgency
        ], dtype=np.float32)

        return obs

    # ── Render ────────────────────────────────────────────────────────────────
    def _render_frame(self, info):
        bar_len = 30
        temp_pct = (self.temp - 15) / (self.profile["critical"] - 15)
        filled = int(temp_pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"[{self.machine_type_id}:{self.profile['name']:6s}] "
            f"Temp: {self.temp:6.1f}°C [{bar}] "
            f"Fan: {self.fan_speed*100:4.0f}% "
            f"Load: {self.load_level*100:4.0f}% "
            f"Action: {info['action_name']:10s} "
            f"Step: {self.step_count:4d}"
        )