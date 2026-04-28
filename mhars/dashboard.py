"""
MHARS Dashboard
================
Live terminal dashboard. Shows real-time temperature, anomaly
score, PPO action, and LLM alert in a clean updating display.

Usage:
    from mhars.dashboard import Dashboard
    dash = Dashboard(system)
    dash.start(source="simulation")   # or "sensor" for real hardware
"""

import os, time, sys
import numpy as np
from mhars.config import Config


class Dashboard:
    """
    Terminal dashboard for MHARS. Updates in place using ANSI codes.

    Args:
        system:      a configured MHARS instance
        refresh_hz:  how many times per second to update (default 2)
    """

    COLORS = {
        "NORMAL":   "\033[32m",   # green
        "WATCH":    "\033[33m",   # yellow
        "WARN":     "\033[33m",   # yellow
        "ALERT":    "\033[91m",   # bright red
        "CRITICAL": "\033[31m",   # red
        "RESET":    "\033[0m",
        "BOLD":     "\033[1m",
        "DIM":      "\033[2m",
        "CYAN":     "\033[36m",
        "BLUE":     "\033[34m",
    }

    def __init__(self, system, refresh_hz: float = 0.5):
        self.system     = system
        self.refresh_hz = refresh_hz
        self._history   = []   # last 20 results

    def start(self, source: str = "simulation", duration_s: int = 60):
        """
        Run the dashboard.

        Args:
            source:     "simulation" uses ThermalEnv, "sensor" reads real hardware
            duration_s: how long to run (seconds). 0 = run forever.
        """
        print("\033[2J\033[H", end="")  # clear screen

        t0        = time.time()
        step      = 0
        gen       = self._make_generator(source)

        try:
            while True:
                temp = next(gen)
                result = self.system.run(temp)
                self._history.append(result)
                if len(self._history) > 20:
                    self._history.pop(0)

                self._render(result, step)
                step += 1

                if duration_s and (time.time() - t0) > duration_s:
                    break

                time.sleep(1.0 / self.refresh_hz)

        except KeyboardInterrupt:
            print("\n\n[Dashboard] Stopped by user.\n")

    # ── Rendering ──────────────────────────────────────────────────────────────
    def _render(self, result, step: int):
        C = self.COLORS
        print("\033[H", end="")   # move cursor to top without clearing

        profile  = Config.MACHINE_PROFILES[self.system.machine_type_id]
        severity = self._severity(result.urgency)
        sev_col  = C.get(severity, C["RESET"])
        bar      = self._temp_bar(result.current_temp, profile)

        lines = [
            f"{C['BOLD']}╔══════════════════════════════════════════════════════╗{C['RESET']}",
            f"{C['BOLD']}║  MHARS — {self.system.machine_name:<10}  Live Monitor{' '*16}║{C['RESET']}",
            f"{C['BOLD']}╚══════════════════════════════════════════════════════╝{C['RESET']}",
            "",
            f"  {C['CYAN']}Temperature{C['RESET']}  {result.current_temp:>6.1f}°C  {bar}",
            f"  {C['CYAN']}Predicted   {C['RESET']}  {result.lstm_prediction:>6.1f}°C  (10 min forecast)",
            f"  {C['CYAN']}Anomaly     {C['RESET']}  {result.anomaly_score:>6.3f}    {self._score_bar(result.anomaly_score, 20)}",
            f"  {C['CYAN']}Urgency     {C['RESET']}  {result.urgency:>6.3f}    {self._score_bar(result.urgency, 20)}",
            f"  {C['CYAN']}Context     {C['RESET']}  {result.context_score:>6.3f}",
            "",
            f"  {C['BOLD']}Decision{C['RESET']}",
            f"  Route   →  {C['BLUE']}{result.route.upper():<6}{C['RESET']}",
            f"  Action  →  {sev_col}{C['BOLD']}{result.action.upper():<12}{C['RESET']}  [{severity}]",
            f"  Latency →  {result.latency_ms:.1f} ms",
            "",
            f"  {C['BOLD']}Alert{C['RESET']}  ({result.llm_source})",
            self._wrap_text(result.alert, width=52, indent=2),
            "",
            f"  {C['DIM']}Step {step:>5}  |  {time.strftime('%H:%M:%S')}{C['RESET']}",
            "",
            f"  {C['BOLD']}Temperature history (last 20 readings){C['RESET']}",
            "  " + self._sparkline(),
            "",
            f"  {C['DIM']}Press Ctrl+C to stop{C['RESET']}",
        ]

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def _severity(self, urgency: float) -> str:
        if urgency >= 0.85:  return "CRITICAL"
        if urgency >= 0.65:  return "ALERT"
        if urgency >= 0.45:  return "WARN"
        if urgency >= 0.25:  return "WATCH"
        return "NORMAL"

    def _temp_bar(self, temp: float, profile: dict, width: int = 20) -> str:
        ratio   = (temp - 15) / (profile["critical"] - 15)
        filled  = int(np.clip(ratio * width, 0, width))
        safe_f  = int((profile["safe_max"] - 15) / (profile["critical"] - 15) * width)
        bar = ""
        for i in range(width):
            char = "█" if i < filled else "░"
            if i < safe_f:
                bar += f"\033[32m{char}\033[0m"
            elif i < int(width * 0.9):
                bar += f"\033[33m{char}\033[0m"
            else:
                bar += f"\033[31m{char}\033[0m"
        pct = int(ratio * 100)
        return f"[{bar}] {pct:3d}%"

    def _score_bar(self, score: float, width: int = 20) -> str:
        filled = int(np.clip(score * width, 0, width))
        color  = "\033[32m" if score < 0.4 else ("\033[33m" if score < 0.7 else "\033[31m")
        return f"{color}{'█' * filled}{'░' * (width - filled)}\033[0m"

    def _sparkline(self) -> str:
        if not self._history:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        temps  = [r.current_temp for r in self._history]
        lo, hi = min(temps), max(temps)
        if hi == lo:
            return blocks[4] * len(temps)
        result = ""
        for t in temps:
            idx = int((t - lo) / (hi - lo) * (len(blocks) - 1))
            result += blocks[idx]
        return result + f"  {lo:.0f}°–{hi:.0f}°C"

    def _wrap_text(self, text: str, width: int = 52, indent: int = 2) -> str:
        if not text:
            return " " * indent + "(no alert)"
        words  = text.split()
        lines  = []
        line   = " " * indent
        for word in words:
            if len(line) + len(word) + 1 > width:
                lines.append(line)
                line = " " * indent + word
            else:
                line += (" " if len(line) > indent else "") + word
        lines.append(line)
        return "\n".join(lines)

    # ── Data generators ────────────────────────────────────────────────────────
    def _make_generator(self, source: str):
        if source == "simulation":
            return self._simulation_generator()
        elif source == "sensor":
            return self._sensor_generator()
        else:
            raise ValueError(f"Unknown source: {source}. Use 'simulation' or 'sensor'.")

    def _simulation_generator(self):
        """Generate realistic temperature from ThermalEnv simulation."""
        from stage1_simulation.gym_env import ThermalEnv
        env = ThermalEnv(
            machine_type_id = self.system.machine_type_id,
            max_steps       = 10_000
        )
        obs, info = env.reset()
        while True:
            action = 0  # let MHARS decide
            obs, _, terminated, truncated, info = env.step(action)
            yield info["temp"]
            if terminated or truncated:
                obs, info = env.reset()

    def _sensor_generator(self):
        """
        Read from real DS18B20 sensor on Raspberry Pi.
        Requires: pip install adafruit-circuitpython-ds18x20
        Falls back to simulation if hardware not found.
        """
        try:
            import board
            import adafruit_ds18x20
            import adafruit_onewire.bus
            ow_bus = adafruit_onewire.bus.OneWireBus(board.D4)
            devices = ow_bus.scan()
            if not devices:
                raise RuntimeError("No DS18B20 sensor found on GPIO4")
            sensor = adafruit_ds18x20.DS18X20(ow_bus, devices[0])
            print("[Dashboard] DS18B20 sensor connected ✓")
            while True:
                yield sensor.temperature
                time.sleep(1.0)
        except Exception as e:
            print(f"[Dashboard] Sensor not available ({e})")
            print("[Dashboard] Falling back to simulation")
            yield from self._simulation_generator()