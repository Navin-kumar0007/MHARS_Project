"""
MHARS Dashboard — FastAPI Backend (v2.0)
=========================================
Provides WebSocket streaming and REST endpoints for the
interactive monitoring dashboard. Exposes every internal
variable of the MHARS AI pipeline for full transparency.

Endpoints:
  WS   /ws/telemetry           Live 1Hz telemetry stream (expanded payload)
  GET  /api/system_status      Machine profile + loaded model status
  POST /api/inject_anomaly     Inject 1 of 5 anomaly types for live demo
  POST /api/switch_machine     Switch between CPU/Motor/Server/Engine
  POST /api/reset              Reset environment to idle state
  GET  /api/action_history     Last 100 RL agent actions
  GET  /api/alert_history      Last 50 LLM-generated alerts
"""

import sys
import os
import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

# Ensure we can import mhars modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from mhars.core import MHARS
from mhars.system_health import SystemHealthMonitor
from stage1_simulation.gym_env import ThermalEnv, MACHINE_PROFILES

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MHARS Dashboard API",
    description="Real-time monitoring backend for the MHARS Digital Twin",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global State ───────────────────────────────────────────────────────────────
class SystemState:
    """Holds the live simulation state, history buffers, and anomaly injection."""

    def __init__(self, machine_type_id: int = 1):
        self.machine_type_id = machine_type_id
        self.mhars = MHARS(machine_type_id=machine_type_id)
        self.env = ThermalEnv(machine_type_id=machine_type_id)
        self.env.reset()

        # Anomaly injection state
        self.anomaly_injection: Optional[str] = None
        self.anomaly_ticks_remaining: int = 0  # For persistent anomalies

        # History buffers (server-side rolling storage)
        self.action_history: deque = deque(maxlen=100)
        self.alert_history: deque = deque(maxlen=50)
        self.telemetry_history: deque = deque(maxlen=200)

        # Live mode: read real hardware temp instead of simulation
        self.live_mode: bool = False

    def reinitialize(self, machine_type_id: int):
        """Switch to a different machine type — reinitializes all AI models."""
        self.machine_type_id = machine_type_id
        self.mhars = MHARS(machine_type_id=machine_type_id)
        self.env = ThermalEnv(machine_type_id=machine_type_id)
        self.env.reset()
        self.anomaly_injection = None
        self.anomaly_ticks_remaining = 0
        self.action_history.clear()
        self.alert_history.clear()
        self.telemetry_history.clear()


state = SystemState(machine_type_id=1)  # Default: Motor


def read_hardware_temp() -> float:
    """Read real CPU temperature / activity from hardware.
    
    Strategy:
    1. Try psutil.sensors_temperatures() (Linux)
    2. Try psutil.cpu_percent() mapped to temp range (cross-platform)
    3. Fallback: use os.getloadavg() mapped to temp range (macOS/Linux)
    """
    # Method 1: psutil sensors (Linux)
    if PSUTIL_AVAILABLE:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except (AttributeError, Exception):
            pass

        # Method 2: psutil CPU percent → temp mapping
        try:
            cpu_pct = psutil.cpu_percent(interval=0.1)
            base_temp = 36.0
            load_heat = cpu_pct * 0.48
            return round(base_temp + load_heat + np.random.normal(0, 0.3), 2)
        except Exception:
            pass

    # Method 3: os.getloadavg (macOS / Linux native, no pip needed)
    try:
        load_1min = os.getloadavg()[0]  # 1-minute load average
        cpu_count = os.cpu_count() or 4
        load_pct = min(100, (load_1min / cpu_count) * 100)
        base_temp = 36.0
        load_heat = load_pct * 0.48
        return round(base_temp + load_heat + np.random.normal(0, 0.3), 2)
    except Exception:
        return 42.0


# ── Request Models ─────────────────────────────────────────────────────────────
class AnomalyRequest(BaseModel):
    type: str  # temperature_spike, bearing_wear, fan_blockage, sensor_drift, power_surge


class MachineRequest(BaseModel):
    machine_type_id: int  # 0=CPU, 1=Motor, 2=Server, 3=Engine


# ── Anomaly Simulation Logic ──────────────────────────────────────────────────
# Each anomaly type defines: temp_delta per tick, duration in ticks, description
ANOMALY_PROFILES = {
    "temperature_spike": {
        "delta": 8.0,
        "duration": 1,       # One-off massive spike
        "description": "Sudden thermal overload (+8°C instant)",
    },
    "bearing_wear": {
        "delta": 1.5,
        "duration": 10,      # Gradual friction heating over 10 seconds
        "description": "Worn bearing causing gradual friction heat (+1.5°C/s for 10s)",
    },
    "fan_blockage": {
        "delta": 0.8,
        "duration": 15,      # Cooling failure persists for 15 seconds
        "description": "Blocked cooling fan — heat dissipation failure (+0.8°C/s for 15s)",
    },
    "sensor_drift": {
        "delta": 0.0,        # No real temp change, but noise is added
        "duration": 12,
        "noise": 3.0,        # ±3°C noise injected into readings
        "description": "Faulty sensor adding ±3°C random noise (12s duration)",
    },
    "power_surge": {
        "delta": 12.0,
        "duration": 1,       # One-off extreme spike
        "description": "Electrical fault causing extreme thermal spike (+12°C instant)",
    },
}


def apply_anomaly_to_temp(current_temp: float) -> float:
    """Apply the active anomaly injection to the temperature."""
    if state.anomaly_injection is None or state.anomaly_ticks_remaining <= 0:
        state.anomaly_injection = None
        state.anomaly_ticks_remaining = 0
        # Normal random variation
        return current_temp + (np.random.random() - 0.48) * 0.8

    profile = ANOMALY_PROFILES.get(state.anomaly_injection, {})
    delta = profile.get("delta", 0.0)
    noise = profile.get("noise", 0.0)

    # Apply temperature delta
    current_temp += delta

    # Apply sensor noise if applicable
    if noise > 0:
        current_temp += np.random.uniform(-noise, noise)

    # Decrement remaining ticks
    state.anomaly_ticks_remaining -= 1
    if state.anomaly_ticks_remaining <= 0:
        state.anomaly_injection = None

    return current_temp


# ── REST Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/system_status")
async def get_system_status():
    """Returns the current machine profile, model status, and available anomalies."""
    profile = MACHINE_PROFILES[state.machine_type_id]
    return {
        "machine_type_id": state.machine_type_id,
        "machine_profile": profile,
        "models_loaded": {
            "isolation_forest": state.mhars._if_model is not None,
            "lstm": state.mhars._lstm is not None,
            "autoencoder": state.mhars._ae_model is not None,
            "vibration_detector": state.mhars._vib_model is not None,
            "ppo_agent": state.mhars._ppo is not None,
            "phi3_llm": state.mhars._llm_gen is not None and getattr(state.mhars._llm_gen, 'use_llm', False),
        },
        "available_machines": {
            mid: mp["name"] for mid, mp in MACHINE_PROFILES.items()
        },
        "available_anomalies": {
            k: v["description"] for k, v in ANOMALY_PROFILES.items()
        },
        "active_anomaly": state.anomaly_injection,
    }


@app.post("/api/inject_anomaly")
async def inject_anomaly(req: AnomalyRequest):
    """Inject one of 5 anomaly types for live demonstration."""
    if req.type not in ANOMALY_PROFILES:
        return {
            "status": "error",
            "message": f"Unknown anomaly type '{req.type}'. "
                       f"Available: {list(ANOMALY_PROFILES.keys())}",
        }
    profile = ANOMALY_PROFILES[req.type]
    state.anomaly_injection = req.type
    state.anomaly_ticks_remaining = profile["duration"]
    return {
        "status": "success",
        "anomaly": req.type,
        "description": profile["description"],
        "duration_ticks": profile["duration"],
    }


@app.post("/api/switch_machine")
async def switch_machine(req: MachineRequest):
    """Switch to a different machine type. Reinitializes the entire AI pipeline."""
    if req.machine_type_id not in MACHINE_PROFILES:
        return {
            "status": "error",
            "message": f"Invalid machine_type_id. Must be 0-3, got {req.machine_type_id}",
        }
    state.reinitialize(req.machine_type_id)
    profile = MACHINE_PROFILES[req.machine_type_id]
    return {
        "status": "success",
        "message": f"Switched to {profile['name']}. All AI models reloaded.",
        "machine_profile": profile,
    }


@app.post("/api/reset")
async def reset_system():
    """Reset the environment to idle state without changing machine type."""
    state.env.reset()
    state.anomaly_injection = None
    state.anomaly_ticks_remaining = 0
    state.action_history.clear()
    state.alert_history.clear()
    state.telemetry_history.clear()
    return {"status": "success", "message": "System reset to idle."}


@app.get("/api/action_history")
async def get_action_history():
    """Return the last 100 RL agent actions with timestamps."""
    return {"actions": list(state.action_history)}


@app.get("/api/alert_history")
async def get_alert_history():
    """Return the last 50 LLM-generated alerts."""
    return {"alerts": list(state.alert_history)}


@app.post("/api/toggle_mode")
async def toggle_mode():
    """Toggle between live hardware mode and simulation demo mode."""
    state.live_mode = not state.live_mode
    mode = "live" if state.live_mode else "demo"
    # When switching to live, reinit MHARS as CPU (the actual machine)
    if state.live_mode:
        state.reinitialize(0)  # CPU profile for real computer
    state.action_history.clear()
    state.alert_history.clear()
    state.telemetry_history.clear()
    return {
        "status": "success",
        "mode": mode,
        "message": f"Switched to {mode} mode." + (
            " Reading real CPU temperature." if state.live_mode
            else " Using simulated thermal environment."
        ),
        "psutil_available": PSUTIL_AVAILABLE,
    }


@app.get("/api/mode")
async def get_mode():
    """Return current mode."""
    return {"mode": "live" if state.live_mode else "demo", "live_mode": state.live_mode}


@app.get("/api/system_health")
async def get_system_health():
    """Return hardware metrics based on the current machine."""
    return SystemHealthMonitor.snapshot(state.machine_type_id)


# ── WebSocket Telemetry Stream ─────────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """
    Streams live telemetry data at 1Hz with the FULL expanded payload.
    Every internal AI variable is exposed for complete dashboard transparency.
    """
    await websocket.accept()
    print("[WebSocket] Client connected.")

    try:
        while True:
            # ── Step 1: Get temperature ────────────────────────────────────
            if state.live_mode:
                # LIVE MODE: read real hardware CPU temperature
                current_temp = read_hardware_temp()
            else:
                # DEMO MODE: simulated environment + anomaly injection
                state.env.temp = apply_anomaly_to_temp(state.env.temp)
                current_temp = state.env.temp

            # ── Step 2: Run the complete MHARS AI pipeline ─────────────────
            # Run in threadpool to prevent blocking the WebSocket event loop during heavy ML inferences
            result = await run_in_threadpool(state.mhars.run, temp_celsius=current_temp, sync_alert=True)

            # ── Step 3: Apply PPO action feedback to the simulation ────────
            action_effects = {
                "throttle": -2.0,
                "increase-fan": -1.0,
                "fan+": -1.0,
                "alert": -0.3,
                "emergency-shutdown": -10.0,
                "shutdown": -10.0,
            }
            if result.action in action_effects:
                state.env.temp += action_effects[result.action]

            # Keep temperature physically reasonable
            state.env.temp = max(20.0, state.env.temp)

            # ── Step 4: Build the EXPANDED payload ─────────────────────────
            # Extract per-model scores from metadata
            meta = result.metadata or {}

            payload = {
                # Core readings
                "timestamp": result.timestamp,
                "machine_type": result.machine_type,
                "machine_type_id": state.machine_type_id,
                "current_temp": round(result.current_temp, 2),
                "lstm_prediction": round(result.lstm_prediction, 2),

                # Individual model scores (Gap G1 fix)
                "if_score": round(meta.get("if_score", 0.0), 4),
                "lstm_score": round(meta.get("lstm_score", 0.0), 4),
                "ae_score": round(meta.get("ae_score", 0.0), 4),
                "vib_score": round(meta.get("vib_score", 0.0), 4),

                # Fused outputs
                "anomaly_score": round(result.anomaly_score, 4),
                "context_score": round(result.context_score, 4),
                "urgency": round(result.urgency, 4),

                # RL Agent decision
                "action": result.action,
                "route": result.route,
                "latency_ms": round(result.latency_ms, 2),

                # LLM output
                "alert": result.alert,
                "llm_source": result.llm_source,

                # PPO observation vector (6 dimensions)
                "raw_obs": [round(v, 4) for v in result.raw_obs],

                # Active anomaly injection info
                "active_anomaly": state.anomaly_injection if not state.live_mode else None,
                "anomaly_ticks_remaining": state.anomaly_ticks_remaining if not state.live_mode else 0,

                # Mode indicator
                "live_mode": state.live_mode,
                
                # Full system health (dynamic for all modes)
                "system_health": SystemHealthMonitor.snapshot(state.machine_type_id),

                # Machine thresholds (for gauge rendering)
                "thresholds": {
                    "idle": MACHINE_PROFILES[state.machine_type_id].get("idle", 40.0),
                    "safe_max": MACHINE_PROFILES[state.machine_type_id].get("safe_max", 80.0),
                    "critical": MACHINE_PROFILES[state.machine_type_id].get("critical", 95.0),
                },
                
                # Full metadata dictionary (contains RUL, XAI contributions, fault_type)
                "metadata": meta,
            }

            # ── Step 5: Store in history buffers ───────────────────────────
            action_entry = {
                "timestamp": result.timestamp,
                "action": result.action,
                "urgency": round(result.urgency, 4),
                "route": result.route,
                "temp": round(result.current_temp, 2),
                "latency_ms": round(result.latency_ms, 2),
            }
            state.action_history.append(action_entry)

            alert_entry = {
                "timestamp": result.timestamp,
                "alert": result.alert,
                "source": result.llm_source,
                "urgency": round(result.urgency, 4),
                "severity": (
                    "critical" if result.urgency >= 0.8
                    else "warning" if result.urgency >= 0.5
                    else "normal"
                ),
            }
            state.alert_history.append(alert_entry)

            state.telemetry_history.append(payload)

            # ── Step 6: Send to frontend ───────────────────────────────────
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)  # Stream at 1Hz

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        import traceback
        traceback.print_exc()
