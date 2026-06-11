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
import json
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from mhars.core import MHARS
from mhars.system_health import SystemHealthMonitor
from stage1_simulation.gym_env import ThermalEnv, MACHINE_PROFILES
from mhars.share_links import ShareLinkManager
from mhars.report_generator import ReportGenerator

app = FastAPI(
    title="MHARS API",
    description="Production-grade thermal monitoring and AI control API",
    version="2.1.0"
)

# ── Runtime configuration (env-driven) ──────────────────────────────────────
def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

MHARS_API_KEY = os.environ.get("MHARS_API_KEY", "").strip()
MHARS_REQUIRE_AUTH = _env_flag("MHARS_REQUIRE_AUTH", False)
MHARS_SYNTHETIC_MODE = _env_flag("MHARS_SYNTHETIC_MODE", False)

# ── CORS (#11 fix: configurable origins, no wildcard in production) ──────────
CORS_ORIGINS = os.environ.get("MHARS_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,null").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

share_manager = ShareLinkManager()

from mhars.auth import authenticate_user, create_access_token, decode_access_token, get_user, ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    # Dev mode: auth disabled and no API key set → act as a default admin.
    if not MHARS_REQUIRE_AUTH and not MHARS_API_KEY:
        return {"username": "dev", "role": "admin"}
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def require_role(allowed_roles: list[str]):
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles and current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Operation not permitted")
        return current_user
    return role_checker

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"]}

@app.get("/api/auth/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user


# ── User management (admin only) ─────────────────────────────────────────────
from mhars.auth import list_users as _list_users, update_user_role as _update_role, \
    delete_user as _delete_user, create_user as _create_user

class NewUserRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"

class RoleUpdateRequest(BaseModel):
    role: str

@app.get("/api/users", dependencies=[Depends(require_role(["admin"]))])
async def get_users():
    return {"users": _list_users()}

@app.post("/api/users", dependencies=[Depends(require_role(["admin"]))])
async def add_user(req: NewUserRequest):
    if req.role not in ("admin", "operator", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role")
    if not _create_user(req.username, req.password, req.role):
        raise HTTPException(status_code=409, detail="User already exists")
    return {"status": "success"}

@app.post("/api/users/{username}/role", dependencies=[Depends(require_role(["admin"]))])
async def change_user_role(username: str, req: RoleUpdateRequest):
    if req.role not in ("admin", "operator", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role")
    if not _update_role(username, req.role):
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success"}

@app.delete("/api/users/{username}", dependencies=[Depends(require_role(["admin"]))])
async def remove_user(username: str, current_user: dict = Depends(get_current_user)):
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    if not _delete_user(username):
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success"}


def verify_ws_token(token: str) -> bool:
    """Verify WebSocket connection credentials.

    Accepts either a valid JWT or the configured static API key. When auth is
    not required and no API key is set (dev mode), connections are allowed.
    """
    if not MHARS_REQUIRE_AUTH and not MHARS_API_KEY:
        return True
    if MHARS_API_KEY and token == MHARS_API_KEY:
        return True
    return decode_access_token(token) is not None

@app.on_event("startup")
async def startup_event():
    # Validate CORS origins
    for origin in CORS_ORIGINS:
        origin = origin.strip()
        if origin and not origin.startswith("http"):
            print(f"  [WARN] Malformed CORS origin: '{origin}'. Expected http(s)://...")

    # Security Warning
    if not MHARS_API_KEY and not MHARS_REQUIRE_AUTH:
        import socket
        try:
            print("\n" + "!"*80)
            print("  SECURITY WARNING: MHARS API is running WITHOUT authentication.")
            print("  If this server is accessible over the network, anyone can control it.")
            print("  To secure it, set MHARS_API_KEY and MHARS_REQUIRE_AUTH=true.")
            print("!"*80 + "\n")
        except:
            pass
    
    if MHARS_SYNTHETIC_MODE:
        print("[INIT] Running in SYNTHETIC MODE (forcing multi-modal proxies)")
    
    # Note: Rate limiting on /api/inject_anomaly uses in-memory state.
    # If running multiple Uvicorn workers, each worker has its own counter.
    # For multi-worker deployments, use a shared store (Redis/file-based).

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
        self._live_temp_ema: Optional[float] = None  # thermal-mass smoothing of load-driven temp

    @property
    def machine_profile(self) -> Dict[str, Any]:
        return MACHINE_PROFILES[self.machine_type_id]

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

# Rate limiter for anomaly injection (#13)
_last_injection_time: float = 0.0
_INJECTION_COOLDOWN: float = 5.0  # seconds between injections


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
@app.get("/api/status")
async def get_status():
    """Lightweight liveness probe (used by Docker healthcheck)."""
    return {"status": "ok", "live_mode": state.live_mode}


@app.get("/api/system_status")
async def get_system_status():
    """Returns the current machine profile, model status, and available anomalies."""
    profile = MACHINE_PROFILES[state.machine_type_id]
    m = state.mhars
    llm_live = m._llm_gen is not None and getattr(m._llm_gen, 'use_llm', False)

    # P4.1: per-model provenance. ok=True when the trained model is the live path;
    # ok=False means a (functional) fallback is in use — surfaced as a badge so
    # placeholder/degraded outputs are never mistaken for the real model.
    def entry(ok, detail):
        return {"ok": bool(ok), "detail": detail}

    model_status = {
        "Forecast (LSTM)":   entry(m._lstm is not None,
                                   ("quantile multi-horizon" if getattr(m, "_lstm_qmode", False)
                                    else (m._lstm_version or "fallback"))),
        "Anomaly AE":        entry(m._ae_model is not None,
                                   "calibrated (EVT)" if getattr(m, "_ae_calib", None) else
                                   ("loaded" if m._ae_model is not None else "fallback")),
        "Vibration":         entry(m._vib_model is not None,
                                   "calibrated (EVT)" if getattr(m, "_vib_calib", None) else
                                   ("loaded" if m._vib_model is not None else "fallback")),
        "Isolation Forest":  entry(getattr(m, "_if_has_retrained", False),
                                   "serving-distribution" if getattr(m, "_if_has_retrained", False) else "warmup proxy"),
        "Fusion":            entry(getattr(m, "_learned_fusion", None) is not None,
                                   "learned attention" if getattr(m, "_learned_fusion", None) else "rule-based"),
        "RUL Predictor":     entry(getattr(m, "_rul_model", None) is not None,
                                   "learned" if getattr(m, "_rul_model", None) else "physics fallback"),
        "Fault Classifier":  entry(getattr(m, "_fault_clf", None) is not None,
                                   "supervised" if getattr(m, "_fault_clf", None) else "rule-based fingerprint"),
        "PPO Agent":         entry(m._ppo is not None, "loaded" if m._ppo is not None else "rule-based"),
        "LLM (Phi-3)":       entry(llm_live, "on-device" if llm_live else "template alerts"),
    }
    degraded = sorted([k for k, v in model_status.items() if not v["ok"]])

    return {
        "machine_type_id": state.machine_type_id,
        "machine_profile": profile,
        "models_loaded": {
            "isolation_forest": state.mhars._if_model is not None,
            "lstm": state.mhars._lstm is not None,
            "autoencoder": state.mhars._ae_model is not None,
            "vibration_detector": state.mhars._vib_model is not None,
            "ppo_agent": state.mhars._ppo is not None,
            "phi3_llm": llm_live,
        },
        "model_status": model_status,
        "models_degraded": degraded,
        "available_machines": {
            mid: mp["name"] for mid, mp in MACHINE_PROFILES.items()
        },
        "available_anomalies": {
            k: v["description"] for k, v in ANOMALY_PROFILES.items()
        },
        "active_anomaly": state.anomaly_injection,
        "synthetic_mode": MHARS_SYNTHETIC_MODE,
    }


@app.get("/api/model_registry")
async def get_model_registry():
    """X.1: model registry — identity (sha + size + mtime) of each live artifact.
    Lets you confirm exactly which trained weights are deployed."""
    import hashlib
    mdir = os.path.join(os.path.dirname(__file__), "..", "models")
    artifacts = {
        "Forecast (LSTM)": "lstm_v2.pt",
        "Anomaly AE": "autoencoder_lstm_v2.pt",
        "Vibration": "vibration_detector.pt",
        "Isolation Forest": "isolation_forest.pkl",
        "Fusion": "learned_fusion.pt",
        "RUL Predictor": "rul_predictor_v2.pt",
        "Fault Classifier": "fault_classifier.pt",
        "PPO Agent": "ppo_thermal.zip",
    }
    out = []
    for name, fn in artifacts.items():
        path = os.path.join(mdir, fn)
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            st = os.stat(path)
            out.append({"name": name, "file": fn, "present": True,
                        "sha": h.hexdigest()[:12], "size_kb": round(st.st_size / 1024, 1),
                        "modified": st.st_mtime})
        else:
            out.append({"name": name, "file": fn, "present": False,
                        "sha": None, "size_kb": 0, "modified": None})
    return {"registry": out}


@app.get("/api/eval_report")
async def get_eval_report():
    """P4.3: return the offline anomaly-detection evaluation report (if present).
    Generated by tools/eval_anomaly.py — F1/ROC-AUC per detector + per-fault rates."""
    path = os.path.join(os.path.dirname(__file__), "..", "models", "eval_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return {"available": True, "report": json.load(f)}
    return {"available": False, "report": None}


@app.post("/api/inject_anomaly", dependencies=[Depends(require_role(["operator", "admin"]))])
async def inject_anomaly(req: AnomalyRequest):
    """Inject one of 5 anomaly types for live demonstration."""
    # Rate limiting (#13)
    global _last_injection_time
    now = time.time()
    if now - _last_injection_time < _INJECTION_COOLDOWN:
        remaining = round(_INJECTION_COOLDOWN - (now - _last_injection_time), 1)
        return {
            "status": "rate_limited",
            "message": f"Please wait {remaining}s before injecting another anomaly.",
        }
    
    if req.type not in ANOMALY_PROFILES:
        return {
            "status": "error",
            "message": f"Unknown anomaly type '{req.type}'. "
                       f"Available: {list(ANOMALY_PROFILES.keys())}",
        }
    profile = ANOMALY_PROFILES[req.type]
    state.anomaly_injection = req.type
    state.anomaly_ticks_remaining = profile["duration"]
    _last_injection_time = now
    return {
        "status": "success",
        "anomaly": req.type,
        "description": profile["description"],
        "duration_ticks": profile["duration"],
    }


@app.post("/api/switch_machine", dependencies=[Depends(require_role(["operator", "admin"]))])
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


@app.post("/api/reset", dependencies=[Depends(require_role(["operator", "admin"]))])
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


@app.post("/api/toggle_mode", dependencies=[Depends(require_role(["operator", "admin"]))])
async def toggle_mode():
    """Toggle between live hardware mode and simulation demo mode."""
    state.live_mode = not state.live_mode
    mode = "live" if state.live_mode else "demo"
    # When switching to live, reinit MHARS as CPU (the actual machine)
    if state.live_mode:
        state.reinitialize(0)  # CPU profile for real computer
    state._live_temp_ema = None  # reset thermal smoothing on mode change
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


@app.get("/api/registry")
async def get_registry():
    """Return the list of all registered nodes in the federated network."""
    return state.mhars._registry.list_all_nodes()


# ── Share Links Endpoints ──────────────────────────────────────────────────────
class ShareLinkRequest(BaseModel):
    label: str
    expires_in_hours: int = 24

@app.post("/api/share/create", dependencies=[Depends(require_role(["admin", "operator"]))])
async def create_share_link(req: ShareLinkRequest, current_user: dict = Depends(get_current_user)):
    """Create a new public share link."""
    try:
        token = share_manager.create_link(current_user["username"], req.label, req.expires_in_hours)
        return {"status": "success", "token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/share/list", dependencies=[Depends(require_role(["admin", "operator"]))])
async def list_share_links(current_user: dict = Depends(get_current_user)):
    """List all active share links."""
    links = share_manager.list_links()
    # Operators can only see their own links
    if current_user["role"] == "operator":
        links = [link for link in links if link.get("creator") == current_user["username"]]
    return {"status": "success", "links": links}

@app.post("/api/share/revoke/{token}", dependencies=[Depends(require_role(["admin", "operator"]))])
async def revoke_share_link(token: str, current_user: dict = Depends(get_current_user)):
    """Revoke a share link."""
    # Ensure they own it or are admin
    links = share_manager.list_links()
    link_data = next((l for l in links if l["token"] == token), None)
    
    if not link_data:
         raise HTTPException(status_code=404, detail="Link not found")
         
    if current_user["role"] != "admin" and link_data["creator"] != current_user["username"]:
        raise HTTPException(status_code=403, detail="Cannot revoke another user's link")
        
    if share_manager.revoke_link(token):
        return {"status": "success", "message": "Link revoked."}
    raise HTTPException(status_code=404, detail="Link not found")

@app.get("/api/share/validate/{token}")
async def validate_share_link(token: str):
    """Validate a public share token (unauthenticated endpoint)."""
    is_valid = share_manager.validate_and_record_access(token)
    if is_valid:
        return {"status": "success", "valid": True}
    raise HTTPException(status_code=401, detail="Invalid or expired link")


@app.get("/api/share/{token}")
async def get_shared_status(token: str):
    """Public read-only status snapshot for a valid share token (no login)."""
    if not share_manager.validate_and_record_access(token):
        raise HTTPException(status_code=410, detail="Invalid or expired link")

    latest = state.telemetry_history[-1] if state.telemetry_history else {}
    meta = latest.get("metadata", {}) or {}
    profile = MACHINE_PROFILES[state.machine_type_id]
    return {
        "machine": profile.get("name", "Unknown"),
        "current_temp": latest.get("current_temp"),
        "health_score": meta.get("health_score"),
        "health_trend": meta.get("health_trend"),
        "action": latest.get("action"),
        "alert": latest.get("alert"),
        "fault_type": meta.get("fault_type"),
        "thresholds": latest.get("thresholds"),
        "timestamp": latest.get("timestamp"),
    }


# ── Report Generation ──────────────────────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/api/report/html", dependencies=[Depends(require_role(["admin", "operator", "viewer"]))])
async def get_html_report():
    """Generates a downloadable HTML diagnostic report."""
    html_content = ReportGenerator.generate_html_report(state)
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/api/report/download", dependencies=[Depends(require_role(["admin", "operator", "viewer"]))])
async def download_html_report():
    """Alias for /api/report/html — forces a file download in the browser."""
    html_content = ReportGenerator.generate_html_report(state)
    headers = {"Content-Disposition": "attachment; filename=mhars_report.html"}
    return HTMLResponse(content=html_content, status_code=200, headers=headers)


# ── Advanced Analytics Endpoints (Phase 4B) ─────────────────────────────────────
def _latest_meta() -> Dict[str, Any]:
    """Return the metadata dict of the most recent telemetry tick, or {}."""
    if not state.telemetry_history:
        raise HTTPException(status_code=503, detail="No telemetry yet — start the stream first.")
    return state.telemetry_history[-1].get("metadata", {}) or {}


@app.get("/api/health_score", dependencies=[Depends(get_current_user)])
async def get_health_score():
    """Composite 0-100 health score with per-component breakdown."""
    meta = _latest_meta()
    return {
        "score": meta.get("health_score"),
        "trend": meta.get("health_trend"),
        "breakdown": meta.get("health_breakdown", {}),
    }


@app.get("/api/trends", dependencies=[Depends(get_current_user)])
async def get_trends():
    """CUSUM/EWMA statistical trend analysis."""
    meta = _latest_meta()
    return {
        "trend_stats": meta.get("trend_stats", {}),
        "concept_drift_detected": meta.get("concept_drift_detected", False),
    }


@app.get("/api/explainability", dependencies=[Depends(get_current_user)])
async def get_explainability():
    """Feature attribution (XAI) for the most recent decision."""
    meta = _latest_meta()
    return {
        "feature_importance": meta.get("feature_importance", {}),
        "contributions": meta.get("contributions", {}),
        "top_contributor": meta.get("top_contributor"),
        "fault_type": meta.get("fault_type"),
    }


@app.get("/api/uncertainty", dependencies=[Depends(get_current_user)])
async def get_uncertainty():
    """MC-Dropout / conformal prediction uncertainty metrics."""
    meta = _latest_meta()
    return {
        "prediction_interval": meta.get("prediction_interval", {}),
        "conformal_boost": meta.get("conformal_boost"),
        "urgency_confidence": meta.get("urgency_confidence"),
        "urgency_variance": meta.get("urgency_variance"),
    }


# ── WebSocket Telemetry Stream ─────────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(default="")):
    """
    Streams live telemetry data at 1Hz with the FULL expanded payload.
    Every internal AI variable is exposed for complete dashboard transparency.
    Auth via: ws://host:port/ws/telemetry?token=YOUR_KEY (when MHARS_API_KEY is set)
    """
    # Verify WebSocket auth
    if not verify_ws_token(token):
        await websocket.close(code=4003, reason="Invalid API key")
        return

    await websocket.accept()
    print("[WebSocket] Client connected.")

    try:
        while True:
            # ── Step 1: Get temperature ────────────────────────────────────
            if state.live_mode:
                # LIVE MODE — real-data path (Apple Silicon exposes no CPU temp
                # sensor, so temperature is a TRANSPARENT thermal model of the
                # machine's REAL CPU load). Inputs (load/RAM/etc) are real psutil.
                if PSUTIL_AVAILABLE:
                    cpu_pct = psutil.cpu_percent(interval=0) / 100.0
                else:
                    cpu_pct = 0.5
                # Thermal model: idle ~42°C → full-load ~88°C, driven by real load.
                target_temp = 42.0 + cpu_pct * 46.0
                # Thermal mass: first-order lag (EMA) so temp ramps/settles like a
                # real chip instead of jumping with instantaneous CPU% spikes.
                if state._live_temp_ema is None:
                    state._live_temp_ema = target_temp
                else:
                    state._live_temp_ema += 0.15 * (target_temp - state._live_temp_ema)
                current_temp = round(state._live_temp_ema, 2)
                from mhars.schemas import SensorReading
                sr = SensorReading(
                    temp_c=current_temp,
                    load_pct=cpu_pct,
                    ambient_c=25.0,
                )
            else:
                # DEMO MODE: simulated environment + anomaly injection
                state.env.temp = apply_anomaly_to_temp(state.env.temp)
                current_temp = state.env.temp
                # Build SensorReading with simulation context
                from mhars.schemas import SensorReading
                sr = SensorReading(
                    temp_c=current_temp,
                    load_pct=getattr(state.env, 'load_level', 0.5),
                    ambient_c=25.0,
                )

            # ── Step 2: Run the complete MHARS AI pipeline ─────────────────
            # Run in threadpool to prevent blocking the WebSocket event loop during heavy ML inferences
            # Pass synthetic mode flag to core via metadata if needed, but here we just use the global flag
            result = await run_in_threadpool(state.mhars.run, temp_celsius=sr, sync_alert=True)

            # ── Step 3: Apply PPO action feedback to the simulation ────────
            # Fix #7: Aligned with Config.ACTIONS (removed phantom "increase-fan")
            action_effects = {
                "throttle": -2.0,
                "fan+": -1.0,
                "alert": -0.3,
                "shutdown": -10.0,
                "emergency-shutdown": -10.0,
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
