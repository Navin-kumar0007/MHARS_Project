"""
MHARS Core Pipeline
====================
The main entry point for the MHARS framework.
Loads all trained models and exposes a single .run() method
that takes raw sensor readings and returns a full decision + alert.

Usage:
    from mhars import MHARS
    system = MHARS(machine_type_id=0)
    result = system.run(sensor_readings)
    print(result.alert)
    print(result.action)
"""

import os, sys, json, time, pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque

# ── Optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mhars.config import Config


# ── Result dataclass — clean output from every .run() call ────────────────────
@dataclass
class MHARSResult:
    """Everything MHARS knows after processing one sensor reading."""
    timestamp:         float
    machine_type:      str
    current_temp:      float
    anomaly_score:     float
    lstm_prediction:   float
    context_score:     float
    urgency:           float
    action:            str
    route:             str           # "edge", "cloud", "both"
    alert:             str
    llm_source:        str           # "phi3-mini" or "template"
    latency_ms:        float
    raw_obs:           List[float]   = field(default_factory=list)
    metadata:          Dict[str, Any] = field(default_factory=dict)

    def is_critical(self) -> bool:
        return self.urgency >= Config.EDGE_URGENCY_THRESHOLD

    def summary(self) -> str:
        return (
            f"[{self.machine_type}] {self.current_temp:.1f}°C | "
            f"anomaly={self.anomaly_score:.2f} | "
            f"action={self.action} | route={self.route} | "
            f"urgency={self.urgency:.2f}"
        )


# ── MHARS main class ───────────────────────────────────────────────────────────
class MHARS:
    """
    Multi-modal Hybrid Adaptive Response System.

    Loads all trained models and runs the full pipeline on demand.
    Designed for both real-time IoT deployment and batch analysis.

    Args:
        machine_type_id: 0=CPU, 1=Motor, 2=Server, 3=Engine
        llm_path:        path to Phi-3 Mini GGUF (None = template fallback)
        verbose:         print detailed output for each step
    """

    def __init__(
        self,
        machine_type_id: int = 0,
        llm_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.machine_type_id = machine_type_id
        self.profile         = Config.MACHINE_PROFILES[machine_type_id]
        self.machine_name    = self.profile["name"]
        self.verbose         = verbose

        # Apply global seed for reproducibility
        import torch
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        # Rolling window for LSTM (last 12 readings)
        self._temp_window = deque(maxlen=Config.LSTM_WINDOW)
        self._steps_since_action = 0

        print(f"[MHARS] Initialising for machine: {self.machine_name}")
        self._load_models(llm_path)
        print(f"[MHARS] Ready ✓\n")

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_models(self, llm_path: Optional[str]):
        """Load all trained models. Gracefully handles missing files."""

        # Isolation Forest
        self._if_model = None
        if os.path.exists(Config.ISOLATION_FOREST):
            with open(Config.ISOLATION_FOREST, 'rb') as f:
                self._if_model = pickle.load(f)
            print(f"  ✓  Isolation Forest loaded")
        else:
            print(f"  ⚠  Isolation Forest not found — skipping noise filter")

        # LSTM — detect hidden_size from checkpoint automatically
        self._lstm = None
        if TORCH_AVAILABLE and os.path.exists(Config.LSTM):
            from mhars.models import ThermalLSTM
            checkpoint = torch.load(Config.LSTM, map_location="cpu")
            # Infer hidden_size from weight shape: weight_ih_l0 is (4*hidden, input)
            hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
            self._lstm = ThermalLSTM(hidden_size=hidden_size)
            self._lstm.load_state_dict(checkpoint)
            self._lstm.eval()
            print(f"  ✓  LSTM loaded (hidden_size={hidden_size})")
        else:
            print(f"  ⚠  LSTM not found — using linear trend prediction")

        # Autoencoder + threshold
        self._ae_model    = None
        self._ae_threshold = 0.05
        if TORCH_AVAILABLE and os.path.exists(Config.AUTOENCODER):
            from mhars.models import ThermalAutoencoder
            self._ae_model = ThermalAutoencoder()
            self._ae_model.load_state_dict(
                torch.load(Config.AUTOENCODER, map_location="cpu"))
            self._ae_model.eval()
            if os.path.exists(Config.AUTOENCODER_META):
                with open(Config.AUTOENCODER_META) as f:
                    meta = json.load(f)
                self._ae_threshold = meta.get("threshold", 0.05)
            print(f"  ✓  Autoencoder loaded (threshold={self._ae_threshold:.5f})")
        else:
            print(f"  ⚠  Autoencoder not found — using simple anomaly score")

        # Vibration Detector
        self._vib_model     = None
        self._vib_mean      = None
        self._vib_std       = None
        self._vib_threshold = 0.01
        if TORCH_AVAILABLE and os.path.exists(Config.VIBRATION_DETECTOR):
            from mhars.models import VibrationDetector
            vib_meta_path = Config.VIBRATION_META
            if os.path.exists(vib_meta_path):
                with open(vib_meta_path) as f:
                    vib_meta = json.load(f)
                n_feat = vib_meta.get("n_features", 5)
                self._vib_model = VibrationDetector(n_features=n_feat)
                self._vib_model.load_state_dict(
                    torch.load(Config.VIBRATION_DETECTOR, map_location="cpu"))
                self._vib_model.eval()
                self._vib_mean      = np.array(vib_meta["mean"],  dtype=np.float32)
                self._vib_std       = np.array(vib_meta["std"],   dtype=np.float32)
                self._vib_threshold = vib_meta.get("threshold", 0.01)
                print(f"  ✓  Vibration Detector loaded (features={n_feat})")
            else:
                print(f"  ⚠  Vibration meta not found — skipping")
        else:
            print(f"  ⚠  Vibration Detector not found — using placeholder")

        # PPO agent
        self._ppo = None
        ppo_path = Config.PPO.replace(".zip", "")
        if os.path.exists(ppo_path + ".zip") or os.path.exists(ppo_path):
            try:
                from stable_baselines3 import PPO as SB3PPO
                self._ppo = SB3PPO.load(ppo_path)
                print(f"  ✓  PPO agent loaded")
            except Exception as e:
                print(f"  ⚠  PPO failed to load: {e}")
        else:
            print(f"  ⚠  PPO not found — using rule-based fallback")

        # LLM
        self._llm_gen = None
        try:
            from mhars.llm import AlertGenerator
            self._llm_gen = AlertGenerator(
                model_path=llm_path or (Config.LLM if os.path.exists(Config.LLM) else None)
            )
        except Exception as e:
            print(f"  ⚠  LLM init failed: {e}")

        # Multi-modal CNN Hotspot Detector
        self._cnn = None
        try:
            from stage2_ml.mobilenet_cnn import ThermalHotspotDetector
            cnn_path = os.path.join("models", "mobilenet_cnn.pt")
            self._cnn = ThermalHotspotDetector(cnn_path if os.path.exists(cnn_path) else None)
            print("  ✓  CNN Hotspot Detector loaded")
        except Exception as e:
            print(f"  ⚠  CNN Hotspot Detector not found: {e}")

        # Multi-modal Audio MFCC Pipeline
        self._audio = None
        try:
            from stage2_ml.audio_mfcc import AudioPipeline
            self._audio = AudioPipeline()
            print("  ✓  Audio MFCC Pipeline loaded")
        except Exception as e:
            print(f"  ⚠  Audio MFCC Pipeline not found: {e}")

    # ── Main pipeline ──────────────────────────────────────────────────────────
    def run(self, temp_celsius: float, extra_scores: Optional[Dict] = None, sync_alert: bool = False) -> MHARSResult:
        """
        Run the full MHARS pipeline on one temperature reading.

        Args:
            temp_celsius:  current temperature in degrees Celsius
            extra_scores:  optional dict with pre-computed scores:
                           {"cnn_score": 0.3, "audio_score": 0.5,
                            "cnn_var": None, "audio_var": None}

        Returns:
            MHARSResult with action, alert, route, and all scores.
        """
        t0 = time.perf_counter()
        extra = extra_scores or {}

        # Step 1 — Normalize temperature
        temp_norm = self._normalize_temp(temp_celsius)
        self._temp_window.append(temp_norm)

        # Step 2 — Isolation Forest noise check
        if_score = self._compute_if_score(temp_norm)

        # Step 3 — LSTM prediction
        lstm_pred_norm, lstm_score = self._compute_lstm_score(temp_norm)
        lstm_pred_celsius = self._denormalize_temp(lstm_pred_norm)

        # Step 4 — Autoencoder anomaly score
        ae_score = self._compute_ae_score()

        # Step 4b — Vibration anomaly score
        vib_score = self._compute_vib_score(temp_norm)

        # Step 4c — Multi-modal inputs (CNN and Audio)
        cnn_score   = extra.get("cnn_score", 0.5)
        cnn_var     = extra.get("cnn_var", None)
        audio_score = extra.get("audio_score", 0.5)
        audio_var   = extra.get("audio_var", None)

        if self._cnn is not None and "cnn_score" not in extra:
            cnn_res = self._cnn.predict_from_temperature(temp_celsius, self.profile["safe_max"])
            cnn_score = cnn_res["hotspot_score"]
            cnn_var   = cnn_res["grid_variance"]
            
        if self._audio is not None and "audio_score" not in extra:
            aud_res = self._audio.process_from_temperature(temp_celsius, self.profile["safe_max"])
            audio_score = aud_res["audio_score"]
            audio_var   = aud_res["audio_variance"]

        # Step 5 — Attention fusion → context score + urgency + XAI
        context, urgency, contributions, top_contributor = self._fuse(
            lstm_score = lstm_score,
            ae_score   = ae_score,
            if_score   = if_score,
            cnn_score  = max(cnn_score, vib_score),  # Fuse max physical stress
            audio_score= audio_score,
            cnn_var    = cnn_var,
            audio_var  = audio_var,
        )

        # Step 5.5 — Anomaly Fingerprinting
        fault_type = self._fingerprint_anomaly(urgency, top_contributor, temp_celsius)

        # Step 6 — RL Router decision
        route = self._route(urgency)

        # Step 7 — PPO action
        obs    = self._build_obs(temp_norm, lstm_pred_norm, ae_score, urgency)
        action = self._decide(obs, temp_celsius)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._steps_since_action = (
            0 if action != "do-nothing" else self._steps_since_action + 1
        )
        
        # Step 7.5 — Estimate RUL
        rul_minutes = self._estimate_rul(temp_celsius, lstm_pred_celsius, self.profile["safe_max"])

        # Create Result object first
        result = MHARSResult(
            timestamp       = time.time(),
            machine_type    = self.machine_name,
            current_temp    = temp_celsius,
            anomaly_score   = ae_score,
            lstm_prediction = lstm_pred_celsius,
            context_score   = context,
            urgency         = urgency,
            action          = action,
            route           = route,
            alert           = "Generating async alert...",
            llm_source      = "async",
            latency_ms      = round(latency_ms, 2),
            raw_obs         = obs.tolist(),
            metadata        = {
                "if_score": if_score, "lstm_score": lstm_score,
                "ae_score": ae_score, "vib_score": vib_score,
                "cnn_score": cnn_score, "audio_score": audio_score,
                "context_score": context, "rul_minutes": rul_minutes,
                "contributions": contributions, "top_contributor": top_contributor,
                "fault_type": fault_type
            }
        )

        # Step 8 — LLM alert
        alert_ctx = {
            "machine_type":   self.machine_name,
            "current_temp":   temp_celsius,
            "predicted_temp": lstm_pred_celsius,
            "anomaly_score":  ae_score,
            "action_name":    action,
            "urgency":        urgency,
        }
        
        if route == "edge":
            result.alert = (f"[EDGE ALERT] {self.machine_name} at {temp_celsius:.1f}°C! "
                            f"Urgent action triggered: {action}.")
            result.llm_source = "edge_template"
        elif self._llm_gen is not None:
            if sync_alert:
                # Synchronous template alert — instant, used by dashboards
                res = self._llm_gen.generate({**alert_ctx, '_force_template': True})
                result.alert = res["alert"]
                result.llm_source = res["source"]
            else:
                # Async LLM alert — used by batch scripts / edge deployment
                def on_alert_ready(res_dict):
                    result.alert = res_dict["alert"]
                    result.llm_source = res_dict["source"]
                    if self.verbose:
                        print(f"\n  [ASYNC ALERT READY] {result.alert}")
                self._llm_gen.generate_async(alert_ctx, callback=on_alert_ready)
        else:
            result.alert = f"[{self.machine_name}] {temp_celsius:.1f}°C — action: {action}."
            result.llm_source = "fallback"

        if self.verbose:
            print(result.summary())

        return result

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _normalize_temp(self, t: float) -> float:
        p = self.profile
        return float(np.clip((t - 15.0) / (p["critical"] - 15.0), 0, 1))

    def _denormalize_temp(self, t_norm: float) -> float:
        p = self.profile
        return float(t_norm * (p["critical"] - 15.0) + 15.0)

    def _compute_if_score(self, temp_norm: float) -> float:
        if self._if_model is None:
            return float(np.clip((temp_norm - 0.3) / 0.7, 0, 1))
        # Build a meaningful 5-sensor feature vector from temperature history
        # instead of feeding [t, t, t, t, t] which makes the IF useless
        window = list(self._temp_window)
        if len(window) >= 5:
            feat = np.array([[
                temp_norm,                                                     # current reading
                float(np.mean(window[-3:])),                                   # short-term avg (3-step)
                float(np.mean(window[-5:])),                                   # medium-term avg (5-step)
                abs(window[-1] - window[-2]) if len(window) >= 2 else 0.0,     # rate of change
                float(np.std(window[-5:])),                                    # recent variability
            ]])
        else:
            feat = np.array([[temp_norm, temp_norm, temp_norm, 0.0, 0.0]])
        raw = self._if_model.decision_function(feat)[0]
        score = -raw
        return float(np.clip(score, 0, 1))

    def _compute_lstm_score(self, temp_norm: float):
        window = list(self._temp_window)
        if len(window) < Config.LSTM_WINDOW:
            # Not enough history yet — use linear trend
            pred_norm = temp_norm + 0.01
        elif self._lstm is not None:
            import torch
            x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                pred_norm = float(self._lstm(x).item())
        else:
            # Simple linear extrapolation
            trend = (window[-1] - window[-3]) / 2 if len(window) >= 3 else 0
            pred_norm = float(np.clip(temp_norm + trend * 10, 0, 1))

        lstm_score = float(np.clip(abs(pred_norm - temp_norm), 0, 1))
        return pred_norm, lstm_score

    def _compute_ae_score(self) -> float:
        window = list(self._temp_window)
        if len(window) < Config.LSTM_WINDOW or self._ae_model is None:
            return 0.2  # default low score when not enough data
        import torch
        x = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            err = self._ae_model.reconstruction_error(x).item()
        score = err / (self._ae_threshold + 1e-8)
        return float(np.clip(score, 0, 1))

    def _compute_vib_score(self, temp_norm: float) -> float:
        """
        Compute vibration anomaly score from synthesized vibration features.
        In a real deployment, these features would come from an accelerometer.
        Here we derive them from temperature dynamics (correlated physics model).
        """
        if self._vib_model is None:
            # Fallback: estimate vibration from temperature trend
            window = list(self._temp_window)
            if len(window) >= 5:
                rate = abs(window[-1] - window[-3]) / 2 if len(window) >= 3 else 0
                return float(np.clip(rate * 2.0, 0, 1))
            return 0.3  # neutral default

        window = list(self._temp_window)
        if len(window) < 5:
            return 0.3  # not enough data yet

        # Synthesize vibration features from temperature dynamics
        # (Real system: these come from accelerometer hardware)
        recent = np.array(window[-12:]) if len(window) >= 12 else np.array(window)
        rms       = float(np.sqrt(np.mean(recent ** 2))) * 2.0      # RMS proxy
        peak2peak = float(np.ptp(recent)) * 3.0                     # peak-to-peak
        crest     = float(np.max(np.abs(recent)) / (rms + 1e-8))    # crest factor
        centroid  = 100.0 + float(np.std(recent)) * 100             # spectral centroid proxy
        std_val   = float(np.std(recent))                           # raw variability

        feat = np.array([rms, peak2peak, crest, centroid, std_val], dtype=np.float32)

        # Normalize using training statistics
        feat_norm = (feat - self._vib_mean) / (self._vib_std + 1e-8)

        import torch
        x = torch.FloatTensor(feat_norm.reshape(1, -1))
        with torch.no_grad():
            error = self._vib_model.reconstruction_error(x).item()
        score = error / (self._vib_threshold + 1e-8)
        return float(np.clip(score, 0, 1))

    def _fuse(self, lstm_score, ae_score, if_score,
              cnn_score=0.5, audio_score=0.5,
              cnn_var=None, audio_var=None):
        def w(var):
            return 1.0 if var is None else 1.0 / (1.0 + var)

        scores  = np.array([lstm_score, ae_score, if_score,
                             cnn_score, audio_score], dtype=np.float32)
        weights = np.array([1.0, 1.0, 1.0, w(cnn_var), w(audio_var)],
                            dtype=np.float32)
        weights /= weights.sum() + 1e-8

        context = float(np.clip(np.dot(weights, scores), 0, 1))
        top2    = np.sort(scores)[-2:]
        urgency = float(np.clip(0.6 * top2[-1] + 0.4 * top2[-2], 0, 1))
        
        # XAI Contributions
        # Prevent division by zero if all scores are 0
        total_impact = np.dot(weights, scores) + 1e-8
        contrib = {
            "trend_forecast": round((weights[0] * scores[0] / total_impact) * 100),
            "pattern_check":  round((weights[1] * scores[1] / total_impact) * 100),
            "outlier_scan":   round((weights[2] * scores[2] / total_impact) * 100),
            "vibration":      round((weights[3] * scores[3] / total_impact) * 100),
            "audio":          round((weights[4] * scores[4] / total_impact) * 100),
        }
        top_contributor = max(contrib, key=contrib.get) if context > 0.05 else "none"

        return context, urgency, contrib, top_contributor

    def _route(self, urgency: float) -> str:
        if urgency >= Config.EDGE_URGENCY_THRESHOLD:
            return "edge"
        elif urgency <= Config.CLOUD_URGENCY_THRESHOLD:
            return "cloud"
        return "both"

    def _build_obs(self, temp_norm, pred_norm, ae_score, urgency) -> np.ndarray:
        return np.array([
            temp_norm,
            pred_norm,
            ae_score,
            self.machine_type_id / 3.0,
            float(np.clip(self._steps_since_action / 100.0, 0, 1)),
            urgency,
        ], dtype=np.float32)

    def _decide(self, obs: np.ndarray, current_temp_raw: float) -> str:
        # HARDWARE SAFETY OVERRIDE:
        # RL agents are prone to out-of-distribution failure.
        # In real-world industrial systems, a strict rule-based
        # safety envelope always overrides the AI.
        p = self.profile
        if current_temp_raw >= p["critical"]:
            return "emergency-shutdown"
        elif current_temp_raw >= p["safe_max"] and obs[5] > 0.8:
            return "shutdown"

        if self._ppo is not None:
            action_id, _ = self._ppo.predict(obs, deterministic=True)
            return Config.ACTIONS[int(action_id)]
        
        # Rule-based fallback when PPO not loaded
        temp_norm = obs[0]
        urgency   = obs[5]
        if urgency > 0.7:
            return "fan+"
        elif urgency > 0.5:
            return "throttle"
        elif urgency > 0.35:
            return "alert"
        return "do-nothing"

    def _estimate_rul(self, current_temp: float, predicted_temp: float, safe_max: float) -> Optional[float]:
        """Estimate remaining useful life in minutes before hitting safe_max."""
        delta_per_10min = predicted_temp - current_temp
        if delta_per_10min <= 0:
            return None  # Temperature falling or stable — no immediate RUL concern
            
        remaining_degrees = safe_max - current_temp
        if remaining_degrees <= 0:
            return 0.0  # Already past threshold
            
        minutes = (remaining_degrees / delta_per_10min) * 10.0
        return round(min(minutes, 999.0), 1)

    def _fingerprint_anomaly(self, urgency: float, top_contributor: str, current_temp: float) -> str:
        """Map abstract ML scores to physical fault signatures."""
        if urgency < 0.4:
            return "Normal Operations"
            
        if top_contributor == "vibration":
            return "Bearing / Mechanical Wear"
        elif top_contributor == "audio":
            return "Acoustic Cavitation / Grinding"
        elif top_contributor == "trend_forecast":
            return "Thermal Runaway"
        elif top_contributor == "pattern_check" and current_temp > self.profile["safe_max"] * 0.8:
            return "Cooling System Failure"
        elif top_contributor == "outlier_scan":
            return "Sensor Glitch / Power Surge"
            
        return "Unknown System Stress"

    # (Alert generation is now inline async in run())

    # ── Convenience methods ────────────────────────────────────────────────────
    def run_sequence(self, temps: List[float]) -> List[MHARSResult]:
        """Run the pipeline on a list of temperature readings in sequence."""
        return [self.run(t) for t in temps]

    def reset(self):
        """Clear the rolling window — call this when switching machines."""
        self._temp_window.clear()
        self._steps_since_action = 0
        print(f"[MHARS] Reset for {self.machine_name}")

    def wait_for_alerts(self):
        """Block until all background LLM alerts have been processed."""
        if self._llm_gen is not None:
            self._llm_gen.wait_for_alerts()