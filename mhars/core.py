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

import os, sys, json, time, pickle, logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from collections import deque

from mhars.health_score import HealthScoreEngine
from mhars.trend_analyzer import TrendAnalyzer
from mhars.explainability import ExplainabilityEngine
from mhars.maintenance_scheduler import MaintenanceScheduler
from stage2_ml.mc_dropout import MCDropoutEstimator


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (Issue #35)."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from mhars.schemas import SensorReading

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
        np.random.seed(Config.SEED)
        if TORCH_AVAILABLE:
            torch.manual_seed(Config.SEED)

        # Rolling window for LSTM (last 12 readings)
        self._temp_window = deque(maxlen=Config.LSTM_WINDOW)
        self._steps_since_action = 0

        # V2 — Multivariate sensor window: stores (12, 5) sensor history
        self._multi_sensor_window = deque(maxlen=Config.LSTM_WINDOW)

        # Issue 4 — Temporal context: full reading history for dT/dt
        self._reading_history: deque = deque(maxlen=60)  # 60 sec @ 1Hz

        # Issue 5 — Online anomaly detection: rolling retrain buffer
        self._if_retrain_buffer: deque = deque(maxlen=500)
        self._if_retrain_interval = Config.IF_RETRAIN_INTERVAL
        self._if_sample_count = 0
        self._if_has_retrained = False  # Cold-start flag: IF pickle is trained on
                                        # CMAPSS multi-sensor, not our 5-feature vector.
                                        # Skip it until online retraining fires once.

        # Throttle registry heartbeat (#17)
        self._last_heartbeat_time: float = 0.0
        self._heartbeat_interval: float = 30.0  # seconds

        # Issue 6 — Uncertainty quantification: track recent score variance
        self._recent_urgencies: deque = deque(maxlen=30)
        self._recent_contexts: deque  = deque(maxlen=30)
        
        # New Feature — Data Drift Detection
        self._ae_score_history: deque = deque(maxlen=100)

        # X.1 — Streaming feature-drift monitor (concept-drift → retrain signal)
        from mhars.drift_monitor import DriftMonitor
        self._drift_monitor = DriftMonitor()

        # R4 — label-free lifelong adaptation: buffer recent NORMAL AE windows.
        self._normal_ae_buffer = deque(maxlen=400)
        self._adapt_count = 0
        self._last_adaptation = None
        self._adapt_cooldown = 0

        # Issue 5 — Structured logging
        self._setup_logger()

        # Issue 8 — Register node in multi-agent registry
        from mhars.registry import AgentRegistry
        self._registry = AgentRegistry()
        self.node_id = f"{self.machine_name}_{os.getpid()}"
        self._registry.register_node(self.node_id, self.machine_name)

        # Phase 4B — New Engines
        self._health_engine = HealthScoreEngine(self.profile)
        # Target mean is roughly normalized operating temp (e.g. 0.5)
        self._trend_analyzer = TrendAnalyzer(target_mean=0.5, std_dev=0.15)
        self._xai_engine = ExplainabilityEngine()
        self._maintenance_scheduler = MaintenanceScheduler(self.profile)
        self._mc_dropout = MCDropoutEstimator(num_samples=20)

        # Phase 3 — Physics-Informed Causal Layer & Digital Twin
        try:
            from stage3_ai.causal_layer import PhysicsCausalLayer
            self._causal_layer = PhysicsCausalLayer(self.profile)
            
            from stage1_simulation.digital_twin import DigitalTwin
            self._digital_twin = DigitalTwin(self.profile)

            from stage3_ai.counterfactual_rca import CounterfactualRCA  # R3
            self._cf_rca = CounterfactualRCA(self.profile)
        except ImportError:
            self._causal_layer = None
            self._digital_twin = None
            self._cf_rca = None

        print(f"[MHARS] Initialising for machine: {self.machine_name}")
        self._load_models(llm_path)
        print(f"[MHARS] Ready ✓\n")

    def _setup_logger(self):
        import logging.handlers
        import queue
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'mhars_events.jsonl')
        
        self.logger = logging.getLogger('mhars_structured')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Prevent duplicate handlers if re-instantiated
        if not getattr(self.logger, 'queue_listener_started', False):
            # 1. Create a queue
            self._log_queue = queue.Queue(-1)
            
            # 2. Create the QueueHandler (attached to the logger)
            queue_handler = logging.handlers.QueueHandler(self._log_queue)
            self.logger.addHandler(queue_handler)
            
            # 3. Create the RotatingFileHandler (runs in background)
            # Max 10MB per file, keep 5 backups
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            
            # 4. Create and start the listener
            self._log_listener = logging.handlers.QueueListener(self._log_queue, file_handler)
            self._log_listener.start()
            self.logger.queue_listener_started = True

    def close(self):
        """Stop the background log listener and release file handles (#18)."""
        if hasattr(self, '_log_listener'):
            self._log_listener.stop()
        if hasattr(self, '_llm_gen') and self._llm_gen is not None:
            self._llm_gen.wait_for_alerts()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_models(self, llm_path: Optional[str]):
        """Load all trained models. Gracefully handles missing files."""
        from mhars.metadata_manager import MetadataManager
        # Pass logger to MetadataManager for structured stale-model events
        metadata = MetadataManager(max_age_days=30, logger=self.logger)

        # Isolation Forest
        self._if_model = None
        if os.path.exists(Config.ISOLATION_FOREST):
            with open(Config.ISOLATION_FOREST, 'rb') as f:
                self._if_model = pickle.load(f)
            metadata.check_model_freshness(Config.ISOLATION_FOREST, "Isolation Forest")
            print(f"  ✓  Isolation Forest loaded")
        else:
            print(f"  ⚠  Isolation Forest not found — skipping noise filter")

        # R1: zero-shot foundation forecaster (opt-in via MHARS_FORECASTER=foundation).
        # Lazy — the model only downloads/loads on first forecast.
        self._foundation = None
        self._prev_fc_band = None       # (p10,p50,p90) of last next-step forecast
        self._foundation_anomaly = 0.0  # zero-shot residual anomaly score
        if getattr(Config, "FORECASTER_BACKEND", "lstm") == "foundation":
            try:
                from stage2_ml.foundation_forecaster import FoundationForecaster
                self._foundation = FoundationForecaster(Config.FOUNDATION_MODEL, quantiles=Config.LSTM_QUANTILES)
                print(f"  ✓  Foundation forecaster enabled ({Config.FOUNDATION_MODEL}, zero-shot)")
            except Exception as e:
                print(f"  ⚠  Foundation forecaster unavailable ({e}) — using trained LSTM")

        # LSTM — prefer TFT (Phase 3), fall back to V2 (BiLSTM+Attention), fall back to V1
        self._lstm = None
        self._lstm_version = None  # "tft", "v2", or "v1"
        self._lstm_horizon = 1      # P1.5: multi-horizon forecast length
        self._lstm_qmode = False    # P2.1: per-step quantile forecaster
        self._lstm_quantiles = None
        self._last_forecast_traj = None  # normalized forward trajectory (p50, t+1…t+H)
        self._last_forecast_band = None  # [(lo,hi)…] normalized p10/p90 per step
        if TORCH_AVAILABLE and getattr(Config, 'TFT_MODEL', False) and os.path.exists(Config.TFT_MODEL):
            from mhars.models import TFTPredictor
            checkpoint = torch.load(Config.TFT_MODEL, map_location="cpu")
            self._lstm = TFTPredictor(num_vars=Config.LSTM_INPUT_SIZE_V2, d_model=Config.TFT_D_MODEL, n_heads=Config.TFT_N_HEADS, num_quantiles=Config.TFT_NUM_QUANTILES)
            self._lstm.load_state_dict(checkpoint)
            self._lstm.eval()
            self._lstm_version = "tft"
            print(f"  ✓  TFT Predictor loaded (Phase 3)")
        elif TORCH_AVAILABLE and os.path.exists(Config.LSTM_V2):
            from mhars.models import ThermalLSTMv2
            checkpoint = torch.load(Config.LSTM_V2, map_location="cpu")
            hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
            input_size = checkpoint["lstm.weight_ih_l0"].shape[1]
            out_h = checkpoint["linear.weight"].shape[0]   # = H (P1.5) or H*Q (P2.1)
            self._lstm = ThermalLSTMv2(input_size=input_size, hidden_size=hidden_size, output_horizon=out_h)
            self._lstm.load_state_dict(checkpoint)
            self._lstm.eval()
            self._lstm_version = "v2"
            # P2.1: a sidecar meta marks quantile mode (head = H * Q).
            v2_meta_path = Config.LSTM_V2.replace(".pt", "_meta.json")
            if os.path.exists(v2_meta_path):
                with open(v2_meta_path) as f:
                    v2_meta = json.load(f)
                self._lstm_quantiles = v2_meta.get("quantiles")
                self._lstm_horizon = v2_meta.get("horizon", out_h)
                self._lstm_qmode = bool(self._lstm_quantiles)
            else:
                self._lstm_horizon = out_h
            _qtag = f", quantiles={len(self._lstm_quantiles)}" if self._lstm_qmode else ""
            print(f"  ✓  LSTM V2 loaded (BiLSTM+Attention, input={input_size}, hidden={hidden_size}, horizon={self._lstm_horizon}{_qtag})")
        elif TORCH_AVAILABLE and os.path.exists(Config.LSTM):
            from mhars.models import ThermalLSTM
            checkpoint = torch.load(Config.LSTM, map_location="cpu")
            hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
            self._lstm = ThermalLSTM(hidden_size=hidden_size)
            self._lstm.load_state_dict(checkpoint)
            self._lstm.eval()
            metadata.check_model_freshness(Config.LSTM, "Thermal LSTM")
            self._lstm_version = "v1"
            print(f"  ✓  LSTM V1 loaded (hidden_size={hidden_size})")
        else:
            print(f"  ⚠  Thermal Predictor not found — using linear trend prediction")

        # Autoencoder — prefer V2 (LSTM-AE), fall back to V1 (linear AE)
        self._ae_model    = None
        self._ae_threshold = 0.05
        self._ae_version   = None
        self._ae_calib     = None
        if TORCH_AVAILABLE and os.path.exists(Config.AUTOENCODER_V2):
            from mhars.models import ThermalAutoencoderLSTM
            if os.path.exists(Config.AUTOENCODER_V2_META):
                with open(Config.AUTOENCODER_V2_META) as f:
                    ae_meta = json.load(f)
                input_size = ae_meta.get("input_size", 5)
                hidden_size = ae_meta.get("hidden_size", 32)
                self._ae_model = ThermalAutoencoderLSTM(
                    input_size=input_size, hidden_size=hidden_size, seq_len=12)
                self._ae_model.load_state_dict(
                    torch.load(Config.AUTOENCODER_V2, map_location="cpu"))
                self._ae_model.eval()
                self._ae_threshold = ae_meta.get("threshold", 0.05)
                self._ae_version = "v2"
                if "calib" in ae_meta:
                    from mhars.anomaly_calibrator import AnomalyCalibrator
                    self._ae_calib = AnomalyCalibrator.from_dict(ae_meta["calib"])
                print(f"  ✓  LSTM-AE V2 loaded (input={input_size}, threshold={self._ae_threshold:.5f}"
                      f"{', calibrated' if self._ae_calib else ''})")
            else:
                print(f"  ⚠  LSTM-AE V2 meta not found — skipping")
        elif TORCH_AVAILABLE and os.path.exists(Config.AUTOENCODER):
            from mhars.models import ThermalAutoencoder
            self._ae_model = ThermalAutoencoder()
            self._ae_model.load_state_dict(
                torch.load(Config.AUTOENCODER, map_location="cpu"))
            self._ae_model.eval()
            self._ae_version = "v1"
            if os.path.exists(Config.AUTOENCODER_META):
                with open(Config.AUTOENCODER_META) as f:
                    meta = json.load(f)
                self._ae_threshold = meta.get("threshold", 0.05)
            metadata.check_model_freshness(Config.AUTOENCODER, "Thermal Autoencoder")
            print(f"  ✓  Autoencoder V1 loaded (threshold={self._ae_threshold:.5f})")
        else:
            print(f"  ⚠  Autoencoder not found — using simple anomaly score")

        # Conformal Prediction — load calibration if available
        self._conformal = None
        if os.path.exists(Config.CONFORMAL_META):
            try:
                from mhars.conformal import ConformalPredictor
                self._conformal = ConformalPredictor.load(Config.CONFORMAL_META)
                print(f"  ✓  Conformal Predictor loaded (coverage={self._conformal.coverage:.0%})")
            except Exception as e:
                print(f"  ⚠  Conformal predictor load failed: {e}")
        else:
            print(f"  ⚠  Conformal predictor not calibrated — no prediction intervals")

        # Vibration Detector
        self._vib_model     = None
        self._vib_mean      = None
        self._vib_std       = None
        self._vib_threshold = 0.01
        self._vib_calib     = None
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
                if "calib" in vib_meta:
                    from mhars.anomaly_calibrator import AnomalyCalibrator
                    self._vib_calib = AnomalyCalibrator.from_dict(vib_meta["calib"])
                metadata.check_model_freshness(Config.VIBRATION_DETECTOR, "Vibration Detector")
                print(f"  ✓  Vibration Detector loaded (features={n_feat}"
                      f"{', calibrated' if self._vib_calib else ''})")
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
            from stage2_ml.efficientnet_cnn import ThermalHotspotDetector
            cnn_path = os.path.join("models", "efficientnet_cnn.pt")
            self._cnn = ThermalHotspotDetector(cnn_path if os.path.exists(cnn_path) else None)
            print("  ✓  CNN Hotspot Detector loaded")
        except Exception as e:
            print(f"  ⚠  CNN Hotspot Detector not found: {e}")

        # Multi-modal Audio MFCC Pipeline
        self._audio = None
        try:
            from stage2_ml.audio_mfcc import AudioPipeline
            self._audio = AudioPipeline()
            print(f"  ✓  Audio MFCC Pipeline loaded")
        except Exception as e:
            print(f"  ⚠  Audio MFCC Pipeline not found: {e}")

        # Phase 2: Learned Attention Fusion
        self._learned_fusion = None
        if TORCH_AVAILABLE and os.path.exists(Config.LEARNED_FUSION_MODEL):
            try:
                from mhars.learned_fusion import LearnedAttentionFusion
                self._learned_fusion = LearnedAttentionFusion(
                    n_modalities=Config.FUSION_N_MODALITIES,
                    d_model=Config.FUSION_D_MODEL,
                    n_heads=Config.FUSION_N_HEADS,
                )
                self._learned_fusion.load_state_dict(
                    torch.load(Config.LEARNED_FUSION_MODEL, map_location="cpu"))
                self._learned_fusion.eval()
                print(f"  ✓  Learned Attention Fusion loaded")
            except Exception as e:
                print(f"  ⚠  Learned Fusion load failed: {e}")
        else:
            print(f"  ⚠  Learned Fusion not found — using rule-based fusion")

        # Phase 2: RUL Predictor V2
        self._rul_model = None
        if TORCH_AVAILABLE and os.path.exists(Config.RUL_MODEL_V2):
            try:
                from mhars.models import RULPredictor
                checkpoint = torch.load(Config.RUL_MODEL_V2, map_location="cpu")
                self._rul_model = RULPredictor()
                self._rul_model.load_state_dict(checkpoint)
                self._rul_model.eval()
                print(f"  ✓  RUL Predictor V2 loaded")
            except Exception as e:
                print(f"  ⚠  RUL Predictor V2 load failed: {e}")

        # P2.4: Supervised fault classifier
        self._fault_clf = None
        self._fault_mean = None
        self._fault_std = None
        if TORCH_AVAILABLE and os.path.exists(Config.FAULT_CLASSIFIER) and os.path.exists(Config.FAULT_CLASSIFIER_META):
            try:
                from mhars.models import FaultClassifier
                with open(Config.FAULT_CLASSIFIER_META) as f:
                    fc_meta = json.load(f)
                self._fault_clf = FaultClassifier(
                    n_features=fc_meta.get("n_features", 10),
                    n_classes=fc_meta.get("n_classes", len(Config.FAULT_CLASSES)))
                self._fault_clf.load_state_dict(torch.load(Config.FAULT_CLASSIFIER, map_location="cpu"))
                self._fault_clf.eval()
                self._fault_mean = np.array(fc_meta["mean"], dtype=np.float32)
                self._fault_std = np.array(fc_meta["std"], dtype=np.float32)
                print(f"  ✓  Fault Classifier loaded (acc={fc_meta.get('val_acc', '?')})")
            except Exception as e:
                print(f"  ⚠  Fault Classifier load failed: {e}")
        else:
            print(f"  ⚠  Fault Classifier not found — using rule-based fingerprint")

        # Phase 2: SAC Agent
        self._sac = None
        sac_path = Config.SAC_MODEL.replace(".zip", "")
        if os.path.exists(sac_path + ".zip") or os.path.exists(sac_path):
            try:
                from stable_baselines3 import SAC as SB3SAC
                self._sac = SB3SAC.load(sac_path)
                print(f"  ✓  SAC agent loaded")
            except Exception as e:
                print(f"  ⚠  SAC failed to load: {e}")

    # ── Main pipeline ──────────────────────────────────────────────────────────
    def run(self, temp_celsius: Union[float, 'SensorReading'] = None,
            extra_scores: Optional[Dict] = None,
            sync_alert: bool = False,
            reading: Optional['SensorReading'] = None) -> MHARSResult:
        """
        Run the full MHARS pipeline on a sensor reading.

        Accepts EITHER:
          - temp_celsius (float) for backward compatibility
          - reading (SensorReading) for full multi-sensor context
          - temp_celsius as a SensorReading object directly

        Returns:
            MHARSResult with action, alert, route, and all scores.
        """
        t0 = time.perf_counter()
        extra = extra_scores or {}

        # Issue 1 — Accept SensorReading or bare float
        if reading is not None:
            sr = reading
        elif isinstance(temp_celsius, SensorReading):
            sr = temp_celsius
        elif temp_celsius is not None:
            sr = SensorReading.from_temp_only(float(temp_celsius))
        else:
            raise ValueError("Must provide temp_celsius or reading")

        # Issue 4 — Auto-compute dT/dt from history if not provided
        if sr.dT_dt is None and len(self._reading_history) >= 2:
            prev = self._reading_history[-1]
            sr.dT_dt = sr.temp_c - prev.temp_c  # °C/sec at 1Hz
        elif sr.dT_dt is None:
            sr.dT_dt = 0.0
        self._reading_history.append(sr)

        temp_celsius_val = sr.temp_c

        # Step 1 — Normalize temperature
        temp_norm = self._normalize_temp(temp_celsius_val)
        self._temp_window.append(temp_norm)

        # V2 — Feed multivariate sensor window (5 normalized sensor values)
        # In production, these come from real sensors; here we synthesize
        # correlated values from the primary temperature for compatibility
        multi_sensor_values = self._build_multi_sensor_vector(sr, temp_norm)
        self._multi_sensor_window.append(multi_sensor_values)

        # Step 2 — Isolation Forest noise check
        if_score = self._compute_if_score(temp_norm)

        # Issue 5 — Online anomaly detection retrain (guarded by Config toggle #3)
        self._if_sample_count += 1
        if self._if_model is not None and Config.IF_ONLINE_RETRAIN:
            window_data = list(self._temp_window)
            if len(window_data) >= 5:
                feat = np.array([[
                    temp_norm,
                    float(np.mean(window_data[-3:])),
                    float(np.mean(window_data[-5:])),
                    abs(window_data[-1] - window_data[-2]) if len(window_data) >= 2 else 0.0,
                    float(np.std(window_data[-5:])),
                ]])
                self._if_retrain_buffer.append(feat[0])
            # Fire the first online retrain early (~100 samples) so the cheap
            # warmup proxy hands off to the real serving-distribution IF quickly,
            # then on the regular interval thereafter (P3.1).
            if len(self._if_retrain_buffer) >= 50 and (
                    self._if_sample_count % self._if_retrain_interval == 0
                    or (not self._if_has_retrained and len(self._if_retrain_buffer) >= 100)):
                self._retrain_if()

        # Step 3 — LSTM prediction (now returns conformal interval + boost)
        lstm_pred_norm, lstm_score, prediction_interval, conformal_boost = self._compute_lstm_score(temp_norm)
        lstm_pred_celsius = self._denormalize_temp(lstm_pred_norm)

        # Load context modulates anomaly interpretation.
        # Idle machines amplify anomaly signals (unexpected heat when no load).
        load_factor = 1.0 + (1.0 - sr.load_pct) * 0.3

        # Step 4 — Autoencoder anomaly score
        ae_score = self._compute_ae_score()

        # Step 4b — Vibration anomaly score
        if sr.vibration_g > 0.0:
            # Map real vibration to score (e.g. >10g is critical)
            vib_score = float(np.clip(sr.vibration_g / 10.0, 0, 1))
        else:
            vib_score = self._compute_vib_score(temp_norm)

        # Step 4c — Multi-modal inputs (CNN and Audio)
        # P1.3: The CNN (ImageNet weights, no thermal fine-tuning) and the audio
        # model are not trained on real thermal-camera / fault-audio data — their
        # "scores" were temperature-derived placeholders that injected ~0.5 noise
        # into 2 of the 6 fusion modalities. They are GATED OUT of fusion until a
        # real dataset is available: default to 0.0 (no contribution) and skip the
        # placeholder inference entirely (also saves per-tick latency). Real values
        # are still honoured when supplied via extra kwargs or the SensorReading,
        # so a true sensor can be plugged in later without code changes.
        cnn_score   = extra.get("cnn_score", 0.0)
        cnn_var     = extra.get("cnn_var", None)

        audio_score = sr.audio_score if sr.audio_score is not None else extra.get("audio_score", 0.0)
        audio_var   = sr.audio_var if sr.audio_var is not None else extra.get("audio_var", None)

        # Step 5 — Attention fusion → context score + XAI
        # Fix #10: Pass CNN and vibration as separate modalities instead of max()
        context, contributions, top_contributor = self._fuse(
            lstm_score  = lstm_score,
            ae_score    = ae_score,
            if_score    = if_score,
            cnn_score   = cnn_score,
            audio_score = audio_score,
            vib_score   = vib_score,
            cnn_var     = cnn_var,
            audio_var   = audio_var,
        )

        # Enhance top_contributor with XAI gradient attribution if applicable
        feature_importance = {}
        if top_contributor == "pattern_check" and self._ae_model is not None and TORCH_AVAILABLE:
            try:
                # Need the tensor input again
                x = torch.FloatTensor([list(self._multi_sensor_window)]) if self._ae_version == "v2" else torch.FloatTensor(list(self._temp_window)).unsqueeze(0)
                feature_importance = self._xai_engine.compute_attribution(self._ae_model, x)
            except Exception as e:
                self.logger.warning(f"XAI Attribution failed for AE: {e}")
        elif top_contributor == "trend_forecast" and self._lstm is not None and TORCH_AVAILABLE:
             try:
                x = torch.FloatTensor([list(self._multi_sensor_window)]) if self._lstm_version in ["v2", "tft"] else torch.FloatTensor(list(self._temp_window)).unsqueeze(0).unsqueeze(-1)
                feature_importance = self._xai_engine.compute_attribution(self._lstm, x)
             except Exception as e:
                 self.logger.warning(f"XAI Attribution failed for LSTM: {e}")

        # Step 5.2 — Base urgency driven by prediction proximity to critical threshold
        base_urgency = (lstm_pred_celsius - 25.0) / (self.profile["critical"] - 25.0)
        base_urgency = max(0.0, base_urgency)
        
        # Apply load factor
        load_factor = 1.0 + (sr.load_pct * 0.5) if hasattr(sr, 'load_pct') else 1.0
        
        urgency = (base_urgency * 0.6 + context * 0.4) * load_factor
        urgency = float(np.clip(urgency, 0, 1))

        # Step 5.5 — Anomaly Fingerprinting
        # P2.4: prefer the learned fault classifier; fall back to the rule-based
        # fingerprint when no model is loaded or its confidence is low.
        fault_features = self._fault_feature_vector(if_score, lstm_score, ae_score, vib_score, context, urgency)
        clf_fault, fault_confidence, anomaly_probability = self._classify_fault(fault_features)
        # R1: when the zero-shot foundation backbone is active, use its
        # distribution-free residual as the anomaly detector — it generalises to
        # unseen machines / real hardware (no false-fire on out-of-distribution
        # data), unlike the sim-trained classifier.
        if self._foundation is not None:
            anomaly_probability = float(self._foundation_anomaly)
        # X.1 — monitor distribution drift on normal-operation samples only.
        self._drift_monitor.update(fault_features, is_normal=(anomaly_probability < 0.5))

        # R4 — buffer recent NORMAL AE windows (label-free) and adapt on drift.
        if self._ae_version == "v2" and anomaly_probability < 0.3 and len(self._multi_sensor_window) >= Config.LSTM_WINDOW:
            self._normal_ae_buffer.append([list(map(float, r)) for r in self._multi_sensor_window])
        if self._adapt_cooldown > 0:
            self._adapt_cooldown -= 1
        if (self._drift_monitor.retrain_recommended and self._adapt_cooldown == 0
                and len(self._normal_ae_buffer) >= 80):
            self.adapt_online()
        if clf_fault is not None and clf_fault != "Normal Operations" and fault_confidence >= Config.FAULT_MIN_CONFIDENCE and urgency >= 0.5:
            fault_type = clf_fault
        elif clf_fault == "Normal Operations" and fault_confidence >= Config.FAULT_MIN_CONFIDENCE:
            fault_type = "Normal Operations"
        else:
            fault_type = self._fingerprint_anomaly(urgency, top_contributor, temp_celsius_val)

        # Step 6 — RL Router decision (apply conformal boost to urgency)
        urgency = float(np.clip(urgency + conformal_boost, 0, 1))
        route = self._route(urgency)

        # Step 7 — PPO action
        obs    = self._build_obs(temp_norm, lstm_pred_norm, ae_score, urgency)
        action = self._decide(obs, temp_celsius_val, sr)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Issue 6 — Uncertainty quantification
        self._recent_urgencies.append(urgency)
        self._recent_contexts.append(context)
        urgency_confidence = 1.0 - float(np.std(list(self._recent_urgencies))) if len(self._recent_urgencies) >= 3 else 0.5
        urgency_variance   = float(np.std(list(self._recent_urgencies))) if len(self._recent_urgencies) >= 3 else 0.0
        self._steps_since_action = (
            0 if action != "do-nothing" else self._steps_since_action + 1
        )
        
        # Step 7.5 — Estimate RUL & Advanced Analytics
        rul_minutes = self._estimate_rul(temp_celsius_val, lstm_pred_celsius, self.profile["safe_max"])

        # R3 — Causal counterfactual RCA (only when concerned, to save cost).
        causal_rca = None
        if self._cf_rca is not None and urgency >= 0.5:
            fan_now = 1.0 if action in ("fan+", "throttle") else 0.0
            try:
                causal_rca = self._cf_rca.analyze(temp_celsius_val, sr.load_pct, fan_now, sr.dT_dt)
            except Exception:
                causal_rca = None

        # Trend Analysis
        trend_stats = self._trend_analyzer.update(temp_norm)
        drift_detected = trend_stats["is_drifting"]
        
        # Health Score & Maintenance Schedule
        health_data = self._health_engine.compute(
            current_temp=temp_celsius_val,
            anomaly_score=ae_score,
            rul_minutes=rul_minutes,
            vib_score=vib_score,
            drift_detected=drift_detected
        )
        
        maintenance_plan = self._maintenance_scheduler.schedule(rul_minutes, health_data["score"])

        # Create Result object first
        result = MHARSResult(
            timestamp       = time.time(),
            machine_type    = self.machine_name,
            current_temp    = temp_celsius_val,
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
                "feature_importance": feature_importance,
                "fault_type": fault_type,
                "fault_confidence": round(fault_confidence, 3),
                # P2.2: calibrated anomaly-detection score (ROC-AUC ~0.93 vs ~0.55
                # for the raw AE/fused scores). Display signal; control logic
                # (urgency/PPO) is intentionally left unchanged.
                "anomaly_probability": round(anomaly_probability, 4),
                # R1 — zero-shot foundation residual anomaly + active backbone
                "foundation_anomaly": round(self._foundation_anomaly, 4),
                "forecaster_backend": "foundation" if self._foundation is not None else "lstm",
                # X.1 — feature-drift / retrain signal
                "drift": self._drift_monitor.snapshot(),
                # R3 — causal counterfactual RCA (root cause + prescribed action)
                "causal_rca": causal_rca,
                # R4 — lifelong adaptation status
                "adaptation": {"count": self._adapt_count, "last": self._last_adaptation,
                               "normal_buffer": len(self._normal_ae_buffer)},
                "health_score": health_data["score"],
                "health_trend": health_data["trend"],
                "health_breakdown": health_data["breakdown"],
                "maintenance_plan": maintenance_plan,
                "trend_stats": trend_stats,
                "features": sr.to_feature_vector(),
                "urgency_confidence": round(urgency_confidence, 3),
                "urgency_variance": round(urgency_variance, 4),
                # Phase 1 — Conformal prediction interval
                "prediction_interval": prediction_interval,
                "conformal_boost": round(conformal_boost, 4),
                # Phase 1 — Model version tracking
                "lstm_version": self._lstm_version or "fallback",
                "ae_version": self._ae_version or "fallback",
                # P1.5 — Direct multi-horizon forecast (denormalized °C trajectory)
                "forecast_trajectory_c": (
                    [round(self._denormalize_temp(v), 2) for v in self._last_forecast_traj]
                    if self._last_forecast_traj else None
                ),
                "forecast_band_c": (
                    [[round(self._denormalize_temp(lo), 2), round(self._denormalize_temp(hi), 2)]
                     for (lo, hi) in self._last_forecast_band]
                    if self._last_forecast_band else None
                ),
                "forecast_horizon_s": self._lstm_horizon if self._lstm_horizon > 1 else None,
            }
        )


        # Step 8 — LLM alert
        # Phase 3 — Causal Reasoning
        causal_reasoning = "Normal operation"
        if hasattr(self, '_causal_layer') and self._causal_layer is not None:
            # Simple assumption: 0 fan speed if not throttling, else 1
            current_fan = 1.0 if action in ["fan+", "throttle"] else 0.0
            causal_res = self._causal_layer.analyze(temp_celsius_val, sr.load_pct, current_fan)
            causal_reasoning = causal_res["root_cause_hypothesis"]

        alert_ctx = {
            "machine_type":     self.machine_name,
            "current_temp":     temp_celsius_val,
            "predicted_temp":   lstm_pred_celsius,
            "anomaly_score":    ae_score,
            "action_name":      action,
            "urgency":          urgency,
            "load_pct":         sr.load_pct,
            "dT_dt":            sr.dT_dt,
            "causal_reasoning": causal_reasoning,
        }
        
        if route == "edge":
            result.alert = (f"[EDGE ALERT] {self.machine_name} at {temp_celsius_val:.1f}°C! "
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
            result.alert = f"[{self.machine_name}] {temp_celsius_val:.1f}°C — action: {action}."
            result.llm_source = "fallback"

        # Write structured log (with numpy-safe encoder #35)
        import dataclasses
        log_entry = dataclasses.asdict(result)
        self.logger.info(json.dumps(log_entry, cls=_NumpySafeEncoder))

        # Update heartbeat (throttled to every 30s — #17)
        now = time.time()
        if now - self._last_heartbeat_time >= self._heartbeat_interval:
            self._registry.register_node(self.node_id, self.machine_name, status=result.action)
            self._last_heartbeat_time = now

        # Add drift flag to metadata
        result.metadata["concept_drift_detected"] = drift_detected

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

    def _build_multi_sensor_vector(self, sr: 'SensorReading', temp_norm: float) -> list:
        """
        Build a 5-element normalized sensor vector for v2 multivariate models.

        Maps to CMAPSS thermal sensors [s2, s3, s4, s7, s11]:
          s4 = primary temperature (temp_norm)
          s2 = load-correlated temperature (LPC outlet)
          s3 = HPC outlet temperature (slightly lagged)
          s7 = total temperature at HPT outlet
          s11 = static temperature at LPT outlet

        In production deployment, these come directly from hardware sensors.
        Here we synthesize correlated values from the SensorReading for compatibility.
        """
        # Primary sensor
        s4 = temp_norm

        # s2: load-correlated (LPC outlet) — higher load → higher temp
        s2 = float(np.clip(temp_norm * (0.85 + 0.15 * sr.load_pct), 0, 1))

        # s3: slightly lagged version (HPC outlet)
        window = list(self._temp_window)
        if len(window) >= 2:
            s3 = float(np.clip((window[-1] + window[-2]) / 2.0, 0, 1))
        else:
            s3 = temp_norm

        # s7: total temperature at HPT outlet — amplified version
        s7 = float(np.clip(temp_norm * 1.05 + 0.02 * sr.dT_dt, 0, 1))

        # s11: static temperature at LPT outlet — ambient-influenced
        ambient_norm = (sr.ambient_c - 15.0) / (self.profile["critical"] - 15.0) if sr.ambient_c else 0.3
        s11 = float(np.clip(temp_norm * 0.9 + ambient_norm * 0.1, 0, 1))

        return [s2, s3, s4, s7, s11]

    def _compute_if_score(self, temp_norm: float) -> float:
        # Phase 2: Per-machine damping replaces blanket CPU/Server bypass.
        # All machine types now get anomaly detection, with sensitivity damping
        # for erratic thermal profiles (CPU, Server).
        damping = Config.ANOMALY_DAMPING_FACTORS.get(self.machine_type_id, 1.0)
            
        # Cold-start bypass: skip the pickle-loaded IF entirely until the online
        # retrain has fired. The pickle was trained on CMAPSS multi-sensor data,
        # not our 5-feature serving vector — using it is both a train/serve
        # mismatch and the per-tick latency hotspot (P3.1). A cheap linear proxy
        # covers the warmup window until the serving-distribution IF is ready.
        if self._if_model is None or not self._if_has_retrained:
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
        return float(np.clip(score * damping, 0, 1))

    def _retrain_if(self):
        """Issue 5 — Online Isolation Forest retraining from rolling buffer.
        Adapts to seasonal/load-cycle drift by periodically refitting
        on the most recent observations."""
        try:
            from sklearn.ensemble import IsolationForest
            X = np.array(list(self._if_retrain_buffer))
            new_model = IsolationForest(
                contamination=Config.IF_CONTAMINATION,
                n_estimators=getattr(Config, "IF_N_ESTIMATORS", 100),
                random_state=Config.SEED,
            )
            new_model.fit(X)
            self._if_model = new_model
            self._if_has_retrained = True  # Cold-start resolved
            if self.verbose:
                print(f"  [IF] Online retrained on {len(X)} samples")
        except Exception as e:
            if self.verbose:
                print(f"  [IF] Retrain failed: {e}")

    def _compute_lstm_score(self, temp_norm: float):
        window = list(self._temp_window)
        prediction_interval = None

        if len(window) < Config.LSTM_WINDOW:
            # Not enough history yet — use linear trend
            pred_norm = temp_norm + 0.01
        elif self._foundation is not None and len(self._reading_history) >= Config.LSTM_WINDOW:
            # R1: zero-shot foundation forecaster on the raw temperature stream.
            try:
                H = Config.LSTM_FORECAST_HORIZON
                temps = [r.temp_c for r in self._reading_history][-33:]
                actual = float(temps[-1])
                # Residual anomaly: score the actual reading vs the PREVIOUS band.
                if self._prev_fc_band is not None:
                    from stage2_ml.foundation_forecaster import FoundationForecaster
                    self._foundation_anomaly = FoundationForecaster.residual_anomaly(actual, *self._prev_fc_band)
                fc = self._foundation.forecast(temps, horizon=H)
                p10, p50, p90 = fc["p10"], fc["p50"], fc["p90"]
                self._lstm_horizon = H
                self._lstm_qmode = False
                pred_norm = self._normalize_temp(float(p50[0]))
                self._last_forecast_traj = [self._normalize_temp(float(v)) for v in p50]
                self._last_forecast_band = [(self._normalize_temp(float(lo)), self._normalize_temp(float(hi)))
                                            for lo, hi in zip(p10, p90)]
                prediction_interval = {
                    "lower": round(float(p10[0]), 2),
                    "upper": round(float(p90[0]), 2),
                    "width_celsius": round(float(p90[0] - p10[0]), 2),
                    "quantile": round(Config.LSTM_QUANTILES[-1] - Config.LSTM_QUANTILES[0], 4),
                }
                self._prev_fc_band = (float(p10[0]), float(p50[0]), float(p90[0]))
                lstm_score = float(np.clip(abs(pred_norm - temp_norm), 0, 1))
                conformal_boost = Config.CONFORMAL_URGENCY_BOOST if prediction_interval["upper"] > self.profile["safe_max"] else 0.0
                return pred_norm, lstm_score, prediction_interval, conformal_boost
            except Exception:
                pass  # fall through to the trained LSTM on any error
            pred_norm = float(np.clip(temp_norm + (window[-1] - window[-3]) / 2 * 10, 0, 1)) if len(window) >= 3 else temp_norm
        elif self._lstm is not None and TORCH_AVAILABLE:
            if self._lstm_version == "tft" and len(self._multi_sensor_window) >= Config.LSTM_WINDOW:
                # Phase 3: TFT multivariate input (batch, 12, 5)
                multi_window = list(self._multi_sensor_window)
                x = torch.FloatTensor([multi_window])  # (1, 12, 5)
                with torch.no_grad():
                    quantiles, var_w, attn_w = self._lstm(x)
                    # Quantiles: [p10, p50, p90]
                    p10 = float(quantiles[:, 0].item())
                    pred_norm = float(quantiles[:, 1].item())
                    p90 = float(quantiles[:, 2].item())
                
                # Built-in interval from TFT
                prediction_interval = {
                    "lower": round(self._denormalize_temp(p10), 2),
                    "upper": round(self._denormalize_temp(p90), 2),
                    "width_celsius": round((p90 - p10) * (self.profile["critical"] - 15.0), 2),
                    "quantile": 0.80, # p90 - p10 coverage
                }
            elif self._lstm_version == "v2" and len(self._multi_sensor_window) >= Config.LSTM_WINDOW:
                # V2: multivariate BiLSTM+Attention input (batch, 12, 5)
                multi_window = list(self._multi_sensor_window)
                x = torch.FloatTensor([multi_window])  # (1, 12, 5)
                with torch.no_grad():
                    out = self._lstm(x)
                if self._lstm_qmode:
                    # P2.1: out is (1, H*Q) → (H, Q) per-step quantiles.
                    nq = len(self._lstm_quantiles)
                    arr = out[0].reshape(self._lstm_horizon, nq)
                    mid = nq // 2
                    p50 = arr[:, mid].tolist()
                    lo = arr[:, 0].tolist()
                    hi = arr[:, -1].tolist()
                    pred_norm = float(p50[0])
                    self._last_forecast_traj = p50
                    self._last_forecast_band = list(zip(lo, hi))
                    # Native next-step quantile interval (replaces MC-Dropout/conformal).
                    prediction_interval = {
                        "lower": round(self._denormalize_temp(lo[0]), 2),
                        "upper": round(self._denormalize_temp(hi[0]), 2),
                        "width_celsius": round((hi[0] - lo[0]) * (self.profile["critical"] - 15.0), 2),
                        "quantile": round(self._lstm_quantiles[-1] - self._lstm_quantiles[0], 4),
                    }
                elif self._lstm_horizon > 1:
                    # P1.5: direct multi-horizon — out is (1, H). Use the next step
                    # for the anomaly score / conformal interval (continuity), and
                    # stash the full forward trajectory for the dashboard.
                    traj = out[0].tolist()
                    pred_norm = float(traj[0])
                    self._last_forecast_traj = traj
                else:
                    pred_norm = float(out.item())
                    self._last_forecast_traj = None
            else:
                # V1: univariate input (batch, 12, 1)
                x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    pred_norm = float(self._lstm(x).item())

            # Epistemic Uncertainty via MC Dropout (skip for multi-horizon output —
            # conformal provides the interval; MCD assumes a scalar head).
            if self._lstm_version in ["v1", "v2"] and self._lstm_horizon == 1:
                 mc_input = torch.FloatTensor([list(self._multi_sensor_window)]) if self._lstm_version == "v2" else torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)
                 mc_res = self._mc_dropout.predict_with_uncertainty(self._lstm, mc_input)
                 # Only use prediction interval from MCD if conformal isn't available
                 if self._conformal is None or not self._conformal.is_calibrated:
                      prediction_interval = {
                          "lower": round(self._denormalize_temp(mc_res["lower_bound"]), 2),
                          "upper": round(self._denormalize_temp(mc_res["upper_bound"]), 2),
                          "width_celsius": round((mc_res["upper_bound"] - mc_res["lower_bound"]) * (self.profile["critical"] - 15.0), 2),
                          "quantile": 0.95,
                          "confidence_score": mc_res["confidence"]
                      }
                      
            # Conformal prediction interval (skip for TFT and for the P2.1
            # quantile forecaster, which supplies its own native interval).
            if self._lstm_version != "tft" and not self._lstm_qmode and self._conformal is not None and self._conformal.is_calibrated:
                interval = self._conformal.predict_interval(pred_norm)
                prediction_interval = {
                    "lower": round(self._denormalize_temp(interval["lower"]), 2),
                    "upper": round(self._denormalize_temp(interval["upper"]), 2),
                    "width_celsius": round(interval["width"] * (self.profile["critical"] - 15.0), 2),
                    "quantile": round(interval["quantile"], 6),
                }
        else:
            # Simple linear extrapolation
            trend = (window[-1] - window[-3]) / 2 if len(window) >= 3 else 0
            pred_norm = float(np.clip(temp_norm + trend * 10, 0, 1))

        lstm_score = float(np.clip(abs(pred_norm - temp_norm), 0, 1))

        # Urgency boost if conformal upper bound exceeds safe_max
        conformal_boost = 0.0
        if prediction_interval is not None:
            if prediction_interval["upper"] > self.profile["safe_max"]:
                conformal_boost = Config.CONFORMAL_URGENCY_BOOST

        return pred_norm, lstm_score, prediction_interval, conformal_boost

    def _compute_ae_score(self) -> float:
        # Phase 2: Per-machine damping replaces blanket CPU/Server bypass.
        damping = Config.ANOMALY_DAMPING_FACTORS.get(self.machine_type_id, 1.0)
            
        if self._ae_version == "v2":
            # V2: LSTM-AE with multivariate sequence input (batch, 12, 5)
            multi_window = list(self._multi_sensor_window)
            if len(multi_window) < Config.LSTM_WINDOW or self._ae_model is None or not TORCH_AVAILABLE:
                return 0.2
            x = torch.FloatTensor([multi_window])  # (1, 12, 5)
            with torch.no_grad():
                err = self._ae_model.reconstruction_error(x).item()
            # P1.4: POT/EVT-calibrated score when available, else raw ratio.
            score = self._ae_calib.score(err) if self._ae_calib else err / (self._ae_threshold + 1e-8)
            return float(np.clip(score * damping, 0, 1))
        else:
            # V1: univariate linear AE (batch, 12)
            window = list(self._temp_window)
            if len(window) < Config.LSTM_WINDOW or self._ae_model is None or not TORCH_AVAILABLE:
                return 0.2
            x = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                err = self._ae_model.reconstruction_error(x).item()
            score = err / (self._ae_threshold + 1e-8)
            return float(np.clip(score * damping, 0, 1))


    def _compute_vib_score(self, temp_norm: float) -> float:
        """
        Compute vibration anomaly score from synthesized vibration features.
        In a real deployment, these features would come from an accelerometer.
        Here we derive them from temperature dynamics (correlated physics model).
        """
        # Phase 2: Per-machine damping replaces blanket CPU/Server bypass.
        damping = Config.ANOMALY_DAMPING_FACTORS.get(self.machine_type_id, 1.0)

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

        # The vibration detector is now retrained on the serving distribution
        # (P1.1), so its reconstruction error is calibrated — an idle machine
        # scores low and faults exceed the threshold. No physics clamp needed.
        if not TORCH_AVAILABLE:
            # Fallback if torch somehow not available
            return float(np.clip(np.mean(np.abs(feat_norm)) * 0.5, 0, 1))
        x = torch.FloatTensor(feat_norm.reshape(1, -1))
        with torch.no_grad():
            error = self._vib_model.reconstruction_error(x).item()
        # P1.4: POT/EVT-calibrated score when available, else raw ratio.
        score = self._vib_calib.score(error) if self._vib_calib else error / (self._vib_threshold + 1e-8)
        return float(np.clip(score * damping, 0, 1))

    def _fuse(self, lstm_score, ae_score, if_score,
              cnn_score=0.5, audio_score=0.5, vib_score=0.3,
              cnn_var=None, audio_var=None):
        """6-modality attention fusion.
        
        Phase 2: Uses learned neural attention when model is available,
        falls back to hand-coded inverse-variance weighting otherwise.
        """
        scores_np = np.array([lstm_score, ae_score, if_score,
                              cnn_score, audio_score, vib_score], dtype=np.float32)

        # Phase 2: Use learned fusion if available
        if self._learned_fusion is not None and TORCH_AVAILABLE:
            context, contributions, top_contributor = self._learned_fusion.fuse_with_xai(scores_np)
            return context, contributions, top_contributor

        # Fallback: hand-coded inverse-variance weighting
        def w(var):
            return 1.0 if var is None else 1.0 / (1.0 + var)

        weights = np.array([1.0, 1.0, 1.0, w(cnn_var), w(audio_var), 1.0],
                            dtype=np.float32)
        weights /= weights.sum() + 1e-8

        context = float(np.clip(np.dot(weights, scores_np), 0, 1))
        
        # XAI Contributions
        total_impact = np.dot(weights, scores_np) + 1e-8
        contributions = {
            "trend_forecast": round(float(weights[0] * scores_np[0] / total_impact) * 100),
            "pattern_check":  round(float(weights[1] * scores_np[1] / total_impact) * 100),
            "outlier_scan":   round(float(weights[2] * scores_np[2] / total_impact) * 100),
            "cnn_hotspot":    round(float(weights[3] * scores_np[3] / total_impact) * 100),
            "audio":          round(float(weights[4] * scores_np[4] / total_impact) * 100),
            "vibration":      round(float(weights[5] * scores_np[5] / total_impact) * 100),
        }
        top_contributor = max(contributions, key=contributions.get) if context > 0.05 else "none"

        return context, contributions, top_contributor

    def _route(self, urgency: float) -> str:
        if urgency >= Config.EDGE_URGENCY_THRESHOLD:
            return "edge"
        elif urgency <= Config.CLOUD_URGENCY_THRESHOLD:
            return "cloud"
        return "both"

    def _build_obs(self, temp_norm, pred_norm, ae_score, urgency) -> np.ndarray:
        # Fix #30: use dynamic machine count instead of hardcoded 3.0
        max_machine_id = max(len(Config.MACHINE_PROFILES) - 1, 1)
        return np.array([
            temp_norm,
            pred_norm,
            ae_score,
            self.machine_type_id / float(max_machine_id),
            float(np.clip(self._steps_since_action / 100.0, 0, 1)),
            urgency,
        ], dtype=np.float32)

    def _decide(self, obs: np.ndarray, current_temp_raw: float, sr: 'SensorReading' = None) -> str:
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
            proposed_action = Config.ACTIONS[int(action_id)]
        else:
            # Rule-based fallback when PPO not loaded
            temp_norm = obs[0]
            urgency   = obs[5]
            if urgency > 0.7:
                proposed_action = "fan+"
            elif urgency > 0.5:
                proposed_action = "throttle"
            elif urgency > 0.35:
                proposed_action = "alert"
            else:
                proposed_action = "do-nothing"
                
        # Phase 3 — Digital Twin Safety Check
        if hasattr(self, '_digital_twin') and self._digital_twin is not None and sr is not None:
            # Estimate current fan speed (0.0 if not cooling, 1.0 if cooling) for simulation
            current_fan = 1.0 if proposed_action in ["fan+", "throttle"] else 0.0
            # Predict next 5 seconds
            trajectory = self._digital_twin.simulate_what_if(
                current_temp_raw, sr.load_pct, current_fan, [proposed_action], steps_per_action=5
            )
            # If the proposed action leads to critical failure in the next 5 seconds, override
            if any(t >= p["critical"] for t in trajectory):
                return "emergency-shutdown"
            elif any(t >= p["safe_max"] for t in trajectory) and proposed_action not in ["fan+", "throttle", "shutdown"]:
                return "throttle"
                
        return proposed_action

    def _estimate_rul(self, current_temp: float, predicted_temp: float, safe_max: float) -> Optional[float]:
        """Estimate remaining useful life in minutes before hitting safe_max.
        
        Phase 2: Uses learned RUL predictor when available (predicts cycles-to-failure
        from multivariate sensor window). Falls back to linear temperature extrapolation.
        """
        # Phase 2: Learned RUL predictor
        if self._rul_model is not None and TORCH_AVAILABLE:
            multi_window = list(self._multi_sensor_window)
            if len(multi_window) >= Config.LSTM_WINDOW:
                x = torch.FloatTensor([multi_window])  # (1, 12, 5)
                with torch.no_grad():
                    rul_cycles = float(self._rul_model(x).item())
                # Convert cycles to minutes (at 1Hz sampling, 1 cycle ≈ 1 second)
                rul_minutes = max(0.0, rul_cycles) / 60.0
                # Only trust the learned value when it's physically meaningful.
                # Near-zero outputs on a stable machine are model noise, not an
                # imminent failure. A low "minutes-to-limit" is only credible when
                # the temperature is actually trending toward the limit — otherwise
                # fall through to physics-based extrapolation, which returns None
                # when the temperature isn't rising (→ dashboard shows "Stable").
                rising = predicted_temp > current_temp + 0.5
                if rul_minutes >= 5.0 and rising:
                    return round(min(rul_minutes, 999.0), 1)

        # Fallback: physics extrapolation from the *sustained* recent trend.
        # A single one-step LSTM forecast is far too noisy to treat as a heating
        # rate (a momentary +10°C swing on an oscillating idle machine would
        # otherwise read as "seconds to limit"). Estimate the real °C/sec slope
        # by least-squares over the last ~30s of actual readings instead.
        temps = [r.temp_c for r in self._reading_history][-30:]
        if len(temps) < 6:
            return None  # not enough history yet — report "Stable"

        n = len(temps)
        xs = list(range(n))            # samples are 1 Hz → x is seconds
        mean_x = sum(xs) / n
        mean_y = sum(temps) / n
        denom = sum((x - mean_x) ** 2 for x in xs) or 1e-6
        slope = sum((xs[i] - mean_x) * (temps[i] - mean_y) for i in range(n)) / denom  # °C/sec

        # Require a genuine, sustained climb (> ~1.2°C/min) before reporting a
        # time-to-limit; otherwise the machine is effectively stable.
        if slope <= 0.02:
            return None

        remaining_degrees = safe_max - current_temp
        if remaining_degrees <= 0:
            return 0.0  # already past threshold

        minutes = (remaining_degrees / slope) / 60.0
        return round(min(minutes, 999.0), 1)

    def adapt_online(self):
        """R4 — label-free self-supervised adaptation of the anomaly AE on the
        buffered recent-normal windows, canary-guarded, with EVT recalibration."""
        if self._ae_model is None or self._ae_version != "v2" or not TORCH_AVAILABLE:
            return {"adopted": False, "reason": "no adaptable AE"}
        if len(self._normal_ae_buffer) < 40:
            return {"adopted": False, "reason": "insufficient normal data", "n": len(self._normal_ae_buffer)}
        from mhars.online_adapt import OnlineAdapter
        windows = np.array(list(self._normal_ae_buffer), dtype=np.float32)
        adopted, new_calib, info = OnlineAdapter.adapt(self._ae_model, windows)
        self._adapt_cooldown = 120  # ~2 min between adaptations
        if adopted:
            self._ae_calib = new_calib
            self._ae_threshold = getattr(new_calib, "z_q", self._ae_threshold)
            self._adapt_count += 1
        self._last_adaptation = {"adopted": bool(adopted), "count": self._adapt_count, **info}
        return self._last_adaptation

    def _fault_feature_vector(self, if_score, lstm_score, ae_score, vib_score, context, urgency):
        """P2.4: per-tick feature vector for the fault classifier — temperature
        dynamics from the reading history plus the per-model anomaly scores."""
        temps = [r.temp_c for r in self._reading_history][-8:]
        dT_dt = (temps[-1] - temps[-2]) if len(temps) >= 2 else 0.0
        slope5 = (temps[-1] - temps[-5]) / 4.0 if len(temps) >= 5 else 0.0
        std8 = float(np.std(temps)) if len(temps) >= 2 else 0.0
        max_dT = float(np.abs(np.diff(temps)).max()) if len(temps) >= 2 else 0.0
        return np.array([dT_dt, slope5, std8, max_dT,
                         if_score, lstm_score, ae_score, vib_score, context, urgency],
                        dtype=np.float32)

    def _classify_fault(self, features):
        """Return (fault_name, confidence, p_fault).
        p_fault = 1 - P(normal) is a calibrated anomaly-detection score (P2.2 —
        far stronger than the raw AE/fused scores: ROC-AUC ~0.93 vs ~0.55).
        (None, 0.0, 0.0) when no classifier is loaded."""
        if self._fault_clf is None or not TORCH_AVAILABLE:
            return None, 0.0, 0.0
        x = (features - self._fault_mean) / (self._fault_std + 1e-8)
        with torch.no_grad():
            probs = torch.softmax(self._fault_clf(torch.FloatTensor(x).unsqueeze(0)), dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        p_fault = float(1.0 - probs[0].item())
        name = Config.FAULT_CLASSES[idx] if idx < len(Config.FAULT_CLASSES) else "Unknown System Stress"
        return name, conf, p_fault

    def _fingerprint_anomaly(self, urgency: float, top_contributor: str, current_temp: float) -> str:
        """Map abstract ML scores to physical fault signatures."""
        if urgency < 0.55:
            return "Normal Operations"
            
        is_compute = self.machine_type_id in [0, 2] # CPU or Server
            
        if top_contributor == "vibration":
            return "Fan Bearing Issue" if is_compute else "Bearing / Mechanical Wear"
        elif top_contributor == "audio":
            return "Coil Whine / Fan Noise" if is_compute else "Acoustic Cavitation / Grinding"
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