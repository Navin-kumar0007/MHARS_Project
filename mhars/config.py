"""
MHARS Configuration
====================
Central config for the entire system. Edit this file to
customise thresholds, model paths, and deployment settings.
Import anywhere with: from mhars.config import Config
"""

import os

class Config:
    # ── Model paths ───────────────────────────────────────────────────────────
    MODELS_DIR           = os.path.join(os.path.dirname(__file__), '..', 'models')
    RESULTS_DIR          = os.path.join(os.path.dirname(__file__), '..', 'results')
    SEED                 = 42
    ISOLATION_FOREST     = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
    LSTM                 = os.path.join(MODELS_DIR, 'lstm.pt')
    AUTOENCODER          = os.path.join(MODELS_DIR, 'autoencoder.pt')
    AUTOENCODER_META     = os.path.join(MODELS_DIR, 'autoencoder_meta.json')
    PPO                  = os.path.join(MODELS_DIR, 'ppo_thermal.zip')
    LLM                  = os.path.join(MODELS_DIR, 'Phi-3-mini-4k-instruct-q4.gguf')
    VIBRATION_DETECTOR   = os.path.join(MODELS_DIR, 'vibration_detector.pt')
    VIBRATION_META       = os.path.join(MODELS_DIR, 'vibration_detector_meta.json')
    CNN_MODEL            = os.path.join(MODELS_DIR, 'efficientnet_cnn.pt')

    # ── V2 Enhanced model paths (Phase 1 Deep Analysis) ──────────────────────
    LSTM_V2              = os.path.join(MODELS_DIR, 'lstm_v2.pt')
    AUTOENCODER_V2       = os.path.join(MODELS_DIR, 'autoencoder_lstm_v2.pt')
    AUTOENCODER_V2_META  = os.path.join(MODELS_DIR, 'autoencoder_lstm_v2_meta.json')
    RUL_MODEL            = os.path.join(MODELS_DIR, 'rul_predictor.pt')
    CONFORMAL_META       = os.path.join(MODELS_DIR, 'conformal_meta.json')

    # ── Multivariate sensor config ────────────────────────────────────────────
    THERMAL_SENSORS      = ["s2", "s3", "s4", "s7", "s11"]
    LSTM_INPUT_SIZE_V2   = 5   # number of sensor channels for BiLSTM
    LSTM_HIDDEN_V2       = 128
    LSTM_LAYERS_V2       = 2

    # ── Conformal prediction ─────────────────────────────────────────────────
    CONFORMAL_COVERAGE   = 0.90   # 90% prediction interval coverage
    CONFORMAL_URGENCY_BOOST = 0.15  # urgency boost when upper bound exceeds safe_max

    import json

    # ── Machine types ─────────────────────────────────────────────────────────
    # Dynamically load from JSON file to allow operators to add profiles
    # without touching source code.
    _machines_file = os.path.join(os.path.dirname(__file__), 'machines.json')
    MACHINE_PROFILES = {}
    MACHINE_TYPES = {}
    
    if os.path.exists(_machines_file):
        try:
            with open(_machines_file, 'r') as f:
                _raw_profiles = json.load(f)
                if _raw_profiles:
                    for k, v in _raw_profiles.items():
                        machine_id = int(k)
                        MACHINE_PROFILES[machine_id] = v
                        MACHINE_TYPES[machine_id] = v["name"]
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Issue 1 — Malformed config fallback
            print(f"  [ERROR] Malformed or invalid machines.json: {e}. Falling back to defaults.")
            _raw_profiles = None # Force fallback below
    
    if not MACHINE_PROFILES:
        # Fallback if file goes missing or was malformed
        MACHINE_TYPES = { 0: "CPU", 1: "Motor", 2: "Server", 3: "Engine" }
        MACHINE_PROFILES = {
            0: {"name": "CPU", "safe_max": 85.0, "critical": 100.0, "idle": 45.0, "thermal_mass_J_K": 12.0, "conv_coeff": 0.08, "target_temp": 65.0, "heat_rate": 2.5},
            1: {"name": "Motor", "safe_max": 80.0, "critical": 95.0, "idle": 40.0, "thermal_mass_J_K": 25.0, "conv_coeff": 0.05, "target_temp": 60.0, "heat_rate": 1.8},
            2: {"name": "Server", "safe_max": 75.0, "critical": 90.0, "idle": 35.0, "thermal_mass_J_K": 18.0, "conv_coeff": 0.07, "target_temp": 55.0, "heat_rate": 1.5},
            3: {"name": "Engine", "safe_max": 100.0, "critical": 115.0, "idle": 60.0, "thermal_mass_J_K": 40.0, "conv_coeff": 0.03, "target_temp": 80.0, "heat_rate": 3.8},
        }

    # ── RL Router thresholds ──────────────────────────────────────────────────
    EDGE_URGENCY_THRESHOLD  = 0.8   # above → edge only  (< 50 ms)
    CLOUD_URGENCY_THRESHOLD = 0.4   # below → cloud only

    # ── Anomaly detection ─────────────────────────────────────────────────────
    IF_CONTAMINATION        = 0.03  # Isolation Forest expected noise ratio
    AE_THRESHOLD_PERCENTILE = 95    # Autoencoder anomaly threshold percentile
    LSTM_WINDOW             = 12    # sliding window size for LSTM
    LSTM_PREDICTION_HORIZON_S = 1   # next-step horizon for the anomaly score (1Hz)
    # P1.5: direct multi-horizon forecast — model predicts the next H steps so the
    # dashboard shows a real forward trajectory (not a single 1-step point).
    LSTM_FORECAST_HORIZON   = 10    # steps ahead (seconds at 1Hz)
    # P2.1: per-step quantiles → native uncertainty band (replaces MC-Dropout).
    # Forecaster head emits H * len(LSTM_QUANTILES) values, reshaped to (H, Q).
    LSTM_QUANTILES          = [0.1, 0.5, 0.9]   # p10 / p50 / p90

    # ── PPO training ──────────────────────────────────────────────────────────
    PPO_TIMESTEPS    = 500_000
    PPO_N_ENVS       = 4
    PPO_CLIP_RANGE   = 0.2
    PPO_LEARNING_RATE = 3e-4

    # Issue 3 — Configurable reward function weights
    # Operators can tune these without touching Python logic
    PPO_REWARD = {
        "safe_bonus":           1.0,   # reward per step temp is in safe range
        "tracking_weight":     -0.5,   # penalty scaled by |temp - target|/target
        "unnecessary_action":  -2.0,   # penalty for intervening when temp is fine
        "breach_penalty":     -10.0,   # hard penalty for critical temp / damage
        "overshoot_scale":     -3.0,   # scaled penalty when between safe_max and critical
        "oscillation_penalty": -0.3,   # penalty for changing action every step
        "fan_energy_cost":     -0.05,  # small ongoing cost for running fan at max
    }

    # ── LLM settings ─────────────────────────────────────────────────────────
    LLM_MAX_TOKENS  = 120
    LLM_TEMPERATURE = 0.3
    LLM_N_CTX       = 512

    # ── Actions ───────────────────────────────────────────────────────────────
    ACTIONS = {
        0: "do-nothing",
        1: "fan+",
        2: "throttle",
        3: "alert",
        4: "shutdown",
        5: "emergency-shutdown",   # hardware safety override (not PPO-selectable)
    }

    # ── Online learning ───────────────────────────────────────────────────────
    IF_ONLINE_RETRAIN       = True   # Toggle online Isolation Forest retraining
    IF_RETRAIN_INTERVAL     = 500    # Retrain every N samples (was 100, too frequent)
    IF_COLD_START_SAMPLES   = 50     # Skip IF pickle until online retrain fires once

    # ── Phase 2: Anomaly detection damping (replaces blanket CPU/Server bypass) ──
    # Factor ∈ (0, 1] applied to IF/AE/Vib scores per machine type.
    # 1.0 = full score (Motors, Engines), <1.0 = damped (CPU, Server).
    ANOMALY_DAMPING_FACTORS = {
        0: 0.30,   # CPU  — high-frequency thermal fluctuations → heavy damping
        1: 1.00,   # Motor — mechanical inertia → full sensitivity
        2: 0.40,   # Server — moderate fluctuations → moderate damping
        3: 1.00,   # Engine — heavy thermal mass → full sensitivity
    }

    # ── Phase 2: Learned Attention Fusion ─────────────────────────────────────
    LEARNED_FUSION_MODEL = os.path.join(MODELS_DIR, 'learned_fusion.pt')
    FUSION_D_MODEL       = 32    # embedding dimension per modality
    FUSION_N_HEADS       = 4     # multi-head attention heads
    FUSION_N_MODALITIES  = 6     # lstm, ae, if, cnn, audio, vibration

    # ── Phase 2: RUL Prediction ──────────────────────────────────────────────
    RUL_MODEL_V2         = os.path.join(MODELS_DIR, 'rul_predictor_v2.pt')
    RUL_MAX_CYCLES       = 125   # piece-wise linear cap per NASA convention

    # ── P2.4: Supervised fault classifier ────────────────────────────────────
    FAULT_CLASSIFIER     = os.path.join(MODELS_DIR, 'fault_classifier.pt')
    FAULT_CLASSIFIER_META = os.path.join(MODELS_DIR, 'fault_classifier_meta.json')
    # Index order == training label encoding (0 = normal).
    FAULT_CLASSES        = ["Normal Operations", "Heat Spike", "Bearing Wear",
                            "Fan Blockage", "Sensor Drift", "Power Surge"]
    FAULT_ANOMALY_KEYS   = ["normal", "temperature_spike", "bearing_wear",
                            "fan_blockage", "sensor_drift", "power_surge"]
    FAULT_MIN_CONFIDENCE = 0.55  # below this → fall back to rule-based fingerprint

    # ── Phase 2: SAC Agent ───────────────────────────────────────────────────
    SAC_MODEL            = os.path.join(MODELS_DIR, 'sac_thermal.zip')

    # ── Phase 2: Enhanced Environment ────────────────────────────────────────
    ENV_V2_OBS_DIM       = 12    # expanded observation space
    PPO_REWARD_V2 = {
        **PPO_REWARD,
        "energy_efficiency_bonus": 0.1,   # reward for low fan speed while safe
        "proactive_cooling_bonus": 0.3,   # reward for cooling before warning zone
        "rul_penalty_scale":      -2.0,   # penalty proportional to low RUL
        "smoothness_reward":      -0.15,  # penalize rapid fan speed oscillations
    }

    # ── Phase 3: TFT Predictor ───────────────────────────────────────────────
    TFT_MODEL            = os.path.join(MODELS_DIR, 'tft_predictor.pt')
    TFT_D_MODEL          = 32
    TFT_N_HEADS          = 4
    TFT_NUM_QUANTILES    = 3