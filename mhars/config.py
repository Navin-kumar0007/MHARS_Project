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

    # ── Machine types ─────────────────────────────────────────────────────────
    MACHINE_TYPES = {
        0: "CPU",
        1: "Motor",
        2: "Server",
        3: "Engine",
    }

    MACHINE_PROFILES = {
        0: {"name": "CPU",    "safe_max": 85.0,  "critical": 100.0, "idle": 45.0},
        1: {"name": "Motor",  "safe_max": 80.0,  "critical": 95.0,  "idle": 40.0},
        2: {"name": "Server", "safe_max": 75.0,  "critical": 90.0,  "idle": 35.0},
        3: {"name": "Engine", "safe_max": 100.0, "critical": 115.0, "idle": 60.0},
    }

    # ── RL Router thresholds ──────────────────────────────────────────────────
    EDGE_URGENCY_THRESHOLD  = 0.8   # above → edge only  (< 50 ms)
    CLOUD_URGENCY_THRESHOLD = 0.4   # below → cloud only

    # ── Anomaly detection ─────────────────────────────────────────────────────
    IF_CONTAMINATION        = 0.03  # Isolation Forest expected noise ratio
    AE_THRESHOLD_PERCENTILE = 95    # Autoencoder anomaly threshold percentile
    LSTM_WINDOW             = 12    # sliding window size for LSTM

    # ── PPO training ──────────────────────────────────────────────────────────
    PPO_TIMESTEPS    = 500_000
    PPO_N_ENVS       = 4
    PPO_CLIP_RANGE   = 0.2
    PPO_LEARNING_RATE = 3e-4

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
    }

    # ── Results / logging ─────────────────────────────────────────────────────
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')