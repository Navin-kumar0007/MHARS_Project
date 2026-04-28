"""
MHARS Trainer
==============
One command to train all models from scratch.
Wraps Stages 2 and 3 into clean train() calls.

Usage:
    from mhars.trainer import MHARSTrainer
    trainer = MHARSTrainer()
    trainer.train_all()           # train everything
    trainer.train_ml_only()       # Isolation Forest + LSTM + Autoencoder
    trainer.train_ppo(machine=0)  # PPO for one machine type
"""

import os, sys
from mhars.config import Config


class MHARSTrainer:
    """Trains all MHARS models and saves them to the models/ directory."""

    def __init__(self, data_path: str = None, verbose: bool = True):
        self.data_path = data_path
        self.verbose   = verbose
        os.makedirs(Config.MODELS_DIR,  exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def train_all(self, ppo_machine: int = 0, ppo_timesteps: int = None):
        """Train every model in the correct order."""
        print("\n[MHARS Trainer] Training all components\n")
        self.train_isolation_forest()
        self.train_lstm()
        self.train_autoencoder()
        self.train_ppo(machine=ppo_machine, timesteps=ppo_timesteps)
        print("\n[MHARS Trainer] All models trained ✓")

    def train_ml_only(self):
        """Train only the ML layer (no PPO)."""
        self.train_isolation_forest()
        self.train_lstm()
        self.train_autoencoder()

    def train_isolation_forest(self):
        print("[Trainer] Training Isolation Forest...")
        from stage2_ml.isolation_forest import run_training
        run_training(model_path=Config.ISOLATION_FOREST)

    def train_lstm(self):
        print("[Trainer] Training LSTM...")
        from stage2_ml.lstm_predictor import run_training
        run_training(model_path=Config.LSTM)

    def train_autoencoder(self):
        print("[Trainer] Training Autoencoder...")
        from stage2_ml.autoencoder import run_training
        run_training(model_path=Config.AUTOENCODER)

    def train_ppo(self, machine: int = 0, timesteps: int = None):
        print(f"[Trainer] Training PPO for machine {machine}...")
        from stage3_ai.ppo_agent import run_training
        run_training(
            machine_type_id = machine,
            timesteps       = timesteps or Config.PPO_TIMESTEPS,
            model_path      = Config.PPO,
        )

    # ── Internal ───────────────────────────────────────────────────────────────
    def _base(self):
        return os.path.join(os.path.dirname(__file__), '..')

    def _stage1_path(self):
        return os.path.join(self._base(), 'stage1_simulation')

    def _stage2_path(self):
        return os.path.join(self._base(), 'stage2_ml')

    def _stage3_path(self):
        return os.path.join(self._base(), 'stage3_ai')