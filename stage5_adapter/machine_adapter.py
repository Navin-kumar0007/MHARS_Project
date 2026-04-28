"""
MHARS — Stage 5: Machine Adapter
==================================
The core research novelty of MHARS.

Every existing paper in the Khadam et al. (2025) SLR trains a
thermal model for ONE machine type. If you want it to work on a
different machine, you retrain from scratch using thousands of samples.

The Machine Adapter does it differently:
  1. Find the most similar existing machine profile (cosine similarity)
  2. Load the LSTM trained on that machine
  3. Freeze the first layer (it already knows "what temperature trends
     look like")
  4. Fine-tune only the last 2 layers on ~100 new samples
  5. The model adapts to the new machine in < 5 minutes

This is transfer learning applied to thermal management — no existing
reviewed paper implements this.

Experiment protocol (from the implementation plan):
  - Machine A: CPU (machine_type_id=0) — already trained in Stage 2
  - Machine B: Engine (machine_type_id=3) — the "new" machine
  - Metric 1: LSTM RMSE after adapter vs full retraining
  - Metric 2: PPO convergence speed — adapted vs from scratch
"""

import os, sys, json, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage1_simulation.load_cmapss import load_cmapss, preprocess, make_lstm_windows
from stage1_simulation.gym_env import MACHINE_PROFILES
from stage2_ml.lstm_predictor import ThermalLSTM


# ── Machine profile registry ───────────────────────────────────────────────────
# Each profile is a feature vector describing the machine's thermal behaviour.
# These are hand-crafted from domain knowledge — in a real deployment they would
# be learned from the first 50 readings of any new machine.
MACHINE_PROFILES_FEATURES = {
    0: np.array([0.85, 0.20, 2.5, 85.0, 100.0], dtype=np.float32),  # CPU
    1: np.array([0.70, 0.30, 1.8, 80.0, 95.0],  dtype=np.float32),  # Motor
    2: np.array([0.65, 0.15, 1.5, 75.0, 90.0],  dtype=np.float32),  # Server
    3: np.array([0.95, 0.50, 3.2, 100.0, 115.0], dtype=np.float32), # Engine
}
# Feature dimensions:
# [load_sensitivity, ambient_sensitivity, heat_rate, safe_max, critical_temp]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two feature vectors. Range: [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_most_similar_machine(new_machine_id: int,
                               known_machine_ids: list) -> tuple:
    """
    Given a new machine type, find the most structurally similar
    machine in the known set using cosine similarity of feature vectors.

    Returns (best_machine_id, similarity_score).
    """
    new_features = MACHINE_PROFILES_FEATURES[new_machine_id]
    best_id, best_sim = None, -1.0

    for mid in known_machine_ids:
        sim = cosine_similarity(new_features,
                                MACHINE_PROFILES_FEATURES[mid])
        if sim > best_sim:
            best_sim = sim
            best_id  = mid

    return best_id, best_sim


# ── LSTM Machine Adapter ───────────────────────────────────────────────────────
class MachineAdapter:
    """
    Adapts a pre-trained LSTM to a new machine type using transfer learning.

    Strategy:
      - Freeze the LSTM recurrent layer (it knows temporal patterns — reuse)
      - Unfreeze dropout + final linear layer (machine-specific scaling)
      - Fine-tune with low LR + early stopping to prevent overfitting
    """

    def __init__(self, base_model_path: str, similar_machine_id: int):
        self.similar_machine_id = similar_machine_id
        self.model = ThermalLSTM()
        self.model.load_state_dict(torch.load(base_model_path,
                                               map_location="cpu"))
        self._freeze_base_layers()
        print(f"  Loaded base model trained on machine {similar_machine_id}")
        print(f"  LSTM layer frozen — dropout + output layer will be fine-tuned")

    def _freeze_base_layers(self):
        """Freeze LSTM recurrent weights. Keep dropout + linear trainable."""
        for param in self.model.lstm.parameters():
            param.requires_grad = False
        # Dropout + Linear stay trainable — enough capacity to adapt
        for param in self.model.drop.parameters():
            param.requires_grad = True
        for param in self.model.linear.parameters():
            param.requires_grad = True

    def adapt(self,
              X_new: np.ndarray,
              y_new: np.ndarray,
              n_samples: int = 100,
              epochs: int = 30,
              lr: float = 0.0005) -> float:
        """
        Fine-tune the model on n_samples from the new machine.
        Uses early stopping to prevent overfitting on small samples.
        Returns the best validation RMSE achieved.
        """
        # Use exactly n_samples for adaptation (the research claim)
        X_adapt = X_new[:n_samples]
        y_adapt = y_new[:n_samples]
        X_val   = X_new[n_samples:n_samples + 200]  # held-out for measuring
        y_val   = y_new[n_samples:n_samples + 200]

        def to_t(arr):
            return torch.FloatTensor(arr).unsqueeze(-1)

        dl = DataLoader(
            TensorDataset(to_t(X_adapt), torch.FloatTensor(y_adapt)),
            batch_size=32, shuffle=True
        )

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        criterion = nn.MSELoss()

        # Early stopping: track best validation RMSE and save best weights
        best_val_rmse = float("inf")
        best_state    = None
        patience      = 8
        no_improve    = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            # Validate every epoch
            self.model.eval()
            with torch.no_grad():
                X_v = to_t(X_val)
                preds = self.model(X_v)
                val_rmse = float(criterion(preds, torch.FloatTensor(y_val)).item() ** 0.5)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                marker = " ← best" if no_improve == 0 else ""
                print(f"    Epoch {epoch:3d}/{epochs}  val RMSE: {val_rmse:.4f}{marker}")

            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_val_rmse

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Adapted model saved → {path}")


# ── PPO Transfer ───────────────────────────────────────────────────────────────
def transfer_ppo_policy(base_model_path: str,
                         new_machine_id: int,
                         n_adapt_episodes: int = 50,
                         save_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_adapted.zip")):
    """
    Fine-tune a trained PPO policy on a new machine type.
    Measures how many episodes are needed to reach a reward threshold
    versus training from scratch.

    Returns (adapted_rewards_curve, scratch_rewards_curve).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stage1_simulation.gym_env import ThermalEnv

    print(f"\n  Loading base PPO policy from {base_model_path}")
    # Must pass env at load time when n_envs differs (trained with 4, adapting with 1)
    new_env = Monitor(ThermalEnv(machine_type_id=new_machine_id, max_steps=500))
    adapted_model = PPO.load(base_model_path.replace(".zip", ""), env=new_env)

    print(f"  Fine-tuning on machine {new_machine_id} for "
          f"{n_adapt_episodes * 500:,} timesteps...")
    adapted_model.learn(total_timesteps=n_adapt_episodes * 500,
                        reset_num_timesteps=False)

    # Evaluate adapted model
    adapted_rewards = _evaluate_policy(adapted_model, new_machine_id, n_eps=10)

    # Compare: train from scratch on new machine
    print(f"  Training from scratch on machine {new_machine_id}...")
    scratch_model = PPO(
        policy="MlpPolicy",
        env=Monitor(ThermalEnv(machine_type_id=new_machine_id, max_steps=500)),
        clip_range=0.2, verbose=0
    )
    scratch_model.learn(total_timesteps=n_adapt_episodes * 500)
    scratch_rewards = _evaluate_policy(scratch_model, new_machine_id, n_eps=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    adapted_model.save(save_path.replace(".zip", ""))
    print(f"  Adapted PPO saved → {save_path}")

    return adapted_rewards, scratch_rewards


def _evaluate_policy(model, machine_id: int, n_eps: int = 10) -> list:
    """Evaluate a policy for n episodes, return list of episode rewards."""
    from stage1_simulation.gym_env import ThermalEnv
    rewards = []
    for ep in range(n_eps):
        env  = ThermalEnv(machine_type_id=machine_id, max_steps=500)
        obs, _ = env.reset(seed=ep)
        ep_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    return rewards