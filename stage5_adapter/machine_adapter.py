"""
MHARS — Stage 5: Machine Adapter (FIXED)
==========================================
Fix applied: ISSUE-5 — cosine similarity now normalizes feature vectors
before computing similarity, so all 5 features contribute equally.

Previous bug: feature vector values ranged from 0.2 to 115.0 — the large
temperature values (safe_max, critical_temp) completely dominated the dot
product, making all similarities return 1.000.

Fix: MinMax normalize each feature dimension across all known machines
before computing cosine similarity.
"""

import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage2_ml'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage3_ai'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from load_cmapss import load_cmapss, preprocess, make_lstm_windows
from lstm_predictor import ThermalLSTM
from gym_env import MACHINE_PROFILES


# ── Machine feature registry ───────────────────────────────────────────────────
# Features: [load_sensitivity, ambient_sensitivity, heat_rate, safe_max, critical_temp]
# These differ enough across machine types that normalized similarity
# should clearly separate them.
MACHINE_PROFILES_FEATURES = {
    0: np.array([0.85, 0.20, 2.5,  85.0, 100.0], dtype=np.float32),  # CPU
    1: np.array([0.70, 0.30, 1.8,  80.0,  95.0], dtype=np.float32),  # Motor
    2: np.array([0.65, 0.15, 1.5,  75.0,  90.0], dtype=np.float32),  # Server
    3: np.array([0.95, 0.50, 3.2, 100.0, 115.0], dtype=np.float32),  # Engine
}


def _normalize_features(feature_dict: dict) -> dict:
    """
    MinMax normalize each feature dimension across all machine profiles.
    This ensures the safe_max (75–100) and critical_temp (90–115) ranges
    don't dominate over load_sensitivity (0.65–0.95) and heat_rate (1.5–3.2).
    Returns a new dict with normalized vectors.
    """
    all_vecs = np.stack(list(feature_dict.values()))   # shape (N_machines, N_features)
    mean = all_vecs.mean(axis=0)
    std = all_vecs.std(axis=0) + 1e-8

    normalized = {}
    for machine_id, vec in feature_dict.items():
        normalized[machine_id] = (vec - mean) / std
    return normalized


# Pre-compute normalized features at module load time
_NORMALIZED_FEATURES = _normalize_features(MACHINE_PROFILES_FEATURES)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_most_similar_machine(new_machine_id: int,
                               known_machine_ids: list) -> tuple:
    """
    Find most similar known machine using normalized cosine similarity.
    Now correctly differentiates machine types instead of always returning 1.0.

    Returns (best_machine_id, similarity_score).
    """
    new_features = _NORMALIZED_FEATURES[new_machine_id]
    best_id, best_sim = None, -1.0

    print(f"\n  Similarity scores for machine {new_machine_id} "
          f"({MACHINE_PROFILES[new_machine_id]['name']}):")
    for mid in known_machine_ids:
        sim = cosine_similarity(new_features, _NORMALIZED_FEATURES[mid])
        name = MACHINE_PROFILES[mid]['name']
        print(f"    vs {name:<8} (id={mid}): {sim:.4f}")
        if sim > best_sim:
            best_sim = sim
            best_id  = mid

    return best_id, best_sim


# ── LSTM Machine Adapter ───────────────────────────────────────────────────────
class MachineAdapter:
    """
    Adapts a pre-trained LSTM to a new machine type using transfer learning.
    Freezes the LSTM layer, fine-tunes only the linear output head.
    """

    def __init__(self, base_model_path: str, similar_machine_id: int):
        self.similar_machine_id = similar_machine_id

        # Auto-detect hidden_size from checkpoint (ISSUE fix from core.py)
        checkpoint = torch.load(base_model_path, map_location="cpu")
        hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
        self.model = ThermalLSTM(hidden_size=hidden_size)
        self.model.load_state_dict(checkpoint)
        self._freeze_base_layers()
        print(f"  Loaded base model (machine {similar_machine_id}, hidden={hidden_size})")
        print(f"  LSTM layer frozen — only output layer will be fine-tuned")

    def _freeze_base_layers(self):
        for param in self.model.lstm.parameters():
            param.requires_grad = False
        for param in self.model.linear.parameters():
            param.requires_grad = True

    def adapt(self, X_new, y_new, n_samples=100, epochs=20, lr=0.005):
        X_adapt = X_new[:n_samples]
        y_adapt = y_new[:n_samples]
        X_val   = X_new[n_samples:n_samples + 200]
        y_val   = y_new[n_samples:n_samples + 200]

        def to_t(arr):
            return torch.FloatTensor(arr).unsqueeze(-1)

        dl = DataLoader(
            TensorDataset(to_t(X_adapt), torch.FloatTensor(y_adapt)),
            batch_size=32, shuffle=True
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in dl:
                optimizer.zero_grad()
                criterion(self.model(xb), yb).backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            val_rmse = float(
                criterion(self.model(to_t(X_val)),
                          torch.FloatTensor(y_val)).item() ** 0.5
            )
        return val_rmse

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Adapted model saved → {path}")


# ── PPO Transfer ───────────────────────────────────────────────────────────────
def transfer_ppo_policy(base_model_path, new_machine_id,
                         n_adapt_episodes=50,
                         save_path="../models/ppo_adapted.zip"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from gym_env import ThermalEnv

    print(f"\n  Loading base PPO from {base_model_path}")
    new_env = Monitor(ThermalEnv(machine_type_id=new_machine_id, max_steps=500))
    adapted_model = PPO.load(base_model_path.replace(".zip", ""), env=new_env)

    print(f"  Fine-tuning on machine {new_machine_id} for "
          f"{n_adapt_episodes * 500:,} timesteps...")
    adapted_model.learn(total_timesteps=n_adapt_episodes * 500,
                        reset_num_timesteps=False)
    adapted_rewards = _evaluate_policy(adapted_model, new_machine_id)

    print(f"  Training from scratch on machine {new_machine_id}...")
    scratch_model = PPO(
        policy="MlpPolicy",
        env=Monitor(ThermalEnv(machine_type_id=new_machine_id, max_steps=500)),
        clip_range=0.2, verbose=0
    )
    scratch_model.learn(total_timesteps=n_adapt_episodes * 500)
    scratch_rewards = _evaluate_policy(scratch_model, new_machine_id)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    adapted_model.save(save_path.replace(".zip", ""))
    print(f"  Adapted PPO saved → {save_path}")

    return adapted_rewards, scratch_rewards


def _evaluate_policy(model, machine_id, n_eps=10):
    from gym_env import ThermalEnv
    rewards = []
    for ep in range(n_eps):
        env = ThermalEnv(machine_type_id=machine_id, max_steps=500)
        obs, _ = env.reset(seed=ep)
        ep_r, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    return rewards


# ── Similarity validation ──────────────────────────────────────────────────────
def validate_similarity():
    """
    Verifies the fix works — similarity values should now be < 1.0
    and different for different machine pairs.
    """
    print("\n── Similarity Validation (ISSUE-5 fix) ──────────────────────")
    print("  Raw feature vectors:")
    for mid, vec in MACHINE_PROFILES_FEATURES.items():
        name = MACHINE_PROFILES[mid]['name']
        print(f"    {name:8s}: {vec}")

    print("\n  Normalized feature vectors:")
    for mid, vec in _NORMALIZED_FEATURES.items():
        name = MACHINE_PROFILES[mid]['name']
        print(f"    {name:8s}: {np.round(vec, 3)}")

    print("\n  Pairwise similarities (normalized):")
    ids = list(MACHINE_PROFILES_FEATURES.keys())
    all_different = True
    sims = set()
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a_name = MACHINE_PROFILES[ids[i]]['name']
            b_name = MACHINE_PROFILES[ids[j]]['name']
            sim = cosine_similarity(
                _NORMALIZED_FEATURES[ids[i]],
                _NORMALIZED_FEATURES[ids[j]]
            )
            sims.add(round(sim, 3))
            print(f"    {a_name:8s} ↔ {b_name:8s}: {sim:.4f}")
            if sim >= 0.9999:
                all_different = False
                print(f"    ⚠  Still returning near-1.0 — check feature vectors")

    if all_different and len(sims) > 1:
        print("\n  ✓  All similarities are distinct and < 1.0 (ISSUE-5 FIXED)")
    else:
        print("\n  ⚠  Some similarities are identical or near 1.0")

    return all_different


if __name__ == "__main__":
    validate_similarity()