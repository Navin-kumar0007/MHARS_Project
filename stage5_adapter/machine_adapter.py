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


# ── LSTM Machine Adapter V1 ────────────────────────────────────────────────
class MachineAdapterV1:
    """
    V1 adapter: Freezes LSTM, fine-tunes only linear head.
    Kept for backward compatibility.
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


# Keep backward-compatible alias
MachineAdapter = MachineAdapterV1


# ── Progressive Machine Adapter V2 (Phase 2) ──────────────────────────────
class ProgressiveMachineAdapter:
    """
    Phase 2: 3-phase progressive unfreezing for better few-shot transfer.
    
    Phase 1 (epochs 1–10):  Only output layer, LR=0.01
    Phase 2 (epochs 11–25): Output + last LSTM layer, LR=0.001
    Phase 3 (epochs 26–30): All layers with tiny LR=0.0001
    
    Supports both V1 (ThermalLSTM) and V2 (ThermalLSTMv2) models,
    auto-detected from checkpoint shape.
    """

    def __init__(self, base_model_path: str, similar_machine_id: int):
        self.similar_machine_id = similar_machine_id
        self.is_v2 = False
        
        checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)
        
        # Auto-detect V1 vs V2 from checkpoint keys
        if "attention_weight.weight" in checkpoint:
            # V2: BiLSTM+Attention
            from mhars.models import ThermalLSTMv2
            hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
            input_size = checkpoint["lstm.weight_ih_l0"].shape[1]
            self.model = ThermalLSTMv2(input_size=input_size, hidden_size=hidden_size)
            self.is_v2 = True
            print(f"  [Adapter] Detected V2 model (BiLSTM+Attention, input={input_size})")
        else:
            # V1: Standard LSTM
            hidden_size = checkpoint["lstm.weight_ih_l0"].shape[0] // 4
            self.model = ThermalLSTM(hidden_size=hidden_size)
            print(f"  [Adapter] Detected V1 model (LSTM, hidden={hidden_size})")

        self.model.load_state_dict(checkpoint)
        self.phase_history = []  # track RMSE per phase
        print(f"  [Adapter] Source machine: {similar_machine_id}")

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_name):
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = True

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def _train_phase(self, dl, optimizer, criterion, epochs, phase_name, verbose=True):
        """Train for a specified number of epochs and return avg loss."""
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for xb, yb in dl:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
                n += len(xb)
        return epoch_loss / max(n, 1)

    def _evaluate(self, X_val, y_val):
        """Compute RMSE on validation data."""
        self.model.eval()
        if self.is_v2:
            x = torch.FloatTensor(X_val)
        else:
            x = torch.FloatTensor(X_val).unsqueeze(-1)
        with torch.no_grad():
            preds = self.model(x).numpy()
        rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
        return rmse

    def adapt(self, X_new, y_new, n_samples=100, verbose=True):
        """
        3-phase progressive adaptation.
        
        Args:
            X_new: Training data. V1: (N, window), V2: (N, window, 5)
            y_new: Target values (N,)
            n_samples: Number of samples to use for adaptation
        
        Returns:
            Final validation RMSE
        """
        X_adapt = X_new[:n_samples]
        y_adapt = y_new[:n_samples]
        X_val = X_new[n_samples:n_samples + 200]
        y_val = y_new[n_samples:n_samples + 200]

        if self.is_v2:
            x_tensor = torch.FloatTensor(X_adapt)
        else:
            x_tensor = torch.FloatTensor(X_adapt).unsqueeze(-1)

        dl = DataLoader(
            TensorDataset(x_tensor, torch.FloatTensor(y_adapt)),
            batch_size=32, shuffle=True,
        )
        criterion = nn.MSELoss()

        # ── Phase 1: Output layer only (10 epochs, LR=0.01) ──
        if verbose:
            print(f"  [Phase 1] Fine-tuning output layer only...")
        self._freeze_all()
        self._unfreeze_layer("linear")
        if self.is_v2:
            self._unfreeze_layer("attention_weight")  # also unfreeze attention head
        optimizer = torch.optim.Adam(self._get_trainable_params(), lr=0.01)
        self._train_phase(dl, optimizer, criterion, epochs=10, phase_name="Phase 1")
        rmse_p1 = self._evaluate(X_val, y_val)
        self.phase_history.append(("Phase 1: Output only", rmse_p1))
        if verbose:
            print(f"    RMSE after Phase 1: {rmse_p1:.4f}")

        # ── Phase 2: Output + last LSTM layer (15 epochs, LR=0.001) ──
        if verbose:
            print(f"  [Phase 2] Unfreezing last LSTM layer...")
        if self.is_v2:
            # BiLSTM: unfreeze layer 1 (last layer)
            self._unfreeze_layer("lstm.weight_ih_l1")
            self._unfreeze_layer("lstm.weight_hh_l1")
            self._unfreeze_layer("lstm.bias_ih_l1")
            self._unfreeze_layer("lstm.bias_hh_l1")
        else:
            self._unfreeze_layer("lstm.weight_hh_l0")
            self._unfreeze_layer("lstm.bias_hh_l0")
        optimizer = torch.optim.Adam(self._get_trainable_params(), lr=0.001)
        self._train_phase(dl, optimizer, criterion, epochs=15, phase_name="Phase 2")
        rmse_p2 = self._evaluate(X_val, y_val)
        self.phase_history.append(("Phase 2: Output + LSTM-last", rmse_p2))
        if verbose:
            print(f"    RMSE after Phase 2: {rmse_p2:.4f}")

        # ── Phase 3: All layers with tiny LR (5 epochs, LR=0.0001) ──
        if verbose:
            print(f"  [Phase 3] Full model fine-tuning...")
        self._unfreeze_all()
        optimizer = torch.optim.Adam(self._get_trainable_params(), lr=0.0001)
        self._train_phase(dl, optimizer, criterion, epochs=5, phase_name="Phase 3")
        rmse_p3 = self._evaluate(X_val, y_val)
        self.phase_history.append(("Phase 3: Full fine-tune", rmse_p3))
        if verbose:
            print(f"    RMSE after Phase 3: {rmse_p3:.4f}")

        return rmse_p3

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Adapted model saved → {path}")


# ── Meta-Learning Adapter (Phase 3 MAML) ───────────────────────────────────
class MetaLearningAdapter(ProgressiveMachineAdapter):
    """
    Phase 3: Model-Agnostic Meta-Learning (MAML) Adapter for extreme few-shot
    adaptation (e.g., < 10 samples).
    
    Uses First-Order MAML (FOMAML) to meta-train the model weights such that
    a few gradient steps on a new machine's data leads to rapid convergence.
    """
    
    def meta_train(self, tasks, meta_lr=0.001, inner_lr=0.01, inner_steps=1, meta_epochs=10):
        """
        tasks: List of (X_support, y_support, X_query, y_query) for different machines
        """
        optimizer = torch.optim.Adam(self._get_trainable_params(), lr=meta_lr)
        criterion = nn.MSELoss()
        
        for epoch in range(meta_epochs):
            meta_loss = 0.0
            optimizer.zero_grad()
            
            for X_s, y_s, X_q, y_q in tasks:
                # 1. Clone model state
                original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                # 2. Inner loop on support set
                inner_opt = torch.optim.SGD(self._get_trainable_params(), lr=inner_lr)
                if self.is_v2:
                    xs = torch.FloatTensor(X_s)
                else:
                    xs = torch.FloatTensor(X_s).unsqueeze(-1)
                ys = torch.FloatTensor(y_s)
                
                for _ in range(inner_steps):
                    inner_opt.zero_grad()
                    loss = criterion(self.model(xs), ys)
                    loss.backward()
                    inner_opt.step()
                
                # 3. Outer loop loss on query set
                if self.is_v2:
                    xq = torch.FloatTensor(X_q)
                else:
                    xq = torch.FloatTensor(X_q).unsqueeze(-1)
                yq = torch.FloatTensor(y_q)
                
                query_loss = criterion(self.model(xq), yq)
                query_loss.backward()
                meta_loss += query_loss.item()
                
                # Restore original state, but keep the accumulated gradients
                self.model.load_state_dict(original_state)
            
            # 4. Meta-update
            optimizer.step()
            print(f"  [MAML] Epoch {epoch+1}/{meta_epochs}, Meta Loss: {meta_loss/len(tasks):.4f}")


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