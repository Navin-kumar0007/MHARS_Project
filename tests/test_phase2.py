"""
MHARS — Phase 2 Tests
======================
Tests for all Phase 2 enhancements:
1. Learned Attention Fusion
2. RUL training pipeline
3. Progressive Machine Adapter
4. ThermalEnvV2
5. CPU/Server bypass removal (anomaly damping)
6. Config additions
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation'))

from mhars.config import Config


# ── Component 1: Learned Attention Fusion ──────────────────────────────────────

class TestLearnedAttentionFusion:
    """Tests for the neural self-attention fusion module."""

    @pytest.fixture
    def fusion_model(self):
        torch = pytest.importorskip("torch")
        from mhars.learned_fusion import LearnedAttentionFusion
        model = LearnedAttentionFusion(n_modalities=6, d_model=32, n_heads=4)
        return model

    def test_forward_shape(self, fusion_model):
        """Forward pass returns correct shapes."""
        torch = pytest.importorskip("torch")
        scores = torch.rand(4, 6)  # batch of 4, 6 modalities
        context, attn_weights = fusion_model(scores)
        assert context.shape == (4,), f"Expected (4,), got {context.shape}"
        assert attn_weights.shape[0] == 4, f"Batch dim mismatch"
        assert attn_weights.shape[1] == 6, f"Expected 6 modalities in attention"
        assert attn_weights.shape[2] == 6, f"Expected 6×6 attention map"

    def test_context_range(self, fusion_model):
        """Context score should be in [0, 1] due to Sigmoid."""
        torch = pytest.importorskip("torch")
        scores = torch.rand(10, 6)
        context, _ = fusion_model(scores)
        assert (context >= 0).all() and (context <= 1).all(), \
            f"Context should be in [0,1], got range [{context.min():.3f}, {context.max():.3f}]"

    def test_fuse_with_xai(self, fusion_model):
        """XAI convenience method returns correct format."""
        scores = np.array([0.5, 0.3, 0.2, 0.8, 0.1, 0.4], dtype=np.float32)
        context, contributions, top_contributor = fusion_model.fuse_with_xai(scores)

        assert isinstance(context, float)
        assert 0.0 <= context <= 1.0
        assert isinstance(contributions, dict)
        assert len(contributions) == 6
        assert all(isinstance(v, int) for v in contributions.values())
        assert isinstance(top_contributor, str)

    def test_attention_weight_sum(self, fusion_model):
        """Attention weights should sum to ~1 per row (softmax)."""
        torch = pytest.importorskip("torch")
        scores = torch.rand(1, 6)
        _, attn_weights = fusion_model(scores)
        row_sums = attn_weights[0].sum(dim=-1)
        for i, s in enumerate(row_sums):
            assert abs(s.item() - 1.0) < 0.15, \
                f"Attention row {i} sums to {s.item():.4f}, expected ~1.0"


class TestFusionTrainingData:
    """Tests for synthetic fusion training data generation."""

    def test_data_shape(self):
        from mhars.learned_fusion import generate_fusion_training_data
        X, y = generate_fusion_training_data(n_samples=100)
        assert X.shape == (100, 6)
        assert y.shape == (100,)

    def test_data_range(self):
        from mhars.learned_fusion import generate_fusion_training_data
        X, y = generate_fusion_training_data(n_samples=500)
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert y.min() >= 0.0
        assert y.max() <= 1.0


# ── Component 2: RUL Training Pipeline ────────────────────────────────────────

class TestRULPipeline:
    """Tests for the RUL training pipeline components."""

    def test_make_rul_windows(self):
        """RUL windows are correctly created with piece-wise cap."""
        from stage1_simulation.load_cmapss import (
            load_cmapss, preprocess_multivariate, make_rul_windows
        )
        df = load_cmapss()
        df = preprocess_multivariate(df)
        X, y_rul, unit_ids = make_rul_windows(df, window=12, rul_cap=125)

        assert X.ndim == 3, f"Expected 3D windows, got {X.ndim}D"
        assert X.shape[1] == 12, f"Expected window=12, got {X.shape[1]}"
        assert X.shape[2] == 5, f"Expected 5 sensors, got {X.shape[2]}"
        assert y_rul.max() <= 125, f"RUL cap violated: max={y_rul.max()}"
        assert y_rul.min() >= 0, f"Negative RUL: min={y_rul.min()}"

    def test_nasa_scoring(self):
        """NASA scoring function penalizes late predictions more."""
        from stage2_ml.rul_trainer import nasa_scoring_function

        y_true = np.array([50.0, 50.0, 50.0])
        # Early prediction (predicted 40, actual 50 → d = -10)
        y_pred_early = np.array([40.0, 40.0, 40.0])
        score_early = nasa_scoring_function(y_true, y_pred_early)

        # Late prediction (predicted 60, actual 50 → d = +10)
        y_pred_late = np.array([60.0, 60.0, 60.0])
        score_late = nasa_scoring_function(y_true, y_pred_late)

        assert score_late > score_early, \
            f"Late predictions should have higher penalty. Early={score_early:.1f}, Late={score_late:.1f}"

    def test_rul_predictor_forward(self):
        """RULPredictor forward pass produces scalar output."""
        torch = pytest.importorskip("torch")
        from mhars.models import RULPredictor

        model = RULPredictor(input_size=5, hidden_size=64, num_layers=2)
        x = torch.rand(4, 12, 5)  # batch=4, window=12, sensors=5
        out = model(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"


# ── Component 3: Progressive Machine Adapter ──────────────────────────────────

class TestProgressiveMachineAdapter:
    """Tests for the 3-phase progressive unfreezing adapter."""

    def test_adapter_v1_backward_compat(self):
        """MachineAdapter alias points to MachineAdapterV1."""
        from stage5_adapter.machine_adapter import MachineAdapter, MachineAdapterV1
        assert MachineAdapter is MachineAdapterV1

    def test_progressive_class_exists(self):
        """ProgressiveMachineAdapter is importable."""
        from stage5_adapter.machine_adapter import ProgressiveMachineAdapter
        assert ProgressiveMachineAdapter is not None


# ── Component 4: ThermalEnvV2 ─────────────────────────────────────────────────

class TestThermalEnvV2:
    """Tests for the enhanced gym environment."""

    def test_v2_obs_shape(self):
        """V2 observation space is 12-dimensional."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        obs, info = env.reset(seed=42)
        assert obs.shape == (12,), f"Expected 12-dim obs, got {obs.shape}"

    def test_v2_action_space(self):
        """V2 action space is continuous Box(2)."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0)
        assert env.action_space.shape == (2,), f"Expected (2,) action, got {env.action_space.shape}"

    def test_v2_episode_runs(self):
        """V2 environment runs a full episode without errors."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 600:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == (12,)
            done = term or trunc
            steps += 1
        assert steps > 0, "Episode completed 0 steps"

    def test_v2_variable_episode_length(self):
        """Variable episodes produce different lengths."""
        from stage1_simulation.gym_env import ThermalEnvV2
        lengths = set()
        for seed in range(10):
            env = ThermalEnvV2(machine_type_id=0, variable_episodes=True)
            env.reset(seed=seed)
            lengths.add(env.max_steps)
        assert len(lengths) > 1, f"Expected variable lengths, got {lengths}"

    def test_v2_degradation(self):
        """Degradation factor increases over time."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        env.reset(seed=42)
        for _ in range(100):
            env.step(env.action_space.sample())
        assert env.degradation_factor > 0, "Degradation should increase over time"


# ── Component 6: Bypass Removal ───────────────────────────────────────────────

class TestAnomalyDamping:
    """Tests for per-machine anomaly damping (replaces blanket bypass)."""

    def test_damping_factors_exist(self):
        """ANOMALY_DAMPING_FACTORS should be in Config."""
        assert hasattr(Config, 'ANOMALY_DAMPING_FACTORS')
        assert len(Config.ANOMALY_DAMPING_FACTORS) == 4

    def test_cpu_damping_less_than_1(self):
        """CPU (id=0) should have damping < 1.0."""
        assert Config.ANOMALY_DAMPING_FACTORS[0] < 1.0

    def test_motor_damping_full(self):
        """Motor (id=1) should have full damping = 1.0."""
        assert Config.ANOMALY_DAMPING_FACTORS[1] == 1.0

    def test_server_damping_less_than_1(self):
        """Server (id=2) should have damping < 1.0."""
        assert Config.ANOMALY_DAMPING_FACTORS[2] < 1.0

    def test_engine_damping_full(self):
        """Engine (id=3) should have full damping = 1.0."""
        assert Config.ANOMALY_DAMPING_FACTORS[3] == 1.0


# ── Config Additions ──────────────────────────────────────────────────────────

class TestPhase2Config:
    """Tests for Phase 2 config additions."""

    def test_fusion_config(self):
        assert Config.FUSION_D_MODEL == 32
        assert Config.FUSION_N_HEADS == 4
        assert Config.FUSION_N_MODALITIES == 6

    def test_rul_config(self):
        assert Config.RUL_MAX_CYCLES == 125
        assert hasattr(Config, 'RUL_MODEL_V2')

    def test_sac_config(self):
        assert hasattr(Config, 'SAC_MODEL')

    def test_env_v2_config(self):
        assert Config.ENV_V2_OBS_DIM == 12

    def test_ppo_reward_v2(self):
        R = Config.PPO_REWARD_V2
        assert "energy_efficiency_bonus" in R
        assert "proactive_cooling_bonus" in R
        assert "rul_penalty_scale" in R
        assert "smoothness_reward" in R
        # V2 should inherit all V1 keys
        for key in Config.PPO_REWARD:
            assert key in R, f"Missing V1 key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
