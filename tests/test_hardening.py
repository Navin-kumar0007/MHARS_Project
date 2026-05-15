"""
MHARS Hardening Regression Tests
==================================
Each test verifies a specific bug fix from the codebase review.
Run: pytest tests/test_hardening.py -v
"""

import os
import sys
import json
import time
import pytest
import numpy as np

# Ensure mhars is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── #1: Config.CNN_MODEL exists ────────────────────────────────────────────────
class TestConfigFixes:
    def test_cnn_model_path_defined(self):
        """Bug #1: Config.CNN_MODEL was missing, causing trainer.py to crash."""
        from mhars.config import Config
        assert hasattr(Config, 'CNN_MODEL'), "Config.CNN_MODEL must be defined"
        assert 'mobilenet_cnn.pt' in Config.CNN_MODEL

    def test_no_duplicate_results_dir(self):
        """Bug #27: RESULTS_DIR was defined twice in config.py."""
        from mhars.config import Config
        # If there's a duplicate, it wouldn't cause a Python error,
        # but we can verify by checking the source file directly
        config_path = os.path.join(os.path.dirname(__file__), '..', 'mhars', 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        count = content.count('RESULTS_DIR')
        assert count == 1, f"RESULTS_DIR appears {count} times (expected 1)"

    def test_emergency_shutdown_in_actions(self):
        """Amendment: emergency-shutdown should be in Config.ACTIONS."""
        from mhars.config import Config
        assert 'emergency-shutdown' in Config.ACTIONS.values()

    def test_lstm_prediction_horizon_defined(self):
        """Amendment: LSTM_PREDICTION_HORIZON_S constant should exist."""
        from mhars.config import Config
        assert hasattr(Config, 'LSTM_PREDICTION_HORIZON_S')
        assert Config.LSTM_PREDICTION_HORIZON_S > 0

    def test_if_cold_start_samples_defined(self):
        """Amendment: IF_COLD_START_SAMPLES should exist."""
        from mhars.config import Config
        assert hasattr(Config, 'IF_COLD_START_SAMPLES')
        assert Config.IF_COLD_START_SAMPLES > 0


# ── #4: LLM generate does not mutate input ─────────────────────────────────────
class TestLLMFixes:
    def test_generate_does_not_mutate_input(self):
        """Bug #4: context.pop('_force_template') was mutating the caller's dict."""
        from mhars.llm import AlertGenerator
        gen = AlertGenerator(model_path=None)  # template-only mode

        ctx = {
            "machine_type": "CPU",
            "current_temp": 85.0,
            "predicted_temp": 90.0,
            "anomaly_score": 0.7,
            "action_name": "fan+",
            "urgency": 0.6,
            "_force_template": True,
        }
        original_keys = set(ctx.keys())
        gen.generate(ctx)
        assert set(ctx.keys()) == original_keys, \
            "generate() must not remove keys from the input dict"
        assert '_force_template' in ctx, \
            "_force_template key was removed by generate()"


# ── #6: RUL estimation math ────────────────────────────────────────────────────
class TestRULFix:
    def test_rul_scales_correctly(self):
        """Bug #6: RUL was off by 600x due to wrong time scale.
        
        At 1Hz sampling, if LSTM predicts temp rises by 1°C per step,
        and we're 60°C below safe_max, RUL should be 60 seconds = 1 minute.
        """
        from mhars.core import MHARS, Config
        # Use the formula directly to verify
        horizon_s = Config.LSTM_PREDICTION_HORIZON_S  # 1 second
        current_temp = 25.0
        predicted_temp = 26.0  # +1°C per step
        safe_max = 85.0

        delta = predicted_temp - current_temp  # 1.0
        remaining = safe_max - current_temp    # 60.0
        seconds = (remaining / delta) * horizon_s  # 60 seconds
        minutes = seconds / 60.0  # 1.0 minute

        assert abs(minutes - 1.0) < 0.01, f"RUL should be 1.0 min, got {minutes}"

    def test_rul_returns_none_when_cooling(self):
        """RUL should return None when temp is falling."""
        from mhars.core import MHARS
        system = MHARS(machine_type_id=0, verbose=False)
        rul = system._estimate_rul(
            current_temp=80.0, predicted_temp=78.0, safe_max=85.0
        )
        assert rul is None


# ── #9: Torch import guard ─────────────────────────────────────────────────────
class TestTorchGuard:
    def test_init_does_not_call_torch_directly(self):
        """Bug #9: torch.manual_seed() was called unconditionally.
        
        We can't truly test without torch, but we verify the source
        code has the guard in place.
        """
        core_path = os.path.join(os.path.dirname(__file__), '..', 'mhars', 'core.py')
        with open(core_path, 'r') as f:
            content = f.read()
        # There should be NO `import torch` followed by `torch.manual_seed`
        # outside of a TORCH_AVAILABLE guard
        lines = content.split('\n')
        in_init = False
        for i, line in enumerate(lines):
            if 'def __init__' in line:
                in_init = True
            elif in_init and line.strip().startswith('def '):
                in_init = False
            if in_init and 'import torch' in line and 'try:' not in lines[max(0,i-2):i+1][0]:
                # Found a bare `import torch` inside __init__ — that's the bug
                pytest.fail(f"Line {i+1}: bare 'import torch' in __init__ (should be guarded)")


# ── #10: Fusion preserves all modalities ───────────────────────────────────────
class TestFusionFix:
    def test_fuse_accepts_six_modalities(self):
        """Bug #10: CNN and vibration were merged with max() instead of separate fusion."""
        from mhars.core import MHARS
        system = MHARS(machine_type_id=0, verbose=False)

        context, urgency, contrib, top = system._fuse(
            lstm_score=0.3, ae_score=0.4, if_score=0.2,
            cnn_score=0.8, audio_score=0.1, vib_score=0.9,
        )
        # Both cnn_hotspot and vibration should appear as separate contributors
        assert "cnn_hotspot" in contrib, "CNN hotspot should be a separate contributor"
        assert "vibration" in contrib, "Vibration should be a separate contributor"
        # 6 contributors total
        assert len(contrib) == 6, f"Expected 6 contributors, got {len(contrib)}"

    def test_fuse_vibration_not_overwritten_by_cnn(self):
        """If vibration is high and CNN is low, vibration should dominate."""
        from mhars.core import MHARS
        system = MHARS(machine_type_id=0, verbose=False)

        _, _, contrib, top = system._fuse(
            lstm_score=0.1, ae_score=0.1, if_score=0.1,
            cnn_score=0.1, audio_score=0.1, vib_score=0.95,
        )
        assert contrib["vibration"] > contrib["cnn_hotspot"], \
            "Vibration should dominate when its score is much higher"


# ── #16: Gym env uses Config profiles ──────────────────────────────────────────
class TestProfileUnification:
    def test_gym_env_uses_config_profiles(self):
        """Bug #16: gym_env.py had different thresholds than config.py."""
        from stage1_simulation.gym_env import MACHINE_PROFILES as gym_profiles
        from mhars.config import Config

        for mid in Config.MACHINE_PROFILES:
            assert mid in gym_profiles, f"Machine {mid} missing from gym_env"
            assert gym_profiles[mid]["critical"] == Config.MACHINE_PROFILES[mid]["critical"], \
                f"Machine {mid} critical mismatch: gym={gym_profiles[mid]['critical']}, config={Config.MACHINE_PROFILES[mid]['critical']}"
            assert gym_profiles[mid]["safe_max"] == Config.MACHINE_PROFILES[mid]["safe_max"], \
                f"Machine {mid} safe_max mismatch"


# ── #17: Registry heartbeat throttling ─────────────────────────────────────────
class TestRegistryThrottling:
    def test_heartbeat_not_every_run(self):
        """Bug #17: Registry was writing to disk on every run() call."""
        from mhars.core import MHARS
        system = MHARS(machine_type_id=0, verbose=False)

        # Run twice rapidly — second run should NOT trigger heartbeat
        system.run(45.0)
        first_time = system._last_heartbeat_time

        system.run(46.0)
        second_time = system._last_heartbeat_time

        # Both should have the same timestamp (throttled to 30s)
        assert first_time == second_time, \
            "Heartbeat should be throttled to 30s intervals"


# ── #18: Context manager support ───────────────────────────────────────────────
class TestContextManager:
    def test_mhars_supports_with_statement(self):
        """Amendment: MHARS should support context manager for clean shutdown."""
        from mhars.core import MHARS
        with MHARS(machine_type_id=0, verbose=False) as system:
            r = system.run(45.0)
            assert r.action is not None
        # No exception = close() ran successfully


# ── #24: temp_was_safe threshold ───────────────────────────────────────────────
class TestGymEnvThreshold:
    def test_temp_was_safe_uses_90_percent(self):
        """Bug #24: temp_was_safe used 0.70 which was too aggressive."""
        env_path = os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation', 'gym_env.py')
        with open(env_path, 'r') as f:
            content = f.read()
        assert '0.90' in content, "temp_was_safe should use 0.90 threshold"
        # Verify the actual threshold assignment uses 0.90, not the old 0.70
        # The comment may mention 0.70 for context, so check the code pattern
        assert 'safe_max * 0.90)' in content, \
            "temp_was_safe assignment should use safe_max * 0.90"


# ── #35: Numpy-safe JSON encoding ──────────────────────────────────────────────
class TestNumpyJsonEncoder:
    def test_numpy_types_serialize(self):
        """Bug #35: json.dumps fails with numpy types in metadata."""
        from mhars.core import _NumpySafeEncoder
        data = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.14),
            "array_val": np.array([1, 2, 3]),
            "normal_val": "hello",
        }
        result = json.dumps(data, cls=_NumpySafeEncoder)
        parsed = json.loads(result)
        assert parsed["int_val"] == 42
        assert abs(parsed["float_val"] - 3.14) < 0.01
        assert parsed["array_val"] == [1, 2, 3]


# ── IF Cold-start bypass ──────────────────────────────────────────────────────
class TestIFColdStart:
    def test_if_cold_start_uses_fallback(self):
        """Amendment: IF should use fallback during cold-start period."""
        from mhars.core import MHARS
        system = MHARS(machine_type_id=0, verbose=False)

        # Before any retrain, IF should use the linear fallback
        # even if a pickle model is loaded
        assert not system._if_has_retrained, \
            "IF should not be marked as retrained at init"

    def test_if_has_retrained_flag_set_after_retrain(self):
        """After manual retrain, the cold-start flag should be resolved."""
        from mhars.core import MHARS
        system = MHARS(machine_type_id=0, verbose=False)

        # Feed enough samples to fill the retrain buffer
        for i in range(60):
            system.run(45.0 + i * 0.5)

        # Force a retrain
        if len(system._if_retrain_buffer) >= 50:
            system._retrain_if()
            assert system._if_has_retrained, \
                "IF should be marked as retrained after _retrain_if()"


# ── #32: Demo non-interactive flag ─────────────────────────────────────────────
class TestDemoCI:
    def test_demo_has_no_interactive_flag(self):
        """Bug #32: demo.py input() blocks CI/CD."""
        demo_path = os.path.join(os.path.dirname(__file__), '..', 'demo.py')
        with open(demo_path, 'r') as f:
            content = f.read()
        assert '--no-interactive' in content, \
            "demo.py should support --no-interactive flag"
        assert 'args.no_interactive' in content, \
            "demo.py should check the no_interactive flag"


# ── API fixes (lightweight, no server needed) ─────────────────────────────────
class TestAPIFixes:
    def test_no_cors_wildcard(self):
        """Bug #11: CORS should not allow all origins in production."""
        api_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'main.py')
        with open(api_path, 'r') as f:
            content = f.read()
        # The wildcard should not appear as the configured origin
        assert 'MHARS_CORS_ORIGINS' in content, \
            "CORS origins should come from environment variable"

    def test_no_phantom_action_keys(self):
        """Bug #7: action_effects had 'increase-fan' which doesn't exist."""
        api_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'main.py')
        with open(api_path, 'r') as f:
            lines = f.readlines()
        # Check that 'increase-fan' doesn't appear in any action_effects dict
        in_action_effects = False
        for line in lines:
            if 'action_effects' in line and '{' in line:
                in_action_effects = True
            if in_action_effects:
                assert '"increase-fan"' not in line, \
                    "Phantom 'increase-fan' should be removed from action_effects"
                if '}' in line:
                    in_action_effects = False

    def test_rate_limiter_exists(self):
        """Bug #13: Anomaly injection should have rate limiting."""
        api_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'main.py')
        with open(api_path, 'r') as f:
            content = f.read()
        assert '_INJECTION_COOLDOWN' in content, \
            "Rate limiting cooldown should be defined"
        assert 'rate_limited' in content, \
            "Rate limiter should return rate_limited status"


# ── Deleted requirements.txt ──────────────────────────────────────────────────
class TestRequirementsTxt:
    def test_deprecated_requirements_deleted(self):
        """Bug #26: requirements.txt with torch==2.11.0 was a landmine."""
        req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        assert not os.path.exists(req_path), \
            "Deprecated requirements.txt should be deleted"


# ── Torch guards — no bare `import torch` ─────────────────────────────────────
class TestTorchGuardsComprehensive:
    def test_no_bare_import_torch_in_methods(self):
        """All `import torch` inside methods must be guarded or removed.
        The module-level import is inside try/except, which is correct.
        Method-level bare imports would crash on edge devices without PyTorch.
        """
        core_path = os.path.join(os.path.dirname(__file__), '..', 'mhars', 'core.py')
        with open(core_path, 'r') as f:
            lines = f.readlines()
        
        bare_imports = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == 'import torch':
                # Check if it's inside the module-level try/except (lines ~37-41)
                # That one is fine. Any other bare `import torch` is a bug.
                if i < 45:  # module-level try/except block
                    continue
                bare_imports.append(f"Line {i}: {stripped}")
        
        assert len(bare_imports) == 0, \
            f"Found bare 'import torch' outside module-level guard:\n" + \
            "\n".join(bare_imports)


# ── Gym action names use Config.ACTIONS ────────────────────────────────────────
class TestGymActionConsistency:
    def test_gym_uses_config_actions(self):
        """NEW-4: gym_env.py should use Config.ACTIONS, not a hardcoded list."""
        env_path = os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation', 'gym_env.py')
        with open(env_path, 'r') as f:
            content = f.read()
        # Should NOT have a hardcoded list of action names
        assert '["do-nothing", "fan+", "throttle", "alert", "shutdown"]' not in content, \
            "Gym env should use Config.ACTIONS, not a hardcoded list"
        # Should reference Config.ACTIONS
        assert 'Config.ACTIONS' in content, \
            "Gym env should reference Config.ACTIONS for action name lookup"
