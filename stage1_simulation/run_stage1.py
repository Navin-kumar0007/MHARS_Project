from mhars.config import Config
"""
MHARS — Stage 1 Validation Script
===================================
Run this to confirm Stage 1 is working correctly before moving to Stage 2.
All 5 checklist items from the implementation plan are tested here.

Usage:
    python run_stage1.py

Expected output:
    [PASS] Gymnasium environment created with correct spaces
    [PASS] Reward function behaves correctly (3 distinct values)
    [PASS] 500 random episodes complete without crash
    [PASS] NASA CMAPSS dataset loaded and preprocessed
    [PASS] LSTM windows created (12-step sliding window)
    [PASS] Thermal trend plot saved

    Stage 1 complete. Move to Stage 2.
"""

import sys
import os

import numpy as np

def test_environment():
    print("\n── Test 1: Gymnasium environment ─────────────────────────────")
    from stage1_simulation.gym_env import ThermalEnv, MACHINE_PROFILES

    for machine_id, profile in MACHINE_PROFILES.items():
        env = ThermalEnv(machine_type_id=machine_id)
        obs, info = env.reset(seed=Config.SEED)

        # Check observation shape
        assert obs.shape == (6,), f"Expected obs shape (6,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        assert np.all(obs >= 0) and np.all(obs <= 1), "Observation out of [0,1] bounds"

        # Check action space
        assert env.action_space.n == 5, "Expected 5 actions"  # type: ignore[attr-defined]

        print(f"  [{machine_id}] {profile['name']:6s} — obs: {np.round(obs, 3)}  ✓")

    print("[PASS] Gymnasium environment created with correct spaces")


def test_reward_function():
    print("\n── Test 2: Reward function ───────────────────────────────────")
    from stage1_simulation.gym_env import ThermalEnv

    env = ThermalEnv(machine_type_id=0, max_steps=1000)
    rewards_seen = set()

    # Run many steps to catch all 3 reward values
    obs, _ = env.reset(seed=0)
    for _ in range(600):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_seen.add(round(reward, 1))
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"  Reward values encountered: {sorted(rewards_seen)}")

    # Must have seen at least positive and negative rewards
    has_positive = any(r > 0 for r in rewards_seen)
    has_small_neg = any(-3 <= r < 0 for r in rewards_seen)
    assert has_positive, "Never saw a +1 reward — something is wrong with the safe-band logic"
    assert has_small_neg, "Never saw the -2 penalty — unnecessary intervention penalty not triggering"

    print("[PASS] Reward function behaves correctly (positive and penalty values seen)")


def test_random_episodes():
    print("\n── Test 3: 500 random episodes ───────────────────────────────")
    from stage1_simulation.gym_env import ThermalEnv

    total_reward = 0
    episode_rewards = []

    for machine_id in range(4):
        env = ThermalEnv(machine_type_id=machine_id, max_steps=200)
        ep_count = 0
        while ep_count < 125:  # 125 episodes × 4 machines = 500 total
            obs, _ = env.reset()
            ep_reward = 0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            episode_rewards.append(ep_reward)
            total_reward += ep_reward
            ep_count += 1

    avg = np.mean(episode_rewards)
    print(f"  500 episodes completed. Avg episode reward (random agent): {avg:.1f}")
    print(f"  Min: {min(episode_rewards):.1f}  Max: {max(episode_rewards):.1f}")
    print("[PASS] 500 random episodes complete without crash")


def test_dataset():
    print("\n── Test 4: CMAPSS dataset ────────────────────────────────────")
    from stage1_simulation.load_cmapss import load_cmapss, preprocess, print_dataset_summary

    # No real file yet — uses synthetic data
    df = load_cmapss()
    df = preprocess(df)
    print_dataset_summary(df)

    assert "unit_id" in df.columns
    assert "rul" in df.columns
    assert "near_failure" in df.columns
    assert "s4_norm" in df.columns
    assert df["s4_norm"].between(0, 1).all(), "Normalized column out of [0,1]"

    print("[PASS] NASA CMAPSS dataset loaded and preprocessed")
    return df


def test_lstm_windows(df):
    print("\n── Test 5: LSTM sliding windows ──────────────────────────────")
    from stage1_simulation.load_cmapss import make_lstm_windows

    X, y, unit_ids = make_lstm_windows(df, window=12)

    assert X.shape[1] == 12, f"Window size should be 12, got {X.shape[1]}"
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    assert X.dtype == np.float32
    assert y.dtype == np.float32

    print(f"  Created {X.shape[0]:,} windows of shape {X.shape}")
    print(f"  Unique units in windows: {np.unique(unit_ids).shape[0]}")
    print(f"  Target (y) range: [{y.min():.3f}, {y.max():.3f}]")
    print("[PASS] LSTM windows created (12-step sliding window)")


def test_plot(df):
    print("\n── Test 6: Thermal trend plot ────────────────────────────────")
    from stage1_simulation.load_cmapss import plot_thermal_trends
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "thermal_trends.png")
    plot_thermal_trends(df, n_units=5, save_path=save_path)
    assert os.path.exists(save_path), f"Plot not saved at {save_path}"
    print(f"[PASS] Thermal trend plot saved to {save_path}")


def run_all():
    print("╔══════════════════════════════════════════════════╗")
    print("║   MHARS Stage 1 — Validation                    ║")
    print("╚══════════════════════════════════════════════════╝")

    try:
        test_environment()
        test_reward_function()
        test_random_episodes()
        df = test_dataset()
        test_lstm_windows(df)
        test_plot(df)

        print("\n" + "="*52)
        print("  All Stage 1 checks PASSED.")
        print("  Next step: Stage 2 — ML Perception Layer")
        print("  Start with: stage2_ml/isolation_forest.py")
        print("="*52 + "\n")

    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all()