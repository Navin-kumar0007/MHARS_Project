"""
MHARS — Phase 2: SAC Agent for Continuous Fan Control
======================================================
Soft Actor-Critic agent trained on ThermalEnvV2 for continuous
fan speed control with discrete emergency actions.

SAC is better suited than PPO for continuous actions because:
- Maximum entropy RL → more robust exploration
- Off-policy → more sample efficient
- Naturally handles continuous action spaces

Usage:
    python -m stage3_ai.sac_agent

Requires:
    pip install stable-baselines3[extra]
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation'))


def train_sac(
    machine_type_id: int = 0,
    total_timesteps: int = 500_000,
    save_path: str = None,
    verbose: bool = True,
):
    """
    Train a SAC agent on ThermalEnvV2 for continuous fan speed control.
    
    Returns the trained model and training metrics.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    from gym_env import ThermalEnvV2

    from mhars.config import Config
    if save_path is None:
        save_path = Config.SAC_MODEL

    if verbose:
        print(f"── SAC Training on ThermalEnvV2 ────────────────────────────────")
        print(f"  Machine type: {machine_type_id}")
        print(f"  Total timesteps: {total_timesteps:,}")

    # Training environment
    train_env = Monitor(ThermalEnvV2(machine_type_id=machine_type_id))
    eval_env = Monitor(ThermalEnvV2(machine_type_id=machine_type_id,
                                      variable_episodes=False))

    # SAC hyperparameters
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,           # soft target update coefficient
        gamma=0.99,          # discount factor
        ent_coef="auto",     # auto-tune entropy coefficient
        target_entropy="auto",
        train_freq=1,
        gradient_steps=1,
        verbose=1 if verbose else 0,
        seed=Config.SEED,
    )

    # Evaluation callback
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path=os.path.join(Config.RESULTS_DIR, "sac_logs"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    # Save final model
    model.save(save_path.replace(".zip", ""))
    if verbose:
        print(f"\n  ✓  SAC model saved → {save_path}")

    return model


def evaluate_sac(
    model_or_path,
    machine_type_id: int = 0,
    n_episodes: int = 50,
    verbose: bool = True,
):
    """
    Evaluate a trained SAC agent on ThermalEnvV2.
    
    Returns dict with reward, energy, and safety metrics.
    """
    from stable_baselines3 import SAC
    from gym_env import ThermalEnvV2

    if isinstance(model_or_path, str):
        model = SAC.load(model_or_path.replace(".zip", ""))
    else:
        model = model_or_path

    rewards, max_temps, avg_fans, violations = [], [], [], []

    for ep in range(n_episodes):
        env = ThermalEnvV2(machine_type_id=machine_type_id, variable_episodes=False)
        obs, _ = env.reset(seed=ep)
        ep_reward, done = 0.0, False
        ep_max_temp = 0.0
        ep_fan_sum = 0.0
        ep_steps = 0
        ep_violations = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            ep_max_temp = max(ep_max_temp, info.get("temp", 0))
            ep_fan_sum += info.get("fan_speed", 0.5)
            ep_steps += 1
            if info.get("temp", 0) > env.profile["safe_max"]:
                ep_violations += 1
            done = term or trunc

        rewards.append(ep_reward)
        max_temps.append(ep_max_temp)
        avg_fans.append(ep_fan_sum / max(ep_steps, 1))
        violations.append(ep_violations)

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_max_temp": float(np.mean(max_temps)),
        "mean_avg_fan": float(np.mean(avg_fans)),
        "mean_violations": float(np.mean(violations)),
        "n_episodes": n_episodes,
    }

    if verbose:
        print(f"\n── SAC Evaluation ({n_episodes} episodes) ──")
        print(f"  Mean reward:    {metrics['mean_reward']:.1f} ± {metrics['std_reward']:.1f}")
        print(f"  Mean max temp:  {metrics['mean_max_temp']:.1f}°C")
        print(f"  Mean avg fan:   {metrics['mean_avg_fan']:.2f}")
        print(f"  Mean violations: {metrics['mean_violations']:.1f}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SAC agent for MHARS")
    parser.add_argument("--machine", type=int, default=0, help="Machine type ID (0-3)")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    parser.add_argument("--evaluate-only", type=str, default=None, help="Path to evaluate")
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_sac(args.evaluate_only, machine_type_id=args.machine)
    else:
        model = train_sac(
            machine_type_id=args.machine,
            total_timesteps=args.timesteps,
        )
        evaluate_sac(model, machine_type_id=args.machine)
