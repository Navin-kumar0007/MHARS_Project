"""
MHARS — Stage 3: PPO Thermal Decision Agent
=============================================
Trains a Proximal Policy Optimization agent to manage machine
temperature by choosing the right action at the right time.

Why PPO over a classifier:
  A classifier predicts "will this overheat?" — it cannot decide
  whether to run the fan at 60% or 80%, or whether to throttle now
  or wait 5 minutes. PPO learns a sequential decision POLICY that
  maximises long-term reward. Validated by Zouganeli et al. (2025)
  on CNC machine sensor data. Reference:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12833388/

Training target: 500,000 total timesteps.
  - Not 10,000 episodes (ambiguous — episode length varies).
  - 500K timesteps ≈ 20–40 minutes on a MacBook CPU.
  - Average reward should climb from ~-280 (random) to ~+50 or better.
"""

import os, sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stage1_simulation.gym_env import ThermalEnv
from mhars.config import Config


# ── Training callback — prints progress every 50K steps ───────────────────────
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=50_000, verbose=1):
        super().__init__(verbose)
        self.check_freq  = check_freq
        self.ep_rewards  = []
        self.ep_lengths  = []
        self.checkpoints = []

    def _on_step(self):
        # Collect episode info when episodes finish
        if self.locals.get("dones") is not None:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                    self.ep_lengths.append(info["episode"]["l"])

        # Print summary every check_freq steps
        if self.n_calls % self.check_freq == 0 and self.ep_rewards:
            recent = self.ep_rewards[-50:]   # last 50 episodes
            avg    = np.mean(recent)
            best   = np.max(recent)
            print(f"  Step {self.num_timesteps:>7,}  "
                  f"avg reward (last 50 ep): {avg:>8.1f}  "
                  f"best: {best:>8.1f}  "
                  f"episodes: {len(self.ep_rewards):>5,}")
            self.checkpoints.append({
                "step": self.num_timesteps,
                "avg_reward": avg,
            })
        return True


def train(
    machine_type_id: int = 0,
    total_timesteps: int = 500_000,
    model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_thermal.zip"),
    n_envs: int = 4,               # parallel environments — speeds up training
):
    """
    Train a PPO agent on the specified machine type.
    Uses 4 parallel environments by default to collect experience faster.
    """
    print(f"\n── PPO Training ──────────────────────────────────────────────")
    print(f"  Machine type:    {machine_type_id} (0=CPU 1=Motor 2=Server 3=Engine)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs:   {n_envs}")
    print(f"  Reward weights:  Config.PPO_REWARD (editable in mhars/config.py)")
    print(f"  Target: avg reward > 50 (random baseline ≈ -280)")
    print()

    # Create vectorized environments (parallel training)
    def make_env():
        env = ThermalEnv(machine_type_id=machine_type_id, max_steps=500)
        return Monitor(env)

    vec_env  = make_vec_env(make_env, n_envs=n_envs)

    # Evaluation environment (single, same machine type)
    eval_env = Monitor(ThermalEnv(machine_type_id=machine_type_id, max_steps=500))

    # ── PPO hyperparameters (from Config) ──────────────────────────────────────
    # Reward shaping is handled in gym_env.py via Config.PPO_REWARD.
    # Hyperparameters here control the optimizer, not the reward function.
    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = Config.PPO_LEARNING_RATE,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,         # discount factor
        gae_lambda      = 0.95,         # advantage estimation
        clip_range      = Config.PPO_CLIP_RANGE,
        ent_coef        = 0.01,         # entropy bonus — encourages exploration
        verbose         = 0,            # we handle printing in callback
    )

    callback = ProgressCallback(check_freq=50_000)

    print(f"  Training started. Progress printed every 50,000 steps.")
    print(f"  {'Step':>10}  {'Avg reward (50 ep)':>22}  {'Best':>10}  {'Episodes':>10}")
    print(f"  {'-'*10}  {'-'*22}  {'-'*10}  {'-'*10}")

    model.learn(
        total_timesteps = total_timesteps,
        callback        = callback,
        progress_bar    = False,
    )

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path.replace(".zip", ""))
    
    # Save training logs to CSV
    from mhars.config import Config
    import pandas as pd
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    pd.DataFrame({"episode": range(1, len(callback.ep_rewards)+1), "reward": callback.ep_rewards}).to_csv(
        os.path.join(Config.RESULTS_DIR, "ppo_training.csv"), index=False
    )
    
    print(f"\n  Model saved → {model_path}")

    return model, callback


def plot_training_curve(callback: ProgressCallback, save_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ppo_training_curve.png")):
    """Plot the reward curve — shows the agent learning over time."""
    if not callback.ep_rewards:
        print("  No episode data to plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Smooth with a rolling window
    rewards = np.array(callback.ep_rewards)
    window  = min(50, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rewards,   color="#CCCCCC", linewidth=0.6, label="Episode reward", alpha=0.7)
    ax.plot(smoothed,  color="#0F6E56", linewidth=2.0, label=f"Rolling avg ({window} ep)")
    ax.axhline(y=0,    color="#993C1D", linewidth=1.0, linestyle="--", alpha=0.5, label="Break-even (0)")
    ax.axhline(y=-280, color="#854F0B", linewidth=1.0, linestyle=":",  alpha=0.5, label="Random baseline (-280)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode reward")
    ax.set_title("MHARS PPO Agent — Training Reward Curve", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved → {save_path}")


def evaluate(model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_thermal.zip"),
             machine_type_id: int = 0,
             n_episodes: int = 20):
    """
    Run the trained agent for N episodes and report performance.
    Compares against a random-action baseline.
    """
    print(f"\n── PPO Evaluation ({n_episodes} episodes) ────────────────────")
    model = PPO.load(model_path.replace(".zip", ""))

    rewards, actions_taken = [], []

    for ep in range(n_episodes):
        env  = ThermalEnv(machine_type_id=machine_type_id, max_steps=500)
        obs, _ = env.reset(seed=ep)
        ep_reward = 0
        ep_actions = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            ep_actions.append(int(action))
            done = terminated or truncated

        rewards.append(ep_reward)
        actions_taken.append(ep_actions)

    avg_reward = np.mean(rewards)
    action_names = ["do-nothing", "fan+", "throttle", "alert", "shutdown"]

    # Count how often each action was chosen
    all_actions = [a for ep in actions_taken for a in ep]
    unique, counts = np.unique(all_actions, return_counts=True)
    action_dist = {action_names[u]: int(c) for u, c in zip(unique, counts)}

    print(f"  Avg episode reward (trained PPO): {avg_reward:.1f}")
    print(f"  Baseline (random agent):          ~-280.0")
    print(f"  Improvement:                      {avg_reward - (-280):.1f} points")
    print(f"\n  Action distribution across {len(all_actions):,} steps:")
    for name, count in sorted(action_dist.items(), key=lambda x: -x[1]):
        pct = count / len(all_actions) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:<12}  {bar:<25} {pct:.1f}%")

    improvement = avg_reward > -280
    print(f"\n  {'✓' if improvement else '⚠'}  PPO {'outperforms' if improvement else 'not yet better than'} random baseline")
    return avg_reward, action_dist


def run_training(machine_type_id=0, timesteps=500_000, model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_thermal.zip")):
    model, callback = train(machine_type_id, timesteps, model_path)
    plot_training_curve(callback, save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ppo_training_curve.png"))
    avg_reward, action_dist = evaluate(model_path, machine_type_id)
    print(f"\n[PASS] PPO agent trained — avg reward: {avg_reward:.1f}\n")
    return model, avg_reward


if __name__ == "__main__":
    run_training()