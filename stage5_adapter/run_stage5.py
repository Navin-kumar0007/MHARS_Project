"""
MHARS — Stage 5: Machine Adapter Experiment
=============================================
Runs the full cross-machine transfer experiment and produces
the results table for your research paper.

Protocol (from the implementation plan, Section 5.2):
  Step 1 — Machine A (CPU, id=0): already trained in Stage 2 & 3
  Step 2 — Machine B (Engine, id=3): the new unseen machine
  Step 3 — Activate Machine Adapter on 100 samples from Machine B
  Step 4 — Measure LSTM RMSE: Adapter vs Full retrain vs No adaptation
  Step 5 — Measure PPO convergence: Adapted vs Scratch (50 episodes)

Run from the stage5_adapter/ folder:
    python run_stage5.py
"""

import os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage1_simulation.load_cmapss import load_cmapss, preprocess, make_lstm_windows
from stage2_ml.lstm_predictor import ThermalLSTM
from machine_adapter import (
    MachineAdapter, find_most_similar_machine,
    transfer_ppo_policy, MACHINE_PROFILES_FEATURES
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def rmse_on_data(model: ThermalLSTM,
                 X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    x_t = torch.FloatTensor(X).unsqueeze(-1)
    y_t = torch.FloatTensor(y)
    with torch.no_grad():
        preds = model(x_t)
    return float(nn.MSELoss()(preds, y_t).item() ** 0.5)


def retrain_from_scratch(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          epochs: int = 30) -> tuple:
    """Full retrain on all available Machine B data. This is the baseline."""
    model     = ThermalLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    def to_t(a): return torch.FloatTensor(a).unsqueeze(-1)

    dl = DataLoader(
        TensorDataset(to_t(X_train), torch.FloatTensor(y_train)),
        batch_size=256, shuffle=True
    )

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    elapsed = time.time() - t0
    val_rmse = rmse_on_data(model, X_val, y_val)
    return model, val_rmse, elapsed


def generate_machine_b_data(machine_type_id: int = 3, seed: int = 99):
    """
    Generate synthetic data for a genuinely different machine (Engine).
    Unlike CPU data, Engine has:
      - Higher absolute baselines (620–1550 vs 550–1380)
      - Nonlinear degradation curve (exponential, not linear)
      - Shorter lifespans (80–200 vs 150–350 cycles)
      - Higher unit-to-unit variance
      - Different sensor correlation structure

    CRITICAL: We normalize using CPU's min/max ranges, NOT Engine's own range.
    This simulates real deployment: the model was trained on CPU-normalized data,
    so when Engine data arrives, it's normalized the same way. This makes the
    Engine data look genuinely different to the LSTM.
    """
    from stage1_simulation.load_cmapss import THERMAL_SENSORS, PRIMARY_TEMP

    rng = np.random.default_rng(seed)
    rows = []

    for unit_id in range(1, 51):
        max_cycle = rng.integers(80, 200)
        baseline_offset = rng.normal(0, 8)

        for cycle in range(1, max_cycle + 1):
            rul = max_cycle - cycle
            frac = 1 - (rul / max_cycle)
            degradation = (frac ** 1.8) * 55

            row = {
                "unit_id": unit_id, "cycle": cycle, "rul": rul,
                "op1": rng.uniform(0.2, 0.9),
                "op2": rng.uniform(22, 28),
                "op3": rng.integers(70, 110),
            }

            engine_bases  = [570, 620, 1430, 1070, 640]    # shifted up 15-50 from CPU baselines
            engine_spread = [14,  16,  25,   20,   13]     # vs CPU: [10, 12, 20, 15, 10]
            engine_deg    = [0.15, 0.4, 1.3,  1.0,  0.55]  # different degradation profile

            for i, sensor in enumerate(["s2", "s3", "s4", "s7", "s11"]):
                noise = rng.normal(0, engine_spread[i] * 0.4)
                row[sensor] = (engine_bases[i] + baseline_offset
                               + degradation * engine_deg[i] + noise)

            for s in ["s1","s5","s6","s8","s9","s10","s12","s13","s14",
                       "s15","s16","s17","s18","s19","s20","s21"]:
                row[s] = rng.normal(55, 7)

            rows.append(row)

    df = pd.DataFrame(rows)

    # Get CPU data normalization ranges (what the LSTM was trained on)
    cpu_df = load_cmapss()
    cpu_ranges = {}
    for sensor in THERMAL_SENSORS:
        cpu_ranges[sensor] = (cpu_df[sensor].min(), cpu_df[sensor].max())

    # Normalize Engine data using CPU ranges — NOT its own min/max.
    # This makes Engine data sit at different positions in normalized space.
    df_norm = df.copy()
    for sensor in THERMAL_SENSORS:
        cpu_min, cpu_max = cpu_ranges[sensor]
        df_norm[f"{sensor}_norm"] = (df[sensor] - cpu_min) / (cpu_max - cpu_min + 1e-8)
        # Clip to [0, 1.5] — Engine values can exceed CPU's range (that's the point)
        df_norm[f"{sensor}_norm"] = df_norm[f"{sensor}_norm"].clip(0, 1.5)

    df_norm["near_failure"] = (df_norm["rul"] < 30).astype(int)

    X, y, _ = make_lstm_windows(df_norm, window=12)
    print(f"  Engine data: {len(df):,} rows, {len(X):,} windows")
    print(f"  Engine s4 normalized range: [{df_norm['s4_norm'].min():.3f}, {df_norm['s4_norm'].max():.3f}]")
    print(f"  (CPU s4 normalized range was [0.000, 1.000] — difference forces adaptation)")
    return X, y


# ── Main experiment ────────────────────────────────────────────────────────────
def run_lstm_experiment():
    print("\n" + "="*60)
    print("  LSTM Experiment — Machine Adapter vs Full Retrain")
    print("="*60)

    BASE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm.pt")
    ADAPT_SAMPLES = 100

    # Check base model exists
    if not os.path.exists(BASE_MODEL):
        print(f"  ERROR: Base LSTM model not found at {BASE_MODEL}")
        print("  Run Stage 2 first: cd ../stage2_ml && python run_stage2.py")
        return None

    # Generate Machine B data (Engine — never seen during Stage 2 training)
    print(f"\n  Generating Machine B (Engine) data...")
    X_B, y_B = generate_machine_b_data(machine_type_id=3)
    print(f"  Machine B windows: {len(X_B):,}")

    # Split: 100 for adaptation, 200 for validation, rest for full retrain
    X_adapt = X_B[:ADAPT_SAMPLES]
    y_adapt = y_B[:ADAPT_SAMPLES]
    X_val   = X_B[ADAPT_SAMPLES : ADAPT_SAMPLES + 200]
    y_val   = y_B[ADAPT_SAMPLES : ADAPT_SAMPLES + 200]
    X_full  = X_B[:2000]   # all data for full retrain baseline
    y_full  = y_B[:2000]

    # ── Baseline 1: No adaptation (raw Machine A model on Machine B) ──────────
    print(f"\n  [1/3] Testing unadapted model (Machine A on Machine B data)...")
    base_model = ThermalLSTM()
    base_model.load_state_dict(torch.load(BASE_MODEL, map_location="cpu"))
    rmse_unadapted = rmse_on_data(base_model, X_val, y_val)
    print(f"  Unadapted RMSE: {rmse_unadapted:.4f}  (~{rmse_unadapted*85:.1f}°C)")

    # ── Baseline 2: Full retrain from scratch ─────────────────────────────────
    print(f"\n  [2/3] Full retrain from scratch on {len(X_full):,} Machine B samples...")
    t0 = time.time()
    _, rmse_scratch, elapsed_scratch = retrain_from_scratch(
        X_full, y_full, X_val, y_val, epochs=30
    )
    print(f"  Scratch RMSE:   {rmse_scratch:.4f}  (~{rmse_scratch*85:.1f}°C)")
    print(f"  Scratch time:   {elapsed_scratch:.1f}s  ({len(X_full)} samples)")

    # ── Machine Adapter: 100 samples only ─────────────────────────────────────
    print(f"\n  [3/3] Machine Adapter — fine-tuning on {ADAPT_SAMPLES} samples...")

    # Find most similar known machine
    known_ids  = [0, 1, 2]   # CPU, Motor, Server (all trained in Stage 2)
    similar_id, sim_score = find_most_similar_machine(3, known_ids)
    print(f"  Most similar known machine: {similar_id} "
          f"(similarity={sim_score:.3f})")

    adapter = MachineAdapter(BASE_MODEL, similar_machine_id=similar_id)

    t0 = time.time()
    rmse_adapted = adapter.adapt(X_B, y_B,
                                  n_samples=ADAPT_SAMPLES,
                                  epochs=40)
    elapsed_adapted = time.time() - t0
    print(f"  Adapted RMSE:   {rmse_adapted:.4f}  (~{rmse_adapted*85:.1f}°C)")
    print(f"  Adapter time:   {elapsed_adapted:.1f}s  ({ADAPT_SAMPLES} samples)")

    # Save adapted model
    adapter.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm_adapted_engine.pt"))

    # ── Results table ─────────────────────────────────────────────────────────
    improvement = ((rmse_unadapted - rmse_adapted) / rmse_unadapted) * 100

    print("\n" + "─"*60)
    print(f"  {'Method':<30} {'RMSE (norm)':>12} {'RMSE (°C)':>10} {'Time':>8} {'Samples':>8}")
    print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*8} {'─'*8}")
    print(f"  {'No adaptation (raw)':<30} {rmse_unadapted:>12.4f} {rmse_unadapted*85:>9.1f}° {'0s':>8} {'0':>8}")
    print(f"  {'Machine Adapter (100 samples)':<30} {rmse_adapted:>12.4f} {rmse_adapted*85:>9.1f}° {elapsed_adapted:>7.1f}s {ADAPT_SAMPLES:>8}")
    print(f"  {'Full retrain (2000 samples)':<30} {rmse_scratch:>12.4f} {rmse_scratch*85:>9.1f}° {elapsed_scratch:>7.1f}s {'2000':>8}")
    print("─"*60)
    print(f"  Machine Adapter reduces error by {improvement:.1f}% vs unadapted")
    print(f"  Machine Adapter uses {len(X_full) // ADAPT_SAMPLES}x fewer samples than full retrain")

    return {
        "rmse_unadapted": rmse_unadapted,
        "rmse_adapted":   rmse_adapted,
        "rmse_scratch":   rmse_scratch,
        "samples_adapted": ADAPT_SAMPLES,
        "samples_scratch": len(X_full),
        "time_adapted":   elapsed_adapted,
        "time_scratch":   elapsed_scratch,
        "improvement_pct": improvement,
    }


def run_ppo_experiment():
    print("\n" + "="*60)
    print("  PPO Experiment — Adapted vs From Scratch")
    print("="*60)

    BASE_PPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_thermal.zip")

    if not os.path.exists(BASE_PPO.replace(".zip", "") + ".zip"):
        # SB3 saves without .zip sometimes
        if not os.path.exists(BASE_PPO):
            print(f"  ERROR: Base PPO model not found at {BASE_PPO}")
            print("  Run Stage 3 first: cd ../stage3_ai && python run_stage3.py")
            return None

    print(f"\n  Running PPO transfer experiment on Engine (machine 3)...")
    print(f"  Three conditions: zero-shot, fine-tuned (10K steps), scratch (10K steps)")
    print(f"  This demonstrates the pre-training advantage of transfer learning\n")

    adapted_rewards, scratch_rewards = transfer_ppo_policy(
        base_model_path = BASE_PPO,
        new_machine_id  = 3,
        n_adapt_episodes = 50,
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_adapted_engine.zip"),
    )

    avg_adapted = np.mean(adapted_rewards)
    avg_scratch = np.mean(scratch_rewards)

    print(f"\n  {'Method':<35} {'Avg reward (10 eps)':>20}")
    print(f"  {'─'*35} {'─'*20}")
    print(f"  {'Adapted from CPU policy (50 eps)':<35} {avg_adapted:>20.1f}")
    print(f"  {'Trained from scratch (50 eps)':<35} {avg_scratch:>20.1f}")
    print(f"  {'Maximum possible reward':<35} {'500.0':>20}")
    print()

    if avg_adapted > avg_scratch:
        advantage = avg_adapted - avg_scratch
        print(f"  ✓  Machine Adapter PPO outperforms scratch by {advantage:.1f} points")
        print(f"     after the same number of timesteps")
    else:
        print(f"  ⚠  Scratch matches adapted — both may have converged already")
        print(f"     This is normal when the task is simple enough")

    return {
        "avg_adapted": avg_adapted,
        "avg_scratch": avg_scratch,
        "adapted_rewards": adapted_rewards,
        "scratch_rewards": scratch_rewards,
    }


def save_results(lstm_results: dict, ppo_results: dict):
    """Save results to a JSON file ready for inclusion in the paper."""
    import json
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "experiment": "MHARS Machine Adapter Validation",
        "machine_a":  "CPU (id=0) — trained in Stage 2 & 3",
        "machine_b":  "Engine (id=3) — new unseen machine",
        "lstm":       lstm_results,
        "ppo":        ppo_results if ppo_results else "Not run",
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "stage5_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {path}")
    print("  Include these numbers in your paper's Table 1")


def print_paper_table(lstm_r: dict, ppo_r: dict):
    """Print the results in the exact format for Table 1 in the paper."""
    print("\n" + "╔" + "═"*62 + "╗")
    print("║  Table 1 — Machine Adapter Results (for your paper)         ║")
    print("╠" + "═"*62 + "╣")
    if lstm_r:
        print(f"║  LSTM RMSE — unadapted model:   {lstm_r['rmse_unadapted']*85:>6.2f}°C              ║")
        print(f"║  LSTM RMSE — Machine Adapter:   {lstm_r['rmse_adapted']*85:>6.2f}°C  (100 samples) ║")
        print(f"║  LSTM RMSE — Full retrain:      {lstm_r['rmse_scratch']*85:>6.2f}°C  (2000 samples)║")
        print(f"║  Error reduction vs unadapted:  {lstm_r['improvement_pct']:>6.1f}%                 ║")
        print(f"║  Sample efficiency:             {lstm_r['samples_scratch']//lstm_r['samples_adapted']}x fewer samples        ║")
        print(f"║  Adaptation time:               {lstm_r['time_adapted']:>6.1f}s                  ║")
    print("╠" + "═"*62 + "╣")
    if ppo_r:
        print(f"║  PPO reward — adapted (50 eps): {ppo_r['avg_adapted']:>6.1f}                  ║")
        print(f"║  PPO reward — scratch (50 eps): {ppo_r['avg_scratch']:>6.1f}                  ║")
        print(f"║  Maximum possible reward:        500.0                    ║")
    print("╚" + "═"*62 + "╝")


def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   MHARS Stage 5 — Machine Adapter Experiment        ║")
    print("║   Machine A: CPU (trained)                          ║")
    print("║   Machine B: Engine (new, unseen)                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    lstm_results = run_lstm_experiment()
    ppo_results  = run_ppo_experiment()

    if lstm_results:
        save_results(lstm_results, ppo_results)
        print_paper_table(lstm_results, ppo_results)

    print("\n  Stage 5 complete.")
    print("  Your MHARS system is now fully implemented.")
    print("  All 5 stages done: Simulation → ML → AI → Adapter ✓\n")


if __name__ == "__main__":
    main()