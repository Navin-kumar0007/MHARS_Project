"""
MHARS — benchmarks/benchmark_runner.py
========================================
Run: python benchmarks/benchmark_runner.py
Run 3 seeds: python benchmarks/benchmark_runner.py --seeds 3
"""

import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage2_ml'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage3_ai'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage5_adapter'))

import numpy as np


SCHEMA = {
    "version":    "1.0",
    "experiment": "MHARS Machine Adapter Benchmark",
    "machine_a":  "CPU (id=0)",
    "machine_b":  "Engine (id=3)",
    "metrics": {
        "lstm_rmse_unadapted":  "RMSE of CPU-trained LSTM on Engine data (no adaptation)",
        "lstm_rmse_adapted":    "RMSE after Machine Adapter fine-tune (100 samples)",
        "lstm_rmse_scratch":    "RMSE after full retrain on Engine data (2000 samples)",
        "lstm_sample_efficiency": "Ratio scratch_samples / adapter_samples",
        "lstm_error_reduction_pct": "% improvement of adapter over unadapted",
        "ppo_reward_adapted":   "Avg PPO reward (adapted, 50 episodes)",
        "ppo_reward_scratch":   "Avg PPO reward (from scratch, 50 episodes)",
        "ppo_advantage":        "ppo_reward_adapted - ppo_reward_scratch",
        "adapter_time_s":       "Wall-clock time for Machine Adapter to run",
    }
}


def run_one_seed(seed: int) -> dict:
    from load_cmapss import load_cmapss, preprocess, make_lstm_windows, _generate_synthetic_cmapss
    from lstm_predictor import ThermalLSTM
    from machine_adapter import MachineAdapter, find_most_similar_machine
    import torch, torch.nn as nn

    print(f"\n  Seed {seed} — running...")

    # Generate Machine B (Engine) data
    df = _generate_synthetic_cmapss(n_units=50, seed=seed)
    from load_cmapss import preprocess
    df = preprocess(df)
    df["s4_norm"] = (df["s4_norm"] * 1.15).clip(0, 1)

    X_B, y_B, _ = make_lstm_windows(df, window=12)
    X_val  = X_B[100:300]
    y_val  = y_B[100:300]
    X_full = X_B[:2000]
    y_full = y_B[:2000]

    def rmse(model, X, y):
        model.eval()
        x_t = torch.FloatTensor(X).unsqueeze(-1)
        y_t = torch.FloatTensor(y)
        with torch.no_grad():
            return float(nn.MSELoss()(model(x_t), y_t).item() ** 0.5)

    # 1. Unadapted
    ckpt  = torch.load("models/lstm.pt", map_location="cpu")
    h     = ckpt["lstm.weight_ih_l0"].shape[0] // 4
    base  = ThermalLSTM(hidden_size=h)
    base.load_state_dict(ckpt)
    rmse_unadapted = rmse(base, X_val, y_val)

    # 2. Machine Adapter (100 samples)
    similar_id, sim = find_most_similar_machine(3, [0, 1, 2])
    adapter = MachineAdapter("models/lstm.pt", similar_id)
    t0      = time.time()
    rmse_adapted = adapter.adapt(X_B, y_B, n_samples=100, epochs=20)
    adapter_time = time.time() - t0

    # 3. Full retrain (2000 samples)
    from lstm_predictor import ThermalLSTM
    scratch = ThermalLSTM(hidden_size=h)
    opt     = torch.optim.Adam(scratch.parameters(), lr=0.001)
    crit    = nn.MSELoss()
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(
        torch.FloatTensor(X_full).unsqueeze(-1),
        torch.FloatTensor(y_full)), batch_size=256, shuffle=True)
    scratch.train()
    for _ in range(30):
        for xb, yb in dl:
            opt.zero_grad(); crit(scratch(xb), yb).backward(); opt.step()
    rmse_scratch = rmse(scratch, X_val, y_val)

    # 4. PPO experiment
    from machine_adapter import transfer_ppo_policy
    adapted_r, scratch_r = transfer_ppo_policy(
        "models/ppo_thermal.zip", 3,
        n_adapt_episodes=50,
        save_path=f"models/ppo_adapted_seed{seed}.zip"
    )
    ppo_adapted = float(np.mean(adapted_r))
    ppo_scratch = float(np.mean(scratch_r))

    improvement = ((rmse_unadapted - rmse_adapted) / rmse_unadapted) * 100

    return {
        "seed":                     seed,
        "timestamp":                time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lstm_rmse_unadapted":      round(rmse_unadapted, 6),
        "lstm_rmse_adapted":        round(rmse_adapted,   6),
        "lstm_rmse_scratch":        round(rmse_scratch,   6),
        "lstm_sample_efficiency":   2000 // 100,
        "lstm_error_reduction_pct": round(improvement, 2),
        "adapter_time_s":           round(adapter_time, 2),
        "ppo_reward_adapted":       round(ppo_adapted, 2),
        "ppo_reward_scratch":       round(ppo_scratch, 2),
        "ppo_advantage":            round(ppo_adapted - ppo_scratch, 2),
        "similar_machine_id":       similar_id,
        "similar_machine_sim":      round(sim, 4),
    }


def run_benchmark(n_seeds: int = 3):
    os.makedirs("benchmarks/results", exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  MHARS Benchmark — {n_seeds} seed(s)")
    print(f"{'='*58}")

    all_results = []
    for seed in range(n_seeds):
        r = run_one_seed(seed)
        all_results.append(r)

        fname = f"benchmarks/results/seed_{seed}_{r['timestamp'].replace(':','-')}.json"
        with open(fname, "w") as f:
            json.dump({**SCHEMA, "result": r}, f, indent=2)
        print(f"  Saved → {fname}")

    # Summary statistics
    keys = ["lstm_rmse_unadapted","lstm_rmse_adapted","lstm_rmse_scratch",
            "ppo_reward_adapted","ppo_reward_scratch","ppo_advantage"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        summary[k] = {
            "mean":   round(float(np.mean(vals)), 4),
            "std":    round(float(np.std(vals)),  4),
            "min":    round(float(np.min(vals)),  4),
            "max":    round(float(np.max(vals)),  4),
        }

    summary_path = "benchmarks/results/summary.json"
    with open(summary_path, "w") as f:
        json.dump({**SCHEMA, "n_seeds": n_seeds, "summary": summary,
                   "individual_runs": all_results}, f, indent=2)

    # Print table
    print(f"\n{'─'*58}")
    print(f"  {'Metric':<35} {'Mean':>8}  {'Std':>6}")
    print(f"  {'─'*35} {'─'*8}  {'─'*6}")
    for k, v in summary.items():
        label = k.replace("_"," ").replace("lstm ","LSTM ").replace("ppo ","PPO ")
        print(f"  {label:<35} {v['mean']:>8.4f}  {v['std']:>6.4f}")
    print(f"{'─'*58}")
    print(f"\n  Summary → {summary_path}")
    print("  Include mean ± std values in paper Table 1\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(n_seeds=args.seeds)