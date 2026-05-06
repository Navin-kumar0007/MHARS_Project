"""
MHARS — Stage 1: NASA CMAPSS Dataset Loader (FIXED)
=====================================================
Fix applied: ISSUE-4 — load_cmapss() now automatically looks for
the real NASA file in data/ folder before falling back to synthetic.

Real dataset download:
  https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
  Save as: data/train_FD001.txt
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Column names for CMAPSS FD001
CMAPSS_COLS = [
    "unit_id", "cycle",
    "op1", "op2", "op3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

THERMAL_SENSORS = ["s2", "s3", "s4", "s7", "s11"]
PRIMARY_TEMP    = "s4"

# ── Auto-detect data path ─────────────────────────────────────────────────────
def _find_cmapss_file():
    """
    Search for train_FD001.txt in common locations relative to this file.
    Returns the path if found, None if not found.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "..", "data", "train_FD001.txt"),
        os.path.join(this_dir, "data", "train_FD001.txt"),
        os.path.join(this_dir, "train_FD001.txt"),
        "data/train_FD001.txt",
        "train_FD001.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def load_cmapss(filepath: str = None) -> pd.DataFrame:
    """
    Load CMAPSS FD001 training data.

    If filepath is provided and exists, uses that.
    If filepath is None, auto-searches for the file.
    If not found anywhere, falls back to synthetic data.
    """
    # Try provided path first
    if filepath and os.path.exists(filepath):
        return _load_real(filepath)

    # Auto-detect
    if filepath is None:
        found = _find_cmapss_file()
        if found:
            return _load_real(found)

    # Fallback to synthetic
    if filepath:
        print(f"CMAPSS file not found at: {filepath}")
    else:
        print("CMAPSS file not found — generating synthetic thermal dataset.")
        print("Download the real data from:")
        print("  https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
        print("  Save as: data/train_FD001.txt")
        print()
    return _generate_synthetic_cmapss()


def _load_real(filepath: str) -> pd.DataFrame:
    print(f"Loading real CMAPSS dataset from: {filepath}")
    df = pd.read_csv(filepath, sep=" ", header=None,
                     names=CMAPSS_COLS + ["_extra1", "_extra2"])
    df = df.drop(columns=["_extra1", "_extra2"], errors="ignore")
    df = df.dropna(axis=1, how="all")

    # Add RUL column
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    print(f"Loaded {len(df):,} rows, {df['unit_id'].nunique()} engine units")
    return df


def _generate_synthetic_cmapss(n_units: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic engine degradation data.
    Temperature rises as engine degrades (RUL decreases).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for unit_id in range(1, n_units + 1):
        max_cycle       = rng.integers(150, 350)
        baseline_offset = rng.normal(0, 5)

        for cycle in range(1, max_cycle + 1):
            rul         = max_cycle - cycle
            degradation = (1 - rul / max_cycle) * 40

            row = {
                "unit_id": unit_id, "cycle": cycle, "rul": rul,
                "op1": rng.uniform(0, 1),
                "op2": rng.uniform(20, 25),
                "op3": rng.integers(60, 100),
            }
            for i, sensor in enumerate(["s2","s3","s4","s7","s11"]):
                base   = [550, 590, 1380, 1020, 610][i]
                spread = [10,  12,  20,   15,   10][i]
                noise  = rng.normal(0, spread * 0.3)
                row[sensor] = base + baseline_offset + degradation * [0.1,0.3,1.0,0.7,0.4][i] + noise

            for s in ["s1","s5","s6","s8","s9","s10","s12","s13",
                      "s14","s15","s16","s17","s18","s19","s20","s21"]:
                row[s] = rng.normal(50, 5)

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} synthetic rows across {n_units} engine units")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for sensor in THERMAL_SENSORS:
        col_min = df[sensor].min()
        col_max = df[sensor].max()
        df[f"{sensor}_norm"] = (df[sensor] - col_min) / (col_max - col_min + 1e-8)
    df["near_failure"] = (df["rul"] < 30).astype(int)
    return df


def make_lstm_windows(df: pd.DataFrame, window: int = 12):
    X, y, unit_ids = [], [], []
    for unit_id, group in df.groupby("unit_id"):
        temps = group[f"{PRIMARY_TEMP}_norm"].values
        for i in range(len(temps) - window):
            X.append(temps[i:i + window])
            y.append(temps[i + window])
            unit_ids.append(unit_id)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(unit_ids))


def plot_thermal_trends(df, n_units=5,
                         save_path="results/thermal_trends.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(n_units, 1, figsize=(12, 3 * n_units), sharex=False)
    if n_units == 1:
        axes = [axes]
    sample_units = df["unit_id"].unique()[:n_units]
    colors = ["#0F6E56","#534AB7","#993C1D","#854F0B","#3B6D11"]

    for ax, unit_id, color in zip(axes, sample_units, colors):
        unit_data = df[df["unit_id"] == unit_id].sort_values("cycle")
        ax.plot(unit_data["cycle"], unit_data[PRIMARY_TEMP],
                color=color, linewidth=1.5, label=f"Unit {unit_id}")
        danger = unit_data[unit_data["rul"] < 30]
        if len(danger) > 0:
            ax.axvspan(danger["cycle"].min(), danger["cycle"].max(),
                       alpha=0.15, color="#993C1D", label="Near-failure (RUL<30)")
        ax.set_ylabel("HPC temp (s4)")
        ax.set_title(f"Engine unit {unit_id} — {len(unit_data)} cycles")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Engine cycle")
    plt.suptitle("NASA CMAPSS — Thermal sensor (s4) degradation trends",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Trend plot saved to: {save_path}")


def print_dataset_summary(df):
    src = "real NASA CMAPSS" if df[PRIMARY_TEMP].max() > 1400 else "synthetic"
    print(f"\n{'='*55}")
    print(f"CMAPSS Dataset Summary  [{src}]")
    print(f"{'='*55}")
    print(f"  Total rows:     {len(df):,}")
    print(f"  Engine units:   {df['unit_id'].nunique()}")
    print(f"  Avg life (cyc): {df.groupby('unit_id')['cycle'].max().mean():.0f}")
    print(f"  Near-failure %: {df['near_failure'].mean()*100:.1f}%")
    print(f"\n  Primary temp ({PRIMARY_TEMP}) statistics:")
    print(f"    Min:  {df[PRIMARY_TEMP].min():.1f}")
    print(f"    Max:  {df[PRIMARY_TEMP].max():.1f}")
    print(f"    Mean: {df[PRIMARY_TEMP].mean():.1f}")
    print(f"    Std:  {df[PRIMARY_TEMP].std():.1f}")
    print(f"{'='*55}\n")