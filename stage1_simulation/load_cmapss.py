"""
MHARS — Stage 1: NASA CMAPSS Dataset Loader
=============================================
Loads the NASA CMAPSS turbofan engine dataset (FD001).
If the real dataset is not yet downloaded, generates realistic
synthetic thermal data so you can start coding immediately.

Real dataset download:
  https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

The dataset has 21 sensor columns. We use the most thermally relevant:
  - s2:  total temperature at fan inlet
  - s3:  total temperature at LPC outlet
  - s4:  total temperature at HPC outlet
  - s7:  total temperature at HPT outlet
  - s11: total temperature at LPT outlet
  - s12: ratio of fuel flow to static pressure

Each row = one operating cycle of one engine unit.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os

# Column names for CMAPSS FD001
CMAPSS_COLS = [
    "unit_id", "cycle",
    "op1", "op2", "op3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

# Thermally relevant sensor columns we will use in MHARS
THERMAL_SENSORS = ["s2", "s3", "s4", "s7", "s11"]
PRIMARY_TEMP    = "s4"   # HPC outlet temperature — most predictive of failure


def load_cmapss(filepath: str | None = None) -> pd.DataFrame:
    """
    Load CMAPSS FD001 training data. If filepath is None or file not found,
    generates synthetic data with the same statistical properties.
    """
    if filepath and os.path.exists(filepath):
        print(f"Loading real CMAPSS dataset from: {filepath}")
        df = pd.read_csv(filepath, sep=" ", header=None, names=CMAPSS_COLS)
        # Drop the two extra empty columns CMAPSS files sometimes contain
        df = df.dropna(axis=1, how="all")
        # Add RUL (Remaining Useful Life) column
        max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()  # type: ignore[union-attr]
        max_cycles.columns = ["unit_id", "max_cycle"]
        df = df.merge(max_cycles, on="unit_id")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df.drop("max_cycle", axis=1, inplace=True)
        print(f"Loaded {len(df):,} rows, {df['unit_id'].nunique()} engine units")
        return df

    print("CMAPSS file not found — generating synthetic thermal dataset.")
    print("Download the real data from:")
    print("  https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
    print()
    return _generate_synthetic_cmapss()


def _generate_synthetic_cmapss(n_units: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic engine degradation data with realistic thermal patterns:
    - Temperature trends upward as engine degrades (RUL decreases)
    - Random noise per cycle
    - Unit-to-unit variation in baseline temperature
    """
    rng = np.random.default_rng(seed)
    rows = []

    for unit_id in range(1, n_units + 1):
        # Each engine has a random lifespan between 150 and 350 cycles
        max_cycle = rng.integers(150, 350)
        # Random unit-level baseline offset
        baseline_offset = rng.normal(0, 5)

        for cycle in range(1, max_cycle + 1):
            rul = max_cycle - cycle
            # Degradation factor: temperature rises as RUL decreases
            degradation = (1 - rul / max_cycle) * 40

            row = {
                "unit_id": unit_id,
                "cycle": cycle,
                "rul": rul,
                "op1": rng.uniform(0, 1),
                "op2": rng.uniform(20, 25),
                "op3": rng.integers(60, 100),
            }
            # Generate thermal sensor readings
            for i, sensor in enumerate(["s2","s3","s4","s7","s11"]):
                base = [550, 590, 1380, 1020, 610][i]
                spread = [10, 12, 20, 15, 10][i]
                noise = rng.normal(0, spread * 0.3)
                row[sensor] = base + baseline_offset + degradation * [0.1,0.3,1.0,0.7,0.4][i] + noise

            # Remaining sensors (non-thermal, fill with plausible values)
            for s in ["s1","s5","s6","s8","s9","s10","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"]:
                row[s] = rng.normal(50, 5)

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} synthetic rows across {n_units} engine units")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize thermal sensor readings to [0, 1] range per sensor.
    Add a binary 'near_failure' label (RUL < 30 cycles).
    """
    df = df.copy()
    for sensor in THERMAL_SENSORS:
        col_min = df[sensor].min()
        col_max = df[sensor].max()
        df[f"{sensor}_norm"] = (df[sensor] - col_min) / (col_max - col_min + 1e-8)

    df["near_failure"] = (df["rul"] < 30).astype(int)
    return df


def make_lstm_windows(df: pd.DataFrame, window: int = 12):
    """
    Create sliding windows of 12 consecutive readings per unit.
    Input X: window of 12 normalized thermal readings
    Target y: primary temperature at the NEXT timestep

    This is exactly the format the LSTM in Stage 2 will consume.
    """
    X, y, unit_ids = [], [], []

    for unit_id, group in df.groupby("unit_id"):
        temps = group[f"{PRIMARY_TEMP}_norm"].values
        for i in range(len(temps) - window):
            X.append(temps[i : i + window])
            y.append(temps[i + window])
            unit_ids.append(unit_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, np.array(unit_ids)


def plot_thermal_trends(df: pd.DataFrame, n_units: int = 5, save_path: str = "thermal_trends.png"):
    """
    Plot the primary temperature sensor over the engine's lifetime for N units.
    Shows how temperature rises as the engine degrades toward failure.
    """
    fig, axes = plt.subplots(n_units, 1, figsize=(12, 3 * n_units), sharex=False)
    if n_units == 1:
        axes = [axes]

    sample_units = df["unit_id"].unique()[:n_units]
    colors = ["#0F6E56", "#534AB7", "#993C1D", "#854F0B", "#3B6D11"]

    for ax, unit_id, color in zip(axes, sample_units, colors):
        unit_data = df[df["unit_id"] == unit_id].sort_values("cycle")  # type: ignore[call-overload]
        ax.plot(unit_data["cycle"], unit_data[PRIMARY_TEMP],
                color=color, linewidth=1.5, label=f"Unit {unit_id}")
        # Mark the danger zone (last 30 cycles before failure)
        danger = unit_data[unit_data["rul"] < 30]
        if len(danger) > 0:
            ax.axvspan(danger["cycle"].min(), danger["cycle"].max(),
                       alpha=0.15, color="#993C1D", label="Near-failure zone (RUL < 30)")
        ax.set_ylabel("HPC temp (s4)")
        ax.set_title(f"Engine unit {unit_id} — {len(unit_data)} cycles", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Engine cycle")
    plt.suptitle("NASA CMAPSS — Primary thermal sensor (s4) degradation trends", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Trend plot saved to: {save_path}")


def print_dataset_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("CMAPSS Dataset Summary")
    print("="*55)
    print(f"  Total rows:      {len(df):,}")
    print(f"  Engine units:    {df['unit_id'].nunique()}")
    print(f"  Avg life (cyc):  {df.groupby('unit_id')['cycle'].max().mean():.0f}")  # type: ignore[union-attr]
    print(f"  Near-failure %:  {df['near_failure'].mean()*100:.1f}%")
    print(f"\n  Primary temp ({PRIMARY_TEMP}) statistics:")
    print(f"    Min:  {df[PRIMARY_TEMP].min():.1f}")
    print(f"    Max:  {df[PRIMARY_TEMP].max():.1f}")
    print(f"    Mean: {df[PRIMARY_TEMP].mean():.1f}")
    print(f"    Std:  {df[PRIMARY_TEMP].std():.1f}")
    print("="*55 + "\n")