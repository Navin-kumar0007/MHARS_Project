"""
MHARS — Publication-Ready Figures Generator
=============================================
Generates IEEE/Springer-quality figures (300 DPI) from training logs
and benchmark results.  Output: PDF + PNG in results/figures/

Run:
    python benchmarks/paper_figures.py

Figures produced:
    Fig 1 — Edge vs Cloud Latency Comparison (bar chart)
    Fig 2 — LSTM Training Convergence (dual-axis line chart)
    Fig 3 — Autoencoder Anomaly Detection Loss (line chart)
    Fig 4 — PPO Reward Curve (smoothed line chart)
    Fig 5 — Machine Adapter LSTM Results (grouped bar chart)
    Fig 6 — PPO Cross-Domain Transfer (grouped bar chart)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt

# ── Academic styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":          12,
    "axes.labelsize":     13,
    "axes.titlesize":     14,
    "legend.fontsize":    11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "lines.linewidth":    2,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Color palette — professional, distinguishable, print-safe
C_PRIMARY   = "#2166AC"   # steel blue
C_SECONDARY = "#B2182B"   # brick red
C_TERTIARY  = "#1B7837"   # forest green
C_ACCENT    = "#762A83"   # purple
C_LIGHT     = "#D6D6D6"   # light gray

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def _save(fig, name: str):
    """Save figure as both PDF (vector, for LaTeX) and PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"  ✓ Saved: figures/{name}.pdf  +  figures/{name}.png")


# ── Figure 1: Latency Comparison ─────────────────────────────────────────────
def fig1_latency():
    """Bar chart: Edge vs Cloud inference latency."""
    labels   = ["Edge\n(Critical Events)", "Cloud\n(Non-Critical Events)"]
    latencies = [10.4, 4306.8]
    colors    = [C_TERTIARY, C_SECONDARY]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, latencies, color=colors, width=0.45,
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    # Value annotations
    for bar, val in zip(bars, latencies):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y * 1.15,
                f"{val:.1f} ms", ha="center", va="bottom",
                fontweight="bold", fontsize=12)

    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("MHARS Hybrid Routing: Edge vs. Cloud Latency")
    ax.set_yscale("log")
    ax.set_ylim(1, 20_000)

    # 50ms threshold line
    ax.axhline(y=50, color=C_ACCENT, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(1.35, 55, "50 ms SLA threshold", fontsize=10, color=C_ACCENT, style="italic")

    fig.tight_layout()
    _save(fig, "fig1_latency")


# ── Figure 2: LSTM Training Convergence ──────────────────────────────────────
def fig2_lstm():
    """LSTM train/val RMSE over epochs."""
    csv = os.path.join(RESULTS_DIR, "lstm_training.csv")
    if not os.path.exists(csv):
        print("  ⚠ Skipping Fig 2: lstm_training.csv not found")
        return

    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df["epoch"], df["train_rmse"], color=C_PRIMARY,
            marker="o", markersize=3, label="Training RMSE")
    ax.plot(df["epoch"], df["val_rmse"], color=C_SECONDARY,
            marker="s", markersize=3, label="Validation RMSE")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Root Mean Square Error (RMSE)")
    ax.set_title("LSTM Temperature Predictor — Training Convergence")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig2_lstm_convergence")


# ── Figure 3: Autoencoder Loss ───────────────────────────────────────────────
def fig3_autoencoder():
    """AE validation MSE over epochs."""
    csv = os.path.join(RESULTS_DIR, "ae_training.csv")
    if not os.path.exists(csv):
        print("  ⚠ Skipping Fig 3: ae_training.csv not found")
        return

    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df["epoch"], df["val_loss"], color=C_ACCENT,
            marker="^", markersize=4, label="Validation MSE")
    ax.fill_between(df["epoch"], df["val_loss"], alpha=0.15, color=C_ACCENT)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Autoencoder Anomaly Detector — Training Loss")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig3_autoencoder_loss")


# ── Figure 4: PPO Reward Curve ───────────────────────────────────────────────
def fig4_ppo():
    """PPO episode reward with rolling average."""
    csv = os.path.join(RESULTS_DIR, "ppo_training.csv")
    if not os.path.exists(csv):
        print("  ⚠ Skipping Fig 4: ppo_training.csv not found")
        return

    df = pd.read_csv(csv)
    window = min(20, len(df))
    rolling = df["reward"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(df["episode"], df["reward"],
            color=C_LIGHT, linewidth=0.8, alpha=0.7, label="Raw Episode Reward")
    ax.plot(df["episode"], rolling,
            color=C_TERTIARY, linewidth=2.5, label=f"Rolling Avg ({window} ep)")

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.4,
               label="Break-even (0)")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("PPO Adaptive Agent — Learning Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig4_ppo_reward")


# ── Figure 5: Machine Adapter LSTM Results ───────────────────────────────────
def fig5_adapter_lstm():
    """Grouped bar chart: LSTM RMSE — Unadapted vs Adapted vs Scratch."""
    json_path = os.path.join(RESULTS_DIR, "stage5_results.json")
    if not os.path.exists(json_path):
        print("  ⚠ Skipping Fig 5: stage5_results.json not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    lstm = data["lstm"]
    methods = ["No Adaptation", "Machine Adapter\n(100 samples)", "Full Retrain\n(2,000 samples)"]
    rmse_celsius = [lstm["rmse_unadapted"] * 85, lstm["rmse_adapted"] * 85, lstm["rmse_scratch"] * 85]
    colors = [C_LIGHT, C_PRIMARY, C_SECONDARY]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, rmse_celsius, color=colors, width=0.5,
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar, val in zip(bars, rmse_celsius):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}°C", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("RMSE (°C)")
    ax.set_title("LSTM Transfer Learning — Machine Adapter Performance")
    ax.set_ylim(0, max(rmse_celsius) * 1.25)

    # Annotation: improvement
    improvement = lstm["improvement_pct"]
    ax.annotate(f"↓ {improvement:.1f}% error\nreduction",
                xy=(1, rmse_celsius[1]), xytext=(1.6, rmse_celsius[0]),
                fontsize=10, color=C_TERTIARY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_TERTIARY, lw=1.5))

    fig.tight_layout()
    _save(fig, "fig5_adapter_lstm")


# ── Figure 6: PPO Transfer Results ──────────────────────────────────────────
def fig6_adapter_ppo():
    """Grouped bar chart: PPO adapted vs scratch rewards."""
    json_path = os.path.join(RESULTS_DIR, "stage5_results.json")
    if not os.path.exists(json_path):
        print("  ⚠ Skipping Fig 6: stage5_results.json not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    ppo = data["ppo"]
    methods  = ["Adapted\n(CPU → Engine)", "From Scratch\n(Random Init)"]
    rewards  = [ppo["avg_adapted"], ppo["avg_scratch"]]
    colors   = [C_PRIMARY, C_SECONDARY]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(methods, rewards, color=colors, width=0.45,
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar, val in zip(bars, rewards):
        y = bar.get_height()
        v_offset = 8 if val >= 0 else -20
        ax.text(bar.get_x() + bar.get_width() / 2, y + v_offset,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.axhline(y=500, color=C_TERTIARY, linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(1.35, 505, "Max reward (500)", fontsize=10, color=C_TERTIARY, style="italic")

    ax.set_ylabel("Average Reward (10 episodes)")
    ax.set_title("PPO Transfer Learning — Adapted vs. Scratch (25K Steps)")
    ax.set_ylim(min(rewards) - 50, 600)

    fig.tight_layout()
    _save(fig, "fig6_adapter_ppo")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   MHARS — Generating Publication-Ready Figures       ║")
    print("║   Output: results/figures/ (PDF + PNG, 300 DPI)      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    fig1_latency()
    fig2_lstm()
    fig3_autoencoder()
    fig4_ppo()
    fig5_adapter_lstm()
    fig6_adapter_ppo()

    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    print("  Use the PDF versions for LaTeX, PNG versions for Word/Google Docs.\n")
