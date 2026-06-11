"""
P1.5 + P2.1 — Train a direct multi-horizon *quantile* thermal forecaster.

ThermalLSTMv2 head emits H * Q values (reshaped to (H, Q)) so it predicts, for
each of the next H steps, the p10/p50/p90 quantiles directly — a real forward
trajectory (p50) WITH a native uncertainty band (p10..p90), trained by the
pinball/quantile loss. This preserves the P1.5 projection and supplies the
P2.1 uncertainty without MC-Dropout. Trained on the serving distribution
(captured from the live pipeline over gym_env), consistent with P1.1.

Run:  python3 tools/train_forecaster.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mhars import MHARS
from mhars.models import ThermalLSTMv2
from mhars.config import Config
from mhars.schemas import SensorReading
from stage1_simulation.gym_env import ThermalEnv

WIN = Config.LSTM_WINDOW
H = Config.LSTM_FORECAST_HORIZON
Q = Config.LSTM_QUANTILES
MODELS = os.path.join(ROOT, "models")
STEPS = 2500
torch.manual_seed(Config.SEED); np.random.seed(Config.SEED)


def pinball_loss(pred_hq, target_h):
    """pred_hq: (B,H,Q)  target_h: (B,H)  → mean pinball loss over H and Q."""
    losses = []
    for qi, q in enumerate(Q):
        err = target_h - pred_hq[:, :, qi]
        losses.append(torch.max((q - 1) * err, q * err))
    return torch.stack(losses, dim=-1).mean()


def collect_seq(machine_id, n_steps):
    """Return contiguous (window, temp_norm) sequence for one machine."""
    sysm = MHARS(machine_type_id=machine_id, llm_path=None, verbose=False)
    env = ThermalEnv(machine_type_id=machine_id); env.reset()
    prof = env.profile
    windows, norms = [], []
    for _ in range(n_steps):
        action = 1 if env.temp > prof["target_temp"] else 0
        env.step(action)
        sr = SensorReading(temp_c=float(env.temp),
                           load_pct=float(getattr(env, "load_level", 0.5)), ambient_c=25.0)
        sysm.run(temp_celsius=sr, sync_alert=True)
        mw = list(sysm._multi_sensor_window)
        if len(mw) == WIN:
            windows.append([list(map(float, r)) for r in mw])
            norms.append(float(mw[-1][2]))
    return windows, norms


def main():
    X, Y = [], []
    for mid in sorted(Config.MACHINE_PROFILES.keys()):
        w, tn = collect_seq(mid, STEPS)
        # Build H-step targets within this machine's contiguous sequence only.
        for i in range(len(w) - H):
            X.append(w[i])
            Y.append(tn[i + 1 : i + 1 + H])   # next H normalized temps
        print(f"  [machine {mid}] {len(w)} windows")
    X = np.array(X, dtype=np.float32)          # (N,12,5)
    Y = np.array(Y, dtype=np.float32)          # (N,H)
    nq = len(Q)
    print(f"Pooled: X={X.shape} Y={Y.shape} (horizon={H}, quantiles={Q})")

    model = ThermalLSTMv2(input_size=5, hidden_size=Config.LSTM_HIDDEN_V2,
                          num_layers=Config.LSTM_LAYERS_V2, output_horizon=H * nq)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)), batch_size=256, shuffle=True)
    model.train()
    for ep in range(50):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb).reshape(-1, H, nq)         # (B,H,Q)
            loss = pinball_loss(pred, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep % 10 == 9:
            print(f"  epoch {ep+1}/50  pinball={tot/len(X):.6f}")
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X)).reshape(-1, H, nq).numpy()
    mid = nq // 2
    rmse_per_h = np.sqrt(((pred[:, :, mid] - Y) ** 2).mean(axis=0))
    # Empirical coverage of the p10..p90 band on the training set.
    cov = ((Y >= pred[:, :, 0]) & (Y <= pred[:, :, -1])).mean()
    print("  p50 per-step RMSE (norm):", " ".join(f"{r:.4f}" for r in rmse_per_h))
    print(f"  p10–p90 empirical coverage: {cov:.3f} (target {Q[-1]-Q[0]:.2f})")

    torch.save(model.state_dict(), os.path.join(MODELS, "lstm_v2.pt"))
    with open(os.path.join(MODELS, "lstm_v2_meta.json"), "w") as f:
        json.dump({"horizon": H, "quantiles": Q, "input_size": 5,
                   "hidden_size": Config.LSTM_HIDDEN_V2}, f, indent=2)
    print(f"\n✓ Multi-horizon quantile forecaster (H={H}, Q={nq}) saved → models/lstm_v2.pt")


if __name__ == "__main__":
    main()
