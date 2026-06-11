"""
P1.5 — Train a direct multi-horizon thermal forecaster.

The previous LSTM emitted a single 1-step-ahead value while the UI claimed a
"+10 min forecast". This trains ThermalLSTMv2 with output_horizon=H so it
predicts the next H normalized temps (t+1 … t+H) directly — a real forward
trajectory. Trained on the serving distribution (captured from the live
pipeline over gym_env), consistent with P1.1.

Run:  python3 tools/train_forecaster.py
"""
import os, sys
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
MODELS = os.path.join(ROOT, "models")
STEPS = 2500
torch.manual_seed(Config.SEED); np.random.seed(Config.SEED)


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
    print(f"Pooled: X={X.shape} Y={Y.shape} (horizon={H})")

    model = ThermalLSTMv2(input_size=5, hidden_size=Config.LSTM_HIDDEN_V2,
                          num_layers=Config.LSTM_LAYERS_V2, output_horizon=H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)), batch_size=256, shuffle=True)
    model.train()
    for ep in range(45):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); pred = model(xb); loss = lossf(pred, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep % 10 == 9:
            print(f"  epoch {ep+1}/45  mse={tot/len(X):.6f}")
    model.eval()

    # Per-horizon RMSE (denormalized-agnostic, in normalized units)
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()
    rmse_per_h = np.sqrt(((pred - Y) ** 2).mean(axis=0))
    print("  per-step RMSE (norm):", " ".join(f"{r:.4f}" for r in rmse_per_h))

    torch.save(model.state_dict(), os.path.join(MODELS, "lstm_v2.pt"))
    print(f"\n✓ Multi-horizon forecaster (H={H}) saved → models/lstm_v2.pt")


if __name__ == "__main__":
    main()
