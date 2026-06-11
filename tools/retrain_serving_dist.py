"""
P1.1 — Align train/serve distribution.

Drives the REAL MHARS pipeline (core.py) over gym_env *normal* rollouts across
all machine profiles, captures the exact feature tensors that inference feeds
to the forecast LSTM, the LSTM-autoencoder and the vibration detector, then
retrains those three models on that captured serving distribution and
recalibrates their thresholds + the conformal predictor.

Because the SAME code path produces both training and serving features, the
out-of-distribution saturation (vib pinned at 1.0, AE that can't separate
normal from fault) is eliminated at the source.

Run:  python3 tools/retrain_serving_dist.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mhars import MHARS
from mhars.models import ThermalLSTMv2, ThermalAutoencoderLSTM, VibrationDetector
from mhars.config import Config
from mhars.conformal import ConformalPredictor
from mhars.schemas import SensorReading
from stage1_simulation.gym_env import ThermalEnv

WIN = Config.LSTM_WINDOW          # 12
MODELS = os.path.join(ROOT, "models")
STEPS_PER_MACHINE = 2500
torch.manual_seed(Config.SEED); np.random.seed(Config.SEED)


def collect(machine_id, n_steps):
    """Roll out one machine under benign control, capture inference features."""
    sysm = MHARS(machine_type_id=machine_id, llm_path=None, verbose=False)
    env = ThermalEnv(machine_type_id=machine_id)
    env.reset()
    prof = env.profile

    windows, temp_norms, vib_feats = [], [], []
    for _ in range(n_steps):
        # Benign thermostat policy → realistic NORMAL operation, no faults.
        action = 1 if env.temp > prof["target_temp"] else 0
        env.step(action)
        sr = SensorReading(temp_c=float(env.temp),
                           load_pct=float(getattr(env, "load_level", 0.5)),
                           ambient_c=25.0)
        sysm.run(temp_celsius=sr, sync_alert=True)

        mw = list(sysm._multi_sensor_window)
        if len(mw) < WIN:
            continue
        windows.append([list(map(float, row)) for row in mw])   # (12,5)
        temp_norms.append(float(mw[-1][2]))                     # s4 = temp_norm

        # Replicate the vibration feature synthesis from core._compute_vib_score
        tw = list(sysm._temp_window)
        recent = np.array(tw[-12:]) if len(tw) >= 12 else np.array(tw)
        rms = float(np.sqrt(np.mean(recent ** 2))) * 2.0
        p2p = float(np.ptp(recent)) * 3.0
        crest = float(np.max(np.abs(recent)) / (rms + 1e-8))
        centroid = 100.0 + float(np.std(recent)) * 100
        std_val = float(np.std(recent))
        vib_feats.append([rms, p2p, crest, centroid, std_val])

    print(f"  [machine {machine_id}] captured {len(windows)} windows")
    return windows, temp_norms, vib_feats


def main():
    all_windows, all_norms, all_vib = [], [], []
    # lstm pairs: window_t → temp_norm_{t+1}; keep per-machine to avoid cross-joins
    lstm_X, lstm_y = [], []

    for mid in sorted(Config.MACHINE_PROFILES.keys()):
        w, tn, vf = collect(mid, STEPS_PER_MACHINE)
        all_windows.extend(w)
        all_norms.extend(tn)
        all_vib.extend(vf)
        for i in range(len(w) - 1):
            lstm_X.append(w[i])
            lstm_y.append(tn[i + 1])

    X = np.array(all_windows, dtype=np.float32)        # (N,12,5)
    Vib = np.array(all_vib, dtype=np.float32)          # (M,5)
    LX = np.array(lstm_X, dtype=np.float32)            # (K,12,5)
    LY = np.array(lstm_y, dtype=np.float32)            # (K,)
    print(f"\nPooled: AE windows={X.shape}, vib={Vib.shape}, lstm pairs={LX.shape}")

    # ── 1. Forecast LSTM v2 (predict next-step temp_norm) ────────────────────
    print("\n[1/4] Training ThermalLSTMv2 (forecast)…")
    lstm = ThermalLSTMv2(input_size=5, hidden_size=Config.LSTM_HIDDEN_V2,
                         num_layers=Config.LSTM_LAYERS_V2)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    # train/val split for conformal calibration
    n = len(LX); idx = np.random.permutation(n); cut = int(n * 0.85)
    tr, ca = idx[:cut], idx[cut:]
    dl = DataLoader(TensorDataset(torch.tensor(LX[tr]), torch.tensor(LY[tr])),
                    batch_size=256, shuffle=True)
    lstm.train()
    for ep in range(40):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); pred = lstm(xb); loss = lossf(pred, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep % 10 == 9:
            print(f"    epoch {ep+1}/40  mse={tot/len(tr):.6f}")
    lstm.eval()
    torch.save(lstm.state_dict(), os.path.join(MODELS, "lstm_v2.pt"))

    # ── 2. Conformal recalibration on held-out residuals ─────────────────────
    print("[2/4] Recalibrating conformal predictor…")
    with torch.no_grad():
        preds = lstm(torch.tensor(LX[ca])).numpy()
    residuals = np.abs(preds - LY[ca])
    cp = ConformalPredictor(coverage=Config.CONFORMAL_COVERAGE, adaptive=True, adaptive_lr=0.003)
    cp.calibrate(residuals)
    cp.save(os.path.join(MODELS, "conformal_meta.json"))

    # ── 3. LSTM-Autoencoder v2 (reconstruction) ──────────────────────────────
    print("[3/4] Training ThermalAutoencoderLSTM (anomaly)…")
    ae = ThermalAutoencoderLSTM(input_size=5, hidden_size=32, seq_len=WIN)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X)), batch_size=256, shuffle=True)
    ae.train()
    for ep in range(40):
        tot = 0.0
        for (xb,) in dl:
            opt.zero_grad(); recon = ae(xb); loss = lossf(recon, xb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep % 10 == 9:
            print(f"    epoch {ep+1}/40  mse={tot/len(X):.6f}")
    ae.eval()
    with torch.no_grad():
        errs = ae.reconstruction_error(torch.tensor(X)).numpy()
    # Threshold so NORMAL data never saturates (a p99 threshold lets ~1% of a
    # 1 Hz stream spike to 1.0 → false alarms). Sit just above the normal max;
    # faults produce much larger error and still exceed it.
    thr = float(max(np.percentile(errs, 99.9), errs.max() * 1.05))
    torch.save(ae.state_dict(), os.path.join(MODELS, "autoencoder_lstm_v2.pt"))
    with open(os.path.join(MODELS, "autoencoder_lstm_v2_meta.json"), "w") as f:
        json.dump({"threshold": thr, "seq_len": WIN, "input_size": 5, "hidden_size": 32,
                   "normal_mean_error": float(errs.mean()),
                   "normal_p50_error": float(np.percentile(errs, 50))}, f, indent=2)
    print(f"    AE threshold(p99)={thr:.6f}  normal_mean={errs.mean():.6f} "
          f"→ median score≈{np.percentile(errs,50)/thr:.2f}")

    # ── 4. Vibration detector (reconstruction) ───────────────────────────────
    print("[4/4] Training VibrationDetector…")
    mean = Vib.mean(axis=0); std = Vib.std(axis=0) + 1e-8
    Vn = (Vib - mean) / std
    vib = VibrationDetector(n_features=5)
    opt = torch.optim.Adam(vib.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(Vn.astype(np.float32))), batch_size=256, shuffle=True)
    vib.train()
    for ep in range(40):
        tot = 0.0
        for (xb,) in dl:
            opt.zero_grad(); recon = vib(xb); loss = lossf(recon, xb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep % 10 == 9:
            print(f"    epoch {ep+1}/40  mse={tot/len(Vn):.6f}")
    vib.eval()
    with torch.no_grad():
        verrs = vib.reconstruction_error(torch.tensor(Vn.astype(np.float32))).numpy()
    vthr = float(max(np.percentile(verrs, 99.9), verrs.max() * 1.05))
    torch.save(vib.state_dict(), os.path.join(MODELS, "vibration_detector.pt"))
    with open(os.path.join(MODELS, "vibration_detector_meta.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "n_features": 5,
                   "threshold": vthr}, f, indent=2)
    print(f"    Vib threshold(p99)={vthr:.6f}  normal_mean={verrs.mean():.6f} "
          f"→ median score≈{np.percentile(verrs,50)/vthr:.2f}")

    print("\n✓ Retraining complete — models written to models/")


if __name__ == "__main__":
    main()
