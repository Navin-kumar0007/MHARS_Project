"""
P2.4 — Train the supervised fault classifier.

Drives the REAL MHARS pipeline through the same demo dynamics the API uses
(anomaly injection + RL action cooling), labelling each tick with the active
fault type, and captures the exact feature vector the classifier sees at
inference (core._fault_feature_vector). Trains a FaultClassifier with class
weighting (normal dominates) and writes model + meta.

Run:  python3 tools/train_fault_classifier.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mhars import MHARS
from mhars.models import FaultClassifier
from mhars.config import Config
from mhars.schemas import SensorReading

KEYS = Config.FAULT_ANOMALY_KEYS                  # [normal, temperature_spike, …]
ANOMALY = {   # mirrors api/main.py ANOMALY_PROFILES
    "temperature_spike": {"delta": 8.0,  "duration": 1,  "noise": 0.0},
    "bearing_wear":      {"delta": 1.5,  "duration": 10, "noise": 0.0},
    "fan_blockage":      {"delta": 0.8,  "duration": 15, "noise": 0.0},
    "sensor_drift":      {"delta": 0.0,  "duration": 12, "noise": 3.0},
    "power_surge":       {"delta": 12.0, "duration": 1,  "noise": 0.0},
}
ACTION_FX = {"throttle": -2.0, "fan+": -1.0, "alert": -0.3, "shutdown": -10.0, "emergency-shutdown": -10.0}
STEPS = 3000
torch.manual_seed(Config.SEED); np.random.seed(Config.SEED)


def collect(machine_id, n_steps):
    sysm = MHARS(machine_type_id=machine_id, llm_path=None, verbose=False)
    prof = Config.MACHINE_PROFILES[machine_id]
    temp = float(prof["idle"])
    active, rem, cooldown = None, 0, 0
    X, y = [], []
    for _ in range(n_steps):
        if active is None and cooldown <= 0 and np.random.random() < 0.07:
            active = np.random.choice(list(ANOMALY.keys()))
            rem = ANOMALY[active]["duration"]

        if active is not None and rem > 0:
            p = ANOMALY[active]
            temp += p["delta"]
            if p["noise"] > 0:
                temp += np.random.uniform(-p["noise"], p["noise"])
            rem -= 1
            label = KEYS.index(active)
            if rem <= 0:
                active, cooldown = None, 8
        else:
            temp += (np.random.random() - 0.48) * 0.8
            label = 0
            if cooldown > 0:
                cooldown -= 1
        temp = max(20.0, temp)

        sr = SensorReading(temp_c=float(temp), load_pct=0.5, ambient_c=25.0)
        res = sysm.run(temp_celsius=sr, sync_alert=True)
        m = res.metadata
        feats = sysm._fault_feature_vector(m["if_score"], m["lstm_score"], m["ae_score"],
                                           m["vib_score"], m["context_score"], res.urgency)
        X.append(feats); y.append(label)

        temp = max(20.0, temp + ACTION_FX.get(res.action, 0.0))
    return X, y


def main():
    X, y = [], []
    for mid in sorted(Config.MACHINE_PROFILES.keys()):
        xs, ys = collect(mid, STEPS)
        X.extend(xs); y.extend(ys)
        print(f"  [machine {mid}] {len(xs)} samples")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    n_classes = len(Config.FAULT_CLASSES)
    counts = np.bincount(y, minlength=n_classes)
    print(f"Pooled X={X.shape}  class counts={counts.tolist()}")

    mean = X.mean(axis=0); std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std

    idx = np.random.permutation(len(Xn)); cut = int(len(Xn) * 0.85)
    tr, va = idx[:cut], idx[cut:]

    # Inverse-frequency class weights (normal dominates, spikes are rare).
    weights = len(y) / (n_classes * np.clip(counts, 1, None))
    wt = torch.tensor(weights, dtype=torch.float32)

    model = FaultClassifier(n_features=X.shape[1], n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=wt)
    dl = DataLoader(TensorDataset(torch.tensor(Xn[tr]), torch.tensor(y[tr])), batch_size=256, shuffle=True)

    for ep in range(60):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        if ep % 15 == 14:
            model.eval()
            with torch.no_grad():
                pv = model(torch.tensor(Xn[va])).argmax(1).numpy()
            acc = (pv == y[va]).mean()
            print(f"  epoch {ep+1}/60  val_acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xn[va])).argmax(1).numpy()
    val_acc = float((pv == y[va]).mean())
    # Per-class recall
    print("  per-class recall:")
    for c in range(n_classes):
        mask = y[va] == c
        rec = (pv[mask] == c).mean() if mask.sum() else float("nan")
        print(f"    {Config.FAULT_CLASSES[c]:<18} n={int(mask.sum()):4d}  recall={rec:.2f}")

    torch.save(model.state_dict(), Config.FAULT_CLASSIFIER)
    with open(Config.FAULT_CLASSIFIER_META, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(),
                   "n_features": int(X.shape[1]), "n_classes": n_classes,
                   "classes": Config.FAULT_CLASSES, "val_acc": round(val_acc, 3)}, f, indent=2)
    print(f"\n✓ Fault classifier saved (val_acc={val_acc:.3f}) → {Config.FAULT_CLASSIFIER}")


if __name__ == "__main__":
    main()
