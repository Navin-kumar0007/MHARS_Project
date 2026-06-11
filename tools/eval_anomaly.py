"""
P2.2 — Anomaly-detection evaluation harness.

Measures the CURRENT detector's quality on labelled data before deciding
whether a heavier SOTA model (USAD/TranAD/Anomaly-Transformer) is warranted.
Drives the real pipeline through demo dynamics with scheduled fault injections,
labels each tick (0 = normal, 1 = any active fault), then reports precision /
recall / F1 / ROC-AUC / PR-AUC for the fused context score and the individual
AE / vibration scores, plus per-fault detection rate.

Run:  python3 tools/eval_anomaly.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mhars import MHARS
from mhars.config import Config
from mhars.schemas import SensorReading

ANOMALY = {
    "temperature_spike": {"delta": 8.0,  "duration": 1,  "noise": 0.0},
    "bearing_wear":      {"delta": 1.5,  "duration": 10, "noise": 0.0},
    "fan_blockage":      {"delta": 0.8,  "duration": 15, "noise": 0.0},
    "sensor_drift":      {"delta": 0.0,  "duration": 12, "noise": 3.0},
    "power_surge":       {"delta": 12.0, "duration": 1,  "noise": 0.0},
}
ACTION_FX = {"throttle": -2.0, "fan+": -1.0, "alert": -0.3, "shutdown": -10.0, "emergency-shutdown": -10.0}
STEPS = 3000
np.random.seed(Config.SEED)


def collect(machine_id, n_steps):
    sysm = MHARS(machine_type_id=machine_id, llm_path=None, verbose=False)
    prof = Config.MACHINE_PROFILES[machine_id]
    temp = float(prof["idle"]); active = None; rem = 0; cooldown = 0
    rows = []
    for _ in range(n_steps):
        if active is None and cooldown <= 0 and np.random.random() < 0.07:
            active = np.random.choice(list(ANOMALY.keys())); rem = ANOMALY[active]["duration"]
        if active is not None and rem > 0:
            p = ANOMALY[active]; temp += p["delta"]
            if p["noise"] > 0: temp += np.random.uniform(-p["noise"], p["noise"])
            rem -= 1; fault = active
            if rem <= 0: active, cooldown = None, 8
        else:
            temp += (np.random.random() - 0.48) * 0.8; fault = None
            if cooldown > 0: cooldown -= 1
        temp = max(20.0, temp)
        res = sysm.run(temp_celsius=SensorReading(temp_c=float(temp), load_pct=0.5, ambient_c=25.0), sync_alert=True)
        m = res.metadata
        # Classifier P(fault) = 1 - P(normal) — reuses the dynamics-feature model.
        p_fault = 0.0
        if sysm._fault_clf is not None:
            import torch
            feats = sysm._fault_feature_vector(m["if_score"], m["lstm_score"], m["ae_score"], m["vib_score"], m["context_score"], res.urgency)
            x = (feats - sysm._fault_mean) / (sysm._fault_std + 1e-8)
            with torch.no_grad():
                probs = torch.softmax(sysm._fault_clf(torch.FloatTensor(x).unsqueeze(0)), dim=-1)[0]
            p_fault = float(1.0 - probs[0].item())
        rows.append((1 if fault else 0, m["context_score"], m["ae_score"], m["vib_score"], res.urgency, fault, p_fault))
        temp = max(20.0, temp + ACTION_FX.get(res.action, 0.0))
    return rows


def prf(y, score, thr):
    pred = score >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def auc_roc(y, s):
    # rank-based ROC-AUC (Mann–Whitney)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def best_f1(y, s):
    ts = np.unique(np.quantile(s, np.linspace(0.5, 0.99, 40)))
    best = (0, 0, 0, 0.0)
    for t in ts:
        p, r, f = prf(y, s, t)
        if f > best[2]: best = (p, r, f, t)
    return best


def main():
    rows = []
    for mid in sorted(Config.MACHINE_PROFILES.keys()):
        rows.extend(collect(mid, STEPS)); print(f"  [machine {mid}] collected")
    y = np.array([r[0] for r in rows])
    ctx = np.array([r[1] for r in rows]); ae = np.array([r[2] for r in rows])
    vib = np.array([r[3] for r in rows]); urg = np.array([r[4] for r in rows])
    pf = np.array([r[6] for r in rows])
    print(f"\nSamples={len(y)}  positives(fault)={int(y.sum())} ({y.mean()*100:.1f}%)\n")

    import json, time
    report = {"generated_at": time.time(), "samples": int(len(y)),
              "positives": int(y.sum()), "detectors": {}, "per_fault": {}}
    for name, s in [("context (fused)", ctx), ("ae_score", ae), ("vib_score", vib), ("urgency", urg), ("clf P(fault)", pf)]:
        p, r, f = prf(y, s, 0.5)
        bp, br, bf, bt = best_f1(y, s)
        roc = auc_roc(y, s)
        print(f"  {name:<16} @0.5: P={p:.2f} R={r:.2f} F1={f:.2f} | best F1={bf:.2f}@{bt:.2f} | ROC-AUC={roc:.3f}")
        report["detectors"][name] = {"p": round(p, 3), "r": round(r, 3), "f1": round(f, 3),
                                     "best_f1": round(bf, 3), "best_thr": round(bt, 3), "roc_auc": round(roc, 3)}

    print("\n  Per-fault detection rate (clf P(fault)>=0.5):")
    for ftype in ANOMALY:
        idx = [i for i, rr in enumerate(rows) if rr[5] == ftype]
        if idx:
            det = float((pf[idx] >= 0.5).mean())
            print(f"    {ftype:<18} n={len(idx):4d}  detected={det:.2f}")
            report["per_fault"][ftype] = {"n": len(idx), "detected": round(det, 3)}

    out = os.path.join(ROOT, "models", "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report written → {out}")


if __name__ == "__main__":
    main()
