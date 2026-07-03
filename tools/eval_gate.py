"""
Eval gate — regression guard for the anomaly detector (industry-readiness).
============================================================================
Runs the labelled fault-injection eval (reusing tools/eval_anomaly.py), then
compares ROC-AUC of the key detectors against a committed baseline. Fails
(exit 1) if any detector regresses beyond tolerance — so a bad model / bad
change is blocked in CI before it ships.

First run bootstraps the baseline (tools/eval_baseline.json) from the current
models and passes. Commit that baseline; later runs gate against it.

Run:      python3 tools/eval_gate.py            # gate (CI)
          python3 tools/eval_gate.py --update   # re-baseline to current
Env:      MHARS_EVAL_STEPS (default 600/machine), MHARS_EVAL_TOL (default 0.05)
"""
import os, sys, json, time, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from mhars.config import Config
import tools.eval_anomaly as ev

BASELINE = os.path.join(ROOT, "tools", "eval_baseline.json")
REPORT = os.path.join(ROOT, "results", "anomaly_eval.json")
# Detectors that gate the build (name → key in the collected rows tuple index).
GATED = {"context (fused)": 1, "clf P(fault)": 6}


def run_eval(steps: int) -> dict:
    rows = []
    for mid in sorted(Config.MACHINE_PROFILES.keys()):
        rows.extend(ev.collect(mid, steps))
        print(f"  [machine {mid}] collected {steps} steps")
    y = np.array([r[0] for r in rows])
    out = {"generated_at": time.time(), "samples": int(len(y)),
           "positives": int(y.sum()), "steps_per_machine": steps, "detectors": {}}
    for name, idx in GATED.items():
        s = np.array([r[idx] for r in rows], dtype=float)
        out["detectors"][name] = {"roc_auc": round(float(ev.auc_roc(y, s)), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="re-baseline to current models")
    args = ap.parse_args()

    steps = int(os.environ.get("MHARS_EVAL_STEPS", "600"))
    tol = float(os.environ.get("MHARS_EVAL_TOL", "0.05"))

    print(f"Running eval gate ({steps} steps/machine, tol={tol}) …")
    report = run_eval(steps)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report → {REPORT}")
    for name, d in report["detectors"].items():
        print(f"    {name:<16} ROC-AUC = {d['roc_auc']:.4f}")

    # Bootstrap or update the baseline.
    if args.update or not os.path.exists(BASELINE):
        with open(BASELINE, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Baseline {'updated' if args.update else 'bootstrapped'} → {BASELINE}")
        print("  (commit this file; future runs gate against it)")
        return 0

    with open(BASELINE) as f:
        base = json.load(f)

    regressed = []
    for name in GATED:
        cur = report["detectors"].get(name, {}).get("roc_auc", 0.0)
        ref = base["detectors"].get(name, {}).get("roc_auc", 0.0)
        delta = cur - ref
        status = "OK" if delta >= -tol else "REGRESSED"
        print(f"    {name:<16} base={ref:.4f} now={cur:.4f} Δ={delta:+.4f}  [{status}]")
        if delta < -tol:
            regressed.append((name, ref, cur, delta))

    if regressed:
        print("\n✗ EVAL GATE FAILED — detector regression beyond tolerance:")
        for name, ref, cur, delta in regressed:
            print(f"    {name}: {ref:.4f} → {cur:.4f} (Δ{delta:+.4f}, tol -{tol})")
        return 1
    print("\n✓ EVAL GATE PASSED — no detector regressed beyond tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
