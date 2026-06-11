"""
P1.4 — Fit POT/EVT anomaly-score calibrators for the AE and vibration models.

Reuses the trained models (no retraining): captures normal reconstruction
errors over a fresh gym_env rollout through the real pipeline, fits an
AnomalyCalibrator to each, and writes the calibration params into the model
meta JSONs so core.py can produce calibrated 0..1 scores.

Run:  python3 tools/fit_calibrators.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from mhars.models import ThermalAutoencoderLSTM, VibrationDetector
from mhars.config import Config
from mhars.anomaly_calibrator import AnomalyCalibrator
from tools.retrain_serving_dist import collect, STEPS_PER_MACHINE

MODELS = os.path.join(ROOT, "models")


def main():
    all_windows, all_vib = [], []
    for mid in sorted(Config.MACHINE_PROFILES.keys()):
        w, _tn, vf = collect(mid, STEPS_PER_MACHINE)
        all_windows.extend(w)
        all_vib.extend(vf)
    X = np.array(all_windows, dtype=np.float32)
    Vib = np.array(all_vib, dtype=np.float32)
    print(f"\nCaptured AE windows={X.shape}, vib={Vib.shape}")

    # ── AE calibrator ────────────────────────────────────────────────────────
    ae_meta_path = os.path.join(MODELS, "autoencoder_lstm_v2_meta.json")
    with open(ae_meta_path) as f:
        ae_meta = json.load(f)
    ae = ThermalAutoencoderLSTM(input_size=ae_meta.get("input_size", 5),
                                hidden_size=ae_meta.get("hidden_size", 32), seq_len=12)
    ae.load_state_dict(torch.load(os.path.join(MODELS, "autoencoder_lstm_v2.pt"), map_location="cpu"))
    ae.eval()
    with torch.no_grad():
        ae_err = ae.reconstruction_error(torch.tensor(X)).numpy()
    ae_cal = AnomalyCalibrator.fit(ae_err)
    ae_meta["calib"] = ae_cal.to_dict()
    with open(ae_meta_path, "w") as f:
        json.dump(ae_meta, f, indent=2)
    print(f"AE calib: p50={ae_cal.p50:.2e} t={ae_cal.t:.2e} z_q={ae_cal.z_q:.2e} xi={ae_cal.xi:.3f}")
    print(f"   normal score median={ae_cal.score(np.median(ae_err)):.2f} "
          f"p99={ae_cal.score(np.percentile(ae_err,99)):.2f} max={ae_cal.score(ae_err.max()):.2f}")

    # ── Vibration calibrator ──────────────────────────────────────────────────
    vib_meta_path = os.path.join(MODELS, "vibration_detector_meta.json")
    with open(vib_meta_path) as f:
        vib_meta = json.load(f)
    mean = np.array(vib_meta["mean"], dtype=np.float32)
    std = np.array(vib_meta["std"], dtype=np.float32)
    vib = VibrationDetector(n_features=vib_meta.get("n_features", 5))
    vib.load_state_dict(torch.load(os.path.join(MODELS, "vibration_detector.pt"), map_location="cpu"))
    vib.eval()
    Vn = ((Vib - mean) / (std + 1e-8)).astype(np.float32)
    with torch.no_grad():
        vib_err = vib.reconstruction_error(torch.tensor(Vn)).numpy()
    vib_cal = AnomalyCalibrator.fit(vib_err)
    vib_meta["calib"] = vib_cal.to_dict()
    with open(vib_meta_path, "w") as f:
        json.dump(vib_meta, f, indent=2)
    print(f"Vib calib: p50={vib_cal.p50:.2e} t={vib_cal.t:.2e} z_q={vib_cal.z_q:.2e} xi={vib_cal.xi:.3f}")
    print(f"   normal score median={vib_cal.score(np.median(vib_err)):.2f} "
          f"p99={vib_cal.score(np.percentile(vib_err,99)):.2f} max={vib_cal.score(vib_err.max()):.2f}")

    print("\n✓ Calibrators fitted and written to model meta.")


if __name__ == "__main__":
    main()
