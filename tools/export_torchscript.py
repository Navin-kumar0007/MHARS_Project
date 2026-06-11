"""
P3.1 — Export TorchScript edge artifacts.

Profiling showed the live 1 Hz path is ~4.3 ms/tick (dominated by the Isolation
Forest, already optimised) — so torch quantization is unnecessary for the
server. This tool exists for EDGE deployment: it TorchScripts the torch models
into portable, dependency-light artifacts (models/edge/*.ts.pt), verifies
numerical parity with the eager models, and benchmarks eager vs scripted so the
gain is measured, not assumed. Not wired into the live loop.

Run:  python3 tools/export_torchscript.py
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from mhars.config import Config
from mhars.models import (ThermalLSTMv2, ThermalAutoencoderLSTM, VibrationDetector,
                          RULPredictor, FaultClassifier)

OUT = os.path.join(ROOT, "models", "edge")
os.makedirs(OUT, exist_ok=True)


def _bench(fn, x, n=300):
    for _ in range(20): fn(x)
    t0 = time.perf_counter()
    for _ in range(n): fn(x)
    return (time.perf_counter() - t0) / n * 1e3  # ms


def export(name, model, example):
    model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
        scripted = torch.jit.freeze(scripted)
        ref = model(example)
        got = scripted(example)
    max_err = float((ref - got).abs().max())
    path = os.path.join(OUT, f"{name}.ts.pt")
    scripted.save(path)
    eager_ms = _bench(lambda x: model(x), example)
    ts_ms = _bench(lambda x: scripted(x), example)
    print(f"  {name:<16} parity_max_err={max_err:.2e}  eager={eager_ms:.3f}ms  ts={ts_ms:.3f}ms  ({eager_ms/ts_ms:.2f}x)")
    return {"max_err": max_err, "eager_ms": round(eager_ms, 4), "ts_ms": round(ts_ms, 4)}


def main():
    M = os.path.join(ROOT, "models")
    report = {}

    # Forecaster (multi-horizon quantile head)
    ck = torch.load(Config.LSTM_V2, map_location="cpu")
    h = ck["lstm.weight_ih_l0"].shape[0] // 4; isz = ck["lstm.weight_ih_l0"].shape[1]
    oh = ck["linear.weight"].shape[0]
    m = ThermalLSTMv2(input_size=isz, hidden_size=h, output_horizon=oh); m.load_state_dict(ck)
    report["lstm_v2"] = export("lstm_v2", m, torch.rand(1, 12, 5))

    # LSTM-AE
    am = json.load(open(Config.AUTOENCODER_V2_META))
    m = ThermalAutoencoderLSTM(input_size=am.get("input_size", 5), hidden_size=am.get("hidden_size", 32), seq_len=12)
    m.load_state_dict(torch.load(Config.AUTOENCODER_V2, map_location="cpu"))
    report["autoencoder_lstm_v2"] = export("autoencoder_lstm_v2", m, torch.rand(1, 12, 5))

    # Vibration
    vm = json.load(open(Config.VIBRATION_META))
    m = VibrationDetector(n_features=vm.get("n_features", 5))
    m.load_state_dict(torch.load(Config.VIBRATION_DETECTOR, map_location="cpu"))
    report["vibration_detector"] = export("vibration_detector", m, torch.rand(1, 5))

    # RUL
    if os.path.exists(Config.RUL_MODEL_V2):
        m = RULPredictor(); m.load_state_dict(torch.load(Config.RUL_MODEL_V2, map_location="cpu"))
        report["rul_predictor_v2"] = export("rul_predictor_v2", m, torch.rand(1, 12, 5))

    # Fault classifier
    if os.path.exists(Config.FAULT_CLASSIFIER):
        fm = json.load(open(Config.FAULT_CLASSIFIER_META))
        m = FaultClassifier(n_features=fm.get("n_features", 10), n_classes=fm.get("n_classes", 6))
        m.load_state_dict(torch.load(Config.FAULT_CLASSIFIER, map_location="cpu"))
        report["fault_classifier"] = export("fault_classifier", m, torch.rand(1, fm.get("n_features", 10)))

    json.dump(report, open(os.path.join(OUT, "export_report.json"), "w"), indent=2)
    worst = max(r["max_err"] for r in report.values())
    print(f"\n✓ {len(report)} models exported → models/edge/  (worst parity err {worst:.2e})")


if __name__ == "__main__":
    main()
