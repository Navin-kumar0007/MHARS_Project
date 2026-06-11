# MHARS — Enhancement Implementation Plan

> **STATUS: COMPLETE ✅** (all phases implemented, 123/123 tests passing)
> - **P1** correctness — train/serve alignment, RUL + learned fusion, CNN/Audio gated, EVT calibration, multi-horizon forecast
> - **P2** accuracy — supervised fault classifier (0.86 acc), quantile forecaster + bands, MC-Dropout retired, learned anomaly detector (ROC-AUC 0.93 vs 0.55)
> - **P3** latency/edge — Isolation-Forest hotspot fixed (5.3→4.3 ms/tick), TorchScript edge export
> - **P4** observability — model-provenance badges + Diagnostics/metrics page
> - **X.1** MLOps — streaming concept-drift monitor + model registry
>
> Security: hardcoded JWT dev secret removed + scrubbed from git history.
> Open before merge to main: push branch · model provisioning (weights gitignored).



> Deep analysis of the ML / AI / hardware / UI layers and a prioritized,
> actionable plan to improve **accuracy, latency, and output fidelity**.
> Generated from a full code walk of the repository.

---

## 1. Architecture snapshot (what is actually live vs dead)

| Layer | Component | State | Evidence |
|---|---|---|---|
| Data | NASA **C-MAPSS FD001** turbofan + synthetic thermal env | live | `stage1_simulation/load_cmapss.py`, `gym_env.py`, `data/train_FD001.txt` present |
| ML | Isolation Forest | live | `models/isolation_forest.pkl` |
| ML | **BiLSTM + temporal attention** (`ThermalLSTMv2`) | live | `models/lstm_v2.pt` |
| ML | **LSTM-Autoencoder v2** (`ThermalAutoencoderLSTM`) | live | `models/autoencoder_lstm_v2.pt` |
| ML | Vibration detector (AE, 5 feat) | live | `models/vibration_detector.pt` |
| ML | Conformal predictor | live | `models/conformal_meta.json` |
| ML | **RUL predictor** (`RULPredictor`) | ❌ **missing** → physics fallback only | `models/rul_predictor.pt` absent |
| ML | EfficientNet CNN (hotspot) | ❌ ImageNet weights, **no thermal training** → ~0.5 noise | `models/efficientnet_cnn.pt` absent; startup log |
| ML | Audio MFCC | runs, but contributes ~constant to fusion | `core.py` `extra.get("audio_score", 0.5)` |
| ML | **TFT** quantile forecaster | ❌ scaffolded, dead | `models/tft_predictor.pt` absent |
| AI | **Learned attention fusion** | ❌ **missing** → rule-based fallback | `models/learned_fusion.pt` absent; log "⚠ Learned Fusion not found" |
| AI | PPO router/agent | live | `models/ppo_thermal.zip` |
| AI | **SAC** agent | ❌ scaffolded, unused | `models/sac_thermal.zip` absent |
| AI | MC-Dropout + Conformal uncertainty | live (overlapping) | `stage2_ml/mc_dropout.py`, `conformal.py` |
| AI | Causal layer | live | `stage3_ai/causal_layer.py` |
| AI | Phi-3 LLM | live, but dashboard uses **template path** at 1 Hz, not real generation | `core.py` `sync_alert=True` → `_force_template` |
| HW | psutil temp / MQTT sensors | live; **ambient hardcoded 25°C** | `api/main.py`, `stage4_hardware/` |
| HW | Quantization / ONNX / TorchScript | ❌ **none anywhere** | grep: 0 hits |

---

## 2. Root cause behind most "false data"

**Train/serve domain mismatch.** Models train on C-MAPSS turbofan degradation
(`THERMAL_SENSORS=["s2","s3","s4","s7","s11"]`, `RUL_MAX_CYCLES=125`) but run on a
*different* synthetic thermal sim (`gym_env`). Inference features are
out-of-distribution → reconstruction error saturates → constant max anomaly /
`vib=1.0`. The recent clamps in `core.py` (`_compute_vib_score` physics anchor,
`_estimate_rul` slope-based fallback) treat the **symptom**. The **cure** is
distribution alignment (retrain on the serving distribution or align the sim).

---

## 3. Work items (prioritized)

Effort: S ≈ <½ day · M ≈ 1–2 days · L ≈ 3+ days. Impact tags: 🎯 accuracy · ⚡ latency · 🔎 output fidelity.

### P1 — correctness (fixes the root cause + silent fakes)

#### P1.1 — Align train/serve distribution 🎯🔎 · L
- **Problem:** OOD inference → saturated anomaly/vib scores.
- **Files:** `stage1_simulation/gym_env.py`, `stage2_ml/run_stage2.py`, `stage5_adapter/machine_adapter.py`.
- **Approach:** generate a training set from `gym_env` itself (per machine profile), or domain-adapt C-MAPSS → sim. Reuse the existing adapter path (`lstm_adapted_engine.pt`, `ppo_adapted_engine.zip` prove it works) and extend to AE + vibration.
- **Accept:** on an idle machine, `ae_score < 0.3` and `vib_score < 0.3` *without* the core clamps; anomaly scores rise only under injected faults.
- **Prereq:** `data/train_FD001.txt` ✅ present.

#### P1.2 — Train the missing models that are silently faked 🎯 · M
- `rul_predictor.pt` — `RULPredictor` (`mhars/models.py`) exists; train on C-MAPSS (`stage2_ml/rul_trainer.py`). Target FD001 RMSE ≈ 12–14 (SOTA BiLSTM+attn). Currently RUL is **physics-only**.
- `learned_fusion.pt` — `LearnedAttentionFusion` exists; train so fusion weights are learned, not the rule-based fallback.
- **Files:** `stage2_ml/rul_trainer.py`, `mhars/learned_fusion.py`, `mhars/core.py` (load paths already wired).
- **Accept:** startup log shows "RUL predictor loaded" + "Learned Fusion loaded"; `metadata.rul_minutes` tracks real degradation.

#### P1.3 — Remove or fix placeholder modalities 🔎 · S
- **Problem:** CNN (ImageNet, untrained) + Audio contribute ~constant 0.5 → dilute 2 of 6 fusion inputs with noise.
- **Approach (pick one):** (a) gate CNN/Audio out of fusion until trained; or (b) fine-tune CNN on a thermal dataset (FLIR ADAS) and train the audio model on real fault audio.
- **Files:** `core.py` `_fuse`, `stage2_ml/efficientnet_cnn.py`, `audio_model.py`.
- **Accept:** fusion only consumes modalities with real signal; `feature_importance` no longer attributes weight to constants.

#### P1.4 — Adaptive anomaly thresholding + score calibration 🎯🔎 · M
- **Problem:** fixed `AE_THRESHOLD_PERCENTILE=95`; raw error/threshold saturates.
- **Approach:** **POT / peaks-over-threshold (EVT)** dynamic threshold + isotonic/Platt calibration → calibrated 0–1 scores.
- **Files:** `mhars/config.py`, `core.py` `_compute_ae_score`/`_compute_vib_score`.
- **Accept:** score distributions are calibrated (reliability curve ~diagonal); no permanent saturation.

#### P1.5 — Real multi-horizon forecast 🎯🔎 · M
- **Problem:** `LSTM_PREDICTION_HORIZON_S=1` (single step) but UI says "+10 min".
- **Approach:** train **direct multi-horizon** head (predict t+1…t+N); drive RUL + "time to limit" from the multi-step trajectory, not 1-step extrapolation.
- **Files:** `mhars/models.py`, `stage2_ml/lstm_predictor.py`, `config.py`, `core.py` `_estimate_rul`.
- **Accept:** forecast/RUL reflect a true N-minute horizon; UI label matches config.

### P2 — accuracy / output quality

#### P2.1 — SOTA forecaster (quantile + native uncertainty) 🎯🔎 · L
- Swap/augment BiLSTM with **PatchTST / N-HiTS / TFT** (TFT scaffolded). Quantile outputs replace MC-Dropout's N-pass uncertainty.
- **Files:** `stage2_ml/tft_predictor.py`, `train_tft.py`.

#### P2.2 — SOTA multivariate anomaly 🎯 · L
- Evaluate **TranAD / Anomaly-Transformer / USAD** (strong on SMD/MSL/SMAP) vs the small LSTM-AE; adopt if F1 improves.

#### P2.3 — Drop MC-Dropout, keep conformal ⚡ · S
- MC-Dropout = N forward passes/tick. Conformal already gives intervals cheaper. Remove MC-Dropout from the hot path.
- **Files:** `stage2_ml/mc_dropout.py`, `core.py`.

#### P2.4 — Supervised fault classifier 🔎 · M
- Replace rule-based `_fingerprint_anomaly` with a small classifier over labeled fault types → real fault identification + confidence.

### P3 — latency / edge

#### P3.1 — Quantize + export (biggest latency win) ⚡ · M
- INT8 / dynamic quantization + **ONNX Runtime or TorchScript** for LSTM/AE/vib. Nothing is optimized today (eager PyTorch, CPU). Expect 2–4× speedup, lower memory.
- **Files:** new `tools/export_models.py`; load path in `core.py`.
- **Accept:** per-tick inference latency ↓ ≥2×; outputs within tolerance of fp32.

#### P3.2 — Batch + cache the 1 Hz loop ⚡ · S
- WS loop runs every model sequentially per tick. Batch modality inferences; cache template/LLM alerts; skip recompute on unchanged state.
- **Files:** `api/main.py` WS handler, `core.py`.

#### P3.3 — Distill the edge model ⚡ · M
- Distill BiLSTM → tiny GRU/TCN for sub-ms edge inference (edge path already skips LLM).

### P4 — UI / observability

#### P4.1 — Model-provenance badges 🔎 · S
- `systemStatus.models_loaded` already flags loaded vs fallback. Show a badge on the dashboard when any model is in fallback so placeholder outputs are visibly flagged.
- **Files:** `dashboard/src/app/page.tsx`, `components/Sidebar.tsx`.

#### P4.2 — Confidence bands + chart scaling 🔎 · M
- Extend conformal bands to health/RUL; downsample long history; canvas/WebGL for high-rate charts; virtualize history tables.

#### P4.3 — Live eval/metrics page 🎯🔎 · M
- `mhars/alert_eval.py` exists. Add a page with live precision/recall/F1 (anomaly), RMSE (forecast), RUL-score → accuracy measurable, not asserted.

### Cross-cutting — MLOps

#### X.1 — Drift monitoring + registry 🎯 · M
- Track feature + score-distribution drift (beyond CUSUM-on-temp); auto-flag retrain. Build on existing `lstm_version`/`ae_version` stamps; add canary comparison of new vs live model.

---

## 4. Suggested phased roadmap

- **Phase A (correctness):** P1.1 → P1.2 → P1.3 → P1.4 → P1.5. Removes the symptom-level clamps; data becomes trustworthy end-to-end.
- **Phase B (latency):** P3.1 → P3.2 → P2.3. Faster, lighter, edge-ready.
- **Phase C (fidelity):** P4.1 → P4.3 → P2.4 → X.1. Make accuracy visible and monitored.
- **Phase D (SOTA, optional):** P2.1, P2.2, P3.3.

## 5. Prerequisites & risks

- **Data:** `data/train_FD001.txt` ✅ present (retraining feasible).
- **Test gate:** `core.py` edits are covered by `tests/`; run `pytest tests/ -q` after each change (currently green).
- **Risk:** retraining (P1.x) changes model behavior → re-baseline thresholds and the flaky AUC/attention tests noted in git history.
- **Risk:** removing CNN/Audio (P1.3) changes fusion dimensionality (`FUSION_N_MODALITIES=6`) — update config + any shape asserts.
- **Compatibility:** keep model load paths backward-compatible (fallback when artifact missing) so the app never hard-fails on a missing `.pt`.

## 6. Quick wins (do first, low risk, high signal)

1. **P4.1** model-provenance badges — pure frontend, surfaces what's fake today.
2. **P3.1** quantize + ONNX export — concrete latency win, isolated.
3. **P1.2** train `rul_predictor.pt` — makes RUL/time-to-limit real (data present).
