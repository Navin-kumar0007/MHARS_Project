# MHARS — Research Roadmap (Frontier Phase)

> Next-phase plan to push MHARS beyond a strong *integrated* system toward
> genuinely novel, publishable capability. Grounded in the current codebase
> (sim → ML → fusion → RL → LLM → edge → federated → observability) and the
> 2023–2026 literature. Builds on the completed P1–P4 + X.1 work.
>
> **Thesis of novelty:** the individual models are mid-tier; the opportunity is
> to (a) swap perception to a *foundation-model* backbone and (b) **invent at
> the seams** — close a *detect → causally diagnose → counterfactually prescribe
> → act → lifelong-adapt* loop on the edge, which no single deployed system does.

---

## 0. Where we stand vs SOTA (baseline for the gap)

| Capability | SOTA reference points | MHARS today | Gap |
|---|---|---|---|
| Forecasting | PatchTST, N-HiTS, iTransformer; foundation: **TimesFM, Chronos, Moirai, Lag-Llama, MOMENT** | BiLSTM+attn, multi-horizon quantile (per-machine) | no zero-shot; OOD on unseen machines / real-CPU |
| Anomaly | Anomaly-Transformer, TranAD, USAD, DCdetector | LSTM-AE + EVT + supervised classifier | sim-trained → OOD in Live mode |
| RUL | CNN+attn / Transformer (~12 RMSE FD001) | learned (~21 RMSE) + physics fallback | accuracy + no uncertainty on RUL |
| Uncertainty | conformal, deep ensembles, evidential DL | conformal + EVT + quantiles | strong; not yet fed into control |
| Control | safe-RL, distributional RL, shielding | PPO router/agent | not uncertainty-aware, no safety proof |
| Reasoning | Time-LLM, GPT4TS, RAG, LLM agents | Phi-3 template alerts | not grounded/agentic |
| Causality | PCMCI, DYNOTEARS, CIRCA, counterfactual RCA | rule-based causal layer | no causal discovery / counterfactuals |
| Lifelong | continual learning, TTA, online conformal | drift monitor (flags only) | does not adapt; cloud retrain only |

---

## Phase R1 — Foundation-model perception backbone  🎯 (do first)

**Goal:** zero-shot forecasting + residual anomaly that generalises to unseen
machines and real hardware **without per-machine training** — directly fixing the
Live-mode OOD detector problem.

- **Approach:** integrate a small TS foundation model (Chronos-Bolt / TimesFM /
  Moirai-small, CPU-friendly). Forecast = model output; anomaly score =
  standardized forecast residual vs its native prediction interval; keep EVT
  calibration on the residual.
- **Files:** new `stage2_ml/foundation_forecaster.py`; wire as a `_lstm_version`
  branch in `mhars/core.py`; `tools/eval_anomaly.py` to compare vs current.
- **Refs:** TimesFM, Chronos, Moirai, Lag-Llama, MOMENT, Timer.
- **Novelty:** zero-shot, label-free PdM perception that transfers across machine
  types and to real hardware — rare in deployed twins.
- **Accept:** Live-mode detector behaves (no false 100%); anomaly ROC-AUC ≥
  current 0.93 on the eval harness with **zero per-machine training**.
- **Effort:** M. **Risk:** model size/latency on edge → use *-bolt/-small + ONNX.

## Phase R2 — Agentic LLM diagnostician (RAG + digital-twin tool-use)  🎯

**Goal:** replace template alerts with an on-device **agent** that diagnoses,
finds root cause, and writes a maintenance plan grounded in evidence.

- **Approach:** the LLM receives structured pipeline state, **retrieves from a
  maintenance-manual corpus (RAG)**, and **calls the digital twin as a tool**
  (`simulate_what_if`) to test hypotheses, then returns diagnosis + root cause +
  plan **with citations**. ReAct-style tool loop; keep Phi-3 (or swap to a
  small tool-use model).
- **Files:** `mhars/llm.py` → agent loop; new `mhars/rag_store.py`; expose twin
  as a tool; `api/main.py` endpoint for the agent transcript.
- **Refs:** ReAct, Toolformer, Time-LLM, GPT4TS, RAG.
- **Novelty:** small on-device agent grounded in a *live* multimodal PdM loop
  with twin tool-use + manual citations.
- **Accept:** for each injected fault, the agent names the correct root cause +
  cites a manual section + proposes a valid action, verified on the 5 fault types.
- **Effort:** M–L.

## Phase R3 — Causal counterfactual RCA → prescriptive RL (close the loop)  🎯

**Goal:** go from "what's wrong" to "the single change that prevents failure",
then execute it.

- **Approach:** **causal discovery** (PCMCI/DYNOTEARS) on the sensor streams to
  build the causal graph; **counterfactual RCA** ("which variable, if held at
  baseline, removes the predicted fault?") via the digital twin; feed the
  **minimal sufficient intervention** to the RL agent.
- **Files:** `stage3_ai/causal_layer.py` (extend), new
  `stage3_ai/counterfactual_rca.py`; hook into RL action selection in `core.py`.
- **Refs:** PCMCI, DYNOTEARS, CIRCA, RCD, causal RL.
- **Novelty:** online **detect → causal diagnose → counterfactual prescribe →
  act** loop — a publishable contribution.
- **Accept:** on injected faults, counterfactual RCA identifies the injected
  cause ≥ X% of the time and the prescribed action reduces predicted RUL loss.
- **Effort:** L.

## Phase R4 — Label-free lifelong edge adaptation

**Goal:** each machine's twin adapts to its own drift **forever, without labels**.

- **Approach:** drift monitor (X.1) **triggers** self-supervised on-device
  fine-tuning (masked reconstruction / TS2Vec contrastive) + automatic conformal
  + EVT recalibration; guarded by a canary/rollback if metrics regress.
- **Files:** `mhars/drift_monitor.py` (trigger), new
  `tools/online_adapt.py`; registry records each adaptation.
- **Refs:** TS2Vec, TF-C, test-time adaptation, online conformal (Gibbs &
  Candès), EWC/replay.
- **Novelty:** label-free, drift-triggered lifelong adaptation closed-loop on edge.
- **Accept:** after induced drift, scores recalibrate automatically (normal
  median returns ~0) with no human labels; canary prevents regressions.
- **Effort:** M–L.

## Phase R5 — Uncertainty-aware safe RL + sustainable multi-objective control  🛡️🌱

**Goal:** trustworthy, provably-safe, energy/carbon-aware control.

- **Approach:** condition the RL policy on conformal/quantile uncertainty + RUL
  distribution (**distributional RL**); wrap in a **formal safety shield** (never
  exceed critical); extend the reward to a **Pareto multi-objective** trade-off
  (safety / energy / longevity / carbon) and surface the Pareto front on the
  dashboard. Router uses *value-of-information*, not just urgency.
- **Files:** `stage3_ai/ppo_agent.py` / `sac_agent.py`, reward in `config.py`,
  new `stage3_ai/safety_shield.py`; Analytics page for the Pareto front.
- **Refs:** distributional RL (QR-DQN/IQN), safe RL / shielding, multi-objective RL.
- **Novelty:** uncertainty-conditioned, formally-shielded, carbon-aware PdM control.
- **Accept:** shield provably blocks any action sequence that would breach
  critical in the twin; dashboard shows the safety/energy trade-off.
- **Effort:** M–L.

## Phase R6 — Actionable counterfactual XAI (quick win)

**Goal:** *"increase cooling now → failure prob 80% → 12%"* explanations.

- **Approach:** query the digital twin for counterfactual outcomes of each
  candidate action; render on the "Why this decision" panel.
- **Files:** `core.py` (twin queries), `dashboard/.../page.tsx`.
- **Effort:** S–M (twin already exists).

---

## The moonshot (the unique contribution)

Integrate **R1 + R2 + R3 + R4**:

> A **self-adapting causal digital twin** — foundation-model perception
> (zero-shot, label-free), an on-device **LLM agent** doing causal root-cause +
> counterfactual prescription grounded in the twin and manuals, with **label-free
> lifelong edge adaptation** and conformal safety guarantees — closed-loop.

Each piece exists in isolation in the literature; the **closed-loop integration
on the edge** is not something any single deployed system does. MHARS is uniquely
positioned because the sim, twin, fusion, RL, LLM, drift, and federated
scaffolding already exist.

---

## Recommended order & rationale

1. **R1** — fixes a real bug (Live OOD) *and* is a frontier upgrade. Highest ROI.
2. **R2** — biggest demo/“wow”; strong thesis story.
3. **R3** — the core research novelty (causal counterfactual loop).
4. **R6** — cheap, high-perceived-value (reuses twin).
5. **R4**, **R5** — deepen lifelong + trustworthy-control claims.

## Prerequisites & risks
- **Compute/edge:** foundation models + agent must stay CPU/edge-viable → prefer
  *-bolt/-small variants + ONNX/quantization (P3 export path already exists).
- **Data:** RAG needs a maintenance-manual corpus; causal discovery needs
  multivariate streams (synthesize from the twin initially).
- **Eval:** every phase gates on the existing `tools/eval_anomaly.py` harness +
  `pytest tests/` (currently 123 green) + live screenshot verification.
- **Scope honesty:** "no model can do this" = the *integration*, not any single
  module; frame contributions as the closed-loop system + the specific seam
  mechanisms (zero-shot residual anomaly, counterfactual-RCA→RL, label-free
  lifelong edge adaptation).

## References (verify newest 2025–26 variants when web access returns)
TimesFM · Chronos · Moirai · Lag-Llama · MOMENT · Timer · UniTS ·
Anomaly-Transformer · TranAD · USAD · DCdetector · TimesNet ·
PatchTST · N-HiTS · iTransformer ·
PCMCI · DYNOTEARS · CIRCA · RCD ·
ReAct · Toolformer · Time-LLM · GPT4TS · RAG ·
TS2Vec · TF-C · online conformal (Gibbs & Candès) ·
distributional RL (QR-DQN/IQN) · safe-RL shielding · multi-objective RL.
