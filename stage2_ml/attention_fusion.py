"""
MHARS — Stage 2: Attention-Based Multi-Modal Fusion
=====================================================
Combines outputs from LSTM, Autoencoder, and Isolation Forest
into a single Global Context Score that the PPO agent consumes.

Why attention over a static weighted sum:
  Static weights cannot handle sensor degradation. If a thermal
  camera is blocked by steam, its variance spikes → attention
  weight drops automatically → other sensors carry more weight.
  Validated by Vaswani et al. (2017) Attention Is All You Need.

In Stage 2 we have no CNN yet (that needs real thermal images).
The CNN slot is a placeholder that returns 0.5 until Stage 4
when the real hardware camera is connected.

Inputs:
  lstm_score   — normalized LSTM prediction deviation [0,1]
  ae_score     — Autoencoder reconstruction error score [0,1]
  if_score     — Isolation Forest anomaly score [0,1]
  cnn_score    — MobileNetV2 hotspot score [0,1] (placeholder=0.5)
  audio_score  — MFCC anomaly score [0,1] (placeholder=0.5)

Output:
  global_context_score — single float [0,1]
    0.0 = machine is healthy and stable
    1.0 = critical — immediate action required
"""

import numpy as np


def attention_weight(score: float, variance: float | None = None) -> float:
    """
    Compute attention weight for one modality.
    If variance is None, we assume the modality is fully reliable.
    High variance (unstable/noisy sensor) → low weight.
    """
    if variance is None or variance == 0:
        return 1.0
    # Weight inversely proportional to variance
    return 1.0 / (1.0 + variance)


def fuse(
    lstm_score:  float,
    ae_score:    float,
    if_score:    float,
    cnn_score:   float = 0.5,    # placeholder until Stage 4 camera
    audio_score: float = 0.5,    # placeholder until Stage 4 microphone
    lstm_var:    float | None = None,
    ae_var:      float | None = None,
    if_var:      float | None = None,
    cnn_var:     float | None = None,
    audio_var:   float | None = None,
) -> dict:
    """
    Compute the fused Global Context Score.

    Returns a dict with:
      global_context_score — main output to PPO agent
      weights              — attention weights (for logging/debugging)
      urgency_score        — fast urgency estimate for RL router
    """
    scores   = np.array([lstm_score, ae_score, if_score, cnn_score, audio_score],
                        dtype=np.float32)
    variances = [lstm_var, ae_var, if_var, cnn_var, audio_var]

    # Compute raw attention weights
    raw_weights = np.array([
        attention_weight(s, v) for s, v in zip(scores, variances)
    ], dtype=np.float32)

    # Normalize weights to sum to 1
    weights = raw_weights / (raw_weights.sum() + 1e-8)

    # Weighted sum → Global Context Score
    global_score = float(np.dot(weights, scores))
    global_score = np.clip(global_score, 0.0, 1.0)

    # Urgency = blend of top-2 scores (captures spikes better than average)
    top2 = np.sort(scores)[-2:]
    urgency = float(0.6 * top2[-1] + 0.4 * top2[-2])
    urgency = np.clip(urgency, 0.0, 1.0)

    return {
        "global_context_score": global_score,
        "urgency_score":        urgency,
        "weights": {
            "lstm":  float(weights[0]),
            "ae":    float(weights[1]),
            "if":    float(weights[2]),
            "cnn":   float(weights[3]),
            "audio": float(weights[4]),
        },
        "scores": {
            "lstm":  float(scores[0]),
            "ae":    float(scores[1]),
            "if":    float(scores[2]),
            "cnn":   float(scores[3]),
            "audio": float(scores[4]),
        }
    }


def interpret(global_score: float) -> str:
    """Human-readable interpretation of the context score."""
    if global_score < 0.3:
        return "HEALTHY — no action needed"
    elif global_score < 0.5:
        return "WATCH   — monitor closely"
    elif global_score < 0.7:
        return "WARN    — consider fan increase"
    elif global_score < 0.85:
        return "ALERT   — intervention recommended"
    else:
        return "CRITICAL — immediate action required"


def run_tests():
    print("\n── Attention Fusion Tests ────────────────────────────────────")

    # Test 1: All healthy
    result = fuse(0.1, 0.1, 0.1, 0.1, 0.1)
    assert result["global_context_score"] < 0.3, "Healthy test failed"
    print(f"  Test 1 (healthy):  score={result['global_context_score']:.3f}  {interpret(result['global_context_score'])}")

    # Test 2: LSTM spiking (thermal trend detected)
    result = fuse(0.9, 0.2, 0.2, 0.5, 0.5)
    print(f"  Test 2 (LSTM spike): score={result['global_context_score']:.3f}  {interpret(result['global_context_score'])}")

    # Test 3: Multiple sensors alarming
    result = fuse(0.85, 0.80, 0.75, 0.5, 0.5)
    assert result["global_context_score"] > 0.6, "Multi-alarm test failed"
    print(f"  Test 3 (multi-alarm): score={result['global_context_score']:.3f}  {interpret(result['global_context_score'])}")

    # Test 4: Camera blocked (high variance → low CNN weight)
    result_blocked = fuse(0.5, 0.5, 0.5, cnn_score=0.9, cnn_var=100.0)
    result_normal  = fuse(0.5, 0.5, 0.5, cnn_score=0.9, cnn_var=None)
    assert result_blocked["global_context_score"] < result_normal["global_context_score"], \
        "Blocked camera should reduce CNN's influence"
    print(f"  Test 4 (camera blocked): score={result_blocked['global_context_score']:.3f}  "
          f"(vs unblocked: {result_normal['global_context_score']:.3f}) → camera weight reduced ✓")

    print("[PASS] Attention fusion working correctly\n")


if __name__ == "__main__":
    run_tests()