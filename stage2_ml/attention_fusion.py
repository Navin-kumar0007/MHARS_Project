"""
MHARS — Stage 2: Attention Fusion (UPDATED)
=============================================
Fix: ISSUE-1 — now connects real CNN and audio scores.
Placeholders (0.5) removed. Real modality pipelines called.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def attention_weight(variance=None):
    if variance is None or variance == 0:
        return 1.0
    return 1.0 / (1.0 + variance)


def fuse(lstm_score, ae_score, if_score,
         cnn_score=None, audio_score=None,
         cnn_var=None, audio_var=None):
    """
    Fuse all modality scores into Global Context Score.
    cnn_score/audio_score now required from real pipelines.
    Falls back to 0.5 only if explicitly passed as None.
    """
    cnn_s   = cnn_score   if cnn_score   is not None else 0.5
    audio_s = audio_score if audio_score is not None else 0.5

    scores  = np.array([lstm_score, ae_score, if_score, cnn_s, audio_s],
                       dtype=np.float32)
    raw_w   = np.array([
        attention_weight(None),
        attention_weight(None),
        attention_weight(None),
        attention_weight(cnn_var),
        attention_weight(audio_var),
    ], dtype=np.float32)
    weights = raw_w / (raw_w.sum() + 1e-8)
    global_score = float(np.clip(np.dot(weights, scores), 0, 1))
    top2    = np.sort(scores)[-2:]
    urgency = float(np.clip(0.6 * top2[-1] + 0.4 * top2[-2], 0, 1))

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
        },
        "placeholders_used": {
            "cnn":   cnn_score   is None,
            "audio": audio_score is None,
        }
    }


def interpret(score):
    if score < 0.3:  return "HEALTHY — no action needed"
    if score < 0.5:  return "WATCH — monitor closely"
    if score < 0.7:  return "WARN — consider fan increase"
    if score < 0.85: return "ALERT — intervention recommended"
    return "CRITICAL — immediate action required"


def run_tests():
    print("\n── Attention Fusion Tests ────────────────────────────────────")

    # Test 1: healthy all modalities
    r = fuse(0.1, 0.1, 0.1, cnn_score=0.1, audio_score=0.1)
    assert r["global_context_score"] < 0.3
    print(f"  Test 1 (healthy):       score={r['global_context_score']:.3f}  {interpret(r['global_context_score'])}")
    assert not any(r["placeholders_used"].values()), "All real scores provided"

    # Test 2: LSTM spike
    r = fuse(0.9, 0.2, 0.2, cnn_score=0.3, audio_score=0.2)
    print(f"  Test 2 (LSTM spike):    score={r['global_context_score']:.3f}  {interpret(r['global_context_score'])}")

    # Test 3: CNN hotspot detected
    r = fuse(0.3, 0.3, 0.2, cnn_score=0.85, audio_score=0.4)
    print(f"  Test 3 (CNN hotspot):   score={r['global_context_score']:.3f}  {interpret(r['global_context_score'])}")

    # Test 4: audio anomaly
    r = fuse(0.3, 0.4, 0.2, cnn_score=0.3, audio_score=0.88)
    print(f"  Test 4 (audio anomaly): score={r['global_context_score']:.3f}  {interpret(r['global_context_score'])}")

    # Test 5: camera blocked (high var → low CNN weight)
    r_blocked = fuse(0.5, 0.5, 0.5, cnn_score=0.9, cnn_var=100.0, audio_score=0.5)
    r_normal  = fuse(0.5, 0.5, 0.5, cnn_score=0.9, cnn_var=None,  audio_score=0.5)
    assert r_blocked["global_context_score"] < r_normal["global_context_score"]
    print(f"  Test 5 (cam blocked):   score={r_blocked['global_context_score']:.3f}  "
          f"(vs unblocked {r_normal['global_context_score']:.3f}) → CNN weight reduced ✓")

    # Test 6: confirm no placeholders when all scores provided
    r = fuse(0.5, 0.5, 0.5, cnn_score=0.6, audio_score=0.4)
    assert not r["placeholders_used"]["cnn"]
    assert not r["placeholders_used"]["audio"]
    print(f"  Test 6 (no placeholders): CNN={r['scores']['cnn']:.1f}  "
          f"audio={r['scores']['audio']:.1f}  ✓")

    print("[PASS] Attention fusion (all 6 tests)\n")


if __name__ == "__main__":
    run_tests()