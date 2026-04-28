"""
MHARS — Stage 3: RL Router (Edge vs Cloud Dispatcher)
=======================================================
Decides where each data batch gets processed based on urgency.

Rule:
  urgency > 0.8  → Edge   (local inference, < 50 ms, critical situations)
  urgency < 0.4  → Cloud  (deep analysis, logging, retraining triggers)
  0.4–0.8        → Both   (run edge AND log to cloud for monitoring)

This is a rule-based router for now. In a future extension it can
itself become a small PPO agent that learns optimal routing based
on network latency and battery state — but that is not needed for
the current research contribution.
"""

import time
import numpy as np


EDGE_THRESHOLD  = 0.8   # above this → edge only
CLOUD_THRESHOLD = 0.4   # below this → cloud only


def route(urgency_score: float) -> dict:
    """
    Given the urgency score from attention fusion, decide the
    processing path and return a routing decision dict.
    """
    if urgency_score >= EDGE_THRESHOLD:
        path     = "edge"
        reason   = f"urgency {urgency_score:.3f} ≥ {EDGE_THRESHOLD} — critical, respond locally"
        latency  = "< 50 ms"
    elif urgency_score <= CLOUD_THRESHOLD:
        path     = "cloud"
        reason   = f"urgency {urgency_score:.3f} ≤ {CLOUD_THRESHOLD} — routine, deep analysis"
        latency  = "1–5 s"
    else:
        path     = "both"
        reason   = f"urgency {urgency_score:.3f} between thresholds — edge action + cloud log"
        latency  = "< 50 ms (edge) + async cloud"

    return {
        "path":          path,
        "urgency_score": urgency_score,
        "reason":        reason,
        "latency":       latency,
        "timestamp":     time.time(),
    }


def simulate_edge_inference(obs: np.ndarray, ppo_model) -> dict:
    """
    Run PPO model locally (edge path).
    Returns action + elapsed time in ms.
    """
    t0 = time.perf_counter()
    action, _ = ppo_model.predict(obs, deterministic=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    action_names = ["do-nothing", "fan+", "throttle", "alert", "shutdown"]
    return {
        "action":     int(action),
        "action_name": action_names[int(action)],
        "latency_ms": round(elapsed_ms, 2),
        "path":       "edge",
    }


def simulate_cloud_log(context_vector: dict, routing: dict):
    """
    Simulate sending data to cloud for logging and deep analysis.
    In real deployment this would be an MQTT publish or HTTP POST.
    """
    payload = {
        "context":  context_vector,
        "routing":  routing,
        "logged_at": time.time(),
    }
    # Placeholder — in Stage 4 this becomes paho-mqtt publish
    return {"status": "logged", "payload_size_bytes": len(str(payload))}


def run_tests():
    print("\n── RL Router Tests ───────────────────────────────────────────")

    test_cases = [
        (0.1,  "cloud",  "Very safe — deep analysis only"),
        (0.35, "cloud",  "Below cloud threshold"),
        (0.5,  "both",   "Middle — edge + cloud"),
        (0.75, "both",   "Elevated but not critical"),
        (0.85, "edge",   "Critical — edge only"),
        (0.99, "edge",   "Maximum urgency"),
    ]

    all_pass = True
    for urgency, expected_path, label in test_cases:
        result = route(urgency)
        status = "✓" if result["path"] == expected_path else "✗"
        if result["path"] != expected_path:
            all_pass = False
        print(f"  {status}  urgency={urgency:.2f}  path={result['path']:<6}  {label}")

    if all_pass:
        print("[PASS] RL Router routing logic correct\n")
    else:
        print("[FAIL] Some routing decisions incorrect\n")
    return all_pass


if __name__ == "__main__":
    run_tests()