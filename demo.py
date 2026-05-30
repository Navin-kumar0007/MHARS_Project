"""
MHARS Framework Demo
=====================
Run this from the MHARS_Project/ root folder:
    python demo.py

Shows the full framework API — import, configure, run, dashboard.
Requires Stage 2 and Stage 3 models to be trained first.
"""

import os, sys
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--synthetic", action="store_true", help="Enable synthetic mode")
parser.add_argument("--no-interactive", action="store_true", help="Skip interactive prompts (for CI/CD)")
args = parser.parse_args()

if args.synthetic:
    os.environ["MHARS_SYNTHETIC_MODE"] = "true"
    print("Running in SYNTHETIC mode...")

from mhars import MHARS, Dashboard, Config

# ── Demo 1: Single reading ─────────────────────────────────────────────────────
print("=" * 58)
print("  Demo 1 — Single reading (no LLM)")
print("=" * 58)

system = MHARS(machine_type_id=0, verbose=False)

test_temps = [45.0, 58.0, 72.0, 84.0, 92.0]
print(f"\n  {'Temp':>6}  {'Action':>12}  {'Route':>6}  {'Urgency':>8}  {'Anomaly':>8}")
print(f"  {'─'*6}  {'─'*12}  {'─'*6}  {'─'*8}  {'─'*8}")

for temp in test_temps:
    r = system.run(temp)
    print(f"  {r.current_temp:>5.1f}°C  {r.action:>12}  "
          f"{r.route:>6}  {r.urgency:>8.3f}  {r.anomaly_score:>8.3f}")

# ── Demo 2: With LLM ───────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  Demo 2 — With Phi-3 Mini LLM alerts")
print("=" * 58)

llm_path = os.path.join("models", "Phi-3-mini-4k-instruct-q4.gguf")
if os.path.exists(llm_path):
    system_llm = MHARS(machine_type_id=1, llm_path=llm_path, verbose=False)
    system_llm.reset()

    scenarios = [
        (45.0, "Normal motor operation"),
        (72.0, "Motor warming up under load"),
        (88.0, "Motor approaching threshold"),
    ]
    for temp, label in scenarios:
        r = system_llm.run(temp)
        print(f"\n  [{label}] {temp}°C → {r.action}")
        print(f"  Initial Alert Status: {r.alert}")
        print(f"  Source: {r.llm_source}  |  Pipeline Latency: {r.latency_ms:.0f} ms")
        
        # Wait up to 5 seconds for the async alert to finish generating
        import time
        for _ in range(50):
            if r.llm_source != "async":
                break
            time.sleep(0.1)
else:
    print(f"  Phi-3 Mini not found at {llm_path}")
    print("  Run with template fallback:")
    system2 = MHARS(machine_type_id=1, verbose=False)
    r = system2.run(88.0)
    print(f"\n  88.0°C → {r.action}")
    print(f"  {r.alert}")

# ── Demo 3: Sequence ───────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  Demo 3 — Temperature sequence (simulate degradation)")
print("=" * 58)

system3 = MHARS(machine_type_id=2, verbose=False)  # Server
degrading = [35, 38, 41, 45, 50, 57, 63, 69, 74, 79, 83, 88]
results   = system3.run_sequence(degrading)

print(f"\n  Simulating server heating up over 12 readings:\n")
for r in results:
    icon = "🔥" if r.is_critical() else ("⚠️ " if r.urgency > 0.5 else "✓ ")
    print(f"  {icon} {r.current_temp:>5.1f}°C  →  {r.action:<12}  "
          f"urgency={r.urgency:.2f}  route={r.route}")

# ── Demo 4: Dashboard (10 seconds) ───────────────────────────────────────────
if not args.no_interactive:
    print("\n" + "=" * 58)
    print("  Demo 4 — Live Dashboard (10 seconds)")
    print("  Press Ctrl+C to stop early")
    print("=" * 58)
    input("\n  Press Enter to start dashboard...")

    system4 = MHARS(machine_type_id=0, verbose=False)
    dash    = Dashboard(system4, refresh_hz=1)
    dash.start(source="simulation", duration_s=10)
else:
    print("\n  [--no-interactive] Skipping Dashboard demo.")

print("\nDemo complete.")
print("To run the full system with your own temperature readings:")
print("  from mhars import MHARS")
print("  system = MHARS(machine_type_id=0)")
print("  result = system.run(your_temp_reading)")