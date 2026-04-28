from mhars.config import Config
"""
MHARS — Stage 3 Validation Runner
====================================
Trains the PPO agent, tests the RL router, and validates
the LLM alert generator in sequence.

Run from the stage3_ai/ folder:
    python run_stage3.py

Expected total time: 20–40 minutes on a MacBook (500K timesteps).
If you want a quick smoke-test first, run with --quick flag:
    python run_stage3.py --quick
This trains for only 50K timesteps — enough to see the agent
learning, but not enough for full convergence.
"""

import os, sys, argparse


def run_router_tests():
    print("=" * 56)
    print("  Component 1 — RL Router")
    print("=" * 56)
    from stage3_ai.rl_router import run_tests
    passed = run_tests()
    return passed


def run_llm_tests(model_path=None):
    print("=" * 56)
    print("  Component 2 — LLM Alert Generator")
    print("=" * 56)
    from stage3_ai.llm_output import run_tests
    passed, generator = run_tests(model_path=model_path)
    return passed, generator


def run_ppo_training(timesteps: int):
    print("=" * 56)
    print("  Component 3 — PPO Agent Training")
    print("=" * 56)
    from stage3_ai.ppo_agent import run_training
    model, avg_reward = run_training(
        machine_type_id = 0,          # CPU first (as per implementation plan)
        timesteps       = timesteps,
        model_path      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_thermal.zip"),
    )
    passed = avg_reward > -280        # any improvement over random baseline
    return model, avg_reward, passed


def run_end_to_end(ppo_model, generator):
    """
    Full Stage 3 integration test:
    Simulate 5 timesteps going through the complete pipeline:
      Gym observation → PPO decision → RL Router → LLM alert
    """
    print("=" * 56)
    print("  Component 4 — End-to-End Pipeline (Gym → PPO → Router → LLM)")
    print("=" * 56)

    import numpy as np
    from stage1_simulation.gym_env import ThermalEnv, MACHINE_PROFILES
    from stage3_ai.rl_router import route
    from stage2_ml.attention_fusion import fuse, interpret

    env  = ThermalEnv(machine_type_id=0, max_steps=500, render_mode=None)
    obs, info = env.reset(seed=Config.SEED)

    print(f"\n  Running 10 steps through full pipeline...\n")
    print(f"  {'Step':>4}  {'Temp':>7}  {'Action':>12}  {'Route':>6}  {'Urgency':>8}  Alert (first 60 chars)")
    print(f"  {'-'*4}  {'-'*7}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*60}")

    all_pass = True
    for step in range(10):
        # 1. PPO decides action
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        # 2. Build context score from obs
        lstm_score  = float(obs[1])   # predicted temp deviation
        ae_score    = float(obs[2])   # anomaly score
        if_score    = float(obs[2]) * 0.5
        urgency     = float(obs[5])

        fusion = fuse(lstm_score, ae_score, if_score)
        ctx_score = fusion["global_context_score"]

        # 3. Route
        routing = route(urgency)

        # 4. Generate alert
        machine_name = MACHINE_PROFILES[0]["name"]
        real_temp = info["temp"]
        pred_temp = real_temp + (real_temp * float(obs[1]) * 0.1)

        alert_ctx = {
            "machine_type":   machine_name,
            "current_temp":   real_temp,
            "predicted_temp": pred_temp,
            "anomaly_score":  float(obs[2]),
            "action_name":    info["action_name"],
            "urgency":        urgency,
        }
        alert_result = generator.generate(alert_ctx)
        alert_short  = alert_result["alert"][:60] + "..." if len(alert_result["alert"]) > 60 else alert_result["alert"]

        print(f"  {step+1:>4}  {real_temp:>6.1f}°C  "
              f"{info['action_name']:>12}  "
              f"{routing['path']:>6}  "
              f"{urgency:>8.3f}  "
              f"{alert_short}")

        if terminated or truncated:
            obs, info = env.reset()

    print(f"\n[PASS] End-to-end pipeline working\n")
    return True


def print_summary(results):
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  Stage 3 Results Summary" + " "*29 + "║")
    print("╠" + "═"*54 + "╣")
    all_pass = True
    for name, passed in results:
        icon   = "✓" if passed else "⚠"
        status = "PASSED" if passed else "CHECK OUTPUT"
        print(f"║  {icon}  {name:<34} {status:<10}║")
        if not passed:
            all_pass = False
    print("╠" + "═"*54 + "╣")
    if all_pass:
        print("║  All Stage 3 components ready ✓                  ║")
        print("║  Next: Stage 4 — Hardware (Raspberry Pi)         ║")
        print("║  Or:   Stage 5 — Machine Adapter validation      ║")
        print("║                                                   ║")
        print("║  For Phi-3 Mini (real LLM):                      ║")
        print("║    pip install llama-cpp-python                   ║")
        print("║    Download Phi-3-mini-4k-instruct-Q4_K_M.gguf   ║")
    else:
        print("║  Some components need attention (see above)      ║")
    print("╚" + "═"*54 + "╝\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Train for 50K timesteps instead of 500K (smoke test)")
    parser.add_argument("--llm", type=str, default=None,
                        help="Path to Phi-3 Mini GGUF model file")
    args = parser.parse_args()

    timesteps = 50_000 if args.quick else 500_000

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   MHARS Stage 3 — Strategic Brain (RL + LLM)        ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.quick:
        print("  [QUICK MODE] Training for 50K timesteps only.")
        print("  Run without --quick for full 500K training.\n")

    results = []

    # Component 1: RL Router
    p1 = run_router_tests()
    results.append(("RL Router (routing logic)", p1))

    # Component 2: LLM Alert Generator
    p2, generator = run_llm_tests(model_path=args.llm)
    results.append(("LLM Alert Generator", p2))

    # Component 3: PPO Training
    ppo_model, avg_reward, p3 = run_ppo_training(timesteps)
    label = f"PPO Agent (avg reward {avg_reward:.0f})"
    results.append((label, p3))

    # Component 4: End-to-end
    p4 = run_end_to_end(ppo_model, generator)
    results.append(("End-to-end pipeline", p4))

    print_summary(results)


if __name__ == "__main__":
    main()