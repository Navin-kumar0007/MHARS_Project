"""
MHARS — Stage 3: LLM Alert Generator
=======================================
Converts structured PPO output into plain language alerts
for maintenance technicians.

Uses Phi-3 Mini (4-bit GGUF) running locally via llama-cpp-python.
No API calls, no internet, no data leaves your machine.

If llama-cpp-python or the model file is not installed yet,
falls back to a template-based generator so the pipeline
keeps running during development.

Install (do this when ready for the real LLM):
    pip install llama-cpp-python
    Download: Phi-3-mini-4k-instruct-Q4_K_M.gguf
    From:     https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

Reference: Abdin et al. (2024) Phi-3 Technical Report.
           https://arxiv.org/abs/2404.14219
"""

import os
import time


# ── Prompt template ────────────────────────────────────────────────────────────
# Filled at runtime with live sensor data + PPO decision.
# Phi-3 uses a specific chat format: <|user|>...<|end|><|assistant|>
# The model completes from the <|assistant|> token onward.
PROMPT_TEMPLATE = """<|user|>
You are a maintenance assistant for industrial machines. Write exactly two plain sentences for a technician. Be direct and specific. No bullet points, no lists.

Situation:
- Machine: {machine_type}
- Current temperature: {current_temp:.1f}°C
- Predicted temperature in 10 min: {predicted_temp:.1f}°C
- Anomaly score: {anomaly_score:.2f} (0=normal, 1=critical)
- Action taken by system: {action_name}
- Urgency: {urgency:.2f}

Write two sentences only.<|end|>
<|assistant|>"""


# ── LLM wrapper ────────────────────────────────────────────────────────────────
class AlertGenerator:
    """
    Wraps Phi-3 Mini (GGUF) with a fallback template generator.
    The pipeline works either way — real LLM improves language quality
    but the template output is sufficient for testing.
    """

    def __init__(self, model_path: str = None):
        self.llm      = None
        self.use_llm  = False
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self._load_llm(model_path)
        else:
            if model_path:
                print(f"  [LLM] Model not found at: {model_path}")
            print("  [LLM] Using template fallback (no LLM loaded)")
            print("  [LLM] To enable real LLM:")
            print("        pip install llama-cpp-python")
            print("        Download Phi-3-mini-4k-instruct-Q4_K_M.gguf")
            print("        from huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")

    def _load_llm(self, model_path: str):
        try:
            from llama_cpp import Llama
            print(f"  [LLM] Loading Phi-3 Mini from: {model_path}")
            self.llm = Llama(
                model_path    = model_path,
                n_ctx         = 512,    # context window — small = fast
                n_threads     = 4,      # CPU threads
                verbose       = False,
            )
            self.use_llm = True
            print("  [LLM] Phi-3 Mini loaded ✓")
        except ImportError:
            print("  [LLM] llama-cpp-python not installed — using template fallback")
        except Exception as e:
            print(f"  [LLM] Failed to load model: {e} — using template fallback")

    def generate(self, context: dict) -> dict:
        """
        Generate a plain language alert from the context dict.
        Returns alert text + generation time in ms.
        """
        machine_type   = context.get("machine_type", "Unknown machine")
        current_temp   = context.get("current_temp",   0.0)
        predicted_temp = context.get("predicted_temp", 0.0)
        anomaly_score  = context.get("anomaly_score",  0.0)
        action_name    = context.get("action_name",    "do-nothing")
        urgency        = context.get("urgency",        0.0)

        t0 = time.perf_counter()

        if self.use_llm:
            alert = self._generate_llm(
                machine_type, current_temp, predicted_temp,
                anomaly_score, action_name, urgency
            )
        else:
            alert = self._generate_template(
                machine_type, current_temp, predicted_temp,
                anomaly_score, action_name, urgency
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "alert":       alert,
            "source":      "phi3-mini" if self.use_llm else "template",
            "latency_ms":  round(elapsed_ms, 1),
            "context":     context,
        }

    def _generate_llm(self, machine_type, current_temp, predicted_temp,
                      anomaly_score, action_name, urgency) -> str:
        prompt = PROMPT_TEMPLATE.format(
            machine_type   = machine_type,
            current_temp   = current_temp,
            predicted_temp = predicted_temp,
            anomaly_score  = anomaly_score,
            action_name    = action_name,
            urgency        = urgency,
        )
        response = self.llm(
            prompt,
            max_tokens  = 120,
            temperature = 0.3,    # low temperature = more factual, less creative
            stop        = ["<|end|>", "<|user|>", "<|system|>"],
        )
        text = response["choices"][0]["text"].strip()
        # Remove any role tags the model might echo back
        for tag in ["<|assistant|>", "<|user|>", "<|end|>"]:
            text = text.replace(tag, "").strip()
        return text

    def _generate_template(self, machine_type, current_temp, predicted_temp,
                           anomaly_score, action_name, urgency) -> str:
        """
        Rule-based template alert — used when no LLM is available.
        Produces readable alerts that are good enough for testing.
        """
        # Determine severity
        if urgency >= 0.8:
            severity = "CRITICAL"
            tone     = "Immediate intervention is required"
        elif urgency >= 0.5:
            severity = "WARNING"
            tone     = "Close monitoring is recommended"
        else:
            severity = "NORMAL"
            tone     = "No immediate action is needed"

        # Build trend description
        delta = predicted_temp - current_temp
        if delta > 5:
            trend = f"rising rapidly (predicted +{delta:.1f}°C in 10 min)"
        elif delta > 0:
            trend = f"slowly rising (predicted +{delta:.1f}°C in 10 min)"
        elif delta < -5:
            trend = f"cooling down (predicted {delta:.1f}°C in 10 min)"
        else:
            trend = "stable"

        # Build action description
        action_map = {
            "do-nothing": "monitoring continues without intervention",
            "fan+":       "fan speed has been increased by 20%",
            "throttle":   "machine load has been reduced by 15%",
            "alert":      "a maintenance alert has been raised",
            "shutdown":   "emergency shutdown sequence has been initiated",
        }
        action_desc = action_map.get(action_name, action_name)

        sentence1 = (
            f"[{severity}] The {machine_type} is currently at {current_temp:.1f}°C "
            f"with an anomaly score of {anomaly_score:.2f}, and temperature is {trend}."
        )
        sentence2 = f"{tone} — {action_desc}."

        return f"{sentence1} {sentence2}"


# ── Quick test ─────────────────────────────────────────────────────────────────
def run_tests(model_path: str = None):
    print("\n── LLM Alert Generator Tests ─────────────────────────────────")

    generator = AlertGenerator(model_path=model_path)

    test_contexts = [
        {
            "machine_type": "CPU",
            "current_temp": 45.0, "predicted_temp": 48.0,
            "anomaly_score": 0.1,  "action_name": "do-nothing",
            "urgency": 0.15,
        },
        {
            "machine_type": "Motor",
            "current_temp": 72.0, "predicted_temp": 79.0,
            "anomaly_score": 0.6,  "action_name": "fan+",
            "urgency": 0.65,
        },
        {
            "machine_type": "Server",
            "current_temp": 88.0, "predicted_temp": 95.0,
            "anomaly_score": 0.92, "action_name": "shutdown",
            "urgency": 0.91,
        },
    ]

    all_pass = True
    for ctx in test_contexts:
        result = generator.generate(ctx)
        print(f"\n  Machine : {ctx['machine_type']}  "
              f"Temp: {ctx['current_temp']}°C  "
              f"Action: {ctx['action_name']}")
        print(f"  Alert   : {result['alert']}")
        print(f"  Source  : {result['source']}  |  Latency: {result['latency_ms']} ms")
        if not result["alert"]:
            all_pass = False

    status = "PASS" if all_pass else "FAIL"
    print(f"\n[{status}] LLM Alert Generator\n")
    return all_pass, generator


if __name__ == "__main__":
    # To test with the real Phi-3 Mini, set the path:
    # run_tests(model_path="/path/to/Phi-3-mini-4k-instruct-Q4_K_M.gguf")
    run_tests(model_path=None)