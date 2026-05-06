"""
MHARS LLM Alert Generator
==========================
Wraps Phi-3 Mini (4-bit GGUF) with a template fallback.
Import: from mhars.llm import AlertGenerator
"""

import os, time, threading, queue

# Phi-3 chat format — required for clean output without role tags
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


class AlertGenerator:
    """
    Generates plain language alerts from structured sensor + action data.

    If Phi-3 Mini GGUF is available, uses it. Falls back to templates
    so the pipeline always works regardless of LLM availability.
    """

    def __init__(self, model_path: str = None):
        self.llm      = None
        self.use_llm  = False

        if model_path and os.path.exists(model_path):
            self._load_llm(model_path)
        else:
            if model_path:
                print(f"  [LLM] Model not found at: {model_path}")
            print("  [LLM] Using template fallback")
            print("  [LLM] To enable Phi-3 Mini:")
            print("        pip install llama-cpp-python")
            print("        Download Phi-3-mini-4k-instruct-q4.gguf from")
            print("        huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")

        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None: break
            ctx, callback = item
            try:
                res = self.generate(ctx)
                if callback:
                    callback(res)
            except Exception as e:
                print(f"[LLM Async Error] {e}")
            finally:
                self._queue.task_done()

    def generate_async(self, context: dict, callback=None):
        """Queue the alert generation for background processing."""
        self._queue.put((context, callback))

    def wait_for_alerts(self):
        """Block until all async alerts have been generated and processed."""
        self._queue.join()

    def _load_llm(self, model_path: str):
        try:
            from llama_cpp import Llama
            print(f"  [LLM] Loading Phi-3 Mini from: {model_path}")
            self.llm = Llama(
                model_path = model_path,
                n_ctx      = 512,
                n_threads  = 4,
                verbose    = False,
            )
            self.use_llm = True
            print("  [LLM] Phi-3 Mini loaded ✓")
        except ImportError:
            print("  [LLM] llama-cpp-python not installed — using template fallback")
        except Exception as e:
            print(f"  [LLM] Load failed: {e} — using template fallback")

    def generate(self, context: dict) -> dict:
        machine_type   = context.get("machine_type",   "Machine")
        current_temp   = context.get("current_temp",   0.0)
        predicted_temp = context.get("predicted_temp", 0.0)
        anomaly_score  = context.get("anomaly_score",  0.0)
        action_name    = context.get("action_name",    "do-nothing")
        urgency        = context.get("urgency",        0.0)

        t0 = time.perf_counter()

        if self.use_llm:
            alert = self._llm_generate(
                machine_type, current_temp, predicted_temp,
                anomaly_score, action_name, urgency
            )
        else:
            alert = self._template_generate(
                machine_type, current_temp, predicted_temp,
                anomaly_score, action_name, urgency
            )

        return {
            "alert":      alert,
            "source":     "phi3-mini" if self.use_llm else "template",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def _llm_generate(self, machine_type, current_temp, predicted_temp,
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
            temperature = 0.3,
            stop        = ["<|end|>", "<|user|>", "<|system|>"],
        )
        text = response["choices"][0]["text"].strip()
        for tag in ["<|assistant|>", "<|user|>", "<|end|>"]:
            text = text.replace(tag, "").strip()
        return text

    def _template_generate(self, machine_type, current_temp, predicted_temp,
                            anomaly_score, action_name, urgency) -> str:
        if urgency >= 0.8:
            severity, tone = "CRITICAL", "Immediate intervention is required"
        elif urgency >= 0.5:
            severity, tone = "WARNING",  "Close monitoring is recommended"
        else:
            severity, tone = "NORMAL",   "No immediate action is needed"

        delta = predicted_temp - current_temp
        if delta > 5:
            trend = f"rising rapidly (+{delta:.1f}°C predicted in 10 min)"
        elif delta > 0:
            trend = f"slowly rising (+{delta:.1f}°C predicted in 10 min)"
        elif delta < -5:
            trend = f"cooling down ({delta:.1f}°C predicted in 10 min)"
        else:
            trend = "stable"

        action_map = {
            "do-nothing": "monitoring continues without intervention",
            "fan+":       "fan speed has been increased by 20%",
            "throttle":   "machine load has been reduced by 15%",
            "alert":      "a maintenance alert has been raised",
            "shutdown":   "emergency shutdown sequence has been initiated",
        }

        s1 = (f"[{severity}] The {machine_type} is at {current_temp:.1f}°C "
              f"(anomaly={anomaly_score:.2f}), temperature is {trend}.")
        s2 = f"{tone} — {action_map.get(action_name, action_name)}."
        return f"{s1} {s2}"