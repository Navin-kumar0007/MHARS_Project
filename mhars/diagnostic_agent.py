"""
R2 — Agentic LLM diagnostician.

Upgrades the alert layer from one-line templates to a grounded diagnostic agent
that USES TOOLS over the live pipeline state:
  1. RAG          → retrieve relevant maintenance-manual passages (with citations)
  2. digital twin → simulate each candidate action's thermal trajectory (what-if)
  3. causal layer → physics-based root-cause hypothesis
then synthesises a diagnosis (root cause + evidence + citations + recommended
action with its simulated outcome + maintenance plan) and a natural-language
narrative — grounded in the retrieved evidence and the twin's simulations rather
than free-form generation. The tool orchestration is deterministic (reliable on
device); the LLM, when present, only writes the final prose.
"""
from __future__ import annotations

from mhars.rag_store import RagStore

_ACTION_ORDER = {"do-nothing": 0, "fan+": 1, "throttle": 2, "shutdown": 3, "emergency-shutdown": 4}


class DiagnosticAgent:
    CANDIDATE_ACTIONS = ["do-nothing", "fan+", "throttle", "shutdown"]

    def __init__(self, mhars):
        self.m = mhars
        self.rag = RagStore()

    def diagnose(self, state: dict) -> dict:
        trace = []
        fault = state.get("fault_type") or "Normal Operations"
        temp = float(state.get("current_temp", 0.0))
        load = float(state.get("load_pct", 0.5))
        urgency = float(state.get("urgency", 0.0))
        rul = state.get("rul_minutes")
        top = state.get("top_contributor") or ""
        safe = float(self.m.profile["safe_max"])
        crit = float(self.m.profile["critical"])
        fan = 1.0 if state.get("action") in ("fan+", "throttle") else 0.0

        # ── Tool 1: retrieve manuals (RAG) ───────────────────────────────────
        query = f"{fault} {top} temperature {temp:.0f} urgency {urgency:.2f}"
        docs = self.rag.retrieve(query, k=3)
        trace.append({"step": "retrieve_manuals", "tool": "RAG", "result": [d["id"] for d in docs]})

        # ── Tool 2: digital-twin what-if per candidate action ────────────────
        what_if, recommended = [], None
        if getattr(self.m, "_digital_twin", None) is not None:
            for act in self.CANDIDATE_ACTIONS:
                traj = self.m._digital_twin.simulate_what_if(temp, load, fan, [act] * 12, steps_per_action=1)
                peak = round(float(max(traj)), 1)
                final = round(float(traj[-1]), 1)  # predicted temp after the action
                what_if.append({"action": act, "peak_c": peak, "final_c": final,
                                "breach": peak >= crit, "safe": final < safe})
            trace.append({"step": "simulate_what_if", "tool": "digital_twin",
                          "result": {w["action"]: w["final_c"] for w in what_if}})
            # Recommend the least-aggressive action whose predicted temp stays safe
            # (else the one that avoids a breach, else the coolest outcome).
            safe_opts = [w for w in what_if if w["safe"] and not w["breach"]]
            ok_opts = [w for w in what_if if not w["breach"]]
            pool = safe_opts or ok_opts or what_if
            recommended = sorted(pool, key=lambda w: _ACTION_ORDER.get(w["action"], 9))[0]

        # ── Tool 3: causal root-cause ────────────────────────────────────────
        causal = None
        if getattr(self.m, "_causal_layer", None) is not None:
            try:
                causal = self.m._causal_layer.analyze(temp, load, fan)
                trace.append({"step": "causal_analysis", "tool": "causal_layer",
                              "result": causal.get("root_cause_hypothesis")})
            except Exception:
                causal = None

        root_cause = (causal or {}).get("root_cause_hypothesis") or fault
        severity = "critical" if urgency >= 0.8 else "warning" if urgency >= 0.5 else "normal"
        narrative, used_llm = self._narrate(state, fault, root_cause, recommended, docs, rul, safe, crit, severity)

        # R3 — causal counterfactual RCA (computed in the pipeline) is the
        # strongest causal signal; surface it + its prescribed minimal intervention.
        rca = state.get("causal_rca")
        if rca and rca.get("root_cause_variable") not in (None, "none"):
            trace.append({"step": "counterfactual_rca", "tool": "digital_twin(SCM)",
                          "result": rca.get("root_cause_variable")})

        return {
            "fault": fault,
            "root_cause": root_cause,
            "causal_rca": rca,
            "severity": severity,
            "evidence": {
                "temperature_c": round(temp, 1), "safe_max_c": safe, "critical_c": crit,
                "urgency": round(urgency, 2), "rul_minutes": rul,
                "top_contributor": top, "anomaly_probability": state.get("anomaly_probability"),
                "causal_fault_probability": (causal or {}).get("fault_probability"),
            },
            "citations": docs,
            "what_if": what_if,
            "recommended_action": recommended,
            "maintenance_plan": state.get("maintenance_plan"),
            "narrative": narrative,
            "llm_grounded": used_llm,
            "trace": trace,
        }

    # ── Narrative synthesis: grounded LLM if available, else grounded template ──
    def _narrate(self, state, fault, root_cause, recommended, docs, rul, safe, crit, severity):
        cite = docs[0]["id"] if docs else "—"
        guidance = docs[0]["text"] if docs else ""
        rec = recommended["action"] if recommended else "monitor"
        rec_peak = f", projected to settle near {recommended['final_c']}°C" if recommended else ""
        rul_txt = (f" Estimated time-to-limit is ~{round(rul)} min." if isinstance(rul, (int, float)) else
                   " The machine is thermally stable.")

        # Grounded template (always available, deterministic, cites a source).
        template = (
            f"Diagnosis ({severity}): {root_cause}. "
            f"Recommended action: {rec}{rec_peak}.{rul_txt} "
            f"Per maintenance reference [{cite}]: {guidance}"
        )

        # If an on-device LLM is loaded, let it write grounded prose from the same
        # evidence; otherwise return the template.
        gen = getattr(self.m, "_llm_gen", None)
        if gen is not None and getattr(gen, "use_llm", False):
            try:
                ctx = {
                    "machine_type": state.get("machine_type", "machine"),
                    "current_temp": float(state.get("current_temp", 0.0)),
                    "predicted_temp": float(state.get("lstm_prediction", state.get("current_temp", 0.0))),
                    "anomaly_score": float(state.get("anomaly_probability", 0.0) or 0.0),
                    "action_name": recommended["action"] if recommended else "do-nothing",
                    "urgency": float(state.get("urgency", 0.0)),
                    "load_pct": float(state.get("load_pct", 0.5)),
                    "dT_dt": 0.0,
                    "causal_reasoning": f"{root_cause}; manual [{cite}]: {guidance[:160]}",
                }
                res = gen.generate(ctx)
                txt = res.get("alert")
                if txt:
                    return f"{txt} [grounded: {cite}]", True
            except Exception:
                pass
        return template, False
