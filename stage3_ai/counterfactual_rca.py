"""
R3 — Causal counterfactual root-cause analysis (RCA) + prescriptive action.

The digital twin's physics IS a structural causal model (SCM): temperature is a
deterministic function of load, cooling, ambient and an exogenous heat term. So
we do Pearl-style interventions — do(X = baseline) — directly on the twin instead
of learning a causal graph from noisy synthetic data.

Procedure:
  1. Recover the exogenous fault-heat Q_ext that explains the OBSERVED dT/dt
     (the part the twin's load/cooling physics cannot account for).
  2. Forward-simulate the factual trajectory → factual peak.
  3. For each intervenable cause (exogenous fault, load, cooling, ambient),
     counterfactually reset it to baseline, re-simulate → counterfactual peak.
  4. Causal contribution of X = factual_peak − counterfactual_peak(X)  (°C of the
     predicted rise that X is responsible for). Root cause = argmax contribution.
  5. Prescribe the minimal intervention that addresses the root cause.

This yields a *causal* attribution (counterfactual effect) — distinct from the
correlational feature-importance XAI — and a prescriptive action grounded in the
physics, closing detect → causally-diagnose → prescribe.
"""
from __future__ import annotations


class CounterfactualRCA:
    def __init__(self, profile: dict):
        self.heat_rate = float(profile.get("heat_rate", 2.0))
        self.conv = float(profile.get("conv_coeff", 0.05))
        self.mass = float(profile.get("thermal_mass_J_K", 25.0))
        self.amb = 25.0
        self.safe = float(profile["safe_max"])
        self.crit = float(profile["critical"])
        self.idle_load = 0.25  # baseline load when "load" is counterfactually removed

    def _q_out(self, temp, fan, ambient):
        eff_h = self.conv * (1.0 + 2.0 * fan)
        return eff_h * (temp - ambient) * 100.0

    def _q_in(self, load):
        return self.heat_rate * load * 100.0

    def _forward(self, temp, load, fan, ambient, q_ext, steps=12):
        peak = temp
        for _ in range(steps):
            dT = (self._q_in(load) + q_ext - self._q_out(temp, fan, ambient)) / self.mass
            temp = min(self.crit + 10.0, max(15.0, temp + dT))
            peak = max(peak, temp)
        return peak

    def analyze(self, temp: float, load: float, fan: float, dT_dt: float | None,
                ambient: float = 25.0) -> dict:
        """Return causal contributions, root cause, and a prescribed action."""
        # 1. Recover exogenous heat explaining the observed rate of change.
        if dT_dt is None:
            dT_dt = 0.0
        q_ext = self.mass * dT_dt - self._q_in(load) + self._q_out(temp, fan, ambient)
        q_ext = max(0.0, q_ext)  # only a heating exogenous term is a "fault" cause

        # 2. Counterfactual PEAK per cause — supporting "what-if" view.
        factual = self._forward(temp, load, fan, ambient, q_ext)
        cfs = {
            "exogenous_fault": self._forward(temp, load, fan, ambient, 0.0),
            "high_load":       self._forward(temp, self.idle_load, fan, ambient, q_ext),
            "weak_cooling":    self._forward(temp, load, 1.0, ambient, q_ext),
        }

        # 3. Causal attribution via the instantaneous heat-imbalance decomposition
        #    (cleaner than peak-deltas once the machine is already hot): split the
        #    heating drivers into fault heat, load heat above baseline, and the
        #    cooling deficit (heat that more cooling could still remove).
        load_excess = max(0.0, self._q_in(load) - self._q_in(self.idle_load))
        cooling_deficit = max(0.0, self._q_out(temp, 1.0, ambient) - self._q_out(temp, fan, ambient))
        drivers = {
            "exogenous_fault": q_ext,
            "high_load": load_excess,
            "weak_cooling": cooling_deficit,
        }
        total = sum(drivers.values()) + 1e-9
        contrib_pct = {k: round(100.0 * v / total) for k, v in drivers.items()}

        heating = (q_ext + self._q_in(load)) > self._q_out(temp, fan, ambient) or temp >= self.safe
        # Root CAUSE = the dominant heat SOURCE (fault vs load). Cooling deficit is
        # a mitigation lever, shown in the contributions but not a "cause".
        sources = {"exogenous_fault": q_ext, "high_load": load_excess}
        root = max(sources, key=sources.get) if heating and max(sources.values()) > 1.0 else "none"

        # Prescription: address the cause (throttle), but prefer the less-aggressive
        # fan+ when extra cooling alone keeps the predicted peak safe.
        if root == "none":
            prescription = "do-nothing"
        elif cfs["weak_cooling"] < self.safe:
            prescription = "fan+"
        else:
            prescription = "throttle"
        # Escalate if even max cooling + idle load can't keep the peak below critical.
        if root != "none" and min(cfs.values()) >= self.crit:
            prescription = "shutdown"

        return {
            "factual_peak_c": round(factual, 1),
            "counterfactual_peak_c": {k: round(v, 1) for k, v in cfs.items()},
            "causal_contributions": contrib_pct,
            "root_cause_variable": root,
            "prescribed_action": prescription,
            "q_ext_estimate": round(q_ext, 1),
        }
