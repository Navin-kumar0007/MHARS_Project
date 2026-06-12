"""
R2 — Maintenance-knowledge retriever (RAG).

A small curated maintenance corpus + a lightweight TF-IDF retriever (no heavy
embedding dependency — reuses scikit-learn, already required). The diagnostic
agent retrieves the passages most relevant to the current fault/symptoms and
cites them, so the LLM's narrative is grounded in referenceable knowledge
rather than free-form generation.
"""
from __future__ import annotations

# Curated maintenance manual passages. `id` is the citation handle; `tags`
# improve retrieval for the MHARS fault vocabulary.
KNOWLEDGE = [
    {"id": "MAN-THERM-01", "title": "Thermal runaway / heat spike",
     "tags": "temperature_spike power_surge thermal runaway overheat heat spike sudden rise critical shutdown",
     "text": "A sudden, large temperature rise indicates thermal runaway, typically from a "
             "workload surge, blocked airflow, or a power-supply fault. Immediate mitigation: "
             "reduce load (throttle) and raise cooling (fan+). If the temperature is within one "
             "step of the critical limit, trigger an emergency shutdown to prevent permanent damage."},
    {"id": "MAN-BEAR-02", "title": "Bearing wear / mechanical degradation",
     "tags": "bearing_wear vibration mechanical wear friction grinding rpm imbalance lubrication",
     "text": "Rising vibration with gradual heat buildup signals bearing wear or shaft imbalance. "
             "Schedule mechanical inspection; check lubrication and alignment. Bearing faults "
             "progress slowly — plan maintenance within the predicted remaining useful life rather "
             "than shutting down immediately."},
    {"id": "MAN-FAN-03", "title": "Cooling fan blockage / airflow failure",
     "tags": "fan_blockage cooling airflow blocked dust filter heat dissipation failure fan",
     "text": "Sustained heating without a load increase points to degraded heat dissipation — a "
             "blocked or failing cooling fan or clogged filter. Increase fan speed as a stopgap, "
             "then clear obstructions / replace the fan. Persistent blockage will eventually breach "
             "the safe limit, so monitor time-to-limit closely."},
    {"id": "MAN-SENS-04", "title": "Sensor drift / noisy readings",
     "tags": "sensor_drift noise faulty sensor calibration spurious erratic readings glitch",
     "text": "Erratic, high-variance readings with no corresponding load change usually indicate a "
             "faulty or drifting sensor rather than a real thermal event. Cross-check against a "
             "second sensor, recalibrate, and avoid aggressive control actions driven by a single "
             "noisy channel."},
    {"id": "MAN-PWR-05", "title": "Power surge / electrical fault",
     "tags": "power_surge electrical fault voltage spike surge instant overheat transient",
     "text": "An instantaneous extreme temperature jump with an electrical signature suggests a "
             "power surge or supply fault. Protect the system: throttle or shut down, inspect the "
             "power delivery, and verify protection circuitry before resuming."},
    {"id": "MAN-ACT-06", "title": "Control actions reference",
     "tags": "action throttle fan shutdown do-nothing cooling control response policy",
     "text": "Control actions, least to most aggressive: do-nothing (monitor), fan+ (increase "
             "cooling, low cost), throttle (reduce workload, moderate impact), shutdown / "
             "emergency-shutdown (halt to prevent damage). Prefer the least aggressive action that "
             "keeps the predicted trajectory below the safe limit."},
    {"id": "MAN-RUL-07", "title": "Remaining useful life & maintenance windows",
     "tags": "rul remaining useful life maintenance window schedule degradation longevity plan",
     "text": "Remaining useful life (RUL) estimates time to the safe limit at the current trend. "
             "Use it to schedule maintenance proactively: long RUL → routine window; short RUL → "
             "expedite service; near-zero RUL with rising temperature → act now."},
]


class RagStore:
    def __init__(self):
        self._docs = KNOWLEDGE
        self._vec = None
        self._mat = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english")
            corpus = [f"{d['title']} {d['tags']} {d['text']}" for d in self._docs]
            self._mat = self._vec.fit_transform(corpus)
        except Exception:
            self._vec = None  # fall back to keyword overlap

    def retrieve(self, query: str, k: int = 3) -> list:
        """Return the top-k most relevant passages: [{id,title,text,score}]."""
        if self._vec is not None:
            import numpy as np
            qv = self._vec.transform([query])
            sims = (self._mat @ qv.T).toarray().ravel()
            order = np.argsort(-sims)[:k]
            return [{**{key: self._docs[i][key] for key in ("id", "title", "text")},
                     "score": round(float(sims[i]), 3)} for i in order if sims[i] > 0] or \
                   [{**{key: self._docs[order[0]][key] for key in ("id", "title", "text")}, "score": 0.0}]
        # keyword-overlap fallback
        q = set(query.lower().split())
        scored = sorted(self._docs, key=lambda d: -len(q & set((d["tags"] + " " + d["title"]).lower().split())))
        return [{**{key: d[key] for key in ("id", "title", "text")}, "score": 0.0} for d in scored[:k]]
