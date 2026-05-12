"""
MHARS — Issue 7: LLM Alert Quality Evaluation Framework
=========================================================
Automated evaluation of Phi-3 Mini generated alerts against
a factual rubric. Compares LLM output to template baseline
to ensure the AI alert is actually better than the template.

Rubric dimensions (each scored 0–1):
  1. Factual accuracy — does the alert state the correct temperature?
  2. Specificity     — does it name the machine type and trend?
  3. Actionability   — does it recommend a concrete next step?
  4. Conciseness     — is it within 2 sentences?
  5. No hallucination— does it avoid inventing nonexistent readings?

Usage:
    from mhars.alert_eval import AlertEvaluator
    evaluator = AlertEvaluator()
    score = evaluator.evaluate(alert_text, context_dict)
    print(score)  # {'total': 0.85, 'factual': 1.0, ...}
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class AlertScore:
    """Score breakdown for one alert evaluation."""
    factual:        float = 0.0   # 0–1: temperature mentioned and correct?
    specificity:    float = 0.0   # 0–1: machine type + trend mentioned?
    actionability:  float = 0.0   # 0–1: concrete recommendation present?
    conciseness:    float = 0.0   # 0–1: within 2 sentences?
    no_hallucination: float = 0.0 # 0–1: no invented data?
    total:          float = 0.0   # weighted average

    def to_dict(self) -> dict:
        return {
            "factual": round(self.factual, 2),
            "specificity": round(self.specificity, 2),
            "actionability": round(self.actionability, 2),
            "conciseness": round(self.conciseness, 2),
            "no_hallucination": round(self.no_hallucination, 2),
            "total": round(self.total, 2),
        }


class AlertEvaluator:
    """
    Rule-based automated evaluator for MHARS alerts.
    Does not require an LLM-as-judge — uses deterministic checks.
    """

    # Action keywords that indicate actionability
    ACTION_KEYWORDS = [
        "increase", "reduce", "throttle", "shutdown", "monitor",
        "inspect", "maintenance", "fan", "cool", "check",
        "intervention", "recommended", "required", "initiated",
    ]

    # Machine types the system knows about
    MACHINE_TYPES = ["cpu", "motor", "server", "engine", "machine"]

    def evaluate(self, alert_text: str, context: Dict[str, Any]) -> AlertScore:
        """
        Evaluate a single alert against the ground-truth context.

        Args:
            alert_text: the generated alert string
            context:    dict with keys: machine_type, current_temp,
                        predicted_temp, anomaly_score, action_name, urgency
        """
        score = AlertScore()

        if not alert_text or not alert_text.strip():
            return score

        text_lower = alert_text.lower()
        current_temp   = context.get("current_temp", 0.0)
        predicted_temp = context.get("predicted_temp", 0.0)
        machine_type   = context.get("machine_type", "Machine").lower()
        urgency        = context.get("urgency", 0.0)

        # 1. Factual accuracy — does the alert contain the correct temperature?
        temp_str = f"{current_temp:.1f}"
        temp_int = str(int(round(current_temp)))
        if temp_str in alert_text or temp_int in alert_text:
            score.factual = 1.0
        elif any(str(int(current_temp) + d) in alert_text for d in [-1, 0, 1]):
            score.factual = 0.5  # close enough (rounding)
        else:
            score.factual = 0.0

        # 2. Specificity — mentions machine type and temperature trend
        type_mentioned = any(mt in text_lower for mt in self.MACHINE_TYPES)
        trend_words = ["rising", "falling", "stable", "cooling", "increasing", "decreasing"]
        trend_mentioned = any(tw in text_lower for tw in trend_words)
        score.specificity = (0.5 * int(type_mentioned) + 0.5 * int(trend_mentioned))

        # 3. Actionability — does it recommend something concrete?
        action_count = sum(1 for kw in self.ACTION_KEYWORDS if kw in text_lower)
        score.actionability = min(1.0, action_count / 2.0)

        # 4. Conciseness — should be 1–3 sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', alert_text) if s.strip()]
        if 1 <= len(sentences) <= 3:
            score.conciseness = 1.0
        elif len(sentences) == 4:
            score.conciseness = 0.5
        else:
            score.conciseness = 0.2

        # 5. No hallucination — check for invented extreme temps
        numbers_in_text = re.findall(r'\b(\d+\.?\d*)\b', alert_text)
        hallucinated = False
        for num_str in numbers_in_text:
            try:
                num = float(num_str)
                # Flag if a number looks like a temperature but is wildly wrong
                if 20 < num < 200 and abs(num - current_temp) > 30 and abs(num - predicted_temp) > 30:
                    hallucinated = True
            except ValueError:
                pass
        score.no_hallucination = 0.0 if hallucinated else 1.0

        # Weighted total
        score.total = (
            0.30 * score.factual +
            0.20 * score.specificity +
            0.25 * score.actionability +
            0.10 * score.conciseness +
            0.15 * score.no_hallucination
        )

        return score

    def compare_llm_vs_template(
        self,
        llm_alert: str,
        template_alert: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare LLM-generated alert against template baseline.
        Returns which one is better and by how much.
        """
        llm_score = self.evaluate(llm_alert, context)
        tpl_score = self.evaluate(template_alert, context)

        winner = "llm" if llm_score.total > tpl_score.total else "template"
        margin = abs(llm_score.total - tpl_score.total)

        return {
            "winner": winner,
            "margin": round(margin, 3),
            "llm_score": llm_score.to_dict(),
            "template_score": tpl_score.to_dict(),
            "recommendation": (
                f"LLM is {margin:.1%} better — use LLM" if winner == "llm"
                else f"Template is {margin:.1%} better — LLM needs prompt tuning"
            ),
        }

    def batch_evaluate(
        self,
        alerts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of alerts. Each item must have:
          {"alert": str, "context": dict}

        Returns aggregate statistics.
        """
        scores = []
        for item in alerts:
            s = self.evaluate(item["alert"], item["context"])
            scores.append(s)

        if not scores:
            return {"count": 0}

        return {
            "count": len(scores),
            "avg_total":          round(sum(s.total for s in scores) / len(scores), 3),
            "avg_factual":        round(sum(s.factual for s in scores) / len(scores), 3),
            "avg_specificity":    round(sum(s.specificity for s in scores) / len(scores), 3),
            "avg_actionability":  round(sum(s.actionability for s in scores) / len(scores), 3),
            "avg_conciseness":    round(sum(s.conciseness for s in scores) / len(scores), 3),
            "avg_no_hallucination": round(sum(s.no_hallucination for s in scores) / len(scores), 3),
            "min_total":          round(min(s.total for s in scores), 3),
            "max_total":          round(max(s.total for s in scores), 3),
        }


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = AlertEvaluator()

    ctx = {
        "machine_type": "Motor",
        "current_temp": 72.5,
        "predicted_temp": 78.0,
        "anomaly_score": 0.35,
        "action_name": "fan+",
        "urgency": 0.55,
    }

    # Test template alert
    tpl = ("[WARNING] The Motor is at 72.5°C (anomaly=0.35), "
           "temperature is slowly rising (+5.5°C). "
           "Close monitoring is recommended — fan speed has been increased by 20%.")
    tpl_score = evaluator.evaluate(tpl, ctx)
    print(f"Template score: {tpl_score.to_dict()}")

    # Test a bad LLM alert (hallucinated)
    bad = "The server is at 120°C and about to explode. Please evacuate immediately."
    bad_score = evaluator.evaluate(bad, ctx)
    print(f"Bad LLM score:  {bad_score.to_dict()}")

    # Compare
    comparison = evaluator.compare_llm_vs_template(bad, tpl, ctx)
    print(f"Winner: {comparison['winner']} by {comparison['margin']:.3f}")
    print(f"Recommendation: {comparison['recommendation']}")
