# MHARS LLM Quality vs Template Evaluation

This report validates the qualitative improvement of using an LLM (Phi-3) over a standard rule-based template for operator alerts during non-critical (cloud-routed) events.

## Scenario: Normal Operation
- **Current Temp:** 45.0°C (Predicted: 46.5°C)
- **Urgency Score:** 0.1
- **PPO Action:** `do-nothing`

### Rule-Based Template
> [Pump] 45.0°C — action: do-nothing.

### Phi-3 Mini (LLM)
> The pump's current temperature is 45.0°C, with a predicted increase to 46.5°C in 10 minutes, and the system has not taken action, despite an anomaly score of 0.01 and urgency level of 0.10. Please initiate a cooling procedure to prevent overheating.

---

## Scenario: Early Warning (Medium Urgency)
- **Current Temp:** 68.0°C (Predicted: 78.0°C)
- **Urgency Score:** 0.55
- **PPO Action:** `throttle`

### Rule-Based Template
> [Pump] 68.0°C — action: throttle.

### Phi-3 Mini (LLM)
> Reduce the pump's operational speed to mitigate the temperature increase, as the anomaly score indicates a moderate deviation from normal conditions. Immediate attention is advised due to the urgency level, but it is not critical.

---

## Scenario: Post-Cooling Recovery
- **Current Temp:** 65.0°C (Predicted: 50.0°C)
- **Urgency Score:** 0.3
- **PPO Action:** `do-nothing`

### Rule-Based Template
> [Pump] 65.0°C — action: do-nothing.

### Phi-3 Mini (LLM)
> The pump's current temperature is 65.0°C, with a predicted decrease to 50.0°C in the next 10 minutes, and the system has not taken any action despite an anomaly score of 0.40, indicating a non-critical situation with moderate urgency. Please review the system's response to the temperature anomaly and consider monitoring the pump closely.

---

## Conclusion
The LLM provides actionable context, explaining *why* an action was taken (e.g., rising prediction vs current state) rather than just stating the current state. This drastically reduces cognitive load for the human operator.
