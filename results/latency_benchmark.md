# MHARS Latency Benchmark (Edge vs Cloud)

This benchmark validates the RL Router's ability to maintain real-time 
responsiveness (< 50ms) during critical thermal events by shifting 
compute to the edge.

## Degradation Sequence Results

| Step | Temp (°C) | Urgency | Route | Internal (ms) | Network (ms) | Total (ms) | Action |
|------|-----------|---------|-------|---------------|--------------|------------|--------|
| 1 | 35.0 | 0.420 | **both** | 5940.4 | 5.0 | **5946.9** | do-nothing |
| 2 | 38.0 | 0.420 | **both** | 3058.8 | 5.0 | **3065.1** | do-nothing |
| 3 | 41.0 | 0.420 | **both** | 3425.3 | 5.0 | **3431.6** | do-nothing |
| 4 | 45.0 | 0.420 | **both** | 4777.4 | 5.0 | **4783.7** | do-nothing |
| 5 | 50.0 | 0.800 | **edge** | 10.8 | 5.0 | **17.1** | do-nothing |
| 6 | 57.0 | 0.800 | **edge** | 3.1 | 5.0 | **9.3** | do-nothing |
| 7 | 63.0 | 0.800 | **edge** | 3.2 | 5.0 | **9.5** | do-nothing |
| 8 | 69.0 | 0.800 | **edge** | 3.0 | 5.0 | **8.7** | do-nothing |
| 9 | 74.0 | 0.800 | **edge** | 3.2 | 5.0 | **9.0** | do-nothing |
| 10 | 79.0 | 0.800 | **edge** | 3.0 | 5.0 | **9.2** | do-nothing |
| 11 | 83.0 | 0.800 | **edge** | 2.9 | 5.0 | **9.2** | do-nothing |
| 12 | 88.0 | 1.000 | **edge** | 5.1 | 5.0 | **11.3** | do-nothing |

## Summary Metrics

- **Average Edge Response Time:** 10.4 ms
- **Average Cloud Response Time:** 4306.8 ms

**Conclusion:** The RL router successfully maintains sub-50ms latency during critical events by utilizing edge routing.
