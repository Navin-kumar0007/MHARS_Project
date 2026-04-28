"""
MHARS — Stage 4: Edge Inference Wrapper
=========================================
Demonstrates how the MHARS system would be initialized
strictly for Edge inference (no LLM, fast response).
"""

import time
from mhars import MHARS

def run_edge_inference(machine_type_id: int = 1, iterations: int = 5):
    print(f"\n── Starting Edge Inference Test (Machine ID: {machine_type_id}) ──")
    
    # On the edge, we DO NOT load the LLM to save memory and latency
    system = MHARS(machine_type_id=machine_type_id, llm_path=None, verbose=False)
    
    print("[Edge Inference] System loaded successfully without LLM.")
    
    temps = [45.0, 55.0, 75.0, 85.0, 92.0]
    
    for temp in temps[:iterations]:
        t0 = time.perf_counter()
        
        # We manually run the system and pretend it's getting sensor data
        result = system.run(temp)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        print(f"  Temp: {temp:>4.1f}°C | Urgency: {result.urgency:>5.3f} | "
              f"Action: {result.action:<10} | Route: {result.route:<6} | "
              f"Latency: {latency_ms:>5.1f}ms")
              
    print("── Edge Inference Test Complete ──\n")
    return True

if __name__ == "__main__":
    run_edge_inference()
