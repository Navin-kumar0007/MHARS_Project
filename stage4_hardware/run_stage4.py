"""
MHARS — Stage 4 Validation Runner
===================================
Tests the IoT connectivity components (MQTT) and the lightweight
Edge inference wrapper.
"""

import time
import threading

def test_edge_inference():
    print("\n" + "=" * 56)
    print("  Component 1 — Edge Inference Wrapper")
    print("=" * 56)
    from stage4_hardware.edge_inference import run_edge_inference
    passed = run_edge_inference()
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Edge Inference working without LLM overhead.")
    return passed

def test_mqtt_pubsub():
    print("\n" + "=" * 56)
    print("  Component 2 — MQTT IoT Communication (Pub/Sub)")
    print("=" * 56)
    try:
        import paho.mqtt.client
    except ImportError:
        print("[WARN] paho-mqtt not installed. Run: pip install paho-mqtt")
        return False
        
    from stage4_hardware.mqtt_subscriber import run_subscriber
    from stage4_hardware.mqtt_publisher import run_publisher
    
    # Run subscriber in background
    print("[Test] Starting Cloud Subscriber thread...")
    sub_thread = threading.Thread(target=run_subscriber, args=(5,))
    sub_thread.start()
    
    # Wait a sec for subscriber to connect
    time.sleep(1)
    
    # Run publisher
    print("[Test] Running Edge Publisher...")
    pub_passed = run_publisher(iterations=3)
    
    sub_thread.join()
    
    passed = pub_passed
    status = "PASS" if passed else "WARN"
    print(f"[{status}] MQTT Pub/Sub test completed.")
    return passed

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   MHARS Stage 4 — Hardware & IoT Deployment         ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    p1 = test_edge_inference()
    p2 = test_mqtt_pubsub()
    
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  Stage 4 Results Summary" + " "*29 + "║")
    print("╠" + "═"*54 + "╣")
    print(f"║  {'✓' if p1 else '⚠'}  {'Edge Inference':<30} {'PASSED' if p1 else 'WARN':<12}║")
    print(f"║  {'✓' if p2 else '⚠'}  {'MQTT IoT Pub/Sub':<30} {'PASSED' if p2 else 'WARN':<12}║")
    print("╚" + "═"*54 + "╝\n")

if __name__ == "__main__":
    main()
