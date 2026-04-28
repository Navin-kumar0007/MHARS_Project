"""
MHARS — Stage 4: Hardware MQTT Publisher
==========================================
Simulates an IoT Edge device (e.g., Raspberry Pi) publishing
raw sensor data and local MHARS decisions to the cloud.
"""

import json
import time
import random
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "mhars/v1/sensor_data"

def on_connect(client, userdata, flags, rc):
    print(f"[Edge Publisher] Connected to MQTT broker with result code {rc}")

def run_publisher(machine_id: str = "PUMP_01", iterations: int = 10):
    client = mqtt.Client(client_id=f"mhars_edge_{machine_id}_{random.randint(1000,9999)}")
    client.on_connect = on_connect
    
    print(f"[Edge Publisher] Connecting to {BROKER}...")
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"[Edge Publisher] Connection failed: {e}. Skipping MQTT tests.")
        return False
        
    client.loop_start()
    
    print(f"[Edge Publisher] Starting transmission on topic: {TOPIC}")
    for i in range(iterations):
        # Simulate local Edge processing
        temp = random.uniform(30.0, 95.0)
        
        payload = {
            "machine_id": machine_id,
            "timestamp": time.time(),
            "metrics": {
                "temperature": round(temp, 2),
                "vibration_rms": round(random.uniform(0.5, 3.5), 3)
            },
            "edge_decision": "do-nothing" if temp < 80 else "throttle"
        }
        
        msg_str = json.dumps(payload)
        client.publish(TOPIC, msg_str)
        print(f"  Published: {msg_str}")
        time.sleep(1.0)
        
    client.loop_stop()
    client.disconnect()
    print("[Edge Publisher] Finished transmission.")
    return True

if __name__ == "__main__":
    run_publisher()
