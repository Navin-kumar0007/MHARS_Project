"""
MHARS — Stage 4: Hardware MQTT Subscriber
===========================================
Simulates the Cloud/Server side receiving data from the
Edge devices and logging it or triggering further Cloud models.
"""

import json
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "mhars/v1/sensor_data"

def on_connect(client, userdata, flags, rc):
    print(f"[Cloud Subscriber] Connected with result code {rc}")
    client.subscribe(TOPIC)
    print(f"[Cloud Subscriber] Subscribed to topic: {TOPIC}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        machine_id = payload.get("machine_id", "UNKNOWN")
        temp = payload.get("metrics", {}).get("temperature", 0.0)
        action = payload.get("edge_decision", "none")
        
        print(f"  [Cloud] Received from {machine_id}: Temp={temp}°C | Edge Action={action}")
        
    except json.JSONDecodeError:
        print(f"  [Cloud] Received invalid JSON: {msg.payload}")

def run_subscriber(duration_s: int = 15):
    client = mqtt.Client(client_id="mhars_cloud_subscriber")
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"[Cloud Subscriber] Connecting to {BROKER}...")
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"[Cloud Subscriber] Connection failed: {e}. Skipping MQTT tests.")
        return False
        
    print(f"[Cloud Subscriber] Listening for {duration_s} seconds...")
    client.loop_start()
    
    import time
    time.sleep(duration_s)
    
    client.loop_stop()
    client.disconnect()
    print("[Cloud Subscriber] Finished listening.")
    return True

if __name__ == "__main__":
    run_subscriber()
