from mhars.system_health import SystemHealthMonitor
import json

print("\n--- Testing macOS Live Hardware Data Fetching ---")
try:
    data = SystemHealthMonitor.snapshot(0)
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error fetching data: {e}")

print("\n--- Testing Simulated Motor Data ---")
try:
    data_motor = SystemHealthMonitor.snapshot(1)
    print(json.dumps(data_motor, indent=2))
except Exception as e:
    print(f"Error fetching data: {e}")
