import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/telemetry"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket.")
            for _ in range(1):
                message = await websocket.recv()
                data = json.loads(message)
                # print keys of metadata
                print("metadata keys:", data.get("metadata", {}).keys())
                print("metadata.contributions:", data.get("metadata", {}).get("contributions"))
                print("urgency:", data.get("urgency"))
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_ws())
