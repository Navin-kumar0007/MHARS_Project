import asyncio
import json
import uvicorn
from multiprocessing import Process
from api.main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")

if __name__ == "__main__":
    p = Process(target=run_server)
    p.start()
    
    import time
    time.sleep(10) # wait for models to load
    
    import websockets
    async def test():
        uri = "ws://127.0.0.1:8001/ws/telemetry"
        try:
            async with websockets.connect(uri) as ws:
                msg = await ws.recv()
                data = json.loads(msg)
                print("metadata keys:", data.get("metadata", {}).keys())
                print("contributions:", data.get("metadata", {}).get("contributions"))
                print("fault_type:", data.get("metadata", {}).get("fault_type"))
        except Exception as e:
            print("Error:", e)
            
    asyncio.run(test())
    p.terminate()
