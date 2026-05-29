from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

def run():
    uvicorn.run(app, host="127.0.0.1", port=8001)

t = threading.Thread(target=run)
t.daemon = True
t.start()
import time
time.sleep(1)
import urllib.request
req = urllib.request.Request("http://127.0.0.1:8001/", method="OPTIONS")
req.add_header("Origin", "null")
req.add_header("Access-Control-Request-Method", "POST")
try:
    resp = urllib.request.urlopen(req)
    print("Headers:", resp.headers)
except Exception as e:
    print("Error:", e)
