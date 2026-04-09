import asyncio
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from camera import Camera
from vlm import VLMChecker
from compiler import QueryCompiler

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

camera = Camera(source=0)
vlm = VLMChecker(model="llava:7b")
compiler = QueryCompiler(model="llava:7b")

# Shared state
active_condition: dict = {}
alert_log: list = []

class QueryRequest(BaseModel):
    query: str

@app.post("/compile")
async def compile_query(req: QueryRequest):
    global active_condition
    condition = compiler.compile(req.query)
    active_condition = condition
    return {"condition": condition}

@app.get("/alerts")
def get_alerts():
    return {"alerts": alert_log[-50:]}  # Last 50 alerts

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_b64 = camera.get_frame_b64()
            if frame_b64 is None:
                await asyncio.sleep(0.1)
                continue

            payload = {"frame": frame_b64, "alert": None}

            # Only run VLM check if a condition is active (throttled to ~1fps)
            if active_condition:
                result = vlm.check(frame_b64, active_condition)
                if result.triggered:
                    alert = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "condition": active_condition,
                        "reason": result.explanation,
                        "frame": frame_b64
                    }
                    alert_log.append(alert)
                    payload["alert"] = alert

            await websocket.send_text(json.dumps({
                "frame": frame_b64,
                "alert": payload["alert"]
            }))

            await asyncio.sleep(1.0)  # 1fps VLM cadence

    except WebSocketDisconnect:
        pass