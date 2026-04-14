import asyncio
import json
import os
import secrets
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from camera import Camera
from vlm import VLMChecker
from compiler import QueryCompiler

API_TOKEN = os.environ.get("NLCAM_API_TOKEN", "")
if not API_TOKEN:
    API_TOKEN = secrets.token_urlsafe(32)
    print(f"[NLCAM] No NLCAM_API_TOKEN set. Generated token: {API_TOKEN}")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

app = FastAPI()
CORS_ORIGINS = os.environ.get("NLCAM_CORS_ORIGINS", "http://localhost:5500").split(",")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

camera = Camera(source=0)
vlm = VLMChecker(model="llava:7b")
compiler = QueryCompiler(model=os.environ.get("NLCAM_COMPILER_MODEL", "mistral"))

# Shared state
active_condition: dict = {}
alert_log: list = []
connected_clients: set[WebSocket] = set()

class QueryRequest(BaseModel):
    query: str

@app.post("/compile", dependencies=[Depends(verify_token)])
async def compile_query(req: QueryRequest):
    global active_condition
    condition = await asyncio.to_thread(compiler.compile, req.query)
    active_condition = condition
    return {"condition": condition}

@app.get("/alerts", dependencies=[Depends(verify_token)])
def get_alerts():
    return {"alerts": alert_log[-50:]}  # Last 50 alerts

async def broadcast(message: str):
    stale = []
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        connected_clients.discard(ws)

async def capture_loop():
    while True:
        frame_b64 = camera.get_frame_b64()
        if frame_b64 is None:
            await asyncio.sleep(0.1)
            continue

        alert = None

        if active_condition:
            result = await asyncio.to_thread(vlm.check, frame_b64, active_condition)
            if result.triggered:
                alert = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "condition": active_condition,
                    "reason": result.explanation,
                    "frame": frame_b64
                }
                alert_log.append(alert)
                if len(alert_log) > 50:
                    alert_log[:] = alert_log[-50:]

        if connected_clients:
            msg = json.dumps({"frame": frame_b64, "alert": alert})
            await broadcast(msg)

        await asyncio.sleep(1.0)

@app.on_event("startup")
async def startup():
    asyncio.create_task(capture_loop())

@app.on_event("shutdown")
async def shutdown():
    camera.release()

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket, token: str = Query(...)):
    if not secrets.compare_digest(token, API_TOKEN):
        await websocket.close(code=1008, reason="Invalid token")
        return
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)