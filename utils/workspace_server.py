"""
FastAPI workspace server — exposes Pi state and camera streams via HTTP/WebSocket.
Run via start_web.py (not directly).
"""
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TNS Workspace Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Injected at process start by init()
_shared_state = None
_shared_frames = None
_cmd_queue = None


def init(shared_state, shared_frames, cmd_queue):
    global _shared_state, _shared_frames, _cmd_queue
    _shared_state = shared_state
    _shared_frames = shared_frames
    _cmd_queue = cmd_queue


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cameras")
def cameras():
    cams = _shared_state.get("cameras", []) if _shared_state else []
    return {"cameras": list(cams)}


@app.post("/trigger/{ws_id}")
def trigger(ws_id: int):
    if ws_id not in (1, 2):
        return {"error": "ws_id must be 1 or 2"}
    _cmd_queue.put({"command": "TRIGGER", "workspace_id": ws_id})
    return {"status": "ok", "workspace_id": ws_id}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    last_snapshot = {}
    try:
        while True:
            snapshot = {
                "ws1": _shared_state.get("ws1_state", "START") if _shared_state else "START",
                "ws2": _shared_state.get("ws2_state", "START") if _shared_state else "START",
                "error1": _shared_state.get("ws1_error", "") if _shared_state else "",
                "error2": _shared_state.get("ws2_error", "") if _shared_state else "",
                "cameras": list(_shared_state.get("cameras", [])) if _shared_state else [],
            }
            if snapshot != last_snapshot:
                await websocket.send_text(json.dumps(snapshot))
                last_snapshot = snapshot
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/stream/camera/{cam_id}")
async def stream_camera(cam_id: int):
    async def generate():
        while True:
            frame = _shared_frames.get(f"raw_{cam_id}") if _shared_frames else None
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytes(frame) + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/stream/workspace/{cam_id}")
async def stream_workspace(cam_id: int):
    async def generate():
        while True:
            frame = _shared_frames.get(f"ws_{cam_id}") if _shared_frames else None
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytes(frame) + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )
