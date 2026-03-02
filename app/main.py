"""
SentinelAI – main.py
"""

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.fusion_engine   import FusionEngine
from app.settings_manager import SettingsManager

BASE_DIR      = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR.parent / "static"

settings_manager = SettingsManager()
fusion_engine    = FusionEngine(settings_manager)

_monitor_thread     : Optional[threading.Thread] = None
_monitor_stop_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _monitor_stop_event.set()
    fusion_engine.stop()


app = FastAPI(title="SentinelAI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings-page", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})


# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/status")
async def get_status():
    status = fusion_engine.get_status()
    # Return OFFLINE when not running so dashboard hides emergency overlay
    if not status["running"]:
        status["state"] = "OFFLINE"
    return JSONResponse(content=status)


# ── Start ──────────────────────────────────────────────────────────────────────
@app.post("/start")
async def start_monitoring():
    global _monitor_thread, _monitor_stop_event
    if fusion_engine.is_running:
        return JSONResponse(content={"message": "Already running"})
    _monitor_stop_event.clear()
    _monitor_thread = threading.Thread(
        target = fusion_engine.run_loop,
        args   = (_monitor_stop_event,),
        daemon = True,
        name   = "SentinelAI-Monitor",
    )
    _monitor_thread.start()
    return JSONResponse(content={"message": "Monitoring started"})


# ── Stop ───────────────────────────────────────────────────────────────────────
@app.post("/stop")
async def stop_monitoring():
    _monitor_stop_event.set()
    fusion_engine.stop()
    return JSONResponse(content={"message": "Monitoring stopped"})


# ── Reset — stop alarm + clear state + stop monitoring ────────────────────────
@app.post("/reset")
async def reset_system():
    global _monitor_thread
    # 1. Clear state and stop alarm immediately
    fusion_engine.reset()
    # 2. Stop the monitoring loop
    _monitor_stop_event.set()
    fusion_engine.stop()
    _monitor_thread = None
    return JSONResponse(content={"message": "reset"})


# ── Cancel ─────────────────────────────────────────────────────────────────────
@app.post("/cancel")
async def cancel_alert():
    fusion_engine.cancel_emergency()
    return JSONResponse(content={"message": "cancelled"})


# ── Settings ───────────────────────────────────────────────────────────────────
@app.get("/settings")
async def get_settings():
    return JSONResponse(content=settings_manager.load())

@app.post("/settings")
async def save_settings(request: Request):
    try:
        data    = await request.json()
        success = settings_manager.save(data)
        if success:
            return JSONResponse(content={"message": "Settings saved"})
        return JSONResponse(content={"message": "Save failed"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)


# ── MJPEG stream ───────────────────────────────────────────────────────────────
def _mjpeg_generator():
    BOUNDARY     = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    OFFLINE_FRAME = _make_offline_frame()
    while True:
        frame = fusion_engine.get_latest_frame()
        yield BOUNDARY + (frame or OFFLINE_FRAME) + b"\r\n"
        time.sleep(0.04)

def _make_offline_frame() -> bytes:
    try:
        import cv2, numpy as np
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = (13, 17, 23)
        cv2.putText(img, "CAMERA OFFLINE", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 80, 80), 1)
        cv2.putText(img, "Press START MONITORING", (18, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 60, 60), 1)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    except Exception:
        return b""

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
