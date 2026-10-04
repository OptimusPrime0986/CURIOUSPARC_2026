"""FastAPI Application Server for CrowdWatch Control-Room System."""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import re
import shutil
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.service import get_pipeline
from src.config import get_config

logger = logging.getLogger("crowdwatch.api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = PROJECT_ROOT / "web"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for background pipeline initialization."""
    pipeline = get_pipeline()
    pipeline.start()
    logger.info("CrowdWatch API server initialized and pipeline started")
    yield
    pipeline.stop()
    logger.info("CrowdWatch API server shutdown complete")


app = FastAPI(
    title="CrowdWatch Early-Warning API",
    description="Privacy-Preserving Crowd Density & Stampede Early-Warning System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local offline access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the control room single-page dashboard."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(index_file)


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Returns instantaneous system telemetry, FPS, and status."""
    pipeline = get_pipeline()
    return pipeline.get_telemetry()


@app.get("/api/config")
async def get_configuration() -> Dict[str, Any]:
    """Returns active system configuration, zones, and calibration data."""
    cfg = get_config()
    return {
        "system": cfg.get("system"),
        "input": cfg.get("input"),
        "counting": cfg.get("counting"),
        "alerts": cfg.get("alerts"),
        "zones": cfg.zones_data,
        "calibration": cfg.calibration_data,
    }


class SourceRequest(BaseModel):
    source_type: str  # "file", "webcam", "rtsp", "url", "preset"
    source_value: str


@app.post("/api/stream/source")
async def set_source(req: SourceRequest) -> Dict[str, str]:
    """Switches the active video ingestion source (webcam, RTSP, mobile stream, or file)."""
    pipeline = get_pipeline()
    val = req.source_value.strip()

    if req.source_type == "webcam":
        src = int(val) if val.isdigit() else 0
    elif req.source_type in ("file", "preset"):
        p = Path(val)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        src = p
    elif req.source_type in ("rtsp", "url", "stream"):
        src = val
    else:
        src = val

    pipeline.change_source(src)
    return {"status": "ok", "new_source": str(src)}


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, str]:
    """Uploads a video from phone/laptop and directly connects it to the AI pipeline."""
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename)
    target_path = UPLOAD_DIR / clean_name

    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("Uploaded video received & saved to: %s", target_path)
    pipeline = get_pipeline()
    pipeline.change_source(target_path)
    return {
        "status": "ok",
        "filename": clean_name,
        "source": str(target_path),
        "message": f"Successfully activated uploaded clip: {clean_name}",
    }


@app.get("/api/videos/list")
async def list_available_videos() -> Dict[str, Any]:
    """Returns list of all available test clips (pre-loaded and user-uploaded)."""
    data_dir = PROJECT_ROOT / "data"
    videos = []
    
    # Pre-loaded in data/
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for f in data_dir.glob(ext):
            videos.append({
                "name": f.name,
                "path": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "type": "sample",
            })
            
    # User uploaded
    if UPLOAD_DIR.exists():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for f in UPLOAD_DIR.glob(ext):
                videos.append({
                    "name": f.name,
                    "path": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "type": "uploaded",
                })
                
    return {"videos": videos}


def mjpeg_frame_generator(stream_mode: str = "heatmap"):
    """Yields continuous JPEG frames formatted as multipart MJPEG stream."""
    pipeline = get_pipeline()
    while True:
        if stream_mode == "raw":
            jpeg_bytes = pipeline.get_latest_raw_jpeg()
        elif stream_mode == "sidebyside":
            jpeg_bytes = pipeline.get_latest_side_by_side_jpeg()
        else:
            jpeg_bytes = pipeline.get_latest_heatmap_jpeg()

        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        import time
        time.sleep(0.04)  # ~25 FPS max stream delivery


@app.get("/api/stream/live")
async def live_stream():
    """Streams real-time MJPEG video with live heatmap overlay (compatibility)."""
    return StreamingResponse(
        mjpeg_frame_generator("heatmap"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/raw")
async def raw_stream():
    """Streams original unaltered CCTV / phone camera feed."""
    return StreamingResponse(
        mjpeg_frame_generator("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/heatmap")
async def heatmap_stream():
    """Streams real-time AI density heatmap overlay."""
    return StreamingResponse(
        mjpeg_frame_generator("heatmap"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/sidebyside")
async def sidebyside_stream():
    """Streams synchronized side-by-side view (Original Footage | AI Heatmap)."""
    return StreamingResponse(
        mjpeg_frame_generator("sidebyside"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Streams live telemetry packets to connected dashboards."""
    await websocket.accept()
    pipeline = get_pipeline()
    try:
        while True:
            data = pipeline.get_telemetry()
            await websocket.send_json(data)
            await asyncio.sleep(0.2)  # 5 Hz telemetry updates
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


# Mount static assets if web directory exists
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

