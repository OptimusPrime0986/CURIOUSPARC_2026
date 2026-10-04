"""FastAPI Application Server for CrowdWatch Control-Room System."""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.service import get_pipeline
from src.config import get_config

logger = logging.getLogger("crowdwatch.api")

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


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
    source_type: str  # "file", "webcam", "rtsp"
    source_value: str


@app.post("/api/stream/source")
async def set_source(req: SourceRequest) -> Dict[str, str]:
    """Switches the active video ingestion source."""
    pipeline = get_pipeline()
    if req.source_type == "webcam":
        src = int(req.source_value) if req.source_value.isdigit() else 0
    elif req.source_type == "file":
        src = Path(req.source_value)
    else:
        src = req.source_value

    pipeline.change_source(src)
    return {"status": "ok", "new_source": str(src)}


def mjpeg_frame_generator():
    """Yields continuous JPEG frames formatted as multipart MJPEG stream."""
    pipeline = get_pipeline()
    while True:
        jpeg_bytes = pipeline.get_latest_jpeg()
        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        import time
        time.sleep(0.04)  # ~25 FPS max stream delivery


@app.get("/api/stream/live")
async def live_stream():
    """Streams real-time MJPEG video with live heatmap overlay."""
    return StreamingResponse(
        mjpeg_frame_generator(),
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
