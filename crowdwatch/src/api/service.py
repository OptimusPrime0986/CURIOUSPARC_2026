"""Core processing pipeline service integrating Capture, Counting, and Telemetry."""

from __future__ import annotations
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.capture.video_stream import VideoStream
from src.config import get_config
from src.counting.clip_ebc import CLIPEBCPredictor
from src.counting.depth_anything import DepthAnythingV2Predictor
from src.counting.scheduler import AdaptiveCountScheduler

logger = logging.getLogger("crowdwatch.pipeline")


class PipelineService:
    """Manages video capture, scheduling, inference, and stream generation."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.device = self.cfg.get("system", "device", "cpu")

        # Capture source
        src_type = self.cfg.get("input", "source_type", "file")
        if src_type == "webcam":
            source = self.cfg.get("input", "camera_index", 0)
        elif src_type == "rtsp":
            source = self.cfg.get("input", "rtsp_url", "")
        else:
            source = Path(self.cfg.get("input", "source_path", "data/sample_crowd.mp4"))
            if not source.is_absolute():
                source = Path(__file__).resolve().parent.parent.parent / source

        self.source = source
        self.stream: Optional[VideoStream] = None
        self.predictor: Optional[CLIPEBCPredictor] = None
        self.scheduler: Optional[AdaptiveCountScheduler] = None
        self.depth_estimator: Optional[DepthAnythingV2Predictor] = None

        # Processed FPS & latency tracking
        self.processed_fps: float = 0.0
        self.latency_ms: float = 0.0
        self.total_count: float = 0.0
        self._fps_history = []
        self._last_processed_time = time.time()

        # Latest rendered frames for MJPEG streams
        self._latest_rendered_jpeg: Optional[bytes] = None
        self._latest_raw_jpeg: Optional[bytes] = None
        self._latest_heatmap_jpeg: Optional[bytes] = None
        self._latest_side_by_side_jpeg: Optional[bytes] = None
        self._render_lock = threading.Lock()
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # WebSocket subscribers
        self._ws_clients: List[asyncio.Queue] = []
        self._ws_lock = threading.Lock()

    def start(self) -> None:
        """Initializes components and starts background processing."""
        if self._is_running:
            return

        logger.info("Initializing PipelineService...")
        # 1. Initialize models
        clip_model_name = self.cfg.get("counting", "clip_model_name", "openai/clip-vit-base-patch16")
        self.predictor = CLIPEBCPredictor(clip_model_name=clip_model_name, device=self.device)

        frame_skip = self.cfg.get("counting", "frame_skip", 2)
        auto_adapt = self.cfg.get("counting", "auto_adapt_frame_skip", True)
        self.scheduler = AdaptiveCountScheduler(
            self.predictor,
            initial_frame_skip=frame_skip,
            auto_adapt=auto_adapt,
        )

        depth_model_id = self.cfg.get("depth", "model_id", "depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_estimator = DepthAnythingV2Predictor(model_id=depth_model_id, device=self.device)

        # 2. Start VideoStream
        target_fps = self.cfg.get("input", "target_fps", 15)
        self.stream = VideoStream(
            source=self.source,
            target_fps=target_fps,
            reconnect_timeout_sec=self.cfg.get("input", "reconnect_timeout_sec", 30.0),
        ).start()

        # 3. Start processing loop
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True, name="PipelineWorker")
        self._worker_thread.start()
        logger.info("PipelineService successfully started")

    def stop(self) -> None:
        """Stops the pipeline service."""
        self._is_running = False
        if self.stream:
            self.stream.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("PipelineService stopped")

    def change_source(self, new_source: Union[int, str, Path]) -> None:
        """Dynamically switches video input source."""
        logger.info("Switching video source to: %s", new_source)
        if self.stream:
            self.stream.stop()
        self.source = new_source
        self.stream = VideoStream(
            source=self.source,
            target_fps=self.cfg.get("input", "target_fps", 15),
            reconnect_timeout_sec=self.cfg.get("input", "reconnect_timeout_sec", 30.0),
        ).start()

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the latest rendered JPEG frame (backward-compatible)."""
        with self._render_lock:
            return self._latest_rendered_jpeg

    def get_latest_raw_jpeg(self) -> Optional[bytes]:
        """Returns the raw input CCTV/camera frame without heatmap."""
        with self._render_lock:
            return self._latest_raw_jpeg or self._latest_rendered_jpeg

    def get_latest_heatmap_jpeg(self) -> Optional[bytes]:
        """Returns the real-time AI density heatmap overlay."""
        with self._render_lock:
            return self._latest_heatmap_jpeg or self._latest_rendered_jpeg

    def get_latest_side_by_side_jpeg(self) -> Optional[bytes]:
        """Returns synchronized side-by-side feed (Original Footage | AI Heatmap)."""
        with self._render_lock:
            return self._latest_side_by_side_jpeg or self._latest_rendered_jpeg

    def get_telemetry(self) -> Dict[str, Union[float, int, str, bool]]:
        """Returns instantaneous system telemetry and risk assessment."""
        is_conn = self.stream.is_connected if self.stream else False
        status = "ONLINE" if is_conn else "RECONNECTING"
        
        # Calculate risk level based on count and density
        count = self.total_count
        if count < 15:
            risk_level = "NORMAL"
            risk_color = "#10b981"
            risk_text = "Safe crowd density. Standard flow maintained."
        elif count < 35:
            risk_level = "WATCH"
            risk_color = "#3b82f6"
            risk_text = "Moderate concentration. Keep monitoring choke points."
        elif count < 60:
            risk_level = "WARNING"
            risk_color = "#f59e0b"
            risk_text = "High density detected. Prepare queue management."
        else:
            risk_level = "CRITICAL"
            risk_color = "#ef4444"
            risk_text = "Critical stampede danger! Initiate immediate crowd dispersion."

        source_display = Path(str(self.source)).name if isinstance(self.source, (Path, str)) else f"Camera #{self.source}"

        return {
            "status": status,
            "is_connected": is_conn,
            "processed_fps": round(self.processed_fps, 1),
            "camera_fps": self.stream.fps if self.stream else 0.0,
            "latency_ms": round(self.latency_ms, 1),
            "total_count": round(self.total_count, 1),
            "frame_skip": self.scheduler.current_frame_skip if self.scheduler else 2,
            "device": self.device,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_action": risk_text,
            "source_name": source_display,
        }

    def _process_loop(self) -> None:
        """Continuous frame processing loop."""
        while self._is_running:
            loop_start = time.perf_counter()

            if not self.stream:
                time.sleep(0.05)
                continue

            is_conn, frame, frame_id = self.stream.read()
            if not is_conn or frame is None:
                # Render reconnecting placeholder screen
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "CAMERA DISCONNECTED - AUTO RECONNECTING [30s TIMEOUT]...",
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 140, 255),
                    2,
                    cv2.LINE_AA,
                )
                _, jpeg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self._render_lock:
                    self._latest_rendered_jpeg = jpeg.tobytes()
                time.sleep(0.1)
                continue

            # Process frame with adaptive scheduler
            density_map, count, is_new, latency = self.scheduler.process_frame(frame, frame_id)
            self.total_count = count
            self.latency_ms = latency

            # Render 1: Clean Raw Frame with subtle camera HUD
            raw_rendered = self._render_raw_view(frame)
            _, raw_jpeg = cv2.imencode(".jpg", raw_rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Render 2: Heatmap Overlay with telemetry HUD
            heatmap_rendered = self._render_overlay(frame, density_map, count, latency)
            _, heatmap_jpeg = cv2.imencode(".jpg", heatmap_rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Render 3: Synchronized Side-by-Side Composite
            sbs_rendered = self._render_side_by_side(raw_rendered, heatmap_rendered)
            _, sbs_jpeg = cv2.imencode(".jpg", sbs_rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])

            with self._render_lock:
                self._latest_raw_jpeg = raw_jpeg.tobytes()
                self._latest_heatmap_jpeg = heatmap_jpeg.tobytes()
                self._latest_side_by_side_jpeg = sbs_jpeg.tobytes()
                self._latest_rendered_jpeg = heatmap_jpeg.tobytes()

            # Measure processed FPS
            now = time.perf_counter()
            dt = now - self._last_processed_time
            self._last_processed_time = now
            if dt > 0:
                self._fps_history.append(1.0 / dt)
                if len(self._fps_history) > 15:
                    self._fps_history.pop(0)
                self.processed_fps = float(np.mean(self._fps_history))

            # Maintain max 15 FPS processing rate to keep CPU balanced
            elapsed = time.perf_counter() - loop_start
            sleep_time = (1.0 / 15.0) - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _render_raw_view(self, frame: np.ndarray) -> np.ndarray:
        """Renders original camera / mobile footage with clean HUD."""
        out = frame.copy()
        h, w = out.shape[:2]
        # Top banner
        cv2.rectangle(out, (0, 0), (w, 36), (15, 23, 42), -1)
        src_name = Path(str(self.source)).name if isinstance(self.source, (Path, str)) else f"Cam #{self.source}"
        cv2.putText(
            out,
            f"ORIGINAL FOOTAGE (CCTV / MOBILE) | Source: {src_name}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )
        # Live badge
        cv2.circle(out, (w - 75, 18), 5, (0, 220, 100), -1)
        cv2.putText(
            out,
            "LIVE",
            (w - 62, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 100),
            1,
            cv2.LINE_AA,
        )
        return out

    def _render_overlay(
        self, frame: np.ndarray, density_map: np.ndarray, count: float, latency: float
    ) -> np.ndarray:
        """Renders density heatmap overlay with HUD indicators."""
        out = frame.copy()
        if density_map is not None:
            norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            out = cv2.addWeighted(out, 0.60, heatmap, 0.40, 0)

        # Draw Control-Room HUD Header
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 36), (15, 23, 42), -1)

        hud_text = (
            f"AI HEATMAP | Est: {count:.0f} ppl | {self.processed_fps:.1f} FPS | "
            f"Lat: {latency:.0f}ms"
        )
        cv2.putText(
            out,
            hud_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )

        # Draw "SIMULATED" badge if synthetic file
        if "sample_crowd" in str(self.source):
            cv2.putText(
                out,
                "[SIMULATED]",
                (w - 120, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        return out

    def _render_side_by_side(self, raw_frame: np.ndarray, heatmap_frame: np.ndarray) -> np.ndarray:
        """Horizontally stacks raw footage and AI heatmap with a divider."""
        # Normalize height if dimensions differ
        h1, w1 = raw_frame.shape[:2]
        h2, w2 = heatmap_frame.shape[:2]
        target_h = min(h1, h2, 480)
        
        scale1 = target_h / h1
        scale2 = target_h / h2
        
        f1 = cv2.resize(raw_frame, (int(w1 * scale1), target_h))
        f2 = cv2.resize(heatmap_frame, (int(w2 * scale2), target_h))
        
        # Center divider border
        divider = np.full((target_h, 6, 3), 40, dtype=np.uint8)
        divider[:, 2:4] = (0, 200, 255) # cyan separator line
        
        return np.hstack([f1, divider, f2])


# Global singleton pipeline instance
_GLOBAL_PIPELINE: Optional[PipelineService] = None


def get_pipeline() -> PipelineService:
    """Gets or initializes the global pipeline service."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = PipelineService()
    return _GLOBAL_PIPELINE
