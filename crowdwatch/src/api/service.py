"""Core processing pipeline service integrating Capture, Counting, Calibration, and Telemetry."""

from __future__ import annotations
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.calibration.manager import CalibrationManager
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

        # Calibration (Phase 1)
        project_root = Path(__file__).resolve().parent.parent.parent
        calib_path = project_root / self.cfg.get("calibration", "homography_file", "config/calibration.json")
        self.calibration_mgr = CalibrationManager(calibration_path=calib_path)
        if calib_path.exists():
            try:
                self.calibration_mgr.load()
                logger.info("Loaded active ground calibration from %s", calib_path)
            except Exception as e:
                logger.warning("Could not auto-load calibration from %s: %s", calib_path, e)

        # Per-zone live metrics (keyed by zone id)
        self.zone_metrics: Dict[str, Dict[str, float]] = {}

        # Zone definitions from config
        self.zones_data: List[Dict[str, Any]] = []
        self._load_zones()

        # Processed FPS & latency tracking
        self.processed_fps: float = 0.0
        self.latency_ms: float = 0.0
        self.total_count: float = 0.0
        self.calibrated_ground_count: float = 0.0
        self._latest_density_map: Optional[np.ndarray] = None
        self._fps_history: List[float] = []
        self._last_processed_time = time.time()

        # Spatial hotspot tracking for accurate safety alignment
        self.red_hotspot_ratio: float = 0.0
        self.yellow_zone_ratio: float = 0.0
        self.blue_free_ratio: float = 1.0
        self.avg_density_m2: float = 0.0

        # Camera drift tracking
        self.drift_detected: bool = False
        self.drift_ratio: float = 1.0
        self._drift_check_counter: int = 0

        # Dynamic camera viewing angle & perspective auto-configuration
        self.current_angle_result = None
        self._angle_check_counter: int = 0

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

    def _load_zones(self) -> None:
        """Load zone polygons from config/zones.json."""
        try:
            zones_raw = self.cfg.zones_data
            self.zones_data = zones_raw.get("zones", []) if isinstance(zones_raw, dict) else []
            for z in self.zones_data:
                self.zone_metrics[z["id"]] = {
                    "people": 0.0, "area_m2": 0.0, "density_m2": 0.0,
                    "flow_speed": 0.0, "ttc": -1.0,
                }
            logger.info("Loaded %d zone(s) from config", len(self.zones_data))
        except Exception as e:
            logger.warning("Could not load zones: %s", e)
            self.zones_data = []

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
        if self.scheduler:
            self.scheduler._last_density_map = None
            self.scheduler._last_count = 0.0
        self.current_angle_result = None
        self._angle_check_counter = 0
        if self.calibration_mgr:
            self.calibration_mgr._angle_estimator.reset()
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

    def _compute_zone_densities(self, density_map: np.ndarray, total_count: float) -> None:
        """Compute per-zone people/m² using calibrated polygon masks."""
        if not self.zones_data or density_map is None:
            return

        for zone in self.zones_data:
            zid = zone["id"]
            polygon = zone.get("polygon", [])
            # Align default Zone 1 with active calibrated ground plane if available
            if zid == "zone_1" and self.calibration_mgr.is_calibrated and len(self.calibration_mgr.image_points) >= 3:
                polygon = self.calibration_mgr.image_points

            if len(polygon) < 3:
                continue

            if self.calibration_mgr.is_calibrated:
                people, area, ppl_m2 = self.calibration_mgr.zone_density(
                    density_map, total_count, polygon,
                )
            else:
                # Fallback: polygon mask fraction × total_count, no real area
                h, w = density_map.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                zone_sum = float(np.sum(density_map[mask > 0]))
                total_sum = float(np.sum(density_map))
                frac = zone_sum / max(total_sum, 1e-6)
                people = round(frac * total_count, 2)
                area = 0.0
                ppl_m2 = 0.0

            self.zone_metrics[zid] = {
                **self.zone_metrics.get(zid, {}),
                "people": people,
                "area_m2": area,
                "density_m2": ppl_m2,
            }

    def get_telemetry(self) -> Dict[str, Union[float, int, str, bool, list]]:
        """Returns instantaneous system telemetry and risk assessment strictly aligned with footage."""
        is_conn = self.stream.is_connected if self.stream else False
        status = "ONLINE" if is_conn else "RECONNECTING"

        # Calibrated floor area and people inside ground polygon
        calibrated = self.calibration_mgr.is_calibrated
        floor_area = max(self.calibration_mgr.floor_area_m2, 1.0) if calibrated else 0.0

        if calibrated and self.calibration_mgr.image_points and len(self.calibration_mgr.image_points) >= 3:
            quad = self.calibration_mgr.image_points
            if self._latest_density_map is not None:
                people_in_ground, density = self.calibration_mgr.calibrator.density_per_m2(
                    self._latest_density_map, self.total_count, quad
                )
                self.calibrated_ground_count = float(round(people_in_ground, 1))
                self.avg_density_m2 = float(round(density, 2))
            else:
                self.calibrated_ground_count = float(round(self.total_count, 1))
                self.avg_density_m2 = float(round(self.total_count / max(floor_area, 1.0), 2))
        else:
            self.calibrated_ground_count = float(round(self.total_count, 1))
            self.avg_density_m2 = 0.0

        red_pct = round(self.red_hotspot_ratio * 100.0, 1)
        yellow_pct = round(self.yellow_zone_ratio * 100.0, 1)
        blue_pct = round(max(0.0, 100.0 - red_pct - yellow_pct), 1)

        # Build per-zone telemetry list
        zones_telemetry: List[Dict[str, Any]] = []
        for zone in self.zones_data:
            zid = zone["id"]
            zm = self.zone_metrics.get(zid, {})
            zones_telemetry.append({
                "id": zid,
                "name": zone.get("name", zid),
                "people": zm.get("people", 0.0),
                "area_m2": zm.get("area_m2", 0.0),
                "density_m2": zm.get("density_m2", 0.0),
                "flow_speed": zm.get("flow_speed", 0.0),
                "ttc": zm.get("ttc", -1.0),
            })

        # Risk thresholds from config
        alert_cfg = self.cfg.get("alerts", "state_thresholds", {})
        watch_thresh = alert_cfg.get("watch_max_density", 3.5) if isinstance(alert_cfg, dict) else 3.5
        warning_thresh = alert_cfg.get("warning_max_density", 5.0) if isinstance(alert_cfg, dict) else 5.0

        # Real-time risk assessment (Fruin LOS / NFPA 130)
        if not calibrated:
            # Without calibration density is meaningless — use heatmap hotspot ratio only
            if red_pct < 8.0:
                risk_level, risk_color, risk_icon = "NORMAL", "#10b981", "🛡️"
                risk_title = "SYSTEM NORMAL"
                risk_text = "Density not calibrated. Heatmap hotspot ratio is low."
                risk_advisory = "CALIBRATE FOR ACCURATE RISK"
            elif red_pct < 20.0:
                risk_level, risk_color, risk_icon = "WATCH", "#38bdf8", "👀"
                risk_title = "ACTIVE OBSERVATION (uncalibrated)"
                risk_text = "Moderate hotspot presence. Calibrate ground plane for accurate ppl/m²."
                risk_advisory = "UNCALIBRATED OBSERVATION"
            else:
                risk_level, risk_color, risk_icon = "WARNING", "#f59e0b", "⚠️"
                risk_title = "ELEVATED HOTSPOTS (uncalibrated)"
                risk_text = "Significant hotspot density detected. Calibrate for precise risk assessment."
                risk_advisory = "CALIBRATE FOR ACCURATE RISK"
        elif self.avg_density_m2 < 1.8 and red_pct < 8.0:
            risk_level, risk_color, risk_icon = "NORMAL", "#10b981", "🛡️"
            risk_title = "SYSTEM NORMAL"
            risk_text = "Safe crowd density. Standard pedestrian flow observed across all zones."
            risk_advisory = "FREE PEDESTRIAN FLOW"
        elif self.avg_density_m2 < 3.0 and red_pct < 16.0:
            risk_level, risk_color, risk_icon = "WATCH", "#38bdf8", "👀"
            risk_title = "ACTIVE OBSERVATION"
            risk_text = "Moderate crowd presence in mid zones. Traffic active but moving steadily."
            risk_advisory = "STEADY CONCOURSE FLOW"
        elif self.avg_density_m2 < 4.5 and red_pct < 28.0:
            risk_level, risk_color, risk_icon = "WARNING", "#f59e0b", "⚠️"
            risk_title = "CAPACITY WARNING"
            risk_text = "High density accumulation in hotspots. Prepare choke-point diversion gates."
            risk_advisory = "CONGESTION WARNING"
        else:
            risk_level, risk_color, risk_icon = "CRITICAL", "#ef4444", "🚨"
            risk_title = "CRITICAL STAMPEDE HAZARD"
            risk_text = "CRITICAL HAZARD: Dense crowd compaction! Initiate immediate physical flow dispersion."
            risk_advisory = "CRITICAL STAMPEDE ALERT"

        source_display = Path(str(self.source)).name if isinstance(self.source, (Path, str)) else f"Camera #{self.source}"

        # Legacy zone_a / zone_b fields for backward compat with existing dashboard
        zone_a_density = 0.0
        zone_b_density = 0.0
        if len(zones_telemetry) >= 1:
            zone_a_density = zones_telemetry[0].get("density_m2", 0.0)
        if len(zones_telemetry) >= 2:
            zone_b_density = zones_telemetry[1].get("density_m2", 0.0)

        return {
            "status": status,
            "is_connected": is_conn,
            "calibrated": calibrated,
            "drift_detected": self.drift_detected,
            "drift_ratio": round(self.drift_ratio, 3),
            "processed_fps": round(self.processed_fps, 1),
            "camera_fps": self.stream.fps if self.stream else 0.0,
            "latency_ms": round(self.latency_ms, 1),
            "total_count": round(self.total_count, 1),
            "calibrated_ground_count": round(self.calibrated_ground_count, 1),
            "density_m2": self.avg_density_m2,
            "floor_area_m2": round(self.calibration_mgr.floor_area_m2, 1) if calibrated else 0.0,
            "red_hotspot_pct": red_pct,
            "yellow_zone_pct": yellow_pct,
            "blue_free_pct": blue_pct,
            "zone_a_density": round(zone_a_density, 2),
            "zone_b_density": round(zone_b_density, 2),
            "zones": zones_telemetry,
            "frame_skip": self.scheduler.current_frame_skip if self.scheduler else 1,
            "device": self.device,
            "risk_level": risk_level,
            "risk_title": risk_title,
            "risk_color": risk_color,
            "risk_icon": risk_icon,
            "risk_action": risk_text,
            "risk_advisory": risk_advisory,
            "source_name": source_display,
            "camera_angle": {
                "view_type": self.current_angle_result.view_type if self.current_angle_result else "HIGH_OBLIQUE",
                "pitch_deg": self.current_angle_result.pitch_deg if self.current_angle_result else 55.0,
                "scale_gradient": self.current_angle_result.scale_gradient if self.current_angle_result else 1.8,
                "label": self.current_angle_result.label if self.current_angle_result else "Elevated Oblique (55°)",
                "auto_configured": not self.calibration_mgr._is_manual,
            },
        }

    def _process_loop(self) -> None:
        """Continuous frame processing loop with resilient error recovery."""
        while self._is_running:
            loop_start = time.perf_counter()

            try:
                if not self.stream:
                    time.sleep(0.05)
                    continue

                is_conn, frame, frame_id = self.stream.read()
                if not is_conn or frame is None:
                    # Render reconnecting placeholder screen
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "CONNECTING VIDEO STREAM...",
                        (140, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    _, jpeg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    with self._render_lock:
                        self._latest_rendered_jpeg = jpeg.tobytes()
                        self._latest_raw_jpeg = jpeg.tobytes()
                        self._latest_heatmap_jpeg = jpeg.tobytes()
                        self._latest_side_by_side_jpeg = jpeg.tobytes()
                    time.sleep(0.1)
                    continue

                # Auto-detect camera viewing angle & ground geometry (every 30 frames or on startup)
                self._angle_check_counter += 1
                if self._angle_check_counter >= 30 or self.current_angle_result is None:
                    self._angle_check_counter = 0
                    self.current_angle_result = self.calibration_mgr.auto_calibrate_from_angle(frame)

                # Process frame with adaptive scheduler and detected camera angle
                density_map, count, is_new, latency = self.scheduler.process_frame(
                    frame, frame_id, angle_result=self.current_angle_result
                )
                self.total_count = count
                self.latency_ms = latency
                self._latest_density_map = density_map

                # Check camera drift periodically (every 30 frames)
                self._drift_check_counter += 1
                if self._drift_check_counter >= 30 and self.calibration_mgr.is_calibrated:
                    self._drift_check_counter = 0
                    drift, ratio = self.calibration_mgr.check_drift(frame)
                    self.drift_detected = drift
                    self.drift_ratio = ratio

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

            except Exception as e:
                logger.exception("Error in processing loop: %s", e)
                time.sleep(0.05)

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
            f"ORIGINAL FOOTAGE (CCTV / MOBILE) | {src_name}",
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
        """Renders calibrated full-spectrum crowd density heatmap overlay.
        - Areas where people are NOT present at all are in cool BLUE.
        - Mid zones where crowd is less/sparse are in GREEN and vivid YELLOW.
        - Clustered / high-density crowd hotspots are in bright RED."""
        out = frame.copy()
        h, w = out.shape[:2]

        if density_map is not None:
            dh, dw = density_map.shape[:2]
            if (dh, dw) != (h, w):
                density_map = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)

            # Piecewise transfer function tailored to COLORMAP_JET
            xp = [0.0, 0.04, 0.16, 0.38, 0.65, 0.90, 1.25]
            yp = [0.0, 12.0, 75.0, 165.0, 205.0, 235.0, 255.0]
            density_u8 = np.interp(density_map, xp, yp).astype(np.uint8)

            # Update per-zone calibrated densities (replaces old left/right half hack)
            self._compute_zone_densities(density_map, count)

            self.red_hotspot_ratio = float((density_u8 >= 195).mean())
            self.yellow_zone_ratio = float(((density_u8 >= 135) & (density_u8 < 195)).mean())
            self.blue_free_ratio = float((density_u8 < 45).mean())

            # Apply COLORMAP_JET
            heatmap = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)
            if heatmap.shape[:2] != (h, w):
                heatmap = cv2.resize(heatmap, (w, h))

            # Translucent blending so video footage remains crisp under the heatmap
            out = cv2.addWeighted(out, 0.58, heatmap, 0.42, 0)

        # Draw Calibrated Ground Plane Polygon and Grid if active
        if self.calibration_mgr.is_calibrated and self.calibration_mgr.image_points and len(self.calibration_mgr.image_points) >= 4:
            quad = self.calibration_mgr.image_points
            pts = np.array(quad, dtype=np.int32)
            # Amber perspective ground boundary
            cv2.polylines(out, [pts], isClosed=True, color=(56, 189, 248), thickness=2, lineType=cv2.LINE_AA)
            for idx, pt in enumerate(pts):
                cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (56, 189, 248), -1)
            # Label on top anchor point
            anchor_x = max(10, int(pts[0][0]))
            anchor_y = max(50, int(pts[0][1]) - 6)
            cv2.putText(
                out,
                f"CALIBRATED FLOOR: {self.calibration_mgr.floor_area_m2:.1f}m2 [{self.calibrated_ground_count:.0f} ppl | {self.avg_density_m2:.2f} ppl/m2]",
                (anchor_x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (56, 189, 248),
                1,
                cv2.LINE_AA,
            )

        # Draw Control-Room HUD Header
        cv2.rectangle(out, (0, 0), (w, 36), (15, 23, 42), -1)

        if self.current_angle_result:
            angle_tag = f" | {self.current_angle_result.view_type} {round(self.current_angle_result.pitch_deg)}°"
        else:
            angle_tag = ""
        calib_tag = "CALIBRATED" if self.calibration_mgr.is_calibrated else "UNCALIBRATED"
        hud_text = (
            f"AI HEATMAP [{calib_tag}{angle_tag}] | {self.calibrated_ground_count:.0f} ppl in zone ({self.avg_density_m2:.2f} ppl/m2) | "
            f"Total: {count:.0f} | {self.processed_fps:.1f} FPS"
        )
        cv2.putText(
            out,
            hud_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )

        if self.drift_detected:
            cv2.rectangle(out, (0, h - 28), (w, h), (0, 0, 180), -1)
            cv2.putText(
                out,
                f"DRIFT WARNING: CAMERA MOVEMENT DETECTED (MATCH {self.drift_ratio * 100:.0f}%) - RE-CALIBRATE",
                (12, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out

    def _render_side_by_side(self, raw_frame: np.ndarray, heatmap_frame: np.ndarray) -> np.ndarray:
        """Horizontally stacks raw footage and AI heatmap with a divider."""
        h1, w1 = raw_frame.shape[:2]
        h2, w2 = heatmap_frame.shape[:2]
        target_h = 360
        
        w1_scaled = max(1, int(w1 * (target_h / h1)))
        w2_scaled = max(1, int(w2 * (target_h / h2)))
        
        f1 = cv2.resize(raw_frame, (w1_scaled, target_h), interpolation=cv2.INTER_LINEAR)
        f2 = cv2.resize(heatmap_frame, (w2_scaled, target_h), interpolation=cv2.INTER_LINEAR)
        
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
