"""Resilient video ingestion stream for webcam, RTSP, and video files (FR-01, NFR-06).

Provides threaded non-blocking frame capture, automatic reconnection within 30s,
and rolling FPS calculation.
"""

from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("crowdwatch.capture")


class VideoStream:
    """Threaded OpenCV video capture with auto-reconnect and FPS tracking."""

    def __init__(
        self,
        source: Union[int, str, Path],
        target_fps: int = 15,
        reconnect_timeout_sec: float = 30.0,
        reconnect_interval_sec: float = 2.0,
        loop_file: bool = True,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        self.source = source
        self.target_fps = max(1, target_fps)
        self.frame_delay = 1.0 / self.target_fps
        self.reconnect_timeout_sec = reconnect_timeout_sec
        self.reconnect_interval_sec = reconnect_interval_sec
        self.loop_file = loop_file
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._is_connected = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame buffer
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._last_frame_time: float = 0.0

        # Diagnostics & FPS
        self._fps: float = 0.0
        self._fps_history = []
        self._fps_window_size = 20
        self._last_fps_calc_time = time.time()
        self._disconnect_time: Optional[float] = None
        self._reconnect_attempts: int = 0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def fps(self) -> float:
        return round(self._fps, 2)

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def start(self) -> "VideoStream":
        """Starts the background capture thread."""
        if self._is_running:
            return self

        self._is_running = True
        self._connect()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="VideoStreamThread")
        self._thread.start()
        logger.info("VideoStream started for source: %s", self.source)
        return self

    def stop(self) -> None:
        """Stops capture and releases hardware resources."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._disconnect()
        logger.info("VideoStream stopped")

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Returns (is_connected, latest_frame, frame_id). Thread-safe non-blocking."""
        with self._lock:
            if self._latest_frame is None:
                return self._is_connected, None, self._frame_id
            return self._is_connected, self._latest_frame.copy(), self._frame_id

    def _connect(self) -> bool:
        """Attempts to open the video capture source."""
        self._disconnect()
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        elif isinstance(src, Path):
            src = str(src)

        try:
            self._cap = cv2.VideoCapture(src)
            if self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._is_connected = True
                        self._disconnect_time = None
                        self._reconnect_attempts = 0
                    logger.info("Successfully connected to video source: %s", self.source)
                    if self.on_reconnect:
                        try:
                            self.on_reconnect()
                        except Exception as e:
                            logger.error("Error in on_reconnect callback: %s", e)
                    return True
        except Exception as e:
            logger.warning("Error opening video source %s: %s", self.source, e)

        self._handle_disconnect()
        return False

    def _disconnect(self) -> None:
        """Closes OpenCV video capture handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _handle_disconnect(self) -> None:
        """Handles connection loss and triggers callbacks."""
        with self._lock:
            was_connected = self._is_connected
            self._is_connected = False
            if was_connected or self._disconnect_time is None:
                self._disconnect_time = time.time()
                logger.warning("Video source disconnected: %s. Entering auto-reconnect mode...", self.source)
                if self.on_disconnect and was_connected:
                    try:
                        self.on_disconnect()
                    except Exception as e:
                        logger.error("Error in on_disconnect callback: %s", e)

    def _capture_loop(self) -> None:
        """Continuous background capture loop with auto-reconnection (NFR-06)."""
        while self._is_running:
            loop_start = time.time()

            if not self._is_connected or self._cap is None or not self._cap.isOpened():
                # Reconnect handler
                elapsed_disconnect = time.time() - (self._disconnect_time or time.time())
                if elapsed_disconnect > self.reconnect_timeout_sec:
                    logger.error(
                        "Auto-reconnect timeout exceeded (%.1fs > %.1fs) for %s",
                        elapsed_disconnect,
                        self.reconnect_timeout_sec,
                        self.source,
                    )

                self._reconnect_attempts += 1
                logger.info("Reconnection attempt #%d for %s", self._reconnect_attempts, self.source)
                success = self._connect()
                if not success:
                    time.sleep(self.reconnect_interval_sec)
                    continue

            # Read frame
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Video file reached end? Loop if configured
                is_file = isinstance(self.source, (str, Path)) and Path(str(self.source)).is_file()
                if is_file and self.loop_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        # Re-open capture to ensure loop on all codecs
                        self._connect()
                        if self._cap:
                            ret, frame = self._cap.read()

                if not ret or frame is None:
                    self._handle_disconnect()
                    time.sleep(self.reconnect_interval_sec)
                    continue

            # Normalize high-resolution mobile videos (e.g., 1080p / 4K) to max 640px for real-time inference
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                scale = 640.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Update latest frame & calculate FPS
            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1
                dt = now - self._last_frame_time
                self._last_frame_time = now

            if dt > 0:
                self._fps_history.append(1.0 / dt)
                if len(self._fps_history) > self._fps_window_size:
                    self._fps_history.pop(0)
                self._fps = float(np.mean(self._fps_history))

            # Maintain target capture FPS
            elapsed = time.time() - loop_start
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
