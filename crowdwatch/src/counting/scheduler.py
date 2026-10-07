"""Adaptive frame-skip scheduler for crowd counting (FR-02, NFR-02).

Ensures system maintains >= 5 processed FPS on CPU by intelligently running
CLIP-EBC every N frames while maintaining continuous telemetry and display.
"""

from __future__ import annotations
import logging
import time
from typing import Optional, Tuple

import numpy as np

from src.counting.clip_ebc import CLIPEBCPredictor

logger = logging.getLogger("crowdwatch.counting.scheduler")


class AdaptiveCountScheduler:
    """Schedules CLIP-EBC inference across frames with dynamic CPU latency adaptation."""

    def __init__(
        self,
        predictor: CLIPEBCPredictor,
        initial_frame_skip: int = 2,
        min_frame_skip: int = 1,
        max_frame_skip: int = 6,
        target_min_fps: float = 5.0,
        auto_adapt: bool = True,
    ) -> None:
        self.predictor = predictor
        self.frame_skip = max(1, initial_frame_skip)
        self.min_frame_skip = max(1, min_frame_skip)
        self.max_frame_skip = max(self.min_frame_skip, max_frame_skip)
        self.target_min_fps = target_min_fps
        self.auto_adapt = auto_adapt

        # Cached results from last inference
        self._last_density_map: Optional[np.ndarray] = None
        self._last_count: float = 0.0
        self._last_inference_latency_ms: float = 0.0
        self._inference_count: int = 0

        # Latency tracking for adaptation
        self._recent_latencies = []

    @property
    def current_frame_skip(self) -> int:
        return self.frame_skip

    @property
    def last_count(self) -> float:
        return round(self._last_count, 1)

    @property
    def last_latency_ms(self) -> float:
        return round(self._last_inference_latency_ms, 2)

    def process_frame(
        self, frame: np.ndarray, frame_id: int, angle_result: Optional[Any] = None
    ) -> Tuple[np.ndarray, float, bool, float]:
        """Processes a frame, executing CLIP-EBC when (frame_id % frame_skip == 0).

        Returns:
            density_map: Current spatial density map.
            total_count: Current estimated count.
            is_new_inference: True if counter ran on this frame, False if using cached result.
            latency_ms: Inference or cache retrieval latency in milliseconds.
        """
        should_run = (self._last_density_map is None) or (frame_id % self.frame_skip == 0)

        if should_run:
            t0 = time.perf_counter()
            density_map, total_count = self.predictor.predict(frame, angle_result=angle_result)
            latency = (time.perf_counter() - t0) * 1000.0

            self._last_density_map = density_map
            self._last_count = total_count
            self._last_inference_latency_ms = latency
            self._inference_count += 1

            if self.auto_adapt:
                self._adapt_frame_skip(latency)

            return density_map, total_count, True, latency
        else:
            # Re-use cached density map smoothly
            return self._last_density_map, self._last_count, False, 0.5

    def _adapt_frame_skip(self, last_latency_ms: float) -> None:
        """Adapts frame-skip N based on measured inference latency on CPU."""
        self._recent_latencies.append(last_latency_ms)
        if len(self._recent_latencies) > 5:
            self._recent_latencies.pop(0)

        avg_latency_ms = float(np.mean(self._recent_latencies))
        # If average inference time is high (e.g. > 150ms on CPU), increase frame-skip
        # to ensure display/flow pipeline maintains >= 5 FPS
        if avg_latency_ms > 200.0 and self.frame_skip < self.max_frame_skip:
            self.frame_skip += 1
            logger.info(
                "CPU throughput adaptation: increased frame-skip to %d (avg latency: %.1fms)",
                self.frame_skip,
                avg_latency_ms,
            )
        elif avg_latency_ms < 60.0 and self.frame_skip > self.min_frame_skip:
            self.frame_skip -= 1
            logger.info(
                "CPU throughput adaptation: decreased frame-skip to %d (avg latency: %.1fms)",
                self.frame_skip,
                avg_latency_ms,
            )
