"""Automatic Camera Angle and Perspective Geometry Estimator.

Analyzes camera pitch angle, perspective scale gradient, and view type
(TOP_DOWN, HIGH_OBLIQUE, LOW_OBLIQUE) to dynamically auto-configure crowd
density estimation, pedestrian footprint scaling, and ground-plane homography.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("crowdwatch.calibration.angle")


@dataclass
class CameraAngleResult:
    """Estimated camera viewing geometry and perspective parameters."""
    view_type: str                   # 'TOP_DOWN', 'HIGH_OBLIQUE', 'LOW_OBLIQUE'
    pitch_deg: float                 # Estimated pitch in degrees (90 = straight down, 0 = horizontal)
    scale_gradient: float            # Ratio of foreground person area to background person area (>= 1.0)
    aspect_ratio_med: float          # Median height/width aspect ratio of pedestrian contours
    confidence: float                # Confidence score in [0.0, 1.0]
    label: str                       # Human-readable label (e.g. "Elevated Oblique (59°)")
    suggested_grid_m: Tuple[float, float] = (6.0, 10.0)  # (real_width_m, real_height_m)
    suggested_quad: Optional[List[List[float]]] = None   # 4 image points [[x, y], ...] for ground trapezoid


class CameraAngleEstimator:
    """Estimates camera pitch angle and perspective scale from visual geometry."""

    def __init__(self, history_size: int = 7) -> None:
        self.history_size = max(1, history_size)
        self._pitch_history: List[float] = []
        self._scale_history: List[float] = []
        self._ar_history: List[float] = []
        self._last_result: Optional[CameraAngleResult] = None

    def reset(self) -> None:
        """Reset historical smoothing on video source switch."""
        self._pitch_history.clear()
        self._scale_history.clear()
        self._ar_history.clear()
        self._last_result = None

    def estimate(self, frame_bgr: np.ndarray) -> CameraAngleResult:
        """Estimates camera pitch, view type, and perspective parameters from a frame.

        Args:
            frame_bgr: Input video frame (uint8 BGR).

        Returns:
            CameraAngleResult with classified view type and auto-calibrated geometry.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            if self._last_result:
                return self._last_result
            return self._default_top_down(640, 360)

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Morphological pedestrian/foreground saliency
        k_size = max(9, int(min(h, w) * 0.04) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        saliency = cv2.GaussianBlur(cv2.add(bh, th), (7, 7), 1.5)

        p_std = float(np.std(saliency))
        thresh = float(np.median(saliency)) + 0.38 * p_std
        ped_mask = (saliency > thresh).astype(np.uint8)

        # 2. Extract pedestrian contour geometry: aspect ratio and vertical position
        contours, _ = cv2.findContours(ped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aspect_ratios: List[float] = []
        y_positions: List[float] = []
        areas: List[float] = []

        for c in contours:
            x, y, bw, bh_box = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area >= 35 and bw >= 4 and bh_box >= 4:
                aspect_ratios.append(bh_box / float(bw))
                y_positions.append((y + bh_box / 2.0) / float(h))
                areas.append(area)

        # 3. Contour-based aspect ratio analysis
        # - Top-down views: people appear as circular heads/shoulders (AR ~ 0.9 - 1.15)
        # - Oblique views: standing people project vertically (AR ~ 1.25 - 2.5)
        median_ar = float(np.median(aspect_ratios)) if aspect_ratios else 1.0
        ar_80 = float(np.percentile(aspect_ratios, 80)) if len(aspect_ratios) >= 5 else median_ar

        # 4. Vertical scale gradient (area vs y depth position)
        if len(y_positions) >= 6:
            y_arr = np.array(y_positions)
            a_arr = np.array(areas)
            top_mask = y_arr < 0.40
            bot_mask = y_arr > 0.60
            if np.sum(top_mask) >= 2 and np.sum(bot_mask) >= 2:
                top_med = float(np.median(a_arr[top_mask]))
                bot_med = float(np.median(a_arr[bot_mask]))
                instant_scale_grad = float(np.clip(bot_med / max(top_med, 10.0), 1.0, 7.0))
            else:
                instant_scale_grad = 1.0
        else:
            instant_scale_grad = 1.0

        # 5. Temporal smoothing across frames
        self._scale_history.append(instant_scale_grad)
        self._ar_history.append(median_ar)
        if len(self._scale_history) > self.history_size:
            self._scale_history.pop(0)
            self._ar_history.pop(0)

        smooth_scale_grad = float(np.median(self._scale_history))
        smooth_ar = float(np.median(self._ar_history))

        # 6. Camera pitch angle classification
        if smooth_scale_grad <= 1.30 and smooth_ar <= 1.15 and ar_80 <= 1.60:
            view_type = "TOP_DOWN"
            pitch_deg = round(85.0 - (smooth_scale_grad - 1.0) * 15.0, 1)
            label = f"Top-Down / Overhead ({round(pitch_deg)}°)"
            real_w, real_h = 8.0, 6.0
            # Centered rectangular ground plane
            quad = [
                [round(0.12 * w, 1), round(0.12 * h, 1)],
                [round(0.88 * w, 1), round(0.12 * h, 1)],
                [round(0.88 * w, 1), round(0.88 * h, 1)],
                [round(0.12 * w, 1), round(0.88 * h, 1)],
            ]
            confidence = 0.92
        elif smooth_scale_grad >= 2.80 or ar_80 >= 2.40:
            view_type = "LOW_OBLIQUE"
            pitch_deg = round(max(20.0, 42.0 - (smooth_scale_grad - 2.8) * 4.5), 1)
            label = f"Low Oblique / Concourse ({round(pitch_deg)}°)"
            real_w, real_h = 5.0, 14.0
            # Deep trapezoidal ground plane
            quad = [
                [round(0.28 * w, 1), round(0.38 * h, 1)],
                [round(0.72 * w, 1), round(0.38 * h, 1)],
                [round(0.94 * w, 1), round(0.92 * h, 1)],
                [round(0.06 * w, 1), round(0.92 * h, 1)],
            ]
            confidence = 0.88
        else:
            view_type = "HIGH_OBLIQUE"
            pitch_deg = round(65.0 - (smooth_scale_grad - 1.3) * 12.0, 1)
            label = f"Elevated Oblique ({round(pitch_deg)}°)"
            real_w, real_h = 6.0, 10.0
            # Moderate perspective trapezoid
            quad = [
                [round(0.20 * w, 1), round(0.32 * h, 1)],
                [round(0.80 * w, 1), round(0.32 * h, 1)],
                [round(0.92 * w, 1), round(0.90 * h, 1)],
                [round(0.08 * w, 1), round(0.90 * h, 1)],
            ]
            confidence = 0.90

        result = CameraAngleResult(
            view_type=view_type,
            pitch_deg=pitch_deg,
            scale_gradient=round(smooth_scale_grad, 2),
            aspect_ratio_med=round(smooth_ar, 2),
            confidence=confidence,
            label=label,
            suggested_grid_m=(real_w, real_h),
            suggested_quad=quad,
        )
        self._last_result = result
        return result

    def _default_top_down(self, w: int, h: int) -> CameraAngleResult:
        """Fallback default geometry for empty frames."""
        return CameraAngleResult(
            view_type="TOP_DOWN",
            pitch_deg=85.0,
            scale_gradient=1.0,
            aspect_ratio_med=1.0,
            confidence=0.5,
            label="Top-Down / Overhead (85°)",
            suggested_grid_m=(8.0, 6.0),
            suggested_quad=[
                [0.15 * w, 0.15 * h],
                [0.85 * w, 0.15 * h],
                [0.85 * w, 0.85 * h],
                [0.15 * w, 0.85 * h],
            ],
        )
