"""Depth Anything V2 Small integration (Apache-2.0).

Used for secondary scale correction, perspective gradient estimation,
and as an input into the confidence indicator (FR-04, FR-15).
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("crowdwatch.counting.depth_anything")


class DepthAnythingV2Predictor:
    """Wrapper for Depth Anything V2 Small depth estimation."""

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
        local_weights_path: Optional[Union[str, Path]] = None,
    ):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = None
        self.processor = None

        logger.info("Initializing Depth Anything V2 Small on %s (License: Apache-2.0)", self.device)
        if TRANSFORMERS_AVAILABLE:
            try:
                try:
                    self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                    self.model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
                except Exception:
                    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                        raise
                    import socket
                    orig_timeout = socket.getdefaulttimeout()
                    try:
                        socket.setdefaulttimeout(4.0)
                        self.processor = AutoImageProcessor.from_pretrained(model_id)
                        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
                    finally:
                        socket.setdefaulttimeout(orig_timeout)

                self.model.to(self.device)
                self.model.eval()
                logger.info("Depth Anything V2 Small loaded successfully from %s", model_id)
            except Exception as e:
                logger.warning("Could not load HuggingFace Depth Anything V2 (%s); using gradient fallback", e)
                self.model = None
        else:
            logger.warning("Transformers not available; using fallback depth estimator")

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Runs monocular depth estimation.

        Returns:
            depth_map: 2D numpy array (float32, 0 to 1 normalized, where 1 is closer and 0 is farther).
            scale_correction: Estimated scale gradient factor from bottom to top of frame.
            depth_confidence: Float in [0.0, 1.0] indicating depth signal clarity.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty or None")

        h, w = frame_bgr.shape[:2]

        if self.model is not None and self.processor is not None:
            # OpenCV BGR -> RGB
            rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            # Interpolate to original frame resolution
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            raw_depth = prediction.squeeze().cpu().numpy().astype(np.float32)

            # Normalize depth to 0..1
            d_min, d_max = raw_depth.min(), raw_depth.max()
            if d_max - d_min > 1e-6:
                depth_map = (raw_depth - d_min) / (d_max - d_min)
            else:
                depth_map = np.zeros_like(raw_depth)
        else:
            # Fallback realistic perspective gradient: objects at the bottom of the image are closer (near 1.0)
            # and objects at the top are further away (near 0.0)
            y_indices = np.linspace(0.1, 1.0, h, dtype=np.float32).reshape(-1, 1)
            depth_map = np.repeat(y_indices, w, axis=1)

        # Scale correction: ratio of average near depth (bottom 20%) to far depth (top 20%)
        top_slice = depth_map[: int(h * 0.2), :]
        bottom_slice = depth_map[int(h * 0.8) :, :]
        top_mean = float(np.mean(top_slice)) if top_slice.size > 0 else 0.1
        bottom_mean = float(np.mean(bottom_slice)) if bottom_slice.size > 0 else 1.0
        scale_correction = float(np.clip(bottom_mean / max(top_mean, 1e-4), 0.5, 4.0))

        # Depth confidence metric: scene depth variance & dynamic range
        depth_std = float(np.std(depth_map))
        depth_confidence = float(np.clip(depth_std * 3.5, 0.2, 0.98))

        return depth_map, scale_correction, depth_confidence
