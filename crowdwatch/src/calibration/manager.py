"""Calibration persistence and lifecycle management.

Wraps :class:`HomographyCalibrator` with JSON save/load, status reporting,
and a convenience API consumed by the pipeline service.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.calibration.homography import HomographyCalibrator

logger = logging.getLogger("crowdwatch.calibration.manager")


class CalibrationManager:
    """Manages calibration lifecycle: create, save, load, and expose to the pipeline."""

    def __init__(self, calibration_path: str | Path = "config/calibration.json") -> None:
        self._path = Path(calibration_path)
        self._calibrator = HomographyCalibrator()
        self._real_width_m: float = 0.0
        self._real_height_m: float = 0.0
        self._calibration_name: str = ""
        self._timestamp: Optional[str] = None

        # Try loading existing calibration on startup
        if self._path.exists():
            try:
                self.load()
            except Exception as e:
                logger.warning("Could not load calibration from %s: %s", self._path, e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator.is_valid

    @property
    def calibrator(self) -> HomographyCalibrator:
        return self._calibrator

    @property
    def floor_area_m2(self) -> float:
        """Total calibrated floor area in m² (the reference rectangle)."""
        if self.is_calibrated:
            return self._real_width_m * self._real_height_m
        return 0.0

    @property
    def real_width_m(self) -> float:
        return self._real_width_m

    @property
    def real_height_m(self) -> float:
        return self._real_height_m

    # ------------------------------------------------------------------
    # Calibration workflow
    # ------------------------------------------------------------------

    def calibrate(
        self,
        image_points: List[List[float]],
        real_width_m: float,
        real_height_m: float,
        reference_frame: Optional[np.ndarray] = None,
        name: str = "Operator Calibration",
    ) -> Dict[str, Any]:
        """Run a new calibration and return results (does *not* auto-save)."""
        result = self._calibrator.calibrate(
            image_points=image_points,
            real_width_m=real_width_m,
            real_height_m=real_height_m,
            reference_frame=reference_frame,
        )
        self._real_width_m = real_width_m
        self._real_height_m = real_height_m
        self._calibration_name = name
        self._timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Persist current calibration to the JSON file."""
        if not self._calibrator.is_valid:
            raise RuntimeError("No valid calibration to save")

        reproj = self._calibrator.reprojection_error()
        confidence = max(0.0, 1.0 - min(0.5, reproj / 10.0))

        data: Dict[str, Any] = {
            "calibration_name": self._calibration_name,
            "units": "meters",
            "real_width_m": self._real_width_m,
            "real_height_m": self._real_height_m,
            "image_points": self._calibrator.image_points.tolist(),
            "world_points": self._calibrator.world_points.tolist(),
            "homography_matrix": self._calibrator.homography_matrix.tolist(),
            "reference_frame_hash": self._calibrator._reference_frame_hash,
            "timestamp": self._timestamp,
            "confidence_score": round(confidence, 4),
            "notes": "4-point perspective ground plane calibration for converting pixel areas into real m²",
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Calibration saved to %s", self._path)
        return self._path

    def load(self) -> Dict[str, Any]:
        """Load calibration from the JSON file and re-initialise the calibrator."""
        if not self._path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        image_points = data.get("image_points", [])
        real_width = data.get("real_width_m", 0)
        real_height = data.get("real_height_m", 0)

        if len(image_points) != 4 or real_width <= 0 or real_height <= 0:
            logger.warning("Calibration file has incomplete data — skipping load")
            return data

        result = self._calibrator.calibrate(
            image_points=image_points,
            real_width_m=real_width,
            real_height_m=real_height,
        )

        self._real_width_m = real_width
        self._real_height_m = real_height
        self._calibration_name = data.get("calibration_name", "Loaded")
        self._timestamp = data.get("timestamp")
        self._calibrator._reference_frame_hash = data.get("reference_frame_hash")

        logger.info(
            "Loaded calibration '%s': %.1f×%.1f m, reproj_err=%.4f px",
            self._calibration_name,
            real_width,
            real_height,
            result["reprojection_error_px"],
        )
        return data

    # ------------------------------------------------------------------
    # Status / API helpers
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return calibration status dict for the API / dashboard."""
        if not self.is_calibrated:
            return {
                "calibrated": False,
                "message": "No calibration available. Mark 4 ground-plane points to calibrate.",
            }
        return {
            "calibrated": True,
            "name": self._calibration_name,
            "real_width_m": self._real_width_m,
            "real_height_m": self._real_height_m,
            "floor_area_m2": self.floor_area_m2,
            "timestamp": self._timestamp,
            "reference_frame_hash": self._calibrator._reference_frame_hash,
        }

    # ------------------------------------------------------------------
    # Zone-level density helpers
    # ------------------------------------------------------------------

    def zone_density(
        self,
        density_map: np.ndarray,
        total_count: float,
        zone_polygon_px: List[List[float]],
    ) -> Tuple[float, float, float]:
        """Compute calibrated zone metrics.

        Returns:
            ``(people_in_zone, zone_area_m2, density_ppl_per_m2)``
        """
        if not self.is_calibrated:
            return 0.0, 0.0, 0.0

        people, density = self._calibrator.density_per_m2(
            density_map, total_count, zone_polygon_px,
        )
        area = self._calibrator.polygon_area_m2(zone_polygon_px)
        return people, round(area, 2), density

    def preview_grid(self, grid_spacing_m: float = 1.0) -> List[Dict[str, Any]]:
        """Return preview grid lines as JSON-serialisable list of segments."""
        if not self.is_calibrated:
            return []
        raw = self._calibrator.generate_preview_grid(
            real_width_m=self._real_width_m,
            real_height_m=self._real_height_m,
            grid_spacing_m=grid_spacing_m,
        )
        return [
            {"start": [round(s[0], 1), round(s[1], 1)],
             "end":   [round(e[0], 1), round(e[1], 1)]}
            for s, e in raw
        ]
