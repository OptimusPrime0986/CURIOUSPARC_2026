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

from src.calibration.angle_estimator import CameraAngleEstimator, CameraAngleResult
from src.calibration.homography import HomographyCalibrator

logger = logging.getLogger("crowdwatch.calibration.manager")


class CalibrationManager:
    """Manages calibration lifecycle: create, save, load, and expose to the pipeline."""

    def __init__(self, calibration_path: str | Path = "config/calibration.json") -> None:
        self._path = Path(calibration_path)
        self._calibrator = HomographyCalibrator()
        self._angle_estimator = CameraAngleEstimator()
        self._latest_angle_result: Optional[CameraAngleResult] = None
        self._is_manual: bool = False
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

    @property
    def image_points(self) -> List[List[float]]:
        """Active 4-point image coordinates of calibrated ground plane."""
        if self._calibrator and self._calibrator.image_points is not None:
            return self._calibrator.image_points.tolist()
        return []

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
        self._is_manual = not name.startswith("Auto-Configured")
        self._timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        return result

    def auto_calibrate_from_angle(
        self, frame: np.ndarray, force: bool = False
    ) -> CameraAngleResult:
        """Auto-configure ground plane homography and perspective parameters from camera angle."""
        angle_res = self._angle_estimator.estimate(frame)
        self._latest_angle_result = angle_res

        # Only apply ground plane quad if forced or if completely uncalibrated and not manual
        if force or (not self.is_calibrated and not self._is_manual):
            self.calibrate(
                image_points=angle_res.suggested_quad,
                real_width_m=angle_res.suggested_grid_m[0],
                real_height_m=angle_res.suggested_grid_m[1],
                reference_frame=frame,
                name=f"Auto-Configured ({angle_res.label})",
            )
            self._is_manual = False

        return angle_res

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Persist current calibration to the JSON file and save reference frame if available."""
        if not self._calibrator.is_valid:
            raise RuntimeError("No valid calibration to save")

        reproj = self._calibrator.reprojection_error()
        confidence = max(0.0, 1.0 - min(0.5, reproj / 10.0))

        ref_filename = None
        if self._calibrator.reference_frame is not None:
            ref_path = self._path.parent / f"{self._path.stem}_ref.jpg"
            cv2.imwrite(str(ref_path), self._calibrator.reference_frame)
            ref_filename = ref_path.name
            logger.info("Saved calibration reference frame to %s", ref_path)

        data: Dict[str, Any] = {
            "calibration_name": self._calibration_name,
            "units": "meters",
            "real_width_m": self._real_width_m,
            "real_height_m": self._real_height_m,
            "image_points": self._calibrator.image_points.tolist(),
            "world_points": self._calibrator.world_points.tolist(),
            "homography_matrix": self._calibrator.homography_matrix.tolist(),
            "reference_frame_hash": self._calibrator._reference_frame_hash,
            "reference_frame_file": ref_filename,
            "timestamp": self._timestamp,
            "confidence_score": round(confidence, 4),
            "reprojection_error_px": round(reproj, 4),
            "notes": "4-point perspective ground plane calibration for converting pixel areas into real m²",
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Calibration saved to %s", self._path)
        return self._path

    def load(self) -> Dict[str, Any]:
        """Load calibration from the JSON file, re-initialise calibrator, and restore reference frame."""
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

        # Restore reference frame if present
        ref_frame = None
        ref_file = data.get("reference_frame_file") or f"{self._path.stem}_ref.jpg"
        ref_path = self._path.parent / ref_file
        if ref_path.exists():
            ref_frame = cv2.imread(str(ref_path))

        result = self._calibrator.calibrate(
            image_points=image_points,
            real_width_m=real_width,
            real_height_m=real_height,
            reference_frame=ref_frame,
        )

        self._real_width_m = real_width
        self._real_height_m = real_height
        self._calibration_name = data.get("calibration_name", "Loaded")
        self._timestamp = data.get("timestamp")
        self._is_manual = True
        if data.get("reference_frame_hash"):
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
        reproj = self._calibrator.reprojection_error()
        angle_dict = {
            "view_type": self._latest_angle_result.view_type if self._latest_angle_result else "OBLIQUE",
            "pitch_deg": self._latest_angle_result.pitch_deg if self._latest_angle_result else 55.0,
            "scale_gradient": self._latest_angle_result.scale_gradient if self._latest_angle_result else 1.8,
            "label": self._latest_angle_result.label if self._latest_angle_result else "Elevated Oblique (55°)",
            "auto_configured": not self._is_manual,
        }
        return {
            "calibrated": True,
            "name": self._calibration_name,
            "real_width_m": self._real_width_m,
            "real_height_m": self._real_height_m,
            "floor_area_m2": self.floor_area_m2,
            "reprojection_error_px": round(reproj, 4),
            "confidence_score": max(0.0, round(1.0 - min(0.5, reproj / 10.0), 4)),
            "timestamp": self._timestamp,
            "reference_frame_hash": self._calibrator._reference_frame_hash,
            "has_reference_frame": self._calibrator.reference_frame is not None or self._calibrator._reference_orb_descriptors is not None,
            "is_manual": self._is_manual,
            "camera_angle": angle_dict,
        }

    def check_drift(self, current_frame: np.ndarray, match_threshold: float = 0.40) -> Tuple[bool, float]:
        """Compare current frame to reference frame. Returns (drift_detected, match_ratio)."""
        if not self.is_calibrated:
            return False, 1.0
        return self._calibrator.check_camera_drift(current_frame, match_threshold=match_threshold)

    # ------------------------------------------------------------------
    # Zone-level density helpers
    # ------------------------------------------------------------------

    def zone_density(
        self,
        density_map: np.ndarray,
        total_count: float,
        zone_polygon_px: List[List[float]],
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """Compute calibrated zone metrics.

        Returns:
            ``(people_in_zone, zone_area_m2, density_ppl_per_m2)``
        """
        if not self.is_calibrated:
            return 0.0, 0.0, 0.0

        people, density = self._calibrator.density_per_m2(
            density_map, total_count, zone_polygon_px, depth_map=depth_map,
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
