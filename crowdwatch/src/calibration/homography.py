"""Ground-plane homography calibration for pixel-to-metre conversion.

Computes a perspective transform from 4 operator-marked image points to
known real-world coordinates, enabling conversion of pixel-space density
maps into calibrated people/m² measurements.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("crowdwatch.calibration.homography")


class HomographyCalibrator:
    """Computes and applies a ground-plane homography from 4 image↔world point pairs."""

    def __init__(self) -> None:
        self._H: Optional[np.ndarray] = None           # 3×3 image → world
        self._H_inv: Optional[np.ndarray] = None        # 3×3 world → image
        self._image_points: Optional[np.ndarray] = None  # (4, 2) float32
        self._world_points: Optional[np.ndarray] = None  # (4, 2) float32
        self._reference_frame_hash: Optional[str] = None
        self._reference_frame: Optional[np.ndarray] = None
        self._reference_orb_descriptors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True when a homography has been computed successfully."""
        return self._H is not None

    @property
    def homography_matrix(self) -> Optional[np.ndarray]:
        """Return a copy of the 3×3 image→world homography, or None."""
        return self._H.copy() if self._H is not None else None

    @property
    def image_points(self) -> Optional[np.ndarray]:
        return self._image_points.copy() if self._image_points is not None else None

    @property
    def world_points(self) -> Optional[np.ndarray]:
        return self._world_points.copy() if self._world_points is not None else None

    @property
    def reference_frame(self) -> Optional[np.ndarray]:
        return self._reference_frame.copy() if self._reference_frame is not None else None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        image_points: List[List[float]],
        real_width_m: float,
        real_height_m: float,
        reference_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Compute homography from 4 image points to a real-world rectangle.

        Args:
            image_points: 4 pixel coordinates ``[[x1,y1], …]`` in clockwise
                order (top-left, top-right, bottom-right, bottom-left of
                the ground rectangle).
            real_width_m:  Width of the ground rectangle in metres.
            real_height_m: Height (depth) of the ground rectangle in metres.
            reference_frame: Optional BGR frame stored for drift detection.

        Returns:
            Dict with ``area_m2``, ``reprojection_error_px``, ``valid``.

        Raises:
            ValueError: If fewer than 4 points or non-positive dimensions.
        """
        if len(image_points) != 4:
            raise ValueError(f"Exactly 4 image points required, got {len(image_points)}")
        if real_width_m <= 0 or real_height_m <= 0:
            raise ValueError(f"Dimensions must be positive: {real_width_m}×{real_height_m}")

        self._image_points = np.array(image_points, dtype=np.float32)
        self._world_points = np.array(
            [
                [0.0, 0.0],
                [real_width_m, 0.0],
                [real_width_m, real_height_m],
                [0.0, real_height_m],
            ],
            dtype=np.float32,
        )

        self._H = cv2.getPerspectiveTransform(self._image_points, self._world_points)
        self._H_inv = cv2.getPerspectiveTransform(self._world_points, self._image_points)

        reproj_error = self.reprojection_error()

        if reference_frame is not None:
            self._store_reference(reference_frame)

        calibrated_area = real_width_m * real_height_m
        logger.info(
            "Calibration complete: %.1f×%.1f m = %.1f m², reproj_error=%.4f px",
            real_width_m,
            real_height_m,
            calibrated_area,
            reproj_error,
        )

        return {
            "area_m2": calibrated_area,
            "reprojection_error_px": round(reproj_error, 4),
            "valid": True,
        }

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def image_to_world(self, points: List[List[float]]) -> List[List[float]]:
        """Transform pixel coordinates to world coordinates (metres)."""
        if self._H is None:
            raise RuntimeError("Calibration not performed yet")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._H)
        return transformed.reshape(-1, 2).tolist()

    def world_to_image(self, points: List[List[float]]) -> List[List[float]]:
        """Transform world coordinates (metres) to pixel coordinates."""
        if self._H_inv is None:
            raise RuntimeError("Calibration not performed yet")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._H_inv)
        return transformed.reshape(-1, 2).tolist()

    # ------------------------------------------------------------------
    # Area & density
    # ------------------------------------------------------------------

    def polygon_area_m2(self, image_polygon: List[List[float]]) -> float:
        """Compute the real-world area (m²) of a polygon given in image pixels.

        Transforms vertices to world coordinates then applies the Shoelace formula.
        """
        if len(image_polygon) < 3:
            raise ValueError("Polygon needs at least 3 vertices")
        world_vertices = self.image_to_world(image_polygon)
        return self._shoelace_area(world_vertices)

    def people_in_zone(
        self,
        density_map: np.ndarray,
        zone_polygon_px: List[List[float]],
    ) -> float:
        """Fraction of total density that falls inside *zone_polygon_px*.

        Multiply the returned value by ``total_count`` to get the estimated
        number of people in the zone.
        """
        h, w = density_map.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(zone_polygon_px, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        zone_sum = float(np.sum(density_map[mask > 0]))
        total_sum = float(np.sum(density_map))

        if total_sum < 1e-6:
            return 0.0
        return zone_sum / total_sum

    def density_per_m2(
        self,
        density_map: np.ndarray,
        total_count: float,
        zone_polygon_px: List[List[float]],
        depth_map: Optional[np.ndarray] = None,
        return_meta: bool = False,
    ) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
        """Compute calibrated people/m² for a polygon zone with optional depth check.

        Args:
            density_map: 2D array of crowd density.
            total_count: Total head count for scaling.
            zone_polygon_px: Polygon vertices in pixel coordinates.
            depth_map: Optional depth map (H, W) used for sanity check.
            return_meta: If True, returns (people, density, metadata_dict).

        Returns:
            ``(people_in_zone, density_per_m2)`` or with metadata dict.
        """
        fraction = self.people_in_zone(density_map, zone_polygon_px)
        people = round(fraction * total_count, 2)
        area = self.polygon_area_m2(zone_polygon_px)

        if area < 0.01:
            ppl_m2 = 0.0
        else:
            ppl_m2 = round(people / area, 2)

        meta: Dict[str, Any] = {
            "area_m2": round(area, 2),
            "density_fraction": round(fraction, 4),
            "depth_sanity_passed": True,
            "depth_confidence": 1.0,
        }

        if depth_map is not None:
            dh, dw = depth_map.shape[:2]
            dmask = np.zeros((dh, dw), dtype=np.uint8)
            pts = np.array(zone_polygon_px, dtype=np.int32)
            cv2.fillPoly(dmask, [pts], 255)
            z_depth = depth_map[dmask > 0]
            if z_depth.size > 0:
                mean_d = float(np.mean(z_depth))
                std_d = float(np.std(z_depth))
                valid = bool(np.isfinite(mean_d) and mean_d > 0.0)
                cv_d = (std_d / mean_d) if mean_d > 1e-4 else 1.0
                conf = float(np.clip(1.0 - cv_d * 0.4, 0.2, 1.0))
                meta.update({
                    "depth_sanity_passed": valid,
                    "depth_mean": round(mean_d, 3),
                    "depth_std": round(std_d, 3),
                    "depth_confidence": round(conf, 3),
                })

        if return_meta:
            return people, ppl_m2, meta
        return people, ppl_m2

    # ------------------------------------------------------------------
    # Preview grid
    # ------------------------------------------------------------------

    def generate_preview_grid(
        self,
        real_width_m: float,
        real_height_m: float,
        grid_spacing_m: float = 1.0,
    ) -> List[Tuple[List[float], List[float]]]:
        """Generate 1 m × 1 m grid lines in image space for visual verification.

        Returns:
            List of ``(start_px, end_px)`` line segments.
        """
        if self._H_inv is None:
            return []

        lines: List[Tuple[List[float], List[float]]] = []

        # Vertical lines (constant x in world)
        x = 0.0
        while x <= real_width_m + 1e-4:
            s = self.world_to_image([[x, 0.0]])[0]
            e = self.world_to_image([[x, real_height_m]])[0]
            lines.append((s, e))
            x += grid_spacing_m

        # Horizontal lines (constant y in world)
        y = 0.0
        while y <= real_height_m + 1e-4:
            s = self.world_to_image([[0.0, y]])[0]
            e = self.world_to_image([[real_width_m, y]])[0]
            lines.append((s, e))
            y += grid_spacing_m

        return lines

    # ------------------------------------------------------------------
    # Camera drift detection
    # ------------------------------------------------------------------

    def check_camera_drift(
        self,
        current_frame: np.ndarray,
        match_threshold: float = 0.40,
    ) -> Tuple[bool, float]:
        """Compare *current_frame* to the calibration reference.

        Returns:
            ``(drift_detected, match_ratio)`` — *match_ratio* near 1.0 means
            no drift; below *match_threshold* means the camera likely moved.
        """
        if self._reference_orb_descriptors is None:
            return False, 1.0

        gray = (
            cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            if len(current_frame.shape) == 3
            else current_frame
        )
        orb = cv2.ORB_create(nfeatures=500)
        _, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or self._reference_orb_descriptors is None:
            return False, 1.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.knnMatch(self._reference_orb_descriptors, descriptors, k=2)
        except cv2.error:
            return False, 1.0

        good = 0
        for pair in matches:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good += 1

        total = max(len(matches), 1)
        match_ratio = good / total
        drift_detected = match_ratio < match_threshold

        if drift_detected:
            logger.warning(
                "Camera drift detected: match_ratio=%.3f (threshold=%.3f). "
                "Calibration may be invalid.",
                match_ratio,
                match_threshold,
            )

        return drift_detected, round(match_ratio, 4)

    # ------------------------------------------------------------------
    # Quality metric
    # ------------------------------------------------------------------

    def reprojection_error(self) -> float:
        """Mean round-trip reprojection error in pixels."""
        if self._H is None or self._image_points is None:
            return float("inf")

        world_pts = self.image_to_world(self._image_points.tolist())
        reprojected = self.world_to_image(world_pts)
        errors = np.linalg.norm(np.array(reprojected) - self._image_points, axis=1)
        return float(np.mean(errors))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shoelace_area(vertices: List[List[float]]) -> float:
        """Polygon area via the Shoelace (Gauss) formula."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    def _store_reference(self, frame: np.ndarray) -> None:
        """Store ORB descriptors + hash of the reference frame."""
        self._reference_frame = frame.copy()
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )
        self._reference_frame_hash = hashlib.md5(
            cv2.resize(gray, (160, 90)).tobytes()
        ).hexdigest()

        orb = cv2.ORB_create(nfeatures=500)
        _, self._reference_orb_descriptors = orb.detectAndCompute(gray, None)
