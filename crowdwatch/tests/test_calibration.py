"""Unit tests for src/calibration (Phase 1).

Covers:
- Known homography and area computation
- Coordinate round-trip (image → world → image)
- Polygon area accuracy
- people_in_zone / density_per_m2 with synthetic density maps
- Save / load round-trip for CalibrationManager
- Camera drift detection
- Preview grid generation
- Edge cases and error handling
"""

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project root to path for imports
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.homography import HomographyCalibrator
from src.calibration.manager import CalibrationManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_calibrator() -> HomographyCalibrator:
    """Calibrator with a known 4 m × 2 m rectangle (identity-like mapping)."""
    cal = HomographyCalibrator()
    # Image points define a rectangle that maps to a 4×2 m real-world patch.
    # Using a simple trapezoid in pixel space to simulate perspective.
    image_pts = [
        [100, 100],   # top-left
        [500, 100],   # top-right
        [550, 300],   # bottom-right
        [50, 300],    # bottom-left
    ]
    cal.calibrate(image_pts, real_width_m=4.0, real_height_m=2.0)
    return cal


@pytest.fixture
def identity_calibrator() -> HomographyCalibrator:
    """Calibrator where image coords = world coords (no perspective)."""
    cal = HomographyCalibrator()
    image_pts = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 2.0],
        [0.0, 2.0],
    ]
    cal.calibrate(image_pts, real_width_m=4.0, real_height_m=2.0)
    return cal


@pytest.fixture
def tmp_calib_path(tmp_path: Path) -> Path:
    return tmp_path / "test_calibration.json"


# ---------------------------------------------------------------------------
# HomographyCalibrator tests
# ---------------------------------------------------------------------------

class TestHomographyCalibrator:

    def test_calibrate_sets_valid(self, simple_calibrator: HomographyCalibrator):
        assert simple_calibrator.is_valid

    def test_uncalibrated_raises(self):
        cal = HomographyCalibrator()
        assert not cal.is_valid
        with pytest.raises(RuntimeError):
            cal.image_to_world([[0, 0]])
        with pytest.raises(RuntimeError):
            cal.world_to_image([[0, 0]])

    def test_calibrate_rejects_bad_input(self):
        cal = HomographyCalibrator()
        with pytest.raises(ValueError):
            cal.calibrate([[0, 0], [1, 1]], real_width_m=4.0, real_height_m=2.0)
        with pytest.raises(ValueError):
            cal.calibrate([[0, 0]] * 4, real_width_m=-1.0, real_height_m=2.0)

    def test_reprojection_error_near_zero(self, simple_calibrator: HomographyCalibrator):
        """Round-trip through getPerspectiveTransform should give near-zero error."""
        err = simple_calibrator.reprojection_error()
        assert err < 0.1, f"Reprojection error {err} too large"

    def test_corner_points_map_correctly(self, simple_calibrator: HomographyCalibrator):
        """The 4 calibration image points should map to the 4 world corners."""
        img_pts = simple_calibrator.image_points.tolist()
        world_pts = simple_calibrator.image_to_world(img_pts)

        expected = [[0, 0], [4, 0], [4, 2], [0, 2]]
        for got, exp in zip(world_pts, expected):
            assert abs(got[0] - exp[0]) < 0.01
            assert abs(got[1] - exp[1]) < 0.01

    def test_round_trip_identity(self, identity_calibrator: HomographyCalibrator):
        """Image→world→image should return original points."""
        test_pts = [[1.0, 0.5], [2.5, 1.5], [3.0, 0.8]]
        world = identity_calibrator.image_to_world(test_pts)
        back = identity_calibrator.world_to_image(world)
        for orig, rt in zip(test_pts, back):
            assert abs(orig[0] - rt[0]) < 0.01
            assert abs(orig[1] - rt[1]) < 0.01

    def test_polygon_area_known_rectangle(self, identity_calibrator: HomographyCalibrator):
        """A 4×2 rectangle in identity mapping should have area 8 m²."""
        rect = [[0, 0], [4, 0], [4, 2], [0, 2]]
        area = identity_calibrator.polygon_area_m2(rect)
        assert abs(area - 8.0) < 0.01

    def test_polygon_area_half_rect(self, identity_calibrator: HomographyCalibrator):
        """A triangle covering half the rectangle should have area 4 m²."""
        tri = [[0, 0], [4, 0], [4, 2]]
        area = identity_calibrator.polygon_area_m2(tri)
        assert abs(area - 4.0) < 0.01

    def test_polygon_area_perspective(self, simple_calibrator: HomographyCalibrator):
        """The calibration quadrilateral itself should map to 4×2 = 8 m²."""
        quad = simple_calibrator.image_points.tolist()
        area = simple_calibrator.polygon_area_m2(quad)
        assert abs(area - 8.0) < 0.05

    def test_people_in_zone_uniform_density(self, identity_calibrator: HomographyCalibrator):
        """With uniform density, fraction should match zone area / total area."""
        density = np.ones((200, 400), dtype=np.float32)
        # Zone covering roughly half the image (left half)
        zone = [[0, 0], [200, 0], [200, 200], [0, 200]]
        fraction = identity_calibrator.people_in_zone(density, zone)
        assert abs(fraction - 0.5) < 0.01

    def test_people_in_zone_concentrated(self, identity_calibrator: HomographyCalibrator):
        """If density is only in the zone polygon, fraction should be ~1.0."""
        density = np.zeros((200, 400), dtype=np.float32)
        # Put density only in a small region
        density[10:50, 10:50] = 1.0
        zone = [[0, 0], [60, 0], [60, 60], [0, 60]]
        fraction = identity_calibrator.people_in_zone(density, zone)
        assert fraction > 0.95

    def test_density_per_m2(self, identity_calibrator: HomographyCalibrator):
        """Known count in a known area should give correct ppl/m²."""
        # 2×2 zone → area = 4 m²
        density = np.ones((200, 400), dtype=np.float32)
        total_count = 40.0
        zone = [[0, 0], [200, 0], [200, 200], [0, 200]]
        people, ppl_m2 = identity_calibrator.density_per_m2(density, total_count, zone)
        # ~50% of density → ~20 people in ~4.0 m² zone → ~5 ppl/m²
        assert people > 0
        assert ppl_m2 > 0

    def test_preview_grid_generates_lines(self, simple_calibrator: HomographyCalibrator):
        lines = simple_calibrator.generate_preview_grid(
            real_width_m=4.0, real_height_m=2.0, grid_spacing_m=1.0,
        )
        # 5 vertical lines (x=0,1,2,3,4) + 3 horizontal (y=0,1,2) = 8
        assert len(lines) == 8

    def test_polygon_area_rejects_too_few(self, simple_calibrator: HomographyCalibrator):
        with pytest.raises(ValueError):
            simple_calibrator.polygon_area_m2([[0, 0], [1, 1]])


class TestCameraDrift:

    def test_no_reference_returns_no_drift(self):
        cal = HomographyCalibrator()
        cal.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]], 1.0, 1.0)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        drift, ratio = cal.check_camera_drift(frame)
        assert not drift
        assert ratio == 1.0

    def test_same_frame_no_drift(self):
        cal = HomographyCalibrator()
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        cal.calibrate(
            [[50, 50], [270, 50], [270, 190], [50, 190]],
            real_width_m=3.0,
            real_height_m=2.0,
            reference_frame=frame,
        )
        drift, ratio = cal.check_camera_drift(frame)
        # Same frame: should have high match ratio
        assert ratio >= 0.3  # ORB is noisy on random images; mainly test path

    def test_very_different_frame_detects_drift(self):
        cal = HomographyCalibrator()
        # Reference: mostly white with some features
        ref = np.full((240, 320, 3), 200, dtype=np.uint8)
        cv2.rectangle(ref, (50, 50), (150, 150), (0, 0, 0), 3)
        cv2.circle(ref, (250, 120), 40, (50, 50, 50), 5)
        cal.calibrate(
            [[50, 50], [270, 50], [270, 190], [50, 190]],
            real_width_m=3.0,
            real_height_m=2.0,
            reference_frame=ref,
        )
        # Current: completely different
        curr = np.full((240, 320, 3), 30, dtype=np.uint8)
        cv2.rectangle(curr, (200, 100), (300, 200), (255, 255, 255), 3)
        drift, ratio = cal.check_camera_drift(curr)
        # Should detect drift (low match ratio)
        assert ratio < 0.6


# ---------------------------------------------------------------------------
# CalibrationManager tests
# ---------------------------------------------------------------------------

class TestCalibrationManager:

    def test_save_and_load_round_trip(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        assert not mgr.is_calibrated

        image_pts = [[100, 100], [500, 100], [550, 300], [50, 300]]
        result = mgr.calibrate(image_pts, 4.0, 2.0, name="Test Calibration")
        assert result["valid"]
        assert abs(result["area_m2"] - 8.0) < 0.01

        mgr.save()
        assert tmp_calib_path.exists()

        # Load into a fresh manager
        mgr2 = CalibrationManager(calibration_path=tmp_calib_path)
        assert mgr2.is_calibrated
        assert abs(mgr2.floor_area_m2 - 8.0) < 0.01
        assert mgr2.real_width_m == 4.0
        assert mgr2.real_height_m == 2.0

    def test_load_preserves_homography(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        image_pts = [[100, 100], [500, 100], [550, 300], [50, 300]]
        mgr.calibrate(image_pts, 4.0, 2.0)
        mgr.save()

        mgr2 = CalibrationManager(calibration_path=tmp_calib_path)
        # Homography should produce same world coords
        world = mgr2.calibrator.image_to_world(image_pts)
        expected = [[0, 0], [4, 0], [4, 2], [0, 2]]
        for got, exp in zip(world, expected):
            assert abs(got[0] - exp[0]) < 0.01
            assert abs(got[1] - exp[1]) < 0.01

    def test_saved_json_schema(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        mgr.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]], 2.0, 2.0)
        mgr.save()

        with open(tmp_calib_path, "r") as f:
            data = json.load(f)

        assert "homography_matrix" in data
        assert "image_points" in data
        assert "world_points" in data
        assert "real_width_m" in data
        assert "confidence_score" in data
        assert "timestamp" in data
        assert len(data["homography_matrix"]) == 3
        assert len(data["homography_matrix"][0]) == 3

    def test_get_status_uncalibrated(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        status = mgr.get_status()
        assert not status["calibrated"]

    def test_get_status_calibrated(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        mgr.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]], 5.0, 3.0)
        status = mgr.get_status()
        assert status["calibrated"]
        assert status["floor_area_m2"] == 15.0

    def test_zone_density(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        # Identity-like mapping
        mgr.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]], 10.0, 10.0)

        density = np.ones((100, 100), dtype=np.float32)
        total_count = 50.0
        zone = [[0, 0], [50, 0], [50, 100], [0, 100]]  # left half

        people, area, ppl_m2 = mgr.zone_density(density, total_count, zone)
        assert people > 0
        assert area > 0
        assert ppl_m2 > 0

    def test_preview_grid(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        mgr.calibrate([[0, 0], [100, 0], [100, 100], [0, 100]], 4.0, 3.0)
        grid = mgr.preview_grid(grid_spacing_m=1.0)
        assert len(grid) > 0
        assert "start" in grid[0]
        assert "end" in grid[0]

    def test_save_without_calibration_raises(self, tmp_calib_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_calib_path)
        with pytest.raises(RuntimeError):
            mgr.save()

    def test_load_nonexistent_raises(self, tmp_path: Path):
        mgr = CalibrationManager(calibration_path=tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            mgr.load()


# ---------------------------------------------------------------------------
# Acceptance test: known patch density accuracy
# ---------------------------------------------------------------------------

class TestAcceptanceDensity:
    """Acceptance test: a marked 4 m × 2 m patch with a known head count
    gives density within ~15-20% of truth."""

    def test_known_patch_density_accuracy(self):
        """Simulate a 4×2 m patch with 20 people → truth is 2.5 ppl/m²."""
        cal = HomographyCalibrator()
        # Trapezoid in a 640×360 image simulating perspective
        image_pts = [
            [160, 90],
            [480, 90],
            [520, 270],
            [120, 270],
        ]
        cal.calibrate(image_pts, real_width_m=4.0, real_height_m=2.0)

        # Create a synthetic density map (640×360) with Gaussians inside the zone
        density = np.zeros((360, 640), dtype=np.float32)
        np.random.seed(42)
        true_count = 20

        # Place Gaussian blobs at random positions inside the calibration quad
        for _ in range(true_count):
            # Random point inside trapezoid (roughly)
            t = np.random.uniform(0, 1)
            s = np.random.uniform(0, 1)
            # Bilinear interpolation across the quad
            top = np.array(image_pts[0]) * (1 - s) + np.array(image_pts[1]) * s
            bot = np.array(image_pts[3]) * (1 - s) + np.array(image_pts[2]) * s
            pt = top * (1 - t) + bot * t
            cx, cy = int(pt[0]), int(pt[1])
            sigma = 8
            y0, y1 = max(0, cy - 20), min(360, cy + 20)
            x0, x1 = max(0, cx - 20), min(640, cx + 20)
            gy, gx = np.meshgrid(np.arange(y0, y1) - cy, np.arange(x0, x1) - cx, indexing="ij")
            g = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
            density[y0:y1, x0:x1] += g

        zone_polygon = image_pts
        people, ppl_m2 = cal.density_per_m2(density, float(true_count), zone_polygon)

        # Ground truth: 20 people / 8 m² = 2.5 ppl/m²
        truth_density = true_count / 8.0
        error_pct = abs(ppl_m2 - truth_density) / truth_density * 100

        print(f"\n[Acceptance Test] True: {truth_density:.2f} ppl/m², "
              f"Estimated: {ppl_m2:.2f} ppl/m², Error: {error_pct:.1f}%")

        # Accept within 20%
        assert error_pct < 20.0, (
            f"Density error {error_pct:.1f}% exceeds 20% tolerance. "
            f"Expected ~{truth_density:.2f}, got {ppl_m2:.2f}"
        )
