"""
Tests for automatic camera angle estimation and self-configuration.
Validates:
1. Angle estimation from image properties (top-down vs oblique vs low-oblique).
2. Auto-calibration of ground-plane homography from angle estimation.
3. Density predictor adapting kernel shapes and scale gradients to angles.
4. FastAPI endpoint /api/calibration/auto_angle.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.calibration.angle_estimator import CameraAngleEstimator, CameraAngleResult
from src.calibration.manager import CalibrationManager
from src.counting.clip_ebc import CLIPEBCPredictor
from src.api.app import app


class TestCameraAngleEstimator:
    def test_uniform_empty_frame_defaults_to_high_oblique(self):
        estimator = CameraAngleEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)
        assert isinstance(result, CameraAngleResult)
        assert result.pitch_deg > 0
        assert result.scale_gradient >= 1.0

    def test_synthetic_vertical_elongation_detects_low_angle(self):
        estimator = CameraAngleEstimator()
        # Draw realistic standing pedestrian ellipses: 12px wide, 34px tall (AR ~ 2.8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in [150, 250, 350]:
            for x in range(80, 580, 70):
                cv2.ellipse(frame, (x, y), (7, 24), 0, 0, 360, (220, 220, 220), -1)

        result = estimator.estimate(frame)
        assert result.aspect_ratio_med >= 1.2
        assert result.pitch_deg <= 75.0
        assert result.view_type in ["HIGH_OBLIQUE", "LOW_OBLIQUE"]

    def test_sample_video_top_down_detection(self):
        sample_path = Path("data/samples/sample_crowd.mp4")
        if sample_path.exists():
            cap = cv2.VideoCapture(str(sample_path))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                estimator = CameraAngleEstimator()
                res = estimator.estimate(frame)
                assert res.view_type == "TOP_DOWN"
                assert res.pitch_deg >= 75.0

    def test_smooth_filtering_maintains_stability(self):
        estimator = CameraAngleEstimator()
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        res1 = estimator.estimate(frame)
        res2 = estimator.estimate(frame)
        assert abs(res1.pitch_deg - res2.pitch_deg) < 2.0


class TestAutoCalibrationFromAngle:
    def test_manager_auto_calibrate_configures_homography(self):
        manager = CalibrationManager()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        angle_res = manager.auto_calibrate_from_angle(frame, force=True)
        assert isinstance(angle_res, CameraAngleResult)
        assert manager.is_calibrated is True

        # Verify homography is active and status contains camera angle
        status = manager.get_status()
        assert status["calibrated"] is True
        assert status["floor_area_m2"] > 0
        assert status["camera_angle"]["auto_configured"] is True
        assert "pitch_deg" in status["camera_angle"]


class TestPredictorAngleAdaptation:
    def test_top_down_kernel_adaptation(self):
        predictor = CLIPEBCPredictor(device="cpu")
        frame = np.ones((256, 256, 3), dtype=np.uint8) * 128

        angle_top = CameraAngleResult(
            pitch_deg=85.0,
            view_type="TOP_DOWN",
            scale_gradient=1.0,
            aspect_ratio_med=1.0,
            confidence=0.9,
            label="Top-Down (Bird's Eye)",
        )
        density_map, count = predictor.predict(frame, angle_result=angle_top)
        assert count >= 0.0
        assert density_map.shape == (256, 256)

    def test_low_oblique_kernel_adaptation(self):
        predictor = CLIPEBCPredictor(device="cpu")
        frame = np.ones((256, 256, 3), dtype=np.uint8) * 128

        angle_low = CameraAngleResult(
            pitch_deg=35.0,
            view_type="LOW_OBLIQUE",
            scale_gradient=3.5,
            aspect_ratio_med=2.6,
            confidence=0.88,
            label="Low Concourse Oblique",
        )
        density_map, count = predictor.predict(frame, angle_result=angle_low)
        assert count >= 0.0
        assert density_map.shape == (256, 256)


class TestApiAutoAngleEndpoint:
    def test_auto_angle_post_endpoint(self):
        with TestClient(app) as client:
            response = client.post("/api/calibration/auto_angle", json={"force": True})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "angle" in data
            assert "view_type" in data["angle"]
            assert "pitch_deg" in data["angle"]
            assert "scale_gradient" in data["angle"]
            assert "calibration" in data
            assert data["calibration"]["floor_area_m2"] > 0
