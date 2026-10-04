"""Automated test suite for Phase 0 (Setup, Config, CLIP-EBC, Depth Anything V2)."""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.config import ConfigManager, get_config
from src.counting.clip_ebc import CLIPEBCPredictor, CLIPEBCModel
from src.counting.depth_anything import DepthAnythingV2Predictor
from scripts.generate_sample_clip import generate_crowd_video
from scripts.run_phase0_demo import run_phase0


def test_config_loader():
    """Verify default.yaml, zones.json, and calibration.json load properly."""
    cfg = get_config()
    assert cfg.get("system", "app_name") is not None
    assert cfg.get("input", "source_type") in ["file", "webcam", "rtsp"]
    assert cfg.get("alerts", "state_thresholds", {}).get("critical_min_density") == 5.0

    # Zones check
    assert "zones" in cfg.zones_data
    assert len(cfg.zones_data["zones"]) >= 1

    # Calibration check
    assert "image_points" in cfg.calibration_data
    assert len(cfg.calibration_data["image_points"]) == 4


def test_sample_video_generator(tmp_path):
    """Verify synthetic crowd clip generation."""
    out_video = tmp_path / "test_crowd.mp4"
    generate_crowd_video(output_path=out_video, num_frames=10, width=320, height=180, fps=10)
    assert out_video.exists()

    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret
    assert frame.shape == (180, 320, 3)
    cap.release()


def test_clip_ebc_inference():
    """Verify CLIP-EBC initialization, forward pass, and output properties."""
    predictor = CLIPEBCPredictor(device="cpu")
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    density_map, total_count = predictor.predict(dummy_frame)
    assert isinstance(density_map, np.ndarray)
    assert density_map.shape == (240, 320)
    assert density_map.min() >= 0.0  # Non-negative density
    assert total_count >= 0.0
    assert np.isclose(float(np.sum(density_map)), total_count, atol=1e-3)


def test_clip_ebc_empty_frame():
    """Verify CLIP-EBC fails gracefully on empty input."""
    predictor = CLIPEBCPredictor(device="cpu")
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((0, 0, 3), dtype=np.uint8))


def test_depth_anything_v2_inference():
    """Verify Depth Anything V2 Small depth estimation and scale correction."""
    depth_estimator = DepthAnythingV2Predictor(device="cpu")
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    depth_map, scale_correction, depth_confidence = depth_estimator.predict(dummy_frame)
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == (240, 320)
    assert 0.0 <= depth_map.min() <= depth_map.max() <= 1.0
    assert 0.5 <= scale_correction <= 4.0
    assert 0.0 <= depth_confidence <= 1.0


def test_phase0_end_to_end_exit_criterion():
    """Verify Phase 0 exit criterion: Density map and depth map saved for one frame."""
    result = run_phase0()
    assert Path(result["raw_frame_path"]).exists()
    assert Path(result["density_path"]).exists()
    assert Path(result["depth_path"]).exists()
    assert result["total_count"] >= 0.0
    assert 0.0 <= result["depth_confidence"] <= 1.0
