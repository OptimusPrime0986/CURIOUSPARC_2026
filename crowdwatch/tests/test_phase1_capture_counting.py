"""Automated tests for Phase 1 (Capture, Counting Scheduler, and API Pipeline)."""

import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.capture.video_stream import VideoStream
from src.counting.clip_ebc import CLIPEBCPredictor
from src.counting.scheduler import AdaptiveCountScheduler
from src.api.app import app
from src.api.service import PipelineService, get_pipeline


@pytest.fixture
def sample_video():
    video_path = Path(__file__).resolve().parent.parent / "data" / "sample_crowd.mp4"
    if not video_path.exists():
        from scripts.generate_sample_clip import generate_crowd_video
        generate_crowd_video(video_path)
    return str(video_path)


def test_video_stream_capture(sample_video):
    """Verify VideoStream ingests video file and yields valid frames."""
    stream = VideoStream(source=sample_video, target_fps=15).start()
    time.sleep(0.5)

    is_conn, frame, frame_id = stream.read()
    assert is_conn is True
    assert frame is not None
    assert frame.shape[2] == 3
    assert frame_id >= 1
    stream.stop()


def test_video_stream_reconnect_simulation():
    """Verify VideoStream handles non-existent or invalid source with graceful reconnect state."""
    disconnect_called = []

    def on_dc():
        disconnect_called.append(True)

    stream = VideoStream(
        source="non_existent_camera_or_file.mp4",
        reconnect_timeout_sec=2.0,
        reconnect_interval_sec=0.2,
        on_disconnect=on_dc,
    ).start()

    time.sleep(0.5)
    is_conn, frame, _ = stream.read()
    assert is_conn is False
    assert frame is None
    stream.stop()


def test_adaptive_count_scheduler_throughput():
    """Verify AdaptiveCountScheduler achieves >= 5 processed FPS (NFR-02)."""
    predictor = CLIPEBCPredictor(device="cpu")
    scheduler = AdaptiveCountScheduler(
        predictor=predictor,
        initial_frame_skip=3,
        auto_adapt=True,
    )

    dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Process 15 consecutive frames and measure total elapsed time
    start_time = time.perf_counter()
    num_frames = 15
    for f in range(num_frames):
        density_map, count, is_new, latency = scheduler.process_frame(dummy_frame, f)
        assert density_map is not None
        assert count >= 0.0

    total_time = time.perf_counter() - start_time
    effective_fps = num_frames / total_time

    # Must achieve >= 5 processed frames per second
    assert effective_fps >= 5.0, f"Expected >= 5.0 FPS, got {effective_fps:.2f} FPS"


def test_fastapi_rest_endpoints():
    """Verify FastAPI status, config, and source endpoints."""
    with TestClient(app) as client:
        # Status endpoint
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "processed_fps" in data
        assert "camera_angle" in data
        assert "view_type" in data["camera_angle"]

        # Config endpoint
        resp_cfg = client.get("/api/config")
        assert resp_cfg.status_code == 200
        cfg_data = resp_cfg.json()
        assert "system" in cfg_data
        assert "zones" in cfg_data

        # Source switch endpoint
        resp_src = client.post(
            "/api/stream/source",
            json={"source_type": "file", "source_value": "data/sample_crowd.mp4"},
        )
        assert resp_src.status_code == 200
        assert resp_src.json()["status"] == "ok"
