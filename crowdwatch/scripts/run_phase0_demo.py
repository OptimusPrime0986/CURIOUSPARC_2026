"""Phase 0 Execution Script: Load CLIP-EBC + Depth V2 Small, run on sample clip, save density + depth outputs."""

import logging
import sys
from pathlib import Path
import cv2
import matplotlib.cm as cm
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.counting.clip_ebc import CLIPEBCPredictor
from src.counting.depth_anything import DepthAnythingV2Predictor
from scripts.generate_sample_clip import generate_crowd_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("crowdwatch.phase0")


def run_phase0():
    logger.info("=== STARTING PHASE 0: SETUP & MODEL VALIDATION ===")

    # 1. Config Loader
    cfg = get_config()
    device = cfg.get("system", "device", "cpu")
    logger.info("Loaded config: %s (device=%s)", cfg.get("system", "app_name"), device)

    # 2. Ensure sample clip exists
    sample_video_path = PROJECT_ROOT / "data" / "real_crowd_test.mp4"
    if not sample_video_path.exists():
        logger.info("Generating synthetic crowd sample clip at %s", sample_video_path)
        generate_crowd_video(sample_video_path)

    # 3. Read sample frame
    cap = cv2.VideoCapture(str(sample_video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame from {sample_video_path}")
    logger.info("Captured test frame: %dx%d pixels", frame.shape[1], frame.shape[0])

    # 4. Initialize CLIP-EBC and generate density map
    clip_model_name = cfg.get("counting", "clip_model_name", "openai/clip-vit-base-patch16")
    counter = CLIPEBCPredictor(clip_model_name=clip_model_name, device=device)
    density_map, total_count = counter.predict(frame)
    logger.info("CLIP-EBC Density estimation complete. Estimated count: %.2f people", total_count)
    logger.info("Density map min: %.5f, max: %.5f, mean: %.5f", density_map.min(), density_map.max(), density_map.mean())

    # 5. Initialize Depth Anything V2 Small and generate depth map
    depth_model_id = cfg.get("depth", "model_id", "depth-anything/Depth-Anything-V2-Small-hf")
    depth_estimator = DepthAnythingV2Predictor(model_id=depth_model_id, device=device)
    depth_map, scale_correction, depth_confidence = depth_estimator.predict(frame)
    logger.info("Depth Anything V2 Small complete. Scale factor: %.2f, Confidence: %.2f", scale_correction, depth_confidence)
    logger.info("Depth map min: %.5f, max: %.5f, mean: %.5f", depth_map.min(), depth_map.max(), depth_map.mean())

    # 6. Save visualizations (Exit criterion for Phase 0)
    out_dir = PROJECT_ROOT / "data" / "phase0_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original frame
    raw_frame_path = out_dir / "01_raw_frame.jpg"
    cv2.imwrite(str(raw_frame_path), frame)

    # Save density map with JET / Turbo colormap
    norm_density = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    density_colored = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)
    density_overlay = cv2.addWeighted(frame, 0.5, density_colored, 0.5, 0)
    cv2.putText(
        density_overlay,
        f"CLIP-EBC Density Map | Est Count: {total_count:.1f} [SIMULATED]",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    density_path = out_dir / "02_density_map.jpg"
    cv2.imwrite(str(density_path), density_overlay)

    # Save depth map with INFERNO colormap
    norm_depth = (depth_map * 255.0).astype(np.uint8)
    depth_colored = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
    cv2.putText(
        depth_colored,
        f"Depth Anything V2 Small | Conf: {depth_confidence:.2f} Scale: {scale_correction:.2f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    depth_path = out_dir / "03_depth_map.jpg"
    cv2.imwrite(str(depth_path), depth_colored)

    # Save raw numpy matrices
    np.save(str(out_dir / "density_map.npy"), density_map)
    np.save(str(out_dir / "depth_map.npy"), depth_map)

    logger.info("Successfully saved Phase 0 artifacts:")
    logger.info(" - Raw frame: %s", raw_frame_path)
    logger.info(" - Density map: %s", density_path)
    logger.info(" - Depth map: %s", depth_path)
    logger.info("=== PHASE 0 EXIT CRITERION SATISFIED ===")

    return {
        "raw_frame_path": raw_frame_path,
        "density_path": density_path,
        "depth_path": depth_path,
        "total_count": total_count,
        "scale_correction": scale_correction,
        "depth_confidence": depth_confidence,
    }


if __name__ == "__main__":
    run_phase0()
