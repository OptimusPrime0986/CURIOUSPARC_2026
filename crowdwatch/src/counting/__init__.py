"""Counting package — CLIP-EBC density estimation, Depth Anything V2, and adaptive scheduler."""

from src.counting.clip_ebc import CLIPEBCModel, CLIPEBCPredictor
from src.counting.depth_anything import DepthAnythingV2Predictor
from src.counting.scheduler import AdaptiveCountScheduler

__all__ = [
    "CLIPEBCModel",
    "CLIPEBCPredictor",
    "DepthAnythingV2Predictor",
    "AdaptiveCountScheduler",
]
